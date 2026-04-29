"""
Long-context scaling sweep — Phase B headline experiment.

Compares all four KV-cache methods at increasing sequence lengths:

    baseline      — FP16 full cache (reference; OOMs first)
    kivi          — KIVI 4-bit asymmetric block-wise quant
    topk          — TokenSelect dynamic top-K (Triton kernels)
    kivi_topk     — Phase B hybrid (design a) — KIVI storage + centroid TopK

For each (method, seq_len) it reports:

    • storage_mb              — KV cache size in MiB on the GPU
    • prefill_s               — wall time of `process_prefill`
    • decode_step_ms          — mean latency of one `process_step`
    • decode_p99_ms           — 99-th percentile decode latency
    • cache_hit_rate          — selection-cache reuse fraction (TopK / hybrid)
    • blocks_dequantized_avg  — per-step mean (hybrid only) — measures the
                                 selective-dequant volume design (a) saves
    • ppl                     — wikitext perplexity *if* a model is provided

Crucially, this sweep keeps `use_selection_cache=False` for the latency
runs so the kernel actually fires every step — fixing the Phase A v1
methodology issue where 97% cache-hit rate masked the kernel cost.

The PPL pass uses `max_length` ≥ 4×budget so the TopK selection actually
engages (Phase A v1 ran with max_length=512 < budget=1664, leaving the
selection logic dormant — same fix-up applied here).

Usage:
    python -m experiments.long_context                     # latency only
    python -m experiments.long_context --model meta-llama/Llama-2-7b-hf
    python -m experiments.long_context --seq-lens 2048 8192 16384
"""
from __future__ import annotations
import argparse
import gc
from pathlib import Path
from typing import Any, List, Optional

import torch

from methods.registry import make_method
from ._common import (
    ExperimentRecord, device_summary, maybe_load_model,
    warmup_then_time, write_records,
)
from .kernel_bench import make_synthetic_kv


# Methods we benchmark in this sweep. Each entry is (registry_name, kwargs)
# — kwargs are merged with the per-call config in `_method_kwargs`.
# kivi_topk   = Phase B design (a), centroid scoring
# kivi_topk_c = Phase B design (c), Triton quant-aware scoring
_METHOD_NAMES: List[str] = [
    "baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c",
]


def _method_kwargs(name: str, K: int, n_sink: int, n_local: int,
                   bits: int, group_size: int, residual_length: int,
                   use_selection_cache: bool) -> dict:
    """Per-method constructor args. Unrecognised kwargs are ignored by factories."""
    common = dict(
        K=K, n_sink=n_sink, n_local=n_local,
        bits=bits, group_size=group_size, residual_length=residual_length,
        use_selection_cache=use_selection_cache,
        # All other ablation flags left at default (full-features ON).
    )
    return common


def _time_method(
    name: str,
    seq_len: int,
    n_layers: int, n_heads: int, head_dim: int,
    K: int, n_sink: int, n_local: int,
    bits: int, group_size: int, residual_length: int,
    n_warmup: int, n_iter: int,
    device: Any,
    use_selection_cache: bool,
) -> Optional[dict]:
    """
    Build synthetic KV at `seq_len`, time prefill + per-step decode for one
    method. Returns a metrics dict, or None if OOM (caught and reported).
    """
    try:
        past_kv = make_synthetic_kv(
            n_layers, n_heads, head_dim, seq_len, device=device,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    method = make_method(
        name,
        **_method_kwargs(name, K, n_sink, n_local,
                         bits, group_size, residual_length,
                         use_selection_cache),
    )

    # Prefill timing (single shot — too expensive to repeat at 32K).
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        recon = method.process_prefill(past_kv)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        del past_kv
        torch.cuda.empty_cache()
        return None

    # Decode-step timing.
    try:
        timing = warmup_then_time(
            lambda: method.process_step(recon, step=1),
            n_warmup=n_warmup, n_iter=n_iter, device=device,
        )
    except torch.cuda.OutOfMemoryError:
        del past_kv, recon
        torch.cuda.empty_cache()
        return None

    storage_b = method.get_kv_size_bytes(past_kv)
    metrics = {
        "storage_mb":      storage_b / (1024 * 1024),
        "prefill_s":       prefill_s,
        "decode_step_ms":  timing["mean_s"] * 1000,
        "decode_p50_ms":   timing["p50"] * 1000,
        "decode_p99_ms":   timing["p99"] * 1000,
        "decode_std_ms":   timing["std_s"] * 1000,
    }
    if hasattr(method, "get_stats"):
        s = method.get_stats()
        metrics["cache_hit_rate"] = s.get("cache_hit_rate", 0.0)
        if "blocks_dequantized" in s and timing["n_iter"] > 0:
            metrics["blocks_dequantized_avg"] = (
                s["blocks_dequantized"] / max(1, s.get("decode_steps", 1))
            )

    # Cleanup before next config (32K runs eat 16+ GiB of KV alone).
    del past_kv, recon, method
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _measure_ppl(
    method_name: str,
    K: int, n_sink: int, n_local: int,
    bits: int, group_size: int, residual_length: int,
    model, tokenizer,
    n_examples: int, max_length: int, device: str,
) -> Optional[float]:
    """PPL on a wikitext slice; max_length is enforced ≥ 4×budget by caller."""
    try:
        from datasets import load_dataset
        from benchmark.runner import compute_method_perplexity
    except Exception as e:
        print(f"[long_context] skipping PPL: {e}")
        return None

    method = make_method(
        method_name,
        **_method_kwargs(method_name, K, n_sink, n_local,
                         bits, group_size, residual_length,
                         use_selection_cache=True),
    )
    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    texts = []
    for ex in wt:
        t = ex["text"].strip()
        if t and not t.startswith("="):
            texts.append(t)
            if len(texts) >= n_examples:
                break
    return compute_method_perplexity(
        model, tokenizer, method, texts,
        device=device, max_length=max_length,
    )


def run_long_context(
    seq_lens: List[int] = (2048, 8192, 16384),
    n_layers: int = 32, n_heads: int = 32, head_dim: int = 128,
    K: int = 1024, n_sink: int = 128, n_local: int = 512,
    bits: int = 4, group_size: int = 32, residual_length: int = 128,
    n_warmup: int = 5, n_iter: int = 15,
    methods: List[str] = None,
    use_selection_cache: bool = False,            # ← Phase A v2 fix
    model_name: Optional[str] = None,
    n_ppl_examples: int = 30,
    max_length: int = 4096,                       # ← Phase A v2 fix
    device: str = "cuda",
    output_dir: Path = Path("results/phase_b/long_context"),
) -> List[ExperimentRecord]:
    if device == "cuda" and not torch.cuda.is_available():
        print("[long_context] CUDA unavailable — running on CPU.")
        device = "cpu"

    methods = list(methods or _METHOD_NAMES)

    print(f"[long_context] seq_lens={list(seq_lens)}")
    print(f"[long_context] methods={methods}")
    print(f"[long_context] use_selection_cache={use_selection_cache} "
          f"(False isolates kernel cost)")
    print(f"[long_context] {device_summary(device)}")

    model, tokenizer = (None, None)
    if model_name is not None:
        budget = K + n_sink + n_local
        if max_length < 4 * budget:
            print(f"[long_context] WARNING: max_length={max_length} < "
                  f"4×budget={4*budget}. TopK won't engage on PPL pass.")
        print(f"[long_context] loading {model_name} for PPL…")
        model, tokenizer = maybe_load_model(model_name, device=device)

    records: List[ExperimentRecord] = []
    for seq_len in seq_lens:
        print(f"\n── seq_len = {seq_len} ──")
        for name in methods:
            metrics = _time_method(
                name=name, seq_len=seq_len,
                n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
                K=K, n_sink=n_sink, n_local=n_local,
                bits=bits, group_size=group_size,
                residual_length=residual_length,
                n_warmup=n_warmup, n_iter=n_iter,
                device=device,
                use_selection_cache=use_selection_cache,
            )
            if metrics is None:
                print(f"  {name:11s}  OOM — skipped")
                records.append(ExperimentRecord(
                    experiment="long_context", method=name,
                    config={"seq_len": seq_len, "K": K,
                            "use_selection_cache": use_selection_cache},
                    metrics={}, notes="OOM",
                ))
                continue

            line = (f"  {name:11s}  "
                    f"mem={metrics['storage_mb']:8.1f} MB  "
                    f"prefill={metrics['prefill_s']:6.3f} s  "
                    f"decode={metrics['decode_step_ms']:7.3f} ms  "
                    f"p99={metrics['decode_p99_ms']:7.3f} ms")
            if "cache_hit_rate" in metrics:
                line += f"  hit={metrics['cache_hit_rate']:.0%}"
            if "blocks_dequantized_avg" in metrics:
                line += f"  blocks/step={metrics['blocks_dequantized_avg']:.1f}"
            print(line)

            ppl = None
            if model is not None:
                ppl = _measure_ppl(
                    method_name=name, K=K, n_sink=n_sink, n_local=n_local,
                    bits=bits, group_size=group_size,
                    residual_length=residual_length,
                    model=model, tokenizer=tokenizer,
                    n_examples=n_ppl_examples,
                    max_length=max_length, device=device,
                )
                if ppl is not None:
                    metrics["ppl"] = ppl
                    print(f"             ppl={ppl:.3f}")

            records.append(ExperimentRecord(
                experiment="long_context", method=name,
                config={"seq_len": seq_len, "K": K,
                        "n_sink": n_sink, "n_local": n_local,
                        "bits": bits, "group_size": group_size,
                        "residual_length": residual_length,
                        "use_selection_cache": use_selection_cache},
                metrics=metrics,
            ))

    out_path = Path(output_dir) / "long_context"
    write_records(records, out_path)
    print(f"\n[long_context] Wrote {out_path}.csv and {out_path}.json")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[2048, 8192, 16384])
    p.add_argument("--methods", type=str, nargs="+",
                   default=_METHOD_NAMES)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads",  type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--K",          type=int, default=1024)
    p.add_argument("--n-sink",     type=int, default=128)
    p.add_argument("--n-local",    type=int, default=512)
    p.add_argument("--bits",            type=int, default=4)
    p.add_argument("--group-size",      type=int, default=32)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter",   type=int, default=15)
    p.add_argument("--use-selection-cache", action="store_true",
                   help="Enable TopK's selection cache (default OFF — isolates "
                        "kernel cost; Phase A v2 fix-up).")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--n-ppl-examples", type=int, default=30)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_b/long_context"))
    args = p.parse_args()

    run_long_context(
        seq_lens=args.seq_lens,
        n_layers=args.n_layers, n_heads=args.n_heads, head_dim=args.head_dim,
        K=args.K, n_sink=args.n_sink, n_local=args.n_local,
        bits=args.bits, group_size=args.group_size,
        residual_length=args.residual_length,
        n_warmup=args.n_warmup, n_iter=args.n_iter,
        methods=args.methods,
        use_selection_cache=args.use_selection_cache,
        model_name=args.model,
        n_ppl_examples=args.n_ppl_examples, max_length=args.max_length,
        device=args.device, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
