"""
Ablation runner — measures the marginal cost of each TopK novelty.

Default sweep ("leave-one-out"): five rows, each disables exactly one
flag and reports the delta vs the full configuration:

    full                              (all five flags ON)
    -head_softmax                     (Novelty 3 disabled)
    -criticality_weights              (Novelty 4, 10 disabled)
    -selection_cache                  (Novelty 5, 11 disabled)
    -sink_tokens                      (n_sink → 0)
    -local_tokens                     (n_local → 0)

A `--full-grid` option exhaustively sweeps all 2^5 combinations for a
complete attribution table. Use sparingly — that's 32 runs.

Two modes are supported:
  1. Kernel-only (no model): times `process_step()` on a synthetic KV
     cache. Fast; isolates the algorithmic cost.
  2. With model: also measures end-to-end PPL on WikiText so quality
     and latency are reported in the same row.

Usage:
    python -m experiments.ablation                                  # kernel-only
    python -m experiments.ablation --model meta-llama/Llama-2-7b-hf # +PPL
    python -m experiments.ablation --full-grid                      # all 32
"""

from __future__ import annotations
import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from methods.registry import make_method
from ._common import (
    ExperimentRecord, device_summary, maybe_load_model,
    warmup_then_time, write_records,
)
from .kernel_bench import make_synthetic_kv


# ── The five ablation flags ──────────────────────────────────────────────────

ABLATION_FLAGS = [
    "use_head_softmax",
    "use_criticality_weights",
    "use_selection_cache",
    "use_sink_tokens",
    "use_local_tokens",
]


def _config_label(flags: Dict[str, bool]) -> str:
    """Compact label like 'full' or '-head_softmax,-sink_tokens'."""
    off = [f.replace("use_", "") for f, v in flags.items() if not v]
    return "full" if not off else ",".join(f"-{x}" for x in off)


def _leave_one_out_configs() -> List[Dict[str, bool]]:
    """[full, -flag_1, -flag_2, ...] — six rows total."""
    full = {f: True for f in ABLATION_FLAGS}
    out = [dict(full)]
    for f in ABLATION_FLAGS:
        cfg = dict(full)
        cfg[f] = False
        out.append(cfg)
    return out


def _full_grid_configs() -> List[Dict[str, bool]]:
    """All 2^5 = 32 combinations of the five flags."""
    out = []
    for combo in itertools.product([False, True], repeat=len(ABLATION_FLAGS)):
        out.append(dict(zip(ABLATION_FLAGS, combo)))
    return out


# ── Kernel-only timing ───────────────────────────────────────────────────────

def _time_kernel_path(
    flags: Dict[str, bool],
    past_kv,
    K: int, n_sink: int, n_local: int,
    use_kernels: bool,
    n_warmup: int, n_iter: int, device: Any,
) -> Dict[str, float]:
    method = make_method(
        "topk",
        K=K, n_sink=n_sink, n_local=n_local,
        refresh_interval=0,
        use_kernels=use_kernels,
        **flags,
    )
    method.process_prefill(past_kv)

    def step():
        method.process_step(past_kv, step=1)

    timing = warmup_then_time(step, n_warmup=n_warmup, n_iter=n_iter,
                              device=device)
    stats = method.get_stats()
    return {
        "latency_ms":     timing["mean_s"] * 1000,
        "throughput_tps": 1.0 / timing["mean_s"] if timing["mean_s"] > 0 else 0.0,
        "p50_ms":         timing["p50"] * 1000,
        "p99_ms":         timing["p99"] * 1000,
        "std_ms":         timing["std_s"] * 1000,
        "cache_hit_rate": stats["cache_hit_rate"],
    }


# ── End-to-end PPL (optional, only when --model is passed) ───────────────────

def _measure_ppl(
    flags: Dict[str, bool],
    model, tokenizer,
    K: int, n_sink: int, n_local: int,
    n_examples: int = 50, max_length: int = 512,
    device: str = "cuda",
) -> Optional[float]:
    """Compute PPL on a small wikitext slice; returns None on any failure."""
    try:
        from datasets import load_dataset
        from benchmark.runner import compute_method_perplexity
    except Exception as e:
        print(f"[ablation] skipping PPL: {e}")
        return None

    method = make_method(
        "topk",
        K=K, n_sink=n_sink, n_local=n_local,
        refresh_interval=0, use_kernels=True,
        **flags,
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


# ── Main ─────────────────────────────────────────────────────────────────────

def run_ablation(
    seq_len: int = 4096,
    n_layers: int = 32, n_heads: int = 32, head_dim: int = 128,
    K: int = 1024, n_sink: int = 128, n_local: int = 512,
    use_kernels: bool = True,
    n_warmup: int = 5, n_iter: int = 20,
    full_grid: bool = False,
    model_name: Optional[str] = None,
    n_ppl_examples: int = 50,
    device: str = "cuda",
    output_dir: Path = Path("results/phase_a/ablation"),
) -> List[ExperimentRecord]:
    if device == "cuda" and not torch.cuda.is_available():
        print("[ablation] CUDA unavailable — running on CPU.")
        device = "cpu"

    configs = _full_grid_configs() if full_grid else _leave_one_out_configs()
    print(f"[ablation] {len(configs)} configurations × seq_len={seq_len}")
    print(f"[ablation] {device_summary(device)}")

    past_kv = make_synthetic_kv(n_layers, n_heads, head_dim, seq_len,
                                device=device)

    model, tokenizer = (None, None)
    if model_name is not None:
        print(f"[ablation] loading {model_name} for PPL measurement…")
        model, tokenizer = maybe_load_model(model_name, device=device)

    records: List[ExperimentRecord] = []
    full_latency_ms: Optional[float] = None

    for flags in configs:
        label = _config_label(flags)
        print(f"\n── {label} ─────────────")

        timing = _time_kernel_path(
            flags, past_kv,
            K=K, n_sink=n_sink, n_local=n_local,
            use_kernels=use_kernels,
            n_warmup=n_warmup, n_iter=n_iter, device=device,
        )
        print(f"  latency = {timing['latency_ms']:7.3f} ms   "
              f"hit_rate = {timing['cache_hit_rate']:.0%}")

        if label == "full":
            full_latency_ms = timing["latency_ms"]

        ppl = None
        if model is not None:
            ppl = _measure_ppl(
                flags, model, tokenizer,
                K=K, n_sink=n_sink, n_local=n_local,
                n_examples=n_ppl_examples, device=device,
            )
            print(f"  PPL    = {ppl:.3f}" if ppl is not None else "  PPL    = (skipped)")

        delta_ms = (
            timing["latency_ms"] - full_latency_ms
            if full_latency_ms is not None else 0.0
        )

        metrics = dict(timing)
        metrics["delta_ms_vs_full"] = delta_ms
        if ppl is not None:
            metrics["ppl"] = ppl

        records.append(ExperimentRecord(
            experiment="ablation",
            method="topk",
            config={"label": label, "seq_len": seq_len,
                    "K": K, "n_sink": n_sink, "n_local": n_local,
                    "use_kernels": use_kernels,
                    **flags},
            metrics=metrics,
        ))

    out_path = Path(output_dir) / "ablation"
    write_records(records, out_path)
    print(f"\n[ablation] Wrote {out_path}.csv and {out_path}.json")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads",  type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--K",        type=int, default=1024)
    p.add_argument("--n-sink",   type=int, default=128)
    p.add_argument("--n-local",  type=int, default=512)
    p.add_argument("--no-kernels", action="store_true",
                   help="Time the PyTorch reference path instead of Triton.")
    p.add_argument("--full-grid", action="store_true",
                   help="Run all 2^5 = 32 ablation combinations.")
    p.add_argument("--model", type=str, default=None,
                   help="HF model name. If set, also reports PPL per row.")
    p.add_argument("--n-ppl-examples", type=int, default=50)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter",   type=int, default=20)
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_a/ablation"))
    args = p.parse_args()

    run_ablation(
        seq_len=args.seq_len,
        n_layers=args.n_layers, n_heads=args.n_heads,
        head_dim=args.head_dim,
        K=args.K, n_sink=args.n_sink, n_local=args.n_local,
        use_kernels=not args.no_kernels,
        n_warmup=args.n_warmup, n_iter=args.n_iter,
        full_grid=args.full_grid,
        model_name=args.model,
        n_ppl_examples=args.n_ppl_examples,
        device=args.device, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
