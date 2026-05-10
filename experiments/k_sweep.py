"""
Accuracy vs K trade-off curve.

Sweeps the TopK selection budget K ∈ {128, 256, 512, 1024, 2048, 4096}
(by default) and records, for each K:

    • per-step decode latency  (kernel-only timing)
    • per-step throughput
    • cache hit rate
    • perplexity on a wikitext slice  (if a model is provided)

This is the headline plot for the report: it lets the reader pick the
operating point at which selection budget pays for itself in quality
without tipping the latency curve.

Usage:
    python -m experiments.k_sweep                          # latency only
    python -m experiments.k_sweep --model meta-llama/Llama-2-7b-hf
    python -m experiments.k_sweep --K-values 128 256 512 1024 2048 4096
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, List, Optional

import torch

from methods.registry import make_method
from ._common import (
    ExperimentRecord, device_summary, maybe_load_model,
    warmup_then_time, write_records,
)
from .kernel_bench import make_synthetic_kv


def _time_one_K(
    K: int,
    past_kv,
    n_sink: int, n_local: int,
    use_kernels: bool,
    n_warmup: int, n_iter: int, device: Any,
) -> dict:
    method = make_method(
        "topk",
        K=K, n_sink=n_sink, n_local=n_local,
        refresh_interval=0, use_kernels=use_kernels,
    )
    method.process_prefill(past_kv)
    timing = warmup_then_time(
        lambda: method.process_step(past_kv, step=1),
        n_warmup=n_warmup, n_iter=n_iter, device=device,
    )
    stats = method.get_stats()
    return {
        "latency_ms":     timing["mean_s"] * 1000,
        "throughput_tps": 1.0 / timing["mean_s"] if timing["mean_s"] > 0 else 0.0,
        "p50_ms":         timing["p50"] * 1000,
        "p99_ms":         timing["p99"] * 1000,
        "std_ms":         timing["std_s"] * 1000,
        "cache_hit_rate": stats["cache_hit_rate"],
    }


def _ppl_one_K(
    K: int, model, tokenizer,
    n_sink: int, n_local: int,
    n_examples: int, max_length: int, device: str,
) -> Optional[float]:
    try:
        from datasets import load_dataset
        from benchmark.runner import compute_method_perplexity
    except Exception as e:
        print(f"[k_sweep] skipping PPL: {e}")
        return None

    method = make_method(
        "topk",
        K=K, n_sink=n_sink, n_local=n_local,
        refresh_interval=0, use_kernels=True,
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


def run_k_sweep(
    K_values: List[int],
    seq_len: int = 4096,
    n_layers: int = 32, n_heads: int = 32, head_dim: int = 128,
    n_sink: int = 128, n_local: int = 512,
    use_kernels: bool = True,
    n_warmup: int = 5, n_iter: int = 20,
    model_name: Optional[str] = None,
    n_ppl_examples: int = 50, max_length: int = 512,
    device: str = "cuda",
    output_dir: Path = Path("results/phase_a/k_sweep"),
) -> List[ExperimentRecord]:
    if device == "cuda" and not torch.cuda.is_available():
        print("[k_sweep] CUDA unavailable — running on CPU.")
        device = "cpu"

    print(f"[k_sweep] K ∈ {K_values}, seq_len={seq_len}")
    print(f"[k_sweep] {device_summary(device)}")

    past_kv = make_synthetic_kv(n_layers, n_heads, head_dim, seq_len,
                                device=device)

    model, tokenizer = (None, None)
    if model_name is not None:
        print(f"[k_sweep] loading {model_name} for PPL…")
        model, tokenizer = maybe_load_model(model_name, device=device)

    records: List[ExperimentRecord] = []
    for K in K_values:
        timing = _time_one_K(
            K, past_kv, n_sink, n_local,
            use_kernels, n_warmup, n_iter, device,
        )
        ppl = (
            _ppl_one_K(K, model, tokenizer, n_sink, n_local,
                       n_ppl_examples, max_length, device)
            if model is not None else None
        )

        line = (f"  K={K:5d}  "
                f"lat={timing['latency_ms']:7.3f} ms  "
                f"tps={timing['throughput_tps']:7.0f}  "
                f"hit={timing['cache_hit_rate']:.0%}")
        if ppl is not None:
            line += f"  ppl={ppl:.3f}"
        print(line)

        metrics = dict(timing)
        if ppl is not None:
            metrics["ppl"] = ppl

        records.append(ExperimentRecord(
            experiment="k_sweep",
            method="topk",
            config={"K": K, "seq_len": seq_len, "n_sink": n_sink,
                    "n_local": n_local, "use_kernels": use_kernels},
            metrics=metrics,
        ))

    out_path = Path(output_dir) / "k_sweep"
    write_records(records, out_path)
    print(f"\n[k_sweep] Wrote {out_path}.csv and {out_path}.json")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--K-values", type=int, nargs="+",
                   default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads",  type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--n-sink",   type=int, default=128)
    p.add_argument("--n-local",  type=int, default=512)
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--n-ppl-examples", type=int, default=50)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter",   type=int, default=20)
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_a/k_sweep"))
    args = p.parse_args()

    run_k_sweep(
        K_values=args.K_values,
        seq_len=args.seq_len,
        n_layers=args.n_layers, n_heads=args.n_heads, head_dim=args.head_dim,
        n_sink=args.n_sink, n_local=args.n_local,
        use_kernels=not args.no_kernels,
        n_warmup=args.n_warmup, n_iter=args.n_iter,
        model_name=args.model,
        n_ppl_examples=args.n_ppl_examples, max_length=args.max_length,
        device=args.device, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
