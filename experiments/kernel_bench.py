"""
Kernel-level decode-step benchmark.

Compares per-decode-step latency for three configurations on the SAME
synthetic KV cache:

    1. baseline           — full attention (no method)
    2. topk_pytorch       — TopK with kernels OFF (PyTorch reference)
    3. topk_triton        — TopK with kernels ON  (fused Triton path)

For each configuration we measure:
    • Total step latency (ms)
    • Time spent in token selection (ms)
    • Time spent in attention proper (ms; baseline = total)
    • Equivalent throughput (tokens / second)

The harness operates directly on `process_step()` of each method
(no model forward pass), isolating the cost of the KV-cache logic
from the LM. This is the kernel-level number we can attribute purely
to algorithmic / kernel choices.

Usage:
    python -m experiments.kernel_bench
    python -m experiments.kernel_bench --seq-lengths 2048 8192 \
        --num-layers 32 --num-heads 32 --head-dim 128 \
        --output-dir results/phase_a/kernel_bench
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from methods.registry import make_method
from ._common import (
    ExperimentRecord, cuda_sync, device_summary,
    warmup_then_time, write_records,
)


# ── Synthetic KV cache builder ───────────────────────────────────────────────

def make_synthetic_kv(
    n_layers: int, n_heads: int, head_dim: int, seq_len: int,
    batch: int = 1, device: Any = "cpu", dtype: torch.dtype = torch.float16,
):
    """Random past_key_values shaped exactly like HF Llama-style caches."""
    return tuple(
        (
            torch.randn(batch, n_heads, seq_len, head_dim,
                        device=device, dtype=dtype),
            torch.randn(batch, n_heads, seq_len, head_dim,
                        device=device, dtype=dtype),
        )
        for _ in range(n_layers)
    )


# ── Per-config measurement ───────────────────────────────────────────────────

def _bench_method(
    method_name: str,
    method_kwargs: Dict[str, Any],
    past_kv,
    n_warmup: int,
    n_iter: int,
    device: Any,
) -> Dict[str, float]:
    """
    Time `process_step` for a single method config.

    The inner loop calls process_step on the same past_kv repeatedly. The
    method's selection cache will hit after the first miss; we report this
    as cache_hit_rate so a low number flags a buggy timing setup.
    """
    method = make_method(method_name, **method_kwargs)
    method.process_prefill(past_kv)

    def step_once():
        method.process_step(past_kv, step=1)

    # Reset stats so warmup steps don't pollute the cache-hit counter.
    timing = warmup_then_time(step_once, n_warmup=n_warmup,
                              n_iter=n_iter, device=device)

    stats = method.get_stats() if hasattr(method, "get_stats") else {}
    timing["cache_hit_rate"] = stats.get("cache_hit_rate", 0.0)
    timing["throughput_tps"] = (
        1.0 / timing["mean_s"] if timing["mean_s"] > 0 else 0.0
    )
    timing["latency_ms"] = timing["mean_s"] * 1000.0
    return timing


# ── Main benchmark routine ───────────────────────────────────────────────────

def run_kernel_bench(
    seq_lengths: List[int],
    n_layers: int = 32,
    n_heads: int = 32,
    head_dim: int = 128,
    K: int = 1024,
    n_sink: int = 128,
    n_local: int = 512,
    n_warmup: int = 5,
    n_iter: int = 20,
    device: str = "cuda",
    output_dir: Path = Path("results/phase_a/kernel_bench"),
    # ── Phase A v2 fix-ups ────────────────────────────────────────────────
    use_selection_cache: bool = True,        # disable to expose real kernel cost
    include_hybrid: bool = False,            # add kivi_topk row when True
) -> List[ExperimentRecord]:
    """
    Run the bench across `seq_lengths` for {baseline, topk-pytorch, topk-triton}.
    Returns a list of ExperimentRecord rows AND writes CSV + JSON to disk.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("[kernel_bench] CUDA unavailable — falling back to CPU "
              "(triton path will be inactive).")
        device = "cpu"

    print(f"[kernel_bench] device={device}  "
          f"shape=(B=1, H={n_heads}, S=*, D={head_dim}) × {n_layers} layers")
    print(f"[kernel_bench] {device_summary(device)}")

    records: List[ExperimentRecord] = []

    # Per-method base config; selection cache is threaded in so the kernel
    # actually fires every step when False (Phase A v2 fix-up).
    topk_base = {
        "K": K, "n_sink": n_sink, "n_local": n_local,
        "refresh_interval": 0,
        "use_selection_cache": use_selection_cache,
    }
    configs = [
        ("baseline", "baseline", {}),
        ("topk_pytorch", "topk", {**topk_base, "use_kernels": False}),
        ("topk_triton",  "topk", {**topk_base, "use_kernels": True}),
    ]
    if include_hybrid:
        configs.append((
            "kivi_topk_hybrid_a", "kivi_topk",
            {**topk_base, "bits": 4, "group_size": 32, "residual_length": 128},
        ))
        configs.append((
            "kivi_topk_hybrid_c", "kivi_topk_c",
            {**topk_base, "bits": 4, "group_size": 32, "residual_length": 128},
        ))
    print(f"[kernel_bench] use_selection_cache={use_selection_cache} "
          f"(False = kernel fires every step)")

    for seq_len in seq_lengths:
        print(f"\n── seq_len = {seq_len} ─────────────────────────────────")
        past_kv = make_synthetic_kv(n_layers, n_heads, head_dim, seq_len,
                                    device=device)

        baseline_ms = None
        for label, name, kwargs in configs:
            timing = _bench_method(name, kwargs, past_kv,
                                   n_warmup, n_iter, device)
            ms = timing["latency_ms"]
            tps = timing["throughput_tps"]
            if label == "baseline":
                baseline_ms = ms
            speedup = baseline_ms / ms if baseline_ms and ms > 0 else 1.0
            print(f"  {label:14s}  "
                  f"{ms:7.3f} ms   "
                  f"({tps:7.0f} tok/s)   "
                  f"speedup={speedup:5.2f}×   "
                  f"hit_rate={timing['cache_hit_rate']:.0%}")

            records.append(ExperimentRecord(
                experiment="kernel_bench",
                method=label,
                config={
                    "seq_len":   seq_len,
                    "n_layers":  n_layers,
                    "n_heads":   n_heads,
                    "head_dim":  head_dim,
                    "K":         K if name == "topk" else None,
                    "use_kernels": kwargs.get("use_kernels"),
                    "device":    device,
                },
                metrics={
                    "latency_ms":     ms,
                    "throughput_tps": tps,
                    "speedup_vs_baseline": speedup,
                    "p50_ms":         timing["p50"] * 1000,
                    "p90_ms":         timing["p90"] * 1000,
                    "p99_ms":         timing["p99"] * 1000,
                    "std_ms":         timing["std_s"] * 1000,
                    "cache_hit_rate": timing["cache_hit_rate"],
                    "n_iter":         timing["n_iter"],
                },
            ))

    out_path = Path(output_dir) / "kernel_bench"
    write_records(records, out_path)
    print(f"\n[kernel_bench] Wrote {out_path}.csv and {out_path}.json")
    return records


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-lengths", type=int, nargs="+",
                   default=[2048, 4096, 8192])
    p.add_argument("--n-layers",  type=int, default=32)
    p.add_argument("--n-heads",   type=int, default=32)
    p.add_argument("--head-dim",  type=int, default=128)
    p.add_argument("--K",         type=int, default=1024)
    p.add_argument("--n-sink",    type=int, default=128)
    p.add_argument("--n-local",   type=int, default=512)
    p.add_argument("--n-warmup",  type=int, default=5)
    p.add_argument("--n-iter",    type=int, default=20)
    p.add_argument("--device",    type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_a/kernel_bench"))
    p.add_argument("--no-selection-cache", action="store_true",
                   help="Disable selection cache so the kernel fires every "
                        "step (exposes real Triton vs PyTorch delta).")
    p.add_argument("--include-hybrid", action="store_true",
                   help="Also bench the Phase B KIVI+TopK hybrid.")
    args = p.parse_args()

    run_kernel_bench(
        seq_lengths=args.seq_lengths,
        n_layers=args.n_layers, n_heads=args.n_heads,
        head_dim=args.head_dim,
        K=args.K, n_sink=args.n_sink, n_local=args.n_local,
        n_warmup=args.n_warmup, n_iter=args.n_iter,
        device=args.device, output_dir=args.output_dir,
        use_selection_cache=not args.no_selection_cache,
        include_hybrid=args.include_hybrid,
    )


if __name__ == "__main__":
    main()
