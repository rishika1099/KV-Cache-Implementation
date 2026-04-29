#!/usr/bin/env python3
"""
Unified CLI for all KV-cache experiments.

Sub-commands map 1:1 to modules in `experiments/`. Every sub-command
accepts the same `--device --output-dir --n-warmup --n-iter` flags so
scripting them in a sweep is trivial.

Examples:

    # Time the kernels in isolation across three seq lengths
    python run_experiment.py kernel-bench --seq-lengths 2048 4096 8192

    # Five-row leave-one-out ablation, kernel-only timing
    python run_experiment.py ablation --K 1024

    # Same ablation but also report PPL per row
    python run_experiment.py ablation --model meta-llama/Llama-2-7b-hf

    # K-sweep with accuracy: needs a real model
    python run_experiment.py k-sweep --model meta-llama/Llama-2-7b-hf

    # PyTorch vs Triton TB traces
    python run_experiment.py profile --seq-len 8192 --n-active-steps 30

    # Run-once smoke test: drives a single decode through process_step
    python run_experiment.py smoke --method topk --k 1024 --context 4096

    # List registered methods
    python run_experiment.py list-methods
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

# Make `methods` and `experiments` importable when this script is invoked
# from anywhere (`python run_experiment.py …`).
sys.path.insert(0, str(Path(__file__).parent))

from methods.registry import list_methods, make_method, describe          # noqa: E402
from experiments import (                                                   # noqa: E402
    kernel_bench, ablation, k_sweep, profile, long_context,
    passkey_retrieval,
)
from experiments._common import device_summary                              # noqa: E402
from experiments.kernel_bench import make_synthetic_kv                      # noqa: E402


# ── Sub-command: kernel-bench ────────────────────────────────────────────────

def _add_kernel_bench(sp):
    p = sp.add_parser("kernel-bench",
                      help="Compare baseline / topk-pytorch / topk-triton "
                           "decode-step latency at multiple seq lengths.")
    p.add_argument("--seq-lengths", type=int, nargs="+",
                   default=[2048, 4096, 8192])
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads",  type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--n-sink",  type=int, default=128)
    p.add_argument("--n-local", type=int, default=512)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter",   type=int, default=20)
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_a/kernel_bench"))
    p.add_argument("--no-selection-cache", action="store_true",
                   help="Disable selection cache (Phase A v2 fix — kernel "
                        "fires every step).")
    p.add_argument("--include-hybrid", action="store_true",
                   help="Also bench the Phase B KIVI+TopK hybrid.")
    p.set_defaults(_run=lambda a: kernel_bench.run_kernel_bench(
        seq_lengths=a.seq_lengths,
        n_layers=a.n_layers, n_heads=a.n_heads, head_dim=a.head_dim,
        K=a.K, n_sink=a.n_sink, n_local=a.n_local,
        n_warmup=a.n_warmup, n_iter=a.n_iter,
        device=a.device, output_dir=a.output_dir,
        use_selection_cache=not a.no_selection_cache,
        include_hybrid=a.include_hybrid,
    ))


# ── Sub-command: ablation ────────────────────────────────────────────────────

def _add_ablation(sp):
    p = sp.add_parser("ablation",
                      help="Leave-one-out (or full-grid) ablation of the "
                           "five TopK novelty flags.")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads",  type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--n-sink",  type=int, default=128)
    p.add_argument("--n-local", type=int, default=512)
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--full-grid", action="store_true",
                   help="Run all 32 combinations.")
    p.add_argument("--model", type=str, default=None,
                   help="HF model name → also reports PPL per row.")
    p.add_argument("--n-ppl-examples", type=int, default=50)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iter",   type=int, default=20)
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_a/ablation"))
    p.set_defaults(_run=lambda a: ablation.run_ablation(
        seq_len=a.seq_len,
        n_layers=a.n_layers, n_heads=a.n_heads, head_dim=a.head_dim,
        K=a.K, n_sink=a.n_sink, n_local=a.n_local,
        use_kernels=not a.no_kernels,
        n_warmup=a.n_warmup, n_iter=a.n_iter,
        full_grid=a.full_grid,
        model_name=a.model, n_ppl_examples=a.n_ppl_examples,
        device=a.device, output_dir=a.output_dir,
    ))


# ── Sub-command: k-sweep ─────────────────────────────────────────────────────

def _add_k_sweep(sp):
    p = sp.add_parser("k-sweep",
                      help="Sweep K ∈ {128, …, 4096}; report latency, "
                           "throughput and (optional) PPL per K.")
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
    p.set_defaults(_run=lambda a: k_sweep.run_k_sweep(
        K_values=a.K_values,
        seq_len=a.seq_len,
        n_layers=a.n_layers, n_heads=a.n_heads, head_dim=a.head_dim,
        n_sink=a.n_sink, n_local=a.n_local,
        use_kernels=not a.no_kernels,
        n_warmup=a.n_warmup, n_iter=a.n_iter,
        model_name=a.model, n_ppl_examples=a.n_ppl_examples,
        max_length=a.max_length,
        device=a.device, output_dir=a.output_dir,
    ))


# ── Sub-command: profile ─────────────────────────────────────────────────────

def _add_profile(sp):
    p = sp.add_parser("profile",
                      help="Capture torch.profiler TB traces for the PyTorch "
                           "and Triton TopK paths.")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads",  type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--n-sink",   type=int, default=128)
    p.add_argument("--n-local",  type=int, default=512)
    p.add_argument("--n-warmup-steps", type=int, default=3)
    p.add_argument("--n-active-steps", type=int, default=10)
    p.add_argument("--no-pytorch", action="store_true")
    p.add_argument("--no-triton",  action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_a/profile"))
    p.set_defaults(_run=lambda a: profile.run_profile(
        seq_len=a.seq_len,
        n_layers=a.n_layers, n_heads=a.n_heads, head_dim=a.head_dim,
        K=a.K, n_sink=a.n_sink, n_local=a.n_local,
        n_warmup_steps=a.n_warmup_steps,
        n_active_steps=a.n_active_steps,
        profile_pytorch=not a.no_pytorch,
        profile_triton=not a.no_triton,
        device=a.device, output_dir=a.output_dir,
    ))


# ── Sub-command: long-context  (Phase B headline experiment) ────────────────

def _add_long_context(sp):
    p = sp.add_parser("long-context",
                      help="Phase B: scale all four methods (baseline / "
                           "kivi / topk / kivi_topk) across long seq_lens.")
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[2048, 8192, 16384])
    p.add_argument("--methods", type=str, nargs="+",
                   default=["baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c"])
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
                   help="Default OFF — isolates kernel cost (Phase A v2 fix).")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--n-ppl-examples", type=int, default=30)
    p.add_argument("--max-length", type=int, default=4096,
                   help="Must be >= 4×budget for TopK to engage on PPL.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_b/long_context"))
    p.set_defaults(_run=lambda a: long_context.run_long_context(
        seq_lens=a.seq_lens, methods=a.methods,
        n_layers=a.n_layers, n_heads=a.n_heads, head_dim=a.head_dim,
        K=a.K, n_sink=a.n_sink, n_local=a.n_local,
        bits=a.bits, group_size=a.group_size,
        residual_length=a.residual_length,
        n_warmup=a.n_warmup, n_iter=a.n_iter,
        use_selection_cache=a.use_selection_cache,
        model_name=a.model,
        n_ppl_examples=a.n_ppl_examples, max_length=a.max_length,
        device=a.device, output_dir=a.output_dir,
    ))


# ── Sub-command: passkey-retrieval  (Phase B quality eval) ─────────────────

def _add_passkey(sp):
    p = sp.add_parser("passkey-retrieval",
                      help="Phase B: needle-in-haystack quality eval that "
                           "actually exercises process_step (fixes the "
                           "PPL methodology gap).")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192])
    p.add_argument("--methods",  type=str, nargs="+",
                   default=["baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c"])
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--n-depths", type=int, default=5)
    p.add_argument("--K",        type=int, default=1024)
    p.add_argument("--n-sink",   type=int, default=128)
    p.add_argument("--n-local",  type=int, default=512)
    p.add_argument("--bits",            type=int, default=4)
    p.add_argument("--group-size",      type=int, default=32)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_b/passkey"))
    p.set_defaults(_run=lambda a: passkey_retrieval.run_passkey_retrieval(
        model_name=a.model,
        seq_lens=a.seq_lens, methods=a.methods,
        n_trials=a.n_trials, n_depths=a.n_depths,
        K=a.K, n_sink=a.n_sink, n_local=a.n_local,
        bits=a.bits, group_size=a.group_size,
        residual_length=a.residual_length,
        max_new_tokens=a.max_new_tokens,
        seed=a.seed, device=a.device, output_dir=a.output_dir,
    ))


# ── Sub-command: smoke (single-shot run; useful for quick checks) ────────────

def _add_smoke(sp):
    p = sp.add_parser("smoke",
                      help="Run a single decode step on a synthetic KV "
                           "cache for one method. Quick sanity check.")
    p.add_argument("--method", type=str, default="topk",
                   choices=list_methods())
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--n-sink",  type=int, default=128)
    p.add_argument("--n-local", type=int, default=512)
    p.add_argument("--context", type=int, default=4096,
                   help="Synthetic KV-cache seq length.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="(Phase C will lift the >1 limit; for now this "
                        "value is recorded but enforced as 1.)")
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--device", type=str, default="cuda")

    def _run(a):
        if a.batch_size != 1:
            print("[smoke] WARNING: batch_size>1 lands in Phase C; "
                  "running with batch_size=1.")
            a.batch_size = 1

        device = a.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[smoke] CUDA unavailable — using CPU.")
            device = "cpu"
        print(f"[smoke] {device_summary(device)}")

        past_kv = make_synthetic_kv(32, 32, 128, a.context, device=device)
        method = make_method(
            a.method,
            K=a.k, n_sink=a.n_sink, n_local=a.n_local,
            refresh_interval=0, use_kernels=not a.no_kernels,
        )
        method.process_prefill(past_kv)
        out = method.process_step(past_kv, step=1)
        sel = out[0][0].shape[2]
        kv_mb = method.get_kv_size_bytes(past_kv) / 1e6
        print(f"[smoke] method={a.method}  context={a.context}  "
              f"selected={sel}  kv={kv_mb:.2f} MB")
        if hasattr(method, "get_stats"):
            print(f"[smoke] stats={method.get_stats()}")
        if hasattr(method, "get_ablation_config"):
            print(f"[smoke] config={method.get_ablation_config()}")
        print("[smoke] PASSED")

    p.set_defaults(_run=_run)


# ── Sub-command: list-methods ────────────────────────────────────────────────

def _add_list_methods(sp):
    p = sp.add_parser("list-methods",
                      help="Print all methods in the registry.")
    def _run(_a):
        for name in list_methods():
            d = describe(name)
            print(f"  {name:14s}  {d['description']}")
    p.set_defaults(_run=_run)


# ── Main dispatch ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp = p.add_subparsers(dest="cmd", required=True)
    _add_kernel_bench(sp)
    _add_ablation(sp)
    _add_k_sweep(sp)
    _add_profile(sp)
    _add_long_context(sp)
    _add_passkey(sp)
    _add_smoke(sp)
    _add_list_methods(sp)

    args = p.parse_args()
    args._run(args)


if __name__ == "__main__":
    main()
