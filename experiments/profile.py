"""
torch.profiler integration for the TopK decode path.

Captures CPU + CUDA traces (kernel launches, memory ops, time per
function) for two configurations side by side and exports them in
TensorBoard format:

    profile/topk_pytorch/   — TopK decode with kernels OFF
    profile/topk_triton/    — TopK decode with kernels ON

Open with:

    tensorboard --logdir results/phase_a/profile

The profiler runs over a configurable number of decode steps. Only the
final `n_active` steps are recorded (warmup/wait phases are skipped) so
the trace is dominated by steady-state behaviour rather than first-call
JIT compilation.

Also writes a plain-text summary table (`*_summary.txt`) ranked by
`self_cuda_time_total` so you can grep the hottest kernel without
opening TensorBoard.

Usage:
    python -m experiments.profile
    python -m experiments.profile --seq-len 8192 --n-active 30
    python -m experiments.profile --no-pytorch    # only profile triton path
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import torch
from torch.profiler import (
    ProfilerActivity, profile, record_function, schedule,
    tensorboard_trace_handler,
)

from methods.registry import make_method
from ._common import device_summary
from .kernel_bench import make_synthetic_kv


def _activities():
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    return acts


def _profile_one(
    label: str,
    use_kernels: bool,
    past_kv,
    K: int, n_sink: int, n_local: int,
    n_warmup_steps: int, n_active_steps: int,
    out_dir: Path, device: Any,
):
    """
    Profile one configuration and write trace + summary to `out_dir`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    method = make_method(
        "topk",
        K=K, n_sink=n_sink, n_local=n_local,
        refresh_interval=0, use_kernels=use_kernels,
    )
    method.process_prefill(past_kv)

    # Profiler schedule:
    #   wait    → no recording (lets things settle)
    #   warmup  → activities tracked but trace discarded (avoids JIT noise)
    #   active  → recorded into the final trace
    sched = schedule(wait=1, warmup=n_warmup_steps, active=n_active_steps)
    handler = tensorboard_trace_handler(str(out_dir))

    print(f"[profile] {label}: tracing {n_active_steps} active steps "
          f"(after {n_warmup_steps} warmup) → {out_dir}")

    with profile(
        activities=_activities(),
        schedule=sched,
        on_trace_ready=handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,    # stacks bloat the trace; opt-in if needed
    ) as prof:
        # Total scheduled steps = wait + warmup + active.
        n_total = 1 + n_warmup_steps + n_active_steps
        for step in range(n_total):
            with record_function("decode_step"):
                method.process_step(past_kv, step=step)
            prof.step()

    # Plain-text summary, ranked by self GPU time when available.
    sort_key = ("self_cuda_time_total" if torch.cuda.is_available()
                else "self_cpu_time_total")
    summary = prof.key_averages().table(sort_by=sort_key, row_limit=25)
    (out_dir / "summary.txt").write_text(summary)
    print(f"[profile] {label}: wrote summary.txt and TB trace ({out_dir})")


def run_profile(
    seq_len: int = 4096,
    n_layers: int = 32, n_heads: int = 32, head_dim: int = 128,
    K: int = 1024, n_sink: int = 128, n_local: int = 512,
    n_warmup_steps: int = 3, n_active_steps: int = 10,
    profile_pytorch: bool = True, profile_triton: bool = True,
    device: str = "cuda",
    output_dir: Path = Path("results/phase_a/profile"),
):
    if device == "cuda" and not torch.cuda.is_available():
        print("[profile] CUDA unavailable — profiling on CPU "
              "(triton path inactive).")
        device = "cpu"

    print(f"[profile] {device_summary(device)}")
    past_kv = make_synthetic_kv(n_layers, n_heads, head_dim, seq_len,
                                device=device)

    if profile_pytorch:
        _profile_one(
            "topk_pytorch", use_kernels=False, past_kv=past_kv,
            K=K, n_sink=n_sink, n_local=n_local,
            n_warmup_steps=n_warmup_steps, n_active_steps=n_active_steps,
            out_dir=output_dir / "topk_pytorch", device=device,
        )
    if profile_triton:
        _profile_one(
            "topk_triton", use_kernels=True, past_kv=past_kv,
            K=K, n_sink=n_sink, n_local=n_local,
            n_warmup_steps=n_warmup_steps, n_active_steps=n_active_steps,
            out_dir=output_dir / "topk_triton", device=device,
        )

    print(f"\n[profile] open with:  tensorboard --logdir {output_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--n-heads",  type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--K",        type=int, default=1024)
    p.add_argument("--n-sink",   type=int, default=128)
    p.add_argument("--n-local",  type=int, default=512)
    p.add_argument("--n-warmup-steps", type=int, default=3)
    p.add_argument("--n-active-steps", type=int, default=10)
    p.add_argument("--no-pytorch", action="store_true",
                   help="Skip the PyTorch reference profile.")
    p.add_argument("--no-triton", action="store_true",
                   help="Skip the Triton kernel profile.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_a/profile"))
    args = p.parse_args()

    run_profile(
        seq_len=args.seq_len,
        n_layers=args.n_layers, n_heads=args.n_heads, head_dim=args.head_dim,
        K=args.K, n_sink=args.n_sink, n_local=args.n_local,
        n_warmup_steps=args.n_warmup_steps,
        n_active_steps=args.n_active_steps,
        profile_pytorch=not args.no_pytorch,
        profile_triton=not args.no_triton,
        device=args.device, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
