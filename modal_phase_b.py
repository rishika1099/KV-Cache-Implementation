"""
Modal entrypoint for Phase B experiments.

Phase B builds on Phase A by adding:

    1. long_context        — scaling sweep (2K/8K/16K/32K) across all four
                              methods incl. the new KIVI+TopK hybrid
    2. long_context_ppl    — same sweep but on real LLaMA-2-7B with
                              max_length ≥ 4×budget so TopK actually engages
    3. kernel_bench_v2     — Phase A v1 fix-up: re-run kernel bench with
                              `use_selection_cache=False` so the Triton vs
                              PyTorch comparison reflects real kernel cost
    4. k_sweep_v2          — Phase A v1 fix-up: re-run K-sweep with PPL
                              using max_length=4096 so selection engages

The first two are Phase B headline plots; the latter two close the
methodology gaps in Phase A (97% selection-cache hit rate masking the
kernel; PPL flat at 8.401 because budget never engaged).

Pulls Phase A image template — same A100-80GB containers, same KV-cache
results volume, same HF cache mount — and reuses the gpu_function helper
for parity. New experiments slot in alongside the Phase A ones in the
same volume folder structure.

Usage:
    modal run modal_phase_b.py                          # all four
    modal run modal_phase_b.py --skip-ppl               # latency only
    modal run modal_phase_b.py --only long_context
    modal run modal_phase_b.py --only kernel_bench_v2

Pull results locally with:
    modal volume get kv-benchmark-results /phase_b ./results/phase_b
    modal volume get kv-benchmark-results /phase_a_v2 ./results/phase_a_v2
"""

import sys
from pathlib import Path

import modal


# ── Modal infrastructure (same image as Phase A) ─────────────────────────────

app = modal.App("kv-benchmark-phase-b")

_base = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "datasets>=2.18.0,<4.0",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "tabulate>=0.9.0",
        "pandas>=2.0.0",
    )
    .add_local_dir(str(_base / "methods"),     "/app/methods")
    .add_local_dir(str(_base / "benchmark"),   "/app/benchmark")
    .add_local_dir(str(_base / "experiments"), "/app/experiments")
)

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)

RESULTS_PATH    = Path("/results")
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
PHASE_B_DIR     = "phase_b"
PHASE_A_V2_DIR  = "phase_a_v2"


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def gpu_function(name: str, timeout: int = 3600, memory: int = 16384):
    """All Phase B GPU functions use the same decorator stack."""
    return app.function(
        image=image,
        gpu="a100-80gb",
        secrets=[modal.Secret.from_name("huggingface")],
        volumes={
            str(RESULTS_PATH):  results_vol,
            str(HF_CACHE_PATH): model_cache_vol,
        },
        timeout=timeout,
        memory=memory,
        name=name,
    )


# ── 1. long_context (latency only — synthetic KV, no model) ──────────────────

@gpu_function("phase_b_long_context", timeout=3600)
def run_long_context_kernel(
    seq_lens: list = [2048, 8192, 16384, 32768],
):
    _bootstrap()
    from experiments.long_context import run_long_context

    out_dir = RESULTS_PATH / PHASE_B_DIR / "long_context"
    records = run_long_context(
        seq_lens=seq_lens,
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        n_warmup=5, n_iter=15,
        use_selection_cache=False,           # ← real kernel cost
        model_name=None,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 2. long_context_ppl (real model — fewer seq lengths, includes PPL) ───────

@gpu_function("phase_b_long_context_ppl", timeout=5400, memory=32768)
def run_long_context_with_ppl(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    seq_lens: list = [2048, 8192],         # 16K+ would need extended-context model
    n_ppl_examples: int = 30,
):
    _bootstrap()
    from experiments.long_context import run_long_context

    out_dir = RESULTS_PATH / PHASE_B_DIR / "long_context_ppl"
    records = run_long_context(
        seq_lens=seq_lens,
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        n_warmup=3, n_iter=10,
        use_selection_cache=True,            # PPL run — cache reuse on
        model_name=model_name,
        n_ppl_examples=n_ppl_examples,
        max_length=4096,                     # ≥ 4×budget(1664) so TopK engages
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 3. kernel_bench_v2 (Phase A fix-up: cache OFF + hybrid included) ─────────

@gpu_function("phase_b_kernel_bench_v2", timeout=1800)
def run_kernel_bench_v2(
    seq_lengths: list = [2048, 4096, 8192, 16384],
):
    _bootstrap()
    from experiments.kernel_bench import run_kernel_bench

    out_dir = RESULTS_PATH / PHASE_A_V2_DIR / "kernel_bench"
    records = run_kernel_bench(
        seq_lengths=seq_lengths,
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        n_warmup=10, n_iter=30,
        device="cuda", output_dir=out_dir,
        use_selection_cache=False,           # kernel fires every step
        include_hybrid=True,                 # add kivi_topk row
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 4. k_sweep_v2 (Phase A fix-up: max_length=4096 so PPL varies) ────────────

@gpu_function("phase_b_k_sweep_v2", timeout=3600, memory=32768)
def run_k_sweep_v2(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    n_ppl_examples: int = 50,
    K_values: list = [128, 256, 512, 1024, 2048, 4096],
):
    _bootstrap()
    from experiments.k_sweep import run_k_sweep

    out_dir = RESULTS_PATH / PHASE_A_V2_DIR / "k_sweep"
    records = run_k_sweep(
        K_values=K_values,
        seq_len=4096,
        n_layers=32, n_heads=32, head_dim=128,
        n_sink=128, n_local=512,
        use_kernels=True,
        n_warmup=5, n_iter=15,
        model_name=model_name, n_ppl_examples=n_ppl_examples,
        max_length=4096,                      # ← was 512; now ≥ budget
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(skip_ppl: bool = False, only: str = ""):
    """
    Dispatch Phase B jobs in parallel on A100-80GB.

    --skip-ppl  →  skip the two model-loading runs (no PPL, faster, cheaper)
    --only X    →  run only experiment X
                   (long_context | long_context_ppl |
                    kernel_bench_v2 | k_sweep_v2)
    """
    jobs = {
        "long_context":      (run_long_context_kernel,    {}),
        "long_context_ppl":  (run_long_context_with_ppl,  {}),
        "kernel_bench_v2":   (run_kernel_bench_v2,        {}),
        "k_sweep_v2":        (run_k_sweep_v2,             {}),
    }

    if only:
        if only not in jobs:
            print(f"Unknown experiment '{only}'. Choose from: {list(jobs)}")
            return
        selected = {only: jobs[only]}
    elif skip_ppl:
        selected = {k: v for k, v in jobs.items()
                    if k not in ("long_context_ppl", "k_sweep_v2")}
    else:
        selected = jobs

    print(f"\n[phase_b] Dispatching {len(selected)} experiments to A100s "
          f"in parallel:\n  {', '.join(selected)}\n")

    handles = {name: fn.spawn(**kw) for name, (fn, kw) in selected.items()}

    print(f"[phase_b] Spawned. Waiting for completion…\n")

    for name, h in handles.items():
        try:
            res = h.get()
            print(f"[phase_b] ✅ {name}: {res}")
        except Exception as e:
            print(f"[phase_b] ❌ {name} failed: {e}")

    print("\n[phase_b] All done. Pull results with:")
    print(f"  modal volume get kv-benchmark-results /{PHASE_B_DIR} "
          f"./results/{PHASE_B_DIR}")
    print(f"  modal volume get kv-benchmark-results /{PHASE_A_V2_DIR} "
          f"./results/{PHASE_A_V2_DIR}")
