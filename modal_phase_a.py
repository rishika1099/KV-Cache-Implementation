"""
Modal entrypoint for Phase A experiments.

Dispatches the four Phase A experiments to A100-80GB containers in
parallel. Each Modal function runs one experiment module from
`experiments/` and writes its CSV + JSON output into a shared modal
volume so the local download script picks them up.

    1. kernel_bench   — decode-step timing for baseline / topk-pytorch / topk-triton
    2. ablation       — leave-one-out (5 flags + full = 6 rows)
    3. k_sweep        — K ∈ {128, 256, 512, 1024, 2048, 4096}
    4. profile        — torch.profiler TB traces for both PyTorch & Triton paths

The model-loading experiments (ablation+PPL, k_sweep+PPL) run with the
real LLaMA-2-7B; kernel_bench and profile run on synthetic KV caches.

Usage:
    modal run modal_phase_a.py                   # all four
    modal run modal_phase_a.py --skip-ppl        # latency-only (no model)
    modal run modal_phase_a.py --only kernel_bench

Pull results locally with:
    modal volume get kv-benchmark-results /phase_a ./results/phase_a
"""

import sys
from pathlib import Path

import modal


# ── Modal infrastructure ─────────────────────────────────────────────────────

app = modal.App("kv-benchmark-phase-a")

_base = Path(__file__).parent

# triton ships with the torch wheel — no separate pip needed
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

RESULTS_PATH  = Path("/results")
HF_CACHE_PATH = Path("/root/.cache/huggingface")
PHASE_DIR     = "phase_a"   # folder inside the volume


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


# ── Shared GPU function decorator ────────────────────────────────────────────

def gpu_function(name: str, timeout: int = 3600, memory: int = 16384):
    """All Phase A GPU functions use the same decorator stack."""
    return app.function(
        image=image,
        gpu="a100-80gb",
        secrets=[modal.Secret.from_name("huggingface")],
        volumes={
            str(RESULTS_PATH):   results_vol,
            str(HF_CACHE_PATH): model_cache_vol,
        },
        timeout=timeout,
        memory=memory,
        name=name,
    )


# ── 1. kernel_bench ──────────────────────────────────────────────────────────

@gpu_function("phase_a_kernel_bench", timeout=1800)
def run_kernel_bench(
    seq_lengths: list = [2048, 4096, 8192, 16384],
    n_warmup: int = 10,
    n_iter: int = 30,
):
    _bootstrap()
    from experiments.kernel_bench import run_kernel_bench

    out_dir = RESULTS_PATH / PHASE_DIR / "kernel_bench"
    records = run_kernel_bench(
        seq_lengths=seq_lengths,
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        n_warmup=n_warmup, n_iter=n_iter,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 2. ablation (latency only) ───────────────────────────────────────────────

@gpu_function("phase_a_ablation_kernel", timeout=1800)
def run_ablation_kernel():
    _bootstrap()
    from experiments.ablation import run_ablation

    out_dir = RESULTS_PATH / PHASE_DIR / "ablation_kernel"
    records = run_ablation(
        seq_len=4096,
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        use_kernels=True,
        n_warmup=10, n_iter=30,
        full_grid=False,
        model_name=None,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 3. ablation + PPL (real model) ───────────────────────────────────────────

@gpu_function("phase_a_ablation_ppl", timeout=3600, memory=32768)
def run_ablation_with_ppl(model_name: str = "meta-llama/Llama-2-7b-hf",
                          n_ppl_examples: int = 50):
    _bootstrap()
    from experiments.ablation import run_ablation

    out_dir = RESULTS_PATH / PHASE_DIR / "ablation_ppl"
    records = run_ablation(
        seq_len=4096,
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        use_kernels=True,
        n_warmup=5, n_iter=15,
        full_grid=False,
        model_name=model_name, n_ppl_examples=n_ppl_examples,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 4. K sweep + PPL ─────────────────────────────────────────────────────────

@gpu_function("phase_a_k_sweep", timeout=3600, memory=32768)
def run_k_sweep_full(model_name: str = "meta-llama/Llama-2-7b-hf",
                     n_ppl_examples: int = 50,
                     K_values: list = [128, 256, 512, 1024, 2048, 4096]):
    _bootstrap()
    from experiments.k_sweep import run_k_sweep

    out_dir = RESULTS_PATH / PHASE_DIR / "k_sweep"
    records = run_k_sweep(
        K_values=K_values,
        seq_len=4096,
        n_layers=32, n_heads=32, head_dim=128,
        n_sink=128, n_local=512,
        use_kernels=True,
        n_warmup=5, n_iter=15,
        model_name=model_name, n_ppl_examples=n_ppl_examples,
        max_length=512,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 5. torch.profiler ────────────────────────────────────────────────────────

@gpu_function("phase_a_profile", timeout=1800)
def run_profile_traces():
    _bootstrap()
    from experiments.profile import run_profile

    out_dir = RESULTS_PATH / PHASE_DIR / "profile"
    run_profile(
        seq_len=8192,
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        n_warmup_steps=3, n_active_steps=20,
        profile_pytorch=True, profile_triton=True,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"out_dir": str(out_dir)}


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(skip_ppl: bool = False, only: str = ""):
    """
    Dispatch Phase A jobs. By default runs all five in parallel.

    --skip-ppl  →  skip the two model-loading runs (cheaper, faster, no PPL)
    --only X    →  run only experiment X (kernel_bench | ablation_kernel
                                          | ablation_ppl | k_sweep | profile)
    """
    # Map of name → (function, kwargs)
    jobs = {
        "kernel_bench":    (run_kernel_bench,    {}),
        "ablation_kernel": (run_ablation_kernel, {}),
        "ablation_ppl":    (run_ablation_with_ppl, {}),
        "k_sweep":         (run_k_sweep_full,    {}),
        "profile":         (run_profile_traces,  {}),
    }

    if only:
        if only not in jobs:
            print(f"Unknown experiment '{only}'. Choose from: {list(jobs)}")
            return
        selected = {only: jobs[only]}
    elif skip_ppl:
        selected = {k: v for k, v in jobs.items()
                    if k not in ("ablation_ppl", "k_sweep")}
    else:
        selected = jobs

    print(f"\n[phase_a] Dispatching {len(selected)} experiments to A100s "
          f"in parallel:\n  {', '.join(selected)}\n")

    # Spawn all in parallel (.spawn returns a FunctionCall handle).
    handles = {name: fn.spawn(**kw) for name, (fn, kw) in selected.items()}

    print(f"[phase_a] Spawned. Waiting for completion…\n")

    # Block on each. Modal multiplexes their stdout into our terminal.
    for name, h in handles.items():
        try:
            res = h.get()
            print(f"[phase_a] ✅ {name}: {res}")
        except Exception as e:
            print(f"[phase_a] ❌ {name} failed: {e}")

    print("\n[phase_a] All done. Pull results with:")
    print(f"  modal volume get kv-benchmark-results /{PHASE_DIR} "
          f"./results/{PHASE_DIR}")
