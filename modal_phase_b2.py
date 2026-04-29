"""
Modal entrypoint for Phase B v2 — closes the methodology + implementation
gaps we discovered after Phase B v1.

What v1 left on the table:
  • Hybrid latency was Python-overhead-bound (per-step torch.cat over the
    centroid list) — the algorithmic story (32× cheaper scoring) was
    invisible. Fixed by pre-stacking the centroid tensor; cat happens at
    most once per block-seal event.
  • PPL was structurally blind to TopK selection because
    `compute_method_perplexity` runs prefill + a single forward on the
    eval window, never calling `process_step`. So PPL was flat across K
    values for every TopK-family method (8.401 constant in Phase A v2).
  • Hybrid had only design (a) — centroid scoring, an approximation. We
    didn't have the per-token-resolution path.

This phase adds:
  1. long_context_v2     — same sweep as v1 with the fixed (a) and the
                            new (c). Both share the FP16 fast path so a
                            head-to-head latency comparison is meaningful.
  2. passkey_retrieval   — needle-in-haystack at 4K/8K, runs through
                            generate_with_method (which DOES exercise
                            process_step). Five depths × five trials × five
                            methods. Real quality signal.
  3. kernel_bench_v3     — same as v2 but includes both hybrids. Selection
                            cache OFF so the kernel fires every step.
  4. ppl_sanity          — short LLaMA-2-7B PPL pass to confirm v2's
                            (a)-fixed and the new (c) don't regress KIVI's
                            7.268 baseline by more than O(0.05). Quick
                            check; the headline quality number is passkey.

Pulls Phase A image template (same A100-80GB containers, same KV-cache
results volume, same HF cache mount). Each function `.spawn()`s in
parallel via `app.local_entrypoint`.

Usage:
    modal run modal_phase_b2.py                           # all four
    modal run modal_phase_b2.py --skip-quality            # latency only
    modal run modal_phase_b2.py --only passkey

Pull results locally with:
    modal volume get kv-benchmark-results /phase_b2 ./results/phase_b2
"""

import sys
from pathlib import Path

import modal


# ── Modal infrastructure (same image as Phase A / B v1) ─────────────────────

app = modal.App("kv-benchmark-phase-b2")

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
PHASE_B2_DIR    = "phase_b2"


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def gpu_function(name: str, timeout: int = 3600, memory: int = 16384):
    """All Phase B2 GPU functions share the same decorator stack."""
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


# ── 1. long_context_v2 — both hybrids head-to-head ──────────────────────────

@gpu_function("phase_b2_long_context", timeout=3600)
def run_long_context_v2(
    seq_lens: list = [2048, 8192, 16384, 32768],
):
    """
    Same harness as v1 long_context. Critically the (a) path now uses the
    pre-stacked centroid tensor — its decode-step latency on this run is
    the *real* design-(a) cost, not the Python-overhead artifact from v1.
    The (c) row is brand-new and fires the Triton quant-aware kernel.
    """
    _bootstrap()
    from experiments.long_context import run_long_context

    out_dir = RESULTS_PATH / PHASE_B2_DIR / "long_context"
    records = run_long_context(
        seq_lens=seq_lens,
        methods=["baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c"],
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        n_warmup=5, n_iter=15,
        use_selection_cache=False,           # ← real kernel cost every step
        model_name=None,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 2. passkey_retrieval — needle-in-haystack quality eval ──────────────────

@gpu_function("phase_b2_passkey", timeout=5400, memory=32768)
def run_passkey(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    seq_lens: list = [4096, 8192],
    n_trials: int = 5,
    n_depths: int = 5,
):
    """
    Goes through `generate_with_method` so process_step is actually
    invoked — fixes the PPL methodology gap that left v1 blind to TopK
    selection quality. Reports per-depth accuracy, which is the
    discriminating quality metric for top-K methods at long context.
    """
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    out_dir = RESULTS_PATH / PHASE_B2_DIR / "passkey"
    records = run_passkey_retrieval(
        model_name=model_name,
        seq_lens=seq_lens,
        methods=["baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c"],
        n_trials=n_trials, n_depths=n_depths,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        max_new_tokens=12,
        seed=0, device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 3. kernel_bench_v3 — both hybrids, no selection cache ───────────────────

@gpu_function("phase_b2_kernel_bench", timeout=1800)
def run_kernel_bench_v3(
    seq_lengths: list = [2048, 4096, 8192, 16384],
):
    """Same shape as v2 but `include_hybrid=True` now adds both (a) and (c)."""
    _bootstrap()
    from experiments.kernel_bench import run_kernel_bench

    out_dir = RESULTS_PATH / PHASE_B2_DIR / "kernel_bench"
    records = run_kernel_bench(
        seq_lengths=seq_lengths,
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        n_warmup=10, n_iter=30,
        device="cuda", output_dir=out_dir,
        use_selection_cache=False,
        include_hybrid=True,                 # both (a) and (c) added
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 4. ppl_sanity — short PPL run to confirm both hybrids are well-behaved ──

@gpu_function("phase_b2_ppl_sanity", timeout=3600, memory=32768)
def run_ppl_sanity(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    n_ppl_examples: int = 30,
):
    """
    Sanity-only PPL pass. NOTE this still has the v1 methodology limitation
    (compute_method_perplexity does not call process_step) — it tells us
    whether our cache-side modifications corrupted something at prefill,
    not whether the decode-time selection is good. The headline quality
    number is in `passkey`.
    """
    _bootstrap()
    from experiments.long_context import run_long_context

    out_dir = RESULTS_PATH / PHASE_B2_DIR / "ppl_sanity"
    records = run_long_context(
        seq_lens=[2048],
        methods=["baseline", "kivi", "kivi_topk", "kivi_topk_c"],
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        n_warmup=2, n_iter=5,
        use_selection_cache=True,
        model_name=model_name,
        n_ppl_examples=n_ppl_examples,
        max_length=4096,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── Local entrypoint ────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(skip_quality: bool = False, only: str = ""):
    """
    --skip-quality  →  skip the two model-loading runs (passkey + ppl_sanity)
    --only X        →  only run experiment X
                       (long_context | passkey | kernel_bench | ppl_sanity)
    """
    jobs = {
        "long_context": (run_long_context_v2,  {}),
        "passkey":      (run_passkey,          {}),
        "kernel_bench": (run_kernel_bench_v3,  {}),
        "ppl_sanity":   (run_ppl_sanity,       {}),
    }

    if only:
        if only not in jobs:
            print(f"Unknown experiment '{only}'. Choose from: {list(jobs)}")
            return
        selected = {only: jobs[only]}
    elif skip_quality:
        selected = {k: v for k, v in jobs.items()
                    if k not in ("passkey", "ppl_sanity")}
    else:
        selected = jobs

    print(f"\n[phase_b2] Dispatching {len(selected)} experiments to A100s "
          f"in parallel:\n  {', '.join(selected)}\n")

    handles = {name: fn.spawn(**kw) for name, (fn, kw) in selected.items()}
    print(f"[phase_b2] Spawned. Waiting for completion…\n")

    for name, h in handles.items():
        try:
            res = h.get()
            print(f"[phase_b2] ✅ {name}: {res}")
        except Exception as e:
            print(f"[phase_b2] ❌ {name} failed: {e}")

    print("\n[phase_b2] All done. Pull results with:")
    print(f"  modal volume get kv-benchmark-results /{PHASE_B2_DIR} "
          f"./results/{PHASE_B2_DIR}")
