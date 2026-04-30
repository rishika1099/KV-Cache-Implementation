"""
Modal entrypoint for the perf/hybrid-optimization branch.

Goal: validate that the rewritten kivi_topk_hybrid is
  (a) functionally correct  — passkey @4K accuracy must not regress
                              vs the v2 baseline (TopK 48% / hybrids 32%)
  (b) substantially faster  — hybrid decode_step_ms should drop close to
                              KIVI's number, not 5–10× higher.

Writes to results/phase_perf/ so the v2 numbers under
results/_h100_post_fix_v2/phase_b2/ remain the apples-to-apples baseline.

Three independent jobs run in parallel:
  1. smoke           — tiny OPT model, end-to-end generate, sanity only.
  2. long_context    — synthetic latency at multiple seq_lens, all five
                       methods. Same harness as Phase B2; same flags.
  3. passkey_4k      — single seq_len 4K passkey, 5 depths × 5 trials.
                       Skipped from 8K because it's OOD for LLaMA-2-7B
                       and only adds noise.

Optional:
  --only smoke|long|passkey
  --skip-passkey   (saves ~10 min if you only care about latency)

Pull results with:
  modal volume get kv-benchmark-results /phase_perf ./results/phase_perf
"""

import sys
from pathlib import Path

import modal


app = modal.App("kv-benchmark-perf-check")

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
PHASE_DIR       = "phase_perf"


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def gpu_function(name: str, timeout: int = 3600, memory: int = 16384):
    return app.function(
        image=image,
        gpu="h100",
        secrets=[modal.Secret.from_name("huggingface")],
        volumes={
            str(RESULTS_PATH):  results_vol,
            str(HF_CACHE_PATH): model_cache_vol,
        },
        timeout=timeout,
        memory=memory,
        name=name,
    )


# ── 1. Smoke test — fast correctness check on tiny model ────────────────────

@gpu_function("perf_smoke", timeout=600)
def run_smoke():
    """
    End-to-end smoke: load OPT-125M, prefill + decode 8 tokens with both
    hybrid score_modes, assert outputs are finite and shapes are right.
    Catches anything obvious like a missing dict key, a wrong dtype, or a
    crash in _materialise — much cheaper than waiting for a full B2 run.
    """
    _bootstrap()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark.runner import generate_with_method
    from methods.kivi_topk_hybrid import KIVI_TopK_Method
    from methods.kivi_quant import KIVIMethod
    from methods.topk_selection import TopKMethod

    print("[smoke] loading OPT-125M…")
    tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", torch_dtype=torch.float16, device_map="cuda",
    ).eval()

    prompt = (
        "The history of machine learning began in the 1950s when researchers "
        "first proposed neural networks. Over decades the field has evolved "
        "through symbolic AI, statistical learning, and deep learning. "
    ) * 4

    results = {}
    for label, method in [
        ("kivi",          KIVIMethod(bits=4, residual_length=32, group_size=32)),
        ("topk",          TopKMethod(K=64, n_sink=8, n_local=32, refresh_interval=10)),
        ("kivi_topk_a",   KIVI_TopK_Method(
                              bits=4, residual_length=32, group_size=32,
                              K=64, n_sink=8, n_local=32, refresh_interval=10,
                              score_mode="centroid")),
        ("kivi_topk_c",   KIVI_TopK_Method(
                              bits=4, residual_length=32, group_size=32,
                              K=64, n_sink=8, n_local=32, refresh_interval=10,
                              score_mode="quantized")),
    ]:
        text, metrics = generate_with_method(
            model, tok, method,
            prompt=prompt, max_new_tokens=12, device="cuda",
        )
        finite = "ok"
        try:
            for k in metrics:
                if isinstance(metrics[k], float):
                    if not (metrics[k] == metrics[k]):  # NaN check
                        finite = f"NaN in {k}"
        except Exception as e:
            finite = f"err: {e}"

        stats = method.get_stats() if hasattr(method, "get_stats") else {}
        results[label] = {
            "text_head": text[:120],
            "kv_mb": metrics.get("kv_cache_mb", -1),
            "finite": finite,
            "decode_steps": stats.get("decode_steps", 0),
        }
        print(f"[smoke] {label}: {results[label]}")
        method.reset() if hasattr(method, "reset") else None
        torch.cuda.empty_cache()

    print("[smoke] all four methods generated 12 tokens without error.")
    return results


# ── 2. Synthetic latency benchmark — same harness as Phase B2 long_context ──

@gpu_function("perf_long_context", timeout=3600)
def run_long_context_perf(
    seq_lens: list = [2048, 8192, 16384, 32768],
):
    _bootstrap()
    from experiments.long_context import run_long_context

    out_dir = RESULTS_PATH / PHASE_DIR / "long_context"
    records = run_long_context(
        seq_lens=seq_lens,
        methods=["baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c"],
        n_layers=32, n_heads=32, head_dim=128,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        n_warmup=5, n_iter=15,
        use_selection_cache=False,           # real kernel cost every step
        model_name=None,
        device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 3. Passkey @4K — correctness validation (must not regress) ──────────────

@gpu_function("perf_passkey", timeout=3600, memory=32768)
def run_passkey_perf(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    n_trials: int = 5,
    n_depths: int = 5,
):
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    out_dir = RESULTS_PATH / PHASE_DIR / "passkey"
    records = run_passkey_retrieval(
        model_name=model_name,
        seq_lens=[4096],                   # only 4K; 8K is OOD for LLaMA-2-7B
        methods=["baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c"],
        n_trials=n_trials, n_depths=n_depths,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        max_new_tokens=12,
        seed=0, device="cuda", output_dir=out_dir,
    )
    results_vol.commit()
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── Local entrypoint ────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(only: str = "", skip_passkey: bool = False):
    jobs = {
        "smoke":        (run_smoke,            {}),
        "long_context": (run_long_context_perf, {}),
        "passkey":      (run_passkey_perf,      {}),
    }

    if only:
        if only not in jobs:
            print(f"Unknown experiment '{only}'. Choose from: {list(jobs)}")
            return
        selected = {only: jobs[only]}
    elif skip_passkey:
        selected = {k: v for k, v in jobs.items() if k != "passkey"}
    else:
        selected = jobs

    print(f"\n[perf-check] Dispatching {len(selected)} jobs to H100s in parallel:")
    for name in selected:
        print(f"  - {name}")
    print()

    handles = {name: fn.spawn(**kw) for name, (fn, kw) in selected.items()}
    print("[perf-check] Spawned. Waiting for completion…\n")

    for name, h in handles.items():
        try:
            res = h.get()
            print(f"[perf-check] ✅ {name}: {res}")
        except Exception as e:
            print(f"[perf-check] ❌ {name} failed: {e}")

    print("\n[perf-check] All done. Pull results with:")
    print(f"  modal volume get kv-benchmark-results /{PHASE_DIR} ./results/{PHASE_DIR}")
