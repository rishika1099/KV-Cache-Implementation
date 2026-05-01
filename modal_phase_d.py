"""
Phase D — retrieval-fix ablation runner.

Targets the depth-0.3/0.5/0.7 zero band on the KIVI+TopK hybrid passkey
result. Tests four orthogonal knobs landed on `research/retrieval-fixes`:

  1. score_mode='maxpool'       per-channel max block summary
  2. two_pass_factor=2          centroid pre-rank → exact quant_score rerank
  3. proxy_history=8 / max      multi-step query pooling
  4. dynamic_sinks=True         importance-picked sinks (prefill K-norm)

Plus the kitchen-sink combination and a baseline K-sweep on the
pre-fix hybrid (to sanity-check that the issue isn't simply budget).

Two modes:
  --mode smoke   single (method, depth, trial), ~3 min, ~$0.20  — derisk first
  --mode full    full ablation grid, ~45-90 min, ~$3-6           — actual run

Pull results with:
  modal volume get kv-benchmark-results /phase_d ./results/phase_d
"""

import sys
from pathlib import Path

import modal


app = modal.App("kv-benchmark-phase-d")

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
        "wandb>=0.17.0",
    )
    .add_local_dir(str(_base / "methods"),     "/app/methods")
    .add_local_dir(str(_base / "benchmark"),   "/app/benchmark")
    .add_local_dir(str(_base / "experiments"), "/app/experiments")
)

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)

RESULTS_PATH  = Path("/results")
HF_CACHE_PATH = Path("/root/.cache/huggingface")
PHASE_DIR     = "phase_d"

# W&B project for these runs. Override per-call with WANDB_PROJECT env var.
WANDB_PROJECT = "kv-cache-benchmark"


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def gpu_function(name: str, timeout: int = 3600, memory: int = 32768):
    """
    Decorator factory for Phase-D GPU jobs. Each job mounts:
      - the HF secret (gated Llama-2 access)
      - the W&B secret (must export WANDB_API_KEY)
    Create the W&B secret once with:
      modal secret create wandb WANDB_API_KEY=<key>
    """
    return app.function(
        image=image,
        gpu="h100",
        secrets=[
            modal.Secret.from_name("huggingface"),
            modal.Secret.from_name("wandb"),
        ],
        volumes={
            str(RESULTS_PATH):  results_vol,
            str(HF_CACHE_PATH): model_cache_vol,
        },
        timeout=timeout,
        memory=memory,
        name=name,
    )


# ── W&B logging helper ──────────────────────────────────────────────────────

def _log_to_wandb(run_name: str, records, *, group: str, tags: list,
                  config: dict | None = None):
    """
    Push a list of ExperimentRecord objects to W&B as one run:
      - per-record points logged sequentially (so the metrics show on a step axis),
      - a `passkey_table` containing every record for grouped views,
      - per-method × per-depth accuracy summaries,
      - overall mean accuracy as run.summary['mean_accuracy'].

    Failure to import wandb or missing API key is fatal — silent fallback
    would defeat the point of "push runs to wandb".
    """
    import os
    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY not set in container env. "
            "Run: modal secret create wandb WANDB_API_KEY=<your-key>"
        )

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
        name=run_name,
        group=group,
        tags=list(tags),
        config=config or {},
        reinit=True,
        job_type="passkey_retrieval",
    )

    columns = ["method", "seq_len", "depth", "K", "n_sink", "n_local",
               "n_trials", "accuracy", "n_correct", "throughput_tps"]
    table = wandb.Table(columns=columns)

    accs: list[float] = []
    per_method_accs: dict[str, list[float]] = {}

    for step, rec in enumerate(records):
        # ExperimentRecord exposes .method, .config, .metrics
        method = rec.method
        cfg    = rec.config or {}
        m      = rec.metrics or {}
        acc    = float(m.get("accuracy", 0.0))
        tput   = float(m.get("throughput_tps", 0.0))

        wandb.log({
            "method":          method,
            "seq_len":         cfg.get("seq_len"),
            "depth":           cfg.get("depth"),
            "K":               cfg.get("K"),
            "accuracy":        acc,
            "n_correct":       m.get("n_correct"),
            "n_trials":        m.get("n_trials"),
            "throughput_tps":  tput,
        }, step=step)

        table.add_data(
            method, cfg.get("seq_len"), cfg.get("depth"),
            cfg.get("K"), cfg.get("n_sink"), cfg.get("n_local"),
            m.get("n_trials"), acc, m.get("n_correct"), tput,
        )
        accs.append(acc)
        per_method_accs.setdefault(method, []).append(acc)

    wandb.log({"passkey_table": table})

    if accs:
        run.summary["mean_accuracy"] = sum(accs) / len(accs)
        run.summary["n_records"]     = len(accs)
    for method, vals in per_method_accs.items():
        run.summary[f"acc/{method}/mean"] = sum(vals) / len(vals)

    run.finish()
    print(f"[wandb] logged {len(records)} records as run '{run_name}' "
          f"under group='{group}'")


# ── 1. Smoke — derisk runtime bugs cheaply ──────────────────────────────────

@gpu_function("phase_d_smoke", timeout=900)
def run_smoke():
    """
    Single-trial passkey on Llama-2-7B with `kivi_topk_all` (kitchen sink).
    If this passes, every individual fix should also work — they're orthogonal.
    Costs ~$0.20 of H100 time vs ~$5 for the full run.
    """
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    out_dir = RESULTS_PATH / PHASE_DIR / "smoke"
    print("[phase-d-smoke] running 1 method × 1 depth × 1 trial …")

    records = run_passkey_retrieval(
        model_name="meta-llama/Llama-2-7b-hf",
        seq_lens=[4096],
        methods=["kivi_topk_all"],          # the everything-on variant
        n_trials=1, n_depths=1,             # minimum
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        max_new_tokens=12,
        seed=0, device="cuda", output_dir=out_dir,
    )
    results_vol.commit()

    _log_to_wandb(
        run_name="phase_d_smoke",
        records=records,
        group="phase_d",
        tags=["phase_d", "smoke", "kivi_topk_all"],
        config={"mode": "smoke", "model": "Llama-2-7b-hf",
                "seq_len": 4096, "K": 1024,
                "n_sink": 128, "n_local": 512,
                "n_trials": 1, "n_depths": 1},
    )
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 2. Full ablation — passkey across all retrieval-fix variants ───────────

@gpu_function("phase_d_full", timeout=7200, memory=32768)
def run_full():
    """
    Full passkey ablation grid: 7 methods × 5 depths × 5 trials = 175 gens.
    Each generation ~10s on Llama-2-7B at 4K, total ~25 min compute plus
    model loads + warmup. Accepts ~45-90 min wall clock.
    """
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    out_dir = RESULTS_PATH / PHASE_DIR / "ablation"
    methods = [
        "kivi",                  # storage-only reference (100% across depths)
        "kivi_topk",             # pre-fix hybrid (depth 0.3/0.5/0.7 = 0%)
        "kivi_topk_maxpool",     # fix #1
        "kivi_topk_twopass",     # fix #2
        "kivi_topk_multiq",      # fix #3
        "kivi_topk_dynsink",     # fix #4
        "kivi_topk_all",         # all four stacked
    ]
    print(f"[phase-d-full] methods = {methods}")

    records = run_passkey_retrieval(
        model_name="meta-llama/Llama-2-7b-hf",
        seq_lens=[4096],
        methods=methods,
        n_trials=5, n_depths=5,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        max_new_tokens=12,
        seed=0, device="cuda", output_dir=out_dir,
    )
    results_vol.commit()

    _log_to_wandb(
        run_name="phase_d_full_ablation",
        records=records,
        group="phase_d",
        tags=["phase_d", "ablation"] + methods,
        config={"mode": "full", "model": "Llama-2-7b-hf",
                "seq_len": 4096, "K": 1024,
                "n_sink": 128, "n_local": 512,
                "n_trials": 5, "n_depths": 5,
                "methods": methods},
    )
    return {"n_rows": len(records), "out_dir": str(out_dir)}


# ── 3. K-sweep — does the pre-fix hybrid recover with a bigger budget? ─────

@gpu_function("phase_d_ksweep", timeout=3600, memory=32768)
def run_ksweep():
    """
    Run the pre-fix `kivi_topk` (centroid scoring, no other knobs) at
    K ∈ {512, 1024, 2048, 3072}. Tells us how much of the failure is
    'budget too small' vs 'scoring fundamentally broken at any budget'.
    Cheap experiment, big diagnostic value.
    """
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    all_records = []
    for K in [512, 1024, 2048, 3072]:
        out_dir = RESULTS_PATH / PHASE_DIR / f"ksweep_K{K}"
        print(f"\n[phase-d-ksweep] K = {K}")
        records = run_passkey_retrieval(
            model_name="meta-llama/Llama-2-7b-hf",
            seq_lens=[4096],
            methods=["kivi_topk"],
            n_trials=5, n_depths=5,
            K=K, n_sink=128, n_local=512,
            bits=4, group_size=32, residual_length=128,
            max_new_tokens=12,
            seed=0, device="cuda", output_dir=out_dir,
        )
        all_records.extend(records)
    results_vol.commit()

    _log_to_wandb(
        run_name="phase_d_ksweep_kivi_topk",
        records=all_records,
        group="phase_d",
        tags=["phase_d", "ksweep", "kivi_topk"],
        config={"mode": "ksweep", "model": "Llama-2-7b-hf",
                "seq_len": 4096,
                "K_values": [512, 1024, 2048, 3072],
                "n_sink": 128, "n_local": 512,
                "n_trials": 5, "n_depths": 5},
    )
    return {"n_rows": len(all_records), "out_dir": str(RESULTS_PATH / PHASE_DIR)}


# ── Local entrypoint ────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(mode: str = "smoke"):
    """
    Modes:
        smoke   — 1 method × 1 depth × 1 trial (~3 min, ~$0.20)
        full    — full ablation 6 methods × 5 × 5 (~45-90 min, ~$3-6)
        ksweep  — K ∈ {512, 1024, 2048, 3072} on pre-fix hybrid (~30 min)
        all     — full + ksweep (parallel)
    """
    if mode == "smoke":
        print("[phase-d] dispatching smoke …")
        h = run_smoke.spawn()
        try:
            res = h.get()
            print(f"[phase-d] ✅ smoke: {res}")
            print("\n[phase-d] smoke OK. Now run with --mode full")
        except Exception as e:
            print(f"[phase-d] ❌ smoke failed: {e}")

    elif mode == "full":
        print("[phase-d] dispatching full ablation …")
        h = run_full.spawn()
        try:
            res = h.get()
            print(f"[phase-d] ✅ full: {res}")
        except Exception as e:
            print(f"[phase-d] ❌ full failed: {e}")

    elif mode == "ksweep":
        print("[phase-d] dispatching K-sweep …")
        h = run_ksweep.spawn()
        try:
            res = h.get()
            print(f"[phase-d] ✅ ksweep: {res}")
        except Exception as e:
            print(f"[phase-d] ❌ ksweep failed: {e}")

    elif mode == "all":
        print("[phase-d] dispatching full + ksweep in parallel …")
        h_full = run_full.spawn()
        h_sweep = run_ksweep.spawn()
        for name, h in [("full", h_full), ("ksweep", h_sweep)]:
            try:
                res = h.get()
                print(f"[phase-d] ✅ {name}: {res}")
            except Exception as e:
                print(f"[phase-d] ❌ {name} failed: {e}")

    else:
        print(f"Unknown mode '{mode}'. Choose: smoke | full | ksweep | all")
        return

    print(f"\n[phase-d] Pull results with:")
    print(f"  modal volume get kv-benchmark-results /{PHASE_DIR} ./results/{PHASE_DIR}")
