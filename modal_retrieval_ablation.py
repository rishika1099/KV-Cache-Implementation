"""
Passkey-retrieval ablation runner on Modal H100.

Tests four orthogonal selection knobs on the KIVI+TopK hybrid against
the pre-fix baseline, plus a K-sweep on the unmodified hybrid.

Modes:
    smoke    1 method × 1 depth × 1 trial
    full     7 methods × 5 depths × 5 trials
    ksweep   kivi_topk at K ∈ {512, 1024, 2048, 3072}
    all      full + ksweep in parallel

Pull results:
    modal volume get kv-benchmark-results /retrieval_ablation \\
        ./results/retrieval_ablation
"""

import sys
from pathlib import Path

import modal


app = modal.App("kv-retrieval-ablation")

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

RESULTS_PATH    = Path("/results")
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
RESULTS_SUBDIR  = "retrieval_ablation"

WANDB_PROJECT = "kv-cache-benchmark"
WANDB_GROUP   = "retrieval_ablation"


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def gpu_function(name: str, timeout: int = 3600, memory: int = 32768):
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


def _log_to_wandb(run_name: str, records, *, tags: list,
                  config: dict | None = None):
    """One W&B run per Modal job. Logs each record as a step + a Table."""
    import os
    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY not set. Run: "
            "modal secret create wandb WANDB_API_KEY=<your-key>"
        )

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
        name=run_name,
        group=WANDB_GROUP,
        tags=list(tags),
        config=config or {},
        reinit=True,
        job_type="passkey_retrieval",
    )

    columns = ["method", "seq_len", "depth", "K", "n_sink", "n_local",
               "n_trials", "accuracy", "n_correct", "throughput_tps"]
    table = wandb.Table(columns=columns)

    accs: list[float] = []
    per_method: dict[str, list[float]] = {}

    for step, rec in enumerate(records):
        cfg, m = rec.config or {}, rec.metrics or {}
        acc  = float(m.get("accuracy", 0.0))
        tput = float(m.get("throughput_tps", 0.0))

        wandb.log({
            "method":         rec.method,
            "seq_len":        cfg.get("seq_len"),
            "depth":          cfg.get("depth"),
            "K":              cfg.get("K"),
            "accuracy":       acc,
            "n_correct":      m.get("n_correct"),
            "n_trials":       m.get("n_trials"),
            "throughput_tps": tput,
        }, step=step)

        table.add_data(
            rec.method, cfg.get("seq_len"), cfg.get("depth"),
            cfg.get("K"), cfg.get("n_sink"), cfg.get("n_local"),
            m.get("n_trials"), acc, m.get("n_correct"), tput,
        )
        accs.append(acc)
        per_method.setdefault(rec.method, []).append(acc)

    wandb.log({"passkey_table": table})

    if accs:
        run.summary["mean_accuracy"] = sum(accs) / len(accs)
        run.summary["n_records"]     = len(accs)
    for method, vals in per_method.items():
        run.summary[f"acc/{method}/mean"] = sum(vals) / len(vals)

    run.finish()


@gpu_function("retrieval_smoke", timeout=900)
def run_smoke():
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    out_dir = RESULTS_PATH / RESULTS_SUBDIR / "smoke"

    records = run_passkey_retrieval(
        model_name="meta-llama/Llama-2-7b-hf",
        seq_lens=[4096],
        methods=["kivi_topk_all"],
        n_trials=1, n_depths=1,
        K=1024, n_sink=128, n_local=512,
        bits=4, group_size=32, residual_length=128,
        max_new_tokens=12,
        seed=0, device="cuda", output_dir=out_dir,
    )
    results_vol.commit()

    _log_to_wandb(
        run_name="smoke",
        records=records,
        tags=["smoke", "kivi_topk_all"],
        config={"mode": "smoke", "model": "Llama-2-7b-hf",
                "seq_len": 4096, "K": 1024,
                "n_sink": 128, "n_local": 512,
                "n_trials": 1, "n_depths": 1},
    )
    return {"n_rows": len(records), "out_dir": str(out_dir)}


@gpu_function("retrieval_full", timeout=7200, memory=32768)
def run_full():
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    out_dir = RESULTS_PATH / RESULTS_SUBDIR / "ablation"
    methods = [
        "kivi",
        "kivi_topk",
        "kivi_topk_maxpool",
        "kivi_topk_twopass",
        "kivi_topk_multiq",
        "kivi_topk_dynsink",
        "kivi_topk_all",
    ]

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
        run_name="ablation",
        records=records,
        tags=["ablation"] + methods,
        config={"mode": "full", "model": "Llama-2-7b-hf",
                "seq_len": 4096, "K": 1024,
                "n_sink": 128, "n_local": 512,
                "n_trials": 5, "n_depths": 5,
                "methods": methods},
    )
    return {"n_rows": len(records), "out_dir": str(out_dir)}


@gpu_function("retrieval_ksweep", timeout=3600, memory=32768)
def run_ksweep():
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    all_records = []
    for K in [512, 1024, 2048, 3072]:
        out_dir = RESULTS_PATH / RESULTS_SUBDIR / f"ksweep_K{K}"
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
        run_name="ksweep_kivi_topk",
        records=all_records,
        tags=["ksweep", "kivi_topk"],
        config={"mode": "ksweep", "model": "Llama-2-7b-hf",
                "seq_len": 4096,
                "K_values": [512, 1024, 2048, 3072],
                "n_sink": 128, "n_local": 512,
                "n_trials": 5, "n_depths": 5},
    )
    return {"n_rows": len(all_records),
            "out_dir": str(RESULTS_PATH / RESULTS_SUBDIR)}


@gpu_function("retrieval_ctx_ksweep", timeout=18000, memory=65536)
def run_ctx_ksweep(seq_len: int, K_values: list):
    """K-sweep on `kivi_topk` at a single seq_len. One model load, N K-values."""
    _bootstrap()
    from experiments.passkey_retrieval import run_passkey_retrieval

    all_records = []
    for K in K_values:
        out_dir = (RESULTS_PATH / RESULTS_SUBDIR
                   / f"ctx_ksweep_S{seq_len}_K{K}")
        records = run_passkey_retrieval(
            model_name="meta-llama/Llama-2-7b-hf",
            seq_lens=[seq_len],
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
        run_name=f"ctx_ksweep_S{seq_len}",
        records=all_records,
        tags=["ctx_ksweep", f"seq_len={seq_len}"],
        config={"mode": "ctx_ksweep", "model": "Llama-2-7b-hf",
                "seq_len": seq_len,
                "K_values": K_values,
                "n_sink": 128, "n_local": 512,
                "n_trials": 5, "n_depths": 5},
    )
    return {"n_rows": len(all_records), "seq_len": seq_len,
            "out_dir": str(RESULTS_PATH / RESULTS_SUBDIR)}


# Per-context K grids. Skip K values that are >= seq_len (would just be
# full-attention selection over the whole prompt).
_CTX_GRID = {
     8192: [1024, 2048, 3072, 4096, 6144],
    16384: [1024, 2048, 4096, 6144, 8192, 12288],
    32768: [1024, 2048, 4096, 8192, 12288, 16384, 24576],
}


@app.local_entrypoint()
def main(mode: str = "smoke"):
    if mode == "smoke":
        res = run_smoke.remote()
        print(f"smoke: {res}")
    elif mode == "full":
        res = run_full.remote()
        print(f"full: {res}")
    elif mode == "ksweep":
        res = run_ksweep.remote()
        print(f"ksweep: {res}")
    elif mode == "ctx_ksweep":
        handles = [
            (s, run_ctx_ksweep.spawn(s, ks))
            for s, ks in _CTX_GRID.items()
        ]
        for s, h in handles:
            try:
                print(f"S={s}: {h.get()}")
            except Exception as e:
                print(f"S={s} failed: {e}")
    elif mode == "all":
        h_full = run_full.spawn()
        h_sweep = run_ksweep.spawn()
        for name, h in [("full", h_full), ("ksweep", h_sweep)]:
            try:
                print(f"{name}: {h.get()}")
            except Exception as e:
                print(f"{name} failed: {e}")
    else:
        print(f"Unknown mode '{mode}'. "
              f"Choose: smoke | full | ksweep | ctx_ksweep | all")
        return

    print(f"\nPull results:")
    print(f"  modal volume get kv-benchmark-results "
          f"/{RESULTS_SUBDIR} ./results/{RESULTS_SUBDIR}")
