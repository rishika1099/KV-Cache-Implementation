#!/usr/bin/env python3
"""
Upload the experiment runs that exist in results/ but are not yet on
the kv-cache-hpml W&B project.

Run locally (one-shot):

    cd KV-Cache-Implementation
    pip install wandb
    wandb login                # paste your W&B API key
    python scripts/wandb_upload_missing.py

This intentionally creates new runs rather than editing existing ones,
so the project history stays append-only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import wandb

PROJECT = "kv-cache-hpml"
ENTITY  = "rm4318-columbia-university"
RESULTS = Path(__file__).resolve().parents[1] / "results"


def _bench_summary_from_records(records: list[dict]) -> dict:
    """Discard the warmup batch and average the rest."""
    stable = records[1:] if len(records) > 1 else records
    if not stable:
        return {}
    keys = stable[0].keys()
    return {f"mean_{k}": sum(r[k] for r in stable) / len(stable) for k in keys}


def _longbench_per_task(records: list[dict]) -> dict:
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r["score"])
    return {t: sum(s) / len(s) for t, s in by_task.items()}


def _upload_throughput(run_name: str, tags: list[str], path: Path,
                       config_extra: dict | None = None) -> None:
    with open(path) as f:
        d = json.load(f)

    cfg = {k: v for k, v in d.items() if k not in {"summary", "raw_batches", "records"}}
    if config_extra:
        cfg.update(config_extra)

    summary = d.get("summary")
    if summary is None and "records" in d:
        # KIVI×TopK style: single record under "records[prefill]"
        first = next(iter(d["records"].values()))
        summary = first
        cfg["prefill_len"] = int(next(iter(d["records"].keys())))

    run = wandb.init(project=PROJECT, entity=ENTITY, name=run_name,
                     tags=tags, config=cfg, reinit=True)

    if d.get("raw_batches"):
        table = wandb.Table(columns=list(d["raw_batches"][0].keys()),
                            data=[list(r.values()) for r in d["raw_batches"]])
        wandb.log({"raw_batches": table})

    if summary:
        for k, v in summary.items():
            wandb.summary[k] = v

    print(f"  ✓ uploaded {run_name}")
    run.finish()


def _upload_longbench(run_name: str, tags: list[str], path: Path,
                      config_extra: dict | None = None) -> None:
    with open(path) as f:
        data = json.load(f)
    # Some MLA longbench files wrap records under a top-level key
    if isinstance(data, dict) and len(data) == 1:
        data = next(iter(data.values()))

    by_task = defaultdict(list)
    for r in data:
        by_task[r["task"]].append(r["score"])
    task_scores = {t: sum(s) / len(s) for t, s in by_task.items()}
    overall = sum(task_scores.values()) / len(task_scores)

    cfg = dict(data[0].get("config") or {})
    cfg["method"] = data[0].get("method", "unknown")
    cfg["n_examples_per_task"] = len(next(iter(by_task.values())))
    if config_extra:
        cfg.update(config_extra)

    run = wandb.init(project=PROJECT, entity=ENTITY, name=run_name,
                     tags=tags, config=cfg, reinit=True)

    pt_table = wandb.Table(columns=["task", "n", "score"],
                           data=[[t, len(by_task[t]), s] for t, s in task_scores.items()])
    wandb.log({"per_task_scores": pt_table, "overall_score": overall})

    for t, s in task_scores.items():
        wandb.summary[f"task/{t}"] = s
    wandb.summary["overall_score"] = overall

    print(f"  ✓ uploaded {run_name}  (overall={overall:.3f})")
    run.finish()


# ── List of (run_name, tags, file, type) ───────────────────────────────────
UPLOADS: list[tuple[str, list[str], str, str]] = [
    # ── LongBench: SnapKV ─────────────────────────────────────────────────
    ("longbench_snapkv_0.4",
     ["longbench", "snapkv", "budget_ratio=0.4"],
     'longbench_snapkv_{"budget_ratio": 0.4, "kernel_size": 7, "observation_window": 32, "sink_size": 0}.json',
     "longbench"),
    ("longbench_snapkv_baseline",
     ["longbench", "snapkv", "baseline"],
     "longbench_snapkv_baseline.json",
     "longbench"),

    # ── LongBench: KIVI×TopK hybrid ────────────────────────────────────────
    ("longbench_kivi_topk_4bit_K1024",
     ["longbench", "kivi_topk", "bits=4", "K=1024"],
     "longbench_kivi_topk_4bit_K1024_sink128_local512.json",
     "longbench"),
    ("longbench_kivi_topk_4bit_K4096",
     ["longbench", "kivi_topk", "bits=4", "K=4096"],
     "longbench_kivi_topk_4bit_K4096_sink128_local4096.json",
     "longbench"),

    # ── Throughput: long context (32K) ────────────────────────────────────
    ("topk_throughput_topk_p32768_g512",
     ["throughput", "topk", "prefill=32768"],
     "topk_throughput_topk_p32768_g512.json",
     "throughput"),
    ("topk_throughput_topk_flash_p65536_g512",
     ["throughput", "topk_flash", "prefill=65536", "long_context"],
     "topk_throughput_topk_flash_p65536_g512.json",
     "throughput"),

    # ── KIVI×TopK hybrid throughput (32K) ─────────────────────────────────
    ("throughput_kivi_topk_baseline_p32768_g512",
     ["throughput", "kivi_topk", "baseline", "prefill=32768"],
     "throughput_kivi_topk_baseline_p32768_g512.json",
     "throughput"),
    ("throughput_kivi_topk_kivi_topk_p32768_g512",
     ["throughput", "kivi_topk", "bits=4", "prefill=32768"],
     "throughput_kivi_topk_kivi_topk_p32768_g512.json",
     "throughput"),

    # ── MLA throughput at gen=256 (cited in report) ───────────────────────
    ("mla_throughput_baseline_g256",
     ["throughput", "mla", "baseline", "gen=256"],
     "mla_throughput_baseline_g256.json",
     "throughput"),
    ("mla_throughput_mla_g256",
     ["throughput", "mla", "gen=256"],
     "mla_throughput_mla_g256.json",
     "throughput"),

    # ── SnapKV throughput sweep ───────────────────────────────────────────
    ("throughput_snapkv_r04_p4096_g512",
     ["throughput", "snapkv", "budget_ratio=0.4", "prefill=4096"],
     "throughput_snapkv_r04_p4096_g512.json",
     "throughput"),
    ("throughput_snapkv_p4096_g512",
     ["throughput", "snapkv", "prefill=4096"],
     "throughput_snapkv_p4096_g512.json",
     "throughput"),
    ("throughput_snapkv_p1024_g512",
     ["throughput", "snapkv", "prefill=1024"],
     "throughput_snapkv_p1024_g512.json",
     "throughput"),
    ("throughput_snapkv_batch_baseline_p4096_g512",
     ["throughput", "snapkv", "batch", "baseline"],
     "throughput_snapkv_batch_baseline_p4096_g512.json",
     "throughput"),
    ("throughput_snapkv_batch_snapkv_p4096_g512",
     ["throughput", "snapkv", "batch", "snapkv"],
     "throughput_snapkv_batch_snapkv_p4096_g512.json",
     "throughput"),
]


def main() -> None:
    print(f"Uploading {len(UPLOADS)} missing runs to {ENTITY}/{PROJECT}\n")
    for run_name, tags, fname, kind in UPLOADS:
        path = RESULTS / fname
        if not path.exists():
            print(f"  ⚠  skipping (missing JSON): {fname}")
            continue
        try:
            if kind == "longbench":
                _upload_longbench(run_name, tags, path)
            else:
                _upload_throughput(run_name, tags, path)
        except Exception as e:
            print(f"  ✗ {run_name}: {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
