#!/usr/bin/env python3
"""
Merge volume backup results + PPL results into a clean results.jsonl,
then run plots and upload to wandb.

Usage: python merge_and_upload.py
"""
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"
BACKUP_DIR = RESULTS_DIR / "volume_backup"
PPL_FILE = RESULTS_DIR / "ppl_results.json"
OUTPUT_FILE = RESULTS_DIR / "results.jsonl"

# Current-run files only (with group_size for KIVI, n_sink for TopK)
CURRENT_RUN_FILES = {
    "baseline": "baseline.jsonl",
    "kivi_4bit": "kivi_3a2c6a6a.jsonl",
    "kivi_2bit": "kivi_db874bcf.jsonl",
    "snapkv_02": "snapkv_08122fe4.jsonl",
    "snapkv_04": "snapkv_17dda4cb.jsonl",
    "snapkv_06": "snapkv_8008430f.jsonl",
    "topk_512": "topk_45477ba1.jsonl",
    "topk_1024": "topk_510b5c5d.jsonl",
    "topk_2048": "topk_c078153a.jsonl",
}


def deduplicate_baseline(records):
    """
    Baseline file has results from multiple runs stacked.
    Keep only the latest run's data: 4 per seq_len for synthetic,
    20 per task for longbench.
    """
    synthetic = [r for r in records if r["prompt_type"] == "synthetic"]
    longbench = [r for r in records if r["prompt_type"] == "longbench"]
    other = [r for r in records if r["prompt_type"] not in ("synthetic", "longbench")]

    # For synthetic: group by seq_len, take last 4
    by_seq = defaultdict(list)
    for r in synthetic:
        by_seq[r["seq_len"]].append(r)
    deduped_syn = []
    for seq_len in sorted(by_seq.keys()):
        deduped_syn.extend(by_seq[seq_len][-4:])

    # For longbench: group by task, take last 20
    by_task = defaultdict(list)
    for r in longbench:
        by_task[r["task"]].append(r)
    deduped_lb = []
    for task in sorted(by_task.keys()):
        deduped_lb.extend(by_task[task][-20:])

    return deduped_syn + deduped_lb + other


def deduplicate_method(records, n_synthetic=12, n_longbench_per_task=20):
    """
    Method files may have stacked results from old+new runs.
    Keep the latest n results per category.
    """
    synthetic = [r for r in records if r["prompt_type"] == "synthetic"]
    longbench = [r for r in records if r["prompt_type"] == "longbench"]
    other = [r for r in records if r["prompt_type"] not in ("synthetic", "longbench")]

    # Synthetic: group by seq_len, take last 4
    by_seq = defaultdict(list)
    for r in synthetic:
        by_seq[r["seq_len"]].append(r)
    deduped_syn = []
    for seq_len in sorted(by_seq.keys()):
        deduped_syn.extend(by_seq[seq_len][-4:])

    # Longbench: group by task, take last 20
    by_task = defaultdict(list)
    for r in longbench:
        by_task[r["task"]].append(r)
    deduped_lb = []
    for task in sorted(by_task.keys()):
        deduped_lb.extend(by_task[task][-20:])

    return deduped_syn + deduped_lb + other


def main():
    all_records = []

    # 1. Load and deduplicate each method's results
    for label, filename in CURRENT_RUN_FILES.items():
        filepath = BACKUP_DIR / filename
        if not filepath.exists():
            print(f"  SKIP {label}: {filepath} not found")
            continue

        with open(filepath) as f:
            records = [json.loads(line) for line in f if line.strip()]

        if label == "baseline":
            records = deduplicate_baseline(records)
        else:
            records = deduplicate_method(records)

        print(f"  {label}: {len(records)} records")
        all_records.extend(records)

    # 2. Add PPL results
    if PPL_FILE.exists():
        with open(PPL_FILE) as f:
            ppl_results = json.load(f)

        for r in ppl_results:
            all_records.append({
                "method": r["method"],
                "config": r["config"],
                "prompt_type": "wikitext",
                "task": "n/a",
                "seq_len": 0,
                "peak_memory_gb": 0,
                "kv_cache_mb": 0,
                "compression_ratio": 1.0,
                "ttft_ms": 0,
                "throughput_tps": 0,
                "per_token_latency_ms": 0,
                "perplexity": round(r["perplexity"], 4),
                "task_score": None,
                "longbench_score": None,
                "tokens_generated": 0,
                "timestamp": "",
            })
        print(f"  PPL: {len(ppl_results)} records added")
    else:
        print(f"  WARNING: {PPL_FILE} not found — no PPL data")

    # 3. Write merged results
    with open(OUTPUT_FILE, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{len(all_records)} total records written to {OUTPUT_FILE}")

    # 4. Summary
    by_method = defaultdict(list)
    for r in all_records:
        by_method[r["method"]].append(r)

    print("\n--- Summary ---")
    for method in sorted(by_method.keys()):
        recs = by_method[method]
        syn = len([r for r in recs if r["prompt_type"] == "synthetic"])
        lb = len([r for r in recs if r["prompt_type"] == "longbench"])
        wiki = len([r for r in recs if r["prompt_type"] == "wikitext"])
        scored = len([r for r in recs if r.get("task_score") is not None])
        ppl_vals = [r["perplexity"] for r in recs if r.get("perplexity")]
        ppl_str = f"PPL={min(ppl_vals):.2f}" if ppl_vals else "no PPL"
        print(f"  {method:10s}: {syn} syn, {lb} lb ({scored} scored), {wiki} wiki — {ppl_str}")


if __name__ == "__main__":
    main()
