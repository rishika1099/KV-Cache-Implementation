#!/usr/bin/env python3
"""Upload KV-cache benchmark results to Weights & Biases."""

import json
import os
from collections import defaultdict
from pathlib import Path

import wandb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_NAME = "kv-cache-benchmark"
RESULTS_FILE = Path(__file__).parent / "results" / "results.jsonl"
PLOTS_DIR = Path(__file__).parent / "results" / "plots"

METRIC_KEYS = [
    "peak_memory_gb",
    "kv_cache_mb",
    "compression_ratio",
    "ttft_ms",
    "throughput_tps",
    "per_token_latency_ms",
    "perplexity",
    "seq_len",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_run_name(method: str, config: dict) -> str:
    """Create a human-readable run name from method + config."""
    if not config:
        return method
    parts = "_".join(f"{k}={v}" for k, v in sorted(config.items()))
    return f"{method}_{parts}"


def make_tags(method: str, config: dict) -> list[str]:
    """Return a list of tags for this run."""
    tags = [method]
    for k, v in sorted(config.items()):
        tags.append(f"{k}={v}")
    return tags


def config_key(config: dict) -> str:
    """Deterministic string key for a config dict."""
    return json.dumps(config, sort_keys=True)


def load_results(path: Path) -> list[dict]:
    """Read all JSONL rows from *path*."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_results(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group rows by (method, config_key)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["method"], config_key(row["config"]))
        groups[key].append(row)
    return groups


def avg(values: list[float | None]) -> float | None:
    """Return the mean of non-None values, or None if empty."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 0. Detect entity from wandb API (requires prior `wandb login`)
    # ------------------------------------------------------------------
    api = wandb.Api()
    entity = api.default_entity
    print(f"Detected wandb entity: {entity}")
    print(f"Project: {PROJECT_NAME}")

    # ------------------------------------------------------------------
    # 1. Load results
    # ------------------------------------------------------------------
    rows = load_results(RESULTS_FILE)
    print(f"Loaded {len(rows)} result rows from {RESULTS_FILE}")

    groups = group_results(rows)
    print(f"Found {len(groups)} unique (method, config) combinations")

    # We will collect summary rows for the final comparison table.
    summary_rows: list[dict] = []

    # ------------------------------------------------------------------
    # 2. For each (method, config) group, create a wandb run
    # ------------------------------------------------------------------
    for (method, cfg_str), group_rows in sorted(groups.items()):
        cfg = json.loads(cfg_str)
        run_name = make_run_name(method, cfg)
        tags = make_tags(method, cfg)

        print(f"\n--- Creating run: {run_name}  ({len(group_rows)} rows) ---")

        run = wandb.init(
            project=PROJECT_NAME,
            entity=entity,
            name=run_name,
            group=method,
            tags=tags,
            config={
                "method": method,
                **cfg,
            },
            reinit=True,
        )

        # Separate synthetic (performance) rows from quality-only rows
        perf_rows = [r for r in group_rows if r.get("prompt_type") == "synthetic" and r.get("seq_len", 0) > 0]
        quality_rows = [r for r in group_rows if r.get("prompt_type") == "wikitext"]

        # Log each individual data point as a step
        for step_idx, row in enumerate(perf_rows):
            log_dict = {}
            for key in METRIC_KEYS:
                val = row.get(key)
                if val is not None:
                    log_dict[key] = val
            # Also log extra fields
            if row.get("prompt_type"):
                log_dict["prompt_type"] = row["prompt_type"]
            if row.get("tokens_generated"):
                log_dict["tokens_generated"] = row["tokens_generated"]
            wandb.log(log_dict, step=step_idx)

        # Compute averages over performance rows for the summary
        avg_metrics = {}
        for key in METRIC_KEYS:
            val = avg([r.get(key) for r in perf_rows])
            if val is not None:
                avg_metrics[f"avg_{key}"] = round(val, 4)

        # Add perplexity from wikitext rows if available
        ppl_values = [r["perplexity"] for r in quality_rows if r.get("perplexity") is not None]
        if ppl_values:
            avg_metrics["perplexity"] = round(sum(ppl_values) / len(ppl_values), 4)

        # Log averages to run summary
        for k, v in avg_metrics.items():
            run.summary[k] = v

        # Build a summary row for the comparison table
        summary_row = {"run_name": run_name, "method": method}
        summary_row.update(cfg)
        summary_row.update(avg_metrics)
        summary_rows.append(summary_row)

        run.finish()

    # ------------------------------------------------------------------
    # 3. Upload plots as a wandb artifact
    # ------------------------------------------------------------------
    if PLOTS_DIR.exists() and any(PLOTS_DIR.iterdir()):
        print("\n--- Uploading plots as artifact ---")
        run = wandb.init(
            project=PROJECT_NAME,
            entity=entity,
            name="plots-artifact-upload",
            job_type="artifact-upload",
            reinit=True,
        )

        artifact = wandb.Artifact(
            name="benchmark-plots",
            type="plots",
            description="KV-cache benchmark visualisation plots",
        )

        for plot_file in sorted(PLOTS_DIR.iterdir()):
            if plot_file.is_file():
                artifact.add_file(str(plot_file))
                print(f"  Added: {plot_file.name}")

        run.log_artifact(artifact)

        # Also log image files directly so they show in the run dashboard
        for plot_file in sorted(PLOTS_DIR.iterdir()):
            if plot_file.suffix.lower() == ".png":
                wandb.log({plot_file.stem: wandb.Image(str(plot_file))})

        run.finish()
    else:
        print(f"\nNo plots found in {PLOTS_DIR}, skipping artifact upload.")

    # ------------------------------------------------------------------
    # 4. Create a summary comparison table
    # ------------------------------------------------------------------
    print("\n--- Creating summary comparison table ---")
    run = wandb.init(
        project=PROJECT_NAME,
        entity=entity,
        name="summary-comparison",
        job_type="summary",
        reinit=True,
    )

    # Determine all columns across summary rows
    all_columns = []
    seen = set()
    for row in summary_rows:
        for k in row:
            if k not in seen:
                all_columns.append(k)
                seen.add(k)

    table = wandb.Table(columns=all_columns)
    for row in summary_rows:
        table.add_data(*[row.get(c) for c in all_columns])

    wandb.log({"method_comparison": table})

    # Also store summary metrics on this run for quick access
    run.summary["num_methods"] = len(summary_rows)
    run.summary["num_total_rows"] = len(rows)

    run.finish()

    # ------------------------------------------------------------------
    # 5. Print project URL
    # ------------------------------------------------------------------
    project_url = f"https://wandb.ai/{entity}/{PROJECT_NAME}"
    print(f"\n{'=' * 60}")
    print(f"Done! View your project at:")
    print(f"  {project_url}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
