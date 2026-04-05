import json
import os
import csv
from pathlib import Path
from datetime import datetime


class MetricsLogger:
    """
    Appends each run as one JSON line to results/results.jsonl.
    Also maintains results/results.csv for easy pandas loading.
    Writes are flushed and fsync'd immediately (crash-safe).
    """

    FIELDS = [
        "method", "config", "prompt_type", "task", "seq_len",
        "peak_memory_gb", "kv_cache_mb", "compression_ratio",
        "ttft_ms", "throughput_tps", "per_token_latency_ms",
        "perplexity", "task_score", "longbench_score",
        "tokens_generated", "timestamp",
    ]

    def __init__(self, results_dir: Path, prefix: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Each container gets a unique prefix to avoid clobbering in
        # concurrent volume writes (e.g. "baseline", "kivi_a1b2c3d4")
        self.jsonl_path = self.results_dir / f"{prefix}.jsonl"
        self.csv_path = self.results_dir / f"{prefix}.csv"

        # Write CSV header if file doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDS)
                writer.writeheader()

    def log(self, method: str, config: dict, prompt_type: str,
            run_metrics: dict, baseline_kv_mb: float = None,
            task: str = "n/a", perplexity: float = None,
            task_score: float = None, longbench_score: float = None):
        """
        Build and persist one result record.
        compression_ratio = baseline_kv_mb / this_kv_mb
        """
        kv_mb = run_metrics.get('kv_cache_mb', 0.0)
        if baseline_kv_mb and kv_mb > 0:
            compression_ratio = baseline_kv_mb / kv_mb
        else:
            compression_ratio = 1.0

        record = {
            "method":               method,
            "config":               config,
            "prompt_type":          prompt_type,
            "task":                 task,
            "seq_len":              run_metrics.get('input_len', 0),
            "peak_memory_gb":       round(run_metrics.get('peak_memory_gb', 0.0), 4),
            "kv_cache_mb":          round(kv_mb, 4),
            "compression_ratio":    round(compression_ratio, 4),
            "ttft_ms":              round(run_metrics.get('ttft_ms', 0.0), 2),
            "throughput_tps":       round(run_metrics.get('throughput_tps', 0.0), 2),
            "per_token_latency_ms": round(run_metrics.get('per_token_latency_ms', 0.0), 2),
            "perplexity":           round(perplexity, 4) if perplexity is not None else None,
            "task_score":           round(task_score, 4) if task_score is not None else None,
            "longbench_score":      round(longbench_score, 4) if longbench_score is not None else None,
            "tokens_generated":     run_metrics.get('tokens_generated', 0),
            "timestamp":            datetime.utcnow().isoformat() + "Z",
        }

        # Append to JSONL (crash-safe)
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())

        # Append to CSV
        flat = {k: json.dumps(v) if isinstance(v, dict) else v
                for k, v in record.items()}
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writerow(flat)
            f.flush()
            os.fsync(f.fileno())

        return record

    def load_completed(self):
        """
        Load all completed runs from results.jsonl.
        Returns: list of record dicts
        """
        records = []
        if not self.jsonl_path.exists():
            return records
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def build_completed_set(self):
        """
        Return a set of (method, config_hash, prompt_id) for --resume logic.
        config_hash is a stable string representation of the config dict.
        """
        completed = set()
        for rec in self.load_completed():
            config_hash = json.dumps(rec.get('config', {}), sort_keys=True)
            # We don't store prompt_id in records; use seq_len + task as proxy
            key = (
                rec.get('method', ''),
                config_hash,
                rec.get('seq_len', 0),
                rec.get('prompt_type', ''),
                rec.get('task', ''),
            )
            completed.add(key)
        return completed
