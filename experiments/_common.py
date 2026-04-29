"""
Shared helpers used by every experiment in this package.

Centralised so that latency measurement, model loading, and result logging
behave identically across the kernel bench, ablation runner, K-sweep, and
profiler. Adding a new experiment means re-using these — never duplicating
the timing or I/O code.
"""

from __future__ import annotations
import csv
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


# ── Timing primitives ────────────────────────────────────────────────────────

def cuda_sync(device: Any = None) -> None:
    """torch.cuda.synchronize that no-ops on CPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextmanager
def cuda_timer(device: Any = None):
    """
    Context manager that records elapsed wall time across CUDA work.

    Usage:
        with cuda_timer() as t:
            ... GPU work ...
        elapsed_seconds = t()
    """
    cuda_sync(device)
    start = time.perf_counter()
    elapsed: List[float] = [0.0]
    try:
        yield lambda: elapsed[0]
    finally:
        cuda_sync(device)
        elapsed[0] = time.perf_counter() - start


def percentiles(values: List[float], qs=(50, 90, 99)) -> Dict[str, float]:
    """Return {pXX: value, ...} percentiles. Empty input → all zeros."""
    if not values:
        return {f"p{q}": 0.0 for q in qs}
    s = sorted(values)
    out = {}
    for q in qs:
        # nearest-rank percentile (no interpolation — robust on small N)
        idx = max(0, min(len(s) - 1, int(round(q / 100 * (len(s) - 1)))))
        out[f"p{q}"] = s[idx]
    return out


def warmup_then_time(
    fn: Callable[[], None],
    n_warmup: int = 5,
    n_iter: int = 20,
    device: Any = None,
) -> Dict[str, float]:
    """
    Warm-up + timed loop. Returns mean / std / p50 / p90 / p99 in seconds.

    `fn` is called with no arguments and must perform the operation under
    test, including any tensor allocation that should be measured.
    """
    for _ in range(n_warmup):
        fn()
    cuda_sync(device)

    samples: List[float] = []
    for _ in range(n_iter):
        cuda_sync(device)
        t0 = time.perf_counter()
        fn()
        cuda_sync(device)
        samples.append(time.perf_counter() - t0)

    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / max(1, len(samples) - 1)
    pct = percentiles(samples)
    return {
        "mean_s": mean,
        "std_s":  math.sqrt(var),
        **pct,
        "n_iter": len(samples),
    }


# ── Result records (uniform across experiments) ──────────────────────────────

@dataclass
class ExperimentRecord:
    """One row of an experiment result table. Flat for easy CSV / DataFrame."""
    experiment: str                     # "kernel_bench", "ablation", "k_sweep"...
    method:     str                     # registry name
    config:     Dict[str, Any] = field(default_factory=dict)
    metrics:    Dict[str, float] = field(default_factory=dict)
    notes:      str = ""

    def flat(self) -> Dict[str, Any]:
        """Flatten into a single dict (CSV-ready)."""
        out = {
            "experiment": self.experiment,
            "method":     self.method,
            "notes":      self.notes,
        }
        for k, v in self.config.items():
            out[f"cfg.{k}"] = v
        for k, v in self.metrics.items():
            out[f"m.{k}"] = v
        return out


def write_records(records: List[ExperimentRecord],
                  out_path: Path,
                  formats: tuple = ("csv", "json")) -> None:
    """Persist a list of ExperimentRecord rows to disk in CSV and/or JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = [r.flat() for r in records]

    if "json" in formats:
        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump(flat, f, indent=2, default=str)

    if "csv" in formats:
        if not flat:
            return
        all_keys: List[str] = []
        for row in flat:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        with open(out_path.with_suffix(".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for row in flat:
                w.writerow(row)


# ── Model loading helper ─────────────────────────────────────────────────────

def maybe_load_model(model_name: Optional[str], device: str = "cuda"):
    """
    Load HF model + tokenizer. Returns (model, tokenizer).
    Pass model_name=None for kernel-only benches that don't need a real LM.
    """
    if model_name is None:
        return None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    mdl.eval()
    return mdl, tok


def device_summary(device: Any = None) -> Dict[str, Any]:
    """Pretty record of the device — included in every output for reproducibility."""
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version":  torch.__version__,
    }
    if torch.cuda.is_available():
        idx = 0 if device is None else (device.index if hasattr(device, "index") else 0)
        info.update({
            "device_name":   torch.cuda.get_device_name(idx),
            "compute_cap":   torch.cuda.get_device_capability(idx),
            "total_mem_gb":  torch.cuda.get_device_properties(idx).total_memory / 1e9,
        })
    try:
        import triton
        info["triton_version"] = triton.__version__
    except ImportError:
        info["triton_version"] = None
    return info
