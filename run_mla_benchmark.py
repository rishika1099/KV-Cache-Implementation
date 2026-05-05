"""
MLA KV-Cache Benchmark — compare base GQA models vs TransMLA-converted models.

Follows configs/default.yaml exactly for dataset settings (sequence_lengths,
synthetic n_per_length, wikitext n_examples, longbench hf_path/tasks/n_per_task)
and model settings (max_new_tokens, device, dtype).

Comparisons
-----------
  llama2  : llama2-7B (MHA baseline) vs llama2-7B-deepseek (MLA, latent KV cache)
  qwen25  : Qwen2.5-7B (GQA baseline) vs Qwen2.5-7B-deepseek (MLA, latent KV cache)

MLA is a different model architecture — it is NOT a compression method applied to the
same model.  Each pair loads two separate models and compares them directly.

Why latent caching matters
--------------------------
Without latent caching the TransMLA-converted model uses DynamicCache and stores
fully-expanded (K, V).  K shape becomes (batch, num_heads, seq, qk_head_dim=192)
— actually MORE memory than the original model's 128-dim heads.

With latent caching only (c_kv_norm, k_rot_roped) is stored per layer:
  (batch, seq, kv_lora_rank=512) + (batch, 1, seq, qk_rope_head_dim=64)
  = 576 scalars/token/layer  ≈  14–18× smaller than full K/V.

Phases (per model pair)
-----------------------
  1. Load config + datasets via DatasetLoader
  2. Baseline  : synthetic throughput/memory  +  WikiText PPL  +  LongBench
  3. MLA-latent: synthetic throughput/memory  +  WikiText PPL  +  LongBench

Output
------
  results/mla_results.jsonl   — one record per (model, phase, prompt)
  results/mla_results.csv     — same data as CSV

Usage
-----
  # Both pairs, config from configs/default.yaml:
  python run_mla_benchmark.py --model_pair llama2 qwen25 --export_longbench_qa results/longbench_qa_llama2_qwen25.json

  # Single pair, override config:
  python run_mla_benchmark.py --model_pair llama2 --config configs/default.yaml

  # Skip LongBench:
  python run_mla_benchmark.py --model_pair llama2 --skip_longbench

  # Dry run (1 prompt, 20 tokens, 2 PPL samples, 2 LongBench per task):
  python run_mla_benchmark.py --model_pair llama2 --dry_run

File-system structure (input)
------------------------------
  run_mla_benchmark.py
  configs/default.yaml               ← read for all dataset / model settings
  benchmark/
    datasets.py                      ← DatasetLoader
    runner.py                        ← _cache_to_tuple helper
  results/
    mla_results.jsonl                ← written by this script
    mla_results.csv
  ../TransMLA_NeurIPS_2025/          ← auto-detected TransMLA repo
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
TRANSMLA_DIR = SCRIPT_DIR / "methods"
sys.path.insert(0, str(SCRIPT_DIR))      # benchmark.runner, benchmark.datasets
sys.path.insert(0, str(TRANSMLA_DIR))    # transmla.*

# ── Default model paths ───────────────────────────────────────────────────────
MODEL_PATHS = {
    "llama2": {
        "base":       "",
        "mla":        "",
        "label_base": "llama2-7B",
        "label_mla":  "llama2-7B-MLA-latent-2048",
    },
    "qwen25": {
        "base":       "",
        "mla":        "",
        "label_base": "qwen2.5-7B",
        "label_mla":  "qwen2.5-7B-MLA-latent-256",
    },
}


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Baseline generate (base model, standard DynamicCache) ─────────────────────

def generate_baseline(model, tokenizer, prompt, max_new_tokens, device):
    """
    Standard HF greedy decode loop using DynamicCache.

    Input : prompt string
    Output: (generated_text: str, metrics: dict)

    metrics keys:
      kv_cache_mb, peak_memory_gb, ttft_ms,
      throughput_tps, per_token_latency_ms, tokens_generated, input_len
    """
    from benchmark.runner import _cache_to_tuple

    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)

    past_kv = prefill_out.past_key_values
    torch.cuda.synchronize(device)
    ttft = time.perf_counter() - t0

    next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = []
    t_decode_start = time.perf_counter()

    for _ in range(max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token.item())
        with torch.no_grad():
            step_out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
        past_kv    = step_out.past_key_values
        next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    kv_tuples = _cache_to_tuple(past_kv)
    kv_bytes  = sum(
        k.element_size() * k.numel() + v.element_size() * v.numel()
        for k, v in kv_tuples
    )

    n_gen       = len(generated_ids)
    decode_time = t_end - t_decode_start
    return tokenizer.decode(generated_ids, skip_special_tokens=True), {
        "kv_cache_mb":          kv_bytes / 1e6,
        "peak_memory_gb":       torch.cuda.max_memory_allocated(device) / 1e9,
        "ttft_ms":              ttft * 1000,
        "throughput_tps":       n_gen / decode_time if decode_time > 0 else 0.0,
        "per_token_latency_ms": decode_time / n_gen * 1000 if n_gen > 0 else 0.0,
        "tokens_generated":     n_gen,
        "input_len":            input_len,
    }


# ── MLA latent generate (MLA model, MLALatentCache) ───────────────────────────

def generate_mla_latent(model, tokenizer, prompt, max_new_tokens, device):
    """
    Greedy decode loop for a TransMLA-converted model using MLALatentCache.

    Creates a fresh MLALatentCache per call.  The cache stores only
    (c_kv_norm, k_rot_roped) per layer — see transmla/transformers/mla_latent.py.
    kv_cache_mb reflects actual latent bytes, not full K/V.

    Input : prompt string
    Output: (generated_text: str, metrics: dict)  — same keys as generate_baseline
    """
    from transmla.transformers.mla_latent import MLALatentCache

    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    cache = MLALatentCache()
    with torch.no_grad():
        prefill_out = model(**inputs, past_key_values=cache, use_cache=True)

    past_kv = prefill_out.past_key_values   # same MLALatentCache, now populated
    torch.cuda.synchronize(device)
    ttft = time.perf_counter() - t0

    next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = []
    t_decode_start = time.perf_counter()

    for _ in range(max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token.item())
        with torch.no_grad():
            step_out = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
        past_kv    = step_out.past_key_values
        next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    n_gen       = len(generated_ids)
    decode_time = t_end - t_decode_start
    return tokenizer.decode(generated_ids, skip_special_tokens=True), {
        "kv_cache_mb":          past_kv.get_cache_bytes() / 1e6,
        "peak_memory_gb":       torch.cuda.max_memory_allocated(device) / 1e9,
        "ttft_ms":              ttft * 1000,
        "throughput_tps":       n_gen / decode_time if decode_time > 0 else 0.0,
        "per_token_latency_ms": decode_time / n_gen * 1000 if n_gen > 0 else 0.0,
        "tokens_generated":     n_gen,
        "input_len":            input_len,
    }


# ── Results logger ────────────────────────────────────────────────────────────

class MLAResultsLogger:
    """
    Appends records to results/mla_results.jsonl and results/mla_results.csv.

    Each record has: pair, model_label, variant, prompt_type, task, seq_len,
    prompt_id, kv_cache_mb, peak_memory_gb, ttft_ms, throughput_tps,
    per_token_latency_ms, tokens_generated, input_len,
    compression_ratio, longbench_score, ppl.
    """
    _FIELDS = [
        "pair", "model_label", "variant", "prompt_type", "task",
        "seq_len", "prompt_id",
        "kv_cache_mb", "peak_memory_gb", "ttft_ms",
        "throughput_tps", "per_token_latency_ms",
        "tokens_generated", "input_len",
        "compression_ratio", "longbench_score", "ppl",
    ]

    def __init__(self, results_dir: Path):
        results_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = results_dir / "mla_results.jsonl"
        self.csv_path   = results_dir / "mla_results.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._FIELDS).writeheader()

    def log(self, record: dict) -> dict:
        row = {f: record.get(f, "") for f in self._FIELDS}
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._FIELDS).writerow(row)
        return row


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_model(model_path, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tqdm.write(f"  Loading base model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    return model, tokenizer


def load_mla_latent_model(mla_path, device, dtype):
    """
    Load TransMLA-converted model with LlamaMLALatentForCausalLM.
    Uses local TransMLA source code — no trust_remote_code needed.
    Config fields kv_lora_rank, qk_rope_head_dim, etc. are read from config.json
    at mla_path via LlamaMLAConfig.from_pretrained().
    """
    from transmla.transformers.llama.configuration_llamamla import LlamaMLAConfig
    from transmla.transformers.llama.modeling_llamamla_latent import LlamaMLALatentForCausalLM
    from transformers import AutoTokenizer

    tqdm.write(f"  Loading MLA latent model: {mla_path}")
    tokenizer = AutoTokenizer.from_pretrained(mla_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = LlamaMLAConfig.from_pretrained(mla_path)
    model  = LlamaMLALatentForCausalLM.from_pretrained(
        mla_path, config=config, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    return model, tokenizer


# ── Run one model through all phases ─────────────────────────────────────────

def run_model_phases(
    model, tokenizer, generate_fn,
    pair, label, variant,
    dataset_loader, config,
    logger, seed, device,
    baseline_kv_by_prompt=None,
    skip_longbench=False,
    longbench_qa_records=None,
):
    """
    Run synthetic, WikiText PPL, and LongBench phases for one model.

    generate_fn: generate_baseline or generate_mla_latent
    baseline_kv_by_prompt: dict prompt_id -> kv_cache_mb from the baseline run,
        used to compute compression_ratio for the MLA run. Pass None for baseline.

    Returns: dict prompt_id -> kv_cache_mb (for passing to the MLA run).
    """
    max_new_tokens = config["model"]["max_new_tokens"]
    kv_by_prompt   = {}

    # ── SYNTHETIC ─────────────────────────────────────────────────────────────
    if config["datasets"]["synthetic"]["enabled"] and dataset_loader.synthetic_prompts:
        tqdm.write(f"\n  [{label}] Synthetic throughput / memory")
        for p in tqdm(dataset_loader.synthetic_prompts, desc=f"{variant} synthetic"):
            set_seeds(seed)
            try:
                _, metrics = generate_fn(model, tokenizer, p["prompt"], max_new_tokens, device)
                baseline_kv = (baseline_kv_by_prompt or {}).get(p["prompt_id"])
                ratio = (
                    baseline_kv / metrics["kv_cache_mb"]
                    if baseline_kv and metrics["kv_cache_mb"] > 0 else None
                )
                kv_by_prompt[p["prompt_id"]] = metrics["kv_cache_mb"]
                logger.log({
                    "pair": pair, "model_label": label, "variant": variant,
                    "prompt_type": "synthetic", "task": "n/a",
                    "seq_len": p["seq_len"], "prompt_id": p["prompt_id"],
                    **metrics,
                    "compression_ratio": ratio,
                })
                tqdm.write(
                    f"    {variant:12s} | seq={p['seq_len']:5d} | "
                    f"kv={metrics['kv_cache_mb']:.1f}MB | "
                    f"tps={metrics['throughput_tps']:.1f} | "
                    f"mem={metrics['peak_memory_gb']:.2f}GB"
                    + (f" | ratio={ratio:.1f}x" if ratio else "")
                )
            except Exception as e:
                tqdm.write(f"    [ERROR] synthetic {p['prompt_id']}: {e}")
                tqdm.write(traceback.format_exc())

    # ── WIKITEXT PPL ──────────────────────────────────────────────────────────
    if config["datasets"]["wikitext"]["enabled"] and dataset_loader.wikitext_examples:
        tqdm.write(f"\n  [{label}] WikiText-103 perplexity")
        set_seeds(seed)
        try:
            ppl = dataset_loader.compute_perplexity(model, tokenizer, device=device)
            tqdm.write(f"    PPL = {ppl:.4f}")
            logger.log({
                "pair": pair, "model_label": label, "variant": variant,
                "prompt_type": "wikitext", "task": "ppl",
                "ppl": ppl,
            })
        except Exception as e:
            tqdm.write(f"    [ERROR] PPL: {e}")

    # ── LONGBENCH ─────────────────────────────────────────────────────────────
    if (not skip_longbench
            and config["datasets"]["longbench"]["enabled"]
            and dataset_loader.longbench_examples):
        tqdm.write(f"\n  [{label}] LongBench")
        for task_name, examples in dataset_loader.longbench_examples.items():
            set_seeds(seed)
            preds, refs = [], []
            for ex in tqdm(examples, desc=f"{variant}/{task_name}", leave=False):
                try:
                    context  = ex.get("context", "")
                    question = ex.get("input", "")
                    answers  = ex.get("answers", [""])
                    if not isinstance(answers, list):
                        answers = [str(answers)]
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    pred, metrics = generate_fn(
                        model, tokenizer, prompt, max_new_tokens, device
                    )
                    preds.append(pred)
                    refs.append(answers)
                    if longbench_qa_records is not None:
                        longbench_qa_records.append({
                            "pair": pair, "model_label": label, "variant": variant,
                            "task": task_name,
                            "prompt": prompt,
                            "correct_answers": answers,
                            "model_answer": pred,
                        })
                    kv_by_prompt[f"{task_name}_{len(preds)}"] = metrics["kv_cache_mb"]
                    logger.log({
                        "pair": pair, "model_label": label, "variant": variant,
                        "prompt_type": "longbench", "task": task_name,
                        "prompt_id": f"{task_name}_{len(preds)}",
                        **metrics,
                    })
                except Exception as e:
                    tqdm.write(f"    [ERROR] {task_name} example: {e}")
                    preds.append("")
                    refs.append([""])

            if preds:
                from benchmark.datasets import DatasetLoader
                score = DatasetLoader.score_longbench(task_name, preds, refs)
                tqdm.write(f"    {task_name:20s} | score={score:.4f}")
                logger.log({
                    "pair": pair, "model_label": label, "variant": variant,
                    "prompt_type": "longbench_score", "task": task_name,
                    "longbench_score": score,
                })

    return kv_by_prompt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KV cache benchmark: base GQA models vs TransMLA latent-cache models"
    )
    parser.add_argument(
        "--model_pair", nargs="+", choices=["llama2", "qwen25"],
        default=["llama2"],
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--skip_longbench", action="store_true",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="1 prompt/len, 20 tokens, 2 PPL samples, 2 LongBench examples/task",
    )
    parser.add_argument(
        "--export_longbench_qa", default=None, metavar="PATH",
        help="If set, write LongBench Q&A records (prompt, correct_answers, model_answer) "
             "to this JSON file after all runs complete.",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.dry_run:
        config["datasets"]["synthetic"]["n_per_length"]   = 1
        config["datasets"]["sequence_lengths"]            = [512, 2048]
        config["datasets"]["wikitext"]["n_examples"]      = 2
        config["datasets"]["longbench"]["n_per_task"]     = 2
        config["model"]["max_new_tokens"]                 = 20
        tqdm.write("[DRY RUN] seq_lens=[512,2048], n=1, max_new_tokens=20, ppl_n=2, lb_n=2")

    seed   = config.get("seed", 42)
    device = config["model"]["device"]
    dtype  = torch.float16 if config["model"].get("dtype", "float16") == "float16" else torch.bfloat16

    results_dir = SCRIPT_DIR / "results"
    logger      = MLAResultsLogger(results_dir)
    longbench_qa_records = [] if args.export_longbench_qa else None

    tqdm.write(f"\n{'='*60}")
    tqdm.write("MLA KV-Cache Benchmark")
    tqdm.write(f"Config:  {config_path}")
    tqdm.write(f"Pairs:   {args.model_pair}")
    tqdm.write(f"Seq lens: {config['datasets']['sequence_lengths']}")
    tqdm.write(f"LongBench tasks: {config['datasets']['longbench']['tasks']}")
    tqdm.write(f"{'='*60}\n")

    for pair in args.model_pair:
        paths = MODEL_PATHS[pair]
        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"Pair: {pair}")
        tqdm.write(f"  base → {paths['base']}")
        tqdm.write(f"  mla  → {paths['mla']}")
        tqdm.write(f"{'─'*60}")

        # ── Load base model + datasets ─────────────────────────────────────────
        tqdm.write("\nLoading base model and datasets...")
        set_seeds(seed)
        base_model, base_tokenizer = load_base_model(paths["base"], device, dtype)

        from benchmark.datasets import DatasetLoader
        dataset_loader = DatasetLoader(config, base_tokenizer)
        dataset_loader.load_all()

        # ── Baseline phases ────────────────────────────────────────────────────
        tqdm.write(f"\n{'='*40}")
        tqdm.write(f"BASELINE: {paths['label_base']}")
        tqdm.write(f"{'='*40}")
        baseline_kv = run_model_phases(
            base_model, base_tokenizer, generate_baseline,
            pair, paths["label_base"], "baseline",
            dataset_loader, config,
            logger, seed, device,
            baseline_kv_by_prompt=None,
            skip_longbench=args.skip_longbench,
            longbench_qa_records=longbench_qa_records,
        )

        del base_model
        torch.cuda.empty_cache()
        tqdm.write("\n  Base model unloaded.")

        # ── MLA latent phases ──────────────────────────────────────────────────
        tqdm.write(f"\n{'='*40}")
        tqdm.write(f"MLA LATENT: {paths['label_mla']}")
        tqdm.write(f"{'='*40}")
        mla_model, mla_tokenizer = load_mla_latent_model(paths["mla"], device, dtype)

        run_model_phases(
            mla_model, mla_tokenizer, generate_mla_latent,
            pair, paths["label_mla"], "mla_latent",
            dataset_loader, config,
            logger, seed, device,
            baseline_kv_by_prompt=baseline_kv,
            skip_longbench=args.skip_longbench,
            longbench_qa_records=longbench_qa_records,
        )

        del mla_model
        torch.cuda.empty_cache()
        tqdm.write("\n  MLA model unloaded.")

    if longbench_qa_records is not None:
        qa_path = Path(args.export_longbench_qa)
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        with open(qa_path, "w") as f:
            json.dump(longbench_qa_records, f, indent=2, ensure_ascii=False)
        tqdm.write(f"LongBench Q&A export: {qa_path} ({len(longbench_qa_records)} records)")

    tqdm.write(f"\n{'='*60}")
    tqdm.write("BENCHMARK COMPLETE")
    tqdm.write(f"Results: {logger.jsonl_path}")
    tqdm.write(f"CSV:     {logger.csv_path}")
    tqdm.write(f"{'='*60}")


if __name__ == "__main__":
    main()
