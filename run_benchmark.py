

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from tabulate import tabulate
from tqdm import tqdm

# ── LOCAL IMPORTS ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.datasets import DatasetLoader
from benchmark.metrics import MetricsLogger
from benchmark.runner import generate_with_method
from methods.baseline import BaselineMethod
from methods.kivi_quant import KIVIMethod
from methods.snapkv_eviction import SnapKVMethod
from methods.topk_selection import TopKMethod
from methods.xkv_svd import XKVMethod


# ── SEED HELPER ───────────────────────────────────────────────────────────────

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── METHOD FACTORY ────────────────────────────────────────────────────────────

def build_method(method_name, cfg):
    if method_name == "baseline":
        return BaselineMethod()
    elif method_name == "kivi":
        return KIVIMethod(
            bits=cfg.get('bits', 4),
            residual_length=cfg.get('residual_length', 128),
        )
    elif method_name == "topk":
        return TopKMethod(
            K=cfg.get('K', 512),
            refresh_interval=cfg.get('refresh_interval', 50),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KV Cache Benchmark")
    parser.add_argument('--config', default='configs/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--methods', nargs='+',
                        choices=['baseline', 'kivi', 'topk'],
                        help='Methods to run (default: all enabled in config)')
    parser.add_argument('--seq_lens', nargs='+', type=int,
                        help='Sequence lengths to test (overrides config)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Quick smoke test: 2 prompts per method')
    parser.add_argument('--skip_longbench', action='store_true',
                        help='Skip LongBench evaluation')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed runs')
    parser.add_argument('--model', default=None,
                        help='Override model name from config')
    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation prompt')
    args = parser.parse_args()



    # ── LOAD CONFIG ───────────────────────────────────────────────────────────
    config_path = Path(__file__).parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.model:
        config['model']['name'] = args.model
    if args.seq_lens:
        config['datasets']['sequence_lengths'] = args.seq_lens
    if args.skip_longbench:
        config['datasets']['longbench']['enabled'] = False

    seed = config.get('seed', 42)
    device = config['model']['device']
    max_new_tokens = config['model']['max_new_tokens']
    model_name = config['model']['name']

    # Dry run override
    if args.dry_run:
        config['datasets']['synthetic']['n_per_length'] = 1
        config['datasets']['sequence_lengths'] = [512, 1024]
        config['datasets']['wikitext']['n_examples'] = 5
        config['datasets']['longbench']['n_per_task'] = 2
        max_new_tokens = 20
        tqdm.write("\n[DRY RUN] Reduced dataset sizes and max_new_tokens=20\n")

    # ── DETERMINE ACTIVE METHODS ──────────────────────────────────────────────
    active_methods = []
    method_order = ['baseline', 'kivi', 'xkv', 'snapkv', 'topk']
    requested = set(args.methods) if args.methods else None

    for mname in method_order:
        mcfg = config['methods'].get(mname, {})
        if not mcfg.get('enabled', True):
            continue
        if requested and mname not in requested:
            continue

        if mname == 'baseline':
            active_methods.append(('baseline', {}))
        else:
            for cfg in mcfg.get('configs', [{}]):
                active_methods.append((mname, cfg))

    tqdm.write(f"\nModel:   {model_name}")
    tqdm.write(f"Device:  {device}")
    tqdm.write(f"Seed:    {seed}")
    tqdm.write(f"Methods: {len(active_methods)} configurations")
    tqdm.write(f"Seq lens: {config['datasets']['sequence_lengths']}")

    # ── ESTIMATE RUNTIME ──────────────────────────────────────────────────────
    n_synthetic = (
        len(config['datasets']['sequence_lengths']) *
        config['datasets']['synthetic']['n_per_length']
    )
    n_runs = len(active_methods) * n_synthetic
    est_minutes = n_runs * 4  # ~4 min per method config per seq_len on H100
    tqdm.write(f"\nEstimated runtime: ~{est_minutes} minutes ({est_minutes/60:.1f} hrs)")
    tqdm.write(f"Total configurations x prompts: {n_runs}")

    if not args.yes and not args.dry_run:
        ans = input("\nContinue? [y/n]: ").strip().lower()
        if ans != 'y':
            tqdm.write("Aborted.")
            return

    # ── LOAD MODEL ────────────────────────────────────────────────────────────
    tqdm.write(f"\nLoading model {model_name}...")
    set_seeds(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    tqdm.write("Model loaded successfully.")

    # ── LOAD DATASETS ─────────────────────────────────────────────────────────
    tqdm.write("\nLoading datasets...")
    dataset_loader = DatasetLoader(config, tokenizer)
    dataset_loader.load_all()

    # ── SETUP RESULTS ─────────────────────────────────────────────────────────
    results_dir = Path(__file__).parent / "results"
    logger = MetricsLogger(results_dir)
    completed_set = logger.build_completed_set() if args.resume else set()

    if args.resume and completed_set:
        tqdm.write(f"\n[RESUME] Found {len(completed_set)} completed runs, skipping them.")

    # ── COLLECT ALL PROMPTS ───────────────────────────────────────────────────
    prompts_to_run = dataset_loader.synthetic_prompts  # list of dicts

    # ── RUN BASELINE FIRST to get kv_cache_mb denominators ───────────────────
    tqdm.write("\n" + "=" * 60)
    tqdm.write("PHASE 1: Running baseline to establish compression denominators")
    tqdm.write("=" * 60)

    baseline_kv_cache = {}  # (seq_len, prompt_id) -> kv_cache_mb

    baseline_method = BaselineMethod()

    for prompt_info in tqdm(prompts_to_run, desc="Baseline prefill"):
        set_seeds(seed)
        prompt = prompt_info['prompt']
        seq_len = prompt_info['seq_len']
        pid = prompt_info['prompt_id']

        try:
            _, metrics = generate_with_method(
                model, tokenizer, baseline_method,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            baseline_kv_cache[(seq_len, pid)] = metrics['kv_cache_mb']
            tqdm.write(
                f"  Baseline | seq={seq_len} | "
                f"mem={metrics['peak_memory_gb']:.2f}GB | "
                f"kv={metrics['kv_cache_mb']:.2f}MB | "
                f"tps={metrics['throughput_tps']:.1f}"
            )
        except Exception as e:
            tqdm.write(f"  [ERROR] Baseline failed on {pid}: {e}")
            baseline_kv_cache[(seq_len, pid)] = 1.0  # fallback

        # Log baseline result
        logger.log(
            method="baseline",
            config={},
            prompt_type="synthetic",
            run_metrics=metrics if '_' not in str(metrics) else {},
            baseline_kv_mb=None,
            task="n/a",
        )

    # ── RUN ALL METHODS ───────────────────────────────────────────────────────
    tqdm.write("\n" + "=" * 60)
    tqdm.write("PHASE 2: Running all method configurations")
    tqdm.write("=" * 60)

    summary_rows = []

    for method_name, method_cfg in tqdm(active_methods, desc="Methods"):
        if method_name == "baseline":
            continue  # already ran

        method_obj = build_method(method_name, method_cfg)
        config_str = json.dumps(method_cfg, sort_keys=True)

        method_kv_list = []
        method_tps_list = []
        method_ppl_list = []

        for prompt_info in tqdm(prompts_to_run, desc=f"{method_name} {config_str}", leave=False):
            set_seeds(seed)
            prompt = prompt_info['prompt']
            seq_len = prompt_info['seq_len']
            pid = prompt_info['prompt_id']

            # Resume check
            if args.resume:
                resume_key = (method_name, config_str, seq_len, "synthetic", "n/a")
                if resume_key in completed_set:
                    continue

            try:
                _, metrics = generate_with_method(
                    model, tokenizer, method_obj,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )

                baseline_mb = baseline_kv_cache.get((seq_len, pid), 1.0)
                rec = logger.log(
                    method=method_name,
                    config=method_cfg,
                    prompt_type="synthetic",
                    run_metrics=metrics,
                    baseline_kv_mb=baseline_mb,
                    task="n/a",
                )

                method_kv_list.append(metrics['kv_cache_mb'])
                method_tps_list.append(metrics['throughput_tps'])

                tqdm.write(
                    f"  {method_name:10s} | cfg={config_str:40s} | "
                    f"seq={seq_len:5d} | "
                    f"mem={metrics['peak_memory_gb']:.2f}GB | "
                    f"kv={metrics['kv_cache_mb']:.2f}MB | "
                    f"ratio={rec['compression_ratio']:.2f}x | "
                    f"tps={metrics['throughput_tps']:.1f}"
                )

            except Exception as e:
                tqdm.write(
                    f"  [ERROR] {method_name} cfg={config_str} seq={seq_len}: {e}"
                )
                tqdm.write(traceback.format_exc())
                # Fall back to baseline for this run — don't crash the benchmark
                try:
                    _, fallback_metrics = generate_with_method(
                        model, tokenizer, BaselineMethod(),
                        prompt=prompt, max_new_tokens=max_new_tokens, device=device,
                    )
                    logger.log(
                        method=f"{method_name}_FALLBACK",
                        config=method_cfg,
                        prompt_type="synthetic",
                        run_metrics=fallback_metrics,
                        baseline_kv_mb=baseline_kv_cache.get((seq_len, pid), 1.0),
                        task="n/a",
                    )
                except Exception as e2:
                    tqdm.write(f"  [ERROR] Fallback also failed: {e2}")

        if method_kv_list:
            avg_kv = sum(method_kv_list) / len(method_kv_list)
            avg_tps = sum(method_tps_list) / len(method_tps_list)
            summary_rows.append([method_name, config_str, f"{avg_kv:.2f}", f"{avg_tps:.1f}"])

    # ── WIKITEXT PERPLEXITY ───────────────────────────────────────────────────
    if config['datasets']['wikitext']['enabled'] and dataset_loader.wikitext_examples:
        tqdm.write("\n" + "=" * 60)
        tqdm.write("PHASE 3: Computing perplexity on WikiText-103")
        tqdm.write("=" * 60)

        for method_name, method_cfg in tqdm(active_methods, desc="PPL eval"):
            set_seeds(seed)
            try:
                ppl = dataset_loader.compute_perplexity(
                    model, tokenizer, device=device
                )
                tqdm.write(f"  {method_name:10s} | cfg={json.dumps(method_cfg)} | PPL={ppl:.2f}")
                # Log perplexity as a separate record
                logger.log(
                    method=method_name,
                    config=method_cfg,
                    prompt_type="wikitext",
                    run_metrics={'input_len': 0, 'peak_memory_gb': 0, 'kv_cache_mb': 0,
                                 'ttft_ms': 0, 'throughput_tps': 0, 'per_token_latency_ms': 0,
                                 'tokens_generated': 0},
                    baseline_kv_mb=None,
                    task="n/a",
                    perplexity=ppl,
                )
            except Exception as e:
                tqdm.write(f"  [ERROR] PPL eval failed for {method_name}: {e}")

    # ── LONGBENCH ─────────────────────────────────────────────────────────────
    if (config['datasets']['longbench']['enabled'] and
            dataset_loader.longbench_examples and
            not args.skip_longbench):
        tqdm.write("\n" + "=" * 60)
        tqdm.write("PHASE 4: LongBench evaluation")
        tqdm.write("=" * 60)

        for method_name, method_cfg in tqdm(active_methods, desc="LongBench"):
            method_obj = build_method(method_name, method_cfg)

            for task_name, examples in dataset_loader.longbench_examples.items():
                set_seeds(seed)
                preds = []
                refs = []

                for ex in tqdm(examples, desc=f"{method_name}/{task_name}", leave=False):
                    try:
                        # LongBench examples have 'context', 'input', 'answers' fields
                        context = ex.get('context', '')
                        question = ex.get('input', '')
                        answers = ex.get('answers', [''])
                        if not isinstance(answers, list):
                            answers = [str(answers)]

                        # Data is pre-filtered to < 4096 tokens — no truncation needed
                        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
                        pred, metrics = generate_with_method(
                            model, tokenizer, method_obj,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            device=device,
                        )
                        preds.append(pred)
                        refs.append(answers)  # full list; score_longbench takes max

                    except Exception as e:
                        tqdm.write(f"  [ERROR] LongBench {task_name} example failed: {e}")
                        preds.append("")
                        refs.append("")

                if preds:
                    score = DatasetLoader.score_longbench(task_name, preds, refs)
                    tqdm.write(
                        f"  {method_name:10s} | task={task_name:20s} | score={score:.4f}"
                    )
                    logger.log(
                        method=method_name,
                        config=method_cfg,
                        prompt_type="longbench",
                        run_metrics={'input_len': 0, 'peak_memory_gb': 0, 'kv_cache_mb': 0,
                                     'ttft_ms': 0, 'throughput_tps': 0, 'per_token_latency_ms': 0,
                                     'tokens_generated': 0},
                        baseline_kv_mb=None,
                        task=task_name,
                        longbench_score=score,
                        task_score=score,
                    )

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    tqdm.write("\n" + "=" * 60)
    tqdm.write("BENCHMARK COMPLETE")
    tqdm.write("=" * 60)

    if summary_rows:
        tqdm.write(tabulate(
            summary_rows,
            headers=["Method", "Config", "Avg KV MB", "Avg TPS"],
            tablefmt="grid",
        ))

    tqdm.write(f"\nResults saved to: {logger.jsonl_path}")
    tqdm.write(f"CSV saved to:     {logger.csv_path}")
    tqdm.write("\nNext step: python plot_results.py --results results/results.jsonl")
    tqdm.write("\n" + "=" * 60)
    tqdm.write("Save results before stopping instance:")
    tqdm.write("  gsutil cp -r results/ gs://YOUR_BUCKET/")
    tqdm.write("=" * 60)


if __name__ == "__main__":
    main()
