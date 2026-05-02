"""
modal_longbench.py

Run LongBench for a single method. Trigger each variant independently.

Examples:
    modal run modal_longbench.py                          # baseline
    modal run modal_longbench.py --method kivi --bits 4
    modal run modal_longbench.py --method kivi --bits 2
    modal run modal_longbench.py --method kivi --bits 2 --residual_length 64
    modal run modal_longbench.py --tasks qasper,triviaqa
    modal run modal_longbench.py --n_per_task 10
"""

import json
import sys
from pathlib import Path

import modal

app = modal.App("kv-longbench")

_base = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "datasets>=2.18.0,<4.0",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "rouge-score>=0.1.2",
    )
    .add_local_dir(str(_base / "methods"),   "/app/methods")
    .add_local_dir(str(_base / "benchmark"), "/app/benchmark")
)

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
RESULTS_PATH    = Path("/results")

HF_DATASET = "vm2825/longbench-raw"
MAX_LENGTH = 4096   # KIVI config/model2maxlen.json — Llama-2-7b-chat-hf
ALL_TASKS  = ["qasper", "multifieldqa_en", "triviaqa", "2wikimqa", "multi_news", "lcc"]

# Task-specific prompt templates (from KIVI / LongBench official)
DATASET2PROMPT = {
    "qasper":           "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en":  "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "triviaqa":         "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "2wikimqa":         "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multi_news":       "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "lcc":              "Please complete the code given below. \n{context}Next line of code:\n",
}

# Task-specific max new tokens (from KIVI / LongBench official)
DATASET2MAXLEN = {
    "qasper":           128,
    "multifieldqa_en":  64,
    "triviaqa":         32,
    "2wikimqa":         32,
    "multi_news":       512,
    "lcc":              64,
}


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def _set_seeds(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@app.function(
    image=image,
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        str(HF_CACHE_PATH): model_cache_vol,
        str(RESULTS_PATH):  results_vol,
    },
    timeout=7200,
    memory=65536,
)
def run_longbench(
    method_name: str,
    method_cfg: dict,
    examples: list,
    model_name: str,
    max_new_tokens: int = 200,
    seed: int = 42,
) -> list:
    _bootstrap()
    _set_seeds(seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from methods.baseline   import BaselineMethod
    from methods.kivi_quant import KIVIMethod
    from benchmark.runner   import generate_with_method
    from benchmark.datasets import DatasetLoader

    print(f"[modal] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    if method_name == "baseline":
        method = BaselineMethod()
    elif method_name == "kivi":
        method = KIVIMethod(
            bits=method_cfg.get("bits", 4),
            residual_length=method_cfg.get("residual_length", 128),
            group_size=method_cfg.get("group_size", 32),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    cfg_str = json.dumps(method_cfg, sort_keys=True)
    results = []

    for ex in examples:
        task           = ex["task"]
        prompt         = ex["prompt"]
        answers        = ex["answers"]
        task_max_tokens = ex.get("max_new_tokens", max_new_tokens)
        _set_seeds(seed)
        try:
            pred, _ = generate_with_method(
                model, tokenizer, method,
                prompt=prompt,
                max_new_tokens=task_max_tokens,
                device="cuda",
            )
            score = DatasetLoader.score_longbench(task, [pred], [answers])
            results.append({
                "task": task, "method": method_name,
                "config": method_cfg, "prompt": prompt,
                "pred": pred, "answers": answers, "score": score,
            })
            print(f"  {method_name:10s} {cfg_str} | {task:20s} | score={score:.3f}")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {method_name} {task}: {e}\n{traceback.format_exc()}")
            results.append({
                "task": task, "method": method_name,
                "config": method_cfg, "prompt": prompt,
                "pred": "", "answers": answers, "score": 0.0,
            })

    # Save to volume
    cfg_tag = method_name if not method_cfg else f"{method_name}_{'_'.join(str(v) for v in method_cfg.values())}"
    out = RESULTS_PATH / f"longbench_{cfg_tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"[modal] Results saved to volume: {out}")

    return results


@app.local_entrypoint()
def main(
    method: str = "baseline",
    bits: int = 4,
    residual_length: int = 128,
    group_size: int = 32,
    model: str = "meta-llama/Llama-2-7b-chat-hf",
    tasks: str = "",
    n_per_task: int = 9999,
    max_new_tokens: int = 200,
):
    from datasets import load_dataset

    # Build method config
    if method == "baseline":
        method_cfg = {}
    elif method == "kivi":
        method_cfg = {"bits": bits, "residual_length": residual_length, "group_size": group_size}
    else:
        raise ValueError(f"Unknown method: {method}")

    from transformers import AutoTokenizer

    task_list = [t.strip() for t in tasks.split(",") if t.strip()] or ALL_TASKS
    cfg_str = json.dumps(method_cfg, sort_keys=True)
    print(f"Method: {method}  cfg={cfg_str}")
    print(f"Tasks:  {task_list}  ({n_per_task} per task)")
    print(f"Model:  {model}")

    print(f"Loading tokenizer for truncation ({model})...")
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Load examples with KIVI-style middle-truncation at MAX_LENGTH
    all_examples = []
    for task in task_list:
        print(f"  Loading {task}...")
        ds = load_dataset(HF_DATASET, task, split="test")
        template = DATASET2PROMPT[task]
        task_max_tokens = DATASET2MAXLEN[task]
        truncated = 0
        count = 0
        for ex in ds:
            if count >= n_per_task:
                break
            answers = ex.get("answers", [""])
            if not isinstance(answers, list):
                answers = [str(answers)]
            prompt = template.format(**ex)
            # Middle-truncation: keep first half + last half (matches KIVI exactly)
            tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokens) > MAX_LENGTH:
                half = MAX_LENGTH // 2
                prompt = (
                    tokenizer.decode(tokens[:half],  skip_special_tokens=True) +
                    tokenizer.decode(tokens[-half:], skip_special_tokens=True)
                )
                truncated += 1
            all_examples.append({
                "task":           task,
                "prompt":         prompt,
                "answers":        answers,
                "max_new_tokens": task_max_tokens,
            })
            count += 1
        print(f"    {count} examples  ({truncated} middle-truncated to {MAX_LENGTH} tokens)")

    print(f"\nTotal: {len(all_examples)} examples — launching container...")

    results = run_longbench.remote(
        method, method_cfg, all_examples, model, max_new_tokens,
    )

    # Also save locally
    out = Path(__file__).parent / "results" / f"longbench_{method}{'_'+cfg_str if method_cfg else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Local results saved to {out}")

    # Summary
    from collections import defaultdict
    task_scores = defaultdict(list)
    for r in results:
        task_scores[r["task"]].append(r["score"])

    print(f"\n── {method} {cfg_str} ───────────────────────────────────────────────")
    for task in task_list:
        vals = task_scores[task]
        avg = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {task:<22}  score={avg:.3f}")
    all_vals = [s for v in task_scores.values() for s in v]
    print(f"  {'overall':<22}  score={sum(all_vals)/len(all_vals):.3f}" if all_vals else "")
