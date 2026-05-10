"""
modal_longbench_snapkv.py

LongBench quality evaluation for SnapKV (budget_ratio=0.4) vs baseline.
6 tasks × 20 examples = 120 examples per method.

Tasks: qasper, multifieldqa_en, triviaqa, 2wikimqa, multi_news, lcc

Run:
    modal run modal_longbench_snapkv.py                     # baseline
    modal run modal_longbench_snapkv.py --method snapkv     # SnapKV 0.4

Output JSON:
    results/longbench_snapkv_baseline.json
    results/longbench_snapkv_snapkv_0.4.json
"""

import json
import sys
from pathlib import Path

import modal

app   = modal.App("kv-longbench-snapkv")
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
MAX_LENGTH  = 4096
ALL_TASKS   = ["qasper", "multifieldqa_en", "triviaqa", "2wikimqa", "multi_news", "lcc"]

DATASET2PROMPT = {
    "qasper":           "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en":  "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "triviaqa":         "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "2wikimqa":         "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multi_news":       "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "lcc":              "Please complete the code given below. \n{context}Next line of code:\n",
}

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
    import random, numpy as np, torch
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
    model_name: str,
    task_list: list,
    n_per_task: int,
    seed: int = 42,
) -> list:
    _bootstrap()
    _set_seeds(seed)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from methods.baseline       import BaselineMethod
    from methods.snapkv_eviction import SnapKVMethod
    from benchmark.runner       import generate_with_method
    from benchmark.datasets     import DatasetLoader

    print(f"[modal] Loading {model_name}  method={method_name}  cfg={method_cfg}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    print(f"[modal] Model loaded. Mem={torch.cuda.memory_allocated()/1e9:.1f} GB")

    if method_name == "baseline":
        method = BaselineMethod()
    elif method_name == "snapkv":
        method = SnapKVMethod(
            budget_ratio=method_cfg.get("budget_ratio", 0.4),
            observation_window=method_cfg.get("observation_window", 32),
            kernel_size=method_cfg.get("kernel_size", 7),
            sink_size=method_cfg.get("sink_size", 0),
            pooling=method_cfg.get("pooling", "avgpool"),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Build prompts
    all_examples = []
    for task in task_list:
        print(f"  Loading {task}...")
        ds = load_dataset(HF_DATASET, task, split="test")
        template    = DATASET2PROMPT[task]
        task_max_nl = DATASET2MAXLEN[task]
        truncated   = 0
        count       = 0
        for ex in ds:
            if count >= n_per_task:
                break
            answers = ex.get("answers", [""])
            if not isinstance(answers, list):
                answers = [str(answers)]
            prompt = template.format(**ex)
            tokens = tokenizer(prompt, truncation=False).input_ids
            if len(tokens) > MAX_LENGTH:
                half = MAX_LENGTH // 2
                prompt = (
                    tokenizer.decode(tokens[:half],  skip_special_tokens=True) +
                    tokenizer.decode(tokens[-half:], skip_special_tokens=True)
                )
                truncated += 1
            all_examples.append({
                "task": task, "prompt": prompt,
                "answers": answers, "max_new_tokens": task_max_nl,
            })
            count += 1
        print(f"    {count} examples  ({truncated} truncated to {MAX_LENGTH} tok)")

    results = []
    for ex in all_examples:
        task           = ex["task"]
        task_max_nl    = ex["max_new_tokens"]
        _set_seeds(seed)
        try:
            pred, metrics = generate_with_method(
                model, tokenizer, method,
                prompt=ex["prompt"],
                max_new_tokens=task_max_nl,
                device="cuda",
            )
            score = DatasetLoader.score_longbench(task, [pred], [ex["answers"]])
            results.append({
                "task":    task,
                "method":  method_name,
                "config":  method_cfg,
                "pred":    pred,
                "answers": ex["answers"],
                "score":   score,
                "metrics": {k: round(v, 4) for k, v in metrics.items()
                            if isinstance(v, float)},
            })
            print(f"  {method_name:10s} | {task:20s} | score={score:.3f}  "
                  f"kv={metrics.get('kv_cache_mb', 0):.0f}MB  "
                  f"ttft={metrics.get('ttft_ms', 0):.0f}ms")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {task}: {e}\n{traceback.format_exc()}")
            results.append({
                "task": task, "method": method_name, "config": method_cfg,
                "pred": "", "answers": ex["answers"], "score": 0.0,
            })

    # Save to volume
    ratio_tag = f"_{method_cfg.get('budget_ratio', 0.4)}" if method_name == "snapkv" else ""
    tag = f"longbench_snapkv_{method_name}{ratio_tag}"
    out = RESULTS_PATH / f"{tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"[modal] Saved → {out}")
    return results


@app.local_entrypoint()
def main(
    method: str          = "baseline",
    budget_ratio: float  = 0.4,
    observation_window: int = 32,
    kernel_size: int     = 7,
    sink_size: int       = 0,
    pooling: str         = "avgpool",
    model: str           = "meta-llama/Llama-2-7b-chat-hf",
    tasks: str           = "",
    n_per_task: int      = 20,
    seed: int            = 42,
):
    """
    LongBench quality eval for SnapKV vs baseline.

        modal run modal_longbench_snapkv.py                     # baseline
        modal run modal_longbench_snapkv.py --method snapkv     # SnapKV 0.4
    """
    if method == "baseline":
        method_cfg = {}
    elif method == "snapkv":
        method_cfg = {
            "budget_ratio":       budget_ratio,
            "observation_window": observation_window,
            "kernel_size":        kernel_size,
            "sink_size":          sink_size,
            "pooling":            pooling,
        }
    else:
        raise ValueError(f"Unknown method: {method}")

    task_list = [t.strip() for t in tasks.split(",") if t.strip()] or ALL_TASKS
    print(f"Method: {method}  cfg={method_cfg}")
    print(f"Tasks:  {task_list}  ({n_per_task} per task)")
    print(f"Model:  {model}")

    results = run_longbench.remote(
        method, method_cfg, model, task_list, n_per_task, seed,
    )

    # Save locally
    ratio_tag = f"_{budget_ratio}" if method == "snapkv" else ""
    out = Path(__file__).parent / "results" / f"longbench_snapkv_{method}{ratio_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")

    # Summary
    from collections import defaultdict
    task_scores = defaultdict(list)
    for r in results:
        task_scores[r["task"]].append(r["score"])

    print(f"\n── {method} {method_cfg} ─────────────────────────────────────────")
    for task in task_list:
        vals = task_scores.get(task, [])
        avg  = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {task:<22}  score={avg:.3f}  (n={len(vals)})")
    all_vals = [s for v in task_scores.values() for s in v]
    if all_vals:
        print(f"  {'overall':<22}  score={sum(all_vals)/len(all_vals):.3f}")
