"""
modal_mla_benchmark.py

Run MLA benchmark (base GQA vs TransMLA latent-cache) on Modal.
Mirrors run_mla_benchmark.py logic; modal_longbench.py is the structural template.

Both --run_baseline and --run_mla default to False.
Pass at least one to actually launch a container.

Examples:
    modal run modal_mla_benchmark.py --run-baseline --run-mla
    modal run modal_mla_benchmark.py --run-baseline
    modal run modal_mla_benchmark.py --run-mla
    modal run modal_mla_benchmark.py --model-pair qwen25 --run-baseline --run-mla
    modal run modal_mla_benchmark.py --run-baseline --run-mla --tasks qasper,triviaqa
    modal run modal_mla_benchmark.py --run-baseline --run-mla --n-per-task 5
    modal run modal_mla_benchmark.py --run-baseline --run-mla --dry-run
"""

import json
import sys
from pathlib import Path

import modal

app = modal.App("kv-mla-benchmark")

_base = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.53.1",
        "datasets>=2.18.0,<4.0",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "rouge-score>=0.1.2",
        "pyyaml>=6.0",
        "tqdm>=4.0",
    )
    .add_local_dir(str(_base / "methods"),   "/app/methods")
    .add_local_dir(str(_base / "benchmark"), "/app/benchmark")
    .add_local_dir(str(_base / "configs"),   "/app/configs")
)

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
RESULTS_PATH    = Path("/results")

ALL_TASKS = ["qasper", "multifieldqa_en", "triviaqa", "2wikimqa", "multi_news", "lcc"]

MODEL_PATHS = {
    "llama2": {
        "base":       "meta-llama/Llama-2-7b-chat-hf",
        "mla":        "botsxc/llama2-7b-chat-mla-2048-8",
        "label_base": "llama2-7B",
        "label_mla":  "llama2-7B-MLA-latent-2048",
    },
    "qwen25": {
        "base":       "Qwen/Qwen2.5-7B-Instruct",
        "mla":        "botsxc/qwen2.5-7b-instruct-mla-256",
        "label_base": "qwen2.5-7B",
        "label_mla":  "qwen2.5-7B-MLA-latent-256",
    },
    "qwen25-1.5b": {
        "base":       "Qwen/Qwen2.5-1.5B-Instruct",
        "mla":        "botsxc/qwen2.5-1.5b-mla",
        "label_base": "qwen2.5-1.5B",
        "label_mla":  "qwen2.5-1.5B-MLA-latent-256",
    },
}

# Task-specific prompt templates (from KIVI / LongBench official)
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

MAX_LENGTH = 2048  # botsxc/llama2-7b-chat-mla-2048-8 was trained with 2048 max positions


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    if "/app/methods" not in sys.path:
        sys.path.insert(0, "/app/methods")


def _set_seeds(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _apply_chat_template(tokenizer, prompt: str, system_prompt: str = "") -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if system_prompt:
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
    return f"[INST] {prompt} [/INST]"


def _load_examples(tokenizer, task_list, n_per_task, hf_dataset_path, use_chat_template: bool = False, system_prompt: str = "", filter_by_length: bool = False):
    from datasets import load_dataset
    all_examples = []
    for task in task_list:
        print(f"  Loading {task}...")
        ds = load_dataset(hf_dataset_path, task, split="test")
        template = DATASET2PROMPT[task]
        task_max_tokens = DATASET2MAXLEN[task]
        truncated = 0
        filtered = 0
        count = 0
        for ex in ds:
            if count >= n_per_task:
                break
            answers = ex.get("answers", [""])
            if not isinstance(answers, list):
                answers = [str(answers)]
            prompt = template.format(**ex)
            tokens = tokenizer(prompt, truncation=False).input_ids
            if len(tokens) > MAX_LENGTH:
                if filter_by_length:
                    filtered += 1
                    continue
                half = MAX_LENGTH // 2
                prompt = (
                    tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                    tokenizer.decode(tokens[-half:], skip_special_tokens=True)
                )
                truncated += 1
            if use_chat_template:
                prompt = _apply_chat_template(tokenizer, prompt, system_prompt)
            all_examples.append({
                "task": task,
                "prompt": prompt,
                "answers": answers,
                "max_new_tokens": task_max_tokens,
            })
            count += 1
        if filter_by_length:
            print(f"    {count} examples  ({filtered} filtered out, >{MAX_LENGTH} tokens)")
        else:
            print(f"    {count} examples  ({truncated} middle-truncated to {MAX_LENGTH} tokens)")
    return all_examples


def _generate_baseline(model, tokenizer, prompt, max_new_tokens, device):
    import time
    import torch
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


def _generate_mla_latent(model, tokenizer, prompt, max_new_tokens, device):
    import time
    import torch
    from transmla.transformers.mla_latent import MLALatentCache

    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    cache = MLALatentCache()
    with torch.no_grad():
        prefill_out = model(**inputs, past_key_values=cache, use_cache=True)

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


def _load_mla_latent_model(mla_path, device, dtype):
    from transmla.transformers.llama.configuration_llamamla import LlamaMLAConfig
    from transmla.transformers.llama.modeling_llamamla_latent import LlamaMLALatentForCausalLM
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(mla_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = LlamaMLAConfig.from_pretrained(mla_path)
    model  = LlamaMLALatentForCausalLM.from_pretrained(
        mla_path, config=config, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    return model, tokenizer


def _run_longbench_phase(model, tokenizer, generate_fn, label, examples, seed):
    from benchmark.datasets import DatasetLoader
    results = []
    for ex in examples:
        task            = ex["task"]
        prompt          = ex["prompt"]
        answers         = ex["answers"]
        task_max_tokens = ex["max_new_tokens"]
        _set_seeds(seed)
        try:
            pred, metrics = generate_fn(model, tokenizer, prompt, task_max_tokens, "cuda")
            score = DatasetLoader.score_longbench(task, [pred], [answers])
            results.append({
                "label": label, "task": task,
                "prompt": prompt, "pred": pred, "answers": answers,
                "score": score, **metrics,
            })
            print(f"  {label:30s} | {task:20s} | score={score:.3f} | kv={metrics['kv_cache_mb']:.1f}MB")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {label} {task}: {e}\n{traceback.format_exc()}")
            results.append({
                "label": label, "task": task,
                "prompt": prompt, "pred": "", "answers": answers, "score": 0.0,
            })
    return results


@app.function(
    image=image,
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        str(HF_CACHE_PATH): model_cache_vol,
        str(RESULTS_PATH):  results_vol,
    },
    timeout=14400,
    memory=65536,
)
def run_mla_benchmark(
    pair: str,
    model_paths: dict,
    task_list: list,
    n_per_task: int,
    hf_dataset_path: str,
    run_baseline: bool,
    run_mla: bool,
    use_chat_template: bool = False,
    system_prompt: str = "",
    seed: int = 42,
    filter_by_length: bool = False,
) -> dict:
    """
    Run LongBench for one model pair (base and/or MLA latent) in a single container.
    Returns {"baseline": [...], "mla_latent": [...]} — absent keys if that variant was skipped.
    """
    _bootstrap()
    _set_seeds(seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── Load dataset examples once using base tokenizer ───────────────────────
    print(f"[modal] Loading tokenizer for dataset preparation: {model_paths['base']}")
    prep_tokenizer = AutoTokenizer.from_pretrained(model_paths["base"])
    if prep_tokenizer.pad_token is None:
        prep_tokenizer.pad_token = prep_tokenizer.eos_token
    examples = _load_examples(prep_tokenizer, task_list, n_per_task, hf_dataset_path, use_chat_template, system_prompt, filter_by_length)
    del prep_tokenizer

    results = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    if run_baseline:
        print(f"\n[modal] Loading base model: {model_paths['base']}")
        tokenizer = AutoTokenizer.from_pretrained(model_paths["base"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_paths["base"], torch_dtype=torch.float16, device_map="cuda"
        )
        model.eval()

        print(f"\n[modal] Running baseline LongBench: {model_paths['label_base']}")
        results["baseline"] = _run_longbench_phase(
            model, tokenizer, _generate_baseline, model_paths["label_base"], examples, seed
        )

        del model
        torch.cuda.empty_cache()
        print("[modal] Base model unloaded.")

    # ── MLA latent ────────────────────────────────────────────────────────────
    if run_mla:
        print(f"\n[modal] Loading MLA latent model: {model_paths['mla']}")
        model, tokenizer = _load_mla_latent_model(model_paths["mla"], "cuda", torch.float16)

        print(f"\n[modal] Running MLA latent LongBench: {model_paths['label_mla']}")
        results["mla_latent"] = _run_longbench_phase(
            model, tokenizer, _generate_mla_latent, model_paths["label_mla"], examples, seed
        )

        del model
        torch.cuda.empty_cache()
        print("[modal] MLA model unloaded.")

    # ── Save to volume ─────────────────────────────────────────────────────────
    variants = "_".join(k for k in ("baseline", "mla_latent") if k in results)
    out = RESULTS_PATH / f"mla_longbench_{pair}_{variants}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"[modal] Results saved to volume: {out}")

    return results


@app.local_entrypoint()
def main(
    model_pair: str = "llama2",
    run_baseline: bool = False,
    run_mla: bool = False,
    tasks: str = "",
    n_per_task: int = 20,
    hf_dataset: str = "vm2825/longbench-raw",
    use_chat_template: bool = False,
    system_prompt: str = "You are a helpful and precise assistant for answering questions.",
    seed: int = 42,
    dry_run: bool = False,
    filter_by_length: bool = False,
):
    if not run_baseline and not run_mla:
        print("[warning] Neither --run_baseline nor --run_mla was set. Nothing to do.")
        return

    if model_pair not in MODEL_PATHS:
        print(f"[error] Unknown model_pair '{model_pair}'. Choose from: {list(MODEL_PATHS)}")
        return

    if dry_run:
        n_per_task = 2
        print("[DRY RUN] n_per_task=2")

    task_list = [t.strip() for t in tasks.split(",") if t.strip()] or ALL_TASKS
    paths = MODEL_PATHS[model_pair]

    print(f"Pair:         {model_pair}")
    print(f"  base  →  {paths['base']}")
    print(f"  mla   →  {paths['mla']}")
    print(f"Run baseline: {run_baseline}")
    print(f"Run MLA:      {run_mla}")
    print(f"Tasks:        {task_list}  ({n_per_task} per task)")
    print(f"\nLaunching container...")

    results = run_mla_benchmark.remote(
        pair=model_pair,
        model_paths=paths,
        task_list=task_list,
        n_per_task=n_per_task,
        hf_dataset_path=hf_dataset,
        run_baseline=run_baseline,
        run_mla=run_mla,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        seed=seed,
        filter_by_length=filter_by_length,
    )

    # ── Save locally ───────────────────────────────────────────────────────────
    variants = "_".join(k for k in ("baseline", "mla_latent") if k in results)
    out = Path(__file__).parent / "results" / f"mla_longbench_{model_pair}_{variants}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Local results saved to {out}")

    # ── Summary ────────────────────────────────────────────────────────────────
    from collections import defaultdict
    for variant, records in results.items():
        task_scores = defaultdict(list)
        for r in records:
            task_scores[r["task"]].append(r["score"])
        label = paths["label_base"] if variant == "baseline" else paths["label_mla"]
        print(f"\n── {label} ({'baseline' if variant == 'baseline' else 'MLA latent'}) ─────────────────────────")
        for task in task_list:
            vals = task_scores.get(task, [])
            avg = sum(vals) / len(vals) if vals else float("nan")
            print(f"  {task:<22}  score={avg:.3f}")
        all_vals = [s for v in task_scores.values() for s in v]
        if all_vals:
            print(f"  {'overall':<22}  score={sum(all_vals)/len(all_vals):.3f}")
