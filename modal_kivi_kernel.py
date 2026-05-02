"""
modal_kivi_kernel.py

Run LongBench with the KIVI kernel implementation (real CUDA GEMV + Triton quantization).
Uses LlamaForCausalLM_KIVI with the compiled kivi_gemv CUDA extension.

Examples:
    modal run modal_kivi_kernel.py                          # KIVI 2-bit, all tasks
    modal run modal_kivi_kernel.py --bits 4
    modal run modal_kivi_kernel.py --bits 2 --residual_length 64
    modal run modal_kivi_kernel.py --tasks qasper --n_per_task 5
"""

import json
import sys
from pathlib import Path

import modal

app = modal.App("kv-kivi-kernel")

_base = Path(__file__).parent

# pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel has nvcc and CUDA headers needed to compile kivi_gemv
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install(
        "transformers==4.44.2",
        "datasets>=2.18.0,<4.0",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "rouge-score>=0.1.2",
        "triton>=2.3.0",
    )
    .add_local_dir(str(_base / "methods"),   "/app/methods",   copy=True)
    .add_local_dir(str(_base / "benchmark"), "/app/benchmark", copy=True)
    # Compile the KIVI CUDA extension and install it so `import kivi_gemv` works
    .run_commands(
        "cd /app/methods/kivi_kernels && "
        "TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' python setup.py install 2>&1 | tail -5"
    )
)

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
RESULTS_PATH    = Path("/results")

HF_DATASET = "vm2825/longbench-raw"
MAX_LENGTH  = 4096
ALL_TASKS   = ["qasper", "multifieldqa_en", "triviaqa", "2wikimqa", "multi_news", "lcc"]

DATASET2PROMPT = {
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n Answer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question "
        "based on the above text, only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any "
        "other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on "
        "the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
}

DATASET2MAXLEN = {
    "qasper":          128,
    "multifieldqa_en":  64,
    "triviaqa":         32,
    "2wikimqa":         32,
    "multi_news":      512,
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


def _generate_kivi(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    """Manual token-by-token greedy decode with KIVI's custom past_key_values (9-tuples)."""
    import torch

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        # Prefill
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        if next_token.item() == tokenizer.eos_token_id:
            return tokenizer.decode(next_token[0], skip_special_tokens=True)

        generated = [next_token.item()]

        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            if next_token.item() == tokenizer.eos_token_id:
                break
            generated.append(next_token.item())

    return tokenizer.decode(generated, skip_special_tokens=True)


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
def run_kivi_kernel(
    bits: int,
    residual_length: int,
    group_size: int,
    examples: list,
    model_name: str,
    seed: int = 42,
) -> list:
    _bootstrap()
    _set_seeds(seed)

    import torch
    from transformers import AutoTokenizer, AutoConfig
    from methods.llama_kivi_model import LlamaForCausalLM_KIVI
    from benchmark.datasets import DatasetLoader

    method_cfg = {"bits": bits, "residual_length": residual_length, "group_size": group_size}
    print(f"[modal] Loading {model_name} with KIVI kernel cfg={method_cfg}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name)
    config.k_bits = bits
    config.v_bits = bits
    config.group_size = group_size
    config.residual_length = residual_length

    model = LlamaForCausalLM_KIVI.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print(f"[modal] Model loaded. Running {len(examples)} examples...")

    results = []
    for i, ex in enumerate(examples):
        task           = ex["task"]
        prompt         = ex["prompt"]
        answers        = ex["answers"]
        task_max_toks  = ex.get("max_new_tokens", 200)
        _set_seeds(seed)
        try:
            pred = _generate_kivi(model, tokenizer, prompt, task_max_toks, "cuda")
            score = DatasetLoader.score_longbench(task, [pred], [answers])
            results.append({
                "task": task, "method": "kivi_kernel",
                "config": method_cfg, "prompt": prompt,
                "pred": pred, "answers": answers, "score": score,
            })
            print(f"  [{i+1}/{len(examples)}] {task:20s} score={score:.3f}  pred={pred[:60]!r}")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {task} example {i}: {e}\n{traceback.format_exc()}")
            results.append({
                "task": task, "method": "kivi_kernel",
                "config": method_cfg, "prompt": prompt,
                "pred": "", "answers": answers, "score": 0.0,
            })

    # Save to volume
    cfg_tag = f"kivi_kernel_{bits}bit_res{residual_length}_gs{group_size}"
    out = RESULTS_PATH / f"longbench_{cfg_tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"[modal] Results saved to volume: {out}")
    return results


@app.local_entrypoint()
def main(
    bits: int = 2,
    residual_length: int = 128,
    group_size: int = 32,
    model: str = "meta-llama/Llama-2-7b-chat-hf",
    tasks: str = "",
    n_per_task: int = 9999,
):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    method_cfg = {"bits": bits, "residual_length": residual_length, "group_size": group_size}
    task_list = [t.strip() for t in tasks.split(",") if t.strip()] or ALL_TASKS
    print(f"Method: kivi_kernel  cfg={json.dumps(method_cfg, sort_keys=True)}")
    print(f"Tasks:  {task_list}  ({n_per_task} per task)")
    print(f"Model:  {model}")

    print(f"Loading tokenizer for truncation ({model})...")
    tokenizer = AutoTokenizer.from_pretrained(model)

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
            tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokens) > MAX_LENGTH:
                half = MAX_LENGTH // 2
                prompt = (
                    tokenizer.decode(tokens[:half],  skip_special_tokens=True) +
                    tokenizer.decode(tokens[-half:], skip_special_tokens=True)
                )
                truncated += 1
            all_examples.append({
                "task": task, "prompt": prompt,
                "answers": answers, "max_new_tokens": task_max_tokens,
            })
            count += 1
        print(f"    {count} examples  ({truncated} middle-truncated to {MAX_LENGTH} tokens)")

    print(f"\nTotal: {len(all_examples)} examples — launching container...")

    results = run_kivi_kernel.remote(
        bits, residual_length, group_size, all_examples, model,
    )

    # Save locally
    cfg_tag = f"kivi_kernel_{bits}bit_res{residual_length}_gs{group_size}"
    out = Path(__file__).parent / "results" / f"longbench_{cfg_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Local results saved to {out}")

    # Summary
    from collections import defaultdict
    task_scores = defaultdict(list)
    for r in results:
        task_scores[r["task"]].append(r["score"])

    cfg_str = json.dumps(method_cfg, sort_keys=True)
    print(f"\n── kivi_kernel {cfg_str} ───────────────────────────────────")
    for task in task_list:
        vals = task_scores[task]
        avg = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {task:<22}  score={avg:.3f}")
    all_vals = [s for v in task_scores.values() for s in v]
    if all_vals:
        print(f"  {'overall':<22}  score={sum(all_vals)/len(all_vals):.3f}")
