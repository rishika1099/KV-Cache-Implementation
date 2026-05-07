"""
modal_longbench_kivi_topk.py

LongBench quality benchmark for the combined KIVI + TopK method
(LlamaForCausalLM_KIVITopK — quantised KV cache + sparse attention).

Uses a custom greedy decode loop because the model stores past_key_values
as a 9-tuple per layer (not a DynamicCache), which is incompatible with
model.generate().

Examples:
    # KIVI 4-bit + TopK K=1024 (default)
    modal run modal_longbench_kivi_topk.py

    # Vary K
    modal run modal_longbench_kivi_topk.py --topk-k 2048
    modal run modal_longbench_kivi_topk.py --topk-k 512

    # KIVI 2-bit
    modal run modal_longbench_kivi_topk.py --bits 2

Output:
    results/longbench_kivi_topk_{bits}bit_K{K}_sink{n_sink}_local{n_local}.json
"""

import json
import sys
from pathlib import Path

import modal

app = modal.App("kv-longbench-kivi-topk")
_base = Path(__file__).parent

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
    "qasper":          "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "triviaqa":        "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "2wikimqa":        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multi_news":      "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "lcc":             "Please complete the code given below. \n{context}Next line of code:\n",
}

DATASET2MAXLEN = {
    "qasper":          128,
    "multifieldqa_en": 64,
    "triviaqa":        32,
    "2wikimqa":        32,
    "multi_news":      512,
    "lcc":             64,
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


def _greedy_generate(model, tokenizer, input_ids, max_new_tokens):
    """
    Manual greedy decode compatible with the 9-tuple KV cache format used by
    LlamaForCausalLM_KIVITopK.  Returns the generated token string (decoded).
    """
    import torch
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    with torch.no_grad():
        outputs    = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_kv    = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = [next_token.item()]
        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_id:
                break
            outputs    = model(
                input_ids=next_token, past_key_values=past_kv,
                use_cache=True, return_dict=True,
            )
            past_kv    = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
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
def run_longbench_kivi_topk(
    model_name: str,
    bits: int,
    group_size: int,
    residual_length: int,
    topk_K: int,
    topk_n_sink: int,
    topk_n_local: int,
    task_list: list,
    n_per_task: int,
    seed: int = 42,
) -> list:
    _bootstrap()
    _set_seeds(seed)

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoConfig
    from methods.llama_kivi_topk_model import LlamaForCausalLM_KIVITopK
    from benchmark.datasets import DatasetLoader

    print(f"[modal] Loading {model_name}...")
    print(f"[modal] KIVI: bits={bits}  group_size={group_size}  residual_length={residual_length}")
    print(f"[modal] TopK: K={topk_K}  n_sink={topk_n_sink}  n_local={topk_n_local}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name)
    config.k_bits          = bits
    config.v_bits          = bits
    config.group_size      = group_size
    config.residual_length = residual_length
    config.topk_K          = topk_K
    config.topk_n_sink     = topk_n_sink
    config.topk_n_local    = topk_n_local

    model = LlamaForCausalLM_KIVITopK.from_pretrained(
        model_name, config=config,
        torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    print(f"[modal] Model loaded. Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    all_examples = []
    for task in task_list:
        print(f"  Loading {task}...")
        ds = load_dataset(HF_DATASET, task, split="test")
        template   = DATASET2PROMPT[task]
        task_maxtok = DATASET2MAXLEN[task]
        truncated  = 0
        count      = 0

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
                    tokenizer.decode(tokens[:half], skip_special_tokens=True) +
                    tokenizer.decode(tokens[-half:], skip_special_tokens=True)
                )
                truncated += 1

            all_examples.append({
                "task":           task,
                "prompt":         prompt,
                "answers":        answers,
                "max_new_tokens": task_maxtok,
            })
            count += 1

        print(f"    {count} examples  ({truncated} middle-truncated to {MAX_LENGTH} tokens)")

    results = []
    for ex in all_examples:
        task        = ex["task"]
        prompt      = ex["prompt"]
        answers     = ex["answers"]
        max_new_tok = ex["max_new_tokens"]
        _set_seeds(seed)

        try:
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=False,
            ).input_ids.to("cuda")

            pred  = _greedy_generate(model, tokenizer, input_ids, max_new_tok)
            score = DatasetLoader.score_longbench(task, [pred], [answers])

            results.append({
                "task":    task,
                "method":  "kivi_topk",
                "config":  {
                    "bits":           bits,
                    "group_size":     group_size,
                    "residual_length": residual_length,
                    "K":              topk_K,
                    "n_sink":         topk_n_sink,
                    "n_local":        topk_n_local,
                },
                "prompt":  prompt,
                "pred":    pred,
                "answers": answers,
                "score":   score,
            })
            print(f"  kivi_topk {task:20s} | score={score:.3f}")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {task}: {e}\n{traceback.format_exc()}")
            results.append({
                "task": task, "method": "kivi_topk",
                "config": {}, "prompt": prompt,
                "pred": "", "answers": answers, "score": 0.0,
            })

    tag = f"longbench_kivi_topk_{bits}bit_K{topk_K}_sink{topk_n_sink}_local{topk_n_local}"
    out = RESULTS_PATH / f"{tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"[modal] Results saved to volume: {out}")

    return results


@app.local_entrypoint()
def main(
    model: str        = "meta-llama/Llama-2-7b-chat-hf",
    bits: int         = 4,
    group_size: int   = 32,
    residual_length: int = 128,
    topk_k: int       = 1024,
    topk_n_sink: int  = 128,
    topk_n_local: int = 512,
    tasks: str        = "",
    n_per_task: int   = 20,
):
    """
    LongBench benchmark for KIVI + TopK combined method.

    Examples:
        modal run modal_longbench_kivi_topk.py                        # 4-bit, K=1024
        modal run modal_longbench_kivi_topk.py --bits 2               # 2-bit, K=1024
        modal run modal_longbench_kivi_topk.py --topk-k 2048          # 4-bit, K=2048
        modal run modal_longbench_kivi_topk.py --tasks qasper,triviaqa
    """
    task_list = [t.strip() for t in tasks.split(",") if t.strip()] or ALL_TASKS
    print(f"Method:  kivi_topk")
    print(f"KIVI:    bits={bits}  group_size={group_size}  residual_length={residual_length}")
    print(f"TopK:    K={topk_k}  n_sink={topk_n_sink}  n_local={topk_n_local}")
    print(f"Tasks:   {task_list}  ({n_per_task} per task)")
    print(f"Model:   {model}")
    print(f"\nLaunching container...")

    results = run_longbench_kivi_topk.remote(
        model, bits, group_size, residual_length,
        topk_k, topk_n_sink, topk_n_local,
        task_list, n_per_task,
    )

    tag   = f"longbench_kivi_topk_{bits}bit_K{topk_k}_sink{topk_n_sink}_local{topk_n_local}"
    out_f = Path(__file__).parent / "results" / f"{tag}.json"
    out_f.parent.mkdir(parents=True, exist_ok=True)
    with open(out_f, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Local results saved to {out_f}")

    from collections import defaultdict
    task_scores = defaultdict(list)
    for r in results:
        task_scores[r["task"]].append(r["score"])

    print(f"\n── kivi_topk  bits={bits}  K={topk_k} ────────────────────────────────────")
    for task in task_list:
        vals = task_scores[task]
        avg  = sum(vals) / len(vals) if vals else float("nan")
        print(f"  {task:<22}  score={avg:.3f}")
    all_vals = [s for v in task_scores.values() for s in v]
    if all_vals:
        print(f"  {'overall':<22}  score={sum(all_vals)/len(all_vals):.3f}")
