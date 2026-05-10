"""
modal_kivi_compare.py  — TEMPORARY comparison script

Matches KIVI pred_long_bench.py exactly:
  - Raw THUDM/LongBench qasper test set (all ~200 examples, no pre-filtering)
  - Middle-truncation at max_length=4096  (first 2048 + last 2048 tokens)
  - No chat template applied for Llama-2-7b-chat-hf (build_chat returns unchanged)
  - max_new_tokens=128 for qasper
  - greedy decoding (num_beams=1, do_sample=False)

Run:
    modal run modal_kivi_compare.py --method baseline
    modal run modal_kivi_compare.py --method kivi2
"""

import json
from pathlib import Path
import modal

app = modal.App("kv-kivi-compare")
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

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
MAX_LENGTH = 4096   # from KIVI config/model2maxlen.json for Llama-2-7b-chat-hf
MAX_NEW_TOKENS = 128  # from KIVI config/dataset2maxlen.json for qasper

QASPER_PROMPT = (
    "You are given a scientific article and a question. Answer the question as concisely as you can, "
    "using a single phrase or sentence if possible. If the question cannot be answered based on the "
    "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
    "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
    "Article: {context}\n\n Answer the question based on the above article as concisely as you can, "
    "using a single phrase or sentence if possible. If the question cannot be answered based on the "
    "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
    "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
    "Question: {input}\n\nAnswer:"
)


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
def run_qasper(method_name: str, method_cfg: dict, examples: list) -> list:
    import sys
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from methods.baseline   import BaselineMethod
    from methods.kivi_quant import KIVIMethod
    from benchmark.runner   import generate_with_method
    from benchmark.datasets import DatasetLoader

    print(f"[modal] Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    if method_name == "baseline":
        method = BaselineMethod()
    elif method_name == "kivi2":
        method = KIVIMethod(
            bits=method_cfg.get("bits", 2),
            residual_length=method_cfg.get("residual_length", 128),
            group_size=method_cfg.get("group_size", 32),
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    results = []
    for i, ex in enumerate(examples):
        prompt  = ex["prompt"]
        answers = ex["answers"]
        try:
            pred, _ = generate_with_method(
                model, tokenizer, method,
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                device="cuda",
            )
            score = DatasetLoader.score_longbench("qasper", [pred], [answers])
            results.append({
                "method": method_name, "config": method_cfg,
                "prompt_len_tokens": ex["prompt_len"],
                "pred": pred, "answers": answers, "score": score,
            })
            print(f"  [{i+1}/{len(examples)}] score={score:.3f}  pred={pred[:60]!r}")
        except Exception as e:
            import traceback
            print(f"  [ERROR] example {i}: {e}\n{traceback.format_exc()}")
            results.append({
                "method": method_name, "config": method_cfg,
                "prompt_len_tokens": ex["prompt_len"],
                "pred": "", "answers": answers, "score": 0.0,
            })

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"\n[modal] {method_name} qasper avg={avg:.4f}  ({len(results)} examples)")

    tag = f"kivi_compare_{method_name}"
    out = RESULTS_PATH / f"{tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    results_vol.commit()
    print(f"[modal] Saved → {out}")
    return results


@app.local_entrypoint()
def main(method: str = "baseline"):
    """
    method: "baseline" or "kivi2"
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if method not in ("baseline", "kivi2"):
        raise ValueError("method must be 'baseline' or 'kivi2'")

    method_cfg = {"bits": 2, "residual_length": 128, "group_size": 32} if method == "kivi2" else {}

    print(f"Loading tokenizer for length check ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading THUDM/LongBench qasper test (raw, no pre-filter)...")
    ds = load_dataset("vm2825/longbench-raw", "qasper", split="test")
    print(f"  Total examples: {len(ds)}")

    examples = []
    skipped  = 0
    for ex in ds:
        prompt = QASPER_PROMPT.format(**ex)

        # KIVI-style: tokenize without truncation, then middle-truncate if needed
        tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        original_len = len(tokens)

        if original_len > MAX_LENGTH:
            half   = MAX_LENGTH // 2
            prompt = (
                tokenizer.decode(tokens[:half],     skip_special_tokens=True) +
                tokenizer.decode(tokens[-half:],    skip_special_tokens=True)
            )
            tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        answers = ex.get("answers", [""])
        if not isinstance(answers, list):
            answers = [str(answers)]

        examples.append({
            "prompt":    prompt,
            "answers":   answers,
            "prompt_len": len(tokens),
            "original_len": original_len,
        })

    truncated = sum(1 for e in examples if e["original_len"] > MAX_LENGTH)
    print(f"  {len(examples)} examples  ({truncated} were middle-truncated to {MAX_LENGTH} tokens)")
    import numpy as np
    lens = [e["prompt_len"] for e in examples]
    print(f"  Prompt length: min={min(lens)}  max={max(lens)}  mean={int(np.mean(lens))}")

    print(f"\nLaunching Modal ({method})...")
    results = run_qasper.remote(method, method_cfg, examples)

    avg = sum(r["score"] for r in results) / len(results) if results else 0
    out = Path(__file__).parent / "results" / f"kivi_compare_{method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n── {method} qasper ──────────────────────────────────")
    print(f"  examples : {len(results)}")
    print(f"  avg score: {avg:.4f}")
    print(f"  saved to : {out}")
