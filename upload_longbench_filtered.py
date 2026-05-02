#!/usr/bin/env python3
"""
Filter LongBench tasks to examples whose tokenized prompt fits within Llama-2-7b's
4096-token context window, then upload to HuggingFace as vm2825/longbench-llama2-filtered.

Usage:
    python upload_longbench_filtered.py

The uploaded dataset mirrors the THUDM/LongBench config-per-task layout so the
existing _load_longbench code (load_dataset(hf_path, task, split="test")) works
without any structural changes.
"""

import json
import os
import zipfile
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset
from huggingface_hub import hf_hub_download

HF_TOKEN  = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise EnvironmentError("Set HF_TOKEN env var before running: export HF_TOKEN=hf_...")
HF_REPO   = "vm2825/longbench-llama2-filtered"
TASKS     = ["qasper", "multifieldqa_en", "triviaqa", "2wikimqa", "multi_news", "lcc"]
MAX_TOKENS = 3600
N_PER_TASK = 20

# Must match the templates used in modal_longbench.py exactly
DATASET2PROMPT = {
    "qasper":           "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en":  "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "triviaqa":         "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "2wikimqa":         "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multi_news":       "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "lcc":              "Please complete the code given below. \n{context}Next line of code:\n",
}


def build_prompt(ex, task):
    return DATASET2PROMPT[task].format(**ex)


def load_task_from_zip(zip_path, task):
    """Read JSONL for one task from the LongBench data.zip."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"data/{task}.jsonl") as f:
            return [json.loads(line) for line in f]


def filter_task(examples, task, tokenizer):
    print(f"[{task}] Total examples: {len(examples)}")
    filtered = []
    for ex in examples:
        prompt   = build_prompt(ex, task)
        n_tokens = len(tokenizer(prompt, truncation=False)["input_ids"])
        if n_tokens < MAX_TOKENS:
            filtered.append(ex)
        if len(filtered) >= N_PER_TASK:
            break
    print(f"[{task}] Kept {len(filtered)}/{N_PER_TASK} examples with < {MAX_TOKENS} tokens")
    if len(filtered) < N_PER_TASK:
        print(f"[{task}] WARNING: only {len(filtered)} examples found — using all of them")
    return filtered


def main():
    print("Downloading THUDM/LongBench data.zip ...")
    zip_path = hf_hub_download(
        "THUDM/LongBench", "data.zip", repo_type="dataset"
    )
    print(f"Using: {zip_path}")

    print("\nLoading Llama-2-7b-hf tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        token=HF_TOKEN,
    )

    for task in TASKS:
        print(f"\n[{task}] Loading from zip ...")
        raw = load_task_from_zip(zip_path, task)

        # Ensure answers is always a list (some tasks store it differently)
        for ex in raw:
            if "answers" in ex and not isinstance(ex["answers"], list):
                ex["answers"] = [ex["answers"]]

        examples = filter_task(raw, task, tokenizer)
        if not examples:
            print(f"[{task}] ERROR: no examples found — skipping upload")
            continue

        ds = Dataset.from_list(examples)
        print(f"[{task}] Uploading {len(ds)} examples to {HF_REPO} (config={task}) ...")
        ds.push_to_hub(
            HF_REPO,
            config_name=task,
            split="test",
            token=HF_TOKEN,
            private=False,
        )
        print(f"[{task}] Done.")

    print(f"\nAll tasks uploaded to https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
