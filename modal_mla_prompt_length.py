"""
modal_mla_prompt_length.py

Summarize RESULTS.md, RESULTS_MLA.md, and TOPK_RESULTS.md with baseline GQA
and MLA latent-cache, recording latency and KV cache metrics for each doc.

Examples:
    modal run modal_mla_prompt_length.py --run-baseline --run-mla
    modal run modal_mla_prompt_length.py --run-mla
    modal run modal_mla_prompt_length.py --run-baseline --run-mla --model-pair qwen25
"""

import json
import sys
from pathlib import Path

import modal

app = modal.App("kv-mla-prompt-length")

_base = Path(__file__).parent

def _load_docs(base: Path) -> dict:
    return {
        "RESULTS":     (base / "results" / "RESULTS.md").read_text(),
        "RESULTS_MLA": (base / "results" / "RESULTS_MLA.md").read_text(),
        "TOPK":        (base / "results" / "TOPK_RESULTS.md").read_text(),
    }

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.53.1",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
    )
    .add_local_dir(str(_base / "methods"), "/app/methods")
)

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
RESULTS_PATH    = Path("/results")

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
}


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    if "/app/methods" not in sys.path:
        sys.path.insert(0, "/app/methods")


MAX_LENGTH = 4096


def _apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if system_prompt:
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
    return f"[INST] {user_prompt} [/INST]"


def _truncate_prompt(tokenizer, prompt: str) -> str:
    tokens = tokenizer(prompt, truncation=False).input_ids
    if len(tokens) <= MAX_LENGTH:
        return prompt
    half = MAX_LENGTH // 2
    return (
        tokenizer.decode(tokens[:half], skip_special_tokens=True)
        + tokenizer.decode(tokens[-half:], skip_special_tokens=True)
    )


def _generate_baseline(model, tokenizer, prompt, max_new_tokens, device):
    import time
    import torch

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

    next_token    = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
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

    kv_bytes = sum(
        k.element_size() * k.numel() + v.element_size() * v.numel()
        for k, v in past_kv
    )
    n_gen       = len(generated_ids)
    decode_time = t_end - t_decode_start
    return tokenizer.decode(generated_ids, skip_special_tokens=True), {
        "input_tokens":         input_len,
        "tokens_generated":     n_gen,
        "ttft_ms":              ttft * 1000,
        "throughput_tps":       n_gen / decode_time if decode_time > 0 else 0.0,
        "per_token_latency_ms": decode_time / n_gen * 1000 if n_gen > 0 else 0.0,
        "kv_cache_mb":          kv_bytes / 1e6,
        "peak_memory_gb":       torch.cuda.max_memory_allocated(device) / 1e9,
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

    next_token    = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
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
        "input_tokens":         input_len,
        "tokens_generated":     n_gen,
        "ttft_ms":              ttft * 1000,
        "throughput_tps":       n_gen / decode_time if decode_time > 0 else 0.0,
        "per_token_latency_ms": decode_time / n_gen * 1000 if n_gen > 0 else 0.0,
        "kv_cache_mb":          past_kv.get_cache_bytes() / 1e6,
        "peak_memory_gb":       torch.cuda.max_memory_allocated(device) / 1e9,
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


def _run_phase(model, tokenizer, generate_fn, label, docs, max_new_tokens, system_prompt):
    results = []
    for doc_name, doc_text in docs.items():
        user_msg = _truncate_prompt(tokenizer, f"Please summarize the following text:\n\n{doc_text}")
        prompt = _apply_chat_template(tokenizer, system_prompt, user_msg)
        try:
            response, metrics = generate_fn(model, tokenizer, prompt, max_new_tokens, "cuda")
            record = {"label": label, "doc": doc_name, "response": response, **metrics}
            print(
                f"  {label:35s} | {doc_name:12s} | tokens={metrics['input_tokens']:4d} "
                f"| kv={metrics['kv_cache_mb']:.1f}MB | ttft={metrics['ttft_ms']:.0f}ms"
            )
        except Exception as e:
            import traceback
            print(f"  [ERROR] {label} {doc_name}: {e}\n{traceback.format_exc()}")
            record = {"label": label, "doc": doc_name, "response": ""}
        results.append(record)
    return results


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
def run_summarization(
    pair: str,
    model_paths: dict,
    docs: dict,
    max_new_tokens: int,
    run_baseline: bool,
    run_mla: bool,
    system_prompt: str = "You are a helpful assistant.",
) -> dict:
    _bootstrap()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    if run_baseline:
        print(f"\n[modal] Loading base model: {model_paths['base']}")
        tokenizer = AutoTokenizer.from_pretrained(model_paths["base"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_paths["base"], torch_dtype=torch.float16, device_map="cuda"
        )
        model.eval()
        results["baseline"] = _run_phase(model, tokenizer, _generate_baseline, model_paths["label_base"], docs, max_new_tokens, system_prompt)
        del model
        torch.cuda.empty_cache()
        print("[modal] Base model unloaded.")

    if run_mla:
        print(f"\n[modal] Loading MLA latent model: {model_paths['mla']}")
        model, tokenizer = _load_mla_latent_model(model_paths["mla"], "cuda", torch.float16)
        results["mla_latent"] = _run_phase(model, tokenizer, _generate_mla_latent, model_paths["label_mla"], docs, max_new_tokens, system_prompt)
        del model
        torch.cuda.empty_cache()
        print("[modal] MLA model unloaded.")

    variants = "_".join(k for k in ("baseline", "mla_latent") if k in results)
    out = RESULTS_PATH / f"mla_summarization_{pair}_{variants}.json"
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
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 200,
):
    if not run_baseline and not run_mla:
        print("[warning] Neither --run-baseline nor --run-mla was set. Nothing to do.")
        return

    if model_pair not in MODEL_PATHS:
        print(f"[error] Unknown model_pair '{model_pair}'. Choose from: {list(MODEL_PATHS)}")
        return

    docs = _load_docs(_base)
    paths = MODEL_PATHS[model_pair]
    print(f"Pair:  {model_pair}")
    print(f"Docs:  {list(docs.keys())}")
    print(f"\nLaunching container...")

    results = run_summarization.remote(
        pair=model_pair,
        model_paths=paths,
        docs=docs,
        max_new_tokens=max_new_tokens,
        run_baseline=run_baseline,
        run_mla=run_mla,
        system_prompt=system_prompt,
    )

    variants = "_".join(k for k in ("baseline", "mla_latent") if k in results)
    out = _base / "results" / f"mla_summarization_{model_pair}_{variants}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLocal results saved to {out}")

    for variant, records in results.items():
        label = paths["label_base"] if variant == "baseline" else paths["label_mla"]
        print(f"\n── {label} ({'baseline' if variant == 'baseline' else 'MLA latent'}) ─────────────────")
        print(f"  {'doc':12s}  {'tokens':>6}  {'kv_mb':>8}  {'ttft_ms':>9}  {'tps':>7}")
        for r in records:
            print(
                f"  {r['doc']:12s}  {r.get('input_tokens', 0):>6}  "
                f"{r.get('kv_cache_mb', 0):>8.1f}  {r.get('ttft_ms', 0):>9.1f}  "
                f"{r.get('throughput_tps', 0):>7.2f}"
            )
