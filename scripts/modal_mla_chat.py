"""
modal_mla_chat.py

Run a single chat prompt against the MLA latent-cache model (or baseline GQA)
on a Modal GPU container.

Examples:
    modal run modal_mla_chat.py
    modal run modal_mla_chat.py --use-mla False
    modal run modal_mla_chat.py --model-pair qwen25
    modal run modal_mla_chat.py --user-prompt "What is the capital of France?"
    modal run modal_mla_chat.py --system-prompt "You are a pirate." --user-prompt "Tell me about treasure."
    modal run modal_mla_chat.py --max-new-tokens 300
"""

import sys
from pathlib import Path

import modal

app = modal.App("kv-mla-chat")

_base = Path(__file__).parent

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

model_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
HF_CACHE_PATH   = Path("/root/.cache/huggingface")

MODEL_PATHS = {
    "llama2": {
        "base": "meta-llama/Llama-2-7b-chat-hf",
        "mla":  "botsxc/llama2-7b-chat-mla-2048-8",
    },
    "qwen25": {
        "base": "Qwen/Qwen2.5-7B",
        "mla":  "botsxc/qwen2.5-7b-instruct-mla-256",
    },
}


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    if "/app/methods" not in sys.path:
        sys.path.insert(0, "/app/methods")


def _apply_chat_template(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Llama-2 chat fallback
    if system_prompt:
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
    return f"[INST] {user_prompt} [/INST]"


@app.function(
    image=image,
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(HF_CACHE_PATH): model_cache_vol},
    timeout=600,
    memory=32768,
)
def run_mla_chat(
    model_path: str,
    use_mla: bool,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> dict:
    _bootstrap()

    import time
    import torch
    from transformers import AutoTokenizer

    device = "cuda"
    dtype  = torch.float16

    print(f"[modal] Loading {'MLA' if use_mla else 'baseline'} model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_mla:
        from transmla.transformers.llama.configuration_llamamla import LlamaMLAConfig
        from transmla.transformers.llama.modeling_llamamla_latent import LlamaMLALatentForCausalLM
        config = LlamaMLAConfig.from_pretrained(model_path)
        model  = LlamaMLALatentForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=dtype, device_map=device
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device
        )
    model.eval()

    prompt = _apply_chat_template(tokenizer, system_prompt, user_prompt)
    print(f"[modal] Prompt ({len(prompt)} chars):\n{prompt}\n")

    inputs    = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    if use_mla:
        from transmla.transformers.mla_latent import MLALatentCache
        cache = MLALatentCache()
        with torch.no_grad():
            prefill_out = model(**inputs, past_key_values=cache, use_cache=True)
    else:
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

    n_gen       = len(generated_ids)
    decode_time = t_end - t_decode_start
    response    = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if use_mla:
        kv_mb = past_kv.get_cache_bytes() / 1e6
    else:
        from benchmark.runner import _cache_to_tuple
        kv_tuples = _cache_to_tuple(past_kv)
        kv_mb = sum(k.element_size()*k.numel() + v.element_size()*v.numel() for k, v in kv_tuples) / 1e6

    metrics = {
        "input_tokens":         input_len,
        "tokens_generated":     n_gen,
        "ttft_ms":              ttft * 1000,
        "throughput_tps":       n_gen / decode_time if decode_time > 0 else 0.0,
        "per_token_latency_ms": decode_time / n_gen * 1000 if n_gen > 0 else 0.0,
        "kv_cache_mb":          kv_mb,
        "peak_memory_gb":       torch.cuda.max_memory_allocated(device) / 1e9,
    }

    print(f"[modal] Response:\n{response}")
    print(f"[modal] Metrics: {metrics}")
    return {"response": response, "metrics": metrics}


@app.local_entrypoint()
def main(
    model_pair: str = "llama2",
    use_mla: bool = True,
    system_prompt: str = "You are a helpful assistant.",
    user_prompt: str = "Explain what multi-head latent attention (MLA) is in two sentences.",
    max_new_tokens: int = 200,
):
    if model_pair not in MODEL_PATHS:
        print(f"[error] Unknown model_pair '{model_pair}'. Choose from: {list(MODEL_PATHS)}")
        return

    model_path = MODEL_PATHS[model_pair]["mla" if use_mla else "base"]

    print(f"Pair:          {model_pair}")
    print(f"Model:         {model_path}")
    print(f"MLA:           {use_mla}")
    print(f"System prompt: {system_prompt}")
    print(f"User prompt:   {user_prompt}")
    print(f"Max new tokens:{max_new_tokens}")
    print(f"\nLaunching container...")

    result = run_mla_chat.remote(
        model_path=model_path,
        use_mla=use_mla,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
    )

    print(f"\n── Response ──────────────────────────────────────────────────────────")
    print(result["response"])
    print(f"\n── Metrics ───────────────────────────────────────────────────────────")
    for k, v in result["metrics"].items():
        print(f"  {k:<25} {v:.3f}")
