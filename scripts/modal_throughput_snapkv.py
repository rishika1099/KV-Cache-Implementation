"""
modal_throughput_snapkv.py

Throughput benchmark for SnapKV at batch size 1 — mirrors
modal_throughput_topk.py structure.  SnapKV evicts KV entries once after
prefill (no per-step logic), so this measures:
  - TTFT        : prefill + attention-weight materialisation + eviction + 1st token
  - Decode tps  : tokens/s over decode steps 2..max_new_tokens (constant-size cache)
  - Total tps   : end-to-end tokens/s
  - Peak GB     : torch.cuda.max_memory_allocated() across the call

NOTE: output_attentions=True is required for SnapKV (it needs the full
attention weight matrix to vote on prefix importance).  This materialises an
extra (batch, heads, seq, seq) tensor per layer during prefill — expect higher
TTFT and peak memory than baseline at the same prefill length.

Examples:
    modal run modal_throughput_snapkv.py
    modal run modal_throughput_snapkv.py --prefill-len 4096 --budget-ratio 0.4
    modal run modal_throughput_snapkv.py --prefill-len 1024 --budget-ratio 0.5

Output:
    results/throughput_snapkv_r<ratio>_p<prefill>_g<gen>.json   (local)
    /results/throughput_snapkv_r<ratio>_p<prefill>_g<gen>.json  (Modal volume)
"""

import json
import sys
import time
from pathlib import Path

import modal

app = modal.App("kv-throughput-snapkv")
_base = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
    )
    .add_local_dir(str(_base / "methods"),   "/app/methods")
    .add_local_dir(str(_base / "benchmark"), "/app/benchmark")
)

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
RESULTS_PATH    = Path("/results")


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def _cache_to_tuple(past_kv):
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            return tuple(
                (past_kv.key_cache[i], past_kv.value_cache[i])
                for i in range(len(past_kv.key_cache))
            )
    except ImportError:
        pass
    return past_kv


def _tuple_to_cache(kv_tuple):
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(kv_tuple, tuple) and len(kv_tuple) > 0:
            cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(kv_tuple):
                cache.update(k, v, layer_idx)
            return cache
    except ImportError:
        pass
    return kv_tuple


def _timed_generate_snapkv(model, tokenizer, method, input_ids, max_new_tokens):
    """
    Greedy decode with SnapKV hooks.  B=1 only.
    Returns (ttft_s, decode_time_s, n_decode, peak_mem_gb).
    """
    import torch
    assert input_ids.shape[0] == 1, "SnapKV throughput bench is B=1 only"

    method.reset()
    device = input_ids.device
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    pad_id = tokenizer.pad_token_id  if tokenizer.pad_token_id  is not None else 0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # ── Prefill + eviction + first token ──────────────────────────────
        t0 = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            output_attentions=True,   # SnapKV needs full attn weights
        )
        past_kv_tuple = _cache_to_tuple(outputs.past_key_values)
        attn_weights  = outputs.attentions          # tuple of (1, heads, seq, seq)
        past_kv_tuple = method.process_prefill(past_kv_tuple, attention_weights=attn_weights)
        past_kv       = _tuple_to_cache(past_kv_tuple)
        next_token    = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_s = time.perf_counter() - t0

        finished = (next_token.squeeze(-1) == eos_id)
        n_decode = 0

        # ── Decode loop (frozen, constant-size KV cache) ──────────────────
        t1 = time.perf_counter()
        for step in range(max_new_tokens - 1):
            if finished.all():
                break
            fwd_kwargs = dict(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
            )
            # RoPE correction: after eviction the cache is shorter than the
            # original prompt so we must supply the true absolute position.
            true_len = getattr(method, "true_seq_length", None)
            if true_len is not None:
                fwd_kwargs["position_ids"] = torch.tensor(
                    [[true_len]], device=device,
                )
            outputs = model(**fwd_kwargs)
            step_kv_tuple = _cache_to_tuple(outputs.past_key_values)
            step_kv_tuple = method.process_step(step_kv_tuple, step=step)
            past_kv       = _tuple_to_cache(step_kv_tuple)
            next_token    = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            finished      = finished | (next_token.squeeze(-1) == eos_id)
            next_token[finished] = pad_id
            n_decode += 1
        torch.cuda.synchronize()
        decode_time_s = time.perf_counter() - t1

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return ttft_s, decode_time_s, n_decode, peak_mem_gb


@app.function(
    image=image,
    gpu="H100",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        str(HF_CACHE_PATH): model_cache_vol,
        str(RESULTS_PATH):  results_vol,
    },
    timeout=3600,
    memory=65536,
)
def run_throughput_snapkv(
    budget_ratio: float,
    observation_window: int,
    kernel_size: int,
    sink_size: int,
    n_batches: int,
    prefill_len: int,
    max_new_tokens: int,
    model_name: str,
) -> dict:
    _bootstrap()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from methods.snapkv_eviction import SnapKVMethod

    print(f"[modal] method=snapkv  budget_ratio={budget_ratio}  "
          f"obs_window={observation_window}  kernel_size={kernel_size}  "
          f"sink_size={sink_size}")
    print(f"[modal] prefill={prefill_len}  max_new_tokens={max_new_tokens}")
    print(f"[modal] batch_size=1  n_batches={n_batches}")
    print(f"[modal] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[modal] Total GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    print(f"[modal] Model loaded. Mem used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    method = SnapKVMethod(
        budget_ratio=budget_ratio,
        observation_window=observation_window,
        kernel_size=kernel_size,
        sink_size=sink_size,
    )

    # Warmup
    print("[modal] Warming up...")
    dummy = torch.randint(100, tokenizer.vocab_size - 100,
                          (1, prefill_len), device="cuda")
    _timed_generate_snapkv(model, tokenizer, method, dummy, 32)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[modal] Warmup done. Mem used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    torch.manual_seed(42)
    batch_records = []

    print(f"\n[modal] ── batch_size=1  ({n_batches} batches) ──")
    for b_idx in range(n_batches):
        input_ids = torch.randint(
            100, tokenizer.vocab_size - 100,
            (1, prefill_len), device="cuda",
        )

        ttft_s, decode_time_s, n_decode, peak_mem_gb = \
            _timed_generate_snapkv(model, tokenizer, method, input_ids, max_new_tokens)

        total_time = ttft_s + decode_time_s
        total_toks = 1 * (1 + n_decode)
        total_tps  = total_toks / total_time
        decode_tps = n_decode / decode_time_s if decode_time_s > 0 else 0
        ttft_ms    = ttft_s * 1000

        batch_records.append({
            "ttft_ms":     ttft_ms,
            "decode_tps":  decode_tps,
            "total_tps":   total_tps,
            "peak_mem_gb": peak_mem_gb,
        })
        flag = " [skip-warmup]" if b_idx == 0 else ""
        print(f"  [{b_idx+1}/{n_batches}] ttft={ttft_ms:.0f}ms  "
              f"decode={decode_tps:.0f} tok/s  total={total_tps:.0f} tok/s  "
              f"peak={peak_mem_gb:.1f} GB{flag}")

        del input_ids
        torch.cuda.empty_cache()

    stable = batch_records[1:] if len(batch_records) > 1 else batch_records
    n = len(stable)
    summary = {
        "mean_ttft_ms":     sum(r["ttft_ms"]     for r in stable) / n,
        "mean_decode_tps":  sum(r["decode_tps"]  for r in stable) / n,
        "mean_total_tps":   sum(r["total_tps"]   for r in stable) / n,
        "mean_peak_mem_gb": sum(r["peak_mem_gb"] for r in stable) / n,
        "n_stable":         n,
    }
    print(f"  → avg  ttft={summary['mean_ttft_ms']:.0f}ms  "
          f"decode={summary['mean_decode_tps']:.0f} tok/s  "
          f"total={summary['mean_total_tps']:.0f} tok/s  "
          f"peak={summary['mean_peak_mem_gb']:.1f} GB  (n={n})")

    payload = {
        "method":             "snapkv",
        "budget_ratio":       budget_ratio,
        "observation_window": observation_window,
        "kernel_size":        kernel_size,
        "sink_size":          sink_size,
        "batch_size":         1,
        "prefill_len":        prefill_len,
        "max_new_tokens":     max_new_tokens,
        "model":              model_name,
        "summary":            summary,
        "raw_batches":        batch_records,
    }

    ratio_tag = str(budget_ratio).replace(".", "")
    tag = f"throughput_snapkv_r{ratio_tag}_p{prefill_len}_g{max_new_tokens}"
    out = RESULTS_PATH / f"{tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    results_vol.commit()
    print(f"\n[modal] Saved → {out}")
    return payload


@app.local_entrypoint()
def main(
    budget_ratio: float  = 0.4,
    observation_window: int = 32,
    kernel_size: int     = 7,
    sink_size: int       = 0,
    n_batches: int       = 3,
    prefill_len: int     = 4096,
    max_new_tokens: int  = 512,
    model: str           = "meta-llama/Llama-2-7b-chat-hf",
):
    """
    SnapKV throughput benchmark at batch size 1.

    Default parameters match the paper's LongBench settings.
    """
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method:           snapkv (B=1)")
    print(f"budget_ratio:     {budget_ratio}")
    print(f"obs_window:       {observation_window}  kernel_size={kernel_size}  sink_size={sink_size}")
    print(f"Prefill:          {prefill_len} tok    Generate: {max_new_tokens} tok")
    print(f"n_batches:        {n_batches}")

    result = run_throughput_snapkv.remote(
        budget_ratio, observation_window, kernel_size, sink_size,
        n_batches, prefill_len, max_new_tokens, model,
    )

    ratio_tag = str(budget_ratio).replace(".", "")
    tag   = f"throughput_snapkv_r{ratio_tag}_p{prefill_len}_g{max_new_tokens}"
    out_f = out_dir / f"{tag}.json"
    with open(out_f, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_f}")

    s = result["summary"]
    print(f"\n── snapkv budget_ratio={budget_ratio} prefill={prefill_len} gen={max_new_tokens} (B=1) ──")
    print(f"  TTFT          {s['mean_ttft_ms']:>8.0f} ms")
    print(f"  Decode tps    {s['mean_decode_tps']:>8.0f} tok/s")
    print(f"  Total tps     {s['mean_total_tps']:>8.0f} tok/s")
    print(f"  Peak mem      {s['mean_peak_mem_gb']:>8.1f} GB")
    print(f"  (over n={s['n_stable']} batches, first batch dropped as warmup)")
