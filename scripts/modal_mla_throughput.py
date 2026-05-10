"""
modal_mla_throughput.py

Throughput benchmark: baseline GQA vs MLA latent-cache, batch_size=1.
Uses random-token prompts with fixed prefill length. Hardcoded to llama2 pair.

Run both methods then compare:
    modal run modal_mla_throughput.py --method baseline
    modal run modal_mla_throughput.py --method mla
    modal run modal_mla_throughput.py --compare

Or sweep prefill lengths:
    modal run modal_mla_throughput.py --method baseline --prefill-lens 512,1024,2048
"""

import json
import sys
import time
from pathlib import Path

import modal

app = modal.App("kv-mla-throughput")
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

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
RESULTS_PATH    = Path("/results")

MODEL_BASE = "meta-llama/Llama-2-7b-chat-hf"
MODEL_MLA  = "botsxc/llama2-7b-chat-mla-2048-8"


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")
    if "/app/methods" not in sys.path:
        sys.path.insert(0, "/app/methods")


def _timed_generate_baseline(model, tokenizer, input_ids, max_new_tokens):
    import torch
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        t0      = time.perf_counter()
        out     = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_s = time.perf_counter() - t0

        n_decode = 0
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_tok.item() == eos_id:
                break
            out      = model(input_ids=next_tok, past_key_values=past_kv,
                             use_cache=True, return_dict=True)
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            n_decode += 1
        torch.cuda.synchronize()
        decode_time_s = time.perf_counter() - t1

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    # KV cache size: sum over layers of (K + V) tensors
    kv_bytes = sum(
        k.element_size() * k.numel() + v.element_size() * v.numel()
        for k, v in past_kv
    )
    return ttft_s, decode_time_s, n_decode, peak_mem_gb, kv_bytes / 1e6


def _timed_generate_mla(model, tokenizer, input_ids, max_new_tokens):
    import torch
    from transmla.transformers.mla_latent import MLALatentCache

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        t0    = time.perf_counter()
        cache = MLALatentCache()
        out   = model(input_ids=input_ids, past_key_values=cache,
                      use_cache=True, return_dict=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_s = time.perf_counter() - t0

        n_decode = 0
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_tok.item() == eos_id:
                break
            out      = model(input_ids=next_tok, past_key_values=past_kv,
                             use_cache=True, return_dict=True)
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            n_decode += 1
        torch.cuda.synchronize()
        decode_time_s = time.perf_counter() - t1

    peak_mem_gb  = torch.cuda.max_memory_allocated() / 1e9
    kv_cache_mb  = past_kv.get_cache_bytes() / 1e6
    return ttft_s, decode_time_s, n_decode, peak_mem_gb, kv_cache_mb


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
def run_throughput(
    method: str,
    prefill_lens: list,
    max_new_tokens: int,
    n_batches: int,
) -> dict:
    _bootstrap()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[modal] method={method}")
    print(f"[modal] prefill_lens={prefill_lens}  max_new_tokens={max_new_tokens}  n_batches={n_batches}")
    print(f"[modal] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[modal] Total GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    if method == "baseline":
        model_path  = MODEL_BASE
        generate_fn = _timed_generate_baseline
        tokenizer   = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="cuda",
        )
    elif method == "mla":
        from transmla.transformers.llama.configuration_llamamla import LlamaMLAConfig
        from transmla.transformers.llama.modeling_llamamla_latent import LlamaMLALatentForCausalLM

        model_path  = MODEL_MLA
        generate_fn = _timed_generate_mla
        tokenizer   = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = LlamaMLAConfig.from_pretrained(model_path)
        model  = LlamaMLALatentForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.float16, device_map="cuda",
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'baseline' or 'mla'.")

    model.eval()
    print(f"[modal] Model loaded. Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Warmup at first prefill length
    print("[modal] Warming up...")
    dummy = torch.randint(100, tokenizer.vocab_size - 100, (1, prefill_lens[0]), device="cuda")
    generate_fn(model, tokenizer, dummy, 32)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[modal] Warmup done. Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    all_results = {}
    torch.manual_seed(42)

    for prefill_len in prefill_lens:
        print(f"\n[modal] ── prefill_len={prefill_len} ──")
        records = []

        for b_idx in range(n_batches):
            input_ids = torch.randint(
                100, tokenizer.vocab_size - 100, (1, prefill_len), device="cuda",
            )
            ttft_s, decode_time_s, n_decode, peak_mem_gb, kv_mb = \
                generate_fn(model, tokenizer, input_ids, max_new_tokens)

            total_time  = ttft_s + decode_time_s
            total_toks  = 1 + n_decode
            total_tps   = total_toks / total_time
            decode_tps  = n_decode / decode_time_s if decode_time_s > 0 else 0.0
            ttft_ms     = ttft_s * 1000

            records.append({
                "ttft_ms":      ttft_ms,
                "decode_tps":   decode_tps,
                "total_tps":    total_tps,
                "peak_mem_gb":  peak_mem_gb,
                "kv_cache_mb":  kv_mb,
            })
            flag = " [warmup]" if b_idx == 0 else ""
            print(f"  [{b_idx+1}/{n_batches}] ttft={ttft_ms:.0f}ms  "
                  f"decode={decode_tps:.1f} tok/s  total={total_tps:.1f} tok/s  "
                  f"peak={peak_mem_gb:.2f} GB  kv={kv_mb:.1f} MB{flag}")

            del input_ids
            torch.cuda.empty_cache()

        stable = records[1:] if len(records) > 1 else records
        n = len(stable)
        agg = {
            "mean_ttft_ms":     sum(r["ttft_ms"]     for r in stable) / n,
            "mean_decode_tps":  sum(r["decode_tps"]  for r in stable) / n,
            "mean_total_tps":   sum(r["total_tps"]   for r in stable) / n,
            "mean_peak_mem_gb": sum(r["peak_mem_gb"] for r in stable) / n,
            "mean_kv_cache_mb": sum(r["kv_cache_mb"] for r in stable) / n,
            "n_stable": n,
        }
        all_results[prefill_len] = agg
        print(f"  → avg  ttft={agg['mean_ttft_ms']:.0f}ms  "
              f"decode={agg['mean_decode_tps']:.1f} tok/s  "
              f"total={agg['mean_total_tps']:.1f} tok/s  "
              f"peak={agg['mean_peak_mem_gb']:.2f} GB  "
              f"kv={agg['mean_kv_cache_mb']:.1f} MB  (n={n})")

    payload = {
        "method":         method,
        "max_new_tokens": max_new_tokens,
        "model":          model_path,
        "results":        {str(k): v for k, v in all_results.items()},
    }

    out = RESULTS_PATH / f"mla_throughput_{method}_g{max_new_tokens}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    results_vol.commit()
    print(f"\n[modal] Saved → {out}")
    return payload


def _print_table(base_res, mla_res):
    all_prefills = sorted(
        set(int(k) for k in base_res["results"]) |
        set(int(k) for k in mla_res["results"])
    )
    W = 120
    print(f"\n{'─'*W}")
    print(f"  generate={base_res['max_new_tokens']} tok  batch_size=1")
    print(f"  baseline: {base_res['model']}")
    print(f"  mla:      {mla_res['model']}")
    print(f"{'─'*W}")
    print(f"  {'Prefill':>8}  "
          f"{'TTFT-Base':>10}  {'TTFT-MLA':>10}  "
          f"{'Dec-Base':>10}  {'Dec-MLA':>10}  {'Dec Spd':>9}  "
          f"{'KV-Base MB':>11}  {'KV-MLA MB':>11}  {'KV Ratio':>9}  "
          f"{'Mem-Base GB':>12}  {'Mem-MLA GB':>11}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*10}  {'─'*9}  "
          f"{'─'*11}  {'─'*11}  {'─'*9}  "
          f"{'─'*12}  {'─'*11}")

    for p in all_prefills:
        b = base_res["results"].get(str(p), {})
        m = mla_res["results"].get(str(p), {})

        def f(r, key, fmt=".0f"):
            return format(r[key], fmt) if r else "—"

        dec_spd = (
            f"{m['mean_decode_tps'] / b['mean_decode_tps']:.2f}x"
            if b and m else "—"
        )
        kv_ratio = (
            f"{b['mean_kv_cache_mb'] / m['mean_kv_cache_mb']:.1f}x"
            if b and m and m['mean_kv_cache_mb'] > 0 else "—"
        )

        print(f"  {p:>8}  "
              f"{f(b,'mean_ttft_ms'):>10}  {f(m,'mean_ttft_ms'):>10}  "
              f"{f(b,'mean_decode_tps'):>10}  {f(m,'mean_decode_tps'):>10}  {dec_spd:>9}  "
              f"{f(b,'mean_kv_cache_mb','.1f'):>11}  {f(m,'mean_kv_cache_mb','.1f'):>11}  {kv_ratio:>9}  "
              f"{f(b,'mean_peak_mem_gb','.2f'):>12}  {f(m,'mean_peak_mem_gb','.2f'):>11}")
    print(f"{'─'*W}")


@app.local_entrypoint()
def main(
    method: str       = "baseline",
    prefill_lens: str = "512,1024,2048",
    max_new_tokens: int = 256,
    n_batches: int    = 4,
    compare: bool     = False,
):
    """
    Run one method or compare saved results.

    Examples:
        modal run modal_mla_throughput.py --method baseline
        modal run modal_mla_throughput.py --method mla
        modal run modal_mla_throughput.py --compare
        modal run modal_mla_throughput.py --method baseline --prefill-lens 512,1024,2048,4096
    """
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if compare:
        b_f = out_dir / f"mla_throughput_baseline_g{max_new_tokens}.json"
        m_f = out_dir / f"mla_throughput_mla_g{max_new_tokens}.json"
        if not b_f.exists() or not m_f.exists():
            print("Missing result files. Run baseline and mla first.")
            print(f"  Expected: {b_f}")
            print(f"  Expected: {m_f}")
            return
        with open(b_f) as f: b_res = json.load(f)
        with open(m_f) as f: m_res = json.load(f)
        _print_table(b_res, m_res)
        return

    pl_list = [int(x) for x in prefill_lens.split(",")]
    print(f"Method: {method}")
    print(f"Prefill lens: {pl_list}  Generate: {max_new_tokens} tok  Batches/len: {n_batches}")

    result = run_throughput.remote(method, pl_list, max_new_tokens, n_batches)

    out_f = out_dir / f"mla_throughput_{method}_g{max_new_tokens}.json"
    with open(out_f, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_f}")

    print(f"\n── {method} (generate={max_new_tokens}) ──")
    print(f"  {'Prefill':>8}  {'TTFT ms':>9}  {'Decode tok/s':>14}  {'KV MB':>8}  {'Peak GB':>9}")
    print(f"  {'─'*8}  {'─'*9}  {'─'*14}  {'─'*8}  {'─'*9}")
    for p, r in result["results"].items():
        print(f"  {int(p):>8}  {r['mean_ttft_ms']:>9.0f}  "
              f"{r['mean_decode_tps']:>14.1f}  "
              f"{r['mean_kv_cache_mb']:>8.1f}  "
              f"{r['mean_peak_mem_gb']:>9.2f}")
