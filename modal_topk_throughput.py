"""
modal_topk_throughput.py

Throughput benchmark: baseline vs TopK (TokenSelect) at long context (32K).
Measures decode latency at batch_size=1 to demonstrate attention-scope savings.

Run:
    modal run modal_topk_throughput.py --method baseline
    modal run modal_topk_throughput.py --method topk
    modal run modal_topk_throughput.py --compare
"""

import json
import sys
import time
from pathlib import Path

import modal

app = modal.App("kv-topk-throughput")
_base = Path(__file__).parent

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel",
        add_python="3.11",
    )
    .pip_install(
        "transformers==4.44.2",
        "accelerate>=0.27.0",
        "numpy>=1.24.0",
        "triton>=2.3.0",
        "packaging",
        "ninja",
    )
    .pip_install("flash-attn>=2.5.0", extra_options="--no-build-isolation")
    .add_local_dir(str(_base / "methods"), "/app/methods", copy=True)
)

results_vol = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
HF_CACHE_PATH = Path("/root/.cache/huggingface")
RESULTS_PATH = Path("/results")


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def _timed_generate(model, tokenizer, input_ids, max_new_tokens):
    """
    Manual greedy decode. Times prefill and decode separately.
    Returns: (ttft_s, decode_time_s, n_decode, peak_mem_gb)
    """
    import torch
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # ── Prefill ───────────────────────────────────────────────
        t0 = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_s = time.perf_counter() - t0

        n_decode = 0

        # ── Decode ────────────────────────────────────────────────
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_id:
                break
            outputs = model(
                input_ids=next_token, past_key_values=past_kv,
                use_cache=True, return_dict=True,
            )
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
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
        str(RESULTS_PATH): results_vol,
    },
    timeout=7200,
    memory=65536,
)
def run_throughput(
    method: str,
    n_examples: int,
    prefill_len: int,
    max_new_tokens: int,
    model_name: str,
    topk_K: int,
    topk_n_sink: int,
    topk_n_local: int,
) -> dict:
    _bootstrap()

    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    print(f"[modal] method={method}")
    print(f"[modal] prefill={prefill_len}  max_new_tokens={max_new_tokens}")
    print(f"[modal] n_examples={n_examples}")
    print(f"[modal] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[modal] Total GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "topk":
        from methods.llama_topk_model import LlamaForCausalLM_TopK
        config = AutoConfig.from_pretrained(model_name)
        config.topk_K = topk_K
        config.topk_n_sink = topk_n_sink
        config.topk_n_local = topk_n_local
        model = LlamaForCausalLM_TopK.from_pretrained(
            model_name, config=config,
            torch_dtype=torch.float16, device_map="cuda",
        )
        print(f"[modal] TopK params: K={topk_K}, n_sink={topk_n_sink}, n_local={topk_n_local}")
        print(f"[modal] Budget: {topk_n_sink + topk_K + topk_n_local} tokens "
              f"({(topk_n_sink + topk_K + topk_n_local)/prefill_len*100:.0f}% of prefill)")
    elif method == "topk_flash":
        from methods.llama_topk_flash_model import LlamaForCausalLM_TopKFlash
        config = AutoConfig.from_pretrained(model_name)
        config.topk_K = topk_K
        config.topk_n_sink = topk_n_sink
        config.topk_n_local = topk_n_local
        model = LlamaForCausalLM_TopKFlash.from_pretrained(
            model_name, config=config,
            torch_dtype=torch.float16, device_map="cuda",
        )
        print(f"[modal] TopKFlash (paged) params: K={topk_K}, n_sink={topk_n_sink}, n_local={topk_n_local}")
        print(f"[modal] Budget: {topk_n_sink + topk_K + topk_n_local} tokens "
              f"({(topk_n_sink + topk_K + topk_n_local)/prefill_len*100:.0f}% of prefill)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda",
        )

    model.eval()
    print(f"[modal] Model loaded. Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Warmup (compile kernels, warm caches)
    print("[modal] Warming up...")
    dummy = torch.randint(100, tokenizer.vocab_size - 100,
                          (1, prefill_len), device="cuda")
    _timed_generate(model, tokenizer, dummy, 16)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[modal] Warmup done. Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Benchmark
    records = []
    torch.manual_seed(42)

    for i in range(n_examples):
        input_ids = torch.randint(
            100, tokenizer.vocab_size - 100,
            (1, prefill_len), device="cuda",
        )

        ttft_s, decode_time_s, n_decode, peak_mem_gb = \
            _timed_generate(model, tokenizer, input_ids, max_new_tokens)

        decode_ms_per_tok = (decode_time_s / n_decode * 1000) if n_decode > 0 else 0
        decode_tps = n_decode / decode_time_s if decode_time_s > 0 else 0

        records.append({
            "ttft_ms": ttft_s * 1000,
            "decode_tps": decode_tps,
            "decode_ms_per_tok": decode_ms_per_tok,
            "n_decode": n_decode,
            "peak_mem_gb": peak_mem_gb,
        })

        print(f"  [{i+1}/{n_examples}] ttft={ttft_s*1000:.0f}ms  "
              f"decode={decode_tps:.1f} tok/s  "
              f"({decode_ms_per_tok:.1f} ms/tok)  "
              f"n_decode={n_decode}  peak={peak_mem_gb:.1f} GB")

        del input_ids
        torch.cuda.empty_cache()

    # Skip first example (warmup outlier), average the rest
    stable = records[1:] if len(records) > 1 else records
    n = len(stable)
    summary = {
        "mean_ttft_ms": sum(r["ttft_ms"] for r in stable) / n,
        "mean_decode_tps": sum(r["decode_tps"] for r in stable) / n,
        "mean_decode_ms_per_tok": sum(r["decode_ms_per_tok"] for r in stable) / n,
        "mean_peak_mem_gb": sum(r["peak_mem_gb"] for r in stable) / n,
        "n_stable": n,
    }
    print(f"\n[modal] ── Summary (excluding example 1) ──")
    print(f"  TTFT:        {summary['mean_ttft_ms']:.0f} ms")
    print(f"  Decode:      {summary['mean_decode_tps']:.1f} tok/s "
          f"({summary['mean_decode_ms_per_tok']:.1f} ms/tok)")
    print(f"  Peak mem:    {summary['mean_peak_mem_gb']:.1f} GB")

    payload = {
        "method": method,
        "prefill_len": prefill_len,
        "max_new_tokens": max_new_tokens,
        "model": model_name,
        "topk_params": {"K": topk_K, "n_sink": topk_n_sink, "n_local": topk_n_local}
                       if method == "topk" else None,
        "budget_tokens": (topk_n_sink + topk_K + topk_n_local) if method == "topk" else prefill_len,
        "records": records,
        "summary": summary,
    }

    tag = f"topk_throughput_{method}_p{prefill_len}_g{max_new_tokens}"
    out = RESULTS_PATH / f"{tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    results_vol.commit()
    print(f"\n[modal] Saved → {out}")
    return payload


def _print_comparison(baseline_res, topk_res):
    """Print side-by-side comparison table."""
    b = baseline_res["summary"]
    t = topk_res["summary"]
    tp = topk_res["topk_params"]

    W = 80
    print(f"\n{'═'*W}")
    print(f"  TokenSelect Throughput Benchmark")
    print(f"  Model: {baseline_res['model']}")
    print(f"  Prefill: {baseline_res['prefill_len']} tokens  →  Generate: {baseline_res['max_new_tokens']} tokens")
    print(f"  TopK: K={tp['K']}, n_sink={tp['n_sink']}, n_local={tp['n_local']} "
          f"(budget={topk_res['budget_tokens']} = "
          f"{topk_res['budget_tokens']/baseline_res['prefill_len']*100:.0f}% of context)")
    print(f"  GPU: H100 80GB, batch_size=1")
    print(f"{'═'*W}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'TopK':>12} {'Speedup':>10}")
    print(f"  {'─'*25} {'─'*12} {'─'*12} {'─'*10}")
    print(f"  {'TTFT (ms)':<25} {b['mean_ttft_ms']:>12.0f} {t['mean_ttft_ms']:>12.0f} "
          f"{b['mean_ttft_ms']/t['mean_ttft_ms']:>9.2f}x")
    print(f"  {'Decode (tok/s)':<25} {b['mean_decode_tps']:>12.1f} {t['mean_decode_tps']:>12.1f} "
          f"{t['mean_decode_tps']/b['mean_decode_tps']:>9.2f}x")
    print(f"  {'Decode (ms/tok)':<25} {b['mean_decode_ms_per_tok']:>12.1f} {t['mean_decode_ms_per_tok']:>12.1f} "
          f"{b['mean_decode_ms_per_tok']/t['mean_decode_ms_per_tok']:>9.2f}x")
    print(f"  {'Peak memory (GB)':<25} {b['mean_peak_mem_gb']:>12.1f} {t['mean_peak_mem_gb']:>12.1f} "
          f"{'—':>10}")
    print(f"{'═'*W}")


@app.local_entrypoint()
def main(
    method: str = "baseline",
    n_examples: int = 5,
    prefill_len: int = 32768,
    max_new_tokens: int = 512,
    model: str = "meta-llama/Llama-2-7b-hf",
    topk_k: int = 4096,
    topk_n_sink: int = 128,
    topk_n_local: int = 4096,
    compare: bool = False,
):
    """
    Benchmark baseline vs TopK at long context.

    Examples:
        modal run modal_topk_throughput.py --method baseline
        modal run modal_topk_throughput.py --method topk
        modal run modal_topk_throughput.py --compare
    """
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"p{prefill_len}_g{max_new_tokens}"

    if compare:
        b_f = out_dir / f"topk_throughput_baseline_{tag}.json"
        t_f = out_dir / f"topk_throughput_topk_{tag}.json"
        if not b_f.exists() or not t_f.exists():
            print(f"Missing result files. Run baseline and topk first.")
            print(f"  Expected: {b_f}")
            print(f"  Expected: {t_f}")
            return
        with open(b_f) as f:
            b_res = json.load(f)
        with open(t_f) as f:
            t_res = json.load(f)
        _print_comparison(b_res, t_res)
        return

    print(f"Method: {method}")
    print(f"Prefill: {prefill_len} tokens, Generate: {max_new_tokens} tokens")
    print(f"Examples: {n_examples} (first skipped as warmup)")
    if method == "topk":
        budget = topk_n_sink + topk_k + topk_n_local
        print(f"TopK budget: {budget} tokens ({budget/prefill_len*100:.0f}% of context)")

    result = run_throughput.remote(
        method, n_examples, prefill_len, max_new_tokens, model,
        topk_k, topk_n_sink, topk_n_local,
    )

    out_f = out_dir / f"topk_throughput_{method}_{tag}.json"
    with open(out_f, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_f}")

    # Print single-method summary
    s = result["summary"]
    print(f"\n── {method} (prefill={prefill_len}, generate={max_new_tokens}) ──")
    print(f"  TTFT:     {s['mean_ttft_ms']:.0f} ms")
    print(f"  Decode:   {s['mean_decode_tps']:.1f} tok/s ({s['mean_decode_ms_per_tok']:.1f} ms/tok)")
    print(f"  Peak mem: {s['mean_peak_mem_gb']:.1f} GB")
