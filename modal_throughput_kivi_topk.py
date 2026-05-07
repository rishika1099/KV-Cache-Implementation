"""
modal_throughput_kivi_topk.py

Throughput benchmark: baseline vs KIVI-4bit vs KIVITopK-4bit at long context.
Batch size 1, sweeps prefill lengths to show how the methods scale.

The key story: at long prefill lengths (8K+) the quantised KV cache in
KIVI / KIVITopK reduces HBM bandwidth during decode (4x at 4-bit vs FP16),
and TopK further reduces the effective attention scope.

Run:
    # individual methods
    modal run modal_throughput_kivi_topk.py --method baseline
    modal run modal_throughput_kivi_topk.py --method kivi
    modal run modal_throughput_kivi_topk.py --method kivi_topk

    # print comparison from saved results
    modal run modal_throughput_kivi_topk.py --compare

Output files:
    results/throughput_kivi_topk_baseline_p<prefil>_g<gen>.json
    results/throughput_kivi_topk_kivi_p<prefil>_g<gen>.json
    results/throughput_kivi_topk_kivi_topk_p<prefil>_g<gen>.json
"""

import json
import sys
import time
from pathlib import Path

import modal

app = modal.App("kv-throughput-kivi-topk")
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
    .run_commands(
        "cd /app/methods/kivi_kernels && "
        "TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' python setup.py install 2>&1 | tail -5"
    )
)

results_vol     = modal.Volume.from_name("kv-benchmark-results", create_if_missing=True)
model_cache_vol = modal.Volume.from_name("hf-model-cache",       create_if_missing=True)
HF_CACHE_PATH   = Path("/root/.cache/huggingface")
RESULTS_PATH    = Path("/results")


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def _timed_generate(model, tokenizer, input_ids, max_new_tokens):
    """
    Manual greedy decode (B=1). Works for baseline, KIVI, and KIVITopK
    because all three expose the same model() interface.
    KIVITopK returns 9-tuples; baseline returns DynamicCache; all fine as
    long as we pass past_key_values back unchanged.

    Returns: (ttft_s, decode_time_s, n_decode, peak_mem_gb)
    """
    import torch
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # ── Prefill + first token ──────────────────────────────────────────
        t0      = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_s  = time.perf_counter() - t0

        n_decode = 0
        # ── Decode loop ────────────────────────────────────────────────────
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_id:
                break
            outputs    = model(
                input_ids=next_token, past_key_values=past_kv,
                use_cache=True, return_dict=True,
            )
            past_kv    = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            n_decode  += 1
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
    timeout=7200,
    memory=65536,
)
def run_throughput(
    method: str,
    prefill_lens: list,
    max_new_tokens: int,
    n_examples: int,
    model_name: str,
    bits: int,
    group_size: int,
    residual_length: int,
    topk_K: int,
    topk_n_sink: int,
    topk_n_local: int,
) -> dict:
    _bootstrap()

    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    print(f"[modal] method={method}  bits={bits}")
    print(f"[modal] prefill_lens={prefill_lens}  max_new_tokens={max_new_tokens}")
    print(f"[modal] n_examples={n_examples} (first skipped as warmup)")
    print(f"[modal] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[modal] Total GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "kivi":
        from methods.llama_kivi_model import LlamaForCausalLM_KIVI
        config = AutoConfig.from_pretrained(model_name)
        config.k_bits          = bits
        config.v_bits          = bits
        config.group_size      = group_size
        config.residual_length = residual_length
        model = LlamaForCausalLM_KIVI.from_pretrained(
            model_name, config=config,
            torch_dtype=torch.float16, device_map="cuda",
        )
        print(f"[modal] KIVI: bits={bits}  group_size={group_size}  residual_length={residual_length}")

    elif method == "kivi_topk":
        from methods.llama_kivi_topk_model import LlamaForCausalLM_KIVITopK
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
        print(f"[modal] KIVITopK: bits={bits}  K={topk_K}  n_sink={topk_n_sink}  n_local={topk_n_local}")
        budget = topk_n_sink + topk_K + topk_n_local
        print(f"[modal] Attention budget: {budget} tokens")

    else:  # baseline
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda",
        )

    model.eval()
    print(f"[modal] Model loaded. Mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    all_records = {}
    torch.manual_seed(42)

    for prefill_len in prefill_lens:
        print(f"\n[modal] ── prefill={prefill_len} ──")

        # Warmup at this prefill length (compiles Triton kernels)
        dummy = torch.randint(
            100, tokenizer.vocab_size - 100, (1, prefill_len), device="cuda"
        )
        _timed_generate(model, tokenizer, dummy, 16)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        records = []
        oom = False
        for i in range(n_examples):
            try:
                input_ids = torch.randint(
                    100, tokenizer.vocab_size - 100,
                    (1, prefill_len), device="cuda",
                )
                ttft_s, decode_time_s, n_decode, peak_mem_gb = \
                    _timed_generate(model, tokenizer, input_ids, max_new_tokens)

                decode_tps        = n_decode / decode_time_s if decode_time_s > 0 else 0
                decode_ms_per_tok = decode_time_s / n_decode * 1000 if n_decode > 0 else 0

                records.append({
                    "ttft_ms":          ttft_s * 1000,
                    "decode_tps":       decode_tps,
                    "decode_ms_per_tok": decode_ms_per_tok,
                    "peak_mem_gb":      peak_mem_gb,
                })
                print(f"  [{i+1}/{n_examples}] ttft={ttft_s*1000:.0f}ms  "
                      f"decode={decode_tps:.1f} tok/s  "
                      f"({decode_ms_per_tok:.1f} ms/tok)  "
                      f"peak={peak_mem_gb:.1f} GB")

                del input_ids
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at prefill={prefill_len}")
                torch.cuda.empty_cache()
                oom = True
                break

        if oom:
            all_records[prefill_len] = {"oom": True}
        else:
            # Skip first example (warmup outlier)
            stable = records[1:] if len(records) > 1 else records
            n = len(stable)
            summary = {
                "mean_ttft_ms":          sum(r["ttft_ms"]           for r in stable) / n,
                "mean_decode_tps":       sum(r["decode_tps"]        for r in stable) / n,
                "mean_decode_ms_per_tok": sum(r["decode_ms_per_tok"] for r in stable) / n,
                "mean_peak_mem_gb":      sum(r["peak_mem_gb"]       for r in stable) / n,
                "n_stable":              n,
            }
            all_records[prefill_len] = summary
            print(f"  → avg  ttft={summary['mean_ttft_ms']:.0f}ms  "
                  f"decode={summary['mean_decode_tps']:.1f} tok/s  "
                  f"({summary['mean_decode_ms_per_tok']:.1f} ms/tok)  "
                  f"peak={summary['mean_peak_mem_gb']:.1f} GB  (n={n})")

    payload = {
        "method":          method,
        "bits":            bits,
        "group_size":      group_size,
        "residual_length": residual_length,
        "topk_params":     {"K": topk_K, "n_sink": topk_n_sink, "n_local": topk_n_local}
                           if method == "kivi_topk" else None,
        "max_new_tokens":  max_new_tokens,
        "model":           model_name,
        "records":         {str(k): v for k, v in all_records.items()},
    }

    # Use largest non-OOM prefill for the filename tag
    valid_prefills = [p for p in prefill_lens if not all_records.get(p, {}).get("oom")]
    tag_p = max(valid_prefills) if valid_prefills else prefill_lens[-1]
    tag = f"throughput_kivi_topk_{method}_p{tag_p}_g{max_new_tokens}"
    out = RESULTS_PATH / f"{tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    results_vol.commit()
    print(f"\n[modal] Saved → {out}")
    return payload


def _print_comparison(results_by_method):
    """Print a side-by-side comparison table across all methods and prefill lengths."""
    all_prefills = sorted(
        set(int(p) for res in results_by_method.values() for p in res["records"])
    )
    methods = list(results_by_method.keys())
    W = 100
    print(f"\n{'═'*W}")
    print(f"  KIVI + TopK Throughput Benchmark  (batch_size=1, B=1)")
    print(f"  GPU: H100 80GB")
    print(f"{'═'*W}")

    hdr = f"  {'Prefill':>8}"
    for m in methods:
        hdr += f"  {'TTFT-'+m:>14}  {'Dec-'+m:>14}"
    print(hdr)

    sep = f"  {'─'*8}"
    for _ in methods:
        sep += f"  {'─'*14}  {'─'*14}"
    print(sep)

    for p in all_prefills:
        row = f"  {p:>8}"
        for m in methods:
            r = results_by_method[m]["records"].get(str(p), {})
            if r.get("oom"):
                row += f"  {'OOM':>14}  {'OOM':>14}"
            elif not r:
                row += f"  {'—':>14}  {'—':>14}"
            else:
                row += f"  {r['mean_ttft_ms']:>14.0f}  {r['mean_decode_tps']:>14.1f}"
        print(row)
    print(f"{'═'*W}")

    # Speedup of kivi_topk vs baseline (decode tps)
    if "baseline" in results_by_method and "kivi_topk" in results_by_method:
        print(f"\n  Speedup (decode tok/s): kivi_topk vs baseline")
        for p in all_prefills:
            b = results_by_method["baseline"]["records"].get(str(p), {})
            t = results_by_method["kivi_topk"]["records"].get(str(p), {})
            if b.get("oom") or t.get("oom") or not b or not t:
                continue
            spd = t["mean_decode_tps"] / b["mean_decode_tps"]
            print(f"    prefill={p:>6}: {spd:.2f}x")


@app.local_entrypoint()
def main(
    method: str       = "baseline",
    prefill_lens: str = "4096,8192,16384,32768",
    max_new_tokens: int = 512,
    n_examples: int   = 4,
    model: str        = "meta-llama/Llama-2-7b-hf",
    bits: int         = 4,
    group_size: int   = 32,
    residual_length: int = 128,
    topk_k: int       = 1024,
    topk_n_sink: int  = 128,
    topk_n_local: int = 512,
    compare: bool     = False,
):
    """
    Throughput benchmark: baseline vs KIVI vs KIVITopK at long context.

    Examples:
        modal run modal_throughput_kivi_topk.py --method baseline
        modal run modal_throughput_kivi_topk.py --method kivi
        modal run modal_throughput_kivi_topk.py --method kivi_topk
        modal run modal_throughput_kivi_topk.py --compare
    """
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if compare:
        results_by_method = {}
        for m in ("baseline", "kivi", "kivi_topk"):
            # Find most recent file for this method
            matches = sorted(out_dir.glob(f"throughput_kivi_topk_{m}_*.json"))
            if not matches:
                print(f"  No results found for {m} — run --method {m} first.")
                continue
            with open(matches[-1]) as f:
                results_by_method[m] = json.load(f)
        if results_by_method:
            _print_comparison(results_by_method)
        return

    p_list = [int(x) for x in prefill_lens.split(",")]
    print(f"Method:  {method}")
    print(f"Prefill lengths: {p_list}  Generate: {max_new_tokens} tok")
    print(f"n_examples: {n_examples} (first per prefill skipped as warmup)")
    if method in ("kivi", "kivi_topk"):
        print(f"KIVI:    bits={bits}  group_size={group_size}  residual_length={residual_length}")
    if method == "kivi_topk":
        budget = topk_n_sink + topk_k + topk_n_local
        print(f"TopK:    K={topk_k}  n_sink={topk_n_sink}  n_local={topk_n_local}  budget={budget}")

    result = run_throughput.remote(
        method, p_list, max_new_tokens, n_examples, model,
        bits, group_size, residual_length,
        topk_k, topk_n_sink, topk_n_local,
    )

    valid_prefills = [p for p in p_list if not result["records"].get(str(p), {}).get("oom")]
    tag_p = max(valid_prefills) if valid_prefills else p_list[-1]
    tag   = f"throughput_kivi_topk_{method}_p{tag_p}_g{max_new_tokens}"
    out_f = out_dir / f"{tag}.json"
    with open(out_f, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_f}")

    print(f"\n── {method} (generate={max_new_tokens}) ──")
    print(f"  {'Prefill':>8}  {'TTFT ms':>10}  {'Decode tok/s':>14}  {'ms/tok':>8}  {'Peak GB':>9}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*14}  {'─'*8}  {'─'*9}")
    for p in p_list:
        r = result["records"].get(str(p), {})
        if r.get("oom"):
            print(f"  {p:>8}  {'OOM':>10}  {'OOM':>14}  {'OOM':>8}  {'OOM':>9}")
        elif r:
            print(f"  {p:>8}  {r['mean_ttft_ms']:>10.0f}  "
                  f"{r['mean_decode_tps']:>14.1f}  "
                  f"{r['mean_decode_ms_per_tok']:>8.1f}  "
                  f"{r['mean_peak_mem_gb']:>9.1f}")
