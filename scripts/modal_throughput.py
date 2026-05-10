"""
modal_throughput.py

Throughput benchmark: baseline vs KIVI-2bit across batch sizes.
Uses artificial random-token prompts with fixed prefill length.

Run one method at a time, results saved to volume + locally:
    modal run modal_throughput.py --method baseline
    modal run modal_throughput.py --method kivi

Then compare:
    modal run modal_throughput.py --compare
"""

import json
import sys
import time
from pathlib import Path

import modal

app = modal.App("kv-throughput")
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
    )
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
    Unified greedy decode loop for both baseline and KIVI models.
    Times prefill (TTFT) and decode phases separately.

    Returns:
        ttft_s       - time from start to first token (prefill, seconds)
        decode_time_s - time for remaining max_new_tokens-1 decode steps
        n_decode     - actual decode steps completed
        peak_mem_gb  - peak GPU memory during the full generation
    """
    import torch
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    pad_id = tokenizer.pad_token_id  if tokenizer.pad_token_id  is not None else 0
    BS     = input_ids.shape[0]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # ── Prefill + first token ──────────────────────────────────────────
        t0 = time.perf_counter()
        outputs    = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_kv    = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [BS,1]
        torch.cuda.synchronize()
        ttft_s = time.perf_counter() - t0

        finished = (next_token.squeeze(-1) == eos_id)
        n_decode = 0

        # ── Decode loop ────────────────────────────────────────────────────
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if finished.all():
                break
            outputs    = model(input_ids=next_token, past_key_values=past_kv,
                               use_cache=True, return_dict=True)
            past_kv    = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            finished   = finished | (next_token.squeeze(-1) == eos_id)
            next_token[finished] = pad_id
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
    bits: int,
    batch_sizes: list,
    n_batches: int,
    prefill_len: int,
    max_new_tokens: int,
    model_name: str,
) -> dict:
    _bootstrap()

    import torch
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    print(f"[modal] method={method}  bits={bits}")
    print(f"[modal] prefill={prefill_len}  max_new_tokens={max_new_tokens}")
    print(f"[modal] batch_sizes={batch_sizes}  n_batches={n_batches}")
    print(f"[modal] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[modal] Total GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if method == "kivi":
        from methods.llama_kivi_model import LlamaForCausalLM_KIVI
        config = AutoConfig.from_pretrained(model_name)
        config.k_bits         = bits
        config.v_bits         = bits
        config.group_size     = 32
        config.residual_length = 128
        model = LlamaForCausalLM_KIVI.from_pretrained(
            model_name, config=config,
            torch_dtype=torch.float16, device_map="cuda",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda",
        )
    model.eval()
    print(f"[modal] Model loaded. "
          f"Mem used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Warmup — compile Triton kernels at actual prefill length
    print("[modal] Warming up (compiles Triton kernels)...")
    dummy = torch.randint(100, tokenizer.vocab_size - 100,
                          (1, prefill_len), device="cuda")
    _timed_generate(model, tokenizer, dummy, 32)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[modal] Warmup done. "
          f"Mem used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    results = {}
    torch.manual_seed(42)

    for bs in batch_sizes:
        print(f"\n[modal] ── batch_size={bs} "
              f"(free mem: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB) ──")
        batch_records = []
        oom = False

        for b_idx in range(n_batches):
            try:
                input_ids = torch.randint(
                    100, tokenizer.vocab_size - 100,
                    (bs, prefill_len), device="cuda",
                )

                ttft_s, decode_time_s, n_decode, peak_mem_gb = \
                    _timed_generate(model, tokenizer, input_ids, max_new_tokens)

                total_time    = ttft_s + decode_time_s
                total_toks    = bs * (1 + n_decode)
                total_tps     = total_toks / total_time
                decode_tps    = (bs * n_decode) / decode_time_s if decode_time_s > 0 else 0
                ttft_ms       = ttft_s * 1000

                batch_records.append({
                    "ttft_ms":      ttft_ms,
                    "decode_tps":   decode_tps,
                    "total_tps":    total_tps,
                    "peak_mem_gb":  peak_mem_gb,
                })
                flag = " [skip-warmup]" if b_idx == 0 else ""
                print(f"  [{b_idx+1}/{n_batches}] ttft={ttft_ms:.0f}ms  "
                      f"decode={decode_tps:.0f} tok/s  total={total_tps:.0f} tok/s  "
                      f"peak={peak_mem_gb:.1f} GB{flag}")

                del input_ids
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at batch_size={bs}")
                torch.cuda.empty_cache()
                oom = True
                break

        if oom:
            results[bs] = {"oom": True}
            break   # larger batch sizes will also OOM
        else:
            # Skip first batch (first-batch warmup / outlier effect)
            stable = batch_records[1:] if len(batch_records) > 1 else batch_records
            n = len(stable)
            results[bs] = {
                "oom":             False,
                "mean_ttft_ms":    sum(r["ttft_ms"]     for r in stable) / n,
                "mean_decode_tps": sum(r["decode_tps"]  for r in stable) / n,
                "mean_total_tps":  sum(r["total_tps"]   for r in stable) / n,
                "mean_peak_mem_gb":sum(r["peak_mem_gb"] for r in stable) / n,
                "n_stable":        n,
            }
            r = results[bs]
            print(f"  → avg  ttft={r['mean_ttft_ms']:.0f}ms  "
                  f"decode={r['mean_decode_tps']:.0f} tok/s  "
                  f"total={r['mean_total_tps']:.0f} tok/s  "
                  f"peak={r['mean_peak_mem_gb']:.1f} GB  (n={n})")

    payload = {
        "method":         method,
        "bits":           bits,
        "prefill_len":    prefill_len,
        "max_new_tokens": max_new_tokens,
        "model":          model_name,
        "results":        {str(k): v for k, v in results.items()},
    }

    tag = f"throughput_{method}_p{prefill_len}_g{max_new_tokens}"
    out = RESULTS_PATH / f"{tag}.json"
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    results_vol.commit()
    print(f"\n[modal] Saved → {out}")
    return payload


def _fmt(r, key, fmt=".0f"):
    if r.get("oom"):
        return "OOM"
    v = r.get(key, 0)
    return format(v, fmt)


def _print_table(baseline_res, kivi_res):
    """Print a comparison table from two result dicts."""
    all_bs = sorted(
        set(int(k) for k in baseline_res["results"]) |
        set(int(k) for k in kivi_res["results"])
    )
    W = 130
    print(f"\n{'─'*W}")
    print(f"  prefill={baseline_res['prefill_len']} tok  "
          f"generate={baseline_res['max_new_tokens']} tok  "
          f"model={baseline_res['model']}")
    print(f"{'─'*W}")
    print(f"  {'BS':>5}  "
          f"{'TTFT-Base ms':>14}  {'TTFT-KIVI ms':>14}  "
          f"{'Dec-Base tok/s':>16}  {'Dec-KIVI tok/s':>16}  {'Dec Speedup':>12}  "
          f"{'Tot-Base tok/s':>16}  {'Tot-KIVI tok/s':>16}  {'Tot Speedup':>12}  "
          f"{'Mem-Base GB':>12}  {'Mem-KIVI GB':>12}")
    print(f"  {'─'*5}  {'─'*14}  {'─'*14}  {'─'*16}  {'─'*16}  {'─'*12}  "
          f"{'─'*16}  {'─'*16}  {'─'*12}  {'─'*12}  {'─'*12}")
    for bs in all_bs:
        b = baseline_res["results"].get(str(bs), {})
        k = kivi_res["results"].get(str(bs), {})

        b_ttft  = _fmt(b, "mean_ttft_ms",    ".0f")
        k_ttft  = _fmt(k, "mean_ttft_ms",    ".0f")
        b_dtps  = _fmt(b, "mean_decode_tps", ".0f")
        k_dtps  = _fmt(k, "mean_decode_tps", ".0f")
        b_ttps  = _fmt(b, "mean_total_tps",  ".0f")
        k_ttps  = _fmt(k, "mean_total_tps",  ".0f")
        b_mem   = _fmt(b, "mean_peak_mem_gb",".1f")
        k_mem   = _fmt(k, "mean_peak_mem_gb",".1f")

        def speedup(b_r, k_r, key):
            if b_r.get("oom") and not k_r.get("oom"):
                return "∞ (OOM)"
            if b_r.get("oom") or k_r.get("oom"):
                return "—"
            s = k_r[key] / b_r[key]
            return f"{s:.2f}x"

        d_spd = speedup(b, k, "mean_decode_tps")
        t_spd = speedup(b, k, "mean_total_tps")

        print(f"  {bs:>5}  "
              f"{b_ttft:>14}  {k_ttft:>14}  "
              f"{b_dtps:>16}  {k_dtps:>16}  {d_spd:>12}  "
              f"{b_ttps:>16}  {k_ttps:>16}  {t_spd:>12}  "
              f"{b_mem:>12}  {k_mem:>12}")
    print(f"{'─'*W}")


@app.local_entrypoint()
def main(
    method: str  = "baseline",
    bits: int    = 2,
    batch_sizes: str = "1,2,4,8,16,32,64,128,256",
    n_batches: int   = 3,
    prefill_len: int = 1024,
    max_new_tokens: int = 512,
    model: str   = "meta-llama/Llama-2-7b-chat-hf",
    compare: bool = False,
):
    """
    Run one method or compare saved results.

    Examples:
        modal run modal_throughput.py --method baseline
        modal run modal_throughput.py --method kivi
        modal run modal_throughput.py --compare
    """
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if compare:
        # Load saved local results and print table
        tag  = f"p{prefill_len}_g{max_new_tokens}"
        b_f  = out_dir / f"throughput_baseline_{tag}.json"
        k_f  = out_dir / f"throughput_kivi_{tag}.json"
        if not b_f.exists() or not k_f.exists():
            print(f"Missing result files. Run baseline and kivi first.")
            print(f"  Expected: {b_f}")
            print(f"  Expected: {k_f}")
            return
        with open(b_f) as f: b_res = json.load(f)
        with open(k_f)  as f: k_res = json.load(f)
        _print_table(b_res, k_res)
        return

    bs_list = [int(x) for x in batch_sizes.split(",")]
    print(f"Method: {method}  bits={bits}")
    print(f"Batch sizes: {bs_list}")
    print(f"Prefill: {prefill_len} tok  Generate: {max_new_tokens} tok  Batches/size: {n_batches}")

    result = run_throughput.remote(
        method, bits, bs_list, n_batches, prefill_len, max_new_tokens, model,
    )

    tag    = f"throughput_{method}_p{prefill_len}_g{max_new_tokens}"
    out_f  = out_dir / f"{tag}.json"
    with open(out_f, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_f}")

    # Single-method table
    print(f"\n── {method} (prefill={prefill_len}, generate={max_new_tokens}) ──")
    print(f"  {'BS':>5}  {'TTFT ms':>10}  {'Decode tok/s':>14}  {'Total tok/s':>12}  {'Peak GB':>9}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*9}")
    for bs, r in result["results"].items():
        if r.get("oom"):
            print(f"  {bs:>5}  {'OOM':>10}  {'OOM':>14}  {'OOM':>12}  {'OOM':>9}")
        else:
            print(f"  {int(bs):>5}  {r['mean_ttft_ms']:>10.0f}  "
                  f"{r['mean_decode_tps']:>14.0f}  "
                  f"{r['mean_total_tps']:>12.0f}  "
                  f"{r['mean_peak_mem_gb']:>9.1f}")
