"""
modal_throughput_snapkv_batch.py

Throughput benchmark: baseline vs SnapKV across batch sizes at prefill=4096.
Uses random-token prompts (equal length, no padding) — valid for any batch size.

Run one method at a time:
    modal run modal_throughput_snapkv_batch.py --method baseline
    modal run modal_throughput_snapkv_batch.py --method snapkv

Then compare:
    modal run modal_throughput_snapkv_batch.py --compare

Output JSON:
    results/throughput_snapkv_batch_baseline_p4096_g512.json
    results/throughput_snapkv_batch_snapkv_p4096_g512.json
"""

import json
import sys
import time
from pathlib import Path

import modal

app   = modal.App("kv-throughput-snapkv-batch")
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


def _timed_generate(model, tokenizer, input_ids, max_new_tokens):
    """Greedy decode with separate TTFT and decode timing."""
    import torch
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    pad_id = tokenizer.pad_token_id  if tokenizer.pad_token_id  is not None else 0
    BS     = input_ids.shape[0]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        t0 = time.perf_counter()
        out        = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_kv    = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_s = time.perf_counter() - t0
        del out

        finished = (next_token.squeeze(-1) == eos_id)
        n_decode  = 0
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if finished.all():
                break
            out        = model(input_ids=next_token, past_key_values=past_kv,
                               use_cache=True, return_dict=True)
            past_kv    = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            finished   = finished | (next_token.squeeze(-1) == eos_id)
            next_token[finished] = pad_id
            n_decode  += 1
            del out
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
    batch_sizes: list,
    n_batches: int,
    prefill_len: int,
    max_new_tokens: int,
    snapkv_window_size: int,
    snapkv_max_capacity_prompt: int,
    snapkv_kernel_size: int,
    snapkv_pooling: str,
    snapkv_sink_size: int,
    model_name: str,
) -> dict:
    _bootstrap()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[modal] method={method}  prefill={prefill_len}  max_new_tokens={max_new_tokens}")
    print(f"[modal] batch_sizes={batch_sizes}  n_batches={n_batches}")
    print(f"[modal] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[modal] Total GPU mem: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa",
    )

    if method == "snapkv":
        from methods.llama_snapkv_model import snapkv_monkey_patch
        snapkv_monkey_patch(
            model,
            window_size=snapkv_window_size,
            max_capacity_prompt=snapkv_max_capacity_prompt,
            kernel_size=snapkv_kernel_size,
            pooling=snapkv_pooling,
            sink_size=snapkv_sink_size,
        )
        print(f"[modal] SnapKV: window={snapkv_window_size}  "
              f"max_capacity_prompt={snapkv_max_capacity_prompt}  "
              f"kernel={snapkv_kernel_size}  pool={snapkv_pooling}")

    model.eval()
    print(f"[modal] Model loaded. Mem={torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Warmup at B=1
    print("[modal] Warming up...")
    dummy = torch.randint(100, tokenizer.vocab_size - 100, (1, prefill_len), device="cuda")
    _timed_generate(model, tokenizer, dummy, 32)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[modal] Warmup done. Mem={torch.cuda.memory_allocated()/1e9:.1f} GB")

    results = {}
    torch.manual_seed(42)

    for bs in batch_sizes:
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        print(f"\n[modal] ── batch_size={bs}  (free mem: {free_gb:.1f} GB) ──")
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

                total_time = ttft_s + decode_time_s
                total_toks = bs * (1 + n_decode)
                decode_tps = (bs * n_decode) / decode_time_s if decode_time_s > 0 else 0
                total_tps  = total_toks / total_time if total_time > 0 else 0
                ttft_ms    = ttft_s * 1000

                batch_records.append({
                    "ttft_ms":      ttft_ms,
                    "decode_tps":   decode_tps,
                    "total_tps":    total_tps,
                    "peak_mem_gb":  peak_mem_gb,
                })
                flag = " [warmup]" if b_idx == 0 else ""
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
            break
        else:
            stable = batch_records[1:] if len(batch_records) > 1 else batch_records
            n = len(stable)
            results[bs] = {
                "oom":              False,
                "mean_ttft_ms":     sum(r["ttft_ms"]     for r in stable) / n,
                "mean_decode_tps":  sum(r["decode_tps"]  for r in stable) / n,
                "mean_total_tps":   sum(r["total_tps"]   for r in stable) / n,
                "mean_peak_mem_gb": sum(r["peak_mem_gb"] for r in stable) / n,
                "n_stable":         n,
            }
            r = results[bs]
            print(f"  → avg  ttft={r['mean_ttft_ms']:.0f}ms  "
                  f"decode={r['mean_decode_tps']:.0f} tok/s  "
                  f"total={r['mean_total_tps']:.0f} tok/s  "
                  f"peak={r['mean_peak_mem_gb']:.1f} GB  (n={n})")

    payload = {
        "method":         method,
        "prefill_len":    prefill_len,
        "max_new_tokens": max_new_tokens,
        "model":          model_name,
        "snapkv_config": {
            "window_size":         snapkv_window_size,
            "max_capacity_prompt": snapkv_max_capacity_prompt,
            "kernel_size":         snapkv_kernel_size,
            "pooling":             snapkv_pooling,
            "sink_size":           snapkv_sink_size,
        } if method == "snapkv" else None,
        "results": {str(k): v for k, v in results.items()},
    }

    tag = f"throughput_snapkv_batch_{method}_p{prefill_len}_g{max_new_tokens}"
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
    return format(r.get(key, 0), fmt)


def _print_comparison(baseline_res, snapkv_res):
    all_bs = sorted(
        set(int(k) for k in baseline_res["results"]) |
        set(int(k) for k in snapkv_res["results"])
    )
    W = 120
    prefill = baseline_res["prefill_len"]
    gen     = baseline_res["max_new_tokens"]
    cfg     = snapkv_res.get("snapkv_config") or {}
    print(f"\n{'─'*W}")
    print(f"  prefill={prefill}  generate={gen}  model={baseline_res['model'].split('/')[-1]}")
    if cfg:
        print(f"  SnapKV: window={cfg.get('window_size')}  "
              f"max_capacity_prompt={cfg.get('max_capacity_prompt')}  "
              f"pool={cfg.get('pooling')}")
    print(f"{'─'*W}")
    print(f"  {'BS':>5}  "
          f"{'Dec-Base':>10}  {'Dec-SnapKV':>10}  {'Dec Speedup':>12}  "
          f"{'Tot-Base':>10}  {'Tot-SnapKV':>10}  {'Tot Speedup':>12}  "
          f"{'Mem-Base':>10}  {'Mem-SnapKV':>10}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}")
    for bs in all_bs:
        b = baseline_res["results"].get(str(bs), {})
        s = snapkv_res["results"].get(str(bs),   {})

        def speedup(b_r, s_r, key):
            if b_r.get("oom") and not s_r.get("oom"):
                return "∞(OOM)"
            if b_r.get("oom") or s_r.get("oom"):
                return "—"
            return f"{s_r[key] / b_r[key]:.2f}x"

        print(f"  {bs:>5}  "
              f"{_fmt(b,'mean_decode_tps'):>10}  {_fmt(s,'mean_decode_tps'):>10}  "
              f"{speedup(b,s,'mean_decode_tps'):>12}  "
              f"{_fmt(b,'mean_total_tps'):>10}  {_fmt(s,'mean_total_tps'):>10}  "
              f"{speedup(b,s,'mean_total_tps'):>12}  "
              f"{_fmt(b,'mean_peak_mem_gb','.1f'):>10}  {_fmt(s,'mean_peak_mem_gb','.1f'):>10}")
    print(f"{'─'*W}")


@app.local_entrypoint()
def main(
    method: str           = "baseline",
    batch_sizes: str      = "1,2,4,8,16,32,64,128",
    n_batches: int        = 3,
    prefill_len: int      = 4096,
    max_new_tokens: int   = 512,
    snapkv_window_size: int         = 32,
    snapkv_max_capacity_prompt: int = -1,   # -1 = auto (0.4 × prefill_len)
    snapkv_kernel_size: int         = 7,
    snapkv_pooling: str             = "avgpool",
    snapkv_sink_size: int           = 0,
    model: str            = "meta-llama/Llama-2-7b-chat-hf",
    compare: bool         = False,
):
    """
    Batch-sweep throughput benchmark for SnapKV vs baseline at 4K prefill.

        modal run modal_throughput_snapkv_batch.py --method baseline
        modal run modal_throughput_snapkv_batch.py --method snapkv
        modal run modal_throughput_snapkv_batch.py --compare
    """
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"p{prefill_len}_g{max_new_tokens}"

    if compare:
        b_f = out_dir / f"throughput_snapkv_batch_baseline_{tag}.json"
        s_f = out_dir / f"throughput_snapkv_batch_snapkv_{tag}.json"
        if not b_f.exists() or not s_f.exists():
            print("Missing result files. Run baseline and snapkv first.")
            return
        with open(b_f) as f: b_res = json.load(f)
        with open(s_f) as f: s_res = json.load(f)
        _print_comparison(b_res, s_res)
        return

    if snapkv_max_capacity_prompt <= 0:
        snapkv_max_capacity_prompt = max(snapkv_window_size + 1, int(0.4 * prefill_len))
        print(f"[entry] auto snapkv_max_capacity_prompt = {snapkv_max_capacity_prompt}")

    bs_list = [int(x) for x in batch_sizes.split(",")]
    print(f"Method: {method}  model: {model}")
    print(f"Batch sizes: {bs_list}")
    print(f"Prefill: {prefill_len}  Generate: {max_new_tokens}  Batches/size: {n_batches}")

    result = run_throughput.remote(
        method, bs_list, n_batches, prefill_len, max_new_tokens,
        snapkv_window_size, snapkv_max_capacity_prompt,
        snapkv_kernel_size, snapkv_pooling, snapkv_sink_size,
        model,
    )

    out_f = out_dir / f"throughput_snapkv_batch_{method}_{tag}.json"
    with open(out_f, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_f}")

    print(f"\n── {method}  prefill={prefill_len}  generate={max_new_tokens} ──")
    print(f"  {'BS':>5}  {'TTFT ms':>10}  {'Decode tok/s':>14}  {'Total tok/s':>12}  {'Peak GB':>9}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*9}")
    for bs, r in result["results"].items():
        if r.get("oom"):
            print(f"  {int(bs):>5}  {'OOM':>10}  {'OOM':>14}  {'OOM':>12}  {'OOM':>9}")
        else:
            print(f"  {int(bs):>5}  {r['mean_ttft_ms']:>10.0f}  "
                  f"{r['mean_decode_tps']:>14.0f}  "
                  f"{r['mean_total_tps']:>12.0f}  "
                  f"{r['mean_peak_mem_gb']:>9.1f}")
