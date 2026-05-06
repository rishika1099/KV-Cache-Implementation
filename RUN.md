# KV Cache Implementation — Replication Guide

This repo benchmarks three KV-cache optimisation methods against a FP16 baseline on
`meta-llama/Llama-2-7b-chat-hf` using [Modal](https://modal.com) for GPU compute.

**Methods covered:**
- **Baseline** — full FP16 attention, no compression
- **KIVI** — 2/4-bit per-token KV quantisation (`methods/kivi_quant.py`, `methods/llama_kivi_model.py`)
- **TopK** — sparse token selection via TokenSelect (`methods/topk_selection.py`, `methods/llama_topk_model.py`)
- **TopK-Flash** — same selection but with paged KV + flash-attn for long-context latency (`methods/llama_topk_flash_model.py`)
- **MLA** — Multi-head Latent Attention latent-KV cache via TransMLA (`methods/transmla/`)

---

## Prerequisites

```bash
pip install modal
modal setup          # authenticate with Modal
modal secret create huggingface HF_TOKEN=<your_hf_token>
```

The Modal volumes `kv-benchmark-results` and `hf-model-cache` are created automatically
on first run (`create_if_missing=True`). Results are also written locally to `results/`.

---

## 1. LongBench Quality (6 tasks x 20 examples = 120 examples per method)

Tasks: `qasper`, `multifieldqa_en`, `triviaqa`, `2wikimqa`, `multi_news`, `lcc`

### 1a. Baseline

```bash
modal run modal_longbench.py
```

Output: `results/longbench_baseline.json`

### 1b. KIVI

```bash
modal run modal_longbench.py --method kivi --bits 4
modal run modal_longbench.py --method kivi --bits 2
```

Output:
- `results/longbench_kivi_{"bits": 4, "group_size": 32, "residual_length": 128}.json`
- `results/longbench_kivi_{"bits": 2, "group_size": 32, "residual_length": 128}.json`

### 1c. TopK (K sweep)

```bash
modal run modal_longbench.py --method topk --top-k 2048 --n-per-task 20
modal run modal_longbench.py --method topk --top-k 1024 --n-per-task 20
modal run modal_longbench.py --method topk --top-k 512  --n-per-task 20
```

Output:
- `results/longbench_topk_{"K": 2048, "cosine_threshold": 0.9, "kernel_size": -1, "n_local": 512, "n_sink": 128}.json`
- `results/longbench_topk_{"K": 1024, "cosine_threshold": 0.9, "kernel_size": -1, "n_local": 512, "n_sink": 128}.json`
- `results/longbench_topk_{"K": 512,  "cosine_threshold": 0.9, "kernel_size": -1, "n_local": 512, "n_sink": 128}.json`

### 1d. MLA (LongBench — baseline vs MLA latent)

Uses `modal_mla_benchmark.py` which runs both models in one invocation.

```bash
# Llama-2-7B: baseline (GQA) vs MLA latent checkpoint
modal run modal_mla_benchmark.py --run-baseline --run-mla

# Dry-run (2 examples per task, quick sanity check)
modal run modal_mla_benchmark.py --run-baseline --run-mla --dry-run
```

Output:
- `results/mla_longbench_llama2_baseline_2048.json`
- `results/mla_longbench_llama2_mla_latent_2048.json`

Models used:
- Baseline: `meta-llama/Llama-2-7b-chat-hf`
- MLA: `botsxc/llama2-7b-chat-mla-2048-8`

---

## 2. Throughput Benchmarks

### 2a. KIVI throughput (batch-size sweep, prefill=1024, gen=512)

```bash
modal run modal_throughput.py --method baseline
modal run modal_throughput.py --method kivi
modal run modal_throughput.py --compare   # prints comparison table
```

Output:
- `results/throughput_baseline_p1024_g512.json`
- `results/throughput_kivi_p1024_g512.json`

Sweeps batch sizes 1, 2, 4, 8, 16, 32, 64, 128 (256 will OOM for baseline).

### 2b. TopK throughput at batch size 1 (non-flash, prefill=4096)

Uses `modal_throughput_topk.py` — companion to 2a, measures TopK at the same
LongBench-style prefill length.

```bash
modal run modal_throughput_topk.py --top-k 1024 --prefill-len 4096 --max-new-tokens 512
```

Output: `results/throughput_topk_K1024_p4096_g512.json`

### 2c. TopK-Flash throughput at long context (32K / 64K prefill)

Uses `modal_topk_throughput.py` with flash-attn + paged KV (`llama_topk_flash_model.py`).
Demonstrates zero-copy selective attention at prefill lengths where selection meaningfully
reduces HBM bandwidth.

```bash
# Baseline at 32K prefill
modal run modal_topk_throughput.py --method baseline --prefill-len 32768

# TopK (standard gather) at 32K prefill
modal run modal_topk_throughput.py --method topk    --prefill-len 32768 --topk-k 4096

# TopK-Flash (paged + flash-attn) at 32K prefill
modal run modal_topk_throughput.py --method topk_flash --prefill-len 32768 --topk-k 4096

# TopK-Flash at 64K prefill
modal run modal_topk_throughput.py --method topk_flash --prefill-len 65536 --topk-k 4096
```

Output:
- `results/topk_throughput_baseline_p32768_g512.json`
- `results/topk_throughput_topk_p32768_g512.json`
- `results/topk_throughput_topk_flash_p32768_g512.json`
- `results/topk_throughput_topk_flash_p65536_g512.json`

Note: requires `flash-attn>=2.5.0` and `triton>=2.3.0`. The Modal image for this script
installs them from the CUDA-devel PyTorch registry image automatically.

### 2d. MLA throughput (prefill sweep, batch size 1)

```bash
modal run modal_mla_throughput.py --method baseline --prefill-lens 512,1024,2048
modal run modal_mla_throughput.py --method mla      --prefill-lens 512,1024,2048
modal run modal_mla_throughput.py --compare
```

Output:
- `results/mla_throughput_baseline_g256.json`
- `results/mla_throughput_mla_g256.json`

---

## 3. Result Files Reference

| File | Description |
|------|-------------|
| `results/RESULTS.md` | KIVI quality + throughput analysis writeup |
| `results/TOPK_RESULTS.md` | TopK LongBench quality analysis writeup |
| `results/RESULTS_MLA.md` | MLA throughput + quality analysis writeup |
| `results/longbench_baseline.json` | Baseline LongBench (n=120) |
| `results/longbench_kivi_{"bits": 4, ...}.json` | KIVI 4-bit LongBench |
| `results/longbench_kivi_{"bits": 2, ...}.json` | KIVI 2-bit LongBench |
| `results/longbench_topk_{"K": 2048, ...}.json` | TopK K=2048 LongBench |
| `results/longbench_topk_{"K": 1024, ...}.json` | TopK K=1024 LongBench |
| `results/longbench_topk_{"K": 512, ...}.json` | TopK K=512 LongBench |
| `results/mla_longbench_llama2_baseline_2048.json` | MLA baseline LongBench |
| `results/mla_longbench_llama2_mla_latent_2048.json` | MLA latent LongBench |
| `results/throughput_baseline_p1024_g512.json` | Baseline throughput batch sweep |
| `results/throughput_kivi_p1024_g512.json` | KIVI throughput batch sweep |
| `results/topk_throughput_baseline_p32768_g512.json` | Baseline at 32K prefill |
| `results/topk_throughput_topk_p32768_g512.json` | TopK (no flash) at 32K prefill |
| `results/topk_throughput_topk_flash_p32768_g512.json` | TopK-Flash at 32K prefill |
| `results/topk_throughput_topk_flash_p65536_g512.json` | TopK-Flash at 64K prefill |
| `results/mla_throughput_baseline_g256.json` | MLA baseline throughput sweep |
| `results/mla_throughput_mla_g256.json` | MLA latent throughput sweep |

---

## 4. Code Structure

```
methods/
  base.py                      # Abstract MethodWrapper interface
  baseline.py                  # No-op baseline
  kivi_quant.py                # KIVI quantisation wrapper
  llama_kivi_model.py          # Custom Llama model with KIVI KV cache
  topk_selection.py            # TopK (TokenSelect) selection wrapper
  llama_topk_model.py          # Custom Llama model with TopK gather-based attention
  llama_topk_flash_model.py    # Custom Llama model with TopK flash-attn paged attention
  kivi_kernels/                # CUDA kernels for KIVI quantised GEMV
  transmla/                    # TransMLA: Multi-head Latent Attention implementation
    transformers/llama/
      configuration_llamamla.py
      modeling_llamamla.py       # MLA full-cache variant
      modeling_llamamla_latent.py # MLA latent-cache variant
      mla.py / mla_latent.py

benchmark/
  datasets.py                  # LongBench + synthetic dataset loaders
  runner.py                    # generate_with_method(), PPL evaluation
  metrics.py                   # Metric helpers

modal_longbench.py             # LongBench runner (baseline / KIVI / TopK)
modal_mla_benchmark.py         # LongBench runner (baseline vs MLA)
modal_throughput.py            # Throughput batch sweep (baseline / KIVI)
modal_throughput_topk.py       # Throughput at B=1 (TopK, matches LongBench prefill)
modal_topk_throughput.py       # Throughput at long context (baseline / TopK / TopK-Flash)
modal_mla_throughput.py        # Throughput prefill sweep (baseline / MLA)
```
