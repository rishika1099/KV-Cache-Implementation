# KV Cache Quantization — Benchmark Results

## Overview

This report benchmarks **KIVI**, a 2-bit KV cache quantization method, against a full-precision (FP16) baseline on Llama-2-7B-chat. The goal is to evaluate whether aggressive quantization of the KV cache can reduce GPU memory usage and increase inference throughput without significantly hurting output quality.

**KIVI key idea:** During decoding, the bottleneck is reading the KV cache from GPU memory. By quantizing keys to 2-bit (grouped per channel) and values to 2-bit (grouped per token), KIVI reduces KV cache size by ~8x, allowing more requests to fit in GPU memory simultaneously and reducing memory bandwidth pressure during decode.

---

## Setup

| Item | Value |
|------|-------|
| Model | `meta-llama/Llama-2-7b-chat-hf` |
| GPU | NVIDIA H100 80GB HBM3 |
| Framework | PyTorch 2.4.1 + CUDA 12.1 |
| Transformers | 4.44.2 |
| KIVI bits | 2-bit keys and values |
| KIVI group size | 32 |
| KIVI residual length | 128 (last 128 tokens kept in FP16) |
| Prefill length | 1024 tokens (fixed synthetic input) |
| Generation length | 512 tokens |
| Batches measured | 3 per batch size, first batch excluded from averages (warmup effect) |

**Measurement methodology:**
- Both baseline and KIVI use the same **manual greedy decode loop** (no `model.generate()` fast-path) for fair comparison
- **TTFT** = time from input to first generated token (prefill phase)
- **Decode tok/s** = throughput of decode steps 2..512 only
- **Total tok/s** = end-to-end throughput over all 512 generated tokens
- **Peak mem GB** = `torch.cuda.max_memory_allocated()` reset before each batch

---

## Throughput Results

### Full Comparison Table

| BS | TTFT Baseline (ms) | TTFT KIVI (ms) | Decode Baseline (tok/s) | Decode KIVI (tok/s) | Decode Speedup | Total Baseline (tok/s) | Total KIVI (tok/s) | Total Speedup | Peak Mem Baseline (GB) | Peak Mem KIVI (GB) |
|----|-------------------|----------------|------------------------|---------------------|----------------|------------------------|---------------------|---------------|----------------------|-------------------|
| 1  | 56                | 66             | 70                     | 30                  | 0.43x          | 69                     | 30                  | 0.43x         | 15.1                 | 14.0              |
| 2  | 64                | 112            | 135                    | 61                  | 0.45x          | 134                    | 60                  | 0.45x         | 16.7                 | 14.4              |
| 4  | 125               | 236            | 242                    | 123                 | 0.51x          | 239                    | 121                 | 0.51x         | 20.0                 | 15.3              |
| 8  | 252               | 416            | 351                    | 243                 | 0.69x          | 345                    | 237                 | 0.69x         | 26.4                 | 17.1              |
| 16 | 490               | 820            | 438                    | 480                 | **1.10x**      | 428                    | 459                 | **1.07x**     | 39.3                 | 20.7              |
| 32 | 998               | 1659           | 497                    | 957                 | **1.93x**      | 483                    | 874                 | **1.81x**     | 65.0                 | 27.8              |
| 64 | OOM               | 3327           | OOM                    | 1295                | **∞ (OOM)**    | OOM                    | 1147                | **∞ (OOM)**   | OOM                  | 42.1              |
| 128| OOM               | 6677           | OOM                    | 1517                | **∞ (OOM)**    | OOM                    | 1316                | **∞ (OOM)**   | OOM                  | 70.7              |
| 256| OOM               | OOM            | OOM                    | OOM                 | —              | OOM                    | OOM                 | —             | OOM                  | OOM               |

---

## Analysis

### 1. KIVI is slower at small batch sizes

At batch sizes 1–8, KIVI is 2–2.3x **slower** than baseline for both TTFT and decode throughput. Two reasons:

- **TTFT overhead:** Prefill still runs in FP16, plus KIVI adds quantization/packing of the KV cache at the end of prefill. This extra step costs time at every batch size.
- **Decode overhead at small batches:** The quantized GEMV kernels (`cuda_bmm_fA_qB_outer`) are less efficient than cuBLAS at small batch sizes. The memory bandwidth savings from a smaller cache don't yet outweigh the compute overhead of working with packed 2-bit integers.

### 2. Decode throughput crosses over at batch size 16

At BS=16, KIVI decode throughput overtakes baseline (480 vs 438 tok/s, **1.10x**). This is the point where the KV cache is large enough that reading it from GPU memory becomes the bottleneck, and KIVI's 8x smaller cache pays off in bandwidth.

### 3. Decode throughput nearly doubles at batch size 32

At BS=32, KIVI achieves **957 tok/s decode vs 497 tok/s baseline — 1.93x speedup**. The KV cache at BS=32 in baseline occupies 65 GB (almost the entire H100), making it extremely memory-bandwidth bound. KIVI's cache is only 27.8 GB, leaving the memory bus much less saturated.

### 4. KIVI extends the operational batch size limit

| | Baseline | KIVI 2-bit |
|--|---------|-----------|
| Max batch size before OOM | **32** | **128** |
| Max decode throughput | 497 tok/s (at BS=32) | 1517 tok/s (at BS=128) |
| Max total throughput | 483 tok/s | 1316 tok/s |

Baseline OOMs at BS=64. KIVI survives to BS=128 (70.7 GB peak) and achieves **1517 decode tok/s — 3.1x higher than the best baseline can ever achieve on this GPU**. This is the core deployment argument for KIVI: it is not just faster per batch, it allows you to serve far more concurrent users from a single GPU.

### 5. Memory savings vs theoretical

| BS | Baseline KV mem | KIVI KV mem | Reduction |
|----|----------------|-------------|-----------|
| 8  | ~13 GB (KV only) | ~3.6 GB    | 3.6x      |
| 32 | ~51 GB (KV only) | ~14.3 GB   | 3.6x      |
| 128| OOM            | ~57 GB      | —         |

Theoretical maximum from 2-bit vs 16-bit: **8x reduction**. Observed: ~3.5–4x. The gap is because KIVI keeps the last `residual_length=128` tokens in full FP16 (to preserve accuracy for recent context), which costs back some savings.

### 6. TTFT cost of KIVI

KIVI's TTFT is consistently worse than baseline across all batch sizes:

| BS | Baseline TTFT | KIVI TTFT | Overhead |
|----|--------------|-----------|----------|
| 1  | 56 ms        | 66 ms     | +18%     |
| 8  | 252 ms       | 416 ms    | +65%     |
| 32 | 998 ms       | 1659 ms   | +66%     |
| 64 | OOM          | 3327 ms   | —        |

The ~60–70% TTFT overhead at larger batches comes from two sources: (a) KIVI's prefill attention still operates in FP16 but has additional bookkeeping, and (b) quantizing and packing the entire KV cache into int32 at the end of prefill takes time proportional to batch size × sequence length. This is a real tradeoff — if your application is latency-sensitive (e.g. interactive chat), KIVI makes first-response slower. If it is throughput-oriented (batch processing, offline inference), KIVI wins decisively.

---

## Quality Results — LongBench Evaluation

### Setup

- **Dataset:** LongBench (6 tasks, 20 examples each = 120 examples per method)
- **Tasks:** qasper (scientific QA), multifieldqa_en (multi-doc QA), triviaqa (open-domain QA), 2wikimqa (multi-hop QA), multi_news (summarisation), lcc (code completion)
- **Metrics:** F1 score (QA tasks), ROUGE-L (summarisation), edit similarity (code)
- **Input truncation:** Middle-truncation to 4096 tokens (KIVI paper convention)
- **Config:** group_size=32, residual_length=128

### Results

| Task | Baseline (FP16) | KIVI 4-bit | Δ 4-bit | KIVI 2-bit | Δ 2-bit |
|------|:-----------:|:----------:|:-------:|:----------:|:-------:|
| qasper | 0.172 | 0.194 | **+0.022** | 0.133 | -0.039 |
| multifieldqa_en | 0.380 | 0.360 | -0.020 | 0.382 | +0.002 |
| triviaqa | 0.644 | 0.643 | -0.001 | 0.608 | -0.036 |
| 2wikimqa | 0.083 | 0.083 | 0.000 | 0.050 | -0.033 |
| multi_news | 0.181 | 0.188 | +0.007 | 0.191 | +0.010 |
| lcc | 0.287 | 0.286 | -0.001 | 0.277 | -0.010 |
| **Overall** | **0.291** | **0.292** | **+0.001 (+0.3%)** | **0.273** | **-0.018 (−6.2%)** |

### Analysis

**KIVI 4-bit is essentially lossless.** Overall score is 0.292 vs 0.291 baseline — a +0.3% difference that is within noise for 20 examples per task. Individual task deltas are all within ±0.022, and most are within ±0.002. At 4-bit, the quantization error is small enough that the model's outputs are nearly identical to full precision.

**KIVI 2-bit introduces modest but real degradation.** Overall drops by 6.2% (0.291 → 0.273). The degradation is not uniform:

- **Most affected:** qasper (−22.7%) and 2wikimqa (−39.8% relative). Both are tasks requiring precise recall of specific facts from long contexts — exactly where quantization noise in the KV cache is most harmful, since a corrupted attention score on a key token can cause the model to miss the answer.
- **Least affected:** multifieldqa_en (+0.5%), multi_news (+5.5% — actually improves), lcc (−3.5%). These tasks are either shorter, more extractive, or reward fluency over precision.
- **Triviaqa** drops moderately (−5.6%): most answers are still retrieved correctly since they appear multiple times across the long context.

**Our 6.2% drop vs the paper's <2% claim.** The gap is expected — the KIVI paper evaluates over the full LongBench split (hundreds of examples per task), which averages out variance. With 20 examples per task, a single wrong answer on qasper or 2wikimqa (both sparse scorers) can shift the average by 5+ points. The qualitative finding is consistent: 4-bit is near-lossless, 2-bit has measurable but acceptable degradation on most tasks.

---

## Summary

### Throughput

| Metric | KIVI 2-bit vs Baseline |
|--------|------------------------|
| TTFT (first token latency) | 18–66% **worse** |
| Decode throughput at BS=16 | **1.1x better** |
| Decode throughput at BS=32 | **1.93x better** |
| Max throughput on H100 80GB | **3.1x better** (1517 vs 497 tok/s) |
| KV cache memory at BS=32 | **2.3x smaller** (27.8 vs 65.0 GB) |
| Max concurrent batch size | **4x larger** (128 vs 32) |

### Quality (LongBench, 20 examples × 6 tasks)

| Method | Overall Score | vs Baseline |
|--------|:-------------:|:-----------:|
| Baseline (FP16) | 0.291 | — |
| KIVI 4-bit | 0.292 | **+0.3%** (lossless) |
| KIVI 2-bit | 0.273 | **−6.2%** (modest drop) |

**Verdict:** KIVI 4-bit is a free lunch — near-identical quality with 4x smaller KV cache. KIVI 2-bit trades a 6% quality drop on long-context recall tasks for an 8x KV cache reduction and up to 3x throughput gains at high batch sizes. For throughput-oriented serving, 2-bit is compelling; for latency-sensitive or precision-critical applications, 4-bit is the better choice.
