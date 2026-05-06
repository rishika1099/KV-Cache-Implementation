## MLA Latent KV Cache — Benchmark Results

## Overview

This report benchmarks **MLA latent KV cache** against a full-precision (FP16) baseline on **Llama-2-7B-chat**, focusing on *single-request latency and KV memory* as context length increases.

**MLA key idea (high level):** Instead of storing full per-layer K/V tensors for every token, MLA stores a **compressed latent representation** that can be expanded as needed during decode. This reduces KV cache footprint substantially (here ~3.9×), which can reduce peak memory and (depending on implementation) change TTFT and decode speed.

---

## Setup

| Item              | Value                                                                             |
| ----------------- | --------------------------------------------------------------------------------- |
| Base model        | `meta-llama/Llama-2-7b-chat-hf`                                                   |
| MLA model         | `botsxc/llama2-7b-chat-mla-2048-8`                                                |
| Harness           | `modal_throughput.py`-style loop (manual decode; no `model.generate()` fast path) |
| Batch size        | **1** (hardcoded; avoids shared-cache-across-batch issue for MLA)                 |
| Sweep axis        | **Prefill length** ∈ {512, 1024, 2048}                                            |
| Generation length | 512 tokens (as in your earlier runs)                                              |
| Metrics           | TTFT (ms), decode time/throughput, KV cache size (MB), peak memory (GB)           |

### Key design choices vs `modal_throughput.py`

* **Batch size fixed to 1** (MLA cache sharing across batch complicates apples-to-apples).
* **Sweeps prefill length** (512/1024/2048) instead of sweeping batch size.
* **Comparison table includes KV cache MB + KV compression ratio** alongside TTFT / decode / peak memory.
* **Runtime image** matches `modal_mla_benchmark.py` stack (debian_slim + transformers; no triton build).

---

## Throughput & Memory Results

| Prefill | TTFT-Base (ms) | TTFT-MLA (ms) | Dec-Base | Dec-MLA | Dec Spd | KV-Base (MB) | KV-MLA (MB) | KV Ratio | Mem-Base (GB) | Mem-MLA (GB) |
| ------: | -------------: | ------------: | -------: | ------: | ------: | -----------: | ----------: | -------: | ------------: | -----------: |
|     512 |             52 |            39 |       34 |      43 |   1.27x |        402.1 |       103.7 |     3.9x |         13.92 |        13.66 |
|    1024 |             88 |            41 |       34 |      42 |   1.26x |        670.6 |       172.9 |     3.9x |         14.19 |        13.79 |
|    2048 |             88 |            76 |       34 |      44 |   1.32x |       1207.4 |       311.3 |     3.9x |         14.79 |        14.05 |

### Takeaways

* **KV cache shrink is consistent:** ~**3.9×** smaller KV across all prefill lengths (402→104MB, 671→173MB, 1207→311MB).
* **Peak memory drops modestly:** about **0.3–0.7 GB** lower with MLA in these runs (because weights/activations dominate at B=1, and KV is only part of the footprint).
* **Decode is faster with MLA here:** **~1.26–1.32×** improvement. This suggests baseline decode is at least partially **memory-bandwidth bound** even at B=1 for these sequence lengths, and MLA’s smaller KV reduces read pressure.
* **TTFT behavior differs by prefill:** MLA is substantially faster at 512/1024 (39 vs 52 ms; 41 vs 88 ms), but closer at 2048 (76 vs 88 ms). That’s consistent with “prefill cost + any MLA-specific bookkeeping” interacting with context length; it’s not purely monotone.

---

## Quality Results — LongBench (Baseline vs MLA Latent)

### Setup

* **Dataset:** LongBench (6 tasks × 20 examples = 120 total per method)
* **Truncation:** middle-truncate to your set token budget (baseline run used 2048; MLA run used 3000 per your logs)
* **Decode:** greedy
* **Metric:** task-specific LongBench metrics, aggregated to “overall”

### Results (your runs)

| Task            |  Baseline | MLA latent | Δ (MLA − Base) |
| --------------- | --------: | ---------: | -------------: |
| qasper          |     0.237 |      0.036 |         -0.201 |
| multifieldqa_en |     0.392 |      0.101 |         -0.291 |
| triviaqa        |     0.373 |      0.064 |         -0.309 |
| 2wikimqa        |     0.117 |      0.043 |         -0.074 |
| multi_news      |     0.193 |      0.103 |         -0.090 |
| lcc             |     0.118 |      0.140 |         +0.022 |
| **overall**     | **0.238** |  **0.081** |     **-0.157** |

### Interpretation

* **Large quality drop overall** in this MLA latent checkpoint relative to baseline on this 120-example slice.
* The degradation is broad across **QA / retrieval-heavy tasks** (qasper, multifieldqa_en, triviaqa), which are typically most sensitive to any change in attention/KV behavior.
* **lcc improves** slightly here (+0.022), which can happen when compression changes inductive bias (or simply due to small-n variance), but it doesn’t offset the big losses elsewhere.

> One important caveat: your baseline LongBench run and MLA LongBench run used **different truncation lengths** (2048 vs 3000 tokens per the logs you pasted). That can shift difficulty. Even with that caveat, the magnitude of the drop is big enough that it likely reflects a real quality regression in this MLA checkpoint/config, not just truncation noise.

---

## Summary

### Throughput / Memory (B=1, sweep prefill length)

* **KV cache:** ~**3.9× smaller**
* **Decode speed:** **~1.26–1.32× faster**
* **Peak mem:** modestly lower (**~0.3–0.7 GB**)

### Quality (LongBench, 20×6)

* **Overall:** **0.238 → 0.081** (large drop on this slice)

**Verdict:** In your current setup, **MLA latent is a clear win on KV size + decode speed**, but the particular MLA latent model/config you evaluated shows a **substantial quality regression** on LongBench.

---

If you want, I can adapt this report into the exact same structure/style as your KIVI writeup (including a “trade-offs” table and a “deployment guidance” section), but using **MLA vs baseline** as the comparison axis.
