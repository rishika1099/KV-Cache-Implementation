# KV-Cache Optimization for LLM Inference

> **Course:** High Performance Machine Learning  
> **Semester:** Spring 2026  
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** Team KV
- **Members:**
  - Rishika Mamidibathula (rm4318) — TopK / TopK-Flash / KIVI×TopK hybrid / Modal+W&B harness
  - Suhas Morisetty (vm2825) — KIVI 2/4-bit quantization / CUDA+Triton kernels / throughput benchmarking
  - Ziheng Wang (zw3153) — MLA latent KV via TransMLA / SnapKV integration / RoPE+GQA plumbing
  - Tony Tian (jt3640) — SnapKV / LongBench evaluation and analysis

## Submission

- **GitHub repository:** [https://github.com/rishika1099/KV-Cache-Implementation](https://github.com/rishika1099/KV-Cache-Implementation)
- **Final report:** [`deliverables/TeamKV_HPML_Final_Report.pdf`](deliverables/TeamKV_HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/TeamKV_HPML_Final_Presentation.pdf`](deliverables/TeamKV_HPML_Final_Presentation.pdf)
- **Experiment-tracking dashboard:** [https://wandb.ai/rm4318-columbia-university/kv-cache-hpml](https://wandb.ai/rm4318-columbia-university/kv-cache-hpml)
- **Companion blog post (bonus):** [https://rishikamamidibathula.substack.com/p/kv-cache-optimization](https://rishikamamidibathula.substack.com/p/kv-cache-optimization)

The final report PDF and the presentation file are checked into the `deliverables/` folder of this repository **and** uploaded to CourseWorks.

---

## 1. Problem Statement

Autoregressive LLM decoding is memory-bandwidth bound: every generated token requires re-reading the entire KV cache from GPU HBM. For Llama-2-7B at batch size 32 and 4K context, the KV cache alone consumes ~65 GB—nearly saturating an H100 80GB. We target **inference** optimization, benchmarking four compression families (quantization, sparse selection, eviction, latent projection) under identical conditions on a single H100 to quantify the memory–quality–throughput tradeoff for each.

---

## 2. Model/Application Description

- **Model architecture:** Llama-2-7B-chat-hf (32 layers, 32 heads, head dim 128, FP16 weights)
- **MLA model:** botsxc/llama2-7b-chat-mla-2048-8 (TransMLA latent checkpoint)
- **Framework:** PyTorch 2.4.1 · Transformers 4.44.2 · CUDA 12.1 · Triton 3.0 · flash-attn 2.5
- **Dataset:** LongBench (6 tasks × 20 examples = 120 examples/method; F1 / ROUGE-L / edit-similarity)
- **Custom layers:** KIVI CUDA+Triton quantized-GEMV kernels; TopK Triton fused scoring kernel; paged KV pool for TopK-Flash
- **Hardware target:** 1× NVIDIA H100 80GB HBM3 (Modal cloud)

---

## 3. Final Results Summary

| Metric | Baseline (FP16) | KIVI 4-bit | KIVI 2-bit | TopK K=1024 | SnapKV 0.4 | MLA (B=1) |
|---|---|---|---|---|---|---|
| LongBench overall | 0.291 | **0.292 (+0.3%)** | 0.273 (−6.2%) | **0.292 (+0.4%)** | **0.295 (+0.3%)** | 0.081* (−65%) |
| KV compression | 1× | ~4× | ~8× | None† | ~2.5× | **3.9×** |
| Decode tok/s (B=1) | 70 | 30 (0.4×) | 27 (0.4×) | 43 (0.6×) | ~70 | **43 (1.3×)**‡ |
| Best batch tput | 497 @ BS=32 | 873 @ BS=32 | **957 @ BS=32** | — | — | — |
| Max BS before OOM | 32 | **128** | **128** | — | — | — |
| Peak mem @ BS=32 | 65.0 GB | **27.8 GB** | **27.8 GB** | 65 GB | ~26 GB | — |

† TopK does not reduce KV storage — only attention scope is sparse.  
\* MLA baseline used 2048-token truncation vs. 4096 for other methods.  
‡ MLA decode speedup measured at gen=256; gen-length dependent.

**Hardware:** 1× NVIDIA H100 80GB HBM3, CUDA 12.1, PyTorch 2.4.1, Ubuntu 22.04 (Modal)

**Headline result:** KIVI 4-bit incurs no measurable LongBench quality drop (0.292 vs 0.291) while delivering 2.3× smaller KV cache and 1.93× decode throughput at BS=32. By extending the maximum serviceable batch size from 32 to 128, KIVI 2-bit raises the single-GPU peak throughput by 3.1× (1,517 vs 497 tok/s).

---

## 4. Repository Structure
