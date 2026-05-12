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
- **Final presentation:** [`deliverables/TeamKV_HPML_Final_Presentation.pdf`](deliverables/KV_Cache_Optimization_Presentation.pdf)
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

```
.
├── README.md
├── LICENSE                        # MIT
├── requirements.txt
├── configs/                       # YAML configs for experiments
├── deliverables/                  # Final report PDF + presentation
│   ├── TeamKV_HPML_Final_Report.pdf
│   └── KV_Cache_Optimization_Presentation.pdf
├── scripts/                       # Modal launch scripts + run helpers
│   ├── modal_longbench.py         # LongBench quality (baseline/KIVI/TopK)
│   ├── modal_longbench_snapkv.py  # LongBench quality (SnapKV)
│   ├── modal_throughput.py        # Throughput batch sweep (baseline/KIVI)
│   ├── modal_topk_throughput.py   # TopK throughput, incl. 32K
│   ├── modal_mla_benchmark.py     # MLA quality
│   ├── modal_mla_throughput.py    # MLA throughput
│   └── ...
├── methods/                       # KV cache method implementations
│   ├── baseline.py
│   ├── kivi_quant.py
│   ├── llama_kivi_model.py
│   ├── topk_selection.py
│   ├── llama_topk_model.py
│   ├── llama_topk_flash_model.py
│   ├── snapkv_eviction.py
│   ├── llama_snapkv_model.py
│   ├── kivi_topk_hybrid.py
│   ├── kivi_kernels/              # CUDA/Triton quantized-GEMV kernels
│   └── transmla/                  # Multi-head Latent Attention (TransMLA)
├── benchmark/                     # Harness: datasets, runner, metrics
├── experiments/                   # Experiment scripts (ablations, sweeps)
├── tests/                         # Unit tests for SnapKV, KIVI quant
├── notebooks/
│   └── mla_chat_demo.ipynb
├── results/                       # JSON result files + analysis markdown
│   ├── RESULTS.md
│   ├── TOPK_RESULTS.md
│   └── RESULTS_MLA.md
└── docs/                          # RUN.md, SUMMARY.md
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
git clone https://github.com/rishika1099/KV-Cache-Implementation.git
cd KV-Cache-Implementation
git checkout final
pip install modal
modal setup          # authenticate with Modal
modal secret create huggingface HF_TOKEN=<your_hf_token>
pip install -r requirements.txt
```

**System requirements:** Python 3.10+, CUDA 12.x, Modal account. All GPU compute runs on Modal H100 containers — no local GPU required.

### B. Experiment Tracking Dashboard

> **Dashboard:** [https://wandb.ai/rm4318-columbia-university/kv-cache-hpml](https://wandb.ai/rm4318-columbia-university/kv-cache-hpml)  
> *Platform:* Weights & Biases — opens publicly without login.

### C. Dataset

LongBench is downloaded automatically by the benchmark harness (`benchmark/datasets.py`) from HuggingFace at run time. No manual download needed.

### D. Reproduce Quality Results (LongBench)

```bash
modal run scripts/modal_longbench.py                             # Baseline
modal run scripts/modal_longbench.py --method kivi --bits 4      # KIVI 4-bit
modal run scripts/modal_longbench.py --method kivi --bits 2      # KIVI 2-bit
modal run scripts/modal_longbench.py --method topk --top-k 1024  # TopK
modal run scripts/modal_longbench_snapkv.py --budget-ratio 0.4   # SnapKV
modal run scripts/modal_mla_benchmark.py --run-baseline --run-mla
```

### E. Reproduce Throughput Results

```bash
# KIVI batch-size sweep (prefill=1024, gen=512)
modal run scripts/modal_throughput.py --method baseline
modal run scripts/modal_throughput.py --method kivi

# TopK-Flash at long context (32K)
modal run scripts/modal_topk_throughput.py --method baseline --prefill-len 32768
modal run scripts/modal_topk_throughput.py --method topk_flash --prefill-len 32768

# MLA throughput
modal run scripts/modal_mla_throughput.py --compare
```

See `docs/RUN.md` for the full parameter reference.

### F. Quickstart: Reproduce Headline Result

```bash
# 1. Install deps
pip install modal && modal setup
modal secret create huggingface HF_TOKEN=<token>

# 2. KIVI 4-bit throughput batch sweep (~10 min on H100)
modal run scripts/modal_throughput.py --method kivi

# 3. KIVI 4-bit LongBench quality (~20 min on H100)
modal run scripts/modal_longbench.py --method kivi --bits 4
```

Results are written to `results/` and logged to W&B.

---

## 6. Results and Observations

- **KIVI 4-bit — quantization with no quality cost:** Lossless LongBench quality (0.292 vs 0.291), 4× smaller KV cache, 1.93× decode throughput at BS=32, max batch size extended from 32 to 128 (3.1× peak throughput gain). Slower at BS=1 (0.4×) due to INT4 GEMV not mapping to tensor-core units.
- **TopK K=1024 — sparse attention without storage savings:** Matches baseline overall quality (0.292) and beats it on triviaqa (+20%) by filtering distracting passages. Zero memory reduction (full FP16 cache retained). Per-step Q·Kᵀ scoring overhead makes it slower at B=1; gains appear at long context (32K+) where the saved attention work outweighs scoring cost.
- **TopK-Flash — long-context efficiency:** Paged KV pool + flash-attn reduces 32K TTFT by 33% (2627→1761 ms) and peak memory by 19% (52→42 GB). Baseline OOMs at 64K; TopK-Flash handles 64K at 65.8 GB.
- **SnapKV 0.4 — best overall quality:** LongBench 0.295 (+0.3%), ~2.5× KV reduction, zero per-step decode overhead. One-shot attention-vote eviction after prefill. Only degrades on qasper (−2 pt) where evidence spans fall outside the 32-token observation window.
- **MLA — memory and latency win, quality needs retraining:** 3.9× KV reduction and 1.26–1.32× decode speedup at B=1 (gen=256). Quality collapses (−65%) because the available checkpoint was calibrated at 256-token context with rank-8 latent; recovering quality requires retraining at ≥4K context.
- **KIVI×TopK hybrid (32K):** 51% memory reduction (25.8 vs 52.1 GB) with a 40% decode slowdown — Q·Kᵀ scoring still runs over all 32K tokens before selection, and INT4 GEMV overhead compounds with selection cost. The hybrid demonstrates that orthogonal compression strategies can stack memory savings, but selection-cost reduction is needed for the latency win.

---

## 7. Notes

### AI Use Disclosure

*Per the HPML AI Use Policy.*

**Did your team use any AI tool in completing this project?**
- [x] Yes, we used AI assistance as described below.

**Tool(s) used:** Claude (Anthropic), GitHub Copilot

**Specific purpose:** Debugging CUDA OOM errors during KIVI kernel integration; clarifying Triton kernel semantics; polishing report prose drafted by team members; reorganizing the README from an earlier flat structure into the HPML template layout.

**Sections affected:** Triton-kernel debugging in `methods/kivi_kernels/`; report §V Discussion (proofreading only); README §3, §4, §6 (structure and prose polish).

**How we verified correctness:** Every reported number was produced by running our own Modal scripts against H100 containers; we re-ran each headline experiment at least twice and confirmed agreement. Profiler-trace interpretations were checked against raw `torch.profiler` traces in `results/`. AI-suggested code was rewritten in our own words and verified to produce identical results to a hand-written reference.

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above. The same disclosure block appears as an appendix in the final report.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

```bibtex
@misc{teamkv2026hpml,
  title  = {Comparative Evaluation of KV Cache Optimization Strategies for Efficient LLM Inference},
  author = {Mamidibathula, Rishika and Morisetty, Suhas and Wang, Ziheng and Tian, Tony},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/rishika1099/KV-Cache-Implementation}
}
```

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
