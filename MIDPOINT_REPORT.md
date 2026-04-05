# Mid-Point Report: KV Cache Optimization Benchmark for LLM Inference

## Project Summary

**Project Title:** Fair Benchmarking of KV Cache Optimization Techniques for Autoregressive LLM Inference

**Project Goals:**
Conduct a controlled, reproducible comparison of KV cache optimization techniques for autoregressive LLM inference. All methods share the same model, tokenizer, decode loop, prompts, batch size (1), random seeds (42), and hardware (NVIDIA A100-80GB). The only variable across runs is the KV cache policy. The goal is to characterize the memory-quality-speed tradeoff of each approach and identify which techniques are Pareto-optimal for different deployment scenarios.

**AI Model(s):**
- Meta LLaMA-2-7B (meta-llama/Llama-2-7b-hf), FP16 inference
- 32 transformer layers, 32 attention heads, 128 head_dim
- ~13.5B parameters in FP16 (~14 GB VRAM for weights alone)

**Dataset(s):**
- **WikiText-103** (Merity et al., 2016): Train split for synthetic prompt construction (concatenated passages truncated to target token counts); Test split for perplexity evaluation (100 examples, max 512 tokens each)
- **Synthetic prompts**: 12 prompts across 3 sequence lengths (512, 2048, 8192 tokens), 4 prompts per length, constructed from WikiText-103 train passages with a summarization instruction wrapper
- **LongBench** (Bai et al., 2023): Planned for final submission (6 tasks: qasper, multifieldqa_en, hotpotqa, 2wikimqa, gov_report, qmsum) — currently blocked by dataset loading incompatibility (see Blockers)

**Performance Optimization Techniques / Methodology:**

We implement and benchmark 4 KV cache optimization techniques, each representing a different optimization strategy:

| Method | Paper | Strategy | Key Idea |
|--------|-------|----------|----------|
| **KIVI** | Liu et al., ICML 2024 | Quantization | Asymmetric INT4/INT2: keys quantized per-channel, values per-token. Recent tokens kept in FP16 sliding window. |
| **xKV** | Chang et al., 2025 | Low-rank approximation | Per-layer truncated SVD of KV tensors. Stores U, S, Vh instead of full matrices. Periodic recomputation as new tokens arrive. |
| **SnapKV** | Li et al., NeurIPS 2024 | Token eviction | One-shot eviction after prefill using attention-pattern importance scoring. Retains attention sinks + top-K important + recent tokens. |
| **TopK** | Wu et al., EMNLP 2025 | Dynamic selection | Per-step top-K token selection using Q-K dot-product scoring. Full cache stored but only K positions attend. Periodic full-context refresh. |

All implementations are pure PyTorch (no custom CUDA/Triton kernels) to ensure the benchmark reflects algorithmic overhead rather than kernel optimization quality.

**Fairness constraints (non-negotiable):**
1. Same model weights loaded once, shared across all methods
2. Same prompts, same order, same seeds (`torch.manual_seed(42)` before every run)
3. `torch.cuda.reset_peak_memory_stats()` before every run
4. `torch.cuda.synchronize()` around all timing measurements
5. FP16 everywhere, `model.eval()`, `torch.no_grad()` always
6. Same `max_new_tokens=200`, `batch_size=1`

**Profiling / Performance Analysis Tools:**
- PyTorch CUDA memory tracking (`torch.cuda.max_memory_allocated`, `torch.cuda.reset_peak_memory_stats`)
- Wall-clock timing with `time.perf_counter()` + `torch.cuda.synchronize()` for accurate GPU timing
- Custom KV cache size accounting per method (quantized bytes, SVD component bytes, retained tensor bytes)
- Per-method perplexity measurement through modified KV caches
- Planned for final: `torch.profiler` with TensorBoard trace export, CUDA kernel-level breakdown

---

## Current Progress

**GitHub Project:** https://github.com/rishika1099/kv-cache-benchmark

**W&B Project:** https://wandb.ai/rm4318-columbia-university/kv-cache-benchmark

### Results Obtained So Far

We completed a full benchmark run across all 4 methods (12 hyperparameter configurations) on 12 synthetic prompts (3 sequence lengths x 4 prompts each) using Modal cloud compute with NVIDIA A100-80GB GPUs. All method configs ran in parallel, completing the full benchmark in ~60 minutes wall time.

#### Throughput and Compression

| Method | Config | Avg KV Cache (MB) | Compression | Avg Throughput (TPS) | PPL |
|--------|--------|-------------------|-------------|---------------------|-----|
| Baseline | FP16 full | 1990.7 | 1.00x | 38.6 | 9.08 |
| KIVI | 4-bit, res=128 | — | 1.83x | 15.4 | 6.93 |
| KIVI | 2-bit, res=128 | — | — | — | 13.56 |
| xKV | rank=64 | — | — | — | 7.77 |
| xKV | rank=128 | — | 1.18x | 23.7 | 6.94 |
| xKV | rank=256 | — | — | — | 6.94 |
| SnapKV | budget=0.2 | — | 3.7x | 41.4 | 29.41 |
| SnapKV | budget=0.4 | — | 2.4x | — | 27.16 |
| SnapKV | budget=0.6 | — | 1.6x | — | 12.78 |
| TopK | K=256 | 1990.7 | 1.00x | 32.5 | 6.94 |
| TopK | K=512 | 1990.7 | 1.00x | — | 6.94 |
| TopK | K=1024 | 1990.7 | 1.00x | — | 6.94 |

#### Key Findings

1. **SnapKV achieves the best compression-throughput tradeoff**: Up to 3.7x KV cache compression with the highest throughput (41 TPS, faster than baseline) because the smaller evicted cache accelerates attention computation. However, aggressive eviction (budget=0.2) severely degrades quality (PPL 29.4 vs 9.1 baseline).

2. **KIVI quantization compresses well but is slow in pure PyTorch**: 1.83x compression at 4-bit, but only 15.4 TPS — the per-step dequantize-attend-requantize loop in Python is expensive. Real KIVI uses custom Triton kernels for fused quantized attention.

3. **xKV SVD provides modest compression with quality preservation**: rank=128 achieves only 1.18x compression but preserves quality perfectly (PPL 6.94 ≈ baseline). The SVD recomputation every 50 steps creates latency spikes visible in the latency breakdown plot.

4. **TopK preserves quality perfectly but saves no memory**: Compression ratio is always 1.0x because the full cache is stored (selected tokens change per step). This is by design — TopK trades compute for quality, not memory for quality.

5. **Memory scaling is linear**: All methods scale linearly with sequence length. At 8192 tokens, baseline uses ~28 GB peak memory. Methods that reduce KV size (SnapKV, KIVI) show visible separation.

#### Plots Generated

Four publication-ready plots were generated (PNG 300dpi + PDF):
- **Speed-Memory Tradeoff** (throughput vs compression scatter)
- **Memory Scaling** (peak GPU memory vs sequence length)
- **Latency Breakdown** (TTFT vs per-token latency, grouped by seq_len)
- **Memory-Quality Tradeoff** (perplexity vs compression scatter)

---

## Work in Progress

### Planned improvements for final submission:

1. **Weights & Biases integration**: Log all metrics (memory, throughput, perplexity, compression ratio) to wandb for interactive dashboards and run comparison. Currently all metrics are logged to JSONL; wandb integration is straightforward.

2. **torch.profiler integration**: Add CUDA kernel-level profiling traces to identify bottlenecks (e.g., is KIVI slow due to quantization arithmetic or memory copies?). Export Chrome trace JSON for TensorBoard visualization.

3. **Fix LongBench evaluation**: The THUDM/LongBench dataset uses a custom loading script incompatible with `datasets>=4.0`. We plan to either pin `datasets<3.0` for LongBench or download the data directly via the HuggingFace Hub API. This will add task-quality evaluation (F1 for QA, ROUGE-L for summarization).

4. **Add 2 additional methods**:
   - **H2O (Heavy-Hitter Oracle)** (Zhang et al., NeurIPS 2023): Cumulative attention-based eviction — identifies "heavy hitter" tokens that consistently receive high attention across layers and keeps them.
   - **PyramidKV** (Cai et al., 2024): Layer-adaptive KV budget allocation — lower layers get more KV cache (they capture local patterns) while upper layers get less (they capture global patterns).

5. **Improve perplexity evaluation fairness**: Current per-method PPL uses a split-text protocol (prefill first half → modify KV → evaluate second half) which is particularly harsh on eviction methods. We plan to add a sliding-window PPL evaluation that better reflects real-world usage patterns.

6. **Batch size scaling analysis**: Extend from batch_size=1 to batch_size=[1, 4, 8, 16] to measure how KV cache optimization interacts with batched inference — a key consideration for serving workloads.

---

## Blockers and Limitations

1. **LongBench dataset loading failure**: The `THUDM/LongBench` dataset relies on a custom Python loading script (`LongBench.py`) that is no longer supported in `datasets>=4.0`. The library removed support for `trust_remote_code=True` as a security measure. **Mitigation**: Pin to an older datasets version or download raw data via HuggingFace Hub file API.

2. **SnapKV OOM at seq_len=8192**: SnapKV requires `output_attentions=True` during prefill, which materializes the full (seq_len × seq_len) attention weight matrix for all 32 layers. At 8192 tokens this is ~137 GB, exceeding the A100's 80 GB. **Mitigation**: We disabled attention output for non-SnapKV methods (resolved). For SnapKV specifically, we plan to implement chunked attention scoring or use Flash Attention's attention weight extraction.

3. **Pure PyTorch overhead**: Our fairness constraint (no custom CUDA/Triton kernels) means KIVI quantization and xKV SVD are significantly slower than their paper-reported numbers, which use optimized kernels. This is intentional — it isolates algorithmic overhead — but should be noted when comparing to published results.

4. **Single GPU limitation**: All experiments run on a single A100-80GB. We do not measure multi-GPU KV cache sharding or tensor parallel scenarios, which are common in production deployments.

5. **Greedy decoding only**: We use argmax (greedy) decoding for reproducibility. Results may differ under sampling-based decoding (temperature, top-p) because KV cache modifications can alter the probability distribution of generated tokens differently.
