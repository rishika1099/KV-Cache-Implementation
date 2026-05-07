# Mid-Point Report: KV Cache Optimization Benchmark for LLM Inference

## Project Summary

**Project Title:** Fair Benchmarking of KV Cache Optimization Techniques for Autoregressive LLM Inference

**Project Goals:**
The objective of this project is to build a controlled, reproducible benchmarking framework that evaluates KV cache optimization techniques for autoregressive LLM inference under identical conditions. Every method shares the same model, tokenizer, prompts, decode loop, batch size (1), random seeds (42), and hardware (NVIDIA A100-80GB). The sole variable between runs is the KV cache management policy. We aim to characterize the memory-quality-speed tradeoff of each technique and determine which approaches offer favorable Pareto tradeoffs for different deployment priorities.

**AI Model(s):**
- Meta LLaMA-2-7B (meta-llama/Llama-2-7b-hf), FP16 inference
- 32 transformer layers, 32 attention heads, 128-dimensional head embeddings
- Approximately 6.7B parameters loaded in FP16

**Dataset(s):**
- **Synthetic prompts**: 12 prompts across 3 sequence lengths (512, 2048, 8192 tokens), 4 prompts per length, constructed from WikiText-103 train passages wrapped with a summarization instruction. These measure throughput, latency, memory usage, and compression ratio.
- **WikiText-103** (Merity et al., 2016): Test split used for perplexity evaluation (20 examples, max 512 tokens each). Perplexity is computed by running each method's modified KV cache through the model's forward pass and measuring cross-entropy loss on ground-truth continuations.
- **LongBench** (Bai et al., 2023): 6 long-context tasks (qasper, multifieldqa_en, hotpotqa, 2wikimqa, gov_report, qmsum) with 20 examples per task. QA tasks are scored using token-level F1; summarization tasks use ROUGE-L. This evaluates whether KV cache modifications degrade downstream task performance on real long-document workloads.

**Performance Optimization Techniques / Methodology:**

We benchmark three KV cache optimization strategies, each representing a fundamentally different approach to reducing the memory or compute cost of attention:

| Method | Paper | Strategy | Core Mechanism |
|--------|-------|----------|----------------|
| **Baseline** | N/A | No optimization | Standard FP16 KV cache, full attention over all past tokens. Serves as the reference point for all comparisons. |
| **KIVI** | Liu et al., ICML 2024 | Quantization | Reduces the precision of cached key-value tensors from FP16 to INT4 or INT2. Keys are quantized per-channel and values per-token, each within independent blocks of a configurable group size. A sliding window of recent tokens is kept in full FP16 precision to preserve short-range attention fidelity. |
| **TopK (TokenSelect)** | Wu et al., EMNLP 2025 | Dynamic sparse attention | Retains the full KV cache in memory but selects only the top-K most relevant tokens per attention head at each decode step. Selection uses Q-dependent dot-product scoring with head-level soft voting. A paged storage layout enables efficient non-contiguous token retrieval, and a selection cache avoids redundant scoring when consecutive queries are similar. |

All implementations are in pure PyTorch without custom CUDA or Triton kernels. This design choice is intentional: it ensures the benchmark reflects the algorithmic characteristics of each method rather than the quality of a particular kernel implementation.

**Fairness constraints enforced across all runs:**
1. Identical model weights loaded once per GPU container
2. Same prompts in the same order with identical random seeds (torch.manual_seed(42))
3. torch.cuda.reset_peak_memory_stats() called before every individual run
4. torch.cuda.synchronize() called around all timing measurements
5. FP16 precision, model.eval(), torch.no_grad() throughout
6. Fixed max_new_tokens=200, batch_size=1

**Profiling / Performance Analysis Tools:**
- PyTorch CUDA memory tracking (torch.cuda.max_memory_allocated, torch.cuda.reset_peak_memory_stats)
- Wall-clock timing with time.perf_counter() paired with torch.cuda.synchronize() for accurate GPU timing
- Custom per-method KV cache size accounting (quantized bytes at actual bit-width, retained FP16 tensors, paged buffer sizes)
- Per-method perplexity measurement through modified KV caches on WikiText-103
- Weights and Biases (wandb) for experiment tracking, metric logging, and plot hosting

---

## Current Progress

**GitHub Project:** https://github.com/rishika1099/kv-cache-benchmark

**W&B Project:** https://wandb.ai/rm4318-columbia-university/kv-cache-benchmark

### What Has Been Completed

We have completed a full benchmark run covering 7 hyperparameter configurations (1 baseline + 2 KIVI + 3 TopK variants + 1 PPL reference) across synthetic prompts, LongBench tasks, and WikiText perplexity evaluation. All runs executed on Modal cloud infrastructure using NVIDIA A100-80GB GPUs, with method configs running in parallel.

The benchmark framework itself is fully operational: prompt generation, method dispatch, metric collection, LongBench scoring (token-level F1 and ROUGE-L), perplexity evaluation, result merging, and automated plot generation all function end-to-end.

### Results Obtained

#### Throughput, Memory, and Compression

| Method | Config | Avg KV Cache (MB) | Compression | Throughput (TPS) | Perplexity |
|--------|--------|-------------------|-------------|-----------------|------------|
| Baseline | FP16 full cache | 1990.7 | 1.00x | 36.1 | 6.94 |
| KIVI | 4-bit, group_size=32, residual=128 | 626.4 | 2.90x | 8.7 | 6.93 |
| KIVI | 2-bit, group_size=32, residual=128 | 386.5 | 4.47x | 8.4 | 13.70 |
| TopK | K=512, n_sink=128, n_local=512 | 1990.7 | 1.00x | 27.9 | 6.94 |
| TopK | K=1024, n_sink=128, n_local=512 | 1990.7 | 1.00x | 27.3 | 6.94 |
| TopK | K=2048, n_sink=128, n_local=512 | 1990.7 | 1.00x | 32.3 | 6.94 |

#### Memory Scaling with Sequence Length (KIVI 4-bit)

| Sequence Length | Baseline KV (MB) | KIVI 4-bit KV (MB) | Compression |
|----------------|-------------------|---------------------|-------------|
| 512 | 380.1 | 160.9 | 2.36x |
| 2048 | 1185.4 | 393.7 | 3.01x |
| 8192 | 4406.6 | 1324.8 | 3.33x |

#### LongBench Task Scores (F1 / ROUGE-L)

| Task | Baseline | KIVI 4-bit | KIVI 2-bit | TopK (K=512) | TopK (K=2048) |
|------|----------|------------|------------|--------------|---------------|
| qasper | 0.051 | 0.050 | 0.050 | 0.051 | 0.051 |
| multifieldqa_en | 0.049 | 0.050 | 0.050 | 0.047 | 0.047 |
| hotpotqa | 0.012 | 0.013 | 0.013 | 0.012 | 0.012 |
| 2wikimqa | 0.020 | 0.033 | 0.033 | 0.020 | 0.020 |
| gov_report | 0.154 | 0.139 | 0.139 | 0.148 | 0.148 |
| qmsum | 0.074 | 0.052 | 0.052 | 0.067 | 0.067 |

### Analysis and Interpretation

**KIVI achieves meaningful memory reduction with a quality-dependent tradeoff.** At 4-bit quantization with block-wise group quantization (group_size=32), KIVI compresses the KV cache by 2.9x on average while preserving perplexity almost exactly (6.93 vs 6.94 baseline). The compression ratio improves with longer sequences because the fixed-size FP16 residual window becomes a smaller fraction of the total cache. At 8192 tokens, compression reaches 3.33x. However, dropping to 2-bit quantization significantly degrades perplexity to 13.70, indicating that 2 bits per element is insufficient to faithfully represent the key-value distribution for this model. The block-wise quantization approach (each group of 32 tokens quantized independently with its own scale and zero-point) is important here: it prevents early tokens from being repeatedly re-quantized as new tokens arrive, giving O(1) amortized cost per new token rather than the O(N) cost of global re-quantization.

**TopK (TokenSelect) preserves quality identically to baseline but does not reduce memory.** All three TopK configurations produce perplexity of 6.94, matching baseline exactly. This is expected by design: TopK stores the complete KV cache and only sparsifies the attention computation at each decode step by selecting the most relevant K tokens per head. The full cache remains in GPU memory, so the compression ratio is 1.0x. The benefit of TopK is computational: by attending to fewer tokens, each decode step involves less attention computation. Our K=2048 configuration achieves 32.3 TPS (89% of baseline's 36.1 TPS), while K=512 runs at 27.9 TPS. The counterintuitive result that smaller K is slower is due to the overhead of the selection mechanism itself (scoring, paged retrieval, head voting), which dominates when K is small relative to the full sequence. At longer sequences where full attention becomes the bottleneck, smaller K values would show greater speedup.

**Throughput overhead of pure PyTorch KIVI is substantial.** KIVI runs at 8.5 TPS, roughly 4x slower than baseline. This is because each decode step requires dequantizing the cached keys and values from INT4/INT2 back to FP16, performing the standard attention computation, and then quantizing the new token's key-value pair for storage. In pure PyTorch, these quantization and dequantization operations are element-wise Python loops over tensor blocks. The original KIVI paper achieves near-baseline throughput using fused Triton kernels that perform quantized attention directly without full dequantization.

**LongBench scores are consistent across methods.** The task-level F1 and ROUGE-L scores show only minor variation between methods, suggesting that for these particular tasks and sequence lengths, the KV cache modifications do not substantially alter the model's downstream generation quality. The scores themselves are modest (highest is gov_report ROUGE-L at 0.15), which reflects the difficulty of these tasks for a 7B parameter model generating with greedy decoding and a 200-token limit.

#### Plots Generated

Five publication-ready plots (PNG 300dpi and PDF vector) are available on the wandb project page:
1. **Memory vs Quality Tradeoff**: Perplexity plotted against compression ratio, showing KIVI 4-bit as the favorable operating point
2. **Throughput vs Compression**: Decode throughput plotted against compression ratio
3. **Memory Scaling**: Peak GPU memory versus sequence length for each method
4. **Latency Breakdown**: Time-to-first-token and per-token latency grouped by sequence length
5. **LongBench Radar**: Per-task scores on a radar chart comparing all methods

---

## Work in Progress

### Custom CUDA Kernel Integration

The most significant gap in our current results is the throughput penalty of pure PyTorch implementations, particularly for KIVI. The original KIVI paper implements fused quantized attention kernels in Triton that avoid the dequantize-attend-requantize overhead entirely. Similarly, the TokenSelect paper provides custom CUDA kernels for paged sparse attention that are critical to achieving the speedups reported in their evaluation.

We plan to integrate these original kernel implementations for the final submission. Specifically:
- **KIVI**: Fused INT4/INT2 attention kernels from the authors' Triton implementation, which perform quantized matrix multiplication directly
- **TopK (TokenSelect)**: Paged dot-product and sparse gather kernels from the authors' CUDA implementation, which avoid the overhead of our current PyTorch-level paging

Our initial decision to implement everything in pure PyTorch within a single unified repository was deliberate: it provides a fair algorithmic comparison where every method pays the same Python overhead tax. If we are unable to get the original CUDA kernels working within our framework, we fall back to comparing the best available implementation of each method (the authors' official code), but we will ensure identical prompts, model weights, seeds, and hardware are used to keep the comparison as fair as possible.

### Additional Evaluation Dimensions

- **torch.profiler integration**: CUDA kernel-level traces to identify whether bottlenecks are in memory copies, arithmetic, or synchronization
- **Batch size scaling**: Extend from batch_size=1 to batch_size=[1, 4, 8, 16] to study how KV cache optimization interacts with batched inference, which is the dominant serving pattern in production

---

## Blockers and Limitations

1. **Pure PyTorch overhead distorts throughput comparisons.** KIVI quantization and TopK token selection incur significant overhead when implemented as Python-level tensor operations. KIVI runs at roughly 4x slower than baseline, and TopK's selection mechanism costs 10-25% throughput, whereas the original papers report near-baseline speeds using custom kernels. Our current numbers reflect algorithmic overhead in Python, not the achievable performance of these techniques. Integrating original CUDA/Triton kernels is the primary remaining work item.

2. **TopK does not reduce memory in its current form.** Because TopK (TokenSelect) maintains the full KV cache and only sparsifies the attention computation, the measured compression ratio is 1.0x. The original paper's memory savings come from a page-level eviction policy that discards pages with consistently low scores, which we have not yet implemented. This means our current TopK results demonstrate the compute-quality tradeoff but not the memory-quality tradeoff.

3. **SnapKV OOM at long sequences.** SnapKV requires output_attentions=True during prefill, which materializes the full (seq_len x seq_len) attention weight matrix for all 32 layers. At 8192 tokens this exceeds the A100's 80 GB capacity. Chunked attention scoring or Flash Attention weight extraction would be needed to support longer contexts.

4. **Single GPU constraint.** All experiments run on one A100-80GB. We do not evaluate multi-GPU KV cache sharding or tensor parallel configurations, which are common in production serving setups.

5. **Greedy decoding only.** We use argmax decoding for reproducibility. Sampling-based generation (temperature, top-p) could interact differently with KV cache modifications, as the altered attention distributions may shift token probabilities in ways that compound over the generated sequence.

6. **LongBench scores are modest overall.** The F1 and ROUGE-L numbers are low across all methods, including baseline. This is partly a function of the model size (7B), the 200-token generation limit, and greedy decoding. For the final report, we may extend the generation budget or evaluate with a larger model to obtain more discriminative task scores.
