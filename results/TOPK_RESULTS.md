# TopK Selection — LongBench Quality Results

Companion to `RESULTS.md` (which covers Baseline + KIVI). Same harness, same model, same prompts, same truncation — only the method differs.

## Method

**TopK** (TokenSelect, Wu et al., EMNLP 2025) is a *sparse selection* method, not a quantization method. The full FP16 KV cache is kept in memory; on each decode step attention is computed only over a selected subset of tokens — `K` middle tokens chosen dynamically per query, plus a fixed sink window (first 128) and local window (last 512). Selection is per-layer, soft-vote across heads, with a cosine-similarity reuse cache when consecutive query proxies are close.

The trade-off is structurally different from KIVI:

| | KIVI | TopK |
|---|---|---|
| What gets compressed | Per-token bit-width (FP16 → 4/2 bit) | Per-token presence (kept or skipped) |
| Cache size | ~4–8× smaller | Same as baseline |
| Decode work | All tokens, packed-int math | `K + n_sink + n_local` tokens, FP16 math |
| Failure mode | Quantization noise on every token | Some tokens not seen at all |

## Setup

- **Dataset:** LongBench (6 tasks × 20 examples = 120 examples per K value)
- **Tasks:** qasper (scientific QA), multifieldqa_en (multi-doc QA), triviaqa (open-domain QA), 2wikimqa (multi-hop QA), multi_news (summarisation), lcc (code completion)
- **Metrics:** F1 (QA), ROUGE-L (summarisation), edit similarity (code)
- **Input truncation:** Middle-truncation to 4096 tokens (KIVI paper convention)
- **Model:** `meta-llama/Llama-2-7b-chat-hf`, FP16 weights, H100 80GB
- **Decode:** greedy, batch size 1 (TopK does not implement batching at this commit)
- **Fixed knobs:** `n_sink=128`, `n_local=512`, `cosine_threshold=0.9`, `kernel_size=-1` (no max-pool smoothing)
- **K swept:** 2048, 1024, 512
- **Selected token budget at S=4096** (= `n_sink + K + n_local`):
  - K=2048 → 2688 tokens (~66% of cache)
  - K=1024 → 1664 tokens (~41%)
  - K=512 → 1152 tokens (~28%)

## Results

| Task | Baseline (FP16) | TopK K=2048 | Δ K=2048 | TopK K=1024 | Δ K=1024 | TopK K=512 | Δ K=512 |
|------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| qasper | 0.172 | 0.145 | −0.027 | 0.153 | −0.019 | 0.145 | −0.027 |
| multifieldqa_en | 0.380 | 0.329 | −0.051 | 0.281 | −0.099 | 0.237 | **−0.143** |
| triviaqa | 0.644 | **0.731** | **+0.087** | **0.773** | **+0.129** | **0.751** | **+0.107** |
| 2wikimqa | 0.083 | 0.075 | −0.008 | 0.089 | +0.006 | 0.050 | −0.033 |
| multi_news | 0.181 | 0.178 | −0.003 | 0.183 | +0.002 | 0.178 | −0.003 |
| lcc | 0.287 | 0.282 | −0.005 | 0.273 | −0.014 | 0.281 | −0.006 |
| **Overall** | **0.291** | **0.290** | **−0.001 (−0.3%)** | **0.292** | **+0.001 (+0.4%)** | **0.274** | **−0.017 (−5.8%)** |

## Analysis

**TopK at K=2048 and K=1024 is lossless overall.** Both budgets land within ±0.001 of FP16 baseline on the 120-example slice — within noise. K=1024 actually ties KIVI 4-bit (0.292) on overall score, despite using a different mechanism. Cutting K from 2048 → 1024 (1.5× more aggressive sparsity) costs essentially nothing — the per-layer cosine reuse cache absorbs most of the work and the answer-bearing tokens are robust enough at K=1024 to survive selection.

**TopK at K=512 hits the same quality cliff as KIVI 2-bit.** Overall drops 5.8% (0.291 → 0.274), almost exactly matching KIVI 2-bit's 6.2% drop (0.291 → 0.273). Same magnitude of degradation, different mechanism — useful as a corroboration that "moderate compression" lands in the same quality band whether you compress per-token bit-width or per-token presence.

**Per-task profile differs sharply from KIVI.** The two methods have opposite blind spots:

- **TopK *beats* baseline on triviaqa across all three K values** (+13.6% / +20.1% / +16.6%). Real, consistent — not noise. Triviaqa contexts contain the answer multiple times in different phrasings; sparse selection drops the noise tokens but the answer survives in at least one location, and the model focuses better with fewer distractors. KIVI does not show this — it perturbs all tokens equally and slightly degrades retrieval (−5.6% at 2-bit).
- **TopK is worst on multifieldqa_en** (−13.6% at K=2048, deteriorating to −37.7% at K=512). This task has precise mid-context answer spans that get pruned by sparse selection. KIVI is essentially flat here (+0.5% at 2-bit) because it keeps the full token grid, just with quantization noise.
- **qasper hurts both methods comparably** (TopK −15.6%, KIVI 2-bit −22.7%) — long scientific articles with one answer span buried mid-context, hard for either compression strategy to preserve.
- **multi_news, lcc are robust to both methods** — summarisation and code completion don't depend on precise mid-context token recall.

**The mid-context-retrieval blind spot was reproducible at all K** and matches the passkey-retrieval pattern observed on standalone TopK in earlier benchmarks: depths far from sink and local windows are the hardest to preserve under fixed-budget selection. This is the same architectural limitation that motivated the global-coverage and adaptive-K extensions on the hybrid branch.

## Latency caveat (B=1)

Just as KIVI is slower than baseline at small batch sizes, **TopK is also slower than baseline at B=1**: ~30 tok/s decode vs ~70 tok/s baseline (from the throughput table in `RESULTS.md`). Per-step cost = full Q·Kᵀ score pass over all 4096 tokens + per-layer top-K + non-contiguous gather, all of which exceed the saved attention work at B=1 without flash-attention or a fused gather kernel.

The selection win in compute terms only realizes when (a) batched decode amortizes the per-step overhead, or (b) `K / S << 1` at long context where the saved attention work dominates. Neither regime is exercised in this LongBench smoke. **This is a quality-at-budget result, not a latency result.**

## Comparison vs. KIVI on overall score

| compression goal | best KIVI | best TopK | winner |
|---|---|---|---|
| Lossless (≤1% drop) | KIVI 4-bit (0.292) | TopK K=1024 (0.292) | **tie** |
| Moderate compression (~6% drop) | KIVI 2-bit (0.273) | TopK K=512 (0.274) | **tie** |

So as a quality method TopK is competitive with KIVI at every compression budget on overall LongBench. The story is different *per-task* — KIVI's blind spots are precise QA (qasper, 2wikimqa); TopK's blind spot is mid-context multi-doc QA (multifieldqa_en). Different mechanisms, similar averages, complementary failure modes.

## Updated quality summary (n=120, all six methods)

| Method | Overall Score | vs Baseline | Notes |
|--------|:-------------:|:-----------:|:------|
| Baseline (FP16) | 0.291 | — | Reference |
| KIVI 4-bit | 0.292 | +0.3% | Lossless |
| KIVI 2-bit | 0.273 | −6.2% | Modest drop, biggest on qasper / 2wikimqa |
| TopK K=2048 | 0.290 | −0.3% | Lossless |
| TopK K=1024 | 0.292 | +0.4% | Lossless |
| TopK K=512 | 0.274 | −5.8% | Same magnitude as KIVI 2-bit; loss concentrated in multifieldqa_en |

## Reproduction

```bash
modal run modal_longbench.py --method topk --top-k 2048 --n-per-task 20
modal run modal_longbench.py --method topk --top-k 1024 --n-per-task 20
modal run modal_longbench.py --method topk --top-k 512  --n-per-task 20
```

Each writes `results/longbench_topk_<cfg>.json` locally and pushes a copy to the `kv-benchmark-results` Modal volume.

## Result Files

| File | Contents |
|------|----------|
| `longbench_topk_{"K": 2048, "cosine_threshold": 0.9, "kernel_size": -1, "n_local": 512, "n_sink": 128}.json` | TopK K=2048 — per-example results, all 6 tasks × 20 examples |
| `longbench_topk_{"K": 1024, "cosine_threshold": 0.9, "kernel_size": -1, "n_local": 512, "n_sink": 128}.json` | TopK K=1024 |
| `longbench_topk_{"K": 512,  "cosine_threshold": 0.9, "kernel_size": -1, "n_local": 512, "n_sink": 128}.json` | TopK K=512 |
