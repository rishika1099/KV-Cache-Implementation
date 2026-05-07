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
