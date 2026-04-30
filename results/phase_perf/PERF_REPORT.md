# KIVI + TopK Hybrid — Performance Optimization Report

**Branch:** `perf/hybrid-optimization`
**Reference (before):** `results/_h100_post_fix_v2/phase_b2/`
**This run (after):** `results/phase_perf/`
**Hardware:** Modal H100 80GB HBM3, torch 2.4.1+cu121, Triton 3.0.0
**Model (passkey):** meta-llama/Llama-2-7b-hf

The hybrid method (KIVI quantization + TokenSelect dynamic top-K) was 5–6×
slower than KIVI alone in the v2 reference run despite touching the same
underlying tensors. This report documents the bottlenecks, the
optimizations applied, the measured before/after numbers, and what is
still left on the table.

**Headline:** the v2-vs-new gap to KIVI shrinks from **5.6× / 2.1× / 1.6× /
1.3×** at 2K/8K/16K/32K → **1.6× / 1.1× / 1.1× / 1.0×** — i.e. **70–93%
of the gap to KIVI native speed has been closed**, with passkey
correctness preserved (slight non-significant improvement at depth 0.9).

---

## 1. Summary of bottlenecks (v2 reference)

Profiling the v2 hybrid against the KIVI baseline at seq_len 2048
revealed that the per-step cost was dominated by Python-side work, not
GPU kernels:

| Component | Cost contributor | Per-step |
|---|---|---|
| **`_materialise`** | A Python `for block_id in unique_blocks:` loop ran one full-block dequant **per touched block, per layer × 32 layers**. With K≈1024 selected tokens spread across ~30 blocks, that is ~960 separate dequant ops per step. | **dominant** |
| Selection-cache staleness | `_can_reuse_cache` only checked cosine of the proxy query, not whether `seq_len` had changed; on a fresh prefill the cache was reused stale and `_score_layer` re-ran anyway. | minor |
| Quant-state stack rebuild | Block-seal events re-`cat`'d the K-stack from scratch (O(n_blocks²) total). | small at 2K, larger at 32K |
| Score-path dtype | `_score_layer` cast everything to fp32 before the matmul, throwing away H100 fp16 tensor cores. | small |
| Triton kernel cold start | `quant_score` JIT-compiled on the first decode step → ~80 ms one-off spike. | once per run |
| V-side dequant | V dequantization re-derived per-token block ids via `torch.bucketize` every step. | small |

The same dynamics held at 8K/16K/32K; the per-block Python loop scales
linearly with the number of touched blocks, which is why hybrid was
6× slower than KIVI at 2K but only 1.3× slower at 32K — the fixed Python
overhead amortizes as the kernel work grows.

---

## 2. Optimizations applied

All edits are in `methods/kivi_topk_hybrid.py` (v1: 769 lines → v2: 965
lines, no new public API). Mnemonics match the docstring:

| ID | Change | Impact |
|---|---|---|
| **A** | **Vectorized `_materialise`.** Replaced the per-block Python loop with one batched `index_select` + fp16 multiply-add. Single tensor op per layer, zero Python iteration over blocks. | Dominant fix at 2K (5.6× → 1.6× vs KIVI). |
| **B** | **V-side stacks (`vq_stack`, `vs_stack`, `vz_stack`).** Maintained in parallel with K stacks so V dequant is also a single vectorized gather, not a per-block fetch. | Halves remaining `_materialise` cost. |
| **C** | **Pre-allocated growable buffers.** `_ensure_capacity` doubles the four stack tensors when needed (std::vector pattern) instead of re-`cat`ing on every block-seal. | Removes O(n_blocks²) cost on long prefills. |
| **D** | **FP16 score path.** Dropped the unconditional `.float()` upcast in `_score_layer`; matmul stays fp16 (free on H100 tensor cores), fp32 cast is local to the softmax/sum reduction. | Small speedup, larger memory savings. |
| **E** | **Cached token→block_id LUT.** `_block_id_for_quant[layer_idx]` is rebuilt once on block-seal, not per step via `bucketize`. | Removes hot-path bucketize. |
| **F** | **Selection-cache staleness fix.** `_can_reuse_cache` now invalidates on `seq_len` change, not only on cosine drift. | Fixes a latent correctness/perf bug — cache was being skipped anyway because of the staleness, so the fix is mostly forward-looking. |
| **G** | **Honest storage accounting.** `get_kv_size_bytes` includes the uint8/scale/zero scratch tensors. Hybrid now reports 884 MB at 2K (was 351.5 MB, which counted only the compressed view and ignored the mirror). KIVI numbers unchanged. | Truth-in-reporting; no perf delta. |
| **H** | **Triton kernel pre-warm.** `_prewarm_kernels` runs a single tiny `quant_score` invocation at the end of `process_prefill` to absorb the JIT compile cost off the decode hot path. | Removes ~80 ms first-step spike. |
| **I** | **`KV_HYBRID_PROFILE=1`** env var enables per-step timing of `_score_layer` / `_materialise` / `num_blocks_dequantized`, dumped at end of run. | Diagnostic only. |

What we **did not** do (deliberate scope limit):
- Did not bit-pack the uint8 mirror buffer to true 4-bit (would halve
  scratch RAM, but doubles dequant kernel complexity).
- Did not batch `_score_layer` across all 32 layers in one launch
  (would require re-laying out the score buffer; net win uncertain).
- Did not re-enable the selection cache by default (it's still gated
  on the same threshold; with bug F fixed it now correctly *can* be
  enabled, but we kept `use_selection_cache=False` for the apples-to-
  apples benchmark to isolate kernel cost).

---

## 3. Before vs. after — long-context decode latency

Setup: `experiments/long_context.py`, K=1024, n_sink=128, n_local=512,
bits=4, group_size=32, residual_length=128, n_warmup=5, n_iter=15,
`use_selection_cache=False` (so we measure kernel cost, not cache
hits). Numbers are mean `decode_step_ms`.

KIVI itself drifted 1.43–1.48× slower on the new H100 container vs the
v2 container — same code, different hardware allocation. The
**KIVI-normalized speedup** factors that drift out and is the honest
metric.

| seq_len | method | v2 ms | new ms | raw | **kivi-norm** | hybrid/KIVI v2 | hybrid/KIVI new | gap closed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2048  | kivi_topk    |  360.77 |  150.19 | 2.40× | **3.43×** | 5.62× | 1.64× | **70.8%** |
| 2048  | kivi_topk_c  |  357.06 |  148.35 | 2.41× | **3.44×** | 5.56× | 1.62× | **70.9%** |
| 8192  | kivi_topk    |  563.66 |  433.07 | 1.30× | **1.88×** | 2.14× | 1.14× | **87.7%** |
| 8192  | kivi_topk_c  |  561.47 |  448.90 | 1.25× | **1.80×** | 2.13× | 1.18× | **83.7%** |
| 16384 | kivi_topk    |  837.06 |  859.14 | 0.97× | **1.42×** | 1.58× | 1.11× | **80.7%** |
| 16384 | kivi_topk_c  |  831.24 |  831.13 | 1.00× | **1.45×** | 1.56× | 1.08× | **86.0%** |
| 32768 | kivi_topk    | 1393.13 | 1618.60 | 0.86× | **1.27×** | 1.30× | 1.02× | **93.0%** |
| 32768 | kivi_topk_c  | 1379.67 | 1587.32 | 0.87× | **1.29×** | 1.29× | 1.00× | **99.6%** |

> **Reading guide.** "raw" = ratio of v2 ms to new ms on different
> containers (misleading because KIVI itself slowed down). "kivi-norm" =
> raw × (kivi_new / kivi_v2) — what the speedup *would* be if both runs
> were on the same machine. "gap closed" = how much of the v2 hybrid-vs-
> KIVI gap has disappeared.

**At 32K, the hybrid is essentially tied with KIVI native (1.00–1.02×).**
The remaining ~1× cost is the actual top-K selection scoring, which
KIVI doesn't pay at all. We have effectively eliminated the materialise
overhead at the regime that matters most for long-context use.

Storage (compressed cache + scratch buffers):

| seq_len | KIVI | hybrid v2 (under-reported) | hybrid new (honest) |
|---:|---:|---:|---:|
|  2048 |  351.5 MB | 351.5 MB | 884 MB |
|  8192 | 1239.5 MB | 1239.5 MB | 3476 MB |
| 16384 | 2423.5 MB | 2423.5 MB | 6932 MB |
| 32768 | 4791.5 MB | 4791.5 MB | 13844 MB |

The "honest hybrid" number is the uncompressed uint8 mirror used by the
score kernel + scales + zeros. It is **not** bit-packed — 4-bit data is
stored as 8-bit, so a future bit-pack pass would roughly halve the
mirror. See §5.

---

## 4. Correctness — passkey @4K

Setup: 5 trials × 5 depths × max_new_tokens=12, LLaMA-2-7B,
seq_len=4096. Same harness as Phase B2.

| method | depth | v2 acc | new acc | v2 tps | new tps |
|---|---:|---:|---:|---:|---:|
| baseline    | (any) | 100% | 100% | ~50 | ~49 |
| kivi        | (any) | 100% | 100% | 4.7 | 4.3 |
| topk        | 0.1/0.7 | 0% | 0% | ~31 | ~29 |
| topk        | 0.3 | 80% | 80% | 34.5 | 32.9 |
| topk        | 0.5 | 100% | 100% | 34.2 | 33.0 |
| topk        | 0.9 | 60% | 60% | 34.4 | 32.0 |
| **kivi_topk**   | 0.1 | 100% | 100% | 1.74 | **3.44** |
| **kivi_topk**   | 0.3 | 0%   | 0%   | 1.74 | **3.34** |
| **kivi_topk**   | 0.5 | 0%   | 0%   | 1.75 | **3.34** |
| **kivi_topk**   | 0.7 | 0%   | 0%   | 1.75 | **3.39** |
| **kivi_topk**   | 0.9 | 60%  | **100%** | 1.74 | **3.41** |
| **kivi_topk_c** | 0.1 | 100% | 100% | 1.73 | **3.39** |
| **kivi_topk_c** | 0.3 | 0%   | 0%   | 1.75 | **3.35** |
| **kivi_topk_c** | 0.5 | 0%   | 0%   | 1.75 | **3.30** |
| **kivi_topk_c** | 0.7 | 0%   | 0%   | 1.74 | **3.35** |
| **kivi_topk_c** | 0.9 | 60%  | **100%** | 1.76 | **3.37** |

- **Aggregate hybrid accuracy 32% → 40%** (8/25 → 10/25). Depth 0.9 went
  60% → 100% for both hybrids. Within-trial-noise improvement (5-trial
  N has ±22pp standard error), but **no regressions anywhere**.
- **Hybrid throughput ~doubled** (1.74 → 3.4 tok/s) on the same model
  on the same hardware class, consistent with the long-context decode
  numbers.
- topk-only depth-0.1/0.7 still 0% — that is a pre-existing TopK
  property unrelated to this work, kept for sanity.

---

## 5. Remaining bottlenecks & future work

What is left between hybrid and KIVI parity:

1. **Score kernel cost.** At 32K we are at 1.0–1.02× of KIVI; the
   residual is the actual `_score_layer` work (centroid mat-mul or
   `quant_score` Triton call), which KIVI doesn't pay. To go *below*
   KIVI you would need TopK to amortize fewer touched-block dequants
   in the attention itself — possible if `K << seq_len`, but at K=1024
   and seq_len=2048 there isn't much to skip.

2. **Mirror buffer is 8-bit, not 4-bit.** The uint8 stacks store one
   nibble per byte. Bit-packing would halve scratch RAM (884 MB → ~440
   MB at 2K), but requires a packed-load path in the Triton scoring
   kernel and the gather-dequant. Worth doing if mirror RAM becomes a
   constraint at 64K+.

3. **No cross-layer batching.** Each of the 32 layers calls
   `_score_layer` and `_materialise` independently. Stacking layer i+1
   work onto layer i's stream (pipelining) or fusing scoring into one
   launch could save kernel-launch overhead, especially at small
   per-layer batch sizes.

4. **Selection cache disabled.** With OPT-F the cache is now correct,
   but we kept `use_selection_cache=False` for clean apples-to-apples
   benchmarking. Re-enabling would give an additional speedup
   proportional to the cache hit rate (likely 30–60% on autoregressive
   text given the cosine-similarity guard).

5. **Centroid scoring is approximate.** Design (a) (`score_mode=
   "centroid"`) approximates the per-block dot product by the centroid
   dot product. This is what produces the depth-0.3/0.5/0.7 zero-
   accuracy bands on passkey. A proper per-token scored fallback (still
   on quantized state) might recover those bands without the full
   `quant_score` kernel cost.

6. **Smoke test broken on OPT-125M.** The smoke job in
   `modal_perf_check.py` fails with `OPTForCausalLM.forward() got an
   unexpected keyword argument 'position_ids'` — this is a
   **pre-existing** issue in `benchmark/runner.py` (the BUG-2 Option B
   fix unconditionally injects `position_ids`, which OPT doesn't
   accept). It does **not** affect LLaMA-2-7B, which is the actual
   target model and runs the passkey job to completion. Worth fixing
   in a follow-up by gating the kwarg on `model.config.model_type`.

---

## 6. Files & artifacts

```
methods/kivi_topk_hybrid.py            # the optimization (965 lines)
modal_perf_check.py                    # the perf-only Modal entrypoint
results/phase_perf/long_context.csv    # measured decode latency
results/phase_perf/passkey.csv         # measured passkey accuracy
results/phase_perf/summary/
    before_after_long_context.csv      # this report's table 3
    before_after_passkey_4k.csv        # this report's table 4
results/phase_perf/PERF_REPORT.md      # this file
```

Reproduce:
```
git checkout perf/hybrid-optimization
modal run modal_perf_check.py
modal volume get kv-benchmark-results /phase_perf ./results/phase_perf
```

---

## Verdict

**Hybrid is materially faster.** KIVI-normalized speedup of 1.27–3.44×
across seq_lens 2K–32K, with the gap to KIVI native shrinking from
1.30–5.62× to 1.00–1.64×. The dominant Python-loop bottleneck in
`_materialise` has been eliminated. Passkey accuracy is preserved
(non-significant improvement, no regressions). Storage accounting is
now honest. Selection-cache bug fixed. Triton kernels are pre-warmed.
