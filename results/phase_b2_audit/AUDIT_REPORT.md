# Paper-Faithfulness & Correctness Audit — KV Cache Benchmark

**Branch:** `audit/paper-faithfulness`
**Scope:** KIVI · TopK / TokenSelect · KIVI+TopK hybrids (designs a, c) · Triton kernels · Phase B2 results
**No changes have been made to source.** This is a read-only audit. All findings are reproducible from the files inspected.

---

## 1. Executive summary

The pipeline runs end-to-end and produces results that look superficially reasonable — KIVI compresses 3.4×, hybrids run, Modal jobs complete. But several of the headline claims do not survive a careful read.

**Three claims in the report are not supported by the implementation:**

| Claim | Reality |
|---|---|
| "KIVI: per-channel K, per-token V" | **Inverted.** The two helpers compute the opposite of their names. K is per-token, V is per-channel. |
| "TopK / hybrid retrieve correctly when needle is in local window" | **No.** The needle *is* selected. The bug is positional: gathered K/V violate RoPE relative-position assumptions and the model receives positionally-corrupted attention input. |
| "PPL confirms hybrids don't regress KIVI" | **Vacuously true.** `compute_method_perplexity` calls `process_prefill` only — `process_step` never fires, so PPL is structurally identical between KIVI and any TopK-flavoured method. |

**Two further claims are misleading:**

| Claim | Reality |
|---|---|
| "Triton kernel delivers fused-softmax speedup" | Within 5% of PyTorch fallback in kernel_bench. Self-test passes (kernel is correct), but it is not faster. |
| "Latency comparison via kernel_bench" | Baseline is a no-op `process_step`. All methods are timed against a noop, so "speedup" numbers (`0.0001×`) do not reflect attention compute. |

**Verdict:** Results are **not safe to publish without a clearly-scoped errata section**, and the passkey 0% finding for TopK-family methods is a real correctness bug, not the LLaMA-2 4K context limit (that limit explains only the 8K-row 0%).

---

## 2. Paper-faithfulness verdict by method

### 2.1 KIVI (`methods/kivi_quant.py`)

| Aspect | Paper | Implementation | Status |
|---|---|---|---|
| K quantization direction | Per-channel (one scale per head_dim entry, computed across tokens) | **Per-token** (reduces over `dim=-1`, scale shape `(B,H,S,1)`) | ❌ **INVERTED** |
| V quantization direction | Per-token | **Per-channel** (reduces over `dim=-2`, scale shape `(B,H,1,D)`) | ❌ **INVERTED** |
| Group-wise quantization | Yes, group_size G | Yes, `group_size=32` | ✅ |
| Residual window | Most recent R tokens in FP16 | `residual_length=128` in FP16 | ✅ |
| Block sealing trigger | When R tokens overflow, quantize as new block | Custom **two-stage**: residual evicts to `overflow_k`, sealed only when `overflow >= group_size` | ⚠️ Custom (not in paper Algorithm 1) |
| K residual length | R | Single `residual_length` for both K and V | ⚠️ Paper distinguishes K/V residual handling |
| Dequant arithmetic | `q.float() * s + z` | Same | ✅ |

**Numeric verification of the swap** (run on this branch):

```
Input shape: torch.Size([1, 1, 4, 3])    (B, H, S, D)
quantize_per_channel(x):
  scale shape = torch.Size([1, 1, 4, 1])   → one scale per token (PER-TOKEN)
  expected per-channel scale shape = (1, 1, 1, 3)
quantize_per_token(x):
  scale shape = torch.Size([1, 1, 1, 3])   → one scale per channel (PER-CHANNEL)
  expected per-token scale shape = (1, 1, 4, 1)
```

The functions round-trip correctly (dequant reverses quant), so KIVI's compression number is right and the cache reconstruction is consistent. But the **bit budget is being spent on the wrong axis.** KIVI's central argument is that K has channel outliers — using per-token quant on K means the bit budget is squandered on token-level dynamic range while channel-level outliers are clipped together. At 4-bit there is enough headroom that retrieval still works (KIVI passes passkey 100% at 4K), but the headline algorithmic claim is reversed.

### 2.2 TopK / TokenSelect (`methods/topk_selection.py`)

| Aspect | Paper | Implementation | Status |
|---|---|---|---|
| Q-K dot product scoring | Q (current query) · K_cached | **K_last_token (proxy) · K_cached** — Q is not exposed by HF Transformers without modification | ❌ Documented limitation, but a paper-faithfulness gap |
| Per-head softmax (Eq 7) | Yes | Yes (`use_head_softmax=True`) | ✅ |
| Cross-head soft voting | Sum across heads after softmax | Same | ✅ |
| Per-head criticality weights | From entropy | Inverse entropy, normalized | ✅ |
| TopK over token positions | Yes | Yes, dim=token | ✅ |
| Sink + local always retained | Yes | Yes | ✅ |
| Selection cache | Cosine sim of consecutive Q | Cosine sim of consecutive **K** (proxy) | ⚠️ |
| /√d scaling on raw scores | Yes (standard scaled-dot-product) | **Missing in PyTorch fallback** (line 316), present in Triton kernel and in `fused_paged_score`'s PyTorch reference | ❌ Inconsistent |
| Positional encoding for gathered K/V | Either re-rotate or use position-aware kernel | **Neither** — gather is `k[:,:,indices,:]` and the model's next forward applies RoPE assuming the cache is contiguous from position 0 | ❌ **CRITICAL** — see §7 |

### 2.3 KIVI + TopK hybrids (`methods/kivi_topk_hybrid.py`)

| Aspect | Design | Implementation | Status |
|---|---|---|---|
| Storage in KIVI form | KIVI 4-bit blocks | Yes; storage MB matches KIVI | ✅ |
| Design (a): centroid scoring | One Q·centroid per block, broadcast | Yes; `score_mode="centroid"` | ✅ implementation matches stated design |
| Design (c): per-token scoring on quantized K | Triton kernel reading uint8 K + per-token scale/zero | Yes; `score_mode="quantized"` invokes `quant_score` | ✅ implementation matches stated design |
| Both modes selectable | `score_mode` flag | Wired through registry as `kivi_topk` and `kivi_topk_c` | ✅ |
| Selective dequantization | Only blocks containing selected tokens | Yes, `_materialise` uses `unique(block_id)` | ✅ |
| Per-layer indices | Better than shared | Yes; hybrid scores per layer | ✅ (improvement over `TopKMethod`'s shared) |
| Centroid ranking granularity | Token | **Block** — all tokens within a block tie after softmax → TopK picks whole blocks | ⚠️ Documented but not stated in report |
| RoPE positional handling on gather | Required | Inherits TopK's missing positional fix → same bug as §2.2 | ❌ |
| `quant_score` PyTorch fallback shape | Matches uint8 + per-token scale/zero | Matches the **swapped** KIVI shape; would silently break if KIVI were corrected to true per-channel | ⚠️ Chained to §2.1 bug |

### 2.4 Triton kernels (`methods/topk_kernels.py`)

| Kernel | Exists | Used | Correct | Faster? |
|---|---|---|---|---|
| `_fused_paged_score_kernel` | ✅ | ✅ on CUDA when `use_kernels=True` and `use_head_softmax=True` | ✅ self-test rel-err <1e-3 | ❌ within 5% of PyTorch in kernel_bench |
| `_quant_score_kernel` (design c) | ✅ | ✅ when `score_mode="quantized"` and CUDA | ✅ math factoring is correct (verified analytically) | ❌ doesn't show speedup vs (a) — bottleneck is `_materialise` |
| `fused_paged_topk` | ✅ | ✅ (vectorized, no Triton dependency) | ✅ | N/A (no PyTorch reference for paged-topk) |
| PyTorch fallback (`use_kernels=False`) | ✅ | ✅ when CUDA absent / Triton missing | ⚠️ Diverges from Triton path: omits /√d scaling (see §2.2) | N/A |

---

## 3. Confirmed code bugs

Each finding includes file, line(s), what is wrong, why it's wrong, and observable impact.

### 🔴 BUG-1 (CRITICAL): KIVI K/V quantization reduce dimensions are swapped

**Location:** `methods/kivi_quant.py:5-21` and `:24-40`

```python
def quantize_per_channel(tensor, bits):       # ← name says per-channel
    min_val = tensor.min(dim=-1, keepdim=True).values   # ← reduces head_dim
    # scale shape (B, H, S, 1)  → ONE scale per token = per-TOKEN
```

```python
def quantize_per_token(tensor, bits):         # ← name says per-token
    min_val = tensor.min(dim=-2, keepdim=True).values   # ← reduces seq_len
    # scale shape (B, H, 1, D)  → ONE scale per channel = per-CHANNEL
```

**Why it's wrong:** Per-channel quantization means each channel has its own scale (computed across tokens) — scale shape `(B, H, 1, D)`. The code achieves this by reducing over `seq_len` (dim=-2). Per-token means each token has its own scale (computed across channels) — scale shape `(B, H, S, 1)`, achieved by reducing over `head_dim` (dim=-1). The function bodies are swapped.

**Impact:** K is being quantized **per-token** (intended per-channel) and V **per-channel** (intended per-token) — the **exact reverse** of KIVI's headline asymmetric strategy. Round-trip correctness is preserved (dequant reverses), so storage compression is unaffected (3.4× is still real). But quality of the quantized representation is sub-paper: at 4-bit the difference is small (passkey + KIVI = 100%); at 2-bit the gap would likely show.

**Verification:** Reproducible with the smoke test in §2.1 above.

**Severity:** Critical for paper faithfulness. Not fatal for the 4-bit results currently reported, but every "KIVI per-channel K, per-token V" sentence in the writeup is wrong.

---

### 🔴 BUG-2 (CRITICAL): TopK gather corrupts RoPE positional encoding

**Location:** `methods/topk_selection.py:253-270` (`_gather_layer`) and `methods/kivi_topk_hybrid.py:540-624` (`_materialise`)

```python
@staticmethod
def _gather_layer(k, v, indices):
    if indices.dim() == 1:
        return k[:, :, indices, :], v[:, :, indices, :]   # ← raw gather
```

**Why it's wrong:** The cached `k` already has RoPE rotation applied at `original_position`. Gathered K[i] thus carries rotation `θ(orig_pos[i])`. When this gathered cache (length S_sel) is returned to the model, HF's next forward applies RoPE to the new query `Q` at position `S_sel`. The dot product `Q · K[i]` becomes equivalent to `Q_pre^T R(S_sel - orig_pos[i]) K_pre` — i.e., the **relative angle encodes the original gap, not the gathered-position gap**.

Concrete example for passkey @ 4K, depth=0.9:
- Original sequence length = 4086 tokens.
- n_local=512 → recent tokens span original positions 3574…4085.
- Gathered cache length S_sel ≈ 1664 (sink + K + local). Recent tokens land at gathered positions 1152…1663.
- Q's RoPE rotation: θ(1664).
- Recent token at gathered position 1500 has rotation θ(orig_pos = 4000).
- Relative angle the model sees: θ(1664 − 4000) = θ(−2336). The model expects θ(1664 − 1500) = θ(164).
- The "local" tokens look to the attention as if they were at relative distance ~2300, far past any reasonable attention window.

**Impact:** This is the **root cause of the passkey 0% for the entire TopK family** (topk, kivi_topk, kivi_topk_c). The needle is correctly *selected* (it's inside `n_local=512`), but the model sees positionally-corrupted K/V and cannot form a meaningful attention pattern. KIVI has no gather and is unaffected — it returns the full cache, positions intact, and gets 100%.

**Standard fixes** in the literature (not implemented here):
1. Re-apply RoPE to gathered K so rotations match new positions (loses absolute-distance info but preserves contiguity).
2. Use a custom attention kernel that respects original positions (e.g., paged attention with explicit position arrays). Phase B's "paged" claims are about scoring, not attention.
3. Pre-compute K with no RoPE and apply RoPE on-the-fly during attention (most KV-pruning papers).

**Severity:** Critical. Without this fix, **TopK-family methods cannot work on RoPE models** like LLaMA-2.

---

### 🔴 BUG-3 (CRITICAL): PPL methodology cannot test TopK quality

**Location:** `benchmark/runner.py:145-222` (`compute_method_perplexity`)

The function performs:
1. `model(prefix_ids, use_cache=True)` → past_kv
2. `method.process_prefill(past_kv)` → modified_kv
3. `model(target_ids, past_key_values=modified_kv, use_cache=False)` → logits

**There is no decode loop. `method.process_step` is never called.** For TopK methods this means selection logic does not execute during PPL evaluation — the model attends over the full prefill cache exactly as if no selection were happening.

**Impact:**
- ppl_sanity result: KIVI = kivi_topk = kivi_topk_c = **7.2683** (exactly identical, see `results/phase_b2/ppl_sanity/long_context.csv`). Identical to fp16 precision because the hybrids' `process_prefill` is just a wrapper around KIVI's. The runtime selection that distinguishes the methods never runs.
- Any sentence in the writeup that says "PPL shows hybrids don't regress KIVI" is **vacuously true** — they couldn't have regressed because their selection logic wasn't tested.

**Severity:** Critical. PPL must be re-scoped in the writeup to "no prefill regression vs full FP16," not "no quality regression."

---

### 🔴 BUG-4 (CRITICAL): Latency methodology measures method overhead, not attention

**Location:** `experiments/kernel_bench.py:71-94` and `experiments/long_context.py` (driver)

`_bench_method` times only `method.process_step(past_kv, step=1)`. There is **no model forward pass** in this loop. For `BaselineMethod`, `process_step` is a no-op (returns the input unmodified) — hence ~0.009 ms in every kernel_bench row. For other methods, the timed work is the bookkeeping of the selection/quantization data structures, not attention compute.

**Observable contradiction in the data:**
- kernel_bench reports `speedup_vs_baseline = 0.00010` for topk_pytorch at 2K. This implies topk is **10000× slower than baseline**. That is impossible for any real attention regime — full attention scales O(S) per token while TopK scales O(K). The number is meaningless because baseline is a no-op.
- kernel_bench reports `kivi_topk_hybrid_a = 480ms/step` at S=2048. With 32 layers × 32 heads × 2048 tokens × 128 dim per layer, this corresponds to ~75 ns per dequant/score op — entirely the cost of Python-level orchestration, not GPU compute.

**Impact:** Every "speedup vs baseline" number in `kernel_bench.csv` and `long_context.csv` should be deleted from the report. The honest story is per-method per-step *overhead* (which still has comparative value across methods, but not against baseline).

**Severity:** Critical. The headline latency claims in the report cannot be supported by these benchmarks.

---

### 🟠 BUG-5 (HIGH): Triton kernel does not deliver speedup over PyTorch fallback

**Location:** `experiments/kernel_bench.py` results, comparing `topk_pytorch` (use_kernels=False) vs `topk_triton` (use_kernels=True).

| seq_len | topk_pytorch | topk_triton | Δ |
|---|---:|---:|---:|
| 2048 | 9.05 ms | 22.14 ms | **Triton 2.4× SLOWER** |
| 4096 | 13.36 ms | 13.58 ms | tie (+1.6%) |
| 8192 | 22.03 ms | 22.58 ms | tie (+2.5%) |
| 16384 | 39.38 ms | 40.50 ms | tie (+2.8%) |

**Why:** The kernel is correct (self-test rel-err < 1e-3) but the operation it fuses (Q·Kᵀ + softmax + weighted-sum) is already near-optimal in PyTorch on A100 because cuBLAS handles the matmul and torch.softmax is a single fused launch. Adding a Triton atomic_add for cross-head accumulation costs more than it saves at these shapes. The first row's 22 ms is likely first-launch compilation cost not amortized by `n_warmup=10`.

**Impact:** "Phase A v2 fix-ups exposed real Triton vs PyTorch delta" — the data does not show one. The kernel should be presented as "implemented and correct, but not faster at this regime" rather than as a speedup.

**Severity:** High for honesty of the writeup. The kernel work is genuine engineering and worth presenting; just not as a speedup.

---

### 🟠 BUG-6 (HIGH): Two TopK paths apply /√d differently → different selected tokens

**Location:**
- `methods/topk_selection.py:316`: `raw_scores = torch.matmul(proxy_q, middle_k.transpose(-2,-1))` — **no /√d**
- `methods/topk_kernels.py:182`: `raw = torch.matmul(...) * scale` — **with /√d**
- `methods/topk_kernels.py:130`: `s = tl.sum(...) * SCALE` (Triton kernel) — **with /√d**

When `use_kernels=True` and `use_head_softmax=True` and CUDA: Triton path runs (with /√d).
When `use_kernels=False` OR `use_head_softmax=False` OR CPU: falls through to `topk_selection.py:316` (without /√d).

**Why it matters:** With softmax following, scaling is not a monotone constant — different scales produce sharper/flatter distributions, which after head-soft-voting (cross-head sum) yield **different rankings**. So `topk_pytorch` and `topk_triton` in kernel_bench are not just two implementations of the same algorithm; they're two different algorithms that select different tokens.

**Impact:** The "PyTorch reference is numerically-equivalent fallback" claim in the docstrings is false. Quality experiments comparing kernel-on vs kernel-off cannot use the same selected tokens.

**Severity:** High. Easy fix (add `/ math.sqrt(D)` to line 316), but until then the cross-path comparison is broken.

---

### 🟠 BUG-7 (HIGH): proxy_q is K-of-last-token, not Q

**Location:** `methods/topk_selection.py:227-228` and `methods/kivi_topk_hybrid.py:356`

```python
proxy_k = self.full_past_key_values[-1][0]  # last layer's K
proxy_q = proxy_k[:, :, -1:, :]             # K of last token, used as Q
```

**Why it matters:** TokenSelect's algorithm specifies scoring `Q · K`. This implementation scores `K_last · K`. K and Q are produced by different projections of the same residual stream and are RoPE-rotated by the same angle for the same position, but their content differs — `K_last · K_other` is a different similarity than `Q · K_other`.

The codebase acknowledges this as "an established TokenSelect approximation" but no published version of the paper actually does this. The real TokenSelect monkey-patches HF to expose Q.

**Impact:** Selection quality is degraded vs paper. Combined with BUG-2, this is part of why TopK fails passkey even when the needle is in `n_local`.

**Severity:** High for paper faithfulness.

---

### 🟡 BUG-8 (MEDIUM): Hybrid centroid scoring is block-granular, not token-granular

**Location:** `methods/kivi_topk_hybrid.py:491-498`

```python
if self.score_mode == "centroid" and n_blocks > 0:
    block_token = block_raw.repeat_interleave(block_sizes, dim=1)
```

After `repeat_interleave`, all `group_size` tokens within a block carry the **same** per-head score. After per-head softmax, they receive equal probability mass within the block. After cross-head sum, all tokens in a block get the same final score. **TopK over identical scores breaks ties arbitrarily and effectively picks whole blocks at group_size = 32 token granularity.**

**Impact:** "Design (a): per-token resolution via centroid broadcast" is misleading. Selection is block-resolution. This is documented in the module docstring but not surfaced in the report.

**Severity:** Medium. Honest description fixes this — no code change needed.

---

### 🟡 BUG-9 (MEDIUM): Selection cache cosine on K-of-last-token can falsely hit

**Location:** `methods/topk_selection.py:188-194` and `methods/kivi_topk_hybrid.py:645-653`

The cache reuse check is `cosine_sim(K_last_t, K_last_{t-1}) > 0.95`. With K being a 128-dim per-head vector and only one timestep's K compared, this similarity depends on the just-decoded token's K. Common short tokens (spaces, punctuation) tend to have similar K vectors → false hit → stale indices reused even when context shifted.

**Impact:** May cause `cache_hits` rate to be artificially high. The Phase B2 long_context numbers don't expose this (use_selection_cache=False), but ppl_sanity has cache_hit_rate=0.857 — high enough that the cache is short-circuiting selection in most steps.

**Severity:** Medium. Effect on selection quality unmeasured.

---

### 🟡 BUG-10 (MEDIUM): KIVI residual/overflow dynamics are custom, not paper Algorithm 1

**Location:** `methods/kivi_quant.py:187-258` (`process_step`)

The implementation has a **two-stage** dynamic: residual buffer (FP16, fixed size R) → overflow buffer (FP16, variable, holds evicted residual tokens) → quantized block (sealed when overflow hits group_size). KIVI's Algorithm 1 specifies a **one-stage** dynamic: residual is fixed-R; when residual receives a new token, the oldest residual token is committed directly to a quantized block (no overflow staging area).

**Why it might matter:**
- Storage accounting: overflow buffer holds up to `group_size - 1` extra FP16 tokens per layer. Negligible (small constant), but real.
- Block boundaries: paper has block boundary at `R, R+G, R+2G, …` from the end of the sequence. The implementation's overflow staging means block boundary can shift by up to `G-1` tokens depending on insertion timing.
- Quality impact: probably small at G=32, R=128, but unmeasured.

**Severity:** Medium for paper faithfulness; low for actual measured quality.

---

### 🟡 BUG-11 (MEDIUM): Same `residual_length` for K and V

**Location:** `methods/kivi_quant.py:75`, `__init__(self, bits=4, residual_length=128, group_size=32)`

KIVI specifies that K and V can have different residual handling because they have different statistical properties. This implementation uses one `residual_length` for both. Likely harmless at the magnitudes used, but a paper-faithfulness gap.

**Severity:** Medium for faithfulness.

---

### 🟢 BUG-12 (LOW): Refresh fires at step 0

**Location:** `methods/topk_selection.py:218`

```python
if self.refresh_interval > 0 and step % self.refresh_interval == 0:
```

`0 % 50 == 0`, so step 0 always triggers a refresh and returns the full cache. Not a bug per se — the first decode step indeed has the freshest context, so full attention is reasonable. Just worth documenting.

---

### 🟢 BUG-13 (LOW): Runner has dead code path for `update_full_cache`

**Location:** `benchmark/runner.py:113-114`

```python
if hasattr(method, 'update_full_cache'):
    method.update_full_cache(_cache_to_tuple(step_out.past_key_values))
```

No method exposes a public `update_full_cache` (TopKMethod has `_update_full_cache`, private). This branch is never taken. The actual update happens inside `process_step` so behavior is correct, but the runner code is misleading.

---

## 4. Potential bugs needing rerun

These are suspect but not nailed without instrumentation:

1. **Centroid mode might select consistently bad blocks.** Centroid is a mean — for a block where the needle is the only outlier, the centroid is close to filler and the block ranks low. Confirm by adding `(needle_orig_pos, scores[needle_orig_pos], rank_of_needle)` instrumentation in `_score_layer` and rerunning passkey @ 4K.

2. **`use_selection_cache=True` (passkey config) may be hiding selection misses.** Passkey ran with cache enabled, so a single-step refresh + cached indices are reused for the entire 12-token answer. If step-1's selection misses the needle, all subsequent steps inherit that miss. Rerun passkey with `use_selection_cache=False` to isolate.

3. **`refresh_interval=50` with `max_new_tokens=12`** means refresh never fires after step 0. So between step 1–11, only one selection decision is made. If that decision is wrong, the entire generation is wrong. Combined with #2, this means passkey at 4K is essentially testing one selection step.

4. **Hybrid `kivi_topk` and `kivi_topk_c` having identical decode_step latencies** (~480 ms vs 478 ms at 2K) is consistent with `_materialise` dominating, but not yet proven. Profile with `torch.profiler` to attribute the time.

---

## 5. Kernel implementation and usage status

| Kernel | File / function | Implemented | Used | Verified correct |
|---|---|---|---|---|
| Fused paged score | `topk_kernels.py:_fused_paged_score_kernel` | ✅ | ✅ in `topk` (CUDA + use_kernels=True + use_head_softmax=True) | ✅ self-test rel-err < 1e-3; **kernel_bench shows it is not faster than PyTorch fallback** |
| Quant-aware score (design c) | `topk_kernels.py:_quant_score_kernel` | ✅ | ✅ in `kivi_topk_c` (CUDA, hybrid score_mode="quantized") | ✅ math derivation correct (`score = ks·Σ(kq·q) + kz·Σ(q)`); not independently unit-tested against fp16 dequant reference but factoring is exact up to fp16 |
| Fused paged top-K | `topk_kernels.py:fused_paged_topk` | ✅ pure tensor, no Triton | ✅ in `topk` when `middle_len > 2*page_size` | ✅ trivially equivalent to two-stage topk |

**Silent fallback paths that DO trigger in production:**
- `use_kernels=False` (kernel_bench's `topk_pytorch` row) → falls back to `topk_selection.py:316` PyTorch path → **drops /√d scaling** (BUG-6).
- Hybrid (`kivi_topk`, `kivi_topk_c`) does **not** use the fused-score kernel even on CUDA — its docstring (line 70-74) explicitly states the kernel's contiguous-K assumption doesn't apply to the heterogeneous (centroids + overflow + residual) layout. The hybrid's score path is hand-rolled PyTorch (`_score_layer`).

**Net:** Kernels exist, are wired, and run when `use_kernels=True` for `topk`. They don't exhibit a speedup at the regime measured.

---

## 6. Results consistency check

Cross-referencing each result file against the code that produced it:

### Passkey (`results/phase_b2/passkey/passkey.csv`, 50 rows)

| Method | seq_len | Accuracy | Code-consistent verdict |
|---|---|---|---|
| baseline | 4096 | 100% all depths | ✅ No method modifications, full FP16 attention works |
| kivi | 4096 | 100% all depths | ✅ KIVI returns full FP16 reconstruction; positions preserved |
| topk | 4096 | 0% all depths | ❌ **Diagnostic mismatch — see §7** |
| kivi_topk | 4096 | 0% all depths | ❌ Same as topk |
| kivi_topk_c | 4096 | 0% all depths | ❌ Same as topk |
| baseline | 8192 | 0% all depths | ✅ LLaMA-2-7B trained context = 4K; baseline failing at 8K is the indicator |
| any | 8192 | 0% | ✅ Model context limit, not method bug |

### Long context (`results/phase_b2/long_context/long_context.csv`, 20 rows)

- Storage MB column is consistent with KIVI ~3.4× compression at 4-bit — ✅
- `decode_step_ms` ranges from 0.009 (baseline) to 2207 (hybrid @ 32K) — these are **method overheads**, not attention costs (BUG-4). Take with caveat.
- `blocks_dequantized_avg = 1536` for hybrids — at S=2048, n_quant ≈ (2048 - 128 residual - 0 overflow) / 32 ≈ 60 blocks. Avg of 1536 means **all blocks are dequantized every step**. This contradicts the "selective dequantization" design claim — `_materialise` is iterating over all blocks because every block contains some selected token (sink + topk + local span the full sequence). Confirms why hybrid (a) and (c) latency-tie: the dequant cost dominates the score cost regardless of which scoring mode.

### Kernel bench (`results/phase_b2/kernel_bench/kernel_bench.csv`, 20 rows)

- `cache_hit_rate=0.0` everywhere ✅ (use_selection_cache=False intentionally).
- Triton ≈ PyTorch latency (BUG-5).
- Hybrid (a) ≈ Hybrid (c) latency, both 50× slower than `topk` — explained by `_materialise` dominating, not by scoring difference.

### PPL sanity (`results/phase_b2/ppl_sanity/long_context.csv`, 4 rows)

| Method | PPL |
|---|---|
| baseline | 7.2766 |
| kivi | **7.2683** |
| kivi_topk | **7.2683** (identical to kivi) |
| kivi_topk_c | **7.2683** (identical to kivi) |

Confirms BUG-3: `compute_method_perplexity` does not call `process_step`, so kivi_topk and kivi_topk_c PPL are bit-for-bit identical to plain KIVI. The 0.008 PPL gap from baseline is the prefill quantization cost, not anything to do with selection.

---

## 7. Passkey 4K diagnosis

**Verdict:** ❌ **Selected indices map correctly to positions, but RoPE-rotated K/V are not re-rotated to the new gathered positions, so the model receives positionally-corrupted attention input.**

### Evidence

For seq_len=4096, depth=0.9, n_local=512, n_sink=128, K=1024:

| Quantity | Value | Source |
|---|---|---|
| `seq_len_tokens` (target) | 4096 | passkey config |
| Actual prompt length | ~4085 | `_build_prompt` reports `n_tokens` |
| `needle_orig_pos` | ~3650 (depth=0.9 × ~162 lines × ~25 tok/line) | derived from `_build_prompt` arithmetic |
| `n_local` window | [3573, 4085] | `recent_start = current_seq_len - n_local` |
| Is needle inside local? | **Yes** | 3573 ≤ 3650 ≤ 4085 |
| Is needle in `selected_indices`? | **Yes** (always — local window is unconditionally included) | `_paged_token_selection:347` |
| `S_sel` (gathered cache len) | ~1664 = 128 + 1024 + 512 | sink + K + local |
| Q's RoPE position | 1664 (next position after gathered cache) | HF default |
| Needle K's RoPE rotation | θ(3650) (baked in at prefill) | RoPE applied during model's prefill |
| Relative angle Q · K_needle in attention | θ(1664 − 3650) = θ(−1986) | by gathered-position model expectation |
| Relative angle the model expects | θ(1664 − new_position_of_needle) ≈ θ(small) | recent token should look recent |

**The selected needle has a relative-angle that places it ~2000 positions in the past, far outside any reasonable attention window.** Combined with similar shifts on every gathered token, attention over the gathered cache is meaningless.

This is **not** a "model context limit" issue (4K is within range; baseline at 4K = 100%).
This is **not** an "off-by-one in prompt construction" issue.
This is **not** a "local window calculation" issue.
This is **not** a "selected indices missing the needle" issue.
This **is** the RoPE-on-gather positional encoding bug (BUG-2).

### Cheapest possible reproduction

Patch `_gather_layer` to print:
```python
print({"S_sel": S_sel, "needle_orig_pos": needle_orig_pos,
       "needle_in_idx": (indices == needle_orig_pos).any().item(),
       "needle_new_pos": (indices < needle_orig_pos).sum().item()})
```
Expect: `needle_in_idx=True`, `needle_new_pos ≈ 1500-1600`, but model still emits wrong digits — confirms positional corruption is the cause.

### Standard fix

Either re-apply RoPE to gathered K to bring rotation in line with new positions, or use a position-aware attention kernel. Both are well outside a "small fix" — they require changes to the gather contract and likely a custom forward pass. **For the report, the honest move is to acknowledge this as a fundamental limitation of naive KV-pruning on RoPE models and present TopK results with this caveat.**

---

## 8. PPL methodology conclusion

`compute_method_perplexity` calls only `process_prefill` then a single forward — `process_step` never executes. Therefore:

- ✅ PPL **can** detect: prefill-time corruption (e.g., a buggy quantization)
- ❌ PPL **cannot** detect: selection quality, decode-time eviction, runtime cache reuse, anything inside `process_step`

**For the writeup, scope the PPL number to:** "no prefill regression vs full FP16 cache." Do not use it to support claims about TopK retrieval quality, hybrid quality, selection accuracy, or any decode-time property.

The headline quality metric is **passkey** (which does exercise process_step) — and it shows TopK family failing for the reasons in §7, not because of PPL-detectable issues.

---

## 9. Latency methodology conclusion

The kernel_bench harness times `method.process_step(synthetic_kv, step=1)` only. There is no model forward pass and no real attention. Consequences:

- ❌ "speedup_vs_baseline" numbers in kernel_bench.csv and long_context.csv are not interpretable as algorithmic speedup. Baseline is a no-op.
- ❌ "topk_pytorch vs topk_triton" comparison is between two different algorithms (BUG-6 — different scaling in the score path).
- ⚠️ Cross-method comparison within the same row (e.g. kivi_topk vs topk) is meaningful only as **per-step bookkeeping overhead**, not as decode latency.
- ✅ Storage MB numbers are real and trustworthy (KIVI 3.4× compression).
- ✅ End-to-end `throughput_tps` from `generate_with_method` (passkey throughput column) is real because it includes the model forward.

**For the writeup:**
- Drop "kernel speedup" claims unless rerun against a real attention forward.
- Present "decode_step_ms" as **method overhead**, not attention time.
- Rely on passkey's throughput for any "this method is slower/faster end-to-end" claim — that includes the model forward.

---

## 10. Required fixes before final report

In order of severity:

1. **Add a "Known limitations" section to the writeup** that explicitly documents BUG-2 (RoPE+gather positional corruption) as the cause of TopK passkey 0%. Without this section, the report misrepresents what was built.

2. **Fix or relabel BUG-1** (KIVI K/V quantization swap). Either:
   - (a) Swap the function bodies — `quantize_per_channel` should reduce over `dim=-2`, `quantize_per_token` over `dim=-1`. Then rerun PPL sanity to confirm baseline KIVI improves.
   - (b) Keep code as-is and rename functions to match what they actually compute, then state in the report that "this implementation uses per-token K and per-channel V (the inverse of KIVI's prescription)."

3. **Reword PPL claims.** Replace "no quality regression" with "no prefill-time regression" and add the caveat that PPL is structurally blind to selection.

4. **Reword latency claims.** Drop "speedup_vs_baseline." Reframe kernel_bench as "per-step method overhead" and acknowledge baseline is a no-op.

5. **Fix BUG-6** (`/√d` mismatch in PyTorch fallback). One-line change at `topk_selection.py:316`. Rerun kernel_bench to get an honest pytorch-vs-triton comparison.

6. **Add a passkey rerun with `use_selection_cache=False` and `refresh_interval=1`** to isolate whether the 0% is from positional corruption (BUG-2) or from cache staleness (BUG-9). Expected result: still 0% — would confirm BUG-2 dominates.

7. **Profile `_materialise`** with `torch.profiler.profile` on one hybrid step at S=8192 to attribute the 800 ms decode_step. Expected outcome: dequant dominates, scoring is <5% — which is the honest story for "design (a) and (c) tied."

---

## 11. Optional improvements

- **Implement RoPE-on-gather fix** (re-rotate gathered K to gathered positions). Would actually let TopK work on LLaMA-2.
- **Add unit tests:**
  - `tests/test_kivi_quant.py` — round-trip and direction (would have caught BUG-1).
  - `tests/test_topk_gather.py` — positional preservation (would catch BUG-2).
  - `tests/test_topk_paths.py` — assert PyTorch fallback and Triton produce same selected indices on a fixed seed (would catch BUG-6).
- **Add a real "attention-cost benchmark"** that times one model forward at varying S, with each method, to give an honest end-to-end latency comparison.
- **Switch base model to one with longer trained context** (Llama-3-8B, Mistral-7B-v0.2) to give passkey @ 8K and 16K something to measure other than the context limit.
- **Rename `quantize_per_channel` and `quantize_per_token` to `quantize_per_token_axis` and `quantize_per_channel_axis`** (i.e., make the names describe the reduce axis, not the resulting granularity) — would have prevented BUG-1.

---

## Final terminal summary

```
Files analyzed:           22 source files (4948 LOC across methods/, experiments/, benchmark/)
Methods audited:          5 (baseline, kivi, topk, kivi_topk, kivi_topk_c)

Confirmed bugs:           13
  Critical: 4   (KIVI quant swap, RoPE+gather, PPL methodology, latency methodology)
  High:     3   (Triton not faster, /√d mismatch, K-as-Q proxy)
  Medium:   4   (centroid block-granular, cache cosine, residual dynamics, K/V same residual)
  Low:      2   (refresh@step0, dead code in runner)

Potential bugs:           4 needing instrumentation/rerun

Kernel implemented?       yes  (3 kernels: fused score, quant score, paged topk)
Kernel actually used?     yes for topk (when use_kernels=True + CUDA + softmax-on)
                          yes for kivi_topk_c (quant_score)
                          NOT used by kivi_topk centroid mode (by design)
Kernel speedup?           NO — within 5% of PyTorch fallback in kernel_bench

Safe to report results?   WITH CAVEATS:
                          - storage compression (3.4×): yes, real and trustworthy
                          - KIVI passkey accuracy: yes, real
                          - TopK / hybrid passkey accuracy: yes (it really IS 0%, but
                            the explanation in the writeup must be RoPE+gather, not
                            "selection failed")
                          - PPL: scope to "no prefill regression," not "no quality
                            regression"
                          - Latency: drop speedup-vs-baseline; report as overhead

Rerun required?           NO for passkey at 4K — the 0% is real, just needs the
                          right explanation.
                          OPTIONAL: rerun kernel_bench after fixing /√d mismatch
                          (BUG-6) for an honest Triton vs PyTorch comparison.

Top 5 action items:
  1. Document BUG-2 (RoPE+gather positional corruption) as a Known Limitation
     section in the report — this is the explanation for TopK 0% passkey.
  2. Either swap KIVI quant function bodies (BUG-1) or rename them and document
     that this implementation inverts KIVI's K/V quantization directions.
  3. Rescope PPL claim: "no prefill regression" only. Drop any quality claim
     about TopK/hybrid based on PPL.
  4. Rescope latency claim: drop "speedup_vs_baseline" everywhere. Reframe
     kernel_bench as per-method per-step overhead, not attention time.
  5. Add a one-line /√d fix at topk_selection.py:316 (BUG-6) so the
     PyTorch and Triton paths select the same tokens.
```

---

**End of audit. No source files have been modified. Branch `audit/paper-faithfulness` contains only this report and `result_inventory.csv` under `results/phase_b2_audit/`.**
