# Phase D — Hybrid Retrieval Fixes

**Branch:** `research/retrieval-fixes` (off `perf/hybrid-optimization`)
**Target:** close the depth-0.3 / 0.5 / 0.7 zero band on the KIVI+TopK
hybrid passkey result reported in `results/phase_perf/passkey.csv`.

---

## 1. The problem

Phase B2 + perf-pass measurement, Llama-2-7B @ 4K, 5 trials × 5 depths,
`K=1024 / n_sink=128 / n_local=512`:

| depth | KIVI | KIVI+TopK (a) | KIVI+TopK (c) |
|---:|---:|---:|---:|
| 0.1 | 100% | **100%** | **100%** |
| 0.3 | 100% |    0%   |    0%   |
| 0.5 | 100% |    0%   |    0%   |
| 0.7 | 100% |    0%   |    0%   |
| 0.9 | 100% | **100%** | **100%** |

(a) = centroid scoring; (c) = exact int×fp dot product (`quant_score`).
Both designs collapse identically on the middle band, so the failure
is **not** in scoring fidelity — `quant_score` already matches dequant
within rounding. The bottleneck has to be upstream.

Diagnosis: under a tight K budget the hybrid *biases the selection
toward the start and end of the prompt*. Two reasons:

1. The **proxy query** is a single most-recent K-vector projected through
   a head-softmax + criticality vote. It works for end-of-prompt
   (depth 0.9) and stays near the sink (depth 0.1), but for a needle
   buried mid-prompt it doesn't construct a query that points at the
   needle.
2. **Block centroids** mean-pool 32 keys; one strong-but-isolated key
   (the needle) gets diluted by 31 unrelated ones. The block summary
   stops being discriminative exactly where it matters.

Either symptom is enough on its own; both contribute.

---

## 2. Four orthogonal fixes (all landed)

| # | Knob | What it does | Cost |
|---:|---|---|---|
| 1 | `score_mode='maxpool'` | Per-channel **max** over each block's 32 keys instead of mean. Preserves spike of the needle key. | 1 extra `[H, n_blocks, D]` fp16 buffer (~1% of KIVI store). |
| 2 | `two_pass_factor=2` | Centroid pre-rank → keep top `2·K_blocks` blocks → exact `quant_score` rerank only those. FAISS-IVF style. | 2× block-budget kernel call (1.4× of one-pass). |
| 3 | `proxy_history=8`, `proxy_pool='max'` | Build proxy query from the last 8 K-vectors max-pooled per channel. Multi-step retrieval cue, not single-step. | 8× single proxy-query memory (~few KB per layer). |
| 4 | `dynamic_sinks=True` | At end of prefill, pick top-`n_sink` by aggregated K-norm across layers, pin those positions. Replaces the dumb "first 128 tokens" rule. | one extra reduction at prefill, free at decode. |

All four are gated by independent flags so the ablation can attribute
each one. The "kitchen sink" composite (`kivi_topk_all`) sets all four.

### Why not change K?

We're going to confirm with `run_ksweep` that K=512/1024/2048/3072
*all* fail at depth 0.5 on the pre-fix hybrid. If the curve is flat in
K, the cure is not "more budget"; it's a better selection signal —
which is what these four fixes target. (If the curve climbs steeply
with K, that's a separate signal and we'll write it up.)

---

## 3. Code shape

All edits in `methods/kivi_topk_hybrid.py`, registered through
`methods/registry.py`:

```
kivi_topk            ← pre-fix hybrid (centroid, single-query, static sinks)
kivi_topk_c          ← (c) variant, exact quant_score, otherwise pre-fix
kivi_topk_maxpool    ← fix #1 only
kivi_topk_twopass    ← fix #2 only
kivi_topk_multiq     ← fix #3 only
kivi_topk_dynsink    ← fix #4 only
kivi_topk_all        ← all four stacked
```

Implementation sketches:

**Maxpool** — per-channel max within each 32-key block, parallel to the
existing centroid path. Stored once at prefill, grown lazily during
decode in `_sync_block_state`.

**Two-pass** — first rank blocks by centroid score, gather only the top
`2·K_blocks` candidates, then run `quant_score` on just those candidate
blocks for an exact per-token score. Top-K over that exact score.

**Multi-query** — keep a rolling buffer of the last 8 K-vectors used to
form the proxy. Pool them with `max` (default for retrieval; `mean`
falls back to single-query behaviour for ablation purity).

**Dynamic sinks** — after prefill, aggregate `‖k‖₂` across all layers
and heads per position, pick the top-`n_sink` positions, pin those.
Replaces the static "first 128" rule. Decoder uses
`self._sink_positions` if set, else falls back.

---

## 4. Validation plan

Modal H100 entrypoint: `modal_phase_d.py`, three modes.

| Mode | Methods × Depths × Trials | Wall | Cost |
|---|---|---|---|
| `smoke` | 1 × 1 × 1 (kitchen sink only) | ~3 min | ~$0.20 |
| `full` | 7 × 5 × 5 = 175 gens | ~45–90 min | ~$3–6 |
| `ksweep` | `kivi_topk` at K∈{512,1024,2048,3072}, 5×5 each | ~30 min | ~$2 |
| `all` | full + ksweep parallel | ~max(full, ksweep) | ~$5–8 |

Pulled with:
```
modal volume get kv-benchmark-results /phase_d ./results/phase_d
```

---

## 5. Results — *pending H100 run*

| method | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| `kivi` (storage-only ref) | – | – | – | – | – |
| `kivi_topk` (pre-fix) | – | – | – | – | – |
| `kivi_topk_maxpool` | – | – | – | – | – |
| `kivi_topk_twopass` | – | – | – | – | – |
| `kivi_topk_multiq` | – | – | – | – | – |
| `kivi_topk_dynsink` | – | – | – | – | – |
| `kivi_topk_all` | – | – | – | – | – |

K-sweep on `kivi_topk` (no fixes):

| K | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---:|---:|---:|---:|---:|---:|
|  512 | – | – | – | – | – |
| 1024 | – | – | – | – | – |
| 2048 | – | – | – | – | – |
| 3072 | – | – | – | – | – |

To be filled once `modal_phase_d.py --mode all` completes.

---

## 6. Deferred — explicit non-goals for this branch

These are real improvements but they're either (a) heavier engineering
than a single ablation can justify, or (b) only worth doing once the
four cheaper fixes land — so they sit behind a flag for now.

- **Per-head top-K with adaptive budget.** Heads differ widely in how
  spread-out their attention is; uniform K wastes budget on near-uniform
  heads and starves spiky ones. Wants a per-head entropy threshold and
  a paged scoring kernel that can do ragged K. Probably the next
  highest-yield fix after these four.

- **Trained / learned selector head.** Replace the hand-built proxy
  query with a small linear head trained once on a few hundred passkey
  prompts. Off-strategy for an HPML systems project, but the natural
  step if these four fixes are still partial.

- **Pyramid / band retention.** SnapKV-style: keep more recent tokens
  fully and decay older ones. Useful for streaming; for fixed-context
  retrieval it mostly trades depth-0.9 perf for depth-0.1 perf, which
  is not what we need.

- **Shadow scoring table** (Triton) — keep an fp8 mirror of K just for
  scoring, dequant only when materialising. Fastest path to an exact
  scoring signal but doubles the half of storage we already have.
  Saves only if the bottleneck is *fidelity* — which our (a) vs (c)
  comparison shows it isn't.

- **Anchor-aware retrieval.** Detect "passkey N" in the prompt and pin
  that span. Cute, but it's a benchmark hack — not a generalisable
  retrieval fix.

These are listed so reviewers can see they were considered and rejected
or queued, not missed.

---

## 7. Reproduction

```bash
git checkout research/retrieval-fixes

# Smoke test (~$0.20, ~3 min)
modal run modal_phase_d.py --mode smoke

# Full ablation (~$5, ~1 hr)
modal run modal_phase_d.py --mode all

# Pull results
modal volume get kv-benchmark-results /phase_d ./results/phase_d
```

CSV: `results/phase_d/{ablation,ksweep_K*}/passkey.csv`.
Re-run the table in §5 from those.
