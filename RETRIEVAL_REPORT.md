# Hybrid retrieval-fix ablation

Branch: `research/retrieval-fixes` (off `perf/hybrid-optimization`)
Goal: lift the 0% mid-depth band on the KIVI+TopK hybrid passkey result.

## TL;DR

Four selection knobs (maxpool / two-pass / multi-query / dynamic sinks)
were added to attack the proxy-query and block-summary design. **None
of them moved the needle at the original K=1024 budget.** A K-sweep on
the unmodified hybrid shows the mid-band recovers fully when K is
raised: K=2048 fixes 4 of 5 depths, K=3072 fixes all 5. So the diagnosis
"scoring is fundamentally broken" was wrong — the design is fine, the
budget was just too tight.

## Setup

`results/retrieval_ablation/`, Llama-2-7B @ 4K, 5 trials × 5 depths,
`n_sink=128 / n_local=512`. Run via `modal_retrieval_ablation.py`,
results pushed to W&B project `kv-cache-benchmark`, group
`retrieval_ablation`.

## Ablation @ K=1024 — accuracy by depth

| method                | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | mean |
|---|---:|---:|---:|---:|---:|---:|
| `kivi`                | 100 | 100 | 100 | 100 | 100 | **100** |
| `kivi_topk`           | 100 |   0 |   0 |   0 | 100 | 40 |
| `kivi_topk_maxpool`   | 100 |   0 |   0 |   0 | 100 | 40 |
| `kivi_topk_twopass`   | 100 |   0 |   0 |   0 | 100 | 40 |
| `kivi_topk_multiq`    | 100 |   0 |   0 |   0 | 100 | 40 |
| `kivi_topk_dynsink`   | 100 |   0 |   0 |   0 | 100 | 40 |
| `kivi_topk_all`       | 100 |   0 |   0 |   0 | 100 | 40 |

Six hybrid variants, identical 40% accuracy. Each variant was
implemented and verified to actually trigger its code path
(maxpool buffer allocated and used, two-pass kernel call ran, query
history rolled, dynamic sinks chosen and pinned). They simply don't
change which tokens win the top-K at this budget.

Reading: at K=1024 the proxy query is noisy enough that the needle
block doesn't reliably make the top-K cut, and changing how that
top-K is computed doesn't help — the underlying signal is below
threshold no matter how it's pooled.

## K-sweep on `kivi_topk` (no fixes)

| K     | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | mean |
|---:|---:|---:|---:|---:|---:|---:|
|  512  | 100 |   0 |   0 |   0 | 100 | 40 |
| 1024  | 100 |   0 |   0 |   0 | 100 | 40 |
| 2048  | 100 | 100 | 100 |   0 | 100 | 80 |
| 3072  | 100 | 100 | 100 | 100 | 100 | **100** |

Step-function recovery. K=512 vs 1024 changes nothing; the jump from
1024 to 2048 fixes three depths; the jump to 3072 fixes the last one.
The proxy-query design works once it has enough headroom.

## Implications

The four fixes were the wrong knob. The right knob was already in the
constructor: `K`. Practical guidance for the hybrid going forward:

- Default K=1024 is too tight for mid-context retrieval. Recommend
  K≥2048 for any retrieval-heavy workload, K=3072 to be safe.
- The cost story matters: at seq_len=4K, K=3072 selects 3072+128+512 =
  3712 of 4096 tokens, so the hybrid is barely saving any work — the
  real value is at long context where the (n_sink + n_local + K) /
  seq_len ratio stays small. The K-sweep should be repeated at 16K and
  32K to see whether K needs to scale with context.
- The four fixes still ship behind flags (default off) and don't
  regress the storage-only or perplexity-only paths. They're cheap to
  carry and might matter under a different scoring regime.

## Reproduction

```bash
modal secret create wandb WANDB_API_KEY=<key>          # one-time
modal run modal_retrieval_ablation.py --mode all       # ~$5–8, ~1h
modal volume get kv-benchmark-results /retrieval_ablation \
    ./results/retrieval_ablation
```

W&B runs:
- ablation: https://wandb.ai/rm4318-columbia-university/kv-cache-benchmark/runs/v3nt58xo
- ksweep:   https://wandb.ai/rm4318-columbia-university/kv-cache-benchmark/runs/ck2ac75v

## Deferred

Same list as before, now with a sharper prior — none of these are worth
trying until the budget question is resolved at 16K / 32K context:

- per-head adaptive top-K
- learned selector head
- pyramid retention
- fp8 shadow scoring table
- anchor-aware retrieval
