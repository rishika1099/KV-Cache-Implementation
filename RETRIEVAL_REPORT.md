# Hybrid retrieval-fix ablation

Branch: `research/retrieval-fixes` (off `perf/hybrid-optimization`)
Goal: lift the 0% mid-depth band on the KIVI+TopK hybrid passkey result
without regressing edge-depth accuracy or perplexity.

## Baseline (pre-fix)

`results/phase_perf/passkey.csv`, Llama-2-7B @ 4K, 5 trials × 5 depths,
`K=1024 / n_sink=128 / n_local=512`:

| depth | KIVI | KIVI+TopK (a) | KIVI+TopK (c) |
|---:|---:|---:|---:|
| 0.1 | 100% | 100% | 100% |
| 0.3 | 100% |   0% |   0% |
| 0.5 | 100% |   0% |   0% |
| 0.7 | 100% |   0% |   0% |
| 0.9 | 100% | 100% | 100% |

(a) and (c) match exactly, so the bottleneck is upstream of scoring
fidelity. Two suspects:

1. The proxy query is built from a single most-recent K-vector — fine
   for end-of-prompt and near-sink lookups, weak when the needle is
   buried mid-prompt.
2. Block centroids mean-pool 32 keys, diluting any single spike key
   (the needle) by 1/32.

## Fixes

| # | Knob | What it does |
|---:|---|---|
| 1 | `score_mode='maxpool'` | per-channel max over each block |
| 2 | `two_pass_factor=2` | centroid pre-rank → exact `quant_score` rerank |
| 3 | `proxy_history=8`, `proxy_pool='max'` | pool last 8 K-vectors into proxy |
| 4 | `dynamic_sinks=True` | pick `n_sink` positions from prefill K-norm |

All four are independently flagged. The composite (`kivi_topk_all`)
sets all four. Defaults are unchanged for `kivi_topk` / `kivi_topk_c`.

## Run

```bash
modal secret create wandb WANDB_API_KEY=<your-key>   # one-time
modal run modal_retrieval_ablation.py --mode smoke   # ~$0.20
modal run modal_retrieval_ablation.py --mode all     # ~$5–8
modal volume get kv-benchmark-results /retrieval_ablation \
    ./results/retrieval_ablation
```

W&B project: `kv-cache-benchmark`, group `retrieval_ablation`.

## Results — pending H100 run

| method | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|
| `kivi` | – | – | – | – | – |
| `kivi_topk` | – | – | – | – | – |
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

## Deferred

- per-head adaptive top-K (entropy-thresholded budget)
- learned selector head (small linear, trained on retrieval prompts)
- pyramid / band retention (decay older tokens)
- fp8 shadow scoring table
- anchor-aware retrieval (benchmark-specific)
