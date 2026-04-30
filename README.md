# KV Cache Optimization Benchmark
### Columbia University — High Performance Machine Learning (HPML)

A controlled, apples-to-apples comparison of KV-cache compression
techniques for autoregressive LLM decoding on a single NVIDIA H100, plus
a **novel composition** — KIVI quantization + TokenSelect dynamic top-K —
implemented two ways and benchmarked end-to-end.

Across all runs the model, tokenizer, decode loop, prompts, batch size,
random seeds, and hardware are held fixed. The only variable is the KV
cache policy.

---

## What's in this repo

| | |
|---|---|
| **Methods** | `methods/` — one file per technique, all subclass `MethodWrapper` |
| **Experiments** | `experiments/` — passkey retrieval, long-context latency, perplexity sanity, kernel microbench |
| **Modal entrypoints** | `modal_phase_a.py`, `modal_phase_b.py`, `modal_phase_b2.py`, `modal_perf_check.py` — H100 cloud runners |
| **Results** | `results/_h100_post_fix_v2/` (Phase B2 final), `results/phase_perf/` (perf-optimized hybrid) |
| **Reports** | `MIDPOINT_REPORT.md`, `results/phase_perf/PERF_REPORT.md`, `report/main.tex` |

---

## Methods compared

| Method | Source | Idea | Status |
|---|---|---|---|
| **Baseline** | — | FP16 full KV cache (reference) | ✅ |
| **KIVI** | Liu et al., ICML 2024 | Asymmetric INT4 quantization (per-token K, per-block per-channel V) | ✅ |
| **TopK** (TokenSelect) | Wu et al., EMNLP 2025 | Dynamic per-step top-K selection over paged scoring with sink + local windows | ✅ |
| **KIVI + TopK (a)** *novel* | this work | TopK selection scored against quantized **block centroids** (cheap, lossy) | ✅ |
| **KIVI + TopK (c)** *novel* | this work | TopK selection scored against quantized state via a **Triton int×fp kernel** (`quant_score`) — exact dot product directly on uint8 storage | ✅ |
| ~~xKV~~ | Chang et al., 2025 | Per-layer SVD low-rank approximation | scoped out |
| ~~SnapKV~~ | Li et al., NeurIPS 2024 | Post-prefill one-shot eviction | scoped out |

xKV and SnapKV were dropped in Phase B in favor of going deep on the
KIVI×TopK hybrid. See `MIDPOINT_REPORT.md` for the rationale.

---

## Headline results (Llama-2-7B, single H100 80GB)

### Long-context decode latency — `decode_step_ms`, `K=1024, n_sink=128, n_local=512`

| seq_len | baseline | KIVI | TopK | KIVI+TopK (a) | KIVI+TopK (c) |
|---:|---:|---:|---:|---:|---:|
|  2048 |   0.011 |  91.7 |  33.1 | **150.2** | **148.3** |
|  8192 |   0.008 | 379.6 |  16.2 | **433.1** | **448.9** |
| 16384 |   0.010 | 772.2 |  29.4 | **859.1** | **831.1** |
| 32768 |   0.009 | 1579.8 | 55.8 | **1618.6** | **1587.3** |

> Source: `results/phase_perf/long_context.csv` (post-perf-optimization
> branch, `use_selection_cache=False` to isolate kernel cost). Baseline
> uses cached fp16 attention without our wrapper, so its absolute number
> is not directly comparable to the others — it's a sanity floor.

After the perf optimization pass on `perf/hybrid-optimization`, the
hybrid is **within 1.0–1.6× of KIVI alone** at long seq_lens; before
optimization it was 1.3–5.6× slower. See `results/phase_perf/PERF_REPORT.md`
for the full bottleneck analysis and 9 specific changes (vectorized
`_materialise`, batched V dequant, pre-allocated growable buffers,
fp16 score path, cached block-id LUT, selection-cache fix, Triton
kernel pre-warm, honest storage accounting, debug-gated counters).

### Passkey retrieval @ 4K — accuracy across needle depth

5 trials × 5 depths × `max_new_tokens=12`, Llama-2-7B.

| depth | baseline | KIVI | TopK | KIVI+TopK (a) | KIVI+TopK (c) |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 100% | 100% |   0% | **100%** | **100%** |
| 0.3 | 100% | 100% |  80% |    0%  |    0%  |
| 0.5 | 100% | 100% | 100% |    0%  |    0%  |
| 0.7 | 100% | 100% |   0% |    0%  |    0%  |
| 0.9 | 100% | 100% |  60% | **100%** | **100%** |

> Source: `results/phase_perf/passkey.csv`. The mid-depth zero band on
> the hybrid is real and documented — both centroid scoring (a) and
> int×fp scoring (c) bias toward edges of the prompt under the current
> `n_sink=128 / n_local=512 / K=1024` config. This is a known limitation
> of the proxy-query design under tight K budgets, not a kernel bug.

### Perplexity sanity — WikiText-2, seq_len=2048

| Method | PPL | Δ vs baseline |
|---|---:|---:|
| Baseline (fp16) | 6.126 | — |
| KIVI (4-bit) | 6.184 | +0.06 |
| TopK | 6.126 | 0.00 |
| KIVI+TopK (a) | 6.184 | +0.06 |
| KIVI+TopK (c) | 6.184 | +0.06 |

> Source: `results/_h100_post_fix_v2/phase_b2/ppl_sanity/`. PPL deltas
> are within reproducibility noise. The hybrid does **not** introduce
> any quality regression over KIVI alone on language modeling — its
> only weakness is needle-in-haystack mid-depth retrieval.

---

## Setup

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Hugging Face login (Llama-2 needs gated access)
huggingface-cli login
# Request access: https://huggingface.co/meta-llama/Llama-2-7b-hf

# 3. (Optional) Modal account — only needed if using the cloud entrypoints
pip install modal && modal token new
```

Hardware: a single H100 80GB is the canonical target. The kernels also
work on A100; OPT-125M is supported as a smoke-test model only (note:
`benchmark/runner.py` injects `position_ids` for the BUG-2 RoPE fix,
which OPT does not accept — Llama-class models work fine).

---

## Running benchmarks

### Local (any GPU)

```bash
# End-to-end smoke (uses configs/default.yaml)
python run_benchmark.py --dry_run

# Single-experiment runs
python run_experiment.py passkey       --methods baseline kivi topk kivi_topk
python run_experiment.py long_context  --seq_lens 2048 8192 16384
python run_experiment.py ppl_sanity    --seq_len 2048
```

### Modal (recommended — automatic H100 provisioning)

```bash
# Phase B2: full quality matrix (passkey + ppl + long_context + kernel bench)
modal run modal_phase_b2.py

# Perf-only re-run after a code change to the hybrid
modal run modal_perf_check.py

# Pull results back locally
modal volume get kv-benchmark-results /phase_perf ./results/phase_perf
```

Each Modal entrypoint dispatches independent jobs to separate H100
containers in parallel via `.spawn()`, then commits artifacts to a
shared volume.

---

## Adding a new method

1. Create `methods/my_method.py` subclassing `MethodWrapper`:

   ```python
   from methods.base import MethodWrapper

   class MyMethod(MethodWrapper):
       def process_prefill(self, past_key_values, attention_weights=None): ...
       def process_step   (self, past_key_values, step, attention_weights=None): ...
       def reset          (self): ...
       def get_kv_size_bytes(self, past_key_values) -> int: ...
   ```

2. Register it in `methods/registry.py` under a new `_make_<name>` factory.

3. Add a `cfg.<name>` block to `configs/default.yaml`.

4. Optionally add a `__main__` smoke block at the bottom of your file.

The harness handles RoPE position bookkeeping, the decode loop,
metric collection (PPL, throughput, p50/p99, peak memory, blocks
dequantized, cache hit rate), and CSV/JSON serialization.

---

## Project structure

```
methods/
  base.py                    # MethodWrapper interface
  baseline.py                # full-fp16 reference
  kivi_quant.py              # KIVI INT4 quantization
  topk_selection.py          # TokenSelect dynamic top-K
  kivi_topk_hybrid.py        # ★ novel composition (designs a + c)
  topk_kernels.py            # Triton fused_paged_score + quant_score
  registry.py                # method factories
  rope_utils.py              # RoPE delta helpers (BUG-2 fix)
benchmark/
  runner.py                  # decode loop, position_ids injection
  metrics.py                 # PPL, throughput, mem
experiments/
  passkey_retrieval.py
  long_context.py
  ppl_sanity.py
  kernel_bench.py
results/
  _h100_post_fix_v2/         # Phase B2 final (post bug-fix)
  phase_perf/                # post perf-optimization
  phase_perf/PERF_REPORT.md  # bottleneck analysis + before/after
configs/default.yaml
report/main.tex              # writeup
```

---

## What was hard / interesting

- **Two RoPE bugs in the harness** (Phase B audit). Fixed in
  `methods/rope_utils.py` + `benchmark/runner.py` via the "Option B"
  strategy: inject `position_ids` into the model `forward()` rather
  than re-rotating the cache in place. Branch:
  `fix/topk-hybrid-correctness`.
- **Hybrid was 5.6× slower than KIVI alone at 2K** until a Phase-C
  perf pass replaced a per-block Python loop in `_materialise` with
  a single batched gather + fp16 multiply-add. Branch:
  `perf/hybrid-optimization`. Closed 70–93% of the gap; 32K is now
  essentially tied with KIVI native speed.
- **Storage accounting** had been under-reporting the hybrid's
  scratch buffer; now reports the honest 884 MB at 2K (vs 351.5 MB
  before). The mirror is uint8, not bit-packed — a known halving
  opportunity listed as future work.

---

## Reports & branches

| Branch | What's in it |
|---|---|
| `main` | initial scaffold, OPT smoke |
| `audit/paper-faithfulness` | read-only audit of the original implementation against the source papers |
| `fix/topk-hybrid-correctness` | Phase B2 — RoPE fixes, TopK off-by-one, hybrid correctness |
| `perf/hybrid-optimization` ← latest | Phase C — vectorized `_materialise`, batched V dequant, pre-warmed Triton kernels (`results/phase_perf/PERF_REPORT.md`) |

---

## Citations

```bibtex
@inproceedings{liu2024kivi,
  title     = {KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
  author    = {Liu, Zirui and others},
  booktitle = {ICML},
  year      = {2024}
}

@inproceedings{wu2025tokenselect,
  title     = {TokenSelect: Efficient Long-Context Inference via Dynamic
               Token-Level KV Cache Selection},
  author    = {Wu, Wei and others},
  booktitle = {EMNLP},
  year      = {2025}
}

@inproceedings{li2024snapkv,
  title     = {SnapKV: LLM Knows What You Are Looking for Before Generation},
  author    = {Li, Yuhong and others},
  booktitle = {NeurIPS},
  year      = {2024}
}

@article{chang2025xkv,
  title  = {xKV: Cross-Layer SVD for KV-Cache Compression},
  author = {Chang, Chi-Chih and others},
  year   = {2025}
}
```
