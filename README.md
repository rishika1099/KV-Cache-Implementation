# KV Cache Optimization Benchmark
### Columbia University — HPML Course

Fair, controlled comparison of 4 KV cache optimization techniques for autoregressive LLM inference on a single NVIDIA H100 GPU. All methods share the same model, tokenizer, decode loop, prompts, batch size, random seeds, and hardware. The only variable that changes across runs is the KV cache policy.

---

## Methods

| Method | Paper | Key Idea |
|--------|-------|----------|
| **Baseline** | — | FP16 full KV cache (reference) |
| **KIVI** | Liu et al., ICML 2024 | Asymmetric INT4/INT2 quantization (per-channel keys, per-token values) |
| **xKV** | Chang et al., 2025 | Per-layer SVD low-rank approximation |
| **SnapKV** | Li et al., NeurIPS 2024 | One-shot post-prefill eviction using attention patterns |
| **TopK** | Wu et al., EMNLP 2025 | Dynamic per-step top-K token selection |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to HuggingFace (required for LLaMA-2 access)
huggingface-cli login

# 3. Request LLaMA-2 access at:
#    https://huggingface.co/meta-llama/Llama-2-7b-hf
#    (approval usually within 1 hour)

# 4. (Optional) If LLaMA-2 is unavailable, use OPT as drop-in:
#    python run_benchmark.py --model facebook/opt-6.7b
```

**GCP setup:** `bash setup_gcp.sh` (sets up instance, downloads weights, configures tmux)

---

## Quick Start

```bash
# Smoke test — 2 prompts, 20 tokens each (~15 min on H100)
python run_benchmark.py --dry_run

# Full benchmark (~6-8 hours on H100)
python run_benchmark.py

# Specific methods only
python run_benchmark.py --methods baseline kivi snapkv

# Resume after interruption
python run_benchmark.py --resume

# Generate plots after benchmark completes
python plot_results.py --results results/results.jsonl
```

---

## Adding a New Method

1. Create `methods/my_method.py` implementing `MethodWrapper`:
   ```python
   from methods.base import MethodWrapper

   class MyMethod(MethodWrapper):
       def process_prefill(self, past_key_values, attention_weights=None):
           ...
       def process_step(self, past_key_values, step, attention_weights=None):
           ...
       def reset(self):
           ...
   ```

2. Register it in `run_benchmark.py`'s `build_method()` function.

3. Add configs to `configs/default.yaml` under `methods:`.

4. Add a `__main__` smoke test block at the bottom of your file.

---

## Results

*(Fill in after running the benchmark)*

| Method | Compression | PPL | Throughput (TPS) | Peak Mem (GB) |
|--------|------------|-----|-----------------|---------------|
| Baseline | 1.0x | — | — | — |
| KIVI (4-bit) | — | — | — | — |
| KIVI (2-bit) | — | — | — | — |
| xKV (rank=128) | — | — | — | — |
| SnapKV (40%) | — | — | — | — |
| TopK (K=512) | — | — | — | — |

---

## Citations

```bibtex
@inproceedings{liu2024kivi,
  title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
  author={Liu, Zirui and others},
  booktitle={ICML},
  year={2024}
}

@article{chang2025xkv,
  title={xKV: Cross-Layer SVD for KV-Cache Compression},
  author={Chang, Chi-Chih and others},
  year={2025}
}

@inproceedings{li2024snapkv,
  title={SnapKV: LLM Knows What You Are Looking for Before Generation},
  author={Li, Yuhong and others},
  booktitle={NeurIPS},
  year={2024}
}

@inproceedings{wu2025tokenselect,
  title={TokenSelect: Efficient Long-Context Inference via Dynamic Token-Level KV Cache Selection},
  author={Wu, Wei and others},
  booktitle={EMNLP},
  year={2025}
}
```
