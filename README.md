# KV Cache Optimization Benchmark


Fair, controlled comparison of 3 KV cache optimization techniques for autoregressive LLM inference on a single NVIDIA A100 GPU. All methods share the same model, tokenizer, decode loop, prompts, batch size, random seeds, and hardware. The only variable that changes across runs is the KV cache policy.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Login to HuggingFace (required for LLaMA-2 access)
huggingface-cli login

```



---

## Quick Start

```bash


python run_benchmark.py

python run_benchmark.py --methods baseline kivi snapkv

python run_benchmark.py --resume


```


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
