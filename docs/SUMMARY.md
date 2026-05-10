# Project Summary

Purpose
- Fair, controlled benchmark comparing three KV-cache optimization techniques (Baseline, KIVI, SnapKV) for autoregressive LLM inference on a single NVIDIA A100 GPU.

Quick setup
- Install deps: `pip install -r requirements.txt`
- Login to HuggingFace: `huggingface-cli login`

Quick start
```bash
python run_benchmark.py
python run_benchmark.py --methods baseline kivi snapkv
python run_benchmark.py --resume
```

Entry points (function-centric)
- [run_benchmark.py](run_benchmark.py): primary CLI that kicks off experiments and parses `--methods` / `--resume` flags.
    - build config (model, methods, seed, benchmark/dataset, sequence length) and log with tqdm
    - load tokenizer and model
    - custom DatasetLoader loading
    - `prompts_to_run = dataset_loader.synthetic_prompts`
```py
for prompt_info in tqdm(prompts_to_run)
        _, metrics = generate_with_method(
                model, tokenizer, baseline_method,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                device=device,
            )
        baseline_kv_cache[(seq_len, pid)] = metrics['kv_cache_mb']

for method_name, method_cfg in tqdm(active_methods, desc="Methods"):
    method_obj = build_method(method_name, method_cfg)
    for prompt_info in tqdm(prompts_to_run)

```

- [benchmark/runner.py](benchmark/runner.py): core runner with `generate_with_method()` and `compute_method_perplexity()` that orchestrate model prefill, decode loop, metrics, and PPL evaluation.
  - `generate_with_method()` pseudocode:
    1. Reset method state (`method.reset()`)
    2. Tokenize prompt and run model prefill (optionally request attentions)
```py
prefill_out = model(
        **inputs,
        use_cache=True,
        output_attentions=need_attn,
    )
```
    3. Convert model `past_key_values` → tuple and call `method.process_prefill()`
    4. Enter decode loop: for each step call model with `past_key_values`, convert → tuple, call `method.process_step()`, update `past_key_values`
    5. Collect metrics (peak memory, kv size, latency, throughput, etc.)

Core library layout
- [benchmark/](benchmark/): dataset loaders and benchmark orchestration
  - [benchmark/runner.py](benchmark/runner.py): generation and PPL helpers
  - [benchmark/datasets.py](benchmark/datasets.py): dataset definitions and loaders
```py
def __init__(self, config, tokenizer):
    self.config = config
    self.tokenizer = tokenizer
    self.synthetic_prompts = []      # list of (prompt_str, seq_len_target)
    self.wikitext_examples = []      # list of str (raw text for ppl eval)
    self.longbench_examples = {}     # task_name -> list of dicts
```
  - [benchmark/metrics.py](benchmark/metrics.py): metric computations and helpers
- [methods/](methods/): KV-cache method implementations and shared interface
  - [methods/base.py](methods/base.py): defines the abstract `MethodWrapper` interface with hooks `process_prefill()`, `process_step()`, `reset()`, optional `needs_attention_weights` and `get_kv_size_bytes()`.
  - [methods/baseline.py](methods/baseline.py): baseline (no-op / vanilla) KV behavior
  - [methods/kivi_quant.py](methods/kivi_quant.py): KIVI quantization implementation (2-bit asymmetric quantization)
  - [methods/topk_selection.py](methods/topk_selection.py): top-k / selection style approaches

Configs & utilities
- [configs/default.yaml](configs/default.yaml): default experiment configuration (model, decode params, dataset, etc.)
- [requirements.txt](requirements.txt): Python dependencies for running benchmarks

How the pieces interact (global picture)
- `run_benchmark.py` loads config and datasets → for each method:
  - instantiate a `MethodWrapper` implementation from `methods/`
  - call `generate_with_method()` in [benchmark/runner.py](benchmark/runner.py)
    - `generate_with_method()` runs model prefill → `method.process_prefill()` → decode loop where each step calls `method.process_step()`
  - collect metrics and (optionally) compute perplexity via `compute_method_perplexity()`

Notes
- Implementations operate by transforming the `past_key_values` representation (runner converts between Transformers `DynamicCache` and plain tuples). See the helpers `_cache_to_tuple()` / `_tuple_to_cache()` in [benchmark/runner.py](benchmark/runner.py).
- `methods/base.py` documents memory-counting via `get_kv_size_bytes()` so reported KV sizes are method-aware.

References & citations
- See [README.md](README.md) for citations (KIVI, SnapKV, xKV, TokenSelect) and additional context.

Next steps
- Run `python run_benchmark.py` or ask me to prepare a commit / run tests.
