
import json
import sys
from pathlib import Path

import modal

app = modal.App("kv-benchmark-ppl")

_base = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.44.2",
        "datasets>=2.18.0,<4.0",
        "accelerate>=0.27.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    )
    .add_local_dir(str(_base / "methods"),   "/app/methods")
    .add_local_dir(str(_base / "benchmark"), "/app/benchmark")
    .add_local_dir(str(_base / "configs"),   "/app/configs")
)

model_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)
HF_CACHE_PATH = Path("/root/.cache/huggingface")


def _bootstrap():
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def _set_seeds(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_method(name, cfg):
    _bootstrap()
    from methods.baseline        import BaselineMethod
    from methods.kivi_quant      import KIVIMethod
    from methods.snapkv_eviction import SnapKVMethod
    from methods.topk_selection  import TopKMethod

    if name == "baseline":
        return BaselineMethod()
    elif name == "kivi":
        return KIVIMethod(
            bits=cfg.get("bits", 4),
            residual_length=cfg.get("residual_length", 128),
            group_size=cfg.get("group_size", 32),
        )
    elif name == "snapkv":
        return SnapKVMethod(
            budget_ratio=cfg.get("budget_ratio", 0.4),
            sink_size=cfg.get("sink_size", 4),
            observation_window=cfg.get("observation_window", 32),
        )
    elif name == "topk":
        return TopKMethod(
            K=cfg.get("K", 512),
            n_sink=cfg.get("n_sink", 128),
            n_local=cfg.get("n_local", 512),
            refresh_interval=cfg.get("refresh_interval", 50),
            page_size=cfg.get("page_size", 64),
            cache_similarity_threshold=cfg.get("cache_similarity_threshold", 0.95),
            chunk_size=cfg.get("chunk_size", 2048),
        )
    else:
        raise ValueError(f"Unknown method: {name}")


@app.function(
    image=image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={str(HF_CACHE_PATH): model_cache_vol},
    timeout=3600,
    memory=65536,
)
def run_ppl(method_name: str, method_cfg: dict, wikitext_texts: list,
            model_name: str, seed: int = 42) -> dict:
    _bootstrap()
    _set_seeds(seed)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[ppl] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    method = _build_method(method_name, method_cfg)
    cfg_str = json.dumps(method_cfg, sort_keys=True)

    from benchmark.runner import compute_method_perplexity
    ppl = compute_method_perplexity(
        model, tokenizer, method, wikitext_texts,
        device="cuda", max_length=512,
    )
    print(f"[ppl] {method_name} {cfg_str}: PPL={ppl:.3f}")
    return {"method": method_name, "config": method_cfg, "perplexity": ppl}


@app.local_entrypoint()
def main():
    from datasets import load_dataset

    model_name = "meta-llama/Llama-2-7b-hf"

    # Load wikitext
    print("Loading WikiText-103 test set...")
    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    wikitext_texts = []
    for ex in wt:
        if ex["text"].strip() and not ex["text"].strip().startswith("="):
            wikitext_texts.append(ex["text"].strip())
            if len(wikitext_texts) >= 20:
                break
    print(f"  {len(wikitext_texts)} examples loaded")

    # All 8 method configs (no xKV)
    configs = [
        ("baseline", {}),
        ("kivi",   {"bits": 4, "residual_length": 128, "group_size": 32}),
        ("kivi",   {"bits": 2, "residual_length": 128, "group_size": 32}),
        ("snapkv", {"budget_ratio": 0.2, "sink_size": 4, "observation_window": 32}),
        ("snapkv", {"budget_ratio": 0.4, "sink_size": 4, "observation_window": 32}),
        ("snapkv", {"budget_ratio": 0.6, "sink_size": 4, "observation_window": 32}),
        ("topk",   {"K": 512,  "n_sink": 128, "n_local": 512, "refresh_interval": 0}),
        ("topk",   {"K": 1024, "n_sink": 128, "n_local": 512, "refresh_interval": 50}),
        ("topk",   {"K": 2048, "n_sink": 128, "n_local": 512, "refresh_interval": 50}),
    ]

    # Launch all in parallel
    handles = {}
    for method_name, method_cfg in configs:
        cfg_str = json.dumps(method_cfg, sort_keys=True)
        key = f"{method_name} {cfg_str}"
        print(f"  Spawning: {key}")
        h = run_ppl.spawn(method_name, method_cfg, wikitext_texts, model_name)
        handles[key] = h

    # Collect results
    ppl_results = []
    for key, h in handles.items():
        try:
            res = h.get()
            ppl_results.append(res)
            print(f"  OK {key}  PPL={res['perplexity']:.3f}")
        except Exception as e:
            print(f"  FAIL {key}: {e}")

    # Write results
    out = Path(__file__).parent / "results" / "ppl_results.json"
    with open(out, "w") as f:
        json.dump(ppl_results, f, indent=2)
    print(f"\n{len(ppl_results)} PPL results written to {out}")

    for r in sorted(ppl_results, key=lambda x: x['perplexity']):
        print(f"  {r['method']:10s} {json.dumps(r['config']):50s}  PPL={r['perplexity']:.3f}")
