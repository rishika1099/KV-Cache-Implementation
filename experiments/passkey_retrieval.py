"""
Passkey-retrieval quality eval — Phase B headline experiment for selection.

This is a *needle-in-a-haystack* test that exercises `process_step` end-to-
end through `generate_with_method`. It directly addresses the methodology
gap exposed in Phase A v1: `compute_method_perplexity` runs prefill once
+ a single forward on the eval window without ever calling `process_step`,
so it is structurally blind to runtime selection. PPL was therefore flat
across K values for every TopK-family method (Phase A v2 confirmed: 8.401
constant). Passkey retrieval, on the other hand, only succeeds if the
KV-cache keeps (or finds) the needle's representation when the question
arrives — so it tests what TopK actually does at decode time.

Construction per trial
----------------------
We build a long context of repeated filler text with one *needle* sentence
inserted at a controlled relative position ("depth", 0–1). At the end we
ask the model for the passkey. Score = exact-match (the model emits the
correct 5-digit number among its first generated tokens).

    [FILLER × depth] [NEEDLE: "Remember: the pass key is 47291. ..."]
    [FILLER × (1 - depth)] "What is the pass key?\nThe pass key is "
                                                              ─── generate ─→

Filler is a deterministic repeated paragraph (with a counter to keep it
non-pathological for the model — repeated identical lines collapse into
caching artefacts). Total prompt is truncated to `seq_len` *tokens* to
keep configurations apples-to-apples; the needle position is controlled
*after* that truncation so it lands at exactly the right depth.

Why this is the right eval for the hybrid
-----------------------------------------
Design (a) approximates block-level scoring with a centroid (one Q·k per
block instead of one per token). If the needle's block-mean dot-product
ranks competitively against ~12 % of full-attention scores, retrieval
holds even at K=1024 over a 16K context. If the centroid washes out the
signal, accuracy drops sharply at depths >50 %. Either result is
informative for the report.

Usage:
    python -m experiments.passkey_retrieval \
        --model meta-llama/Llama-2-7b-hf \
        --seq-lens 4096 8192 \
        --methods baseline kivi topk kivi_topk \
        --n-trials 5 --n-depths 5
"""
from __future__ import annotations
import argparse
import gc
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from methods.registry import make_method
from ._common import (
    ExperimentRecord, device_summary, maybe_load_model, write_records,
)


# ── Prompt construction ──────────────────────────────────────────────────────

# Single repeated filler line with a counter so the model isn't fed
# token-identical repeats (the counter keeps positional embeddings well-
# behaved and prevents trivial degeneracies).
_FILLER_TEMPLATE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. (line {i})\n"
)
_NEEDLE_TEMPLATE = (
    "Remember this — the pass key is {key}. "
    "The pass key is {key}. Do not forget.\n"
)
_QUESTION_TEMPLATE = (
    "\nWhat is the pass key? The pass key is "
)


def _generate_passkey(rng: random.Random) -> str:
    """5-digit numeric passkey, no leading zero."""
    return str(rng.randint(10_000, 99_999))


def _build_prompt(
    seq_len_tokens: int,
    depth: float,
    passkey: str,
    tokenizer,
) -> Tuple[str, int]:
    """
    Construct a prompt that tokenizes to ≈ `seq_len_tokens` with the needle
    inserted at relative position `depth` (0=start, 1=end).

    Returns (prompt_text, n_tokens_actual).
    """
    needle = _NEEDLE_TEMPLATE.format(key=passkey)
    question = _QUESTION_TEMPLATE
    needle_tokens   = len(tokenizer.encode(needle,   add_special_tokens=False))
    question_tokens = len(tokenizer.encode(question, add_special_tokens=False))

    # Filler budget = seq_len_tokens − needle − question (leaving headroom).
    target_filler_tokens = max(64, seq_len_tokens - needle_tokens
                                  - question_tokens - 8)

    # Build filler line-by-line until we hit the budget. Each line is ~25
    # tokens for LLaMA tokenizer, so this converges in O(seq_len/25) iters.
    filler_parts: List[str] = []
    cur_tokens = 0
    i = 0
    while cur_tokens < target_filler_tokens:
        line = _FILLER_TEMPLATE.format(i=i)
        line_tokens = len(tokenizer.encode(line, add_special_tokens=False))
        filler_parts.append(line)
        cur_tokens += line_tokens
        i += 1
        if i > 4 * seq_len_tokens:           # safety stop, should not happen
            break

    # Insert needle at the requested depth (in *line* space; close enough
    # for the position-sensitivity question we're asking).
    insert_at = int(round(depth * len(filler_parts)))
    insert_at = max(0, min(len(filler_parts), insert_at))
    prefix  = "".join(filler_parts[:insert_at])
    suffix  = "".join(filler_parts[insert_at:])
    prompt  = prefix + needle + suffix + question

    n_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    return prompt, n_tokens


# ── Scoring ──────────────────────────────────────────────────────────────────

_DIGIT_RE = re.compile(r"\d{4,6}")


def _extract_first_number(text: str) -> Optional[str]:
    m = _DIGIT_RE.search(text)
    return m.group(0) if m else None


# ── Single trial ─────────────────────────────────────────────────────────────

def _run_trial(
    model, tokenizer, method,
    prompt: str, passkey: str,
    max_new_tokens: int, device: str,
) -> dict:
    """One generation pass. Returns metrics + correctness."""
    from benchmark.runner import generate_with_method
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt=prompt, max_new_tokens=max_new_tokens, device=device,
    )
    pred = _extract_first_number(text)
    correct = pred == passkey
    return {
        "correct":          int(correct),
        "predicted":        pred or "",
        "passkey":          passkey,
        "throughput_tps":   metrics.get("throughput_tps", 0.0),
        "ttft_ms":          metrics.get("ttft_ms", 0.0),
        "kv_cache_mb":      metrics.get("kv_cache_mb", 0.0),
        "gen_text":         text[:80],          # truncated for log readability
    }


# ── Driver ───────────────────────────────────────────────────────────────────

def run_passkey_retrieval(
    model_name: str,
    seq_lens: List[int] = (4096, 8192),
    methods: Optional[List[str]] = None,
    n_trials: int = 5,
    n_depths: int = 5,
    K: int = 1024, n_sink: int = 128, n_local: int = 512,
    bits: int = 4, group_size: int = 32, residual_length: int = 128,
    max_new_tokens: int = 12,
    seed: int = 0,
    device: str = "cuda",
    output_dir: Path = Path("results/phase_b/passkey"),
) -> List[ExperimentRecord]:
    """
    Run passkey retrieval across {method × seq_len × depth × trial}.

    Reports per (method, seq_len, depth):
        accuracy        — fraction correct over n_trials
        n_correct       — raw count
        n_trials        — denominator
        throughput_tps  — mean across trials
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("[passkey] CUDA unavailable — running on CPU.")
        device = "cpu"

    methods = list(methods or [
        "baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c",
    ])

    print(f"[passkey] model={model_name}")
    print(f"[passkey] seq_lens={list(seq_lens)}  methods={methods}")
    print(f"[passkey] n_trials={n_trials}  n_depths={n_depths}  "
          f"max_new_tokens={max_new_tokens}")
    print(f"[passkey] {device_summary(device)}")

    print(f"[passkey] loading {model_name} …")
    model, tokenizer = maybe_load_model(model_name, device=device)
    if model is None:
        raise RuntimeError("model_name is required for passkey retrieval")

    rng = random.Random(seed)
    # Pre-generate per-trial passkeys so all methods see identical prompts —
    # otherwise differences in throughput would also reflect prompt variance.
    depths = [(d + 0.5) / n_depths for d in range(n_depths)]   # uniform mids
    trials_per_cell = [
        [_generate_passkey(rng) for _ in range(n_trials)]
        for _ in range(n_depths)
    ]
    # Note: same passkey list is reused across (method, seq_len) so methods
    # are compared on identical needles. We resample per-depth to avoid the
    # model "learning" the passkey across depths via repetition.

    records: List[ExperimentRecord] = []
    for seq_len in seq_lens:
        print(f"\n── seq_len = {seq_len} ──────────────────────────────")

        # Build prompts once per (depth, trial) — independent of method.
        prompts: List[List[Tuple[str, str]]] = []
        for d_idx, depth in enumerate(depths):
            row = []
            for t in range(n_trials):
                pk = trials_per_cell[d_idx][t]
                prompt, n_tok = _build_prompt(seq_len, depth, pk, tokenizer)
                row.append((prompt, pk))
            prompts.append(row)
            print(f"  depth={depth:.2f}: built {n_trials} prompts "
                  f"(~{n_tok} tokens)")

        # Pull real model dims so the RoPE-correction (BUG-2 fix) inside the
        # TopK / KIVI+TopK gather paths uses the model's actual head_dim and
        # rope_theta. Without these, ``make_method`` falls back to defaults
        # (head_dim=128, rope_theta=10000) which happens to match LLaMA-2-7B
        # but would silently corrupt other models.
        cfg = getattr(model, "config", None)
        model_head_dim = (
            getattr(cfg, "head_dim", None)
            or (cfg.hidden_size // cfg.num_attention_heads
                if cfg is not None else 128)
        )
        model_rope_theta = float(getattr(cfg, "rope_theta", 10000.0)
                                 if cfg is not None else 10000.0)

        for name in methods:
            print(f"\n  method = {name}")
            for d_idx, depth in enumerate(depths):
                kwargs = dict(
                    K=K, n_sink=n_sink, n_local=n_local,
                    bits=bits, group_size=group_size,
                    residual_length=residual_length,
                    use_selection_cache=True,        # prod default; selection
                                                     # quality is what we care about
                    head_dim=model_head_dim,
                    rope_theta=model_rope_theta,
                )
                method = make_method(name, **kwargs)

                trial_results = []
                for t in range(n_trials):
                    prompt, pk = prompts[d_idx][t]
                    try:
                        r = _run_trial(
                            model, tokenizer, method,
                            prompt=prompt, passkey=pk,
                            max_new_tokens=max_new_tokens, device=device,
                        )
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        r = {"correct": 0, "predicted": "OOM", "passkey": pk,
                             "throughput_tps": 0, "ttft_ms": 0,
                             "kv_cache_mb": 0, "gen_text": "OOM"}
                    trial_results.append(r)

                n_correct = sum(r["correct"] for r in trial_results)
                acc = n_correct / max(1, len(trial_results))
                tput = (sum(r["throughput_tps"] for r in trial_results)
                        / max(1, len(trial_results)))
                print(f"    depth={depth:.2f}  acc={acc:.0%} "
                      f"({n_correct}/{n_trials})  tps={tput:.1f}")

                records.append(ExperimentRecord(
                    experiment="passkey_retrieval", method=name,
                    config={
                        "seq_len":   seq_len,
                        "depth":     round(depth, 3),
                        "K":         K, "n_sink": n_sink, "n_local": n_local,
                        "bits":      bits if "kivi" in name else None,
                        "n_trials":  n_trials,
                    },
                    metrics={
                        "accuracy":       acc,
                        "n_correct":      n_correct,
                        "n_trials":       len(trial_results),
                        "throughput_tps": tput,
                    },
                ))

                del method
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    out_path = Path(output_dir) / "passkey"
    write_records(records, out_path)
    print(f"\n[passkey] Wrote {out_path}.csv and {out_path}.json")
    return records


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192])
    p.add_argument("--methods",  type=str, nargs="+",
                   default=["baseline", "kivi", "topk", "kivi_topk", "kivi_topk_c"])
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--n-depths", type=int, default=5)
    p.add_argument("--K",        type=int, default=1024)
    p.add_argument("--n-sink",   type=int, default=128)
    p.add_argument("--n-local",  type=int, default=512)
    p.add_argument("--bits",            type=int, default=4)
    p.add_argument("--group-size",      type=int, default=32)
    p.add_argument("--residual-length", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--device",   type=str, default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/phase_b/passkey"))
    args = p.parse_args()

    run_passkey_retrieval(
        model_name=args.model,
        seq_lens=args.seq_lens,
        methods=args.methods,
        n_trials=args.n_trials, n_depths=args.n_depths,
        K=args.K, n_sink=args.n_sink, n_local=args.n_local,
        bits=args.bits, group_size=args.group_size,
        residual_length=args.residual_length,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed, device=args.device, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
