import time
import torch
from tqdm import tqdm


# ── DynamicCache ↔ tuple conversion ──────────────────────────────────────────
# Transformers ≥ 4.36 introduced DynamicCache as the default return type for
# past_key_values. Our MethodWrapper implementations work with plain tuples of
# (K, V) tensors.  These helpers bridge the gap so runner.py works with any
# version of HuggingFace Transformers (4.x and 5.x).

def _cache_to_tuple(past_kv):
    """DynamicCache → tuple of (K, V) per layer.  No-op if already a tuple."""
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            return tuple(
                (past_kv.key_cache[i], past_kv.value_cache[i])
                for i in range(len(past_kv.key_cache))
            )
    except ImportError:
        pass
    return past_kv


def _tuple_to_cache(kv_tuple):
    """Tuple of (K, V) → DynamicCache so the model's next forward() is happy."""
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(kv_tuple, tuple) and len(kv_tuple) > 0:
            cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(kv_tuple):
                cache.update(k, v, layer_idx)
            return cache
    except ImportError:
        pass
    return kv_tuple


def generate_with_method(model, tokenizer, method, prompt,
                         max_new_tokens=200, device='cuda'):
    """
    Unified generation loop shared by ALL methods.
    No method reimplements this loop.

    Returns: (generated_text: str, metrics: dict)
    """
    method.reset()
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_len = inputs['input_ids'].shape[1]

    # Memory reset
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # ── PREFILL ──────────────────────────────────────────────
    t0 = time.perf_counter()

    # Only request attention weights if the method needs them (SnapKV).
    # output_attentions=True materialises (batch, heads, seq, seq) per layer
    # which OOMs at seq_len >= ~6000 on 80 GB.
    need_attn = getattr(method, 'needs_attention_weights', False)

    with torch.no_grad():
        prefill_out = model(
            **inputs,
            use_cache=True,
            output_attentions=need_attn,
        )

    # Convert DynamicCache → tuple for methods, then back for the model
    past_kv_tuple = _cache_to_tuple(prefill_out.past_key_values)
    attn_weights = prefill_out.attentions if need_attn else None
    past_kv_tuple = method.process_prefill(
        past_kv_tuple,
        attention_weights=attn_weights,
    )
    past_kv = _tuple_to_cache(past_kv_tuple)

    torch.cuda.synchronize(device)
    ttft = time.perf_counter() - t0

    # ── DECODE LOOP ──────────────────────────────────────────
    next_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = []

    # BUG-2 fix (Option B / StreamingLLM-style position injection).
    # When a method gathers a sparse subset of the cache (TopK / Hybrid), the
    # cache shrinks from S_full to S_sel, but the K tensors keep their
    # *original* RoPE phases. HF would otherwise rotate Q at the truncated
    # cache length, breaking the relative-offset assumption RoPE is built on.
    # We instead pass `position_ids = [[true_pos]]` so HF rotates Q (and the
    # new K being appended) at the *true* absolute position, matching the
    # phases already baked into the cache. Methods set
    # `apply_rope_correction=False` so the cache is left untouched.
    true_pos = input_len  # absolute position of the first decode token

    t_decode_start = time.perf_counter()

    for step in range(max_new_tokens):
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token.item())

        position_ids = torch.tensor(
            [[true_pos]], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            step_out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
                position_ids=position_ids,
                output_attentions=False,  # not needed during decode
            )

        # Convert for method hooks, then back for the model
        step_kv_tuple = _cache_to_tuple(step_out.past_key_values)
        step_kv_tuple = method.process_step(
            step_kv_tuple,
            step=step,
            attention_weights=None,
        )
        past_kv = _tuple_to_cache(step_kv_tuple)

        # BUG-13 fix: removed dead `update_full_cache` branch — no method
        # implements that hook; the new-token append happens inside each
        # method's own `process_step` (e.g. TopKMethod._update_full_cache,
        # KIVIMethod's residual/overflow/block ingestion). The vestigial
        # branch caused confusion during the audit and never fired in
        # practice. See AUDIT_REPORT.md (BUG-13).

        next_token = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        true_pos += 1

    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    # ── METRICS ──────────────────────────────────────────────
    n_generated = len(generated_ids)
    total_time = t_end - t0
    decode_time = t_end - t_decode_start

    # get_kv_size_bytes expects tuples
    kv_bytes = method.get_kv_size_bytes(_cache_to_tuple(past_kv))

    metrics = {
        'peak_memory_gb':        torch.cuda.max_memory_allocated(device) / 1e9,
        'kv_cache_mb':           kv_bytes / 1e6,
        'ttft_ms':               ttft * 1000,
        'total_time_s':          total_time,
        'decode_time_s':         decode_time,
        'tokens_generated':      n_generated,
        'throughput_tps':        n_generated / decode_time if decode_time > 0 else 0,
        'per_token_latency_ms':  decode_time / n_generated * 1000 if n_generated > 0 else 0,
        'input_len':             input_len,
    }

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, metrics


def compute_method_perplexity(model, tokenizer, method, texts,
                              device='cuda', max_length=512):
    """
    Measure perplexity *through* a method's modified KV cache — token-by-token.

    For each text:
      1. Run prefill on the first half → get KV cache
      2. Apply method.process_prefill() to compress/modify KV
      3. For each remaining token, run a single-token forward through the
         method's `process_step`-produced cache, then accumulate the NLL
         of the *next* token under that distribution.

    This actually exercises the decode-time path that ablation methods
    care about. The previous implementation ran a single bulk forward on
    the whole second half, which bypassed `process_step` entirely — that
    is why PPL of `kivi`, `kivi_topk` and `kivi_topk_c` came out *exactly*
    equal in `ppl_sanity/long_context.csv` (they all silently fell back
    to KIVI's prefill reconstruction). See AUDIT_REPORT.md (BUG-3).

    Baseline should match vanilla PPL; lossy methods will be higher.
    """
    import math

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    need_attn = getattr(method, 'needs_attention_weights', False)

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors='pt',
                truncation=True, max_length=max_length,
            ).to(device)
            ids = enc['input_ids']
            if ids.shape[1] < 4:
                continue

            # Split: first half for prefill, second half for token-by-token eval.
            split = ids.shape[1] // 2
            prefix_ids = ids[:, :split]
            target_ids = ids[:, split:]                  # (1, T_eval)
            T_eval = target_ids.shape[1]
            if T_eval < 2:
                continue

            # Prefill on prefix.
            method.reset()
            prefill_out = model(
                input_ids=prefix_ids,
                use_cache=True,
                output_attentions=need_attn,
            )

            past_kv_tuple = _cache_to_tuple(prefill_out.past_key_values)
            attn_weights = prefill_out.attentions if need_attn else None
            past_kv_tuple = method.process_prefill(
                past_kv_tuple,
                attention_weights=attn_weights,
            )
            past_kv = _tuple_to_cache(past_kv_tuple)

            # Score the very first eval token against the prefill logits.
            # `prefill_out.logits[:, -1, :]` predicts the token at position
            # `split` — i.e. `target_ids[:, 0]`.
            first_log_probs = torch.log_softmax(
                prefill_out.logits[:, -1, :], dim=-1,
            )
            total_nll += -first_log_probs[0, target_ids[0, 0]].item()
            total_tokens += 1

            # Step through the remaining eval tokens, calling process_step
            # so that the method's selection / quantisation / re-rotation
            # path is exercised between every forward.
            #
            # BUG-2 fix: pass position_ids so HF rotates Q (and the new K) at
            # the true absolute position, matching the original-position
            # phases already baked into the (possibly sparse) cache. See the
            # generate_with_method comment for the full rationale.
            for t in range(T_eval - 1):
                input_token = target_ids[:, t : t + 1]
                # Original-sequence position of `input_token`.
                true_pos = split + t
                position_ids = torch.tensor(
                    [[true_pos]], dtype=torch.long, device=device,
                )
                step_out = model(
                    input_ids=input_token,
                    past_key_values=past_kv,
                    use_cache=True,
                    position_ids=position_ids,
                    output_attentions=False,
                )

                step_kv_tuple = _cache_to_tuple(step_out.past_key_values)
                step_kv_tuple = method.process_step(
                    step_kv_tuple,
                    step=t,
                    attention_weights=None,
                )
                past_kv = _tuple_to_cache(step_kv_tuple)

                # `step_out.logits[:, -1, :]` predicts `target_ids[:, t+1]`.
                log_probs = torch.log_softmax(
                    step_out.logits[:, -1, :], dim=-1,
                )
                total_nll += -log_probs[0, target_ids[0, t + 1]].item()
                total_tokens += 1

    if total_tokens == 0:
        return float('inf')

    return math.exp(total_nll / total_tokens)
