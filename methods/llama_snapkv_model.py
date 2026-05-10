"""
llama_snapkv_model.py — SnapKV via monkey-patch on stock LlamaForCausalLM.

Usage:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa",
    )
    snapkv_monkey_patch(model,
        window_size=32, max_capacity_prompt=409,
        kernel_size=7, pooling="avgpool", sink_size=0,
    )

Decode runs 100% stock HF code — same DynamicCache, same SDPA kernel, same
Python paths. Only prefill is intercepted: after computing the full causal
attention output, K/V are compressed before being stored in the DynamicCache.

Two additional patches are applied:
  - model.model.forward: resets per-layer kv_seq_len counters on new sequences
    and injects correct position_ids so RoPE is applied at the true (original,
    pre-compression) token positions during decode.
  - model.prepare_inputs_for_generation: uses kv_seq_len for past_length so
    .generate() trims input_ids correctly.
"""
import math
import torch
import torch.nn.functional as F
from typing import Optional
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import DynamicCache


# =============================================================================
# SnapKV compression kernel (unchanged)
# =============================================================================

def _snapkv_compress(
    key_states: torch.Tensor,            # (B, num_kv_heads,   L, D)
    value_states: torch.Tensor,          # (B, num_kv_heads,   L, D)
    query_states: torch.Tensor,          # (B, num_attn_heads, L, D)
    num_key_value_groups: int,
    window_size: int,
    max_capacity_prompt: int,
    kernel_size: int,
    pooling: str,
    sink_size: int = 0,
):
    bsz, num_kv_heads, q_len, head_dim = key_states.shape
    num_attn_heads = query_states.shape[1]
    assert num_attn_heads == num_kv_heads * num_key_value_groups

    key_states_attn = repeat_kv(key_states, num_key_value_groups)

    obs_q = query_states[..., -window_size:, :]
    attn_weights = torch.matmul(obs_q, key_states_attn.transpose(2, 3)) / math.sqrt(head_dim)

    mask = torch.full(
        (window_size, window_size),
        torch.finfo(attn_weights.dtype).min,
        device=attn_weights.device,
    )
    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    attn_weights[..., -window_size:, -window_size:] += mask[None, None]

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    vote = attn_weights[..., :, : q_len - window_size].sum(dim=-2)

    if pooling == "avgpool":
        vote_pooled = F.avg_pool1d(vote, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    elif pooling == "maxpool":
        vote_pooled = F.max_pool1d(vote, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    k_select = max(1, max_capacity_prompt - window_size - sink_size)
    indices = vote_pooled.topk(k_select, dim=-1).indices  # (B, num_attn_heads, k_select)

    if sink_size > 0:
        sink_idx = torch.arange(sink_size, device=indices.device, dtype=indices.dtype
                                ).view(1, 1, -1).expand(bsz, num_attn_heads, -1)
        indices = torch.cat([sink_idx, indices], dim=-1)

    if num_key_value_groups > 1:
        n_rep   = num_key_value_groups
        k_total = indices.shape[-1]
        indices = indices.view(bsz, num_kv_heads, n_rep, k_total)
        indices = indices.reshape(bsz, num_kv_heads, n_rep * k_total)[..., :k_total]

    idx_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    prefix_len   = q_len - window_size
    k_selected   = key_states[..., :prefix_len, :].gather(dim=2, index=idx_expanded)
    v_selected   = value_states[..., :prefix_len, :].gather(dim=2, index=idx_expanded)

    k_compressed = torch.cat([k_selected, key_states[..., -window_size:, :]],   dim=2)
    v_compressed = torch.cat([v_selected, value_states[..., -window_size:, :]], dim=2)
    return k_compressed, v_compressed


# =============================================================================
# Monkey-patch helpers
# =============================================================================

def _patch_attn_forward(attn):
    """
    Replace attn.forward with a thin wrapper:
      - prefill (kv_seq_len == 0): compute full SDPA, compress K/V, store in cache
      - decode: call the saved original forward unchanged
    """
    _orig = attn.forward  # bound method — called as-is for decode

    def patched_forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        if attn.kv_seq_len != 0:
            # ── DECODE: 100% stock HF ──────────────────────────────────────
            attn.kv_seq_len += 1
            return _orig(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache, **kwargs,
            )

        # ── PREFILL ────────────────────────────────────────────────────────
        bsz, q_len, _ = hidden_states.size()

        query_states = attn.q_proj(hidden_states).view(
            bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(
            bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(
            bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

        cos, sin = attn.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Full causal attention for the output
        k_full = repeat_kv(key_states,   attn.num_key_value_groups)
        v_full = repeat_kv(value_states, attn.num_key_value_groups)
        attn_out = F.scaled_dot_product_attention(
            query_states, k_full, v_full,
            attn_mask=None, dropout_p=0.0, is_causal=True,
        )
        del k_full, v_full

        # Compress K/V and store in DynamicCache
        if use_cache and past_key_value is not None:
            if q_len > attn.snapkv_max_capacity_prompt and q_len > attn.snapkv_window_size:
                k_c, v_c = _snapkv_compress(
                    key_states, value_states, query_states,
                    num_key_value_groups=attn.num_key_value_groups,
                    window_size=attn.snapkv_window_size,
                    max_capacity_prompt=attn.snapkv_max_capacity_prompt,
                    kernel_size=attn.snapkv_kernel_size,
                    pooling=attn.snapkv_pooling,
                    sink_size=attn.snapkv_sink_size,
                )
                del key_states, value_states
                past_key_value.update(k_c, v_c, attn.layer_idx)
            else:
                past_key_value.update(key_states, value_states, attn.layer_idx)

        attn.kv_seq_len = q_len

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, attn.hidden_size)
        attn_out = attn.o_proj(attn_out)
        return attn_out, None, past_key_value

    attn.forward = patched_forward


def _patch_llama_model_forward(model):
    """
    Wrap model.model.forward to:
      1. Reset kv_seq_len counters at the start of each new sequence.
      2. Inject correct position_ids during decode so RoPE uses the true
         (original, pre-compression) token positions.
    """
    _orig = model.model.forward  # bound method

    def new_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Reset counters at the start of every new sequence (prefill).
        if past_key_values is None or (
            isinstance(past_key_values, DynamicCache)
            and past_key_values.get_seq_length() == 0
        ):
            for layer in model.model.layers:
                layer.self_attn.kv_seq_len = 0

        # During decode, DynamicCache.get_seq_length() returns the compressed
        # length (e.g. 409), not the original (e.g. 1024).  We need the true
        # position for RoPE, so inject position_ids from kv_seq_len.
        if (
            position_ids is None
            and past_key_values is not None
            and isinstance(past_key_values, DynamicCache)
        ):
            cached_len   = past_key_values.get_seq_length()
            original_len = model.model.layers[0].self_attn.kv_seq_len
            if original_len > cached_len:           # compression is active
                src = input_ids if input_ids is not None else inputs_embeds
                seq_len = src.shape[1]
                position_ids = torch.arange(
                    original_len, original_len + seq_len,
                    dtype=torch.long, device=src.device,
                ).unsqueeze(0)

        return _orig(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    model.model.forward = new_forward


def _patch_prepare_inputs(model):
    """
    Replace prepare_inputs_for_generation so that .generate() trims input_ids
    using the true (original) sequence length rather than the compressed cache
    length, and resets kv_seq_len on the very first call.
    """
    def new_prepare(
        input_ids, past_key_values=None,
        attention_mask=None, inputs_embeds=None, **kwargs,
    ):
        if past_key_values is None:
            for layer in model.model.layers:
                layer.self_attn.kv_seq_len = 0
            past_length = 0
        else:
            past_length = model.model.layers[0].self_attn.kv_seq_len

        if input_ids.shape[1] > past_length:
            input_ids = input_ids[:, past_length:]
        else:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = (
            {"inputs_embeds": inputs_embeds}
            if inputs_embeds is not None and past_key_values is None
            else {"input_ids": input_ids}
        )
        model_inputs.update({
            "position_ids":    position_ids,
            "past_key_values": past_key_values,
            "use_cache":       kwargs.get("use_cache"),
            "attention_mask":  attention_mask,
        })
        return model_inputs

    model.prepare_inputs_for_generation = new_prepare


# =============================================================================
# Public API
# =============================================================================

def snapkv_monkey_patch(
    model,
    window_size: int = 32,
    max_capacity_prompt: int = 512,
    kernel_size: int = 7,
    pooling: str = "avgpool",
    sink_size: int = 0,
):
    """
    Patch a stock LlamaForCausalLM (loaded with attn_implementation='sdpa')
    to apply SnapKV KV-cache compression during prefill.

    After calling this, decode is 100% identical to the stock HF model —
    same DynamicCache, same SDPA kernels, same Python execution path.
    """
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.kv_seq_len              = 0
        attn.snapkv_window_size      = window_size
        attn.snapkv_max_capacity_prompt = max_capacity_prompt
        attn.snapkv_kernel_size      = kernel_size
        attn.snapkv_pooling          = pooling
        attn.snapkv_sink_size        = sink_size
        _patch_attn_forward(attn)

    _patch_llama_model_forward(model)
    _patch_prepare_inputs(model)
