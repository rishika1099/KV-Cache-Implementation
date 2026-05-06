"""
Custom Llama model with TokenSelect (Wu et al., EMNLP 2025) attention.

Unlike the wrapper approach in topk_selection.py, this patches the attention
layer directly. This gives access to real Q tensors, correct RoPE positions,
and per-layer independent selection during decode.

Scoring uses a Triton kernel that fuses Q·K^T across all query heads and
aggregates scores without materialising the expanded K tensor (no repeat_kv).

Usage:
    from transformers import AutoConfig
    from methods.llama_topk_model import LlamaForCausalLM_TopK

    config = AutoConfig.from_pretrained(model_name)
    config.topk_K = 4096
    config.topk_n_sink = 128
    config.topk_n_local = 4096
    model = LlamaForCausalLM_TopK.from_pretrained(
        model_name, config=config, torch_dtype=torch.float16, device_map="cuda"
    )
"""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ── Triton scoring kernel ────────────────────────────────────────────────────
# Computes aggregated attention scores for token selection.
#
# For each KV position j in [0, SEQ_LEN):
#   score[j] = sum_over_q_heads( softmax( Q[h,:] · K[kv_head(h), j, :] / sqrt(d) ) )
#
# The kernel tiles over the head dimension (BLOCK_D) and KV positions (BLOCK_S).
# It handles GQA by iterating over all Q heads that share each KV head.
# Output: (SEQ_LEN,) float32 aggregated scores ready for topk.

if HAS_TRITON:
    @triton.jit
    def _topk_score_kernel(
        Q_ptr,          # (num_q_heads, head_dim)  — single query token
        K_ptr,          # (num_kv_heads, seq_len, head_dim)
        Out_ptr,        # (seq_len,) — aggregated scores
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        seq_len,
        head_dim: tl.constexpr,
        inv_sqrt_d,     # 1/sqrt(head_dim), float32
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        s_off = pid * BLOCK_S + tl.arange(0, BLOCK_S)  # KV positions
        s_mask = s_off < seq_len

        # Accumulate scores across all Q heads
        acc = tl.zeros([BLOCK_S], dtype=tl.float32)
        groups_per_kv = num_q_heads // num_kv_heads

        for kv_h in range(num_kv_heads):
            # Load K[kv_h, s_off, :] — shape (BLOCK_S, BLOCK_D)
            d_off = tl.arange(0, BLOCK_D)
            d_mask = d_off < head_dim
            # K layout: (num_kv_heads, seq_len, head_dim) — row-major
            k_ptrs = K_ptr + kv_h * seq_len * head_dim + s_off[:, None] * head_dim + d_off[None, :]
            k_vals = tl.load(k_ptrs, mask=s_mask[:, None] & d_mask[None, :], other=0.0)

            for g in range(groups_per_kv):
                q_h = kv_h * groups_per_kv + g
                # Load Q[q_h, :] — shape (BLOCK_D,)
                q_ptrs = Q_ptr + q_h * head_dim + d_off
                q_vals = tl.load(q_ptrs, mask=d_mask, other=0.0)

                # dot product: (BLOCK_S,)
                dots = tl.sum(k_vals * q_vals[None, :], axis=1) * inv_sqrt_d

                # Online softmax contribution — we use exp and normalise later
                dots = tl.where(s_mask, dots, float('-inf'))
                dots_max = tl.max(dots, axis=0)
                exp_dots = tl.exp(dots - dots_max)
                exp_sum = tl.sum(exp_dots, axis=0)
                # Normalised softmax scores for this head
                softmax_scores = exp_dots / (exp_sum + 1e-8)
                acc += softmax_scores

        tl.store(Out_ptr + s_off, acc, mask=s_mask)

    def triton_topk_score(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Compute aggregated attention scores using the Triton kernel.

        Args:
            query: (num_q_heads, head_dim) — single decode token query
            key:   (num_kv_heads, seq_len, head_dim) — middle KV cache slice

        Returns:
            scores: (seq_len,) float32 aggregated soft-voting scores
        """
        num_q_heads, head_dim = query.shape
        num_kv_heads, seq_len, _ = key.shape
        inv_sqrt_d = 1.0 / math.sqrt(head_dim)

        out = torch.empty(seq_len, dtype=torch.float32, device=query.device)

        BLOCK_D = triton.next_power_of_2(head_dim)
        BLOCK_S = 128
        grid = (triton.cdiv(seq_len, BLOCK_S),)

        _topk_score_kernel[grid](
            query.contiguous(), key.contiguous(), out,
            num_q_heads, num_kv_heads, seq_len, head_dim,
            inv_sqrt_d,
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        )
        return out

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import DynamicCache


# ── Attention with TokenSelect ────────────────────────────────────────────────

class LlamaAttention_TopK(nn.Module):
    """Llama attention with per-layer TokenSelect during decode."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True

        self.topk_K = getattr(config, 'topk_K', 4096)
        self.topk_n_sink = getattr(config, 'topk_n_sink', 128)
        self.topk_n_local = getattr(config, 'topk_n_local', 4096)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[2]

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # ─── DECODE ───────────────────────────────────────────
            K_full = torch.cat([past_key_value[0], key_states], dim=2)
            V_full = torch.cat([past_key_value[1], value_states], dim=2)
            kv_seq_len = K_full.shape[2]

            total_budget = self.topk_n_sink + self.topk_K + self.topk_n_local

            if kv_seq_len <= total_budget:
                # Full attention — cache fits in budget
                K_exp = repeat_kv(K_full, self.num_key_value_groups)
                V_exp = repeat_kv(V_full, self.num_key_value_groups)
                attn_output = F.scaled_dot_product_attention(
                    query_states, K_exp, V_exp, is_causal=False,
                )
            else:
                # TokenSelect: score → select → attend on subset
                attn_output = self._topk_attention(query_states, K_full, V_full, kv_seq_len)

            past_key_value = (K_full, V_full, kv_seq_len) if use_cache else None

        else:
            # ─── PREFILL ──────────────────────────────────────────
            K_exp = repeat_kv(key_states, self.num_key_value_groups)
            V_exp = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                query_states, K_exp, V_exp, is_causal=True,
            )
            past_key_value = (key_states, value_states, kv_seq_len) if use_cache else None

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    def _topk_attention(self, query_states, K_full, V_full, kv_seq_len):
        """Score middle tokens with Triton kernel, select top-K, attend to subset with SDPA."""
        device = K_full.device
        sink_end = self.topk_n_sink
        local_start = kv_seq_len - self.topk_n_local
        middle_len = local_start - sink_end

        # Score middle tokens
        middle_K = K_full[:, :, sink_end:local_start, :]  # (1, num_kv_heads, middle_len, d)

        if HAS_TRITON:
            # Triton path: fused Q·K scoring + softmax aggregation (no repeat_kv)
            # query_states: (1, num_q_heads, 1, d) → (num_q_heads, d)
            q_2d = query_states[0, :, 0, :]
            k_3d = middle_K[0]  # (num_kv_heads, middle_len, d)
            agg = triton_topk_score(q_2d, k_3d)  # (middle_len,)
        else:
            # PyTorch fallback
            middle_K_exp = repeat_kv(middle_K, self.num_key_value_groups)
            scores = torch.matmul(
                query_states, middle_K_exp.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            scores = scores.squeeze(2)
            agg = torch.softmax(scores, dim=-1).sum(dim=1)[0]

        # Flat top-K
        k_actual = min(self.topk_K, middle_len)
        topk_local = agg.topk(k_actual).indices
        topk_global = topk_local + sink_end

        # Build sorted index: sink ∪ topk ∪ local
        idx = torch.cat([
            torch.arange(0, sink_end, device=device),
            topk_global,
            torch.arange(local_start, kv_seq_len, device=device),
        ])
        idx, _ = idx.sort()

        # Gather selected K, V and attend with SDPA (fused kernel)
        K_sel = K_full[:, :, idx, :]
        V_sel = V_full[:, :, idx, :]
        K_sel_exp = repeat_kv(K_sel, self.num_key_value_groups)
        V_sel_exp = repeat_kv(V_sel, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states, K_sel_exp, V_sel_exp, is_causal=False,
        )
        return attn_output


# ── Decoder layer / Model / CausalLM (mirror KIVI pattern) ───────────────────

class LlamaDecoderLayer_TopK(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention_TopK(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False, use_cache=False, **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache, **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class LlamaModel_TopK(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer_TopK(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, use_cache=None,
        output_attentions=None, output_hidden_states=None, return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Specify input_ids or inputs_embeds")

        # Cache format: tuple of (K_full, V_full, kv_seq_len) per layer
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,  # prefill uses is_causal; decode q_len=1 needs no mask
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
        )


class LlamaForCausalLM_TopK(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_TopK(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None,
        use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None,
    ):
        # Convert DynamicCache if HF injected one
        if isinstance(past_key_values, DynamicCache):
            past_key_values = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).float()

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[..., 1:].contiguous().view(-1).to(logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, **kwargs,
    ):
        if isinstance(past_key_values, DynamicCache):
            past_key_values = None

        if past_key_values is not None:
            past_length = past_key_values[0][2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs
