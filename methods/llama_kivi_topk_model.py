"""
methods/llama_kivi_topk_model.py

Custom Llama model combining KIVI quantised KV cache with TokenSelect
(TopK) sparse attention.

How the two methods compose inside the attention module:

  KIVI stores past keys and values in a compact quantised format using
  two Triton/CUDA kernels:
    - triton_quantize_and_pack_along_last_dim  — pack FP16 → int32
    - cuda_bmm_fA_qB_outer                    — Q @ K^T and attn @ V
                                                 directly on packed integers

  This means during every decode step the attention scores (Q @ K^T) for
  all quantised tokens are computed WITHOUT ever materialising the full
  FP16 key cache.  The score tensor has shape (B, nh, 1, total_seq_len).

  TopK selection is applied to these already-computed scores:
    1. Compute attn_weights with KIVI kernels (same as LlamaAttention_KIVI).
    2. Divide the sequence into sink | middle | local regions.
    3. Soft-vote (softmax per head, sum across heads) to rank middle tokens.
    4. Identify the top-K middle token indices (_select_topk).
    5. Gather attn_weights at [sink | top-K middle | local] indices → budget-size slice.
    6. Softmax over budget positions only.
    7. Dequantise ONLY the selected V rows (_gather_v) and do a small matmul.
       Non-selected V tokens are never read from HBM.

  Benefits vs pure KIVI:
    - Quality: fewer low-relevance tokens pollute the context window.
    - V bandwidth: proportional to budget (n_sink+K+n_local), not seq_len.
      At 32 K context with budget=1664: ~20× fewer V token reads.
  Benefits vs pure TopK wrapper:
    - Memory: KV stored quantised (4× at 4-bit, 8× at 2-bit vs FP16).
    - Scoring: uses REAL Q@K scores, not a K-proxy — more accurate selection.
  Note on K: Q@K^T still reads ALL quantised K (necessary to score them).
    Only the V step is sparse.

Config fields required (set on AutoConfig before from_pretrained):
    config.k_bits          : int  (2 or 4)
    config.v_bits          : int  (2 or 4)
    config.group_size      : int  (e.g. 32)
    config.residual_length : int  (must be multiple of group_size, e.g. 128)
    config.topk_K          : int  (middle tokens selected per step, e.g. 1024)
    config.topk_n_sink     : int  (first tokens always kept, e.g. 128)
    config.topk_n_local    : int  (last tokens always kept, e.g. 512)

Usage:
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    config.k_bits = 4;  config.v_bits = 4
    config.group_size = 32;  config.residual_length = 128
    config.topk_K = 1024;  config.topk_n_sink = 128;  config.topk_n_local = 512
    model = LlamaForCausalLM_KIVITopK.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", config=config,
        torch_dtype=torch.float16, device_map="cuda",
    )

Requires: the kivi_gemv CUDA extension (built via methods/kivi_kernels/setup.py).
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from methods.kivi_kernels.new_pack import triton_quantize_and_pack_along_last_dim
from methods.kivi_kernels.matmul import cuda_bmm_fA_qB_outer

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LLAMA_INPUTS_DOCSTRING,
    add_start_docstrings_to_model_forward,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    CrossEntropyLoss,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.cache_utils import DynamicCache


class LlamaAttention_KIVITopK(nn.Module):
    """
    Multi-headed attention with KIVI quantised KV cache and TopK selection.
    Identical to LlamaAttention_KIVI except the decode path applies a
    soft-vote top-K mask to attn_weights before the final softmax.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size         = config.hidden_size
        self.num_heads           = config.num_attention_heads
        self.head_dim            = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta          = config.rope_theta
        self.is_causal           = True

        # KIVI quantisation parameters
        self.k_bits          = config.k_bits
        self.v_bits          = config.v_bits
        self.group_size      = config.group_size
        self.residual_length = config.residual_length

        # TopK selection parameters
        self.topk_K      = config.topk_K
        self.topk_n_sink = config.topk_n_sink
        self.topk_n_local = config.topk_n_local

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim,           bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size,           bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    # ── TopK selection ────────────────────────────────────────────────────────

    def _select_topk(
        self,
        attn_weights: torch.Tensor,
        total_seq_len: int,
    ):
        """
        Identify which token positions to attend to: sink | top-K middle | local.

        attn_weights: (B, nh, 1, total_seq_len) — scaled Q@K^T with causal mask.

        Returns a 1-D LongTensor of absolute indices [0, total_seq_len) ordered
        as [sink ... | selected_middle ... | local ...], or None when the full
        context fits within the budget (no selection needed).
        """
        budget = self.topk_n_sink + self.topk_K + self.topk_n_local
        if total_seq_len <= budget:
            return None

        sink_end     = min(self.topk_n_sink, total_seq_len)
        recent_start = max(total_seq_len - self.topk_n_local, sink_end)
        middle_len   = recent_start - sink_end

        if middle_len <= 0:
            return None

        device = attn_weights.device

        # Soft-vote: aggregate softmax scores across heads to rank middle tokens.
        # Uses real Q@K^T scores — more accurate than a K-proxy.
        middle_scores = attn_weights[0, :, 0, sink_end:recent_start]  # (nh, M)
        aggregated    = F.softmax(middle_scores, dim=-1).sum(dim=0)    # (M,)

        k_actual      = min(self.topk_K, middle_len)
        # Sort selected indices for sequential memory access during gather.
        topk_rel_idx  = aggregated.topk(k_actual).indices.sort().values
        middle_abs_idx = topk_rel_idx + sink_end

        sink_idx  = torch.arange(sink_end, device=device)
        local_idx = torch.arange(recent_start, total_seq_len, device=device)
        return torch.cat([sink_idx, middle_abs_idx, local_idx])

    def _gather_v(
        self,
        selected_idx: torch.Tensor,
        value_states_quant,
        value_states_full: torch.Tensor,
        value_scale,
        value_mn,
        kv_seq_len: int,
    ) -> torch.Tensor:
        """
        Gather value vectors at selected_idx from the quantised + FP16 split.

        V layout:
          tokens [0 .. n_quant)          → value_states_quant (int32 packed along head_dim)
          tokens [n_quant .. kv_seq_len) → value_states_full  (FP16 residual)

        Only the selected rows are dequantised — V HBM reads scale with budget,
        not with total sequence length.

        Returns: (B, nh, len(selected_idx), head_dim) float16
        """
        from methods.kivi_kernels.new_pack import unpack_and_dequant_vcache

        n_full  = value_states_full.shape[-2]
        n_quant = kv_seq_len - n_full

        B, nh   = value_states_full.shape[:2]
        device  = value_states_full.device
        budget  = selected_idx.shape[0]

        V_sel = torch.empty(
            B, nh, budget, self.head_dim,
            dtype=value_states_full.dtype, device=device,
        )

        quant_mask = selected_idx < n_quant
        full_mask  = ~quant_mask

        if quant_mask.any() and value_states_quant is not None:
            q_idx   = selected_idx[quant_mask]
            q_rows  = value_states_quant[:, :, q_idx, :]   # (B, nh, n_q_sel, D//fps)
            s_rows  = value_scale[:, :, q_idx, :]           # (B, nh, n_q_sel, D//gs)
            mn_rows = value_mn[:, :, q_idx, :]              # (B, nh, n_q_sel, D//gs)
            V_sel[:, :, quant_mask, :] = unpack_and_dequant_vcache(
                q_rows, s_rows, mn_rows, self.group_size, self.v_bits,
            )

        if full_mask.any():
            f_idx = selected_idx[full_mask] - n_quant
            V_sel[:, :, full_mask, :] = value_states_full[:, :, f_idx, :]

        return V_sel

    # ── Forward ───────────────────────────────────────────────────────────────

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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated. Use `attention_mask` instead."
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,           self.head_dim).transpose(1, 2)
        key_states   = key_states.view(  bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]   # stored total seq len

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        assert self.num_key_value_groups == 1, (
            "LlamaAttention_KIVITopK requires GQA groups == 1 "
            "(all KV heads, as in Llama-2-7B)"
        )

        # ══════════════════════════════════════════════════════════════════════
        # DECODE STEP  (past_key_value is not None)
        # ══════════════════════════════════════════════════════════════════════
        if past_key_value is not None:
            (key_states_quant_trans, key_states_full,
             key_scale_trans, key_mn_trans,
             value_states_quant, value_states_full,
             value_scale, value_mn,
             _kv_seq_len_stored) = past_key_value

            # ── Attention scores: Q @ quantised K ─────────────────────────────
            if key_states_quant_trans is not None:
                # cuda kernel: Q (B,nh,1,D) × K_packed_trans (B,nh,D,T//fps) → (B,nh,1,T_quant)
                att_qkquant = cuda_bmm_fA_qB_outer(
                    self.group_size,
                    query_states,
                    key_states_quant_trans,
                    key_scale_trans,
                    key_mn_trans,
                    self.k_bits,
                )
            else:
                att_qkquant = None

            # ── Append new key to FP16 residual buffer ────────────────────────
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            # Q @ FP16 residual K  → (B, nh, 1, residual_len)
            att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))

            # ── Concatenate all scores ────────────────────────────────────────
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            # ── Apply causal attention mask ───────────────────────────────────
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask shape mismatch: expected "
                        f"{(bsz, 1, q_len, kv_seq_len)}, got {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # ── Output: attn @ V (sparse gather) ─────────────────────────────
            # Append new V to FP16 residual so kv_seq_len positions align.
            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]

            # Find which token positions to attend to.
            # Returns sorted absolute indices [sink|top-K middle|local], or None
            # when the full context fits within the budget.
            selected_idx = self._select_topk(attn_weights, kv_seq_len)

            if selected_idx is not None:
                # ── Sparse path ───────────────────────────────────────────────
                # Gather attention weights at the selected positions only,
                # then softmax over that budget-sized slice.
                attn_weights_sel = attn_weights[:, :, :, selected_idx]   # (B,nh,1,budget)
                attn_weights_sel = torch.max(
                    attn_weights_sel,
                    torch.tensor(torch.finfo(attn_weights_sel.dtype).min),
                )
                attn_weights_sel = F.softmax(
                    attn_weights_sel, dim=-1, dtype=torch.float32,
                ).to(query_states.dtype)

                # Dequantise only the selected V rows — HBM reads ∝ budget.
                V_sel = self._gather_v(
                    selected_idx,
                    value_states_quant, value_states_full,
                    value_scale, value_mn,
                    kv_seq_len,
                )
                attn_output = torch.matmul(attn_weights_sel, V_sel)      # (B,nh,1,D)

            else:
                # ── Dense path: budget >= context, use KIVI kernels ───────────
                attn_weights = torch.max(
                    attn_weights,
                    torch.tensor(torch.finfo(attn_weights.dtype).min),
                )
                attn_weights = F.softmax(
                    attn_weights, dim=-1, dtype=torch.float32,
                ).to(query_states.dtype)

                if value_states_quant is None:
                    attn_output = torch.matmul(attn_weights, value_states_full)
                else:
                    attn_output = cuda_bmm_fA_qB_outer(
                        self.group_size,
                        attn_weights[:, :, :, :-value_full_length],
                        value_states_quant,
                        value_scale,
                        value_mn,
                        self.v_bits,
                    )
                    attn_output += torch.matmul(
                        attn_weights[:, :, :, -value_full_length:],
                        value_states_full,
                    )

            # ── KIVI K flush: quantise residual when it reaches exactly R ─────
            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                # triton kernel: pack K.T (B,nh,D,R) → (B,nh,D,R//fps)
                k_quant_new, k_scale_new, k_mn_new = triton_quantize_and_pack_along_last_dim(
                    key_states_full.transpose(2, 3).contiguous(),
                    self.group_size,
                    self.k_bits,
                )
                key_states_full = None   # residual is now empty
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, k_quant_new], dim=3)
                    key_scale_trans        = torch.cat([key_scale_trans,        k_scale_new], dim=3)
                    key_mn_trans           = torch.cat([key_mn_trans,           k_mn_new],    dim=3)
                else:
                    key_states_quant_trans = k_quant_new
                    key_scale_trans        = k_scale_new
                    key_mn_trans           = k_mn_new

            # ── KIVI V flush: quantise oldest token when residual exceeds R ───
            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                # triton kernel: pack V[0] (B,nh,1,D) → (B,nh,1,D//fps)
                v_quant_new, v_scale_new, v_mn_new = triton_quantize_and_pack_along_last_dim(
                    value_states_full[:, :, :1, :].contiguous(),
                    self.group_size,
                    self.v_bits,
                )
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, v_quant_new], dim=2)
                    value_scale        = torch.cat([value_scale,        v_scale_new], dim=2)
                    value_mn           = torch.cat([value_mn,           v_mn_new],    dim=2)
                else:
                    value_states_quant = v_quant_new
                    value_scale        = v_scale_new
                    value_mn           = v_mn_new

        # ══════════════════════════════════════════════════════════════════════
        # PREFILL STEP  (past_key_value is None)
        # Use scaled_dot_product_attention so flash-attn handles the causal
        # mask in O(T) memory — avoids OOM at long context (32K+).
        # TopK is NOT applied at prefill (context-processing step).
        # ══════════════════════════════════════════════════════════════════════
        else:
            # SDPA uses flash-attn when available and is_causal=True, giving
            # O(T) memory vs O(T²) for explicit torch.matmul(Q, K^T).
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            ).to(query_states.dtype)

            # ── Quantise K after prefill (KIVI algorithm 1) ─────────────────
            r = key_states.shape[-2] % self.residual_length
            if r != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant  = None
                    key_states_full   = key_states
                else:
                    key_states_quant  = key_states[:, :, :-r, :].contiguous()
                    key_states_full   = key_states[:, :, -r:, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full  = None

            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = \
                    triton_quantize_and_pack_along_last_dim(
                        key_states_quant.transpose(2, 3).contiguous(),
                        self.group_size,
                        self.k_bits,
                    )
            else:
                key_states_quant_trans = key_scale_trans = key_mn_trans = None

            # ── Quantise V after prefill ─────────────────────────────────────
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full  = value_states
                value_scale = value_mn = None
            else:
                v_quant_part       = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full  = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = \
                    triton_quantize_and_pack_along_last_dim(
                        v_quant_part, self.group_size, self.v_bits,
                    )

        # ── Pack cache as 9-tuple (same layout as LlamaAttention_KIVI) ────────
        past_key_value = (
            key_states_quant_trans, key_states_full,
            key_scale_trans,        key_mn_trans,
            value_states_quant,     value_states_full,
            value_scale,            value_mn,
            kv_seq_len,
        ) if use_cache else None

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output shape mismatch: expected "
                f"{(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# ─────────────────────────────────────────────────────────────────────────────
# Decoder layer and full model  (boilerplate identical to llama_kivi_model.py)
# ─────────────────────────────────────────────────────────────────────────────

class LlamaDecoderLayer_KIVITopK(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size    = config.hidden_size
        self.self_attn      = LlamaAttention_KIVITopK(config=config)
        self.mlp            = LlamaMLP(config)
        self.input_layernorm       = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated. Use `attention_mask`.")

        residual    = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual    = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class LlamaModel_KIVITopK(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size  = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_KIVITopK(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):  return self.embed_tokens
    def set_input_embeddings(self, v): self.embed_tokens = v

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache            if use_cache            is not None else self.config.use_cache
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]   # kv_seq_len from layer 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns    = () if output_attentions    else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_KIVITopK(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model      = LlamaModel_KIVITopK(config)
        self.vocab_size = config.vocab_size
        self.lm_head    = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):        return self.model.embed_tokens
    def set_input_embeddings(self, v):     self.model.embed_tokens = v
    def get_output_embeddings(self):       return self.lm_head
    def set_output_embeddings(self, v):    self.lm_head = v
    def set_decoder(self, decoder):        self.model = decoder
    def get_decoder(self):                 return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits        = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, **kwargs
    ):
        if isinstance(past_key_values, DynamicCache):
            past_key_values = past_key_values.to_legacy_cache()
            if len(past_key_values) == 0:
                past_key_values = None

        if past_key_values is not None:
            past_length = past_key_values[0][-1]
            remove_prefix = past_length if input_ids.shape[1] > past_length else input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        model_inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None and past_key_values is None else {"input_ids": input_ids}
        model_inputs.update({
            "position_ids":   position_ids,
            "past_key_values": past_key_values,
            "use_cache":      kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(
                ps.index_select(0, beam_idx.to(ps.device)) if isinstance(ps, torch.Tensor) else ps
                for ps in layer_past
            )
            for layer_past in past_key_values
        )
