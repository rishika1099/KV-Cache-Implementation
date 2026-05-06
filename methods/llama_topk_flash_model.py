"""
Custom Llama model with TokenSelect attention using paged KV + flash_attn.

Unlike llama_topk_model.py which gathers selected KV (expensive copy),
this uses flash_attn_with_kvcache + block_table for zero-copy selective
attention. Only the selected pages are read from HBM — no gather needed.

Decode flow per layer:
  1. Append new token KV to paged pool
  2. Score middle K with Triton kernel (reads K pages)
  3. Aggregate to page-level scores, select top-K pages
  4. Build subset block_table (sink ∪ selected ∪ local pages)
  5. flash_attn_with_kvcache reads only those pages (zero copy)

Requires: flash-attn >= 2.5.0, triton >= 2.3.0

Usage:
    config = AutoConfig.from_pretrained(model_name)
    config.topk_K = 4096
    config.topk_n_sink = 128
    config.topk_n_local = 4096
    model = LlamaForCausalLM_TopKFlash.from_pretrained(
        model_name, config=config, torch_dtype=torch.float16, device_map="cuda"
    )
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import DynamicCache

from flash_attn import flash_attn_func, flash_attn_with_kvcache

from methods.llama_topk_model import HAS_TRITON
if HAS_TRITON:
    from methods.llama_topk_model import triton_topk_score


def _round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


# ── Attention with paged KV + flash_attn ─────────────────────────────────────

class LlamaAttention_TopKFlash(nn.Module):
    """Llama attention with per-layer TokenSelect using paged KV + flash_attn."""

    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.page_size = getattr(config, 'topk_page_size', 256)  # flash_attn requires divisible by 256
        # Round to page boundaries for clean paged selection
        self.topk_n_sink = _round_up(getattr(config, 'topk_n_sink', 128), self.page_size)
        self.topk_n_local = _round_up(getattr(config, 'topk_n_local', 4096), self.page_size)
        self.topk_K = _round_up(getattr(config, 'topk_K', 4096), self.page_size)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        # Paged KV state (lazily initialized)
        self.k_pool = None
        self.v_pool = None
        self._block_table = []  # logical page idx → physical page idx
        self._seq_len = 0
        self._next_page = 0

    def _init_pool(self, max_seq_len, dtype, device):
        max_pages = (max_seq_len + self.page_size - 1) // self.page_size + 32
        self.k_pool = torch.zeros(
            max_pages, self.page_size, self.num_key_value_heads, self.head_dim,
            dtype=dtype, device=device,
        )
        self.v_pool = torch.zeros(
            max_pages, self.page_size, self.num_key_value_heads, self.head_dim,
            dtype=dtype, device=device,
        )

    def _reset_kv(self):
        self._block_table = []
        self._seq_len = 0
        self._next_page = 0

    def _append_kv(self, key_states, value_states):
        """Append KV to paged pool. key/value: (1, nkv, new_len, hd)."""
        k = key_states[0].transpose(0, 1).contiguous()  # (new_len, nkv, hd)
        v = value_states[0].transpose(0, 1).contiguous()
        new_len = k.shape[0]

        # Fast path: page-aligned start and full pages available
        page_offset = self._seq_len % self.page_size
        if page_offset == 0 and new_len >= self.page_size:
            n_full = new_len // self.page_size
            remainder = new_len % self.page_size
            start_page = self._next_page

            for _ in range(n_full):
                self._block_table.append(self._next_page)
                self._next_page += 1

            self.k_pool[start_page:start_page + n_full] = k[:n_full * self.page_size].view(
                n_full, self.page_size, self.num_key_value_heads, self.head_dim)
            self.v_pool[start_page:start_page + n_full] = v[:n_full * self.page_size].view(
                n_full, self.page_size, self.num_key_value_heads, self.head_dim)
            self._seq_len += n_full * self.page_size

            if remainder > 0:
                self._block_table.append(self._next_page)
                pid = self._next_page
                self._next_page += 1
                self.k_pool[pid, :remainder] = k[n_full * self.page_size:]
                self.v_pool[pid, :remainder] = v[n_full * self.page_size:]
                self._seq_len += remainder
        else:
            # Slow path: handle non-aligned appends (decode tokens)
            written = 0
            while written < new_len:
                po = self._seq_len % self.page_size
                if po == 0:
                    self._block_table.append(self._next_page)
                    self._next_page += 1
                pid = self._block_table[-1]
                chunk = min(self.page_size - po, new_len - written)
                self.k_pool[pid, po:po + chunk] = k[written:written + chunk]
                self.v_pool[pid, po:po + chunk] = v[written:written + chunk]
                written += chunk
                self._seq_len += chunk

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, int]:
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        is_prefill = past_key_value is None

        if is_prefill:
            # ─── PREFILL ──────────────────────────────────────────
            # Initialize / reset paged pool
            if self.k_pool is None:
                self._init_pool(q_len + 1024, key_states.dtype, device)
            self._reset_kv()
            self._append_kv(key_states, value_states)

            # flash_attn_func handles GQA natively (no repeat_kv needed)
            q_f = query_states.transpose(1, 2)   # (1, seq, nheads, hd)
            k_f = key_states.transpose(1, 2)     # (1, seq, nkv, hd)
            v_f = value_states.transpose(1, 2)
            attn_output = flash_attn_func(q_f, k_f, v_f, causal=True)
            attn_output = attn_output.transpose(1, 2)  # (1, nheads, seq, hd)

            past_key_value = self._seq_len if use_cache else None
        else:
            # ─── DECODE ───────────────────────────────────────────
            self._append_kv(key_states, value_states)

            total_budget = self.topk_n_sink + self.topk_K + self.topk_n_local

            if self._seq_len <= total_budget:
                # Full attention via flash_attn paged
                attn_output = self._full_flash_attention(query_states, device)
            else:
                # TopK selection → flash_attn on subset pages
                attn_output = self._topk_flash_attention(query_states, device)

            past_key_value = self._seq_len if use_cache else None

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    def _full_flash_attention(self, query_states, device):
        """Full attention using flash_attn_with_kvcache on all pages."""
        bt = torch.tensor([self._block_table], dtype=torch.int32, device=device)
        seqlen = torch.tensor([self._seq_len], dtype=torch.int32, device=device)
        q_f = query_states.transpose(1, 2)  # (1, 1, nheads, hd)
        out = flash_attn_with_kvcache(q_f, self.k_pool, self.v_pool,
                                       cache_seqlens=seqlen, block_table=bt)
        return out.transpose(1, 2)

    def _topk_flash_attention(self, query_states, device):
        """Score middle K → select top-K pages → flash_attn on subset."""
        ps = self.page_size
        n_sink_pages = self.topk_n_sink // ps
        n_local_pages = self.topk_n_local // ps
        n_total_pages = len(self._block_table)
        n_middle_pages = n_total_pages - n_sink_pages - n_local_pages

        if n_middle_pages <= 0:
            return self._full_flash_attention(query_states, device)

        # ── 1. Gather middle K from pages for scoring ────────────
        mid_start = n_sink_pages           # first middle page (logical)
        mid_end = n_total_pages - n_local_pages  # one past last middle page
        # Physical pages are contiguous (sequential allocation)
        phys_start = self._block_table[mid_start]
        phys_end = self._block_table[mid_end - 1] + 1
        # k_pool slice: (n_middle_pages, page_size, nkv, hd)
        middle_k = self.k_pool[phys_start:phys_end]
        middle_len = n_middle_pages * ps
        # Reshape to (middle_len, nkv, hd) → (nkv, middle_len, hd) for Triton
        middle_k = middle_k.reshape(middle_len, self.num_key_value_heads, self.head_dim)
        middle_k = middle_k.permute(1, 0, 2).contiguous()

        # ── 2. Score with Triton kernel ──────────────────────────
        q_2d = query_states[0, :, 0, :]  # (nq_heads, hd)
        if HAS_TRITON:
            agg = triton_topk_score(q_2d, middle_k)  # (middle_len,) float32
        else:
            # PyTorch fallback
            from methods.llama_topk_model import repeat_kv
            middle_k_exp = middle_k.unsqueeze(0)  # (1, nkv, middle_len, hd)
            from transformers.models.llama.modeling_llama import repeat_kv
            middle_k_exp = repeat_kv(middle_k_exp, self.num_key_value_groups)
            scores = torch.matmul(
                query_states, middle_k_exp.transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            scores = scores.squeeze(2)
            agg = torch.softmax(scores, dim=-1).sum(dim=1)[0]

        # ── 3. Page-level score aggregation (max per page) ───────
        # agg is (middle_len,) = (n_middle_pages * page_size,)
        page_scores = agg.view(n_middle_pages, ps).max(dim=1).values  # (n_middle_pages,)

        # ── 4. Select top-K pages ────────────────────────────────
        k_pages = min(self.topk_K // ps, n_middle_pages)
        selected_middle = page_scores.topk(k_pages).indices  # relative to middle start

        # ── 5. Build subset block_table ──────────────────────────
        sink_pages = torch.arange(n_sink_pages, device=device)
        selected_global = selected_middle + n_sink_pages  # logical page indices
        local_pages = torch.arange(n_total_pages - n_local_pages, n_total_pages, device=device)

        all_selected = torch.cat([sink_pages, selected_global, local_pages])
        all_selected, _ = all_selected.sort()

        # Map logical → physical page indices
        bt_full = torch.tensor(self._block_table, dtype=torch.int32, device=device)
        subset_bt = bt_full[all_selected.long()].unsqueeze(0)  # (1, n_selected)

        # cache_seqlens: all selected pages are full except possibly the last
        n_selected = all_selected.shape[0]
        last_page_fill = self._seq_len % ps
        if last_page_fill == 0:
            last_page_fill = ps
        subset_seqlen = (n_selected - 1) * ps + last_page_fill

        # ── 6. Flash attention on selected pages (zero copy) ─────
        q_f = query_states.transpose(1, 2)  # (1, 1, nheads, hd)
        seqlen_t = torch.tensor([subset_seqlen], dtype=torch.int32, device=device)
        out = flash_attn_with_kvcache(q_f, self.k_pool, self.v_pool,
                                       cache_seqlens=seqlen_t, block_table=subset_bt)
        return out.transpose(1, 2)  # (1, nheads, 1, hd)


# ── Decoder layer / Model / CausalLM ─────────────────────────────────────────

class LlamaDecoderLayer_TopKFlash(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention_TopKFlash(config=config, layer_idx=layer_idx)
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


class LlamaModel_TopKFlash(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer_TopKFlash(config, layer_idx=i)
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

        # past_key_values: tuple of int (seq_len per layer) or None
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0]

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
                attention_mask=None,
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


class LlamaForCausalLM_TopKFlash(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_TopKFlash(config)
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
            past_length = past_key_values[0]  # int: seq_len from layer 0
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
