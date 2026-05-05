"""
MLALatentAttention and MLALatentCache — MLA attention variant that caches
compressed latent representations instead of full reconstructed (K, V) tensors.

Background
----------
Standard MLAAttention (mla.py) calls past_key_value.update(key_states, value_states, ...)
which stores the fully-expanded (K, V) in a DynamicCache.  This negates the memory
advantage of MLA: with default params (qk_head_dim=192, v_head_dim=128, 32 heads) the
stored K/V is *larger* than the original GQA cache.

This file caches only the low-rank latents that the attention already computes:
  - c_kv_norm  : kv_a_layernorm(kv_a_proj(hidden))  shape (batch, seq, kv_lora_rank)
  - k_rot_roped: apply_rope(k_rot_raw)               shape (batch, 1, seq, qk_rope_head_dim)

At every decode step the latents are expanded to full K/V on-the-fly via kv_b_proj.
This trades compute for memory (kv_b_proj is O(seq) per step), but for KV-cache memory
benchmarking the savings are the key metric.

Memory comparison — default conversion of llama2-7B (32 layers, 32 heads):
  Full K/V cache  : 32 × (32 × 192 + 32 × 128) × 2 B = 20 480 B/token/layer
                    × 32 layers → 655 360 B/token ≈ 640 KB/token
  Latent cache    : 32 × (512 + 64) × 2 B = 1 152 B/token/layer
                    × 32 layers → 36 864 B/token ≈ 36 KB/token
  Compression     : ~17.8× vs full-KV MLA; ~14.2× vs original llama2-7B GQA baseline

For Qwen2.5-7B (28 layers, 4 KV heads in original GQA):
  Baseline GQA    : 28 × (4 × 128 + 4 × 128) × 2 B ≈ 57 344 B/token
  Latent cache    : 28 × (512 + 64) × 2 B ≈ 32 256 B/token
  Compression     : ~1.8× vs Qwen GQA baseline (GQA is already very efficient)

Usage
-----
    from transmla.transformers.mla_latent import MLALatentCache
    from transmla.transformers.llama.modeling_llamamla_latent import LlamaMLALatentForCausalLM
    from transmla.transformers.llama.configuration_llamamla import LlamaMLAConfig

    config = LlamaMLAConfig.from_pretrained(mla_model_path)
    model  = LlamaMLALatentForCausalLM.from_pretrained(mla_model_path, config=config)

    cache = MLALatentCache()
    out   = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    # cache is now populated with prefill latents; reuse for decode:
    out2  = model(input_ids=next_token, past_key_values=out.past_key_values, use_cache=True)
    print(f"KV cache memory: {cache.get_cache_bytes() / 1e6:.2f} MB")
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack

from transformers.models.gemma2.modeling_gemma2 import eager_attention_forward, logger
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave

from .mla import MLAAttention


# ── Latent Cache ──────────────────────────────────────────────────────────────

class MLALatentCache(Cache):
    """
    Per-layer KV cache that stores compressed latents (c_kv_norm, k_rot_roped)
    instead of full (K, V) tensors.

    Stored per layer:
      c_kv_norm   – normalized latent  : (batch, seq, kv_lora_rank)
      k_rot_roped – RoPE-applied k_rot : (batch, 1, seq, qk_rope_head_dim)

    Interface
    ---------
    update_latent(c_kv_new, k_rot_new, layer_idx)  → (c_kv_all, k_rot_all)
        Append new-token latents and return the full accumulated tensors.
        Called from MLALatentAttention.forward() instead of the standard update().

    get_cache_bytes() → int
        Total bytes used across all layers (both tensors).

    Implements the HF Cache ABC so the model's forward() can call
    get_seq_length() / get_max_length() for causal-mask construction.
    """

    def __init__(self):
        super().__init__()
        self._c_kv_norm:   dict[int, torch.Tensor] = {}   # layer → (B, S, kv_lora_rank)
        self._k_rot_roped: dict[int, torch.Tensor] = {}   # layer → (B, 1, S, qk_rope_head_dim)

    # ── latent-specific update (called by MLALatentAttention) ─────────────────

    def update_latent(
        self,
        c_kv_new_norm:   torch.Tensor,   # (batch, new_seq, kv_lora_rank)
        k_rot_new_roped: torch.Tensor,   # (batch, 1, new_seq, qk_rope_head_dim)
        layer_idx:       int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new tokens' latents to the per-layer cache and return the full
        accumulated (c_kv_norm_all, k_rot_roped_all) for use in attention.

        Concatenation axes:
          c_kv_norm   → dim 1  (sequence axis)
          k_rot_roped → dim 2  (sequence axis, keepin batch/head dims)
        """
        if layer_idx not in self._c_kv_norm:
            self._c_kv_norm[layer_idx]   = c_kv_new_norm
            self._k_rot_roped[layer_idx] = k_rot_new_roped
        else:
            self._c_kv_norm[layer_idx] = torch.cat(
                [self._c_kv_norm[layer_idx], c_kv_new_norm], dim=1
            )
            self._k_rot_roped[layer_idx] = torch.cat(
                [self._k_rot_roped[layer_idx], k_rot_new_roped], dim=2
            )
        return self._c_kv_norm[layer_idx], self._k_rot_roped[layer_idx]

    # ── memory measurement ────────────────────────────────────────────────────

    def get_cache_bytes(self) -> int:
        """Total bytes occupied by all stored latent tensors."""
        total = 0
        for t in self._c_kv_norm.values():
            total += t.element_size() * t.numel()
        for t in self._k_rot_roped.values():
            total += t.element_size() * t.numel()
        return total

    # ── HF Cache ABC interface ────────────────────────────────────────────────

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Not used. MLALatentAttention calls update_latent() instead."""
        raise RuntimeError(
            "MLALatentCache.update() was called directly. This cache must be used "
            "with MLALatentAttention, which calls update_latent() instead of update()."
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Cached sequence length (number of tokens stored so far)."""
        if layer_idx in self._c_kv_norm:
            return self._c_kv_norm[layer_idx].shape[1]
        # Fall back to layer 0 if queried before layer_idx has been populated
        if self._c_kv_norm:
            return next(iter(self._c_kv_norm.values())).shape[1]
        return 0

    def get_max_length(self) -> Optional[int]:
        return None  # unbounded dynamic cache

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    @property
    def seen_tokens(self) -> int:
        """Compatibility shim — some HF internals read this attribute."""
        return self.get_seq_length(0)

    def to_legacy_cache(self):
        raise NotImplementedError(
            "MLALatentCache stores latents, not (K, V) pairs, and cannot be "
            "converted to the legacy tuple-of-tuples cache format."
        )


# ── Latent-caching Attention ──────────────────────────────────────────────────

class MLALatentAttention(MLAAttention):
    """
    Drop-in replacement for MLAAttention that caches compressed latents
    (c_kv_norm, k_rot_roped) when past_key_value is a MLALatentCache.

    Forward logic (latent-cache path):
      1. Compute query states q_pass, q_rot   (identical to MLAAttention)
      2. kv_a_proj_with_mqa(hidden) → c_kv_raw, k_rot_raw
      3. c_kv_norm   = kv_a_layernorm(c_kv_raw)          if qk_latent_layernorm
      4. k_rot_roped = apply_rotary_pos_emb_interleave(k_rot_raw)
      5. cache.update_latent(c_kv_norm, k_rot_roped, layer_idx)
         → returns all accumulated (c_kv_all_norm, k_rot_all_roped)
      6. kv_b_proj(c_kv_all_norm) → (K_nope_all, V_all)   ← expand latents for attention
      7. key_states   = cat(K_nope_all,  k_rot_all_roped.expand(num_heads))
      8. Standard attention over all cached + current tokens
      9. o_proj → output

    If past_key_value is NOT an MLALatentCache, falls back to the standard
    MLAAttention.forward() (full K/V stored in DynamicCache).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # ── fall-back to standard path if not using latent cache ──────────────
        if not isinstance(past_key_value, MLALatentCache):
            return super().forward(
                hidden_states, position_embeddings, attention_mask,
                past_key_value, cache_position, **kwargs,
            )

        # ── latent-cache path ─────────────────────────────────────────────────
        batch_size, seq_length = hidden_states.shape[:-1]

        # 1. Query projection
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        elif self.qk_latent_layernorm:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_b_proj(self.q_a_proj(hidden_states))

        q_states = q_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # 2. Compressed KV projection
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        # compressed_kv : (batch, seq, kv_lora_rank + qk_rope_head_dim)
        c_kv_raw, k_rot_raw = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # 3. Normalize latent (pre-computed for efficient caching)
        c_kv_new_norm = self.kv_a_layernorm(c_kv_raw) if self.qk_latent_layernorm else c_kv_raw
        # c_kv_new_norm : (batch, seq, kv_lora_rank)

        # 4. Apply RoPE to current-token k_rot
        k_rot_raw = k_rot_raw.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot_new_roped = apply_rotary_pos_emb_interleave(q_rot, k_rot_raw, cos, sin)
        # k_rot_new_roped : (batch, 1, seq, qk_rope_head_dim) — position-encoded

        # 5. Update latent cache and get full accumulated latents
        c_kv_all_norm, k_rot_all_roped = past_key_value.update_latent(
            c_kv_new_norm, k_rot_new_roped, self.layer_idx
        )
        # c_kv_all_norm  : (batch, full_seq, kv_lora_rank)
        # k_rot_all_roped: (batch, 1, full_seq, qk_rope_head_dim)
        full_seq = c_kv_all_norm.shape[1]

        # 6. Expand latents to full K and V for attention
        #    kv_b_proj : (batch, full_seq, num_heads*(qk_nope_head_dim + v_head_dim))
        kv_full = self.kv_b_proj(c_kv_all_norm)
        kv_full = kv_full.view(
            batch_size, full_seq, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        # kv_full : (batch, num_heads, full_seq, qk_nope_head_dim + v_head_dim)

        k_nope_all, value_states = torch.split(
            kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # 7. Assemble full key states
        k_rot_all_expanded = k_rot_all_roped.expand(
            batch_size, self.num_heads, full_seq, self.qk_rope_head_dim
        )
        query_states = torch.cat((q_pass, q_rot), dim=-1)
        # query_states : (batch, num_heads, seq_length, qk_head_dim)
        key_states    = torch.cat((k_nope_all, k_rot_all_expanded), dim=-1)
        # key_states   : (batch, num_heads, full_seq, qk_head_dim)

        # 8. Flash-attention padding (when head dims differ)
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        # 9. Attention
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support "
                    "`output_attentions=True`. Falling back to eager attention."
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softcap=getattr(self.config, "attn_logit_softcapping", None),
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
