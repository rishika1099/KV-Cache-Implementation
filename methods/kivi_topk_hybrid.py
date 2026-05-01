"""
KIVI + TopK hybrid — design (a) centroid scoring + design (c) quant-aware scoring.

Combines KIVI's block-wise asymmetric quantization with TokenSelect's
top-K attention selection. The novelty over naïve composition (KIVI →
dequant full → TopK over FP16) is that scoring runs *directly on the
quantized state*, never paying the cost of dequantizing the entire
historical cache:

    • residual region (FP16, last `residual_length` tokens):
        exact   Q · K_resᵀ
    • overflow buffer (FP16, < group_size pending tokens):
        exact   Q · K_overᵀ
    • quantized blocks (4-bit / 2-bit, group_size tokens each):
        approx  Q · centroidᵀ      (design a)
        or  exact int×fp Triton kernel on uint8 storage  (design c)

After TopK selects K indices over the combined score vector, only the
selected tokens are dequantized — never whole blocks, never the full
cache.

═══════════════════════════════════════════════════════════════════════════
PERFORMANCE OPTIMIZATIONS (perf/hybrid-optimization branch)
═══════════════════════════════════════════════════════════════════════════
The previous reference implementation paid an O(n_unique_blocks · n_layers)
Python-loop cost in `_materialise` (one full-block dequant per touched
block per layer × 32 layers) which dominated decode latency.  This module
replaces that loop with a fully vectorised single-shot gather + dequant
that operates directly on pre-stacked uint8 tensors — one tensor op per
layer, zero Python iteration over blocks.

Key changes vs. v1:
  • V-side scale/zero stacks (`vs_stack`, `vz_stack`) maintained in
    parallel with `kq_stack` so V dequant is also a single batched op.
  • Per-token block-id LUT (`block_id_for_quant[layer_idx]`) cached on
    block-seal events so `bucketize` runs once per seal, not per step.
  • Pre-allocated growable buffers for the four stacked tensors (capacity
    grows by chunks rather than re-cat'ing every block-seal).
  • FP16 matmul in `_score_layer` (drop unconditional `.float()`
    upcasts; H100 fp16 tensor cores are free, fp32 cast only happens
    around the softmax/sum reduction).
  • Selection-cache staleness fix: `_can_reuse_cache` now invalidates
    when seq_len has changed since the cached query, not just when
    cosine similarity drops below threshold.
  • Storage accounting includes hybrid scratch tensors.
  • Optional debug profiling via `KV_HYBRID_PROFILE=1` (logs per-step
    time_scoring / time_materialise / num_blocks_dequantized to stderr).

Selection knobs (all default OFF):
  • score_mode="maxpool"   per-channel max block summary
  • two_pass_factor=N      centroid pre-rank → exact quant_score rerank
  • proxy_history=N        pool the last N K-vectors into the proxy query
  • dynamic_sinks=True     pick n_sink positions from prefill K-norm
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import torch

from .base import MethodWrapper
from .kivi_quant import KIVIMethod, dequantize
from .rope_utils import apply_rope_delta
from .topk_kernels import quant_score


# Debug profiling toggle. When enabled, _score_layer / _materialise time is
# accumulated in self.stats and dumped to stderr at the end of the run.
_DEBUG_PROFILE = os.environ.get("KV_HYBRID_PROFILE", "0") == "1"


def _cuda_sync():
    """Cheap CUDA sync used only when KV_HYBRID_PROFILE=1."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class KIVI_TopK_Method(MethodWrapper):
    """
    Hybrid KIVI-quantized storage + TopK selection. See module docstring.

    Constructor args mirror the union of `KIVIMethod` and `TopKMethod`:
      KIVI side : bits, residual_length, group_size
      TopK side : K, n_sink, n_local, refresh_interval,
                  cache_similarity_threshold,
                  use_head_softmax, use_criticality_weights,
                  use_selection_cache, use_sink_tokens, use_local_tokens
    """

    # Pre-alloc growth chunk (in number of blocks) for the stacked buffers.
    # Larger ⇒ fewer reallocations, more upfront memory. 64 ≈ 2K context
    # worth of blocks at group_size=32; doubles on growth, like std::vector.
    _CAPACITY_CHUNK = 64

    def __init__(
        self,
        # ── KIVI parameters ──
        bits: int = 4,
        residual_length: int = 128,
        group_size: int = 32,
        # ── TopK parameters ──
        K: int = 1024,
        n_sink: int = 128,
        n_local: int = 512,
        refresh_interval: int = 50,
        cache_similarity_threshold: float = 0.95,
        # ── Hybrid scoring path ──
        score_mode: str = "centroid",            # "centroid" | "maxpool" | "quantized"
        two_pass_factor: int = 0,                # 0 disables. else top-(factor*K_blocks)
                                                 # are reranked exactly with quant_score.
        proxy_history: int = 1,                  # >1 pools the last N K-vectors.
        proxy_pool: str = "mean",                # "mean" | "max"
        dynamic_sinks: bool = False,             # pick sinks from prefill K-norm
        # ── Position-encoding parameters (BUG-2 fix) ──
        head_dim: int = 128,
        rope_theta: float = 10000.0,
        apply_rope_correction: bool = False,
        # ── Ablation flags ──
        use_head_softmax: bool = True,
        use_criticality_weights: bool = True,
        use_selection_cache: bool = True,
        use_sink_tokens: bool = True,
        use_local_tokens: bool = True,
    ):
        if score_mode not in ("centroid", "maxpool", "quantized"):
            raise ValueError(
                f"score_mode must be 'centroid' | 'maxpool' | 'quantized', "
                f"got {score_mode!r}"
            )
        if proxy_pool not in ("mean", "max"):
            raise ValueError(
                f"proxy_pool must be 'mean' or 'max', got {proxy_pool!r}"
            )
        if two_pass_factor < 0:
            raise ValueError(f"two_pass_factor must be >= 0, got {two_pass_factor}")
        if proxy_history < 1:
            raise ValueError(f"proxy_history must be >= 1, got {proxy_history}")
        self.score_mode = score_mode
        self.two_pass_factor = two_pass_factor
        self.proxy_history = proxy_history
        self.proxy_pool = proxy_pool
        self.dynamic_sinks = dynamic_sinks
        self.kivi = KIVIMethod(
            bits=bits, residual_length=residual_length, group_size=group_size,
        )
        self.bits = bits
        self.group_size = group_size
        self.residual_length = residual_length

        self.K = K
        self.n_sink = n_sink if use_sink_tokens else 0
        self.n_local = n_local if use_local_tokens else 0
        self.refresh_interval = refresh_interval
        self.cache_similarity_threshold = cache_similarity_threshold

        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.apply_rope_correction = apply_rope_correction

        self.use_head_softmax        = use_head_softmax
        self.use_criticality_weights = use_criticality_weights
        self.use_selection_cache     = use_selection_cache
        self.use_sink_tokens         = use_sink_tokens
        self.use_local_tokens        = use_local_tokens

        # Selection state
        self.step_counter = 0
        self.prev_query: torch.Tensor | None = None
        self.prev_seq_len: int | None = None
        self.cached_indices: Dict[int, torch.Tensor] | None = None
        self.head_weights: torch.Tensor | None = None

        # ── Stacked block state (per layer) ──
        # K side: per-token quant ⇒ scale/zero are (B,H,group_size,1) per block,
        #         which stack along dim=2 to (B,H,n_quant,1) ─ "per-token" scalars.
        #         kq_stack[l]: (B,H,capacity,D) uint8
        #         ks_stack[l]: (B,H,capacity,1) fp16
        #         kz_stack[l]: (B,H,capacity,1) fp16
        # V side: per-channel quant ⇒ scale/zero are (B,H,1,D) per block, which
        #         stack along dim=2 to (B,H,n_blocks,D) ─ "per-block per-channel".
        #         vq_stack[l]: (B,H,capacity,D) uint8
        #         vs_stack[l]: (B,H,n_block_capacity,D) fp16
        #         vz_stack[l]: (B,H,n_block_capacity,D) fp16
        # Centroids (design a only): (B,H,n_block_capacity,D) fp16
        self.kq_stack: Dict[int, torch.Tensor] = {}
        self.ks_stack: Dict[int, torch.Tensor] = {}
        self.kz_stack: Dict[int, torch.Tensor] = {}
        self.vq_stack: Dict[int, torch.Tensor] = {}
        self.vs_stack: Dict[int, torch.Tensor] = {}
        self.vz_stack: Dict[int, torch.Tensor] = {}
        self.centroids_k: Dict[int, torch.Tensor] = {}
        self.maxpool_k: Dict[int, torch.Tensor] = {}
        # Rolling (B, H, proxy_history, D) buffer for multi-step proxy.
        self._query_history: torch.Tensor | None = None
        # Sorted positions for dynamic sinks; None = use static [0, n_sink).
        self._sink_positions: torch.Tensor | None = None

        # Logical sizes (≤ capacity).
        self._n_quant_per_layer: Dict[int, int] = {}
        self._n_blocks_per_layer: Dict[int, int] = {}
        # Per-layer block sizes list — needed only for the partial-first-block
        # case from KIVI's prefill; uniform = group_size for decode-sealed.
        self.block_sizes: Dict[int, List[int]] = {}
        # Per-layer LUT mapping each token slot in [0, n_quant) to its block_id.
        # Rebuilt only on _sync_block_state (block-seal events), not per step.
        self._block_id_for_quant: Dict[int, torch.Tensor] = {}

        # Diagnostics
        self.stats = {
            "decode_steps":         0,
            "full_attention_steps": 0,
            "refresh_steps":        0,
            "cache_hits":            0,
            "cache_misses":          0,
            "blocks_dequantized":    0,        # cumulative legacy counter
            # Perf counters (only populated when KV_HYBRID_PROFILE=1)
            "time_scoring_ms":       0.0,
            "time_materialise_ms":   0.0,
            "time_sync_ms":          0.0,
            "tokens_dequantized":    0,        # = n_selected_tokens in quant region
        }

    # ── Buffer growth helper ────────────────────────────────────────────────
    @staticmethod
    def _ensure_capacity(buf: torch.Tensor, needed: int, growth: int) -> torch.Tensor:
        """
        Grow a stacked buffer in-place along dim=2 if its physical capacity
        is less than `needed`. Doubles on growth (amortised O(1) per append).

        We can't truly grow a torch tensor in-place; what we do instead is
        allocate a new bigger buffer once and copy over. With `growth`
        chunking, the cost is amortised to O(1) per block-seal event.
        """
        cap = buf.shape[2]
        if cap >= needed:
            return buf
        new_cap = max(needed, cap * 2 if cap > 0 else growth)
        new_shape = list(buf.shape)
        new_shape[2] = new_cap
        new_buf = torch.empty(new_shape, dtype=buf.dtype, device=buf.device)
        if cap > 0:
            new_buf[:, :, :cap, :] = buf
        return new_buf

    # ── Prefill ──────────────────────────────────────────────────────────────

    def process_prefill(self, past_key_values, attention_weights=None):
        """
        Build the KIVI quantized cache and seed the stacked block state used
        for fast scoring + selective dequantization on later decode steps.
        Returns the FP16 reconstruction (the model needs it for the prefill
        attention pass itself).
        """
        recon = self.kivi.process_prefill(past_key_values)
        self.step_counter = 0
        self.prev_query = None
        self.prev_seq_len = None
        self.cached_indices = None
        self._query_history = None
        self._sink_positions = None

        self.kq_stack.clear()
        self.ks_stack.clear()
        self.kz_stack.clear()
        self.vq_stack.clear()
        self.vs_stack.clear()
        self.vz_stack.clear()
        self.centroids_k.clear()
        self.maxpool_k.clear()
        self._n_quant_per_layer.clear()
        self._n_blocks_per_layer.clear()
        self.block_sizes.clear()
        self._block_id_for_quant.clear()

        for layer_idx, state in self.kivi.cache.items():
            blocks_k = state['k_blocks']
            blocks_v = state['v_blocks']
            B, H, _, D = state['residual_k'].shape
            dev = state['residual_k'].device
            dtype = state['residual_k'].dtype

            self.block_sizes[layer_idx] = [qk.shape[2] for qk, _, _ in blocks_k]
            n_blocks = len(blocks_k)
            n_quant = sum(self.block_sizes[layer_idx])
            self._n_quant_per_layer[layer_idx] = n_quant
            self._n_blocks_per_layer[layer_idx] = n_blocks

            # Pre-alloc capacity in chunks, sized to fit prefill plus headroom.
            tok_cap = max(n_quant + self._CAPACITY_CHUNK * self.group_size,
                          self._CAPACITY_CHUNK * self.group_size)
            blk_cap = max(n_blocks + self._CAPACITY_CHUNK,
                          self._CAPACITY_CHUNK)

            self.kq_stack[layer_idx] = torch.empty(
                (B, H, tok_cap, D), dtype=torch.uint8, device=dev,
            )
            self.ks_stack[layer_idx] = torch.empty(
                (B, H, tok_cap, 1), dtype=torch.float16, device=dev,
            )
            self.kz_stack[layer_idx] = torch.empty(
                (B, H, tok_cap, 1), dtype=torch.float16, device=dev,
            )
            self.vq_stack[layer_idx] = torch.empty(
                (B, H, tok_cap, D), dtype=torch.uint8, device=dev,
            )
            self.vs_stack[layer_idx] = torch.empty(
                (B, H, blk_cap, D), dtype=torch.float16, device=dev,
            )
            self.vz_stack[layer_idx] = torch.empty(
                (B, H, blk_cap, D), dtype=torch.float16, device=dev,
            )
            self.centroids_k[layer_idx] = torch.empty(
                (B, H, blk_cap, D), dtype=dtype, device=dev,
            )
            self.maxpool_k[layer_idx] = torch.empty(
                (B, H, blk_cap, D), dtype=dtype, device=dev,
            )

            if n_blocks > 0:
                # ── K side: cat once into the buffer ──
                self.kq_stack[layer_idx][:, :, :n_quant, :] = torch.cat(
                    [qk for qk, _, _ in blocks_k], dim=2,
                )
                self.ks_stack[layer_idx][:, :, :n_quant, :] = torch.cat(
                    [sk for _, sk, _ in blocks_k], dim=2,
                )
                self.kz_stack[layer_idx][:, :, :n_quant, :] = torch.cat(
                    [zk for _, _, zk in blocks_k], dim=2,
                )

                # ── V side: stack uint8 along tokens, scale/zero along blocks ──
                self.vq_stack[layer_idx][:, :, :n_quant, :] = torch.cat(
                    [qv for qv, _, _ in blocks_v], dim=2,
                )
                # V scale/zero are per-block per-channel. Per-block shape is
                # (B,H,1,D); stacking along dim=2 gives (B,H,n_blocks,D).
                self.vs_stack[layer_idx][:, :, :n_blocks, :] = torch.cat(
                    [sv for _, sv, _ in blocks_v], dim=2,
                )
                self.vz_stack[layer_idx][:, :, :n_blocks, :] = torch.cat(
                    [zv for _, _, zv in blocks_v], dim=2,
                )

                centroids = torch.cat([
                    self._block_centroid(qk, sk, zk)
                    for qk, sk, zk in blocks_k
                ], dim=2)
                self.centroids_k[layer_idx][:, :, :n_blocks, :] = centroids
                maxpools = torch.cat([
                    self._block_maxpool(qk, sk, zk)
                    for qk, sk, zk in blocks_k
                ], dim=2)
                self.maxpool_k[layer_idx][:, :, :n_blocks, :] = maxpools

                self._block_id_for_quant[layer_idx] = self._build_token_block_ids(
                    self.block_sizes[layer_idx], dev,
                )
            else:
                self._block_id_for_quant[layer_idx] = torch.empty(
                    (0,), dtype=torch.long, device=dev,
                )

        if self.use_criticality_weights:
            self._compute_head_weights(recon)
        else:
            self.head_weights = None

        if self.dynamic_sinks and self.use_sink_tokens and self.n_sink > 0:
            self._compute_dynamic_sinks(recon, self.n_sink)
        else:
            self._sink_positions = None

        # Pre-warm Triton kernels with a dummy launch on the actual stacked
        # tensors so the first real decode step doesn't pay JIT compile cost.
        self._prewarm_kernels()

        return recon

    @staticmethod
    def _block_centroid(qk, sk, zk) -> torch.Tensor:
        """One-time centroid for a quantized block: mean key over group_size."""
        deq = dequantize(qk, sk, zk)            # (B, H, group_size, D)
        return deq.mean(dim=2, keepdim=True)    # (B, H, 1, D)

    @staticmethod
    def _block_maxpool(qk, sk, zk) -> torch.Tensor:
        """Per-channel max over a quantized block. Returns (B, H, 1, D)."""
        deq = dequantize(qk, sk, zk)
        return deq.max(dim=2, keepdim=True).values

    def _compute_dynamic_sinks(self, recon, n_sink: int) -> None:
        """
        Score positions by mean K L2-norm across layers and heads, store
        the top `n_sink` (sorted) as `self._sink_positions`. Falls back to
        None on any error so scoring uses static [0, n_sink).
        """
        try:
            n_layers = len(recon)
            if n_layers == 0 or n_sink == 0:
                return
            S = recon[0][0].shape[2]
            agg = torch.zeros(S, device=recon[0][0].device, dtype=torch.float32)
            for layer_idx in range(n_layers):
                k = recon[layer_idx][0]                  # (B, H, S, D)
                agg += k.float().norm(dim=-1).mean(dim=(0, 1))
            agg /= n_layers
            n_pick = min(n_sink, S)
            self._sink_positions = agg.topk(n_pick).indices.sort().values.to(
                torch.long
            )
        except Exception:
            self._sink_positions = None

    @staticmethod
    def _build_token_block_ids(block_sizes: List[int], device) -> torch.Tensor:
        """
        Build a (n_quant,) long tensor mapping each token slot to its block_id.
        Called only on block-seal events; cheap.
        """
        if not block_sizes:
            return torch.empty((0,), dtype=torch.long, device=device)
        ids = torch.cat([
            torch.full((sz,), bid, dtype=torch.long, device=device)
            for bid, sz in enumerate(block_sizes)
        ])
        return ids

    def _prewarm_kernels(self) -> None:
        """
        Trigger a no-op pass through the score/dequant path so Triton
        compiles before the first real decode step.  Skipped silently if
        there's no quantized state yet (e.g. very short prefills).
        """
        if not self.kivi.cache:
            return
        try:
            layer0 = next(iter(self.kivi.cache.values()))
            B, H, _, D = layer0['residual_k'].shape
            dev = layer0['residual_k'].device
            n_quant = self._n_quant_per_layer.get(0, 0)
            if n_quant == 0:
                return
            dummy_q = torch.zeros((1, H, 1, D), device=dev, dtype=torch.float16)
            kq = self.kq_stack[0][:, :, :n_quant, :]
            ks = self.ks_stack[0][:, :, :n_quant, :]
            kz = self.kz_stack[0][:, :, :n_quant, :]
            wants_quant_kernel = (
                self.score_mode == "quantized" or self.two_pass_factor > 0
            )
            if wants_quant_kernel and dev.type == "cuda":
                _ = quant_score(dummy_q, kq, ks, kz)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception:
            # Pre-warm is best-effort; never fail the pipeline because of it.
            pass

    def _sync_block_state(self) -> None:
        """
        Update centroid + quant-stack state when KIVI seals new blocks.

        Called once per process_step. The common case (no new block) hits
        a tight fast-path early-out. When blocks have been sealed, we
        in-place write into pre-allocated buffers (growing them only when
        capacity is exceeded).
        """
        if _DEBUG_PROFILE:
            _cuda_sync()
            t0 = time.perf_counter()

        for layer_idx, state in self.kivi.cache.items():
            n_existing = self._n_blocks_per_layer.get(layer_idx, 0)
            n_total = len(state['k_blocks'])
            if n_total <= n_existing:
                continue       # No new block this step → fast path

            new_blocks_k = state['k_blocks'][n_existing:n_total]
            new_blocks_v = state['v_blocks'][n_existing:n_total]
            new_token_count = sum(qk.shape[2] for qk, _, _ in new_blocks_k)
            old_n_quant = self._n_quant_per_layer.get(layer_idx, 0)
            new_n_quant = old_n_quant + new_token_count
            new_n_blocks = n_total

            # ── Grow buffers if needed ──
            grow_tok = self._CAPACITY_CHUNK * self.group_size
            grow_blk = self._CAPACITY_CHUNK
            self.kq_stack[layer_idx] = self._ensure_capacity(
                self.kq_stack[layer_idx], new_n_quant, grow_tok)
            self.ks_stack[layer_idx] = self._ensure_capacity(
                self.ks_stack[layer_idx], new_n_quant, grow_tok)
            self.kz_stack[layer_idx] = self._ensure_capacity(
                self.kz_stack[layer_idx], new_n_quant, grow_tok)
            self.vq_stack[layer_idx] = self._ensure_capacity(
                self.vq_stack[layer_idx], new_n_quant, grow_tok)
            self.vs_stack[layer_idx] = self._ensure_capacity(
                self.vs_stack[layer_idx], new_n_blocks, grow_blk)
            self.vz_stack[layer_idx] = self._ensure_capacity(
                self.vz_stack[layer_idx], new_n_blocks, grow_blk)
            self.centroids_k[layer_idx] = self._ensure_capacity(
                self.centroids_k[layer_idx], new_n_blocks, grow_blk)
            self.maxpool_k[layer_idx] = self._ensure_capacity(
                self.maxpool_k[layer_idx], new_n_blocks, grow_blk)

            # ── Append new K-side data ──
            cursor_tok = old_n_quant
            for (qk, sk, zk) in new_blocks_k:
                bs = qk.shape[2]
                self.kq_stack[layer_idx][:, :, cursor_tok:cursor_tok+bs, :] = qk
                self.ks_stack[layer_idx][:, :, cursor_tok:cursor_tok+bs, :] = sk
                self.kz_stack[layer_idx][:, :, cursor_tok:cursor_tok+bs, :] = zk
                blk_slot = n_existing + (cursor_tok - old_n_quant) // self.group_size
                self.centroids_k[layer_idx][:, :, blk_slot, :] = (
                    self._block_centroid(qk, sk, zk).squeeze(2)
                )
                self.maxpool_k[layer_idx][:, :, blk_slot, :] = (
                    self._block_maxpool(qk, sk, zk).squeeze(2)
                )
                cursor_tok += bs

            # ── Append new V-side data ──
            cursor_tok = old_n_quant
            cursor_blk = n_existing
            for (qv, sv, zv) in new_blocks_v:
                bs = qv.shape[2]
                self.vq_stack[layer_idx][:, :, cursor_tok:cursor_tok+bs, :] = qv
                # sv,zv: (B,H,1,D) → write into block slot
                self.vs_stack[layer_idx][:, :, cursor_blk:cursor_blk+1, :] = sv
                self.vz_stack[layer_idx][:, :, cursor_blk:cursor_blk+1, :] = zv
                cursor_tok += bs
                cursor_blk += 1

            # Bookkeeping
            for (qk, _, _) in new_blocks_k:
                self.block_sizes[layer_idx].append(qk.shape[2])
            self._n_quant_per_layer[layer_idx] = new_n_quant
            self._n_blocks_per_layer[layer_idx] = new_n_blocks

            # Rebuild token→block_id LUT (only on seal events).
            self._block_id_for_quant[layer_idx] = self._build_token_block_ids(
                self.block_sizes[layer_idx],
                self.kq_stack[layer_idx].device,
            )

        if _DEBUG_PROFILE:
            _cuda_sync()
            self.stats["time_sync_ms"] += (time.perf_counter() - t0) * 1000.0

    # Back-compat alias.
    _sync_centroids = _sync_block_state

    def _compute_head_weights(self, past_key_values):
        """Per-head criticality (entropy-inverse). Lifted from TopKMethod."""
        k = past_key_values[-1][0]                 # (B, H, S, D)
        _, heads, seq_len, head_dim = k.shape

        if seq_len < 2:
            self.head_weights = torch.ones(heads, device=k.device, dtype=k.dtype)
            return

        n_sample = min(64, seq_len)
        sample_idx = torch.linspace(0, seq_len - 1, n_sample,
                                    device=k.device).long()
        q_sample = k[:, :, sample_idx, :]
        scores = torch.matmul(q_sample, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        eps = 1e-8
        ent = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=-1).squeeze(0)
        inv_ent = 1.0 / (ent.float() + 1e-8)
        self.head_weights = (inv_ent / inv_ent.sum()).to(k.dtype)

    # ── Decode step ──────────────────────────────────────────────────────────

    def process_step(self, past_key_values, step, attention_weights=None):
        """Hybrid decode step. See module docstring for the four phases."""
        # Phase 1: KIVI ingests new tokens; we ignore its FP16 reconstruction.
        self.kivi.process_step(past_key_values, step)
        self._sync_block_state()
        self.step_counter += 1
        self.stats["decode_steps"] += 1

        layer0 = self.kivi.cache[0]
        n_quant       = self._n_quant_per_layer.get(0, 0)
        n_overflow    = layer0['overflow_k'].shape[2]
        n_residual    = layer0['residual_k'].shape[2]
        seq_len       = n_quant + n_overflow + n_residual
        n_blocks      = self._n_blocks_per_layer.get(0, 0)
        total_budget  = self.n_sink + self.K + self.n_local

        # Cache small enough → return full reconstruction.
        if seq_len <= total_budget:
            self.stats["full_attention_steps"] += 1
            return self._reconstruct_full()

        # Periodic refresh.
        if (self.refresh_interval > 0
                and step > 0
                and step % self.refresh_interval == 0):
            self.prev_query = None
            self.prev_seq_len = None
            self.cached_indices = None
            self.stats["refresh_steps"] += 1
            return self._reconstruct_full()

        latest_k = past_key_values[-1][0][:, :, -1:, :]      # (B, H, 1, D)

        if self.proxy_history <= 1:
            proxy_q = latest_k
        else:
            if (self._query_history is None
                    or self._query_history.shape[1] != latest_k.shape[1]
                    or self._query_history.shape[3] != latest_k.shape[3]
                    or self._query_history.device != latest_k.device):
                self._query_history = latest_k.expand(
                    -1, -1, self.proxy_history, -1
                ).contiguous().clone()
            else:
                self._query_history = torch.cat(
                    [self._query_history[:, :, 1:, :], latest_k], dim=2
                )
            if self.proxy_pool == "max":
                proxy_q = self._query_history.max(dim=2, keepdim=True).values
            else:
                proxy_q = self._query_history.mean(dim=2, keepdim=True)

        # Selection cache lookup (now also seq_len-aware — see _can_reuse_cache).
        if self._can_reuse_cache(proxy_q, seq_len):
            self.stats["cache_hits"] += 1
            return self._gather_selected(self.cached_indices)

        self.stats["cache_misses"] += 1

        if _DEBUG_PROFILE:
            _cuda_sync()
            t0 = time.perf_counter()

        per_layer_idx = self._hybrid_select(
            proxy_q, n_blocks, n_overflow, n_residual, seq_len,
        )

        if _DEBUG_PROFILE:
            _cuda_sync()
            self.stats["time_scoring_ms"] += (time.perf_counter() - t0) * 1000.0

        self.prev_query = proxy_q.detach().clone()
        self.prev_seq_len = seq_len
        self.cached_indices = per_layer_idx

        return self._gather_selected(per_layer_idx)

    # ── Hybrid scoring + selection ───────────────────────────────────────────

    def _hybrid_select(
        self,
        proxy_q: torch.Tensor,
        n_blocks: int, n_overflow: int, n_residual: int, seq_len: int,
    ) -> Dict[int, torch.Tensor]:
        """Score on quantized state, return {layer_idx: indices (S_sel,)}."""
        device = proxy_q.device
        n_sink_eff = self.n_sink if self.use_sink_tokens else 0
        n_local_eff = self.n_local if self.use_local_tokens else 0
        recent_start = max(seq_len - n_local_eff, 0)

        if (self.dynamic_sinks
                and self._sink_positions is not None
                and n_sink_eff > 0):
            sp = self._sink_positions
            sink_idx = sp[(sp < seq_len) & (sp < recent_start)]
        else:
            sink_end = min(n_sink_eff, recent_start)
            sink_idx = torch.arange(0, sink_end, device=device)

        recent_idx = torch.arange(recent_start, seq_len, device=device)

        if sink_idx.numel() + recent_idx.numel() >= seq_len:
            base_idx = torch.cat([sink_idx, recent_idx]).unique().sort().values
            return {l: base_idx for l in self.kivi.cache}

        pinned_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if sink_idx.numel() > 0:
            pinned_mask[sink_idx] = True
        if recent_idx.numel() > 0:
            pinned_mask[recent_idx] = True

        per_layer_idx: Dict[int, torch.Tensor] = {}
        for layer_idx, state in self.kivi.cache.items():
            scores = self._score_layer(
                layer_idx, proxy_q, state,
                n_blocks, n_overflow, n_residual,
            )                                                # (seq_len,)
            scores = scores.masked_fill(pinned_mask, float("-inf"))

            k_actual = min(self.K, int((~pinned_mask).sum().item()))
            if k_actual > 0:
                mid_topk = scores.topk(k_actual).indices
            else:
                mid_topk = torch.empty((0,), dtype=torch.long, device=device)

            all_idx = torch.cat([sink_idx, mid_topk, recent_idx]) \
                          .unique().sort().values
            per_layer_idx[layer_idx] = all_idx
        return per_layer_idx

    def _score_layer(
        self,
        layer_idx: int,
        proxy_q: torch.Tensor,
        state: dict,
        n_blocks: int, n_overflow: int, n_residual: int,
    ) -> torch.Tensor:
        """
        Compute a (seq_len,) score vector for one layer.

        Layout (must match the gather order):
            [block_0..block_{n_blocks-1} (n_quant tokens),
             overflow_0..overflow_{n_overflow-1},
             residual_0..residual_{n_residual-1}]
        """
        head_dim = proxy_q.shape[-1]
        scale_factor = head_dim ** 0.5
        H = proxy_q.shape[1]
        n_quant = self._n_quant_per_layer.get(layer_idx, 0)

        # `block_token` is populated as (H, n_quant) by the end of this
        # block regardless of which scoring mode is active.
        block_token: torch.Tensor | None = None

        if self.score_mode in ("centroid", "maxpool"):
            if n_blocks > 0:
                summary = (self.centroids_k[layer_idx][:, :, :n_blocks, :]
                           if self.score_mode == "centroid"
                           else self.maxpool_k[layer_idx][:, :, :n_blocks, :])
                # (1,H,1,D) @ (B,H,D,n_blocks) → (1,H,1,n_blocks) → (H, n_blocks)
                block_raw = torch.matmul(
                    proxy_q, summary.transpose(-2, -1)
                ).squeeze(2).squeeze(0) / scale_factor
            else:
                block_raw = proxy_q.new_zeros((H, 0))

            # Two-pass rerank: top-(factor*K_blocks) blocks from the cheap
            # score path get exact `quant_score` over their tokens; the rest
            # are masked with -inf so they can't win the downstream topk.
            if (self.two_pass_factor > 0 and n_blocks > 0 and n_quant > 0):
                K_blocks = max(1, self.K // self.group_size)
                n_keep_blocks = min(n_blocks, self.two_pass_factor * K_blocks)
                blk_score = block_raw.float().sum(dim=0)
                top_blocks = blk_score.topk(n_keep_blocks).indices

                blk_ids = self._block_id_for_quant[layer_idx]
                cand_mask = torch.isin(blk_ids, top_blocks)
                cand_idx = cand_mask.nonzero(as_tuple=True)[0]

                if cand_idx.numel() > 0:
                    kq_full = self.kq_stack[layer_idx][:, :, :n_quant, :]
                    ks_full = self.ks_stack[layer_idx][:, :, :n_quant, :]
                    kz_full = self.kz_stack[layer_idx][:, :, :n_quant, :]
                    kq_c = kq_full.index_select(2, cand_idx)
                    ks_c = ks_full.index_select(2, cand_idx)
                    kz_c = kz_full.index_select(2, cand_idx)
                    exact_c = quant_score(proxy_q, kq_c, ks_c, kz_c)

                    block_token = torch.full(
                        (H, n_quant), float("-inf"),
                        device=block_raw.device, dtype=torch.float32,
                    )
                    block_token.index_copy_(1, cand_idx, exact_c.float())

        else:  # "quantized" — exact over all tokens
            if n_quant > 0:
                kq = self.kq_stack[layer_idx][:, :, :n_quant, :]
                ks = self.ks_stack[layer_idx][:, :, :n_quant, :]
                kz = self.kz_stack[layer_idx][:, :, :n_quant, :]
                block_raw = quant_score(proxy_q, kq, ks, kz)        # (H, n_quant)
            else:
                block_raw = proxy_q.new_zeros((H, 0))

        # ── Overflow (exact) ──
        if n_overflow > 0:
            over_raw = torch.matmul(
                proxy_q, state['overflow_k'].transpose(-2, -1)
            ).squeeze(2).squeeze(0) / scale_factor              # (H, n_overflow)
        else:
            over_raw = proxy_q.new_zeros((H, 0))

        # ── Residual (exact) ──
        if n_residual > 0:
            res_raw = torch.matmul(
                proxy_q, state['residual_k'].transpose(-2, -1)
            ).squeeze(2).squeeze(0) / scale_factor              # (H, n_residual)
        else:
            res_raw = proxy_q.new_zeros((H, 0))

        # ── Concat in cache order ──
        if block_token is None:
            if self.score_mode in ("centroid", "maxpool") and n_blocks > 0:
                block_ids = self._block_id_for_quant[layer_idx]
                block_token = block_raw.index_select(1, block_ids)
            else:
                block_token = block_raw

        # block_raw / block_token may differ in dtype from over/res depending on
        # the kernel return type (quant_score → fp32, others → fp16). Promote
        # everything to a common dtype for the cat.
        common = torch.float32
        block_token = block_token.to(common)
        over_raw    = over_raw.to(common)
        res_raw     = res_raw.to(common)
        full = torch.cat([block_token, over_raw, res_raw], dim=1)   # (H, seq_len)

        # ── Head soft voting + criticality ──
        if self.use_head_softmax:
            normalized = torch.softmax(full, dim=-1)
        else:
            normalized = full

        if self.use_criticality_weights and self.head_weights is not None:
            normalized = normalized * self.head_weights.float().unsqueeze(-1)

        return normalized.sum(dim=0)                                # (seq_len,)

    # ── Selected-slice gather (vectorised selective dequant) ───────────────

    def _gather_selected(self, per_layer_idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Build (B,H,K_sel,D) per-layer past_kv via single-shot dequant."""
        if _DEBUG_PROFILE:
            _cuda_sync()
            t0 = time.perf_counter()

        result = []
        for layer_idx, state in self.kivi.cache.items():
            idx = (per_layer_idx if isinstance(per_layer_idx, torch.Tensor)
                   else per_layer_idx[layer_idx])
            sel_k, sel_v = self._materialise(layer_idx, state, idx)
            result.append((sel_k, sel_v))

        if _DEBUG_PROFILE:
            _cuda_sync()
            self.stats["time_materialise_ms"] += (time.perf_counter() - t0) * 1000.0
        return tuple(result)

    def _materialise(
        self, layer_idx: int, state: dict, idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selectively dequantize and gather one layer's (K, V).

        Vectorised single-shot path:  given the global indices of the selected
        tokens, gather their uint8 K/V slices and per-token (K) / per-block (V)
        scale/zero from the pre-stacked buffers, then run one fp16 multiply-add
        to produce the final K/V.  No Python loop over blocks; no full-block
        dequantization.
        """
        n_quant = self._n_quant_per_layer.get(layer_idx, 0)
        n_over  = state['overflow_k'].shape[2]
        n_res   = state['residual_k'].shape[2]

        in_quant_mask = idx < n_quant
        in_over_mask  = (idx >= n_quant) & (idx < n_quant + n_over)
        in_res_mask   = idx >= n_quant + n_over

        pieces_k: List[torch.Tensor] = []
        pieces_v: List[torch.Tensor] = []
        positions: List[torch.Tensor] = []

        # ── Quantized region: SINGLE batched gather + dequant ──
        if in_quant_mask.any():
            quant_idx = idx[in_quant_mask]                          # (S_q,)
            n_sel = quant_idx.shape[0]

            # K side: per-token scale/zero (B,H,n_quant,1) → gather along dim=2.
            kq = self.kq_stack[layer_idx][:, :, :n_quant, :]
            ks = self.ks_stack[layer_idx][:, :, :n_quant, :]
            kz = self.kz_stack[layer_idx][:, :, :n_quant, :]
            sel_qk = kq.index_select(2, quant_idx)                  # (B,H,S_q,D) uint8
            sel_sk = ks.index_select(2, quant_idx)                  # (B,H,S_q,1) fp16
            sel_zk = kz.index_select(2, quant_idx)                  # (B,H,S_q,1) fp16
            sel_k_quant = sel_qk.to(torch.float16) * sel_sk + sel_zk

            # V side: per-block-per-channel scale/zero (B,H,n_blocks,D).
            # Need block_id for each selected token to index into vs/vz_stack.
            block_ids = self._block_id_for_quant[layer_idx].index_select(0, quant_idx)
            vq = self.vq_stack[layer_idx][:, :, :n_quant, :]
            n_blocks = self._n_blocks_per_layer[layer_idx]
            vs = self.vs_stack[layer_idx][:, :, :n_blocks, :]
            vz = self.vz_stack[layer_idx][:, :, :n_blocks, :]
            sel_qv = vq.index_select(2, quant_idx)                  # (B,H,S_q,D) uint8
            sel_sv = vs.index_select(2, block_ids)                  # (B,H,S_q,D) fp16
            sel_zv = vz.index_select(2, block_ids)                  # (B,H,S_q,D) fp16
            sel_v_quant = sel_qv.to(torch.float16) * sel_sv + sel_zv

            pieces_k.append(sel_k_quant)
            pieces_v.append(sel_v_quant)
            positions.append(in_quant_mask.nonzero(as_tuple=False).squeeze(1))

            # Bookkeeping (legacy counter kept for backward compat reporting)
            self.stats["tokens_dequantized"] += n_sel
            self.stats["blocks_dequantized"] += int(
                torch.unique(block_ids).numel()
            )

        # ── Overflow region: FP16 gather ──
        if in_over_mask.any():
            over_idx = idx[in_over_mask] - n_quant
            pieces_k.append(state['overflow_k'].index_select(2, over_idx))
            pieces_v.append(state['overflow_v'].index_select(2, over_idx))
            positions.append(in_over_mask.nonzero(as_tuple=False).squeeze(1))

        # ── Residual region: FP16 gather ──
        if in_res_mask.any():
            res_idx = idx[in_res_mask] - n_quant - n_over
            pieces_k.append(state['residual_k'].index_select(2, res_idx))
            pieces_v.append(state['residual_v'].index_select(2, res_idx))
            positions.append(in_res_mask.nonzero(as_tuple=False).squeeze(1))

        # Assemble in original idx order so positional info matches `idx`.
        out_k = torch.cat(pieces_k, dim=2)
        out_v = torch.cat(pieces_v, dim=2)
        order = torch.cat(positions).argsort()
        out_k = out_k.index_select(2, order)
        out_v = out_v.index_select(2, order)

        # Optional RoPE re-rotation (Option A escape hatch — disabled by default).
        if self.apply_rope_correction:
            S_full = n_quant + n_over + n_res
            S_sel = out_k.shape[2]
            shift = S_sel - S_full
            delta = torch.full((S_sel,), float(shift),
                               dtype=torch.float32, device=out_k.device)
            out_k = apply_rope_delta(out_k, delta, rope_theta=self.rope_theta)

        return out_k, out_v

    def _reconstruct_full(self):
        """Fall back to KIVI's full FP16 reconstruction (used pre-budget)."""
        result = []
        for state in self.kivi.cache.values():
            parts_k, parts_v = [], []
            if state['k_blocks']:
                parts_k.append(self.kivi._dequantize_blocks(state['k_blocks']))
                parts_v.append(self.kivi._dequantize_blocks(state['v_blocks']))
            if state['overflow_k'].shape[2] > 0:
                parts_k.append(state['overflow_k'])
                parts_v.append(state['overflow_v'])
            parts_k.append(state['residual_k'])
            parts_v.append(state['residual_v'])
            result.append((torch.cat(parts_k, dim=2),
                           torch.cat(parts_v, dim=2)))
        return tuple(result)

    # ── Selection cache (cosine similarity + seq_len) ───────────────────────

    def _can_reuse_cache(self, current_query: torch.Tensor, current_seq_len: int) -> bool:
        """
        Can we reuse the previously-cached selection?

        v2 fix: also invalidate when seq_len has changed since the cached
        query was scored. Without this guard, freshly-appended tokens would
        not appear in the sparse cache returned to the model — silent
        correctness bug for long generations / step-by-step PPL.
        """
        if not self.use_selection_cache:
            return False
        if (self.prev_query is None
                or self.cached_indices is None
                or self.prev_seq_len is None):
            return False
        if current_seq_len != self.prev_seq_len:
            return False
        q_curr = current_query.squeeze(2).float()
        q_prev = self.prev_query.squeeze(2).float()
        cos = torch.nn.functional.cosine_similarity(q_curr, q_prev, dim=-1)
        return cos.mean().item() >= self.cache_similarity_threshold

    # ── Reset / accounting ──────────────────────────────────────────────────

    def reset(self):
        self.kivi.reset()
        self.step_counter = 0
        self.prev_query = None
        self.prev_seq_len = None
        self.cached_indices = None
        self.head_weights = None
        self._query_history = None
        self._sink_positions = None
        self.kq_stack.clear()
        self.ks_stack.clear()
        self.kz_stack.clear()
        self.vq_stack.clear()
        self.vs_stack.clear()
        self.vz_stack.clear()
        self.centroids_k.clear()
        self.maxpool_k.clear()
        self._n_quant_per_layer.clear()
        self._n_blocks_per_layer.clear()
        self.block_sizes.clear()
        self._block_id_for_quant.clear()
        # Preserve perf counter keys but zero them.
        for k in list(self.stats.keys()):
            self.stats[k] = 0 if not isinstance(self.stats[k], float) else 0.0

    def get_kv_size_bytes(self, past_key_values):
        """
        Storage = KIVI compressed state + hybrid scratch tensors actually
        held in GPU memory during decode.

        v2: previously delegated entirely to KIVI, undercounting the parallel
        kq_stack / ks_stack / kz_stack / vq_stack / vs_stack / vz_stack /
        centroids_k buffers by ~2x.  Now we count them at their *logical*
        size (n_quant or n_blocks slots, not their pre-allocated capacity)
        so the number tracks what the algorithm actually needs.
        """
        total = self.kivi.get_kv_size_bytes(past_key_values)

        for layer_idx in self.kq_stack:
            n_quant  = self._n_quant_per_layer.get(layer_idx, 0)
            n_blocks = self._n_blocks_per_layer.get(layer_idx, 0)

            kq = self.kq_stack[layer_idx]
            ks = self.ks_stack[layer_idx]
            kz = self.kz_stack[layer_idx]
            vq = self.vq_stack[layer_idx]
            vs = self.vs_stack[layer_idx]
            vz = self.vz_stack[layer_idx]
            cs = self.centroids_k[layer_idx]
            mp = self.maxpool_k.get(layer_idx)

            # Note: these are MIRRORS of KIVI's k_blocks / v_blocks data, so
            # in raw GPU bytes they double-count.  We report them anyway
            # because the GPU does hold both (KIVI's per-block list of
            # tensors *and* our stacked buffers).  The fp16 baseline number
            # remains the apples-to-apples comparison.
            B, H, _, D = kq.shape
            per_token_bytes = lambda t: B * H * n_quant * t.shape[-1] * t.element_size()
            per_block_bytes = lambda t: B * H * n_blocks * t.shape[-1] * t.element_size()
            total += per_token_bytes(kq)        # uint8 mirror of K
            total += B * H * n_quant * 1 * ks.element_size()
            total += B * H * n_quant * 1 * kz.element_size()
            total += per_token_bytes(vq)        # uint8 mirror of V
            total += per_block_bytes(vs)
            total += per_block_bytes(vz)
            total += per_block_bytes(cs)
            if mp is not None:
                total += per_block_bytes(mp)

            if self._query_history is not None and layer_idx == 0:
                total += self._query_history.numel() * self._query_history.element_size()
        return total

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def get_ablation_config(self) -> dict:
        design_label = {
            "centroid":  "(a)",
            "quantized": "(c)",
            "maxpool":   "(m)",
        }.get(self.score_mode, self.score_mode)
        return {
            "method": "kivi_topk_hybrid",
            "design": design_label,
            "score_mode":        self.score_mode,
            "bits":              self.bits,
            "group_size":        self.group_size,
            "residual_length":   self.residual_length,
            "K":                 self.K,
            "n_sink":            self.n_sink,
            "n_local":           self.n_local,
            "two_pass_factor":   self.two_pass_factor,
            "proxy_history":     self.proxy_history,
            "proxy_pool":        self.proxy_pool,
            "dynamic_sinks":     self.dynamic_sinks,
            "use_head_softmax":        self.use_head_softmax,
            "use_criticality_weights": self.use_criticality_weights,
            "use_selection_cache":     self.use_selection_cache,
            "use_sink_tokens":         self.use_sink_tokens,
            "use_local_tokens":        self.use_local_tokens,
        }

    def get_stats(self) -> dict:
        s = dict(self.stats)
        scoring = s["cache_hits"] + s["cache_misses"]
        s["cache_hit_rate"] = (
            s["cache_hits"] / scoring if scoring > 0 else 0.0
        )
        if _DEBUG_PROFILE and s["decode_steps"] > 0:
            n = s["decode_steps"]
            sys.stderr.write(
                f"[hybrid-perf] steps={n}  "
                f"score={s['time_scoring_ms']/n:.3f}ms  "
                f"materialise={s['time_materialise_ms']/n:.3f}ms  "
                f"sync={s['time_sync_ms']/n:.3f}ms  "
                f"avg_tokens_dequant={s['tokens_dequantized']/max(n,1):.1f}\n"
            )
        return s


if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path
    _sys.path.insert(0, str(Path(__file__).parent.parent))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark.runner import generate_with_method

    model_name = "facebook/opt-125m"
    print(f"Loading {model_name} for KIVI+TopK hybrid smoke test…")
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    m.eval()

    method = KIVI_TopK_Method(
        bits=4, residual_length=16, group_size=32,
        K=32, n_sink=4, n_local=16, refresh_interval=10,
    )
    text, metrics = generate_with_method(
        m, tok, method,
        prompt="The history of machine learning began",
        max_new_tokens=10, device="cuda",
    )
    print(f"Generated: {text}")
    print(f"KV cache MB: {metrics['kv_cache_mb']:.4f}")
    print(f"Stats: {method.get_stats()}")
    print("KIVI+TopK hybrid smoke test PASSED")
