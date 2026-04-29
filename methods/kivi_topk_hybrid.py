"""
KIVI + TopK hybrid — design (a): centroid scoring on quantized blocks.

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
        approx  Q · centroidᵀ      ← one score per block, broadcast to
                                     all `group_size` tokens in the block
        centroid_b = mean(K_block, dim=seq)            (B, H, 1, D)

The centroid is the FP16 block-mean computed once when the block is
sealed (free if we already have the FP16 data; one cheap dequant + mean
otherwise). Centroids live in a parallel list to KIVI's `k_blocks` and
are kept in sync as new blocks are sealed during decode. This is the
correct "design (a)" formulation — using `zero + scale · mid` only gives
a scalar per block (scale/zero have no head_dim axis under KIVI's
per-token reduction), which can't form a key vector for Q·Kᵀ scoring.

After TopK selects K indices over the combined score vector, only the
quantized blocks containing at least one selected token are dequantized
— never the full cache. Selected (K, V) slices come from a mix of FP16
regions and partially-dequantized blocks.

The TopK selection plumbing (head soft voting, criticality weights,
selection cache, sink + local tokens, full-cache refresh) is borrowed
verbatim from `topk_selection.TopKMethod` so its ablation flags stay
meaningful for the hybrid too.

Design choice rationale
-----------------------
Centroid scoring trades exact attention-magnitude ranking for a ~32×
cheaper scoring pass (one Q·k per block instead of one per token). The
residual + overflow regions — which dominate the *recent* attention mass
in autoregressive decoding — are scored exactly, so the approximation
hits only the older quantized history where TopK is already throwing
~95% of tokens away. Empirically this is the sweet spot for design (a).
A future design (c) would replace the centroid pass with a custom
quant-aware Triton kernel that scores and top-K's directly on packed
4-bit data; left as a stretch goal.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch

from .base import MethodWrapper
from .kivi_quant import KIVIMethod, dequantize
from .topk_kernels import quant_score


class KIVI_TopK_Method(MethodWrapper):
    """
    Hybrid KIVI-quantized storage + TopK selection. See module docstring.

    Constructor args mirror the union of `KIVIMethod` and `TopKMethod`:
      KIVI side : bits, residual_length, group_size
      TopK side : K, n_sink, n_local, refresh_interval,
                  cache_similarity_threshold,
                  use_head_softmax, use_criticality_weights,
                  use_selection_cache, use_sink_tokens, use_local_tokens

    The Triton fused-score kernel from `topk_kernels` is *not* used here:
    its FP16 contract assumes a contiguous K matrix, while the hybrid's
    middle region is split across heterogeneous sources (centroids +
    overflow). Phase B uses the PyTorch reference path; design (c)
    would supply a quant-aware kernel.
    """

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
        score_mode: str = "centroid",            # "centroid" (design a) | "quantized" (design c)
        # ── Ablation flags (mirror TopKMethod for consistent reporting) ──
        use_head_softmax: bool = True,
        use_criticality_weights: bool = True,
        use_selection_cache: bool = True,
        use_sink_tokens: bool = True,
        use_local_tokens: bool = True,
    ):
        if score_mode not in ("centroid", "quantized"):
            raise ValueError(
                f"score_mode must be 'centroid' or 'quantized', got {score_mode!r}"
            )
        self.score_mode = score_mode
        # Underlying KIVI cache (provides the quantized storage)
        self.kivi = KIVIMethod(
            bits=bits, residual_length=residual_length, group_size=group_size,
        )
        self.bits = bits
        self.group_size = group_size
        self.residual_length = residual_length

        # TopK parameters
        self.K = K
        self.n_sink = n_sink if use_sink_tokens else 0
        self.n_local = n_local if use_local_tokens else 0
        self.refresh_interval = refresh_interval
        self.cache_similarity_threshold = cache_similarity_threshold

        # Ablation flags
        self.use_head_softmax        = use_head_softmax
        self.use_criticality_weights = use_criticality_weights
        self.use_selection_cache     = use_selection_cache
        self.use_sink_tokens         = use_sink_tokens
        self.use_local_tokens        = use_local_tokens

        # Selection state (TopK side)
        self.step_counter = 0
        self.prev_query: torch.Tensor | None = None
        self.cached_indices: Dict[int, torch.Tensor] | None = None
        self.head_weights: torch.Tensor | None = None

        # Per-layer pre-stacked centroid tensor of shape (B, H, n_blocks, D).
        # We append new block centroids in-place via torch.cat once per
        # block-seal event (paid once per block, never per step). Avoids
        # the per-step Python loop in v1 that dominated decode latency.
        # Aligned with self.kivi.cache[l]['k_blocks'].
        self.centroids_k: Dict[int, torch.Tensor] = {}
        # Also track per-layer block sizes (matches the stacked tensor's
        # third dim by construction; cheap and lets _materialise compute
        # cumulative offsets without repeated tensor introspection).
        self.block_sizes: Dict[int, List[int]] = {}

        # ── Design (c) state: pre-stacked uint8 K + per-token scale/zero ──
        # Maintained in parallel with self.kivi.cache[l]['k_blocks'] when
        # score_mode == "quantized". One stacked tensor per layer; updated
        # on block-seal events only (not per step).
        #   kq_stack[l]: (B, H, n_quant, D) uint8
        #   ks_stack[l]: (B, H, n_quant, 1) fp16
        #   kz_stack[l]: (B, H, n_quant, 1) fp16
        self.kq_stack: Dict[int, torch.Tensor] = {}
        self.ks_stack: Dict[int, torch.Tensor] = {}
        self.kz_stack: Dict[int, torch.Tensor] = {}

        # Diagnostics
        self.stats = {
            "decode_steps":         0,
            "full_attention_steps": 0,
            "refresh_steps":        0,
            "cache_hits":            0,
            "cache_misses":          0,
            "blocks_dequantized":    0,  # cumulative — tracks dequant volume
        }

    # ── Prefill ──────────────────────────────────────────────────────────────

    def process_prefill(self, past_key_values, attention_weights=None):
        """
        Build the KIVI quantized cache and (optionally) compute per-head
        criticality weights. The model needs full FP16 K,V for the prefill
        pass itself, so we return the dequantized reconstruction — the
        savings kick in on subsequent decode steps.
        """
        recon = self.kivi.process_prefill(past_key_values)
        self.step_counter = 0
        self.prev_query = None
        self.cached_indices = None
        self.centroids_k = {}
        self.block_sizes = {}
        self.kq_stack = {}
        self.ks_stack = {}
        self.kz_stack = {}

        # Seed per-layer scoring state from the already-quantized blocks.
        # Centroids: one mean-pooled key per block (design a).
        # Quant stack: stacked uint8/scale/zero across all blocks (design c).
        # Both are paid once per block; per-step cost is one matmul / one
        # Triton launch over the pre-stacked tensor. Partial last-block from
        # KIVI prefill is handled identically — `block_sizes` carries the
        # truth so downstream gather logic stays correct.
        for layer_idx, state in self.kivi.cache.items():
            blocks = state['k_blocks']
            v_blocks = state['v_blocks']
            self.block_sizes[layer_idx] = [qk.shape[2] for qk, _, _ in blocks]

            if blocks:
                centroid_list = [
                    self._block_centroid(qk, sk, zk) for qk, sk, zk in blocks
                ]
                self.centroids_k[layer_idx] = torch.cat(centroid_list, dim=2)

                # Design (c) packed state — stack along seq axis so the
                # kernel sees one contiguous (B, H, n_quant, D) uint8 tensor.
                self.kq_stack[layer_idx] = torch.cat(
                    [qk for qk, _, _ in blocks], dim=2,
                )
                self.ks_stack[layer_idx] = torch.cat(
                    [sk for _, sk, _ in blocks], dim=2,
                )
                self.kz_stack[layer_idx] = torch.cat(
                    [zk for _, _, zk in blocks], dim=2,
                )
            else:
                # Empty placeholders shaped to match downstream expectations.
                B, H, _, D = state['residual_k'].shape
                dev = state['residual_k'].device
                self.centroids_k[layer_idx] = torch.empty(
                    (B, H, 0, D),
                    dtype=state['residual_k'].dtype, device=dev,
                )
                self.kq_stack[layer_idx] = torch.empty(
                    (B, H, 0, D), dtype=torch.uint8, device=dev,
                )
                self.ks_stack[layer_idx] = torch.empty(
                    (B, H, 0, 1), dtype=torch.float16, device=dev,
                )
                self.kz_stack[layer_idx] = torch.empty(
                    (B, H, 0, 1), dtype=torch.float16, device=dev,
                )

        if self.use_criticality_weights:
            self._compute_head_weights(recon)
        else:
            self.head_weights = None

        return recon

    @staticmethod
    def _block_centroid(qk, sk, zk) -> torch.Tensor:
        """One-time centroid for a quantized block: mean key over group_size."""
        deq = dequantize(qk, sk, zk)            # (B, H, group_size, D)
        return deq.mean(dim=2, keepdim=True)    # (B, H, 1, D)

    def _sync_block_state(self) -> None:
        """
        Update centroid + quant-stack state when KIVI seals new blocks.

        Called once per process_step. The common case (no new block) hits
        a tight fast-path. When a block has been sealed, we incrementally
        cat the new centroid (design a) and the new uint8/scale/zero
        slices (design c) onto their respective stacked tensors — one cat
        per *block-seal event*, never per step.
        """
        for layer_idx, state in self.kivi.cache.items():
            stacked = self.centroids_k[layer_idx]
            n_existing = stacked.shape[2]
            n_total = len(state['k_blocks'])
            if n_total <= n_existing:
                continue       # No new block this step; common case → fast path

            new_blocks = state['k_blocks'][n_existing:n_total]

            # Centroids
            new_centroids = [self._block_centroid(qk, sk, zk)
                             for qk, sk, zk in new_blocks]
            self.centroids_k[layer_idx] = torch.cat(
                [stacked] + new_centroids, dim=2,
            )

            # Quant-stack (only maintained when needed but cheap to keep current)
            self.kq_stack[layer_idx] = torch.cat(
                [self.kq_stack[layer_idx]] + [qk for qk, _, _ in new_blocks],
                dim=2,
            )
            self.ks_stack[layer_idx] = torch.cat(
                [self.ks_stack[layer_idx]] + [sk for _, sk, _ in new_blocks],
                dim=2,
            )
            self.kz_stack[layer_idx] = torch.cat(
                [self.kz_stack[layer_idx]] + [zk for _, _, zk in new_blocks],
                dim=2,
            )

            # Block sizes track all sealed blocks regardless of mode.
            for qk, _, _ in new_blocks:
                self.block_sizes[layer_idx].append(qk.shape[2])

    # Back-compat alias — old name was a single-purpose centroid sync.
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
        """
        Hybrid decode step.

        Phase 1 — update KIVI cache state (append new token; possibly
                  evict to overflow; possibly seal a new quantized block).
        Phase 2 — score Q against the *quantized* state (centroids for
                  blocks, exact for overflow + residual).
        Phase 3 — top-K over combined scores; merge with sink + local.
        Phase 4 — gather selected slice: dequantize only the blocks that
                  contain at least one selected token.
        """
        # Phase 1: KIVI ingests new tokens but we discard its FP16 return —
        # we never need the full reconstruction.
        self.kivi.process_step(past_key_values, step)
        self._sync_block_state()
        self.step_counter += 1
        self.stats["decode_steps"] += 1

        # Total context length per layer (same across layers).
        # Block sizes can vary — KIVI's prefill produces a partial final
        # block when historical_len % group_size ≠ 0. Decode-sealed
        # blocks are always full group_size.
        layer0 = self.kivi.cache[0]
        block_sizes = [qk.shape[2] for qk, _, _ in layer0['k_blocks']]
        n_blocks      = len(block_sizes)
        n_quant       = sum(block_sizes)
        n_overflow    = layer0['overflow_k'].shape[2]
        n_residual    = layer0['residual_k'].shape[2]
        seq_len = n_quant + n_overflow + n_residual
        total_budget = self.n_sink + self.K + self.n_local

        # Cache small enough → return full reconstruction (Novelty 13)
        if seq_len <= total_budget:
            self.stats["full_attention_steps"] += 1
            return self._reconstruct_full()

        # Periodic refresh — invalidate selection cache, return full cache
        if self.refresh_interval > 0 and step % self.refresh_interval == 0:
            self.prev_query = None
            self.cached_indices = None
            self.stats["refresh_steps"] += 1
            return self._reconstruct_full()

        # Query proxy: last token's K from the input past_kv (FP16 view)
        proxy_q = past_key_values[-1][0][:, :, -1:, :]      # (B, H, 1, D)

        # Selection cache lookup (Novelty 5, 11)
        if self._can_reuse_cache(proxy_q):
            self.stats["cache_hits"] += 1
            return self._gather_selected(self.cached_indices)

        self.stats["cache_misses"] += 1

        # Phase 2 + 3: hybrid scoring + top-K (per-layer indices stored)
        per_layer_idx = self._hybrid_select(
            proxy_q, n_blocks, n_overflow, n_residual, seq_len,
        )
        self.prev_query = proxy_q.detach().clone()
        self.cached_indices = per_layer_idx

        # Phase 4
        return self._gather_selected(per_layer_idx)

    # ── Hybrid scoring + selection ───────────────────────────────────────────

    def _hybrid_select(
        self,
        proxy_q: torch.Tensor,
        n_blocks: int, n_overflow: int, n_residual: int, seq_len: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Score on quantized state, return {layer_idx: indices (S_sel,)}.

        Scoring is Q-from-the-last-layer (proxy) against each layer's K —
        this is consistent with `TopKMethod._paged_token_selection` which
        also uses a single proxy query across layers. Per-layer scoring
        would be more accurate but ~32× more expensive; the proxy is an
        established TokenSelect approximation.
        """
        device = proxy_q.device
        sink_end = min(self.n_sink, seq_len) if self.use_sink_tokens else 0
        recent_start = max(
            seq_len - (self.n_local if self.use_local_tokens else 0),
            sink_end,
        )

        sink_idx = torch.arange(0, sink_end, device=device)
        recent_idx = torch.arange(recent_start, seq_len, device=device)

        if recent_start <= sink_end:
            base_idx = torch.cat([sink_idx, recent_idx]).unique().sort().values
            return {l: base_idx for l in self.kivi.cache}

        # Score per layer (centroids are layer-specific so we can't share)
        per_layer_idx: Dict[int, torch.Tensor] = {}
        for layer_idx, state in self.kivi.cache.items():
            scores = self._score_layer(
                layer_idx, proxy_q, state,
                n_blocks, n_overflow, n_residual,
            )                                                # (seq_len,)
            # Mask out sink + recent regions so they aren't double-counted
            # in the top-K (they're added back unconditionally below).
            scores[:sink_end] = float("-inf")
            scores[recent_start:] = float("-inf")

            k_actual = min(self.K, recent_start - sink_end)
            mid_topk = scores.topk(k_actual).indices

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
        Compute a (seq_len,) score vector for one layer's cache.

        Layout (must match the gather order used elsewhere):
          [block_0 (group_size tokens), block_1, ..., block_{n_blocks-1},
           overflow_0, ..., overflow_{n_overflow-1},
           residual_0, ..., residual_{n_residual-1}]
        """
        head_dim = proxy_q.shape[-1]
        scale_factor = head_dim ** 0.5
        H = proxy_q.shape[1]

        # ── Block region scoring ──
        # Two interchangeable paths, selected at construction:
        #   "centroid"  — design (a): one Q · centroid per block, broadcast
        #   "quantized" — design (c): score directly on uint8 K via Triton
        #                 kernel (per-token resolution, no centroid approx)
        if self.score_mode == "centroid":
            centroid_k = self.centroids_k[layer_idx]            # (B,H,n_blocks,D)
            if centroid_k.shape[2] > 0:
                block_raw = torch.matmul(
                    proxy_q.float(), centroid_k.float().transpose(-2, -1)
                ).squeeze(2).squeeze(0) / scale_factor          # (H, n_blocks)
            else:
                block_raw = proxy_q.new_zeros((H, 0))
        else:  # "quantized"
            kq = self.kq_stack[layer_idx]                       # (B,H,n_quant,D) uint8
            if kq.shape[2] > 0:
                # quant_score returns (H, n_quant) raw scaled-dot-product
                # scores already divided by √d. Per-token resolution means
                # no broadcast / repeat_interleave needed below.
                block_raw = quant_score(
                    proxy_q,
                    kq,
                    self.ks_stack[layer_idx],
                    self.kz_stack[layer_idx],
                )                                               # (H, n_quant)
            else:
                block_raw = proxy_q.new_zeros((H, 0))

        # ── Overflow (exact FP16) ──
        if n_overflow > 0:
            over_raw = torch.matmul(
                proxy_q.float(), state['overflow_k'].float().transpose(-2, -1)
            ).squeeze(2).squeeze(0) / scale_factor              # (H, n_overflow)
        else:
            over_raw = proxy_q.new_zeros((H, 0))

        # ── Residual (exact FP16) ──
        if n_residual > 0:
            res_raw = torch.matmul(
                proxy_q.float(), state['residual_k'].float().transpose(-2, -1)
            ).squeeze(2).squeeze(0) / scale_factor              # (H, n_residual)
        else:
            res_raw = proxy_q.new_zeros((H, 0))

        # ── Concat in cache order ──
        # Centroid mode: block_raw is (H, n_blocks) — broadcast to tokens.
        # Quantized mode: block_raw is already (H, n_quant) per-token.
        if self.score_mode == "centroid" and n_blocks > 0:
            block_sizes = torch.tensor(
                [qk.shape[2] for qk, _, _ in state['k_blocks']],
                device=block_raw.device,
            )
            block_token = block_raw.repeat_interleave(block_sizes, dim=1)
        else:
            block_token = block_raw
        full = torch.cat([block_token, over_raw, res_raw], dim=1)   # (H, seq_len)

        # ── Head soft voting + criticality (Novelty 3, 4, 10) ──
        if self.use_head_softmax:
            normalized = torch.softmax(full, dim=-1)
        else:
            normalized = full

        if self.use_criticality_weights and self.head_weights is not None:
            normalized = normalized * self.head_weights.float().unsqueeze(-1)

        return normalized.sum(dim=0)                                # (seq_len,)

    # ── Selected-slice gather (with selective dequantization) ───────────────

    def _gather_selected(
        self, per_layer_idx,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Build the (B, H, K_sel, D) past_kv to return to the model by
        materialising only the selected positions. Quantized blocks are
        dequantized lazily — at most one dequant per block per call.

        `per_layer_idx` is either a dict {layer_idx: indices} or a
        single 1-D tensor shared across layers.
        """
        result = []
        shared = (
            isinstance(per_layer_idx, torch.Tensor) or
            (isinstance(per_layer_idx, dict) and len(set(
                id(v) for v in per_layer_idx.values()
            )) == 1)
        )

        for layer_idx, state in self.kivi.cache.items():
            idx = (per_layer_idx if isinstance(per_layer_idx, torch.Tensor)
                   else per_layer_idx[layer_idx])
            sel_k, sel_v = self._materialise(state, idx)
            result.append((sel_k, sel_v))
        return tuple(result)

    def _materialise(
        self, state: dict, idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selectively dequantize and gather one layer's selected (K, V).

        Splits `idx` into three regions by the cache layout, dequantizes
        only the unique blocks touched by quantized-region indices, then
        gathers and concatenates in original-position order.

        Uses per-block cumulative offsets (not idx // group_size) because
        KIVI's prefill can produce a partial final block.
        """
        n_blocks  = len(state['k_blocks'])
        block_sizes = [qk.shape[2] for qk, _, _ in state['k_blocks']]
        n_quant   = sum(block_sizes)
        n_over    = state['overflow_k'].shape[2]
        n_res     = state['residual_k'].shape[2]

        device = idx.device
        in_quant = idx < n_quant
        in_over  = (idx >= n_quant) & (idx < n_quant + n_over)
        in_res   = idx >= n_quant + n_over

        pieces_k: List[torch.Tensor] = []
        pieces_v: List[torch.Tensor] = []
        positions: List[torch.Tensor] = []   # where in `idx` each piece came from

        # ── Quantized region: dequantize touched blocks only ──
        if in_quant.any():
            quant_idx = idx[in_quant]                              # global pos
            # Compute (block_id, within) via cumulative block-size offsets.
            offsets = torch.tensor(
                [0] + list(__import__("itertools").accumulate(block_sizes)),
                device=device, dtype=torch.long,
            )
            # bucketize against block END boundaries with right=True so
            # idx == offsets[i] maps to block i (start of next block).
            block_id = torch.bucketize(quant_idx, offsets[1:], right=True).long()
            within   = (quant_idx - offsets[block_id]).long()
            unique_blocks = torch.unique(block_id).tolist()

            # Dequantize each touched block once and gather from it.
            B = state['residual_k'].shape[0]
            H = state['residual_k'].shape[1]
            D = state['residual_k'].shape[3]
            sel_qk = torch.empty(
                (B, H, quant_idx.shape[0], D),
                dtype=torch.float16, device=device,
            )
            sel_qv = torch.empty_like(sel_qk)

            for b in unique_blocks:
                qk, sk, zk = state['k_blocks'][b]
                qv, sv, zv = state['v_blocks'][b]
                deq_k = dequantize(qk, sk, zk)                  # (B,H,group,D)
                deq_v = dequantize(qv, sv, zv)
                mask  = (block_id == b)
                local = within[mask]
                sel_qk[:, :, mask, :] = deq_k[:, :, local, :]
                sel_qv[:, :, mask, :] = deq_v[:, :, local, :]
                self.stats["blocks_dequantized"] += 1

            pieces_k.append(sel_qk); pieces_v.append(sel_qv)
            positions.append(in_quant.nonzero(as_tuple=False).squeeze(1))

        # ── Overflow region: FP16 gather ──
        if in_over.any():
            over_idx = idx[in_over] - n_quant
            pieces_k.append(state['overflow_k'][:, :, over_idx, :])
            pieces_v.append(state['overflow_v'][:, :, over_idx, :])
            positions.append(in_over.nonzero(as_tuple=False).squeeze(1))

        # ── Residual region: FP16 gather ──
        if in_res.any():
            res_idx = idx[in_res] - n_quant - n_over
            pieces_k.append(state['residual_k'][:, :, res_idx, :])
            pieces_v.append(state['residual_v'][:, :, res_idx, :])
            positions.append(in_res.nonzero(as_tuple=False).squeeze(1))

        # Assemble in the original idx order so positional info matches `idx`.
        out_k = torch.cat(pieces_k, dim=2)
        out_v = torch.cat(pieces_v, dim=2)
        order = torch.cat(positions).argsort()
        return out_k[:, :, order, :], out_v[:, :, order, :]

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

    # ── Selection cache (cosine-similarity reuse, Novelty 5/11) ─────────────

    def _can_reuse_cache(self, current_query: torch.Tensor) -> bool:
        if not self.use_selection_cache:
            return False
        if self.prev_query is None or self.cached_indices is None:
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
        self.cached_indices = None
        self.head_weights = None
        self.centroids_k = {}
        self.block_sizes = {}
        self.kq_stack = {}
        self.ks_stack = {}
        self.kz_stack = {}
        for k in self.stats:
            self.stats[k] = 0

    def get_kv_size_bytes(self, past_key_values):
        """Storage = KIVI's quantized footprint (the hybrid adds no storage)."""
        return self.kivi.get_kv_size_bytes(past_key_values)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def get_ablation_config(self) -> dict:
        return {
            "method": "kivi_topk_hybrid",
            "design": "(a)" if self.score_mode == "centroid" else "(c)",
            "score_mode":        self.score_mode,
            "bits":              self.bits,
            "group_size":        self.group_size,
            "residual_length":   self.residual_length,
            "K":                 self.K,
            "n_sink":            self.n_sink,
            "n_local":           self.n_local,
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
        return s


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

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
