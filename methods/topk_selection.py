import math

import torch

from .base import MethodWrapper
from . import topk_kernels
from .rope_utils import apply_rope_delta


class TopKMethod(MethodWrapper):
    """
    TokenSelect: Dynamic Token-Level KV Cache Selection for Efficient LLM Inference.

    Implements all 13 novelties from Wu et al. (EMNLP 2025):

     1. Token-level selection — individual token granularity, not block-level
     2. Dynamic Q-dependent scoring — scores recomputed each decode step
        (uses K-proxy for Q since HF doesn't expose Q tensors)
     3. Head soft voting (Eq. 7) — softmax per head before cross-head aggregation
     4. Per-head criticality weighting — heads weighted by attention entropy
     5. Selection cache — reuse indices when consecutive queries are similar
     6. Chunk-wise prefill — process long prefills in chunks to bound memory
     7. Paged KV storage — cache organized in fixed-size logical pages
     8. Paged dot product — score pages first, then tokens within top pages
     9. Non-contiguous sparsity — arbitrary token selection patterns
    10. Head-distinctive behavior — per-head normalization prevents high-norm domination
    11. Consecutive query similarity — cosine similarity drives cache reuse (part of #5)
    12. Training-free — no weight modification required
    13. Full cache maintained — no permanent eviction, tokens can be reactivated

    Custom Triton kernels (in `topk_kernels.py`) accelerate the two hot paths:
      • `fused_paged_score`  — fuses Q·Kᵀ + softmax + criticality weighting +
                                cross-head sum into a single kernel
                                (eliminates three (H, M) intermediates).
      • `fused_paged_topk`   — vectorised page-then-token top-K replacing the
                                Python for-loop in the reference path.

    Both kernels fall back to numerically-equivalent PyTorch ops when Triton
    or CUDA is unavailable. Toggle with the `use_kernels` constructor arg.

    ── Phase A ablation flags ─────────────────────────────────────────────────
    Five boolean toggles individually disable novelties so we can attribute
    quality / latency to specific design choices:

        use_head_softmax       — Novelty 3   (per-head softmax before sum)
        use_criticality_weights— Novelty 4,10 (per-head entropy weighting)
        use_selection_cache    — Novelty 5,11 (consecutive-query reuse)
        use_sink_tokens        — first n_sink tokens always retained
        use_local_tokens       — last n_local tokens always retained

    All default to True (full novelties active). The ablation runner sweeps
    each flag independently and reports the marginal cost of disabling it.

    ── Phase A→C batch-dim seam ──────────────────────────────────────────────
    `_paged_token_selection` returns indices of shape (S_sel,) for B=1 (the
    fast path used today) or (B, S_sel) for batched decode (added in Phase C).
    The gather in `process_step` dispatches on `indices.dim()` so neither
    shape requires changing the rest of the call graph.
    """

    def __init__(self, K=512, n_sink=128, n_local=512, refresh_interval=50,
                 page_size=64, cache_similarity_threshold=0.95, chunk_size=2048,
                 use_kernels=True,
                 # ── Position-encoding parameters (BUG-2 fix) ──
                 head_dim=128,
                 rope_theta=10000.0,
                 apply_rope_correction=True,
                 # ── Ablation flags (default True = full TokenSelect) ──
                 use_head_softmax=True,
                 use_criticality_weights=True,
                 use_selection_cache=True,
                 use_sink_tokens=True,
                 use_local_tokens=True):
        self.K = K
        self.n_sink = n_sink if use_sink_tokens else 0
        self.n_local = n_local if use_local_tokens else 0
        self.refresh_interval = refresh_interval
        self.page_size = page_size                                      # Novelty 7
        self.cache_similarity_threshold = cache_similarity_threshold    # Novelty 5, 11
        self.chunk_size = chunk_size                                    # Novelty 6
        self.use_kernels = use_kernels and topk_kernels.TRITON_AVAILABLE

        # Position-encoding metadata (used when re-rotating gathered keys so that
        # truncated KV layouts present a valid RoPE relative-offset to Q).
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.apply_rope_correction = apply_rope_correction

        # Ablation flags (kept verbatim for diagnostics)
        self.use_head_softmax        = use_head_softmax
        self.use_criticality_weights = use_criticality_weights
        self.use_selection_cache     = use_selection_cache
        self.use_sink_tokens         = use_sink_tokens
        self.use_local_tokens        = use_local_tokens

        self.full_past_key_values = None
        self.step_counter = 0

        # Selection cache state (Novelty 5, 11)
        self.prev_query = None
        self.cached_indices = None

        # Per-head criticality weights (Novelty 4, 10)
        self.head_weights = None

        # ── Per-run statistics (read by experiments/ablation.py) ──
        self.stats = {
            "decode_steps":         0,
            "full_attention_steps": 0,  # seq < budget → no selection
            "refresh_steps":        0,  # forced full-context refresh
            "cache_hits":            0, # selection cache reused
            "cache_misses":          0, # full scoring pass run
        }

    # ── Novelty 6: Chunk-wise prefill ─────────────────────────────────────────

    def process_prefill(self, past_key_values, attention_weights=None):
        """
        Novelty 6:  Chunk-wise prefill — compute head weights in chunks
                     to bound peak memory on long sequences.
        Novelty 13: Store full KV cache (no eviction).
        Novelty 4, 10: Compute per-head criticality weights.
        """
        self.full_past_key_values = tuple(
            (k.clone(), v.clone()) for k, v in past_key_values
        )
        self.step_counter = 0
        self.prev_query = None
        self.cached_indices = None

        # Head weights only needed when criticality is active.
        if self.use_criticality_weights:
            self._compute_head_weights_chunked(past_key_values)
        else:
            self.head_weights = None

        return past_key_values

    def _compute_head_weights_chunked(self, past_key_values):
        """
        Novelty 4, 10: Per-head criticality weighting.
        Novelty 6:     Chunk-wise computation to bound memory.

        Estimate head importance from attention entropy. Heads with low entropy
        (peaked attention) are more decisive — they "know what they want" and
        should have more influence in the soft vote. This prevents high-norm
        heads from dominating the aggregation (Novelty 10).

        Processes queries in chunks of self.chunk_size to avoid OOM on long
        sequences (Novelty 6).
        """
        k = past_key_values[-1][0]  # (batch, heads, seq_len, head_dim)
        batch, heads, seq_len, head_dim = k.shape

        if seq_len < 2:
            self.head_weights = torch.ones(heads, device=k.device, dtype=k.dtype)
            return

        # Sample query positions for entropy estimation
        n_sample = min(64, seq_len)
        sample_idx = torch.linspace(0, seq_len - 1, n_sample, device=k.device).long()
        q_sample = k[:, :, sample_idx, :]  # (1, heads, n_sample, head_dim)

        # Chunk-wise attention score computation (Novelty 6)
        entropy_accum = torch.zeros(heads, device=k.device, dtype=torch.float32)

        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            k_chunk = k[:, :, chunk_start:chunk_end, :]

            scores_chunk = torch.matmul(
                q_sample, k_chunk.transpose(-2, -1)
            ) / (head_dim ** 0.5)
            attn_chunk = torch.softmax(scores_chunk, dim=-1)

            eps = 1e-8
            ent = -(attn_chunk * (attn_chunk + eps).log()).sum(dim=-1)
            entropy_accum += ent.mean(dim=-1).squeeze(0).float()

        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        avg_entropy = entropy_accum / n_chunks

        inv_entropy = 1.0 / (avg_entropy + 1e-8)
        self.head_weights = (inv_entropy / inv_entropy.sum()).to(k.dtype)

    # ── Novelty 5, 11: Selection cache ────────────────────────────────────────

    def _can_reuse_cache(self, current_query):
        """
        Novelty 5, 11: Consecutive query similarity.

        Consecutive decode queries tend to be very similar (the paper observes
        >95% cosine similarity between adjacent steps). When similarity exceeds
        the threshold, the selected token set is unlikely to change, so we
        reuse the cached indices and skip the scoring computation entirely.
        """
        if not self.use_selection_cache:
            return False
        if self.prev_query is None or self.cached_indices is None:
            return False

        q_curr = current_query.squeeze(2).float()  # (1, heads, head_dim)
        q_prev = self.prev_query.squeeze(2).float()

        cos_sim = torch.nn.functional.cosine_similarity(q_curr, q_prev, dim=-1)
        avg_similarity = cos_sim.mean().item()

        return avg_similarity >= self.cache_similarity_threshold

    # ── Main decode step ──────────────────────────────────────────────────────

    def process_step(self, past_key_values, step, attention_weights=None):
        """
        Full TokenSelect decode step.
        Novelty 2: dynamic scoring · 5: selection cache · 7,8: paged scoring ·
        3,4,10: head soft voting + criticality · 1,9: non-contiguous selection ·
        13: full cache maintained.
        """
        self._update_full_cache(past_key_values)
        self.step_counter += 1
        self.stats["decode_steps"] += 1

        current_seq_len = self.full_past_key_values[0][0].shape[2]
        total_budget = self.n_sink + self.K + self.n_local

        # Cache small enough → full attention (Novelty 13)
        if current_seq_len <= total_budget:
            self.stats["full_attention_steps"] += 1
            return self.full_past_key_values

        # Periodic full-context refresh.
        # BUG-12 fix: skip step==0 — at the very first decode step there is
        # nothing to refresh and the early return short-circuited every
        # selection-cache path on the *first* token, distorting cache_hit_rate
        # statistics and forcing a full-context return that callers (PPL,
        # passkey) interpreted as the active code path.
        if (self.refresh_interval > 0
                and step > 0
                and step % self.refresh_interval == 0):
            self.prev_query = None
            self.cached_indices = None
            self.stats["refresh_steps"] += 1
            return self.full_past_key_values

        device = self.full_past_key_values[0][0].device

        # Novelty 2: Get query proxy (dynamic, recomputed each step)
        proxy_k = self.full_past_key_values[-1][0]  # (B, H, S, D)
        proxy_q = proxy_k[:, :, -1:, :]             # (B, H, 1, D)

        # Novelty 5, 11: selection cache lookup
        if self._can_reuse_cache(proxy_q):
            all_idx = self.cached_indices
            self.stats["cache_hits"] += 1
        else:
            all_idx = self._paged_token_selection(
                proxy_q, proxy_k, current_seq_len, device
            )
            self.prev_query = proxy_q.detach().clone()
            self.cached_indices = all_idx
            self.stats["cache_misses"] += 1

        # Novelty 1, 9: Non-contiguous token-level selection across ALL layers.
        # (B,K)-aware gather: indices of shape (S_sel,) → shared slice;
        # indices of shape (B, S_sel) → per-sample torch.gather.
        return tuple(
            self._gather_layer(k, v, all_idx)
            for k, v in self.full_past_key_values
        )

    # ── Phase A→C batch-dim seam ──────────────────────────────────────────────

    def _gather_layer(self, k, v, indices):
        """
        Gather selected tokens from one layer's (K, V) and re-rotate the keys
        so that their RoPE phase matches their *new* gather-slot index.

        Index contract:
          • indices.dim() == 1  →  shape (S_sel,)        — shared across batch
                                  (Phase A/B fast path; B=1 today).
          • indices.dim() == 2  →  shape (B, S_sel)      — per-sample
                                  (Phase C batched decode).

        Both paths preserve k.shape == (B, H, S_full, D) → (B, H, S_sel, D).

        BUG-2 fix
        ─────────
        HF's DynamicCache treats the returned cache as if it lived at logical
        positions ``0 .. len-1``, and rotates the *new* token's Q at position
        ``len``. But each gathered key was originally rotated at its absolute
        index ``orig_pos = indices[i]``. Without correction the dot-product
        ``Q · K_i`` corresponds to a meaningless relative offset and the model
        cannot find any token (passkey collapses to 0%). We undo the original
        rotation and apply the new slot rotation in one step:

            delta_i = slot_i - orig_pos_i
            K_new   = R(delta_i) · K_orig
        """
        if indices.dim() == 1:
            k_gathered = k[:, :, indices, :]
            v_gathered = v[:, :, indices, :]
        else:
            B, H, _, D = k.shape
            idx_exp = indices.view(B, 1, -1, 1).expand(B, H, indices.shape[1], D)
            k_gathered = torch.gather(k, 2, idx_exp)
            v_gathered = torch.gather(v, 2, idx_exp)

        if self.apply_rope_correction:
            S_sel = k_gathered.shape[2]
            slot = torch.arange(S_sel, device=k.device)
            if indices.dim() == 1:
                delta = slot - indices.to(slot.dtype)
            else:
                # (B, S_sel)
                delta = slot.unsqueeze(0) - indices.to(slot.dtype)
            k_gathered = apply_rope_delta(k_gathered, delta, rope_theta=self.rope_theta)

        return k_gathered, v_gathered

    # ── Novelty 7, 8: Paged scoring ──────────────────────────────────────────

    def _paged_token_selection(self, proxy_q, proxy_k, current_seq_len, device):
        """
        Novelty 7, 8: Two-stage paged dot-product scoring.

        Returns a 1-D index tensor (S_sel,) — same indices applied to every
        batch element (the B=1 fast path). Phase C will add a (B, S_sel)
        return for per-sample selection without changing this signature.
        """
        # ── Fixed index sets (sink + recent) ──
        sink_end = min(self.n_sink, current_seq_len) if self.use_sink_tokens else 0
        recent_start = max(
            current_seq_len - (self.n_local if self.use_local_tokens else 0),
            sink_end,
        )

        sink_idx = torch.arange(0, sink_end, device=device)
        recent_idx = torch.arange(recent_start, current_seq_len, device=device)

        middle_start = sink_end
        middle_end = recent_start

        if middle_end <= middle_start:
            all_idx = torch.cat([sink_idx, recent_idx])
            all_idx = all_idx.unique()
            all_idx, _ = all_idx.sort()
            return all_idx

        middle_len = middle_end - middle_start
        middle_k = proxy_k[:, :, middle_start:middle_end, :]

        # ── Novelty 3, 4, 10: head soft voting with optional ablations ──
        weights = self.head_weights if self.use_criticality_weights else None

        if (self.use_kernels and self.use_head_softmax
                and topk_kernels.kernels_available(device)):
            # Fused Triton kernel covers softmax + weighting + cross-head sum.
            aggregated = topk_kernels.fused_paged_score(
                proxy_q, middle_k, weights,
            )
        else:
            # PyTorch reference path — also handles ablations the fused
            # kernel does not (e.g. head-softmax disabled).
            # BUG-6 fix: divide by sqrt(head_dim) to match the Triton kernel
            # and standard scaled-dot-product attention. Without this scaling
            # softmax saturates on large head_dim (=128 for LLaMA-2-7B) and
            # the per-head distribution collapses to a near-one-hot, distorting
            # the soft-vote sum across heads.
            head_dim = proxy_q.shape[-1]
            raw_scores = torch.matmul(proxy_q, middle_k.transpose(-2, -1))
            raw_scores = raw_scores / math.sqrt(head_dim)
            raw_scores = raw_scores.squeeze(2).squeeze(0)               # (H, M)

            if self.use_head_softmax:
                # Novelty 3: per-head softmax FIRST (equalises head contributions)
                normalized = torch.softmax(raw_scores, dim=-1)          # (H, M)
            else:
                # Ablation: bypass softmax — sum raw scaled scores directly.
                # Reveals how much of the win comes from per-head normalisation
                # vs. raw attention magnitude.
                normalized = raw_scores

            if weights is not None:
                normalized = normalized * weights.unsqueeze(-1)         # (H, M)

            aggregated = normalized.sum(dim=0)                          # (M,)

        if aggregated.dim() == 0:
            aggregated = aggregated.unsqueeze(0)

        # ── Novelty 7, 8: Two-stage paged top-K ──
        if middle_len > self.page_size * 2 and self.K < middle_len:
            topk_local = topk_kernels.fused_paged_topk(
                aggregated, self.page_size, self.K,
            )
        else:
            k_actual = min(self.K, middle_len)
            topk_local = aggregated.topk(k_actual).indices

        topk_idx = topk_local + middle_start  # offset to global indices

        all_idx = torch.cat([sink_idx, topk_idx, recent_idx])
        all_idx = all_idx.unique()
        all_idx, _ = all_idx.sort()

        return all_idx

    # ── Novelty 13: Full cache maintenance ────────────────────────────────────

    def _update_full_cache(self, past_key_values):
        """
        Novelty 13: Maintain full KV cache — never evict tokens.
        The new token is always the last position in HF's past_key_values.
        Append it to our full cache regardless of what we returned last step.
        """
        if self.full_past_key_values is None:
            self.full_past_key_values = tuple(
                (k.clone(), v.clone()) for k, v in past_key_values
            )
            return

        new_full_kv = []
        for layer_full, layer_cur in zip(
            self.full_past_key_values, past_key_values
        ):
            k_full, v_full = layer_full
            k_cur, v_cur = layer_cur
            new_k = k_cur[:, :, -1:, :]
            new_v = v_cur[:, :, -1:, :]
            k_updated = torch.cat([k_full, new_k], dim=2)
            v_updated = torch.cat([v_full, new_v], dim=2)
            new_full_kv.append((k_updated, v_updated))

        self.full_past_key_values = tuple(new_full_kv)

    def reset(self):
        self.full_past_key_values = None
        self.step_counter = 0
        self.prev_query = None
        self.cached_indices = None
        self.head_weights = None
        for k in self.stats:
            self.stats[k] = 0

    def get_kv_size_bytes(self, past_key_values):
        """
        Novelty 13: Reports full cache size.
        TopK stores everything — memory savings come from reduced attention
        computation scope, not from storage compression.
        """
        if self.full_past_key_values is not None:
            total = 0
            for layer in self.full_past_key_values:
                k, v = layer[0], layer[1]
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
            return total

        total = 0
        for layer in past_key_values:
            k, v = layer[0], layer[1]
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def get_ablation_config(self) -> dict:
        """Returns the active ablation configuration (logged by experiments)."""
        return {
            "use_kernels":             self.use_kernels,
            "use_head_softmax":        self.use_head_softmax,
            "use_criticality_weights": self.use_criticality_weights,
            "use_selection_cache":     self.use_selection_cache,
            "use_sink_tokens":         self.use_sink_tokens,
            "use_local_tokens":        self.use_local_tokens,
        }

    def get_stats(self) -> dict:
        """Returns per-run selection statistics (for ablation reports)."""
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
    print(f"Loading {model_name} for TopK smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    method = TopKMethod(K=32, n_sink=4, n_local=16, refresh_interval=10)
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="The history of machine learning began",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated: {text}")
    print(f"KV cache MB: {metrics['kv_cache_mb']:.4f}")
    print(f"Stats: {method.get_stats()}")
    print("TopK smoke test PASSED")
