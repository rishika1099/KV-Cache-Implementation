import torch
from .base import MethodWrapper


class TopKMethod(MethodWrapper):
    """
    TokenSelect: Dynamic Token-Level KV Cache Selection (Wu et al., EMNLP 2025).
    Reference: https://arxiv.org/abs/2411.02886
    Official repo: https://github.com/pzs19/TokenSelect

    Pure PyTorch implementation of the core selection algorithm:
      - Per-layer independent token selection
      - Soft voting: softmax per head, sum across heads (Eq. 7)
      - Flat top-K with optional max_pool1d smoothing
      - Per-layer cache reuse via cosine similarity of query fingerprints

    Architecture note (wrapper approach):
      This operates as a past_key_values wrapper, not an attention-module hook.
      To fix RoPE position misalignment when returning sliced caches, the
      method exposes `true_seq_length` which the runner uses to pass correct
      `position_ids` to the model forward. This ensures new Q/K tokens get
      RoPE applied at the true sequence position, not the sliced-cache length.

      Remaining limitation vs. the official implementation:
      Q proxy: uses each layer's latest K vector as a proxy for Q, since
      real Q tensors are not accessible at the wrapper level. The official
      implementation uses the actual Q tensor averaged across seq dim.
    """

    def __init__(self, K=2048, n_sink=128, n_local=512,
                 cosine_threshold=0.9, kernel_size=-1):
        self.K = K
        self.n_sink = n_sink
        self.n_local = n_local
        self.cosine_threshold = cosine_threshold  # paper: 0.9
        self.kernel_size = kernel_size             # -1 = disabled (paper default)

        self.full_past_key_values = None
        self.step_counter = 0

        # Per-layer selection cache (paper: query_fingerprints_cache per layer)
        self.prev_queries = None    # list[Tensor | None], one per layer
        self.cached_indices = None  # list[Tensor | None], one per layer

        # RoPE fix: runner reads this to pass correct position_ids.
        # Set to the true full cache length when returning a sliced cache,
        # None when returning full cache (no correction needed).
        self.true_seq_length = None

    # ── Prefill ───────────────────────────────────────────────────────────────

    def process_prefill(self, past_key_values, attention_weights=None):
        """Store full KV cache. No selection during prefill."""
        self.full_past_key_values = tuple(
            (k.clone(), v.clone()) for k, v in past_key_values
        )
        self.step_counter = 0
        n_layers = len(past_key_values)
        self.prev_queries = [None] * n_layers
        self.cached_indices = [None] * n_layers
        return past_key_values

    # ── Decode step ───────────────────────────────────────────────────────────

    def process_step(self, past_key_values, step, attention_weights=None):
        """
        Per-layer independent token selection (paper Section 3.2).
        Each layer scores its own K cache and selects its own top-K.
        """
        self._update_full_cache(past_key_values)
        self.step_counter += 1

        current_seq_len = self.full_past_key_values[0][0].shape[2]
        total_budget = self.n_sink + self.K + self.n_local

        # Short-circuit: full attention if cache fits in budget
        if current_seq_len <= total_budget:
            self.true_seq_length = None  # no slicing → no position correction
            return self.full_past_key_values

        device = self.full_past_key_values[0][0].device

        # Per-layer selection
        result = []
        for layer_idx, (k_full, v_full) in enumerate(self.full_past_key_values):
            # Q proxy: this layer's latest K vector (wrapper limitation)
            proxy_q = k_full[:, :, -1:, :]  # (batch, kv_heads, 1, head_dim)

            # Per-layer cache reuse (paper: cosine similarity check)
            if self._can_reuse_layer_cache(layer_idx, proxy_q):
                idx = self.cached_indices[layer_idx]
            else:
                idx = self._select_tokens(
                    proxy_q, k_full, current_seq_len, device
                )
                self.prev_queries[layer_idx] = proxy_q.squeeze(2).detach().clone()
                self.cached_indices[layer_idx] = idx

            result.append((k_full[:, :, idx, :], v_full[:, :, idx, :]))

        # RoPE fix: tell runner the true position for the next token
        self.true_seq_length = current_seq_len
        return tuple(result)

    # ── Per-layer cache reuse ─────────────────────────────────────────────────

    def _can_reuse_layer_cache(self, layer_idx, current_query):
        """
        Paper: compare query fingerprints between consecutive decode steps.
        If cosine similarity > threshold, reuse cached indices (skip selection).
        Tracked independently per layer.
        """
        if self.prev_queries[layer_idx] is None or self.cached_indices[layer_idx] is None:
            return False

        q_curr = current_query.squeeze(2).float()   # (batch, heads, head_dim)
        q_prev = self.prev_queries[layer_idx].float()

        cos_sim = torch.nn.functional.cosine_similarity(q_curr, q_prev, dim=-1)
        return cos_sim.mean().item() >= self.cosine_threshold

    # ── Token selection (per-layer) ───────────────────────────────────────────

    def _select_tokens(self, proxy_q, k_full, current_seq_len, device):
        """
        Paper Algorithm 1: token-level selection with soft voting.

        1. Score middle tokens: Q · K^T / sqrt(d)
        2. Softmax per head (equalizes head contributions)
        3. Sum across heads (soft voting)
        4. Optional max_pool1d smoothing
        5. Flat top-K
        6. Return: [sink] ∪ [top-K from middle] ∪ [local]
        """
        head_dim = proxy_q.shape[-1]

        # Fixed regions
        sink_end = min(self.n_sink, current_seq_len)
        recent_start = max(current_seq_len - self.n_local, sink_end)

        sink_idx = torch.arange(0, sink_end, device=device)
        recent_idx = torch.arange(recent_start, current_seq_len, device=device)

        middle_start = sink_end
        middle_end = recent_start

        # No middle tokens → just sink + local
        if middle_end <= middle_start:
            all_idx = torch.cat([sink_idx, recent_idx])
            all_idx, _ = all_idx.sort()
            return all_idx

        middle_len = middle_end - middle_start
        middle_k = k_full[:, :, middle_start:middle_end, :]

        # Score: Q · K^T / sqrt(d)
        raw_scores = torch.matmul(
            proxy_q, middle_k.transpose(-2, -1)
        ) / (head_dim ** 0.5)
        # (batch, heads, 1, middle_len) → (batch, heads, middle_len)
        raw_scores = raw_scores.squeeze(2)

        # Soft voting (paper Eq. 7): softmax per head, then sum across heads
        scores = torch.softmax(raw_scores, dim=-1)   # (batch, heads, middle_len)
        aggregated = scores.sum(dim=1).squeeze(0)     # (middle_len,)

        if aggregated.dim() == 0:
            aggregated = aggregated.unsqueeze(0)

        # Optional max_pool1d smoothing (paper: KERNEL_SIZE parameter)
        if self.kernel_size > 0 and middle_len > self.kernel_size:
            aggregated = torch.nn.functional.max_pool1d(
                aggregated.unsqueeze(0).unsqueeze(0),
                kernel_size=self.kernel_size,
                stride=1,
                padding=(self.kernel_size - 1) // 2,
            ).squeeze(0).squeeze(0)

        # Flat top-K (no paged two-stage)
        k_actual = min(self.K, middle_len)
        topk_local = aggregated.topk(k_actual).indices
        topk_idx = topk_local + middle_start

        # Concatenate: sink ∪ topk ∪ recent (non-overlapping ranges → sort only)
        all_idx = torch.cat([sink_idx, topk_idx, recent_idx])
        all_idx, _ = all_idx.sort()
        return all_idx

    # ── Full cache maintenance ────────────────────────────────────────────────

    def _update_full_cache(self, past_key_values):
        """
        Append the newest token (last position) from each layer's current
        KV cache to our full cache. The full cache is never pruned.
        """
        if self.full_past_key_values is None:
            self.full_past_key_values = tuple(
                (k.clone(), v.clone()) for k, v in past_key_values
            )
            return

        new_full_kv = []
        for (k_full, v_full), (k_cur, v_cur) in zip(
            self.full_past_key_values, past_key_values
        ):
            k_updated = torch.cat([k_full, k_cur[:, :, -1:, :]], dim=2)
            v_updated = torch.cat([v_full, v_cur[:, :, -1:, :]], dim=2)
            new_full_kv.append((k_updated, v_updated))
        self.full_past_key_values = tuple(new_full_kv)

    # ── Reset / metrics ───────────────────────────────────────────────────────

    def reset(self):
        self.full_past_key_values = None
        self.step_counter = 0
        self.prev_queries = None
        self.cached_indices = None
        self.true_seq_length = None

    def get_kv_size_bytes(self, past_key_values):
        """Reports full cache size (no compression, savings via attention scope)."""
        if self.full_past_key_values is not None:
            return sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for k, v in self.full_past_key_values
            )
        return sum(
            k.numel() * k.element_size() + v.numel() * v.element_size()
            for k, v in past_key_values
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from benchmark.runner import generate_with_method

    # OPT-125M for quick smoke test. RoPE models (Llama, Qwen) also work
    # correctly — the runner passes true position_ids via method.true_seq_length.
    model_name = "facebook/opt-125m"
    print(f"Loading {model_name} for TopK smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    method = TopKMethod(K=32, n_sink=4, n_local=16)
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="The history of machine learning began",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated: {text}")
    print(f"KV cache MB: {metrics['kv_cache_mb']:.4f}")
    print("TopK smoke test PASSED")
