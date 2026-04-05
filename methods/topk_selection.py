import torch
from .base import MethodWrapper


class TopKMethod(MethodWrapper):
    """
    Top-K token selection for KV cache attention.

    Inspired by: TokenSelect (Wu et al., EMNLP 2025) —
    "Efficient Long-Context Inference via Dynamic Token-Level KV Cache Selection"
    Simplified to: pure PyTorch, no custom kernels.

    CORE IDEA:
    At each decode step, score all KV positions using approximate Q·K dot products.
    Only return top-K positions to attention. Masked tokens remain stored in the
    full cache — they can be reactivated at refresh steps.
    Nothing is permanently deleted.

    IMPORTANT: TopK stores the FULL cache at all times.
    compression_ratio ≈ 1.0 by design.
    This method trades memory savings for quality preservation.
    """

    def __init__(self, K=512, refresh_interval=50):
        self.K = K
        self.refresh_interval = refresh_interval
        self.full_past_key_values = None
        self.step_counter = 0

    def process_prefill(self, past_key_values, attention_weights=None):
        # Store the full KV cache — return unchanged (full attention during prefill)
        self.full_past_key_values = tuple(
            (k.clone(), v.clone()) for k, v in past_key_values
        )
        self.step_counter = 0
        return past_key_values

    def process_step(self, past_key_values, step, attention_weights=None):
        # Update full cache with newly appended tokens from HF
        # past_key_values contains the cache after HF appended the new token
        self._update_full_cache(past_key_values)
        self.step_counter += 1

        current_seq_len = self.full_past_key_values[0][0].shape[2]

        # If cache is small enough, use full attention
        if current_seq_len <= self.K:
            return self.full_past_key_values

        # Periodic full-context refresh
        if self.refresh_interval > 0 and step % self.refresh_interval == 0:
            return self.full_past_key_values

        # Score token positions using last layer's K as a proxy
        proxy_k = self.full_past_key_values[-1][0]  # (1, heads, seq_len, head_dim)
        proxy_q = proxy_k[:, :, -1:, :]             # (1, heads, 1, head_dim)

        scores = torch.matmul(proxy_q, proxy_k.transpose(-2, -1))  # (1, heads, 1, seq_len)
        scores = scores.mean(dim=1).squeeze()  # (seq_len,)

        # Ensure scores is 1D even for seq_len=1
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        # Always keep last 16 tokens regardless of score (recency bias)
        recent_mask = torch.zeros(current_seq_len, device=scores.device, dtype=torch.bool)
        recent_mask[-min(16, current_seq_len):] = True

        # Select top-K indices
        k_select = min(self.K, current_seq_len)
        topk_indices = scores.topk(k_select).indices
        selected = torch.zeros(current_seq_len, device=scores.device, dtype=torch.bool)
        selected[topk_indices] = True
        selected[recent_mask] = True
        selected_indices = selected.nonzero(as_tuple=True)[0].sort().values

        # Slice ALL layers to selected indices
        selected_kv = tuple(
            (k[:, :, selected_indices, :], v[:, :, selected_indices, :])
            for k, v in self.full_past_key_values
        )
        return selected_kv

    def _update_full_cache(self, past_key_values):
        """
        Sync full_past_key_values with the latest past_key_values from HF.
        HF appends the new token's KV to whatever we returned last step.
        We need to capture that new token and add it to our full cache.

        Strategy: the new token is always the LAST position. Since we may have
        returned a sliced cache last step, we identify new tokens by comparing
        lengths and always take the last token from past_key_values.
        """
        if self.full_past_key_values is None:
            self.full_past_key_values = tuple(
                (k.clone(), v.clone()) for k, v in past_key_values
            )
            return

        full_len = self.full_past_key_values[0][0].shape[2]
        current_len = past_key_values[0][0].shape[2]

        # The new token is always the last position in past_key_values
        # (regardless of whether we returned a sliced or full cache)
        new_full_kv = []
        for layer_idx, (layer_full, layer_cur) in enumerate(
            zip(self.full_past_key_values, past_key_values)
        ):
            k_full, v_full = layer_full
            k_cur, v_cur = layer_cur

            # Extract the newest token (last position in current kv)
            new_k = k_cur[:, :, -1:, :]
            new_v = v_cur[:, :, -1:, :]

            # Append to full cache
            k_updated = torch.cat([k_full, new_k], dim=2)
            v_updated = torch.cat([v_full, new_v], dim=2)
            new_full_kv.append((k_updated, v_updated))

        self.full_past_key_values = tuple(new_full_kv)

    def reset(self):
        self.full_past_key_values = None
        self.step_counter = 0

    def get_kv_size_bytes(self, past_key_values):
        """
        Always return full cache size. TopK stores everything — memory savings are ~zero.
        This is an important benchmark result: TopK trades memory for quality preservation.
        """
        if self.full_past_key_values is not None:
            total = 0
            for layer in self.full_past_key_values:
                k, v = layer[0], layer[1]
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
            return total

        # Fallback to passed past_key_values
        total = 0
        for layer in past_key_values:
            k, v = layer[0], layer[1]
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total


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

    method = TopKMethod(K=32, refresh_interval=10)
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="The history of machine learning began",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated: {text}")
    print(f"KV cache MB: {metrics['kv_cache_mb']:.4f}")
    print("TopK smoke test PASSED")
