import torch
from .base import MethodWrapper


class SnapKVMethod(MethodWrapper):
    """
    SnapKV: One-shot KV eviction after prefill.

    Inspired by: SnapKV (Li et al., NeurIPS 2024) —
    "LLM Knows What You Are Looking for Before Generation"

    ALGORITHM:
      After prefill, use attention patterns from the last `observation_window`
      prompt tokens to score all earlier token positions. Evict low-importance
      tokens once. Decode proceeds with no further modification.

    NOTE: Requires output_attentions=True during prefill. The runner already
    passes this flag for all methods.
    """

    def __init__(self, budget_ratio=0.4, sink_size=4, observation_window=32):
        self.budget_ratio = budget_ratio
        self.sink_size = sink_size
        self.observation_window = observation_window

    @property
    def needs_attention_weights(self) -> bool:
        return True

    def process_prefill(self, past_key_values, attention_weights=None):
        if attention_weights is None:
            # Fallback: no eviction if attention weights unavailable
            return past_key_values

        retained_kv = []
        for layer_idx, (layer_kv, layer_attn) in enumerate(
            zip(past_key_values, attention_weights)
        ):
            k, v = layer_kv[0], layer_kv[1]
            seq_len = k.shape[2]

            # layer_attn shape: (batch, heads, query_len, key_len)
            # Use last observation_window query positions
            obs_window = min(self.observation_window, seq_len)
            obs_attn = layer_attn[:, :, -obs_window:, :]  # (1, heads, obs, seq_len)

            # Importance per token: mean over heads and observation queries
            importance = obs_attn.mean(dim=2).mean(dim=1).squeeze(0)  # (seq_len,)

            # Compute budget
            budget = max(
                self.sink_size + obs_window,
                int(self.budget_ratio * seq_len)
            )
            K = max(0, budget - self.sink_size - obs_window)

            # Sink indices: always retain first sink_size positions
            sink_indices = list(range(self.sink_size))

            # Recent indices: always retain last observation_window positions
            recent_start = max(self.sink_size, seq_len - obs_window)
            recent_indices = list(range(recent_start, seq_len))

            # Middle positions: everything not in sinks or recent window
            protected = set(sink_indices) | set(recent_indices)
            middle_positions = [i for i in range(seq_len) if i not in protected]

            if K > 0 and middle_positions:
                middle_importance = importance[middle_positions]
                k_select = min(K, len(middle_positions))
                top_k_local = middle_importance.topk(k_select).indices
                top_k_indices = [middle_positions[i] for i in top_k_local.tolist()]
            else:
                top_k_indices = []

            # Union and sort retained indices
            retained = sorted(set(sink_indices) | set(top_k_indices) | set(recent_indices))
            retained_tensor = torch.tensor(retained, device=k.device, dtype=torch.long)

            k_retained = k[:, :, retained_tensor, :]
            v_retained = v[:, :, retained_tensor, :]
            retained_kv.append((k_retained, v_retained))

        return tuple(retained_kv)

    def process_step(self, past_key_values, step, attention_weights=None):
        # Identity: new tokens appended by HF decode loop are always kept.
        # No per-step logic needed. This is SnapKV's key advantage.
        return past_key_values

    def reset(self):
        pass

    def get_kv_size_bytes(self, past_key_values):
        """Count bytes of retained K and V tensors only."""
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
    print(f"Loading {model_name} for SnapKV smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    method = SnapKVMethod(budget_ratio=0.5, sink_size=4, observation_window=16)
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="The history of machine learning began",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated: {text}")
    print(f"KV cache MB: {metrics['kv_cache_mb']:.2f}")
    print("SnapKV smoke test PASSED")
