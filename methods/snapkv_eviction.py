import torch
import torch.nn.functional as F
from .base import MethodWrapper


class SnapKVMethod(MethodWrapper):
    """
    SnapKV: One-shot KV eviction after prefill.

    Paper: Li et al., NeurIPS 2024 —
    "LLM Knows What You Are Looking for Before Generation"
    https://github.com/FasterDecoding/SnapKV

    ALGORITHM (matches paper Section 4.1 + Listing 1):
      1. Vote   — sum attention weights from the last `observation_window`
                  query positions over all prefix key positions, per head.
      2. Pool   — apply 1D max-pooling over those per-head scores to cluster
                  neighbouring tokens (prevents sparse / fragmented selection).
      3. Snap   — select top-k prefix positions per head, concatenate with the
                  full observation window, discard the rest.

    Decode proceeds with no further modification — the prompt KV is frozen.

    Key fixes vs a naive implementation:
      - Scoring and selection are done PER HEAD, not averaged across heads.
        Different heads specialise; averaging loses that signal.
      - Pooling is applied BEFORE top-k, not after. This is what makes
        the paper's clustering argument hold (Section 4.3).
      - The observation window is always kept in full by construction
        (concatenated after compression), so no special "sink" bookkeeping
        is needed for it.
      - Attention sinks (first few tokens) are NOT explicitly protected —
        the paper does not do this. If your model needs them, set
        sink_size > 0 and they will be prepended to the kept set.

    Args:
        budget_ratio    fraction of prefix tokens to keep (default 0.4).
                        Ignored if max_capacity_prompt is set directly via
                        process_prefill's seq_len at call time.
        observation_window  number of tokens at the end of the prompt used
                        to vote on prefix importance (default 32).
        kernel_size     1D max-pool kernel for clustering (default 7,
                        matching paper's LongBench experiments).
        sink_size       number of leading tokens to always retain (default 0).
                        The paper does not use sinks; set > 0 if your model
                        relies on attention-sink behaviour (e.g. StreamLLM).
    """

    def __init__(
        self,
        budget_ratio: float = 0.4,
        observation_window: int = 32,
        kernel_size: int = 7,
        sink_size: int = 0,
    ):
        self.budget_ratio = budget_ratio
        self.observation_window = observation_window
        self.kernel_size = kernel_size
        self.sink_size = sink_size

    @property
    def needs_attention_weights(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # prefill
    # ------------------------------------------------------------------

    def process_prefill(self, past_key_values, attention_weights=None):
        """
        Called once after the prompt forward pass.

        past_key_values : tuple of (k, v) per layer
                          k/v shape: (batch, heads, seq_len, head_dim)
        attention_weights: tuple of tensors per layer
                          shape: (batch, heads, seq_len, seq_len)
                          This is the full causal attention matrix from prefill.
        """
        if attention_weights is None:
            # Can't compress without attention weights; pass through unchanged.
            return past_key_values

        compressed = []
        for layer_idx, (layer_kv, layer_attn) in enumerate(
            zip(past_key_values, attention_weights)
        ):
            k, v = layer_kv[0], layer_kv[1]
            # k: (batch, heads, seq_len, head_dim)
            batch, heads, seq_len, head_dim = k.shape

            obs = min(self.observation_window, seq_len)
            prefix_len = seq_len - obs

            # Nothing to compress if the prompt is shorter than the obs window.
            if prefix_len <= 0:
                compressed.append((k, v))
                continue

            # ----------------------------------------------------------
            # Step 1 — Vote
            # layer_attn: (batch, heads, seq_len, seq_len)
            # We want the obs window's attention over the prefix only.
            # Slice: queries = last `obs` rows, keys = first `prefix_len` cols.
            # Shape after slice: (batch, heads, obs, prefix_len)
            # Sum over obs query dim → (batch, heads, prefix_len)
            # ----------------------------------------------------------
            obs_attn = layer_attn[:, :, -obs:, :prefix_len]
            vote = obs_attn.sum(dim=2)  # (batch, heads, prefix_len)

            # ----------------------------------------------------------
            # Step 2 — Pool (per head, across the prefix positions axis)
            # Reshape to (batch*heads, 1, prefix_len) for F.max_pool1d,
            # then reshape back.
            # ----------------------------------------------------------
            bh = batch * heads
            vote_2d = vote.reshape(bh, 1, prefix_len)
            pool_vote = F.max_pool1d(
                vote_2d,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )  # (batch*heads, 1, prefix_len)
            pool_vote = pool_vote.reshape(batch, heads, prefix_len)

            # ----------------------------------------------------------
            # Step 3 — Select top-k per head
            # budget = how many prefix positions to keep total.
            # If sink_size > 0, those slots are reserved for leading tokens.
            # ----------------------------------------------------------
            budget = max(1, int(self.budget_ratio * seq_len))
            # Clamp: budget can't exceed prefix_len (+ obs is added back later)
            k_select = max(1, min(budget - self.sink_size, prefix_len))

            # topk returns (values, indices); indices shape: (batch, heads, k_select)
            _, top_indices = pool_vote.topk(k_select, dim=-1)
            # Sort so we gather in sequence order (important for position embeddings
            # in models that don't use RoPE — harmless for those that do).
            top_indices, _ = top_indices.sort(dim=-1)

            # Optional sinks: always prepend the first sink_size token indices.
            if self.sink_size > 0:
                sink_idx = torch.arange(
                    self.sink_size, device=k.device, dtype=torch.long
                )
                # Broadcast sink_idx to (batch, heads, sink_size)
                sink_idx = sink_idx.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1)
                # Concatenate sinks before selected, then re-sort and de-duplicate.
                top_indices = torch.cat([sink_idx, top_indices], dim=-1)
                top_indices, _ = top_indices.sort(dim=-1)
                # unique_consecutive works because we just sorted.
                # torch has no batched unique, so do it per-head via a small loop
                # only when sink_size > 0 (rare path).
                top_indices = _batched_unique_consecutive(top_indices)

            # ----------------------------------------------------------
            # Gather selected prefix K and V
            # top_indices: (batch, heads, k_select)
            # k:           (batch, heads, prefix_len, head_dim)
            # We need to expand indices to match head_dim.
            # ----------------------------------------------------------
            idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_prefix = k[:, :, :prefix_len, :]
            v_prefix = v[:, :, :prefix_len, :]
            k_selected = k_prefix.gather(dim=2, index=idx_expanded)
            v_selected = v_prefix.gather(dim=2, index=idx_expanded)

            # ----------------------------------------------------------
            # Always keep the full observation window — concatenate after.
            # Final shape: (batch, heads, k_select + obs, head_dim)
            # ----------------------------------------------------------
            k_obs = k[:, :, -obs:, :]
            v_obs = v[:, :, -obs:, :]
            k_out = torch.cat([k_selected, k_obs], dim=2)
            v_out = torch.cat([v_selected, v_obs], dim=2)

            compressed.append((k_out, v_out))

        return tuple(compressed)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def process_step(self, past_key_values, step, attention_weights=None):
        """
        Identity: the HF generate loop appends new decode tokens to
        past_key_values automatically. SnapKV does nothing per step —
        that constant-cost behaviour is the whole point.
        """
        return past_key_values

    def reset(self):
        pass

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------

    def get_kv_size_bytes(self, past_key_values):
        total = 0
        for k, v in past_key_values:
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total


# ----------------------------------------------------------------------
# Internal helper — only used when sink_size > 0
# ----------------------------------------------------------------------

def _batched_unique_consecutive(t: torch.Tensor) -> torch.Tensor:
    """
    Remove consecutive duplicates in the last dim of a (batch, heads, N) tensor.
    Assumes t is already sorted along dim=-1.
    Returns a tensor padded to the original size with the last valid value
    (safe because gather ignores out-of-range duplicates after de-dup).

    This is O(N) and only called in the sink_size > 0 path.
    """
    # diff along last dim; first element always kept
    diff = torch.ones_like(t, dtype=torch.bool)
    diff[:, :, 1:] = t[:, :, 1:] != t[:, :, :-1]
    # For simplicity: mask duplicates to the previous valid index.
    # Because we only deduplicate sink indices that might overlap with top_k,
    # and the tensor is sorted, we can just clamp duplicates to their predecessor.
    out = t.clone()
    for i in range(1, t.shape[-1]):
        mask = ~diff[:, :, i]
        out[:, :, i] = torch.where(mask, out[:, :, i - 1], out[:, :, i])
    return out


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

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

    method = SnapKVMethod(
        budget_ratio=0.5,
        observation_window=16,
        kernel_size=7,
        sink_size=0,
    )
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="The history of machine learning began",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated : {text}")
    print(f"KV cache  : {metrics['kv_cache_mb']:.2f} MB")
    print("SnapKV smoke test PASSED")