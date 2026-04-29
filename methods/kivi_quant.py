import torch
from .base import MethodWrapper


def quantize_per_channel(tensor, bits):
    """
    Quantize along head_dim (channel) dimension.
    tensor shape: (batch, heads, seq_len, head_dim)
    Each (batch, head, token) gets its own scale/zero across channels.
    """
    min_val = tensor.min(dim=-1, keepdim=True).values
    max_val = tensor.max(dim=-1, keepdim=True).values
    scale = (max_val - min_val) / (2 ** bits - 1)
    scale = scale.clamp(min=1e-8)
    zero = min_val
    quantized = ((tensor - zero) / scale).round().clamp(0, 2 ** bits - 1)
    return (
        quantized.to(torch.uint8),
        scale.to(torch.float16),
        zero.to(torch.float16),
    )


def quantize_per_token(tensor, bits):
    """
    Quantize along seq_len (token) dimension.
    tensor shape: (batch, heads, seq_len, head_dim)
    Each (batch, head, channel) gets its own scale/zero across tokens.
    """
    min_val = tensor.min(dim=-2, keepdim=True).values
    max_val = tensor.max(dim=-2, keepdim=True).values
    scale = (max_val - min_val) / (2 ** bits - 1)
    scale = scale.clamp(min=1e-8)
    zero = min_val
    quantized = ((tensor - zero) / scale).round().clamp(0, 2 ** bits - 1)
    return (
        quantized.to(torch.uint8),
        scale.to(torch.float16),
        zero.to(torch.float16),
    )


def dequantize(quantized, scale, zero):
    return quantized.to(torch.float16) * scale + zero


class KIVIMethod(MethodWrapper):
    """
    KIVI: Asymmetric KV cache quantization with block-wise group quantization.

    Implements all three core concepts from Liu et al. (ICML 2024):
    "A Tuning-Free Asymmetric 2bit Quantization for KV Cache"

    1. Asymmetric quantization strategies:
       - Keys: per-channel (dim=-1, head_dim axis) — captures channel outliers
       - Values: per-token (dim=-2, seq_len axis) — captures token outliers

    2. Residual window (hybrid precision):
       - Most recent `residual_length` tokens kept in FP16
       - Historical tokens quantized to `bits`-bit
       - Oldest residual token evicted to quantized cache when window overflows

    3. Block-wise / group-wise quantization (NEW):
       - Historical tokens organized into blocks of `group_size` tokens
       - Each block has independent scale and zero-point parameters
       - When tokens overflow from residual, they accumulate in an FP16 buffer
       - When buffer reaches `group_size`, it is quantized as a NEW block
       - Existing blocks are NEVER dequantized and re-quantized
       - O(1) per-token amortized cost instead of O(N) re-quantization
       - Better accuracy: scale parameters aren't skewed by distant tokens

    Pure PyTorch — no Triton/CUDA kernels.
    """

    def __init__(self, bits=4, residual_length=128, group_size=32):
        self.bits = bits
        self.residual_length = residual_length
        self.group_size = group_size
        self.cache = {}  # layer_idx -> dict with block-wise state

    # ── Block-wise quantization helpers ───────────────────────────────────────

    def _quantize_blocks_per_channel(self, tensor):
        """
        Split tensor into blocks of group_size along seq_len,
        quantize each block independently with per-channel parameters.
        Returns list of (quantized, scale, zero) tuples.
        """
        seq_len = tensor.shape[2]
        blocks = []
        for start in range(0, seq_len, self.group_size):
            end = min(start + self.group_size, seq_len)
            block = tensor[:, :, start:end, :]
            q, s, z = quantize_per_channel(block, self.bits)
            blocks.append((q, s, z))
        return blocks

    def _quantize_blocks_per_token(self, tensor):
        """
        Split tensor into blocks of group_size along seq_len,
        quantize each block independently with per-token parameters.
        Returns list of (quantized, scale, zero) tuples.
        """
        seq_len = tensor.shape[2]
        blocks = []
        for start in range(0, seq_len, self.group_size):
            end = min(start + self.group_size, seq_len)
            block = tensor[:, :, start:end, :]
            q, s, z = quantize_per_token(block, self.bits)
            blocks.append((q, s, z))
        return blocks

    def _dequantize_blocks(self, blocks):
        """Dequantize all blocks and concatenate along seq_len."""
        if not blocks:
            return None
        pieces = [dequantize(q, s, z) for q, s, z in blocks]
        return torch.cat(pieces, dim=2)

    # ── Prefill ───────────────────────────────────────────────────────────────

    def process_prefill(self, past_key_values, attention_weights=None):
        self.cache = {}
        result = []

        for layer_idx, layer_kv in enumerate(past_key_values):
            k, v = layer_kv[0].half(), layer_kv[1].half()
            seq_len = k.shape[2]

            if seq_len <= self.residual_length:
                # Entire sequence fits in residual — keep as FP16
                res_k = k.to(torch.float16)
                res_v = v.to(torch.float16)
                self.cache[layer_idx] = {
                    'k_blocks': [],
                    'v_blocks': [],
                    'overflow_k': torch.empty(
                        (k.shape[0], k.shape[1], 0, k.shape[3]),
                        dtype=torch.float16, device=k.device,
                    ),
                    'overflow_v': torch.empty(
                        (v.shape[0], v.shape[1], 0, v.shape[3]),
                        dtype=torch.float16, device=v.device,
                    ),
                    'residual_k': res_k,
                    'residual_v': res_v,
                }
                result.append((res_k, res_v))
                continue

            # Split into historical and residual
            hist_k = k[:, :, :-self.residual_length, :]
            hist_v = v[:, :, :-self.residual_length, :]
            res_k = k[:, :, -self.residual_length:, :].to(torch.float16)
            res_v = v[:, :, -self.residual_length:, :].to(torch.float16)

            # Block-wise quantization of historical tokens
            k_blocks = self._quantize_blocks_per_channel(hist_k)
            v_blocks = self._quantize_blocks_per_token(hist_v)

            self.cache[layer_idx] = {
                'k_blocks': k_blocks,
                'v_blocks': v_blocks,
                'overflow_k': torch.empty(
                    (k.shape[0], k.shape[1], 0, k.shape[3]),
                    dtype=torch.float16, device=k.device,
                ),
                'overflow_v': torch.empty(
                    (v.shape[0], v.shape[1], 0, v.shape[3]),
                    dtype=torch.float16, device=v.device,
                ),
                'residual_k': res_k,
                'residual_v': res_v,
            }

            # Reconstruct for the model
            deq_k = self._dequantize_blocks(k_blocks)
            deq_v = self._dequantize_blocks(v_blocks)
            reconstructed_k = torch.cat([deq_k, res_k], dim=2)
            reconstructed_v = torch.cat([deq_v, res_v], dim=2)
            result.append((reconstructed_k, reconstructed_v))

        return tuple(result)

    # ── Decode step ───────────────────────────────────────────────────────────

    def process_step(self, past_key_values, step, attention_weights=None):
        result = []

        for layer_idx, layer_kv in enumerate(past_key_values):
            k, v = layer_kv[0], layer_kv[1]

            if layer_idx not in self.cache:
                result.append((k, v))
                continue

            state = self.cache[layer_idx]
            new_k = k[:, :, -1:, :].to(torch.float16)
            new_v = v[:, :, -1:, :].to(torch.float16)

            # Append new token to residual
            res_k = torch.cat([state['residual_k'], new_k], dim=2)
            res_v = torch.cat([state['residual_v'], new_v], dim=2)

            # If residual exceeds limit, evict oldest token to overflow buffer
            if res_k.shape[2] > self.residual_length:
                evict_k = res_k[:, :, :1, :]
                evict_v = res_v[:, :, :1, :]
                res_k = res_k[:, :, 1:, :]
                res_v = res_v[:, :, 1:, :]

                # Append evicted token to overflow buffer
                state['overflow_k'] = torch.cat(
                    [state['overflow_k'], evict_k], dim=2
                )
                state['overflow_v'] = torch.cat(
                    [state['overflow_v'], evict_v], dim=2
                )

                # When overflow reaches group_size, quantize as a new block
                # Existing blocks are NEVER re-quantized (O(1) amortized)
                if state['overflow_k'].shape[2] >= self.group_size:
                    block_k = state['overflow_k'][:, :, :self.group_size, :]
                    block_v = state['overflow_v'][:, :, :self.group_size, :]

                    q_k, s_k, z_k = quantize_per_channel(block_k, self.bits)
                    q_v, s_v, z_v = quantize_per_token(block_v, self.bits)

                    state['k_blocks'].append((q_k, s_k, z_k))
                    state['v_blocks'].append((q_v, s_v, z_v))

                    # Keep any remaining overflow beyond group_size
                    state['overflow_k'] = state['overflow_k'][:, :, self.group_size:, :]
                    state['overflow_v'] = state['overflow_v'][:, :, self.group_size:, :]

            state['residual_k'] = res_k
            state['residual_v'] = res_v

            # Reconstruct full KV: quantized blocks + overflow (FP16) + residual
            parts_k = []
            parts_v = []

            if state['k_blocks']:
                deq_k = self._dequantize_blocks(state['k_blocks'])
                deq_v = self._dequantize_blocks(state['v_blocks'])
                parts_k.append(deq_k)
                parts_v.append(deq_v)

            if state['overflow_k'].shape[2] > 0:
                parts_k.append(state['overflow_k'])
                parts_v.append(state['overflow_v'])

            parts_k.append(res_k)
            parts_v.append(res_v)

            full_k = torch.cat(parts_k, dim=2)
            full_v = torch.cat(parts_v, dim=2)
            result.append((full_k, full_v))

        return tuple(result)

    def reset(self):
        self.cache = {}

    def get_kv_size_bytes(self, past_key_values):
        """
        Count storage:
        - Quantized blocks: uint8 data at actual bit-width + FP16 scale/zero per block
        - Overflow buffer: FP16 (not yet quantized)
        - Residual: FP16
        """
        total = 0
        for layer_idx, state in self.cache.items():
            # Quantized blocks
            for q, s, z in state['k_blocks']:
                total += int(q.numel() * self.bits / 8)  # actual bit-width
                total += s.numel() * s.element_size()
                total += z.numel() * z.element_size()
            for q, s, z in state['v_blocks']:
                total += int(q.numel() * self.bits / 8)
                total += s.numel() * s.element_size()
                total += z.numel() * z.element_size()

            # Overflow buffer (FP16, not yet quantized)
            total += state['overflow_k'].numel() * state['overflow_k'].element_size()
            total += state['overflow_v'].numel() * state['overflow_v'].element_size()

            # Residual (FP16)
            total += state['residual_k'].numel() * state['residual_k'].element_size()
            total += state['residual_v'].numel() * state['residual_v'].element_size()

        # Fallback if cache is empty
        if not self.cache:
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
    print(f"Loading {model_name} for KIVI smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    method = KIVIMethod(bits=4, residual_length=16, group_size=32)
    text, metrics = generate_with_method(
        model, tokenizer, method,
        prompt="The history of machine learning began",
        max_new_tokens=10,
        device="cuda",
    )
    print(f"Generated: {text}")
    print(f"KV cache MB: {metrics['kv_cache_mb']:.4f}")
    print("KIVI smoke test PASSED")
