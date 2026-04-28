import torch
from .base import MethodWrapper


def quantize_per_channel(tensor, bits):
    """
    Per-channel quantization for Keys.
    tensor shape: (batch, heads, seq_len, head_dim)
    Scale is computed along seq_len (dim=-2) so each head_dim channel gets
    its own scale/zero across all tokens — matches KIVI paper key quantization.
    Scale shape: (batch, heads, 1, head_dim)
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


def quantize_per_token(tensor, bits, group_size):
    """
    Per-token quantization for Values, with head_dim grouped.
    tensor shape: (batch, heads, seq_len, head_dim)
    head_dim is split into chunks of group_size; each (token, head_dim_group)
    pair gets its own scale/zero — matches reference KIVI value quantization.
    Returns:
      quantized: uint8, (batch, heads, seq_len, head_dim)
      scale:     fp16,  (batch, heads, seq_len, head_dim // group_size)
      zero:      fp16,  (batch, heads, seq_len, head_dim // group_size)
    """
    batch, heads, seq_len, head_dim = tensor.shape
    assert head_dim % group_size == 0, (
        f"head_dim ({head_dim}) must be divisible by group_size ({group_size})"
    )
    n_groups = head_dim // group_size

    grouped = tensor.view(batch, heads, seq_len, n_groups, group_size)
    min_val = grouped.min(dim=-1, keepdim=True).values
    max_val = grouped.max(dim=-1, keepdim=True).values
    scale = ((max_val - min_val) / (2 ** bits - 1)).clamp(min=1e-8)
    zero = min_val
    quantized = ((grouped - zero) / scale).round().clamp(0, 2 ** bits - 1)
    return (
        quantized.view(batch, heads, seq_len, head_dim).to(torch.uint8),
        scale.squeeze(-1).to(torch.float16),
        zero.squeeze(-1).to(torch.float16),
    )


def dequantize(quantized, scale, zero):
    """Dequantize per-channel keys: scale/zero shape (batch, heads, 1, head_dim)."""
    return quantized.to(torch.float16) * scale + zero


def dequantize_v(quantized, scale, zero, group_size):
    """
    Dequantize per-token-grouped values.
    quantized: uint8, (batch, heads, seq_len, head_dim)
    scale/zero: fp16, (batch, heads, seq_len, head_dim // group_size)
    """
    batch, heads, seq_len, head_dim = quantized.shape
    n_groups = head_dim // group_size
    grouped = quantized.view(batch, heads, seq_len, n_groups, group_size).to(torch.float16)
    return (grouped * scale.unsqueeze(-1) + zero.unsqueeze(-1)).view(
        batch, heads, seq_len, head_dim
    )


class KIVIMethod(MethodWrapper):
    """
    KIVI: Asymmetric KV cache quantization with block-wise group quantization.

    Implements all three core concepts from Liu et al. (ICML 2024):
    "A Tuning-Free Asymmetric 2bit Quantization for KV Cache"

    Implementation follows the paper's Algorithm 1 exactly:

    1. Asymmetric quantization strategies (per Algorithm 1):
       - Keys:   GroupQuant(dim=channel, numGroup=l/G) — scale per channel,
                 grouped along seq_len (group_size G)
       - Values: GroupQuant(dim=token,   numGroup=d/G) — scale per token,
                 grouped along head_dim (group_size G)

    2. Paper-style residuals (XKr, XVr):
       - XKr: K residual buffer, oscillates 0..R-1 tokens (FP16). When it
              reaches R tokens, all R are quantized as one batch (R/G groups
              of G tokens) and XKr resets to empty.
       - XVr: V residual, always R tokens after warmup. New tokens push the
              oldest out, which get per-token-grouped-quantized one at a time.

    3. Prefill split (per KeyQuant in Algorithm 1):
       - K: r = l % R; XKg = K[:l-r] (multiple of R); XKr = K[l-r:] (residual)
       - V: XVg = V[:l-R];   XVr = V[l-R:]                  (or all in XVr if l <= R)

    Constraint: residual_length must be a multiple of group_size.
    Pure PyTorch — no Triton/CUDA kernels.
    """

    def __init__(self, bits=4, residual_length=128, group_size=32):
        self.bits = bits
        self.residual_length = residual_length
        self.group_size = group_size
        assert residual_length % group_size == 0, (
            f"residual_length ({residual_length}) must be a multiple of "
            f"group_size ({group_size}) — required by the KIVI algorithm."
        )
        self.cache = {}  # layer_idx -> dict with paper-style state (XKr, XVr, blocks)

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
        Split tensor into blocks of group_size along seq_len.
        Within each block, every token is quantized with head_dim grouped
        into chunks of group_size (per-token-grouped, matches reference KIVI V).
        Returns list of (quantized, scale, zero) tuples.
        """
        seq_len = tensor.shape[2]
        blocks = []
        for start in range(0, seq_len, self.group_size):
            end = min(start + self.group_size, seq_len)
            block = tensor[:, :, start:end, :]
            q, s, z = quantize_per_token(block, self.bits, self.group_size)
            blocks.append((q, s, z))
        return blocks

    def _dequantize_blocks(self, blocks):
        """Dequantize K blocks (per-channel) and concatenate along seq_len."""
        if not blocks:
            return None
        pieces = [dequantize(q, s, z) for q, s, z in blocks]
        return torch.cat(pieces, dim=2)

    def _dequantize_v_blocks(self, blocks):
        """Dequantize V blocks (per-token-grouped along head_dim)."""
        if not blocks:
            return None
        pieces = [dequantize_v(q, s, z, self.group_size) for q, s, z in blocks]
        return torch.cat(pieces, dim=2)

    # ── Prefill ───────────────────────────────────────────────────────────────

    def process_prefill(self, past_key_values, attention_weights=None):
        """
        Paper Algorithm 1 — Prefill:
            For Keys:    r = l % R; XKg = XK[:l-r]; XKr = XK[l-r:]
                         Q(XKg) = GroupQuant(XKg, dim=channel, numGroup=(l-r)//G)
            For Values:  XVg = XV[:l-R];  XVr = XV[l-R:]   (or all in XVr if l <= R)
                         Q(XVg) = GroupQuant(XVg, dim=token, numGroup=d//G)
        """
        self.cache = {}
        result = []
        R = self.residual_length

        for layer_idx, layer_kv in enumerate(past_key_values):
            k = layer_kv[0].to(torch.float16)
            v = layer_kv[1].to(torch.float16)
            B, H, l, D = k.shape

            # ── K split:  XKg = K[:l-r],  XKr = K[l-r:]   (r = l % R) ──
            r = l % R
            XKg = k[:, :, :l - r, :]   # multiple of R (hence of G); may be empty if l < R
            XKr = k[:, :, l - r:, :]   # 0 to R-1 tokens

            # ── V split:  XVr = last min(l, R) tokens, XVg = the rest ──
            if l <= R:
                XVg = v[:, :, :0, :]   # empty
                XVr = v
            else:
                XVg = v[:, :, :l - R, :]
                XVr = v[:, :, l - R:, :]

            k_blocks = self._quantize_blocks_per_channel(XKg) if XKg.shape[2] > 0 else []
            v_blocks = self._quantize_blocks_per_token(XVg)   if XVg.shape[2] > 0 else []

            self.cache[layer_idx] = {
                'k_blocks': k_blocks,
                'v_blocks': v_blocks,
                'XKr': XKr,
                'XVr': XVr,
            }

            # Reconstruct full KV for the model's first decode step
            parts_k, parts_v = [], []
            if k_blocks:
                parts_k.append(self._dequantize_blocks(k_blocks))
            if XKr.shape[2] > 0:
                parts_k.append(XKr)
            if v_blocks:
                parts_v.append(self._dequantize_v_blocks(v_blocks))
            if XVr.shape[2] > 0:
                parts_v.append(XVr)

            full_k = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
            full_v = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]
            result.append((full_k, full_v))

        return tuple(result)

    # ── Decode step ───────────────────────────────────────────────────────────

    def process_step(self, past_key_values, step, attention_weights=None):
        """
        Paper Algorithm 1 — Decoding:
            XKr ← Concat([XKr, tK]);  if len(XKr) == R: quantize all R tokens, reset XKr
            XVr ← Concat([XVr, tV]);  if len(XVr)  > R: quantize XVr[:-R], keep last R
        """
        result = []
        R = self.residual_length

        for layer_idx, layer_kv in enumerate(past_key_values):
            k, v = layer_kv[0], layer_kv[1]

            if layer_idx not in self.cache:
                result.append((k, v))
                continue

            state = self.cache[layer_idx]
            new_k = k[:, :, -1:, :].to(torch.float16)
            new_v = v[:, :, -1:, :].to(torch.float16)

            # ── Append new token to residuals ──
            XKr = torch.cat([state['XKr'], new_k], dim=2)
            XVr = torch.cat([state['XVr'], new_v], dim=2)

            # ── K: when residual hits R, quantize all R tokens (in R/G groups), reset ──
            if XKr.shape[2] == R:
                new_k_blocks = self._quantize_blocks_per_channel(XKr)
                state['k_blocks'].extend(new_k_blocks)
                XKr = XKr[:, :, :0, :]   # empty, preserves shape (B, H, 0, D)

            # ── V: when residual exceeds R, quantize the oldest tokens, keep last R ──
            if XVr.shape[2] > R:
                v_to_quant = XVr[:, :, :-R, :]
                new_v_blocks = self._quantize_blocks_per_token(v_to_quant)
                state['v_blocks'].extend(new_v_blocks)
                XVr = XVr[:, :, -R:, :]

            state['XKr'] = XKr
            state['XVr'] = XVr

            # ── Reconstruct full KV for the model ──
            parts_k, parts_v = [], []
            if state['k_blocks']:
                parts_k.append(self._dequantize_blocks(state['k_blocks']))
            if XKr.shape[2] > 0:
                parts_k.append(XKr)
            if state['v_blocks']:
                parts_v.append(self._dequantize_v_blocks(state['v_blocks']))
            if XVr.shape[2] > 0:
                parts_v.append(XVr)

            full_k = torch.cat(parts_k, dim=2) if len(parts_k) > 1 else parts_k[0]
            full_v = torch.cat(parts_v, dim=2) if len(parts_v) > 1 else parts_v[0]
            result.append((full_k, full_v))

        return tuple(result)

    def reset(self):
        self.cache = {}

    def get_kv_size_bytes(self, past_key_values):
        """
        Count storage:
        - Quantized K blocks: uint8 data at actual bit-width + FP16 scale/zero per block
        - Quantized V blocks: uint8 data + FP16 scale/zero (head_dim/group_size scales per token)
        - XKr: FP16 K residual (0 to R-1 tokens)
        - XVr: FP16 V residual (always R tokens after warmup)
        """
        total = 0
        for layer_idx, state in self.cache.items():
            for q, s, z in state['k_blocks']:
                total += int(q.numel() * self.bits / 8)
                total += s.numel() * s.element_size()
                total += z.numel() * z.element_size()
            for q, s, z in state['v_blocks']:
                total += int(q.numel() * self.bits / 8)
                total += s.numel() * s.element_size()
                total += z.numel() * z.element_size()

            total += state['XKr'].numel() * state['XKr'].element_size()
            total += state['XVr'].numel() * state['XVr'].element_size()

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
