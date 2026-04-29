"""RoPE delta-rotation helpers for TopK / Hybrid gather paths.

When we gather a sparse subset of past KV positions and concatenate them into
a contiguous tensor of length ``S_sel``, attention will treat them as if they
sat at indices ``0..S_sel-1``. But each gathered key ``K_i`` was originally
rotated with RoPE at its true absolute position ``p_i``. So unless we re-rotate
the key, the dot-product ``Q · K_i`` corresponds to the wrong relative offset.

For LLaMA-style RoPE the rotation is applied to *complex pairs* of channels.
``rotate_half`` matches HuggingFace's convention (split halves, swap-and-negate),
so to add an extra rotation by ``delta`` to an already-rotated key we use:

    K_corrected = K * cos(delta) + rotate_half(K) * sin(delta)

where ``delta`` is the position shift we want to apply. To map an original
position ``p_orig`` to a target position ``p_new`` we use ``delta = p_new - p_orig``.
"""

from __future__ import annotations

import torch


def _build_inv_freq(head_dim: int, rope_theta: float, device, dtype=torch.float32) -> torch.Tensor:
    """LLaMA RoPE inverse frequencies for the *paired* layout of ``rotate_half``."""
    return 1.0 / (
        rope_theta ** (
            torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
        )
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HuggingFace LLaMA convention: split last dim in half and swap with sign flip."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_delta(
    k: torch.Tensor,
    delta_positions: torch.Tensor,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Apply an additional RoPE rotation by ``delta`` to an already-RoPE'd key tensor.

    Args:
        k: ``(B, H, S_sel, D)`` keys that were already rotated at their original
            positions.
        delta_positions: integer tensor of shape ``(S_sel,)`` (or broadcastable to
            ``(B, S_sel)``) giving the per-token rotation delta in *positions*.
            Use ``new_pos - orig_pos`` to remap to a new layout.
        rope_theta: base used when training the model (LLaMA-2: 10000.0).

    Returns:
        ``k`` rotated by ``delta_positions`` -- same shape and dtype as input.
    """
    if k.dim() != 4:
        raise ValueError(f"apply_rope_delta expects (B,H,S,D) keys, got {tuple(k.shape)}")

    D = k.shape[-1]
    if D % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {D}")

    inv_freq = _build_inv_freq(D, rope_theta, k.device)  # (D/2,)
    delta = delta_positions.to(k.device, dtype=torch.float32)
    if delta.dim() == 1:
        # (S_sel,) -> (1, S_sel)
        delta = delta.unsqueeze(0)
    # delta: (B_or_1, S_sel)  ->  freqs: (B_or_1, S_sel, D/2)
    freqs = delta.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    # Repeat to match the rotate_half pairing (HF convention concatenates halves).
    emb = torch.cat([freqs, freqs], dim=-1)  # (B_or_1, S_sel, D)
    cos = emb.cos().to(k.dtype).unsqueeze(1)  # (B_or_1, 1, S_sel, D)
    sin = emb.sin().to(k.dtype).unsqueeze(1)
    return k * cos + _rotate_half(k) * sin
