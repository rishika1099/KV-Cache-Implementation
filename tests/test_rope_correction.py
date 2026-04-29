"""Unit tests for the RoPE delta-rotation helper in methods/rope_utils.py.

These run on CPU and verify the algebraic property we rely on for BUG-2:

    apply_rope_delta(K_at_p, delta=q-p) ≡ rotate K from position p to q.

We test by composing two LLaMA-style RoPE rotations with delta math and
checking we get the same tensor as a direct rotation to the target position.

Run with:
    PYTHONPATH=. python tests/test_rope_correction.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.rope_utils import (  # noqa: E402
    _build_inv_freq, _rotate_half, apply_rope_delta,
)


def rope_apply(x: torch.Tensor, pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Standard LLaMA RoPE applied at absolute position(s) `pos`.

    Args:
        x: (B, H, S, D)
        pos: (S,) absolute positions
    """
    D = x.shape[-1]
    inv_freq = _build_inv_freq(D, theta, x.device)  # (D/2,)
    freqs = pos.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # (S, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)                    # (S, D)
    cos = emb.cos().to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


def test_delta_equals_direct_rotation():
    """K rotated to p, then delta-rotated by (q-p), should equal K rotated to q."""
    torch.manual_seed(0)
    B, H, S, D = 1, 4, 7, 64
    theta = 10000.0

    x = torch.randn(B, H, S, D, dtype=torch.float32)
    orig_pos = torch.tensor([0, 5, 17, 102, 333, 999, 4095])
    target_pos = torch.tensor([3, 9, 200, 50, 1024, 1000, 5000])

    k_at_orig = rope_apply(x, orig_pos, theta=theta)
    k_at_target_direct = rope_apply(x, target_pos, theta=theta)

    delta = (target_pos - orig_pos).to(torch.float32)
    k_at_target_via_delta = apply_rope_delta(k_at_orig, delta, rope_theta=theta)

    err = (k_at_target_via_delta - k_at_target_direct).abs().max().item()
    print(f"  max |delta-rot − direct-rot| = {err:.3e}")
    # Bound is loose-but-tight enough to detect formula errors. RoPE at very
    # high absolute positions (~5000) loses a few bits to fp32 trig roundoff.
    assert err < 1e-3, f"Delta rotation diverged: {err}"


def test_zero_delta_is_identity():
    """delta=0 must leave K untouched."""
    torch.manual_seed(1)
    x = torch.randn(1, 2, 5, 64)
    delta = torch.zeros(5)
    out = apply_rope_delta(x, delta)
    err = (out - x).abs().max().item()
    print(f"  max |out − x| (zero delta) = {err:.3e}")
    assert err < 1e-6


def test_round_trip_to_zero():
    """K rotated to p, then delta-rotated by -p, should match K (delta to 0)."""
    torch.manual_seed(2)
    x = torch.randn(1, 2, 4, 64)
    pos = torch.tensor([10, 50, 150, 1000])
    k_at_p = rope_apply(x, pos)
    k_back = apply_rope_delta(k_at_p, -pos.to(torch.float32))
    err = (k_back - x).abs().max().item()
    print(f"  max |round-trip − x| = {err:.3e}")
    assert err < 1e-4


def test_constant_shift_preserves_qk_dotproduct():
    """The key correctness property of the BUG-2 fix.

    Setup: Q at original full position S_full, K_i at sparse original
    positions. After gather + constant-shift re-rotation, HF's view is
    that K' lives at slots 0..S_sel-1 and Q' is at slot S_sel.

    Constant shift = S_sel - S_full applied to every K. We verify that
    Q'·K'_i (the post-fix value) matches Q·K_i (the un-truncated value)
    within fp32 noise, for all i.

    This is the property a per-token shift would *break* (it would
    compact the layout and lose the relative offsets), and the property
    that lets the model still find the needle in passkey retrieval.
    """
    torch.manual_seed(3)
    B, H, D = 1, 1, 64
    S_full = 4096
    S_sel = 16
    theta = 10000.0

    indices = torch.tensor(sorted([
        1, 5, 100, 500, 1000, 1500, 2000, 2500,
        3000, 3200, 3400, 3600, 3800, 3900, 4000, 4090,
    ]))
    assert indices.numel() == S_sel

    # Underlying unrotated Q (at full length) and K (at the sparse positions).
    q_unrot = torch.randn(B, H, 1, D)
    k_unrot = torch.randn(B, H, S_sel, D)

    # Original (un-truncated) computation.
    q_full = rope_apply(q_unrot, torch.tensor([S_full]), theta=theta)
    k_full = rope_apply(k_unrot, indices, theta=theta)
    dot_orig = torch.matmul(q_full, k_full.transpose(-2, -1)).squeeze()  # (S_sel,)

    # Post-fix computation: Q rotated by HF at slot S_sel; K rotated by
    # constant shift = S_sel - S_full applied to its original-position phase.
    q_slot = rope_apply(q_unrot, torch.tensor([S_sel]), theta=theta)
    k_at_orig = rope_apply(k_unrot, indices, theta=theta)
    shift = float(S_sel - S_full)
    delta = torch.full((S_sel,), shift, dtype=torch.float32)
    k_corrected = apply_rope_delta(k_at_orig, delta, rope_theta=theta)
    dot_post = torch.matmul(q_slot, k_corrected.transpose(-2, -1)).squeeze()

    err = (dot_orig - dot_post).abs().max().item()
    print(f"  max |dot_orig - dot_post| (constant-shift) = {err:.3e}")
    # fp32 dot of D=64 vectors with a few-thousand-position rotation: 1e-3 ok.
    assert err < 1e-3


def test_per_token_shift_breaks_qk_dotproduct():
    """Confirms the failure mode of the *first* fix attempt.

    Per-token delta_i = slot_i - orig_pos_i compacts the layout. Q at slot
    S_sel against K_i at slot i would give relative offset S_sel - i,
    not S_full - orig_pos_i. The dot product diverges from the
    un-truncated reference. This test demonstrates the divergence so
    that, if anyone reverts to the per-token formulation, the unit
    suite fails loudly.
    """
    torch.manual_seed(4)
    B, H, D = 1, 1, 64
    S_full = 4096
    S_sel = 16
    theta = 10000.0

    indices = torch.tensor(sorted([
        1, 5, 100, 500, 1000, 1500, 2000, 2500,
        3000, 3200, 3400, 3600, 3800, 3900, 4000, 4090,
    ]))

    q_unrot = torch.randn(B, H, 1, D)
    k_unrot = torch.randn(B, H, S_sel, D)

    q_full = rope_apply(q_unrot, torch.tensor([S_full]), theta=theta)
    k_full = rope_apply(k_unrot, indices, theta=theta)
    dot_orig = torch.matmul(q_full, k_full.transpose(-2, -1)).squeeze()

    # Per-token delta (the buggy first attempt).
    q_slot = rope_apply(q_unrot, torch.tensor([S_sel]), theta=theta)
    k_at_orig = rope_apply(k_unrot, indices, theta=theta)
    slot = torch.arange(S_sel, dtype=torch.float32)
    delta_per_token = slot - indices.to(torch.float32)
    k_buggy = apply_rope_delta(k_at_orig, delta_per_token, rope_theta=theta)
    dot_buggy = torch.matmul(q_slot, k_buggy.transpose(-2, -1)).squeeze()

    err = (dot_orig - dot_buggy).abs().max().item()
    print(f"  max |dot_orig - dot_buggy| (per-token shift) = {err:.3e}")
    # We *expect* a large divergence — something on the order of a few
    # standard deviations of a random dot product.
    assert err > 0.1, (
        "Per-token shift unexpectedly preserves the dot product. "
        "Did someone change apply_rope_delta semantics?"
    )


def main():
    print("test_delta_equals_direct_rotation ...")
    test_delta_equals_direct_rotation()
    print("  OK\n")

    print("test_zero_delta_is_identity ...")
    test_zero_delta_is_identity()
    print("  OK\n")

    print("test_round_trip_to_zero ...")
    test_round_trip_to_zero()
    print("  OK\n")

    print("test_constant_shift_preserves_qk_dotproduct ...")
    test_constant_shift_preserves_qk_dotproduct()
    print("  OK\n")

    print("test_per_token_shift_breaks_qk_dotproduct ...")
    test_per_token_shift_breaks_qk_dotproduct()
    print("  OK\n")

    print("All RoPE-correction tests PASSED.")


if __name__ == "__main__":
    main()
