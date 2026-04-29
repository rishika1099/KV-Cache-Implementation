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


def test_realistic_topk_gather_scenario():
    """Simulate the actual TopK gather: K rotated at sparse positions,
    re-rotated to dense slot indices. Q at dense slot S_sel must produce
    the *same dot-product up to RoPE shift* as Q would have produced at
    the original full length S against original-position keys, modulo
    the relative-position compression that sparse selection inevitably
    introduces.

    What we *can* assert is internal consistency: after correction,
    Q·K_i should equal (rope_apply(unrotated_q, slot_S) ·
    rope_apply(unrotated_k_i, slot_i))^T — i.e. we sit on a clean
    RoPE manifold rather than the corrupted one we'd get without the
    fix.
    """
    torch.manual_seed(3)
    D = 64
    S_full = 4096
    S_sel = 16

    # Pick a sparse subset of original positions.
    indices = torch.tensor(
        sorted([1, 5, 100, 500, 1000, 1500, 2000, 2500,
                3000, 3200, 3400, 3600, 3800, 3900, 4000, 4090])
    )
    assert indices.numel() == S_sel

    # Underlying (unrotated) K at those positions.
    k_unrot = torch.randn(1, 1, S_sel, D)
    # K as it lives in cache (rotated at original positions).
    k_at_orig = rope_apply(k_unrot, indices)

    # After gather + RoPE-delta correction.
    slot = torch.arange(S_sel)
    delta = (slot - indices).to(torch.float32)
    k_corrected = apply_rope_delta(k_at_orig, delta)

    # What we should get if we'd rotated `k_unrot` directly at slot positions.
    k_at_slot_direct = rope_apply(k_unrot, slot)

    err = (k_corrected - k_at_slot_direct).abs().max().item()
    print(f"  TopK-gather scenario max err = {err:.3e}")
    assert err < 1e-3


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

    print("test_realistic_topk_gather_scenario ...")
    test_realistic_topk_gather_scenario()
    print("  OK\n")

    print("All RoPE-correction tests PASSED.")


if __name__ == "__main__":
    main()
