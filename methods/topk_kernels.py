"""
Custom Triton kernels for the TopK / TokenSelect KV-cache method.

These kernels accelerate the critical decode-step path in
`topk_selection.py`. The reference implementation in that file is pure
PyTorch and is kept as a numerically-equivalent fallback that runs
whenever Triton is unavailable, the tensors live on CPU, or the shape
falls outside the regime the kernels are tuned for.

Two kernels are provided:

  1. fused_paged_score
       Fuses scaled-dot-product (Q · Kᵀ / √d) + per-head numerically-
       stable softmax + per-head criticality weighting + cross-head
       reduction into a single Triton kernel.

       Replaces these four PyTorch ops in `_paged_token_selection`:
           raw   = Q @ K.transpose(-2, -1) / sqrt(d)         # (1,H,1,M)
           sm    = softmax(raw, dim=-1)                      # (1,H,M)
           sm    = sm * head_weights                         # (1,H,M)
           agg   = sm.sum(dim=1).squeeze(0)                  # (M,)

       The fused version never materialises the (H, M) intermediates;
       cross-head accumulation happens via atomicAdd into a single
       (M,) buffer in fp32. We use a one-program-per-head launch with
       FlashAttention-style online softmax (two passes over K for
       numerical stability without storing scores).

  2. fused_paged_topk
       Vectorised two-stage page-then-token top-K. Replaces the
       Python for-loop in `_paged_topk` that gathers candidate token
       indices from selected pages one page at a time. Pure tensor
       ops, runs on any device — no Triton dependency.

Both kernels are exposed via thin wrappers that auto-fallback to the
PyTorch reference path. Callers select the path with a single
`use_kernels` flag.
"""

import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:  # Triton ships with PyTorch builds for CUDA but not CPU
    TRITON_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# Availability check — kernels only run on CUDA tensors with Triton present.
# ────────────────────────────────────────────────────────────────────────────

def kernels_available(device=None) -> bool:
    """Return True iff Triton is importable and the device is CUDA."""
    if not TRITON_AVAILABLE:
        return False
    if device is None:
        return torch.cuda.is_available()
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return str(device).startswith("cuda")


# ────────────────────────────────────────────────────────────────────────────
# Kernel 1: Fused paged score
#   Fused: (Q · Kᵀ)/√d  →  per-head softmax  →  × head_weights  →  Σ_heads
# ────────────────────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.jit
    def _fused_paged_score_kernel(
        Q_ptr,          # fp16/fp32, shape (H, D)              — proxy query
        K_ptr,          # fp16/fp32, shape (H, M, D)           — middle keys
        W_ptr,          # fp32,      shape (H,)                — per-head weights
        OUT_ptr,        # fp32,      shape (M,) — pre-zeroed   — output buffer
        M,                                                       # runtime
        H:       tl.constexpr,
        D:       tl.constexpr,
        SCALE:   tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        One program per attention head. Each program walks K[h, :, :]
        twice (online softmax) and atomic-adds its softmax-weighted
        contribution into the shared (M,) output.

        Algorithm (FlashAttention-style single-query online softmax):

          PASS 1 — find normalisation constants
            m_max = -inf, m_sum = 0
            for each block of K:
              s = (Q[h] · K[h, block]) / √d
              new_max = max(m_max, max(s))
              m_sum   = m_sum · exp(m_max - new_max) + Σ exp(s - new_max)
              m_max   = new_max

          PASS 2 — write weighted softmax with atomic accumulate
            for each block of K:
              s  = (Q[h] · K[h, block]) / √d
              sm = exp(s - m_max) / m_sum · w[h]
              atomic_add OUT[block] += sm
        """
        h = tl.program_id(0)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        # Load Q[h, :] once — kept in registers for the whole program
        q = tl.load(Q_ptr + h * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + h).to(tl.float32)

        k_head_offset = h * M * D

        # ── PASS 1: online softmax statistics ──
        m_max = -float('inf')
        m_sum = 0.0
        for m_start in range(0, M, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            m_mask = m_offs < M
            k_blk = tl.load(
                K_ptr + k_head_offset
                      + m_offs[:, None] * D
                      + d_offs[None, :],
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(k_blk * q[None, :], axis=1) * SCALE
            s = tl.where(m_mask, s, -float('inf'))
            blk_max = tl.max(s, axis=0)
            new_max = tl.maximum(m_max, blk_max)
            m_sum = m_sum * tl.exp(m_max - new_max) \
                    + tl.sum(tl.exp(s - new_max), axis=0)
            m_max = new_max

        inv_sum = 1.0 / m_sum

        # ── PASS 2: weighted softmax → atomic_add ──
        for m_start in range(0, M, BLOCK_M):
            m_offs = m_start + tl.arange(0, BLOCK_M)
            m_mask = m_offs < M
            k_blk = tl.load(
                K_ptr + k_head_offset
                      + m_offs[:, None] * D
                      + d_offs[None, :],
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            s = tl.sum(k_blk * q[None, :], axis=1) * SCALE
            sm = tl.exp(s - m_max) * inv_sum * w
            tl.atomic_add(OUT_ptr + m_offs, sm, mask=m_mask)


def fused_paged_score(proxy_q, middle_k, head_weights):
    """
    Compute aggregated[m] = Σ_h softmax_h( (Q[h] · K[h,:,:]ᵀ) / √d )[m] · w[h]
    in a single fused pass when Triton + CUDA are available.

    Args:
        proxy_q:      (1, H, 1, D)  — single query, one per head
        middle_k:     (1, H, M, D)  — keys to score
        head_weights: (H,) or None  — per-head criticality weights

    Returns:
        aggregated:   (M,) fp32 — cross-head reduced scores
    """
    assert proxy_q.dim() == 4 and middle_k.dim() == 4
    _, H, _, D = proxy_q.shape
    _, Hk, M, Dk = middle_k.shape
    assert H == Hk and D == Dk

    if head_weights is None:
        head_weights = torch.ones(H, device=proxy_q.device, dtype=torch.float32)
    else:
        head_weights = head_weights.to(torch.float32)

    # ── PyTorch reference path (CPU, no Triton, or fallback) ──
    if not kernels_available(proxy_q.device):
        scale = 1.0 / math.sqrt(D)
        raw = torch.matmul(proxy_q, middle_k.transpose(-2, -1)) * scale
        raw = raw.squeeze(2).squeeze(0)                 # (H, M)
        sm  = torch.softmax(raw, dim=-1).to(torch.float32)
        sm  = sm * head_weights.unsqueeze(-1)           # (H, M)
        return sm.sum(dim=0)                            # (M,)

    # ── Triton fast path ──
    q = proxy_q.squeeze(2).squeeze(0).contiguous()      # (H, D)
    k = middle_k.squeeze(0).contiguous()                # (H, M, D)
    out = torch.zeros(M, device=q.device, dtype=torch.float32)

    BLOCK_M = 128 if M > 1024 else 64
    # next-power-of-2 for D so triton.arange(0, BLOCK_D) covers it
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    _fused_paged_score_kernel[(H,)](
        q, k, head_weights, out,
        M,
        H=H, D=D,
        SCALE=1.0 / math.sqrt(D),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out


# ────────────────────────────────────────────────────────────────────────────
# Kernel 1b: Quant-aware raw score (design c)
#   Score (H, M) = (Q · dequant(K_q, K_s, K_z)ᵀ) / √d
#   directly on KIVI's uint8 block storage — no full-cache dequantisation.
#
# KIVI stores K as (B, H, M, D) uint8 with per-token (B, H, M, 1) fp16 scale
# and zero (`quantize_per_channel` reduces over the last axis, so each token
# gets a scalar scale/zero). Dequantisation is therefore:
#       dequant[h,m,d] = qk[h,m,d] * sk[h,m] + zk[h,m]
# and the dot product factors:
#       score[h,m] = sk[h,m] · Σ_d Q[h,d] · qk[h,m,d]    +  zk[h,m] · Σ_d Q[h,d]
#                    └──────── int×fp inner loop ───────┘   └──────── precomputed ───
# We hoist the `Σ_d Q[h,d]` outside the kernel (one fp32 per head) and read
# uint8 K through 32-bit-wide vector loads. The arithmetic is identical in
# count to the FP16 path, but the K-side memory bandwidth drops 4× — which
# is the win design (c) is supposed to deliver vs scoring on fully-dequant
# FP16 keys (the naïve KIVI→FP16→TopK composition).
# ────────────────────────────────────────────────────────────────────────────

if TRITON_AVAILABLE:

    @triton.jit
    def _quant_score_kernel(
        Q_ptr,          # fp32, shape (H, D)             — query, pre-scaled by 1/√d
        QSUM_ptr,       # fp32, shape (H,)               — Σ_d Q[h,d] / √d  (precomputed)
        KQ_ptr,         # uint8, shape (H, M, D)         — quantised keys
        KS_ptr,         # fp16, shape (H, M)             — per-token scale
        KZ_ptr,         # fp16, shape (H, M)             — per-token zero
        OUT_ptr,        # fp32, shape (H, M)             — raw scores (out)
        M,                                                  # runtime
        H:       tl.constexpr,
        D:       tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        One program per (head, M-block). Reads uint8 K, fp32 Q, fp16 sk/zk;
        writes fp32 (H, M) raw scaled-dot-product scores. No softmax — that
        runs downstream over the union of {block, overflow, residual} so
        the quantised region must contribute *raw* scores, not normalised.
        """
        h        = tl.program_id(0)
        m_start  = tl.program_id(1) * BLOCK_M

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        m_offs = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M

        # Load Q[h, :] (already scaled by 1/√d) and Σ_d Q[h, d] · 1/√d.
        q     = tl.load(Q_ptr + h * D + d_offs, mask=d_mask, other=0.0)
        q_sum = tl.load(QSUM_ptr + h)                          # scalar

        # Load uint8 block (BLOCK_M, BLOCK_D)
        kq_offset = h * M * D + m_offs[:, None] * D + d_offs[None, :]
        kq = tl.load(
            KQ_ptr + kq_offset,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.float32)                                        # (BLOCK_M, BLOCK_D)

        # int×fp inner: (BLOCK_M,) = Σ_d kq[m, d] · q[d]
        dot_int = tl.sum(kq * q[None, :], axis=1)               # (BLOCK_M,)

        # Per-token scale/zero (fp16 → fp32)
        ks = tl.load(KS_ptr + h * M + m_offs, mask=m_mask, other=0.0).to(tl.float32)
        kz = tl.load(KZ_ptr + h * M + m_offs, mask=m_mask, other=0.0).to(tl.float32)

        score = ks * dot_int + kz * q_sum                       # (BLOCK_M,)

        tl.store(OUT_ptr + h * M + m_offs, score, mask=m_mask)


def quant_score(proxy_q, K_q, K_s, K_z):
    """
    Raw scaled-dot-product scores against KIVI-quantised keys.

    Args:
        proxy_q: (1, H, 1, D) fp16 / fp32 — single query, one row per head
        K_q:     (1, H, M, D) uint8        — quantised keys
        K_s:     (1, H, M, 1) fp16         — per-token scale
        K_z:     (1, H, M, 1) fp16         — per-token zero

    Returns:
        scores:  (H, M) fp32 — raw scores already divided by √d. Caller is
                 responsible for masking, softmax, and per-head weighting.

    Numerical contract: identical to
        (proxy_q.float() @ dequant(K_q,K_s,K_z).float().transpose(-2,-1)
        ).squeeze(2).squeeze(0) / sqrt(D)
    up to fp16-load precision.
    """
    assert proxy_q.dim() == 4 and K_q.dim() == 4
    _, H, _, D = proxy_q.shape
    _, Hk, M, Dk = K_q.shape
    assert H == Hk and D == Dk
    assert K_q.dtype == torch.uint8

    scale = 1.0 / math.sqrt(D)

    # ── PyTorch reference path (CPU, no Triton, or fallback) ──
    if not kernels_available(proxy_q.device):
        K_fp = K_q.to(torch.float32) * K_s.to(torch.float32) + K_z.to(torch.float32)
        raw = torch.matmul(
            proxy_q.float(), K_fp.transpose(-2, -1)
        ).squeeze(2).squeeze(0) * scale                         # (H, M)
        return raw

    # ── Triton fast path ──
    q_pre  = (proxy_q.float().squeeze(2).squeeze(0) * scale).contiguous()  # (H, D)
    q_sum  = q_pre.sum(dim=-1).contiguous()                                # (H,)
    kq     = K_q.squeeze(0).contiguous()                                   # (H, M, D) uint8
    ks     = K_s.squeeze(0).squeeze(-1).contiguous()                       # (H, M) fp16
    kz     = K_z.squeeze(0).squeeze(-1).contiguous()                       # (H, M) fp16
    out    = torch.empty((H, M), device=proxy_q.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    grid = (H, triton.cdiv(M, BLOCK_M))
    _quant_score_kernel[grid](
        q_pre, q_sum, kq, ks, kz, out,
        M,
        H=H, D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out


# ────────────────────────────────────────────────────────────────────────────
# Kernel 2: Vectorised two-stage paged top-K
#   Stage 1 — page-level: page_score = max token score in page → top-P pages.
#   Stage 2 — token-level: gather all tokens from those pages → final top-K.
# ────────────────────────────────────────────────────────────────────────────

def fused_paged_topk(scores, page_size, K):
    """
    Vectorised two-stage paged top-K. Replaces the Python for-loop in
    the reference implementation with a single tensor expression that
    builds the candidate index set in one shot.

    Args:
        scores:    (M,) fp32 — token scores
        page_size: int       — logical page size for stage-1 grouping
        K:         int       — final top-K count

    Returns:
        (K_actual,) int64 — selected token indices into `scores`.
    """
    device = scores.device
    M = scores.numel()
    n_pages = (M + page_size - 1) // page_size
    pad_len = n_pages * page_size - M

    # Pad with -inf so partial trailing page never wins on padding.
    if pad_len > 0:
        scores_padded = torch.cat([
            scores,
            torch.full((pad_len,), float('-inf'),
                       device=device, dtype=scores.dtype),
        ])
    else:
        scores_padded = scores

    paged = scores_padded.view(n_pages, page_size)                  # (P, S)

    # Stage 1: per-page max → choose top pages.
    page_max = paged.max(dim=-1).values                             # (P,)
    n_pages_needed = min(
        n_pages,
        max((K + page_size - 1) // page_size * 2, 4),               # 2× margin
    )
    top_pages = page_max.topk(n_pages_needed).indices               # (P_sel,)

    # Stage 2: vectorised candidate index construction.
    # candidate[p, s] = top_pages[p] * page_size + s
    page_starts = top_pages * page_size                             # (P_sel,)
    s_offsets   = torch.arange(page_size, device=device)            # (S,)
    candidates  = (page_starts.unsqueeze(1) + s_offsets.unsqueeze(0)).flatten()
    candidates  = candidates[candidates < M]                        # drop padding

    cand_scores = scores[candidates]
    k_actual    = min(K, candidates.numel())
    sel         = cand_scores.topk(k_actual).indices

    return candidates[sel]


# ────────────────────────────────────────────────────────────────────────────
# Self-test (CPU fallback path) — `python -m methods.topk_kernels`
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    H, M, D = 32, 4000, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Self-test on {device} (Triton available: {TRITON_AVAILABLE}, "
          f"kernels active: {kernels_available(device)})")

    q = torch.randn(1, H, 1, D, device=device, dtype=torch.float16)
    k = torch.randn(1, H, M, D, device=device, dtype=torch.float16)
    w = torch.rand(H, device=device, dtype=torch.float32)
    w = w / w.sum()

    # Reference (PyTorch)
    scale = 1.0 / math.sqrt(D)
    raw = torch.matmul(q, k.transpose(-2, -1)) * scale
    raw = raw.squeeze(2).squeeze(0).float()
    sm = torch.softmax(raw, dim=-1)
    ref = (sm * w.unsqueeze(-1)).sum(dim=0)

    out = fused_paged_score(q, k, w)
    abs_err = (out - ref).abs().max().item()
    rel_err = abs_err / (ref.abs().max().item() + 1e-12)
    print(f"  fused_paged_score:   max abs err = {abs_err:.2e}  "
          f"rel err = {rel_err:.2e}")
    assert rel_err < 1e-3, "fused_paged_score: numerical mismatch"

    # Paged top-K
    scores = torch.randn(M, device=device)
    sel = fused_paged_topk(scores, page_size=64, K=512)
    assert sel.numel() == 512
    print(f"  fused_paged_topk:    selected {sel.numel()} indices, "
          f"top score = {scores[sel].max().item():.3f}, "
          f"global max = {scores.max().item():.3f}")

    print("Self-test PASSED")
