"""
Fused Triton kernel: sparse gather + unpack + dequant + weighted sum over
selected rows of a quantized V-cache.

Replaces the three-step PyTorch path:
  1. gather rows  →  (B,nh,budget,D) FP16 buffer   (big alloc)
  2. unpack int32 → int
  3. dequant + matmul

With a single kernel that streams each selected V row, unpacks on-the-fly,
multiplies by the attention weight, and accumulates — writing only the
(B,nh,D) output vector.

V storage layout (produced by triton_quantize_and_pack_along_last_dim with
pack_dim=3, matching KIVI's vcache convention):
  shape : (B, nh, T, head_dim // fps)   int32
  scale : (B, nh, T, head_dim // group_size)  fp16
  mn    : (B, nh, T, head_dim // group_size)  fp16

Packing: for each token row, the head_dim values are grouped into blocks of
group_size; within each group the values are packed into int32 words:
  packed[j] holds values[j*fps : (j+1)*fps]
  values[i]  = (packed[i // fps] >> ((i % fps) * bits)) & ((1 << bits) - 1)
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _sparse_weighted_sum_kernel(
    v_quant_ptr,        # (BH, n_quant, packed_dim) int32
    v_scale_ptr,        # (BH, n_quant, n_groups)   fp16
    v_mn_ptr,           # (BH, n_quant, n_groups)   fp16
    attn_w_ptr,         # (BH, budget_q)             fp16
    idx_ptr,            # (budget_q,)                int32
    out_ptr,            # (BH, head_dim)             fp16  (output)
    n_quant,
    budget_q,
    packed_dim,         # = head_dim // fps  (runtime, not constexpr)
    # ── compile-time constants ─────────────────────────────────────────────
    n_groups:       tl.constexpr,   # head_dim // group_size
    group_size:     tl.constexpr,   # e.g. 32
    ints_per_group: tl.constexpr,   # group_size // fps  (packed ints per group)
    fps:            tl.constexpr,   # 32 // bits
    bits:           tl.constexpr,   # 2 or 4
    head_dim:       tl.constexpr,   # e.g. 128
):
    """
    Grid: (B*nh, n_groups).
    Each program accumulates the weighted dequantized sum for one
    (batch*head, group-of-dims) slice over all budget_q selected tokens.
    """
    bh = tl.program_id(0)   # flattened batch*head index
    g  = tl.program_id(1)   # group index within head_dim

    acc = tl.zeros((group_size,), dtype=tl.float32)

    # Within a group, position i comes from packed int j_vals[i] at bit slot k_vals[i].
    i_vals = tl.arange(0, group_size)
    j_vals = i_vals // fps   # [0,0,...(fps times)...,1,1,..., ..., ints_per_group-1,...]
    k_vals = i_vals % fps    # [0,1,...,fps-1, 0,1,...,fps-1, ...]
    mask_int = (1 << bits) - 1

    for i in range(budget_q):
        # Row index into v_quant for this selected token.
        row = tl.load(idx_ptr + i).to(tl.int32)
        # Attention weight (scalar, already softmax-normalised).
        w   = tl.load(attn_w_ptr + bh * budget_q + i).to(tl.float32)

        # Dequant parameters: one scale and one min per group.
        sg_base = bh * n_quant * n_groups + row * n_groups + g
        scale = tl.load(v_scale_ptr + sg_base).to(tl.float32)
        mn    = tl.load(v_mn_ptr    + sg_base).to(tl.float32)

        # Load ints_per_group packed int32 words for this (row, group).
        # j_vals gathers each int word once per fps elements — cheap, stays in L1.
        pq_base  = bh * n_quant * packed_dim + row * packed_dim + g * ints_per_group
        gathered = tl.load(v_quant_ptr + pq_base + j_vals)   # (group_size,) int32

        # Unpack and dequantize.
        vals    = ((gathered >> (k_vals * bits)) & mask_int).to(tl.float32)
        dequant = vals * scale + mn

        acc += w * dequant

    # Store (group_size,) fp16 result for this (bh, group).
    out_base = bh * head_dim + g * group_size
    tl.store(out_ptr + out_base + tl.arange(0, group_size), acc.to(tl.float16))


def sparse_weighted_sum_quant(
    v_quant:    torch.Tensor,   # (B, nh, n_quant, packed_dim) int32
    v_scale:    torch.Tensor,   # (B, nh, n_quant, n_groups)   fp16
    v_mn:       torch.Tensor,   # (B, nh, n_quant, n_groups)   fp16
    attn_w:     torch.Tensor,   # (B, nh, budget_q)            fp16
    quant_idx:  torch.Tensor,   # (budget_q,)                  int64/int32
    bits:       int,
    group_size: int,
) -> torch.Tensor:
    """
    Fused sparse gather + unpack + dequant + weighted sum.

    For each selected token in quant_idx, reads the packed int32 V row,
    unpacks on-the-fly, dequantizes with per-group scale/mn, multiplies by
    the corresponding attention weight, and accumulates into the output.

    Returns: (B, nh, head_dim) float16.
    """
    B, nh, n_quant, packed_dim = v_quant.shape
    BH       = B * nh
    budget_q = int(quant_idx.shape[0])
    fps      = 32 // bits
    head_dim = packed_dim * fps
    n_groups       = head_dim  // group_size
    ints_per_group = group_size // fps

    assert group_size % fps == 0, "group_size must be divisible by fps (32 // bits)"
    assert (group_size & (group_size - 1)) == 0, "group_size must be a power of 2"

    v_quant_f = v_quant.reshape(BH, n_quant, packed_dim).contiguous()
    v_scale_f = v_scale.reshape(BH, n_quant, n_groups).contiguous()
    v_mn_f    = v_mn.reshape(   BH, n_quant, n_groups).contiguous()
    attn_w_f  = attn_w.reshape( BH, budget_q).contiguous()
    idx_i32   = quant_idx.to(torch.int32).contiguous()

    out = torch.empty(BH, head_dim, dtype=torch.float16, device=v_quant.device)

    grid = (BH, n_groups)
    _sparse_weighted_sum_kernel[grid](
        v_quant_f, v_scale_f, v_mn_f,
        attn_w_f, idx_i32, out,
        n_quant, budget_q, packed_dim,
        n_groups=n_groups,
        group_size=group_size,
        ints_per_group=ints_per_group,
        fps=fps,
        bits=bits,
        head_dim=head_dim,
        num_warps=4,
    )

    return out.reshape(B, nh, head_dim)
