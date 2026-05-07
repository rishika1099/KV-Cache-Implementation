"""
Fused Triton kernel: sparse gather + unpack + dequant + weighted sum over
selected rows of a quantized V-cache.

Design (v2 — budget-parallel):
  Grid  : (B*nh,  ceil(budget_q / BLOCK_BQ))
  Threads: BLOCK_BQ  (= 1 warp, 32 threads by default)

  Each program handles BLOCK_BQ budget tokens × all head_dim dims.
  The budget dimension is parallelised across programs → good occupancy.
  Each program atomic-adds its partial (group_size,) result per group into
  the float32 output, which is later cast to fp16.

  Previous v1 design had grid=(B*nh, n_groups)=128 programs each doing
  budget_q=8320 serial iterations — terrible GPU utilisation.

V storage layout (KIVI vcache convention, pack_dim=3):
  v_quant : (B, nh, T, head_dim // fps)        int32
  v_scale : (B, nh, T, head_dim // group_size) fp16
  v_mn    : (B, nh, T, head_dim // group_size) fp16

Packing: packed[j] holds values[j*fps : (j+1)*fps]
  values[i] = (packed[i // fps] >> ((i % fps) * bits)) & mask
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _sparse_ws_kernel(
    v_quant_ptr,    # (BH, n_quant, packed_dim) int32
    v_scale_ptr,    # (BH, n_quant, n_groups)   fp16
    v_mn_ptr,       # (BH, n_quant, n_groups)   fp16
    attn_w_ptr,     # (BH, budget_q)             fp16
    idx_ptr,        # (budget_q,)                int32
    out_ptr,        # (BH, head_dim)             fp32  — zero-initialised, atomic_add
    n_quant,
    budget_q,
    packed_dim,
    # compile-time constants
    n_groups:       tl.constexpr,
    group_size:     tl.constexpr,
    ints_per_group: tl.constexpr,
    fps:            tl.constexpr,
    bits:           tl.constexpr,
    head_dim:       tl.constexpr,
    BLOCK_BQ:       tl.constexpr,   # tokens per program tile (= 1 warp = 32)
):
    bh      = tl.program_id(0)
    bq_tile = tl.program_id(1)

    bq_start = bq_tile * BLOCK_BQ
    bq_offs  = bq_start + tl.arange(0, BLOCK_BQ)   # (BLOCK_BQ,)
    bq_mask  = bq_offs < budget_q                   # (BLOCK_BQ,) bool

    # Load row indices and attention weights for this budget tile.
    rows = tl.load(idx_ptr + bq_offs, mask=bq_mask, other=0).to(tl.int32)   # (BLOCK_BQ,)
    ws   = tl.load(attn_w_ptr + bh * budget_q + bq_offs,
                   mask=bq_mask, other=0.0).to(tl.float32)                   # (BLOCK_BQ,)

    # Unpack helpers: for each element i in [0, group_size):
    #   j_vals[i] = which packed int32 within the group
    #   k_vals[i] = which bit slot within that int32
    mask_int = (1 << bits) - 1
    i_vals   = tl.arange(0, group_size)            # (group_size,)
    j_vals   = i_vals // fps                       # (group_size,)
    k_vals   = i_vals % fps                        # (group_size,)

    # Loop over head_dim groups (n_groups iterations, e.g. 4 for D=128, gs=32).
    # This is a short compile-time loop.
    for g in range(n_groups):
        # Per-group scale and min for each token: (BLOCK_BQ,)
        sg_base = bh * n_quant * n_groups
        scale = tl.load(v_scale_ptr + sg_base + rows * n_groups + g,
                        mask=bq_mask, other=0.0).to(tl.float32)
        mn    = tl.load(v_mn_ptr    + sg_base + rows * n_groups + g,
                        mask=bq_mask, other=0.0).to(tl.float32)

        # 2-D gather of packed int32s: shape (BLOCK_BQ, group_size).
        # rows[:, None] * packed_dim : base of each token's row in packed storage.
        # g * ints_per_group          : start of this group within the row.
        # j_vals[None, :]             : which packed int for each output element.
        pq_offs = (bh * n_quant * packed_dim
                   + rows[:, None] * packed_dim
                   + g * ints_per_group
                   + j_vals[None, :])                                         # (BLOCK_BQ, group_size)
        gathered = tl.load(v_quant_ptr + pq_offs,
                           mask=bq_mask[:, None], other=0)                    # (BLOCK_BQ, group_size) int32

        # Unpack bits and dequantize.
        vals    = ((gathered >> (k_vals[None, :] * bits)) & mask_int).to(tl.float32)
        dequant = vals * scale[:, None] + mn[:, None]                         # (BLOCK_BQ, group_size)

        # Weighted reduction over budget tile → (group_size,) partial sum.
        partial = tl.sum(dequant * ws[:, None], axis=0)                       # (group_size,)

        # Accumulate into global fp32 output via atomic_add.
        out_offs = bh * head_dim + g * group_size + tl.arange(0, group_size)
        tl.atomic_add(out_ptr + out_offs, partial)


def sparse_weighted_sum_quant(
    v_quant:    torch.Tensor,   # (B, nh, n_quant, packed_dim) int32
    v_scale:    torch.Tensor,   # (B, nh, n_quant, n_groups)   fp16
    v_mn:       torch.Tensor,   # (B, nh, n_quant, n_groups)   fp16
    attn_w:     torch.Tensor,   # (B, nh, budget_q)            fp16
    quant_idx:  torch.Tensor,   # (budget_q,)                  int64/int32
    bits:       int,
    group_size: int,
    BLOCK_BQ:   int = 32,
) -> torch.Tensor:
    """
    Fused sparse gather + unpack + dequant + weighted sum.

    Grid (BH, ceil(budget_q / BLOCK_BQ)) — budget parallelised across programs.
    Each program atomic-adds a partial sum; output is converted fp32 → fp16.

    Returns: (B, nh, head_dim) float16.
    """
    B, nh, n_quant, packed_dim = v_quant.shape
    BH       = B * nh
    budget_q = int(quant_idx.shape[0])
    fps      = 32 // bits
    head_dim = packed_dim * fps
    n_groups       = head_dim  // group_size
    ints_per_group = group_size // fps

    assert group_size % fps == 0,                   "group_size must be divisible by fps (32 // bits)"
    assert (group_size & (group_size - 1)) == 0,    "group_size must be a power of 2"
    assert (BLOCK_BQ  & (BLOCK_BQ  - 1)) == 0,      "BLOCK_BQ must be a power of 2"

    v_quant_f = v_quant.reshape(BH, n_quant, packed_dim).contiguous()
    v_scale_f = v_scale.reshape(BH, n_quant, n_groups).contiguous()
    v_mn_f    = v_mn.reshape(   BH, n_quant, n_groups).contiguous()
    attn_w_f  = attn_w.reshape( BH, budget_q).contiguous()
    idx_i32   = quant_idx.to(torch.int32).contiguous()

    # fp32 accumulator (atomic_add requires float32 for correctness)
    out_f32 = torch.zeros(BH, head_dim, dtype=torch.float32, device=v_quant.device)

    n_budget_tiles = triton.cdiv(budget_q, BLOCK_BQ)
    grid = (BH, n_budget_tiles)

    _sparse_ws_kernel[grid](
        v_quant_f, v_scale_f, v_mn_f,
        attn_w_f, idx_i32, out_f32,
        n_quant, budget_q, packed_dim,
        n_groups=n_groups,
        group_size=group_size,
        ints_per_group=ints_per_group,
        fps=fps,
        bits=bits,
        head_dim=head_dim,
        BLOCK_BQ=BLOCK_BQ,
        num_warps=BLOCK_BQ // 32,
    )

    return out_f32.to(torch.float16).reshape(B, nh, head_dim)
