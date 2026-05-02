# Adapted from KIVI: https://github.com/jy-yuan/KIVI
# cuda_bmm_fA_qB_outer uses the compiled kivi_gemv CUDA extension.
# triton_bmm_fA_qB_outer requires group_size % 64 == 0 and is NOT used (incompatible with group_size=32).
import torch
import triton
import triton.language as tl


@triton.jit
def qbvm_kernel(
    bits,
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    M, N, K,
    stride_abatch, stride_am, stride_ak,
    stride_bbatch, stride_bk, stride_bn,
    stride_cbatch, stride_cm, stride_cn,
    stride_scales_b, stride_scales_k, stride_scales_g,
    stride_zeros_b, stride_zeros_k, stride_zeros_g,
    groupsize,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    feat_per_int = 32 // bits
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    pid_n = pid % num_pid_n
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_batch_offset = pid_batch * stride_abatch
    b_batch_offset = pid_batch * stride_bbatch
    c_batch_offset = pid_batch * stride_cbatch
    a_ptr = a_ptr + a_batch_offset
    b_ptr = b_ptr + b_batch_offset
    c_ptr = c_ptr + c_batch_offset
    a_ptrs = a_ptr + (offs_k[:, None] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + (offs_bn[None, :] // feat_per_int) * stride_bn)
    shifter = (offs_bn % feat_per_int) * bits
    scales_ptr = scales_ptr + pid_batch * stride_scales_b + ((offs_bn[None, :] // groupsize)) * stride_scales_g
    zeros_ptr = zeros_ptr + pid_batch * stride_zeros_b + ((offs_bn[None, :] // groupsize)) * stride_zeros_g
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    num = 0xFF >> (8 - bits)
    for pid_k in range(0, num_pid_k):
        offs_bk = (offs_k[:, None] + pid_k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=offs_bk < K, other=0.)
        b = tl.load(b_ptrs, mask=offs_bk < K, other=0.)
        ptr = scales_ptr + offs_bk * stride_scales_k
        scales = tl.load(ptr, mask=offs_bk < K, other=0.)
        ptr = zeros_ptr + offs_bk * stride_zeros_k
        zeros = tl.load(ptr, mask=offs_bk < K, other=0.)
        b = (b >> shifter[None, :]) & num
        b = b * scales + zeros
        accumulator += tl.sum(a * b, 0)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cn * offs_cn
    c_mask = offs_cn < N
    tl.store(c_ptrs, c, mask=c_mask)


def cuda_bmm_fA_qB_outer(
    group_size: int,
    fA: torch.FloatTensor,
    qB: torch.IntTensor,
    scales: torch.FloatTensor,
    zeros: torch.FloatTensor,
    bits: int,
) -> torch.FloatTensor:
    """
    Q @ K^T where K is quantized (outer-dim grouping, per-channel quant).

    fA    : (B, nh, M, K)   float16
    qB    : (B, nh_kv, K, N//feat_per_int)   int32
    scales: (B, nh_kv, K, G)  float16   G = N // group_size
    zeros : (B, nh_kv, K, G)  float16
    Returns (B, nh, M, N) float16
    """
    import kivi_gemv  # compiled CUDA extension

    assert len(fA.shape) == 4 and len(qB.shape) == 4
    B, nh, M, K = fA.shape
    nh_kv = qB.shape[1]
    feat_per_int = 32 // bits
    fA = fA.view(-1, M, K).contiguous()
    N = qB.shape[-1] * feat_per_int
    qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
    flatten_B = B * nh_kv
    scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
    zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
    assert bits in [2, 4]
    assert nh % nh_kv == 0
    c = kivi_gemv.gemv_forward_cuda_outer_dim(fA, qB, scales, zeros, bits, group_size, nh, nh_kv)
    return c.view(B, nh, c.shape[-2], c.shape[-1])
