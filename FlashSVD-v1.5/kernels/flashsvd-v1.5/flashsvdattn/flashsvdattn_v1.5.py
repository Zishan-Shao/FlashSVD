#!/usr/bin/env python3
# flashsvdattn_v1.5.py – rank-aware Flash-SVD attention (mask-friendly, optimized)
#
# Optimized kernel: V-portion in SRAM, iterate over P tiles → reduces V loads, higher throughput.
# Run from repo root with PYTHONPATH=. or add repo to path.

import sys
import os
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Use optimized kernel (V-portion reuse) - disabled: Triton/LLVM issues with loop structure
_USE_OPTIMIZED_KERNEL = False


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _pad_to_multiple(x: int, m: int) -> int:
    return m * _ceil_div(x, m)

_ST_CACHE: dict[tuple[int, int, int, torch.dtype, torch.device, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


# ────────────────────────────────────────────────────────────────────
# v1.6-style kernel: rank-space scores + rank-space V accumulate + single lift
# scores: (Pq @ (Vq @ Vk^T) + (bq @ Vk^T)) @ Pk^T
# out:    softmax(scores) @ Pv @ Vv + bv
# Notes:
#   - bk only adds a per-query constant shift to scores -> softmax-invariant, safe to ignore.
#   - bv can be added after softmax since row-sum(p)=1 for valid queries.
# ────────────────────────────────────────────────────────────────────
@triton.jit
def _flashsvdattn_v16_rank_kernel(
    # P factors (BHMR): [B,H,M,R]
    Pq_ptr,
    Pk_ptr,
    Pv_ptr,
    # precomputed per-head terms
    #   S = Vq @ Vk^T  [H,R,R]
    #   t = bq @ Vk^T  [H,R]
    S_ptr,
    t_ptr,
    # value factors
    Vv_ptr,  # [H,R,D]
    bv_ptr,  # [H,D]
    # output [B*H,M,D]
    Out_ptr,
    # mask [B,H,1,M] (int32/bool, True=valid)
    mask_ptr,
    sMb,
    sMh,
    sMq,
    sMk,
    # Pq strides
    sQb,
    sQh,
    sQm,
    sQr,
    # Pk strides
    sKb,
    sKh,
    sKn,
    sKr,
    # Pv strides
    sVb2,
    sVh2,
    sVn2,
    sVr2,
    # S strides [H,R,R]
    sSh,
    sSr0,
    sSr1,
    # t strides [H,R]
    sTh,
    sTr,
    # Vv strides [H,R,D]
    sVvh,
    sVvr,
    sVvd,
    # bv strides [H,D]
    sBvh,
    sBvd,
    # Out strides [B*H,M,D]
    sOb,
    sOh,
    sOm,
    # sizes/meta
    seqlen,
    r_dim,
    d_dim,
    nheads,
    softmax_scale,
    # constexprs
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_h = off_bh % nheads
    row_off = pid_m * BLOCK_M

    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)

    # Per-query padding mask
    pad_q_ptrs = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_m * sMk
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m < seqlen, other=0).to(tl.int32)
    pad_q = pad_q_i > 0

    # Load Pq tile [BM, R]
    Pq_ptrs = (
        Pq_ptr
        + off_b * sQb
        + off_h * sQh
        + offs_m[:, None] * sQm
        + offs_r[None, :] * sQr
    )
    mask_q = (offs_m[:, None] < seqlen) & (offs_r[None, :] < r_dim)
    Pq_blk = tl.load(Pq_ptrs, mask=mask_q, other=0.0)

    # Load S = Vq @ Vk^T  [R, R] for this head
    S_ptrs = S_ptr + off_h * sSh + offs_r[:, None] * sSr0 + offs_r[None, :] * sSr1
    mask_s = (offs_r[:, None] < r_dim) & (offs_r[None, :] < r_dim)
    S_blk = tl.load(S_ptrs, mask=mask_s, other=0.0)

    # A = Pq @ S + t, then zero-out padded queries
    A = tl.dot(Pq_blk, S_blk, out_dtype=tl.float32)
    t_ptrs = t_ptr + off_h * sTh + offs_r * sTr
    t_blk = tl.load(t_ptrs, mask=offs_r < r_dim, other=0.0).to(tl.float32)
    A = A + t_blk[None, :]
    A = A * pad_q[:, None].to(tl.float32)
    A = A.to(Pq_blk.dtype)

    # Online softmax accumulators (rank-space)
    acc_r = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Iterate over K/V blocks
    for start_n in range(0, seqlen, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_abs = start_n + offs_n

        # key validity mask from [B,H,1,M] (broadcast over query dim)
        mask_ptrs = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_n_abs * sMk
        mask_i32 = tl.load(mask_ptrs, mask=offs_n_abs < seqlen, other=0).to(tl.int32)
        key_ok = mask_i32 > 0

        # Load Pk, Pv tiles [BN, R]
        Pk_ptrs = (
            Pk_ptr
            + off_b * sKb
            + off_h * sKh
            + offs_n_abs[:, None] * sKn
            + offs_r[None, :] * sKr
        )
        Pv_ptrs = (
            Pv_ptr
            + off_b * sVb2
            + off_h * sVh2
            + offs_n_abs[:, None] * sVn2
            + offs_r[None, :] * sVr2
        )
        mask_kv = (offs_n_abs[:, None] < seqlen) & (offs_r[None, :] < r_dim)
        Pk_blk = tl.load(Pk_ptrs, mask=mask_kv, other=0.0)
        Pv_blk = tl.load(Pv_ptrs, mask=mask_kv, other=0.0)

        # scores = A @ Pk^T  (rank-space), then apply masks
        qk = tl.dot(A, tl.trans(Pk_blk)).to(tl.float32) * softmax_scale
        if CAUSAL:
            causal = offs_m[:, None] >= offs_n_abs[None, :]
            keep = key_ok[None, :] & causal
        else:
            keep = key_ok[None, :]
        qk = tl.where(keep, qk, tl.full(qk.shape, -float("inf"), tl.float32))

        # online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * exp_diff + tl.sum(p, axis=1)
        acc_r = acc_r * exp_diff[:, None] + tl.dot(p.to(A.dtype), Pv_blk, out_dtype=tl.float32)
        m_i = m_new

    den = tl.where(l_i > 0, l_i, 1.0)
    out_r = acc_r / den[:, None]

    # lift once: out = out_r @ Vv + bv
    Vv_ptrs = Vv_ptr + off_h * sVvh + offs_r[:, None] * sVvr + offs_d[None, :] * sVvd
    mask_vv = (offs_r[:, None] < r_dim) & (offs_d[None, :] < d_dim)
    Vv_blk = tl.load(Vv_ptrs, mask=mask_vv, other=0.0)
    out = tl.dot(out_r.to(A.dtype), Vv_blk, out_dtype=tl.float32)

    bv_ptrs = bv_ptr + off_h * sBvh + offs_d * sBvd
    bv_blk = tl.load(bv_ptrs, mask=offs_d < d_dim, other=0.0).to(tl.float32)
    out = out + bv_blk[None, :]
    out = out * pad_q[:, None].to(tl.float32)

    out_ptrs = Out_ptr + off_bh * sOb + offs_m[:, None] * sOh + offs_d[None, :] * sOm
    tl.store(out_ptrs, out.to(A.dtype), mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < d_dim))

# ────────────────────────────────────────────────────────────────────
# Optimized kernel: V-portion in outer loop, traverse P (key blocks)
# Reduces V loads from O(num_key_blocks * R/BR) to O(R/BR) per key group
# ────────────────────────────────────────────────────────────────────


@triton.jit
def _flashsvdattn_v15_kernel(
    Pq_ptr, Vq_ptr, bq_ptr,
    Pk_ptr, Vk_ptr, bk_ptr,
    Pv_ptr, Vv_ptr, bv_ptr,
    Out_ptr,
    mask_ptr, sMb, sMh, sMq, sMk,
    sQb, sQh, sQm, sQr,
    sVqb, sVqh, sVqr, sVqd,
    sBqb, sBqh, sBqd,
    sKb, sKh, sKn, sKr,
    sVkb, sVkh, sVkr, sVkd,
    sBkb, sBkh, sBkd,
    sVb2, sVh2, sVn2, sVr2,
    sVvb, sVvh, sVvr, sVvd,
    sBvb, sBvh, sBvd,
    sOb, sOh, sOm,
    seqlen, r_dim, nheads, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // nheads
    off_h = off_bh % nheads
    row_off = start_m * BLOCK_M
    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    r_idx = tl.arange(0, BLOCK_R)

    # Query padding mask
    pad_q_ptrs = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_m * sMk
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m < seqlen, other=0).to(tl.int32)
    pad_q = pad_q_i > 0

    # 1) Q tile (baseline: one block, V reuse less critical)
    q = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for r_start in range(0, r_dim, BLOCK_R):
        mask_r = (r_start + r_idx) < r_dim
        Pq_ptrs = Pq_ptr + off_b * sQb + off_h * sQh + (row_off + tl.arange(0, BLOCK_M))[:, None] * sQm + (r_start + r_idx)[None, :] * sQr
        Vq_ptrs = Vq_ptr + off_b * sVqb + off_h * sVqh + (r_start + r_idx)[:, None] * sVqr + offs_d[None, :] * sVqd
        Pq_sub = tl.load(Pq_ptrs, mask=(offs_m[:, None] < seqlen) & mask_r[None, :], other=0.0).to(tl.float32)
        Vq_sub = tl.load(Vq_ptrs, mask=mask_r[:, None], other=0.0).to(tl.float32)
        q += tl.dot(Pq_sub, Vq_sub)
    bq_ptrs = bq_ptr + off_b * sBqb + off_h * sBqh + offs_d * sBqd
    q += tl.load(bq_ptrs).to(tl.float32)[None, :]
    q = q * pad_q[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # V-portion in outer loop, traverse P (key blocks) - process 2 key blocks per V load
    num_key_blocks = (seqlen + BLOCK_M - 1) // BLOCK_M
    key_group_start = 0
    while key_group_start < num_key_blocks:
        k_g0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        v_g0 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        k_g1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        v_g1 = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for r_start in range(0, r_dim, BLOCK_R):
            mask_r = (r_start + r_idx) < r_dim
            Vk_ptrs = Vk_ptr + off_b * sVkb + off_h * sVkh + (r_start + r_idx)[:, None] * sVkr + offs_d[None, :] * sVkd
            Vv_ptrs = Vv_ptr + off_b * sVvb + off_h * sVvh + (r_start + r_idx)[:, None] * sVvr + offs_d[None, :] * sVvd
            Vk_sub = tl.load(Vk_ptrs, mask=mask_r[:, None], other=0.0).to(tl.float32)
            Vv_sub = tl.load(Vv_ptrs, mask=mask_r[:, None], other=0.0).to(tl.float32)

            # g=0
            kb0 = key_group_start
            start_n0 = kb0 * BLOCK_M
            offs_n0 = tl.arange(0, BLOCK_M)
            Pk_ptrs0 = Pk_ptr + off_b * sKb + off_h * sKh + (start_n0 + offs_n0)[:, None] * sKn + (r_start + r_idx)[None, :] * sKr
            Pv_ptrs0 = Pv_ptr + off_b * sVb2 + off_h * sVh2 + (start_n0 + offs_n0)[:, None] * sVn2 + (r_start + r_idx)[None, :] * sVr2
            Pk_sub0 = tl.load(Pk_ptrs0, mask=(offs_n0[:, None] < seqlen) & mask_r[None, :], other=0.0).to(tl.float32)
            Pv_sub0 = tl.load(Pv_ptrs0, mask=(offs_n0[:, None] < seqlen) & mask_r[None, :], other=0.0).to(tl.float32)
            k_g0 += tl.dot(Pk_sub0, Vk_sub)
            v_g0 += tl.dot(Pv_sub0, Vv_sub)

            # g=1
            kb1 = key_group_start + 1
            start_n1 = kb1 * BLOCK_M
            offs_n1 = tl.arange(0, BLOCK_M)
            Pk_ptrs1 = Pk_ptr + off_b * sKb + off_h * sKh + (start_n1 + offs_n1)[:, None] * sKn + (r_start + r_idx)[None, :] * sKr
            Pv_ptrs1 = Pv_ptr + off_b * sVb2 + off_h * sVh2 + (start_n1 + offs_n1)[:, None] * sVn2 + (r_start + r_idx)[None, :] * sVr2
            Pk_sub1 = tl.load(Pk_ptrs1, mask=(offs_n1[:, None] < seqlen) & mask_r[None, :], other=0.0).to(tl.float32)
            Pv_sub1 = tl.load(Pv_ptrs1, mask=(offs_n1[:, None] < seqlen) & mask_r[None, :], other=0.0).to(tl.float32)
            k_g1 += tl.dot(Pk_sub1, Vk_sub)
            v_g1 += tl.dot(Pv_sub1, Vv_sub)

        bk_ptrs = bk_ptr + off_b * sBkb + off_h * sBkh + offs_d * sBkd
        bv_ptrs = bv_ptr + off_b * sBvb + off_h * sBvh + offs_d * sBvd
        k_g0 += tl.load(bk_ptrs).to(tl.float32)[None, :]
        v_g0 += tl.load(bv_ptrs).to(tl.float32)[None, :]
        k_g1 += tl.load(bk_ptrs).to(tl.float32)[None, :]
        v_g1 += tl.load(bv_ptrs).to(tl.float32)[None, :]

        # Merge block 0
        start_n0 = key_group_start * BLOCK_M
        offs_n0 = tl.arange(0, BLOCK_M)
        mask_ptrs0 = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + (start_n0 + offs_n0) * sMk
        mask_i32_0 = tl.load(mask_ptrs0, mask=offs_n0 < seqlen, other=0).to(tl.int32)
        mask_vals0 = mask_i32_0 > 0
        causal0 = offs_m[:, None] >= (start_n0 + offs_n0)[None, :]
        key_mask0 = mask_vals0[None, :] & causal0
        qk0 = tl.dot(q, tl.trans(k_g0)) * softmax_scale
        qk0 = tl.where(key_mask0, qk0, tl.full(qk0.shape, float("-inf"), dtype=tl.float32))
        m_new0 = tl.maximum(m_i, tl.max(qk0, axis=1))
        exp_diff0 = tl.exp(m_i - m_new0)
        p0 = tl.exp(qk0 - m_new0[:, None])
        l_i = l_i * exp_diff0 + tl.sum(p0, axis=1)
        acc = acc * exp_diff0[:, None] + tl.dot(p0, v_g0)
        m_i = m_new0

        # Merge block 1 (when invalid, key_mask1 masks to -inf so p1=0, no effect)
        start_n1 = (key_group_start + 1) * BLOCK_M
        offs_n1 = tl.arange(0, BLOCK_M)
        mask_ptrs1 = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + (start_n1 + offs_n1) * sMk
        mask_i32_1 = tl.load(mask_ptrs1, mask=offs_n1 < seqlen, other=0).to(tl.int32)
        mask_vals1 = mask_i32_1 > 0
        causal1 = offs_m[:, None] >= (start_n1 + offs_n1)[None, :]
        key_mask1 = mask_vals1[None, :] & causal1
        qk1 = tl.dot(q, tl.trans(k_g1)) * softmax_scale
        qk1 = tl.where(key_mask1, qk1, tl.full(qk1.shape, float("-inf"), dtype=tl.float32))
        m_new1 = tl.maximum(m_i, tl.max(qk1, axis=1))
        exp_diff1 = tl.exp(m_i - m_new1)
        p1 = tl.exp(qk1 - m_new1[:, None])
        l_i = l_i * exp_diff1 + tl.sum(p1, axis=1)
        acc = acc * exp_diff1[:, None] + tl.dot(p1, v_g1)
        m_i = m_new1

        key_group_start += 2

    den = tl.where(l_i > 0, l_i, 1.0)
    out = acc / den[:, None]
    out = out * pad_q[:, None]
    out_ptrs = Out_ptr + off_bh * sOb + offs_m[:, None] * sOh + offs_d[None, :] * sOm
    tl.store(out_ptrs, out, mask=offs_m[:, None] < seqlen)


# ────────────────────────────────────────────────────────────────────
# 1. Triton wrapper - v1.5 uses same tiles as v1, with num_warps/stages tuning
# ────────────────────────────────────────────────────────────────────
BLOCK_M = 64
BLOCK_R = 64

def _contig(t): return t.contiguous() if not t.is_contiguous() else t
def _contig_last_dim(t): return t.contiguous() if t.stride(-1) != 1 else t

def flash_svd_attention(Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask,
                        *, causal: bool = False, block_m=BLOCK_M, block_n=None, block_r=BLOCK_R,
                        num_warps: int = 4, num_stages: int = 2):
    B,H,M,R = Pq.shape
    D       = Vq.shape[-1]
    scale   = 1.0/math.sqrt(D)
    if block_n is None:
        block_n = block_m
    block_m = int(block_m)
    block_n = int(block_n)
    if block_m < 16 or block_n < 16:
        raise ValueError(f"block_m/block_n must be >= 16, got block_m={block_m}, block_n={block_n}")

    # P factors may be strided views (e.g., fused QKV rank projection); only require rank dim contiguous
    Pq, Pk, Pv = map(_contig_last_dim, (Pq, Pk, Pv))
    if int(block_r) != R:
        raise ValueError(f"block_r must match Pq.shape[-1] (got block_r={block_r}, R={R})")

    # Prefer weight tensors without batch expansion: [H,R,D] and [H,D]
    if Vq.ndim == 4:
        if Vq.shape[0] != 1:
            raise ValueError(f"Vq must be [H,R,D] or [1,H,R,D], got {tuple(Vq.shape)}")
        Vq = Vq[0]
    if Vk.ndim == 4:
        if Vk.shape[0] != 1:
            raise ValueError(f"Vk must be [H,R,D] or [1,H,R,D], got {tuple(Vk.shape)}")
        Vk = Vk[0]
    if Vv.ndim == 4:
        if Vv.shape[0] != 1:
            raise ValueError(f"Vv must be [H,R,D] or [1,H,R,D], got {tuple(Vv.shape)}")
        Vv = Vv[0]
    if bq.ndim == 3:
        if bq.shape[0] != 1:
            raise ValueError(f"bq must be [H,D] or [1,H,D], got {tuple(bq.shape)}")
        bq = bq[0]
    if bv.ndim == 3:
        if bv.shape[0] != 1:
            raise ValueError(f"bv must be [H,D] or [1,H,D], got {tuple(bv.shape)}")
        bv = bv[0]

    Vq, Vk, Vv = map(_contig, (Vq, Vk, Vv))
    bq, bv = map(_contig, (bq, bv))
    if Vq.shape != (H, R, D) or Vk.shape != (H, R, D) or Vv.shape != (H, R, D):
        raise ValueError(f"expected Vq/Vk/Vv to be [H,R,D]=[{H},{R},{D}], got Vq={tuple(Vq.shape)} Vk={tuple(Vk.shape)} Vv={tuple(Vv.shape)}")
    if bq.shape != (H, D) or bv.shape != (H, D):
        raise ValueError(f"expected bq/bv to be [H,D]=[{H},{D}], got bq={tuple(bq.shape)} bv={tuple(bv.shape)}")

    base = mask if mask.ndim==4 else mask[:, :1, :].unsqueeze(2)
    m4   = base.to(torch.int32).expand(B,H,1,M)
    if m4.stride(1) or m4.stride(2):
        m4 = m4.as_strided(m4.size(), (m4.stride(0),0,0,m4.stride(3)))
    sMb,sMh,sMq,sMk = m4.stride()

    # Precompute per-head rank-space terms:
    #   S[h] = Vq[h] @ Vk[h].T   [R,R]
    #   t[h] = bq[h] @ Vk[h].T   [R]
    # (bk is softmax-invariant -> ignore)
    global _ST_CACHE
    if torch.is_grad_enabled():
        S = (Vq @ Vk.transpose(-1, -2)).contiguous()
        t = (bq.unsqueeze(1) @ Vk.transpose(-1, -2)).squeeze(1).contiguous()
    else:
        key = (Vq.data_ptr(), Vk.data_ptr(), bq.data_ptr(), Vq.dtype, Vq.device, Vq.shape[1], Vq.shape[2])
        cached = _ST_CACHE.get(key)
        if cached is None:
            with torch.no_grad():
                S = (Vq @ Vk.transpose(-1, -2)).contiguous()
                t = (bq.unsqueeze(1) @ Vk.transpose(-1, -2)).squeeze(1).contiguous()
            _ST_CACHE[key] = (S, t)
        else:
            S, t = cached

    # Output in input dtype (avoid extra fp32->fp16 cast kernel)
    Out = torch.empty((B * H, M, D), device=Pq.device, dtype=Pq.dtype)

    # Strides
    sQb, sQh, sQm, sQr = Pq.stride()
    sKb, sKh, sKn, sKr = Pk.stride()
    sVb2, sVh2, sVn2, sVr2 = Pv.stride()
    sSh, sSr0, sSr1 = S.stride()
    sTh, sTr = t.stride()
    sVvh, sVvr, sVvd = Vv.stride()
    sBvh, sBvd = bv.stride()
    sOb, sOh, sOm = Out.stride()

    # Tile sizes (pad rank & head-dim to tensorcore-friendly multiples)
    block_r_eff = int(block_r)
    block_d_eff = int(_pad_to_multiple(D, 16))
    block_r_pad = int(_pad_to_multiple(block_r_eff, 16))

    grid = (triton.cdiv(M, block_m), B * H)
    _flashsvdattn_v16_rank_kernel[grid](
        Pq,
        Pk,
        Pv,
        S,
        t,
        Vv,
        bv,
        Out,
        m4,
        sMb=sMb, sMh=sMh, sMq=sMq, sMk=sMk,
        sQb=sQb, sQh=sQh, sQm=sQm, sQr=sQr,
        sKb=sKb, sKh=sKh, sKn=sKn, sKr=sKr,
        sVb2=sVb2, sVh2=sVh2, sVn2=sVn2, sVr2=sVr2,
        sSh=sSh, sSr0=sSr0, sSr1=sSr1,
        sTh=sTh, sTr=sTr,
        sVvh=sVvh, sVvr=sVvr, sVvd=sVvd,
        sBvh=sBvh, sBvd=sBvd,
        sOb=sOb, sOh=sOh, sOm=sOm,
        seqlen=M,
        r_dim=block_r_eff,
        d_dim=D,
        nheads=H,
        softmax_scale=scale,
        CAUSAL=int(causal),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r_pad,
        BLOCK_D=block_d_eff,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return Out.view(B, H, M, D)

# ────────────────────────────────────────────────────────────────────
# 2. Flash-SVD block
# ────────────────────────────────────────────────────────────────────
class FlashSVDBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rank: int,
        d_ff: int,
        *,
        attn_block_m: int = BLOCK_M,
        attn_block_n: int | None = None,
        attn_num_warps: int = 4,
        attn_num_stages: int = 2,
    ):
        super().__init__()
        assert d_model % n_heads == 0 and rank<=d_model//n_heads
        self.d_model,self.H,self.R = d_model,n_heads,rank
        self.dh = d_model//n_heads
        self.attn_block_m = int(attn_block_m)
        self.attn_block_n = int(attn_block_n) if attn_block_n is not None else int(attn_block_m)
        self.attn_num_warps = int(attn_num_warps)
        self.attn_num_stages = int(attn_num_stages)

        def mk():
            P = nn.Parameter(torch.randn(1,n_heads,d_model,rank)*0.02)
            V = nn.Parameter(torch.randn(1,n_heads,rank,self.dh)*0.02)
            b = nn.Parameter(torch.zeros (1,n_heads,       self.dh))
            return P,V,b
        self.Pq,self.Vq,self.bq = mk()
        self.Pk,self.Vk,self.bk = mk()
        self.Pv,self.Vv,self.bv = mk()

        self.proj_out = nn.Linear(d_model,d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff), nn.GELU(), nn.Linear(d_ff,d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask, *, causal: bool = False):
        B,M,_ = x.shape
        H,R,dh = self.H,self.R,self.dh

        # Fuse Q/K/V rank projections into one GEMM:
        #   [B*M, H*3R] = x2d @ W_qkv, where W_qkv is [H*3R, d_model]
        # Avoid (B,H)-batched GEMMs with tiny N=3R (slow when R is small).
        if torch.is_grad_enabled():
            P_qkv = torch.cat((self.Pq, self.Pk, self.Pv), dim=-1)[0]  # [H,d_model,3R]
            W_qkv = P_qkv.permute(0, 2, 1).reshape(H * 3 * R, self.d_model).contiguous()
        else:
            W_qkv = getattr(self, "_W_qkv_cache", None)
            if W_qkv is None or W_qkv.dtype != x.dtype or W_qkv.device != x.device:
                P_qkv = torch.cat((self.Pq, self.Pk, self.Pv), dim=-1)[0]
                W_qkv = P_qkv.permute(0, 2, 1).reshape(H * 3 * R, self.d_model).contiguous()
                self._W_qkv_cache = W_qkv

        x2d = x.reshape(B * M, self.d_model)
        tmp2d = F.linear(x2d, W_qkv)  # [B*M, H*3R]
        tmp_qkv = tmp2d.view(B, M, H, 3, R).permute(0, 2, 1, 3, 4)  # [B,H,M,3,R]
        tmp_q = tmp_qkv[:, :, :, 0, :]
        tmp_k = tmp_qkv[:, :, :, 1, :]
        tmp_v = tmp_qkv[:, :, :, 2, :]

        attn = flash_svd_attention(
            tmp_q, self.Vq[0], self.bq[0],
            tmp_k, self.Vk[0], self.bk[0],
            tmp_v, self.Vv[0], self.bv[0],
            mask=mask, block_r=R, causal=causal,
            block_m=self.attn_block_m, block_n=self.attn_block_n,
            num_warps=self.attn_num_warps, num_stages=self.attn_num_stages,
        )                       # [B,H,M,dh]
        attn = attn.transpose(1,2).reshape(B,M,self.d_model)
        y = self.ln1(x + self.proj_out(attn))
        return self.ln2(y + self.ffn(y))

# ────────────────────────────────────────────────────────────────────
# 3. Dense baseline
# ────────────────────────────────────────────────────────────────────
class BaselineBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model,n_heads,batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff), nn.GELU(), nn.Linear(d_ff,d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        pad = ~mask.squeeze(1).squeeze(1)
        attn, _ = self.mha(x, x, x, key_padding_mask=pad, need_weights=False)
        y = self.ln1(x + attn)
        return self.ln2(y + self.ffn(y))

# ────────────────────────────────────────────────────────────────────
# 4. Rank-aware transplant  (slice **rows**, not columns!)
# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def transplant_weights(dense: BaselineBlock, flash: FlashSVDBlock):
    Wqkv = dense.mha.in_proj_weight    # [3d_model, d_model]
    bqkv = dense.mha.in_proj_bias
    Wo,bo = dense.mha.out_proj.weight, dense.mha.out_proj.bias

    d_model, H = flash.d_model, flash.H
    dh,R = flash.dh, flash.R

    def W_head(p,h):  # rows for this head, then transpose → [d_model, dh]
        rows = slice(p*d_model + h*dh, p*d_model + (h+1)*dh)
        return Wqkv[rows, :].t().contiguous()

    def b_head(p,h):
        rows = slice(p*d_model + h*dh, p*d_model + (h+1)*dh)
        return bqkv[rows]

    for p,(P,V,b) in enumerate([(flash.Pq,flash.Vq,flash.bq),
                                (flash.Pk,flash.Vk,flash.bk),
                                (flash.Pv,flash.Vv,flash.bv)]):  # Q,K,V
        for h in range(H):
            W = W_head(p,h).float()               # [d_model, dh]
            U,S,Vt = torch.linalg.svd(W, full_matrices=False)
            P[0,h].copy_((U[:, :R] * S[:R]).to(P.dtype))  # U·Σ
            V[0,h].copy_(Vt[:R, :].to(V.dtype))           # [R, dh]
            b[0,h].copy_(b_head(p,h).to(b.dtype))

    flash.proj_out.weight.copy_(Wo); flash.proj_out.bias.copy_(bo)
    flash.ffn[0].weight.copy_(dense.ffn[0].weight)
    flash.ffn[0].bias  .copy_(dense.ffn[0].bias)
    flash.ffn[2].weight.copy_(dense.ffn[2].weight)
    flash.ffn[2].bias  .copy_(dense.ffn[2].bias)
    flash.ln1.load_state_dict(dense.ln1.state_dict())
    flash.ln2.load_state_dict(dense.ln2.state_dict())

# ────────────────────────────────────────────────────────────────────
# 5. Quick test
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    B,M = 8,128
    d_model,H,d_ff = 768,12,3072
    x = torch.randn(B,M,d_model, device=dev, dtype=torch.float16)

    mask4 = torch.zeros(B,1,1,M,device=dev,dtype=torch.bool)
    mask4[..., :96] = True           # any true length

    dense = BaselineBlock(d_model,H,d_ff).to(dev).half()

    for R in [64, 54, 48, 32, 28, 16]:
        flash = FlashSVDBlock(d_model,H,R,d_ff).to(dev).half()
        transplant_weights(dense, flash)

        with torch.no_grad():
            yd = dense (x, mask4).float()
            yf = flash (x, mask4).float()
        rel = (yf - yd).norm() / yd.norm()
        print(f"rank {R:2d}  → rel-err {rel:.4e}")
        
    # pick one projection (0=Q, 1=K, 2=V) and one head h
    proj, head = 0, 0               # change if you like

    Wqkv = dense.mha.in_proj_weight  # [3·d_model, d_model]
    d_model, H = flash.d_model, flash.H
    dh = d_model // H

    # rows that belong to this (proj, head) pair
    rows = slice(proj*d_model + head*dh, proj*d_model + (head+1)*dh)
    W = Wqkv[rows, :].t().contiguous().float()   # shape [d_model, dh]

    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    for R in [64, 32, 16, 1]:
        W_hat = (U[:, :R] @ torch.diag(S[:R]) @ Vt[:R]).to(W.dtype)
        rel_w = (W - W_hat).norm() / W.norm()
        print(f"W  rank {R:2d} → weight rel-err {rel_w:.2f}")
