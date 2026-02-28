#!/usr/bin/env python3
# flashsvdattn_bwd.py — rank-aware Flash-SVD attention (mask-friendly)
# Triton 3.3.1 compatible

import math, time
import torch
from torch.utils.checkpoint import checkpoint as checkpoint_fn
import triton
import triton.language as tl

# -----------------------------
# Tunables / defaults
# -----------------------------
BLOCK_M = 64
BLOCK_R = 64

def _contig(t): return t.contiguous() if not t.is_contiguous() else t

# -----------------------------
# Tile loader:  (P @ V + b) over R in BLOCK_R chunks
# -----------------------------
@triton.jit
def load_tiles(
    P_ptr, V_ptr, bias_ptr,
    sPb, sPh, sPm, sPr,
    sVb, sVh, sVr, sVd,
    sBb, sBh, sBd,
    BLOCK_X: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
    full_len, r_dim, off_b, off_h, row_offset,
):
    offs_x = tl.arange(0, BLOCK_X)
    offs_d = tl.arange(0, BLOCK_D)
    r_idx  = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_X, BLOCK_D), dtype=tl.float32)
    for r_start in range(0, r_dim, BLOCK_R):
        mask_r = (r_start + r_idx) < r_dim
        P_ptrs = (
            P_ptr + off_b*sPb + off_h*sPh
                  + (row_offset+offs_x)[:,None]*sPm
                  + (r_start+r_idx)[None,:]*sPr
        )
        V_ptrs = (
            V_ptr + off_b*sVb + off_h*sVh
                  + (r_start+r_idx)[:,None]*sVr
                  + offs_d[None,:]*sVd
        )
        P_sub = tl.load(P_ptrs, mask=mask_r[None,:], other=0.).to(tl.float32)
        V_sub = tl.load(V_ptrs, mask=mask_r[:,None], other=0.).to(tl.float32)
        acc += tl.dot(P_sub, V_sub)
    b_ptrs = bias_ptr + off_b*sBb + off_h*sBh + offs_d*sBd
    acc  += tl.load(b_ptrs).to(tl.float32)[None,:]
    return acc

# -----------------------------
# Forward kernel: streaming softmax
# -----------------------------
@triton.jit
def _demo_attn_kernel(
    # Q
    Pq_ptr, Vq_ptr, bq_ptr,
    # K
    Pk_ptr, Vk_ptr, bk_ptr,
    # V
    Pv_ptr, Vv_ptr, bv_ptr,
    # Out
    Out_ptr,
    # mask [B,1,1,N] broadcast & strides
    mask_ptr, sMb, sMh, sMq, sMk,
    # Q strides
    sQb, sQh, sQm, sQr,
    sVqb, sVqh, sVqr, sVqd,
    sBqb, sBqh, sBqd,
    # K strides
    sKb, sKh, sKn, sKr,
    sVkb, sVkh, sVkr, sVkd,
    sBkb, sBkh, sBkd,
    # V strides
    sVb2, sVh2, sVn2, sVr2,
    sVvb, sVvh, sVvr, sVvd,
    sBvb, sBvh, sBvd,
    # Out strides
    sOb, sOh, sOm,
    # sizes
    seqlen, r_dim, nheads, softmax_scale,
    # tiles
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b  = off_bh // nheads
    off_h  = off_bh %  nheads
    row_off = pid_m * BLOCK_M

    # Q tile [BLOCK_M, D]
    q = load_tiles(
        Pq_ptr, Vq_ptr, bq_ptr,
        sQb, sQh, sQm, sQr,
        sVqb, sVqh, sVqr, sVqd,
        sBqb, sBqh, sBqd,
        BLOCK_M, BLOCK_R, BLOCK_D,
        seqlen, r_dim, off_b, off_h, row_off,
    )

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, seqlen, BLOCK_M):
        offs_n = tl.arange(0, BLOCK_M)
        # mask [B,1,1,N] broadcast over H & M
        mask_ptrs = mask_ptr + off_b*sMb + off_h*sMh + 0*sMq + (start_n + offs_n)*sMk
        mask_i32  = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0

        # K / V tiles for this key block
        k = load_tiles(
            Pk_ptr, Vk_ptr, bk_ptr,
            sKb, sKh, sKn, sKr,
            sVkb, sVkh, sVkr, sVkd,
            sBkb, sBkh, sBkd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )
        v = load_tiles(
            Pv_ptr, Vv_ptr, bv_ptr,
            sVb2, sVh2, sVn2, sVr2,
            sVvb, sVvh, sVvr, sVvd,
            sBvb, sBvh, sBvd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None, :], qk, neg_inf)

        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:,None]), axis=1)
        acc      = acc * exp_diff[:,None] + tl.dot(tl.exp(qk - m_new[:,None]), v)
        m_i      = m_new

    den = tl.reshape(l_i, (BLOCK_M,1))
    out = acc / den

    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = Out_ptr + off_bh*sOb + offs_m[:,None]*sOh + offs_d[None,:]*sOm
    tl.store(out_ptrs, out, mask=offs_m[:,None] < seqlen)

def flash_svd_attention(Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask,
                        *, causal: bool = False, block_m=BLOCK_M, block_r=BLOCK_R):
    B,H,M,R = Pq.shape
    D       = Vq.shape[-1]
    scale   = 1.0/math.sqrt(D)

    Pq,Vq,bq = map(_contig,(Pq,Vq,bq))
    Pk,Vk,bk = map(_contig,(Pk,Vk,bk))
    Pv,Vv,bv = map(_contig,(Pv,Vv,bv))

    base = mask if mask.ndim==4 else mask[:, :1, :].unsqueeze(2)
    m4   = base.to(torch.int32).expand(B,H,1,M)
    if m4.stride(1) or m4.stride(2):
        m4 = m4.as_strided(m4.size(), (m4.stride(0),0,0,m4.stride(3)))
    sMb,sMh,sMq,sMk = m4.stride()

    Out = torch.empty(B*H, M, D, device=Pq.device, dtype=torch.float32)
    args = [
        Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv,
        Out, m4, sMb,sMh,sMq,sMk,
        *Pq.stride(), *Vq.stride(), *bq.stride(),
        *Pk.stride(), *Vk.stride(), *bk.stride(),
        *Pv.stride(), *Vv.stride(), *bv.stride(),
        *Out.stride(), M, R, H, scale,
    ]
    grid = ((M + block_m - 1)//block_m, B*H)
    _demo_attn_kernel[grid](*args, BLOCK_M=block_m, BLOCK_R=block_r, BLOCK_D=D, CAUSAL=int(causal))
    return Out.view(B,H,M,D).to(Pq.dtype)

# -----------------------------
# Backward kernels (two independent variants)
# -----------------------------
_BWD_CONFIGS = [
    #triton.Config({'BLOCK_M': 64, 'BLOCK_R': 64}, num_warps=4, num_stages=1),
    #triton.Config({'BLOCK_M': 64, 'BLOCK_R': 32}, num_warps=4, num_stages=1),
    #triton.Config({'BLOCK_M': 32, 'BLOCK_R': 64}, num_warps=4, num_stages=1),
    triton.Config({'BLOCK_M': 32, 'BLOCK_R': 32}, num_warps=2, num_stages=1),
]

@triton.autotune(configs=_BWD_CONFIGS, key=['seqlen','r_dim','BLOCK_D'])
@triton.jit
def _flash_svd_attn_bwd_kernel_auto(
    # Inputs
    Pq_ptr, Vq_ptr, bq_ptr,
    Pk_ptr, Vk_ptr, bk_ptr,
    Pv_ptr, Vv_ptr, bv_ptr,
    # Upstream dOut
    dOut_ptr,
    # mask [B,1,1,N]
    mask_ptr, sMb, sMh, sMq, sMk,
    # Q strides
    sQb, sQh, sQm, sQr,
    sVqb, sVqh, sVqr, sVqd,
    sBqb, sBqh, sBqd,
    # K strides
    sKb, sKh, sKn, sKr,
    sVkb, sVkh, sVkr, sVkd,
    sBkb, sBkh, sBkd,
    # V strides
    sVb2, sVh2, sVn2, sVr2,
    sVvb, sVvh, sVvr, sVvd,
    sBvb, sBvh, sBvd,
    # dOut strides
    sdOb, sdOh, sdOm,
    # Output grads
    dPq_ptr, dVq_ptr, dbq_ptr,
    dPk_ptr, dVk_ptr, dbk_ptr,
    dPv_ptr, dVv_ptr, dbv_ptr,
    # sizes/meta
    seqlen, r_dim, nheads, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b  = off_bh // nheads
    off_h  = off_bh %  nheads
    row_off = pid_m * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    r_idx  = tl.arange(0, BLOCK_R)
    mask_m = (row_off + offs_m) < seqlen

    # Q and dO tiles
    q = load_tiles(
        Pq_ptr, Vq_ptr, bq_ptr,
        sQb, sQh, sQm, sQr,
        sVqb, sVqh, sVqr, sVqd,
        sBqb, sBqh, sBqd,
        BLOCK_M, BLOCK_R, BLOCK_D,
        seqlen, r_dim, off_b, off_h, row_off,
    )
    dO_ptrs = dOut_ptr + off_bh*sdOb + (row_off + offs_m)[:,None]*sdOh + offs_d[None,:]*sdOm
    dO = tl.load(dO_ptrs, mask=mask_m[:,None], other=0.).to(tl.float32)

    # PASS 0: recompute m_i, l_i
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, seqlen, BLOCK_M):
        offs_n = tl.arange(0, BLOCK_M)
        mask_ptrs = mask_ptr + off_b*sMb + off_h*sMh + 0*sMq + (start_n + offs_n)*sMk
        mask_i32  = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0

        k = load_tiles(
            Pk_ptr, Vk_ptr, bk_ptr,
            sKb, sKh, sKn, sKr,
            sVkb, sVkh, sVkr, sVkd,
            sBkb, sBkh, sBkd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None,:], qk, neg_inf)

        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:,None]), axis=1)
        m_i      = m_new

    # PASS 1: dV + dp_sum
    dp_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, seqlen, BLOCK_M):
        offs_n = tl.arange(0, BLOCK_M)
        mask_ptrs = mask_ptr + off_b*sMb + off_h*sMh + 0*sMq + (start_n + offs_n)*sMk
        mask_i32  = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0

        k = load_tiles(
            Pk_ptr, Vk_ptr, bk_ptr,
            sKb, sKh, sKn, sKr,
            sVkb, sVkh, sVkr, sVkd,
            sBkb, sBkh, sBkd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )
        v = load_tiles(
            Pv_ptr, Vv_ptr, bv_ptr,
            sVb2, sVh2, sVn2, sVr2,
            sVvb, sVvh, sVvr, sVvd,
            sBvb, sBvh, sBvd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None,:], qk, neg_inf)
        p  = tl.exp(qk - m_i[:,None]) / tl.reshape(l_i, (BLOCK_M,1))

        dP = tl.dot(dO, tl.trans(v))            # [M_blk, N_blk]
        dp_sum += tl.sum(dP * p, axis=1)

        dV_dense = tl.dot(tl.trans(p), dO)      # [N_blk, D]

        # Map to {Pv, Vv, bv}
        for r_start in range(0, r_dim, BLOCK_R):
            r_mask = (r_start + r_idx) < r_dim

            Vv_ptrs = Vv_ptr + off_b*sVvb + off_h*sVvh + (r_start+r_idx)[:,None]*sVvr + offs_d[None,:]*sVvd
            Vv_sub  = tl.load(Vv_ptrs, mask=r_mask[:,None], other=0.).to(tl.float32)

            Pv_ptrs = Pv_ptr + off_b*sVb2 + off_h*sVh2 + (start_n+offs_n)[:,None]*sVn2 + (r_start+r_idx)[None,:]*sVr2
            Pv_sub  = tl.load(Pv_ptrs, mask=(offs_n[:,None] < seqlen) & r_mask[None,:], other=0.).to(tl.float32)

            dPv_sub  = tl.dot(dV_dense, tl.trans(Vv_sub))         # [N_blk, R_blk]
            dPv_ptrs = dPv_ptr + off_b*sVb2 + off_h*sVh2 + (start_n+offs_n)[:,None]*sVn2 + (r_start+r_idx)[None,:]*sVr2
            tl.atomic_add(dPv_ptrs, dPv_sub, mask=(offs_n[:,None] < seqlen) & r_mask[None,:])

            dVv_sub  = tl.dot(tl.trans(Pv_sub), dV_dense)         # [R_blk, D]
            dVv_ptrs = dVv_ptr + off_b*sVvb + off_h*sVvh + (r_start+r_idx)[:,None]*sVvr + offs_d[None,:]*sVvd
            tl.atomic_add(dVv_ptrs, dVv_sub, mask=r_mask[:,None])

        dbv_vec  = tl.sum(dV_dense, axis=0)
        dbv_ptrs = dbv_ptr + off_b*sBvb + off_h*sBvh + offs_d*sBvd
        tl.atomic_add(dbv_ptrs, dbv_vec)

    # PASS 2: dQ + dK then map to {Pq,Vq,bq} and {Pk,Vk,bk}
    dQ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in range(0, seqlen, BLOCK_M):
        offs_n = tl.arange(0, BLOCK_M)
        mask_ptrs = mask_ptr + off_b*sMb + off_h*sMh + 0*sMq + (start_n + offs_n)*sMk
        mask_i32  = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0

        k = load_tiles(
            Pk_ptr, Vk_ptr, bk_ptr,
            sKb, sKh, sKn, sKr,
            sVkb, sVkh, sVkr, sVkd,
            sBkb, sBkh, sBkd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )
        v = load_tiles(
            Pv_ptr, Vv_ptr, bv_ptr,
            sVb2, sVh2, sVn2, sVr2,
            sVvb, sVvh, sVvr, sVvd,
            sBvb, sBvh, sBvd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None,:], qk, neg_inf)
        p  = tl.exp(qk - m_i[:,None]) / tl.reshape(l_i, (BLOCK_M,1))

        dP = tl.dot(dO, tl.trans(v))
        dS = (dP - dp_sum[:,None]) * p

        dQ += softmax_scale * tl.dot(dS, k)
        dK_blk = softmax_scale * tl.dot(tl.trans(dS), q)

        # map dK to {Pk, Vk, bk}
        for r_start in range(0, r_dim, BLOCK_R):
            r_mask = (r_start + r_idx) < r_dim

            Vk_ptrs = Vk_ptr + off_b*sVkb + off_h*sVkh + (r_start+r_idx)[:,None]*sVkr + offs_d[None,:]*sVkd
            Vk_sub  = tl.load(Vk_ptrs, mask=r_mask[:,None], other=0.).to(tl.float32)

            Pk_ptrs = Pk_ptr + off_b*sKb + off_h*sKh + (start_n+offs_n)[:,None]*sKn + (r_start+r_idx)[None,:]*sKr
            Pk_sub  = tl.load(Pk_ptrs, mask=(offs_n[:,None] < seqlen) & r_mask[None,:], other=0.).to(tl.float32)

            dPk_sub  = tl.dot(dK_blk, tl.trans(Vk_sub))
            dPk_ptrs = dPk_ptr + off_b*sKb + off_h*sKh + (start_n+offs_n)[:,None]*sKn + (r_start+r_idx)[None,:]*sKr
            tl.atomic_add(dPk_ptrs, dPk_sub, mask=(offs_n[:,None] < seqlen) & r_mask[None,:])

            dVk_sub  = tl.dot(tl.trans(Pk_sub), dK_blk)
            dVk_ptrs = dVk_ptr + off_b*sVkb + off_h*sVkh + (r_start+r_idx)[:,None]*sVkr + offs_d[None,:]*sVkd
            tl.atomic_add(dVk_ptrs, dVk_sub, mask=r_mask[:,None])

        dbk_vec  = tl.sum(dK_blk, axis=0)
        dbk_ptrs = dbk_ptr + off_b*sBkb + off_h*sBkh + offs_d*sBkd
        tl.atomic_add(dbk_ptrs, dbk_vec)

    # map dQ -> {Pq, Vq, bq}
    for r_start in range(0, r_dim, BLOCK_R):
        r_mask = (r_start + r_idx) < r_dim

        Vq_ptrs = Vq_ptr + off_b*sVqb + off_h*sVqh + (r_start+r_idx)[:,None]*sVqr + offs_d[None,:]*sVqd
        Vq_sub  = tl.load(Vq_ptrs, mask=r_mask[:,None], other=0.).to(tl.float32)

        dPq_sub  = tl.dot(dQ, tl.trans(Vq_sub))
        dPq_ptrs = dPq_ptr + off_b*sQb + off_h*sQh + (row_off+offs_m)[:,None]*sQm + (r_start+r_idx)[None,:]*sQr
        tl.store(dPq_ptrs, dPq_sub, mask=(mask_m[:,None] & r_mask[None,:]))

        Pq_ptrs = Pq_ptr + off_b*sQb + off_h*sQh + (row_off+offs_m)[:,None]*sQm + (r_start+r_idx)[None,:]*sQr
        Pq_sub  = tl.load(Pq_ptrs, mask=(mask_m[:,None] & r_mask[None,:]), other=0.).to(tl.float32)

        dVq_sub  = tl.dot(tl.trans(Pq_sub), dQ)
        dVq_ptrs = dVq_ptr + off_b*sVqb + off_h*sVqh + (r_start+r_idx)[:,None]*sVqr + offs_d[None,:]*sVqd
        tl.atomic_add(dVq_ptrs, dVq_sub, mask=r_mask[:,None])

    dbq_vec  = tl.sum(dQ, axis=0)
    dbq_ptrs = dbq_ptr + off_b*sBqb + off_h*sBqh + offs_d*sBqd
    tl.atomic_add(dbq_ptrs, dbq_vec)

@triton.jit
def _flash_svd_attn_bwd_kernel_fixed(
    # Inputs
    Pq_ptr, Vq_ptr, bq_ptr,
    Pk_ptr, Vk_ptr, bk_ptr,
    Pv_ptr, Vv_ptr, bv_ptr,
    # Upstream dOut
    dOut_ptr,
    # mask [B,1,1,N]
    mask_ptr, sMb, sMh, sMq, sMk,
    # Q strides
    sQb, sQh, sQm, sQr,
    sVqb, sVqh, sVqr, sVqd,
    sBqb, sBqh, sBqd,
    # K strides
    sKb, sKh, sKn, sKr,
    sVkb, sVkh, sVkr, sVkd,
    sBkb, sBkh, sBkd,
    # V strides
    sVb2, sVh2, sVn2, sVr2,
    sVvb, sVvh, sVvr, sVvd,
    sBvb, sBvh, sBvd,
    # dOut strides
    sdOb, sdOh, sdOm,
    # Output grads
    dPq_ptr, dVq_ptr, dbq_ptr,
    dPk_ptr, dVk_ptr, dbk_ptr,
    dPv_ptr, dVv_ptr, dbv_ptr,
    # sizes/meta
    seqlen, r_dim, nheads, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # identical body to AUTO (duplicated intentionally to avoid helper)
    pid_m  = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b  = off_bh // nheads
    off_h  = off_bh %  nheads
    row_off = pid_m * BLOCK_M

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    r_idx  = tl.arange(0, BLOCK_R)
    mask_m = (row_off + offs_m) < seqlen

    q = load_tiles(Pq_ptr, Vq_ptr, bq_ptr,
                   sQb, sQh, sQm, sQr,
                   sVqb, sVqh, sVqr, sVqd,
                   sBqb, sBqh, sBqd,
                   BLOCK_M, BLOCK_R, BLOCK_D,
                   seqlen, r_dim, off_b, off_h, row_off)
    dO_ptrs = dOut_ptr + off_bh*sdOb + (row_off + offs_m)[:,None]*sdOh + offs_d[None,:]*sdOm
    dO = tl.load(dO_ptrs, mask=mask_m[:,None], other=0.).to(tl.float32)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, seqlen, BLOCK_M):
        offs_n = tl.arange(0, BLOCK_M)
        mask_ptrs = mask_ptr + off_b*sMb + off_h*sMh + 0*sMq + (start_n + offs_n)*sMk
        mask_i32  = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0
        k = load_tiles(Pk_ptr, Vk_ptr, bk_ptr,
                       sKb, sKh, sKn, sKr,
                       sVkb, sVkh, sVkr, sVkd,
                       sBkb, sBkh, sBkd,
                       BLOCK_M, BLOCK_R, BLOCK_D,
                       seqlen, r_dim, off_b, off_h, start_n)
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None,:], qk, neg_inf)
        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:,None]), axis=1)
        m_i      = m_new

    dp_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, seqlen, BLOCK_M):
        offs_n = tl.arange(0, BLOCK_M)
        mask_ptrs = mask_ptr + off_b*sMb + off_h*sMh + 0*sMq + (start_n + offs_n)*sMk
        mask_i32  = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0
        k = load_tiles(Pk_ptr, Vk_ptr, bk_ptr,
                       sKb, sKh, sKn, sKr,
                       sVkb, sVkh, sVkr, sVkd,
                       sBkb, sBkh, sBkd,
                       BLOCK_M, BLOCK_R, BLOCK_D,
                       seqlen, r_dim, off_b, off_h, start_n)
        v = load_tiles(Pv_ptr, Vv_ptr, bv_ptr,
                       sVb2, sVh2, sVn2, sVr2,
                       sVvb, sVvh, sVvr, sVvd,
                       sBvb, sBvh, sBvd,
                       BLOCK_M, BLOCK_R, BLOCK_D,
                       seqlen, r_dim, off_b, off_h, start_n)
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None,:], qk, neg_inf)
        p  = tl.exp(qk - m_i[:,None]) / tl.reshape(l_i, (BLOCK_M,1))
        dP = tl.dot(dO, tl.trans(v))
        dp_sum += tl.sum(dP * p, axis=1)
        dV_dense = tl.dot(tl.trans(p), dO)
        for r_start in range(0, r_dim, BLOCK_R):
            r_mask = (r_start + r_idx) < r_dim
            Vv_ptrs = Vv_ptr + off_b*sVvb + off_h*sVvh + (r_start+r_idx)[:,None]*sVvr + offs_d[None,:]*sVvd
            Vv_sub  = tl.load(Vv_ptrs, mask=r_mask[:,None], other=0.).to(tl.float32)
            Pv_ptrs = Pv_ptr + off_b*sVb2 + off_h*sVh2 + (start_n+offs_n)[:,None]*sVn2 + (r_start+r_idx)[None,:]*sVr2
            Pv_sub  = tl.load(Pv_ptrs, mask=(offs_n[:,None] < seqlen) & r_mask[None,:], other=0.).to(tl.float32)
            dPv_sub  = tl.dot(dV_dense, tl.trans(Vv_sub))
            dPv_ptrs = dPv_ptr + off_b*sVb2 + off_h*sVh2 + (start_n+offs_n)[:,None]*sVn2 + (r_start+r_idx)[None,:]*sVr2
            tl.atomic_add(dPv_ptrs, dPv_sub, mask=(offs_n[:,None] < seqlen) & r_mask[None,:])
            dVv_sub  = tl.dot(tl.trans(Pv_sub), dV_dense)
            dVv_ptrs = dVv_ptr + off_b*sVvb + off_h*sVvh + (r_start+r_idx)[:,None]*sVvr + offs_d[None,:]*sVvd
            tl.atomic_add(dVv_ptrs, dVv_sub, mask=r_mask[:,None])
        dbv_vec  = tl.sum(dV_dense, axis=0)
        dbv_ptrs = dbv_ptr + off_b*sBvb + off_h*sBvh + offs_d*sBvd
        tl.atomic_add(dbv_ptrs, dbv_vec)

    dQ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for start_n in range(0, seqlen, BLOCK_M):
        offs_n = tl.arange(0, BLOCK_M)
        mask_ptrs = mask_ptr + off_b*sMb + off_h*sMh + 0*sMq + (start_n + offs_n)*sMk
        mask_i32  = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0
        k = load_tiles(Pk_ptr, Vk_ptr, bk_ptr,
                       sKb, sKh, sKn, sKr,
                       sVkb, sVkh, sVkr, sVkd,
                       sBkb, sBkh, sBkd,
                       BLOCK_M, BLOCK_R, BLOCK_D,
                       seqlen, r_dim, off_b, off_h, start_n)
        v = load_tiles(Pv_ptr, Vv_ptr, bv_ptr,
                       sVb2, sVh2, sVn2, sVr2,
                       sVvb, sVvh, sVvr, sVvd,
                       sBvb, sBvh, sBvd,
                       BLOCK_M, BLOCK_R, BLOCK_D,
                       seqlen, r_dim, off_b, off_h, start_n)
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None,:], qk, neg_inf)
        p  = tl.exp(qk - m_i[:,None]) / tl.reshape(l_i, (BLOCK_M,1))
        dP = tl.dot(dO, tl.trans(v))
        dS = (dP - dp_sum[:,None]) * p
        dQ += softmax_scale * tl.dot(dS, k)
        dK_blk = softmax_scale * tl.dot(tl.trans(dS), q)
        for r_start in range(0, r_dim, BLOCK_R):
            r_mask = (r_start + r_idx) < r_dim
            Vk_ptrs = Vk_ptr + off_b*sVkb + off_h*sVkh + (r_start+r_idx)[:,None]*sVkr + offs_d[None,:]*sVkd
            Vk_sub  = tl.load(Vk_ptrs, mask=r_mask[:,None], other=0.).to(tl.float32)
            Pk_ptrs = Pk_ptr + off_b*sKb + off_h*sKh + (start_n+offs_n)[:,None]*sKn + (r_start+r_idx)[None,:]*sKr
            Pk_sub  = tl.load(Pk_ptrs, mask=(offs_n[:,None] < seqlen) & r_mask[None,:], other=0.).to(tl.float32)
            dPk_sub  = tl.dot(dK_blk, tl.trans(Vk_sub))
            dPk_ptrs = dPk_ptr + off_b*sKb + off_h*sKh + (start_n+offs_n)[:,None]*sKn + (r_start+r_idx)[None,:]*sKr
            tl.atomic_add(dPk_ptrs, dPk_sub, mask=(offs_n[:,None] < seqlen) & r_mask[None,:])
            dVk_sub  = tl.dot(tl.trans(Pk_sub), dK_blk)
            dVk_ptrs = dVk_ptr + off_b*sVkb + off_h*sVkh + (r_start+r_idx)[:,None]*sVkr + offs_d[None,:]*sVkd
            tl.atomic_add(dVk_ptrs, dVk_sub, mask=r_mask[:,None])
        dbk_vec  = tl.sum(dK_blk, axis=0)
        dbk_ptrs = dbk_ptr + off_b*sBkb + off_h*sBkh + offs_d*sBkd
        tl.atomic_add(dbk_ptrs, dbk_vec)

    for r_start in range(0, r_dim, BLOCK_R):
        r_mask = (r_start + r_idx) < r_dim
        Vq_ptrs = Vq_ptr + off_b*sVqb + off_h*sVqh + (r_start+r_idx)[:,None]*sVqr + offs_d[None,:]*sVqd
        Vq_sub  = tl.load(Vq_ptrs, mask=r_mask[:,None], other=0.).to(tl.float32)
        dPq_sub  = tl.dot(dQ, tl.trans(Vq_sub))
        dPq_ptrs = dPq_ptr + off_b*sQb + off_h*sQh + (row_off+offs_m)[:,None]*sQm + (r_start+r_idx)[None,:]*sQr
        tl.store(dPq_ptrs, dPq_sub, mask=(mask_m[:,None] & r_mask[None,:]))
        Pq_ptrs = Pq_ptr + off_b*sQb + off_h*sQh + (row_off+offs_m)[:,None]*sQm + (r_start+r_idx)[None,:]*sQr
        Pq_sub  = tl.load(Pq_ptrs, mask=(mask_m[:,None] & r_mask[None,:]), other=0.).to(tl.float32)
        dVq_sub  = tl.dot(tl.trans(Pq_sub), dQ)
        dVq_ptrs = dVq_ptr + off_b*sVqb + off_h*sVqh + (r_start+r_idx)[:,None]*sVqr + offs_d[None,:]*sVqd
        tl.atomic_add(dVq_ptrs, dVq_sub, mask=r_mask[:,None])
    dbq_vec  = tl.sum(dQ, axis=0)
    dbq_ptrs = dbq_ptr + off_b*sBqb + off_h*sBqh + offs_d*sBqd
    tl.atomic_add(dbq_ptrs, dbq_vec)

def flash_svd_attention_backward(
    Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask, dOut, *,
    block_m=BLOCK_M, block_r=BLOCK_R, tune=True
):
    B,H,M,R = Pq.shape
    D = Vq.shape[-1]
    scale = 1.0/math.sqrt(D)

    base = mask if mask.ndim==4 else mask[:, :1, :].unsqueeze(2)
    m4   = base.to(torch.int32).expand(B,H,1,M)
    if m4.stride(1) or m4.stride(2):
        m4 = m4.as_strided(m4.size(), (m4.stride(0),0,0,m4.stride(3)))
    sMb,sMh,sMq,sMk = m4.stride()

    dPq = torch.zeros_like(Pq, dtype=torch.float32)
    dVq = torch.zeros_like(Vq, dtype=torch.float32)
    dbq = torch.zeros(B,H,D, device=Pq.device, dtype=torch.float32)

    dPk = torch.zeros_like(Pk, dtype=torch.float32)
    dVk = torch.zeros_like(Vk, dtype=torch.float32)
    dbk = torch.zeros(B,H,D, device=Pq.device, dtype=torch.float32)

    dPv = torch.zeros_like(Pv, dtype=torch.float32)
    dVv = torch.zeros_like(Vv, dtype=torch.float32)
    dbv = torch.zeros(B,H,D, device=Pq.device, dtype=torch.float32)

    dOutBH = dOut.contiguous().view(B*H, M, D)

    if tune:
        grid = lambda META: (((M + META['BLOCK_M'] - 1) // META['BLOCK_M']), B*H)
        _flash_svd_attn_bwd_kernel_auto[grid](
            Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv,
            dOutBH,
            m4, sMb, sMh, sMq, sMk,
            *Pq.stride(), *Vq.stride(), *bq.stride(),
            *Pk.stride(), *Vk.stride(), *bk.stride(),
            *Pv.stride(), *Vv.stride(), *bv.stride(),
            *dOutBH.stride(),
            dPq, dVq, dbq, dPk, dVk, dbk, dPv, dVv, dbv,
            M, R, H, scale,
            BLOCK_D=Vq.shape[-1],
        )
    else:
        grid = (((M + block_m - 1)//block_m), B*H)
        _flash_svd_attn_bwd_kernel_fixed[grid](
            Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv,
            dOutBH,
            m4, sMb, sMh, sMq, sMk,
            *Pq.stride(), *Vq.stride(), *bq.stride(),
            *Pk.stride(), *Vk.stride(), *bk.stride(),
            *Pv.stride(), *Vv.stride(), *bv.stride(),
            *dOutBH.stride(),
            dPq, dVq, dbq, dPk, dVk, dbk, dPv, dVv, dbv,
            M, R, H, scale,
            BLOCK_M=block_m, BLOCK_R=block_r, BLOCK_D=Vq.shape[-1],
        )

    in_dtype = Pq.dtype
    return (
        dPq.to(in_dtype), dVq.to(in_dtype), dbq.to(in_dtype),
        dPk.to(in_dtype), dVk.to(in_dtype), dbk.to(in_dtype),
        dPv.to(in_dtype), dVv.to(in_dtype), dbv.to(in_dtype),
    )

# -----------------------------
# Torch streaming forward (for debugging)
# -----------------------------
def _normalize_key_mask(mask, N: int, device):
    if mask.ndim == 1:
        key = mask
    elif mask.ndim == 2:
        key = mask[0]
    elif mask.ndim == 3:
        key = mask[0,0]
    elif mask.ndim == 4:
        key = mask[0,0,0]
    else:
        raise ValueError(f"Unsupported mask.ndim={mask.ndim}")
    key = key.to(torch.bool).reshape(-1)
    if key.numel() != N:
        raise ValueError(f"Key mask length {key.numel()} != N={N}")
    return key.to(device)

def streaming_forward_torch(Q, K, V, mask, block_m, scale, print_block_stats=False):
    M, D = Q.shape
    N = K.shape[0]
    out = torch.empty(M, D, device=Q.device, dtype=torch.float32)
    mask_key = _normalize_key_mask(mask, N, Q.device)

    for m0 in range(0, M, block_m):
        m1 = min(m0 + block_m, M)
        q = Q[m0:m1].to(torch.float32)
        acc = torch.zeros((m1-m0, D), device=Q.device, dtype=torch.float32)
        m_i = torch.full((m1-m0,), float('-inf'), device=Q.device)
        l_i = torch.zeros((m1-m0,), device=Q.device)

        for n0 in range(0, N, block_m):
            n1 = min(n0 + block_m, N)
            k = K[n0:n1].to(torch.float32)
            v = V[n0:n1].to(torch.float32)

            qk = (q @ k.T) * scale
            msub = mask_key[n0:n1]
            if (~msub).any():
                qk = torch.where(msub.unsqueeze(0), qk, torch.full_like(qk, float('-inf')))

            m_new = torch.maximum(m_i, qk.max(dim=1).values)
            exp_diff = torch.exp(m_i - m_new)
            e = torch.exp(qk - m_new[:,None])
            l_i  = l_i * exp_diff + e.sum(dim=1)
            acc  = acc * exp_diff[:,None] + (e @ v)
            m_i  = m_new

            if print_block_stats:
                print(f"  [keys {n0}:{n1}] m_i(min/mean/max)={m_i.min().item():.3e}/"
                      f"{m_i.mean().item():.3e}/{m_i.max().item():.3e} "
                      f"l_i(min/mean/max)={l_i.min().item():.3e}/"
                      f"{l_i.mean().item():.3e}/{l_i.max().item():.3e}")

        out[m0:m1] = acc / l_i[:, None].clamp_min(1e-30)
    return out

# -----------------------------
# Dense reference (fp32)
# -----------------------------
def dense_forward(Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask):
    Pq32, Vq32, bq32 = Pq.float(), Vq.float(), bq.float()
    Pk32, Vk32, bk32 = Pk.float(), Vk.float(), bk.float()
    Pv32, Vv32, bv32 = Pv.float(), Vv.float(), bv.float()
    Q = Pq32 @ Vq32 + bq32.unsqueeze(2)
    K = Pk32 @ Vk32 + bk32.unsqueeze(2)
    V = Pv32 @ Vv32 + bv32.unsqueeze(2)
    D = Q.shape[-1]; scale = 1.0/math.sqrt(D)
    base = mask if mask.ndim==4 else mask[:, :1, :].unsqueeze(2)
    logits = (Q @ K.transpose(-1,-2)) * scale
    logits = logits.masked_fill(~base.to(torch.bool), float('-inf'))
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    out = probs @ V
    return out  # keep fp32 for comparison

# -----------------------------
# Metrics & debug utilities
# -----------------------------
def rel_fro(a: torch.Tensor, b: torch.Tensor, eps=1e-12) -> float:
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    num = torch.linalg.vector_norm(a32 - b32)
    den = torch.linalg.vector_norm(b32).clamp_min(eps)
    return (num / den).item()

def tensor_stats(name, t):
    t32 = t.detach().float()
    finite = torch.isfinite(t32)
    total = t32.numel()
    fin_count = finite.sum().item()
    print(f"[dbg] {name:<12s} shape={tuple(t32.shape)} "
          f"finite={fin_count}/{total} "
          f"min/mean/max={t32[finite].min().item() if fin_count else float('nan'):.3e}/"
          f"{t32[finite].mean().item() if fin_count else float('nan'):.3e}/"
          f"{t32[finite].max().item() if fin_count else float('nan'):.3e}")

# -----------------------------
# Check & bench
# -----------------------------
def check_correctness_and_bench(
    B=1, H=1, M=128, R=64, D=64, dtype=torch.float16, device='cuda',
    iters=20, per_head_mask=False, tune=True, debug=True, print_block_stats=False,
    use_checkpoint=False, measure_peak_mem=True
):
    device_obj = torch.device(device)

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Eagerly create CUDA context to silence cuBLAS warning
    _ = torch.randn(1, device=device)

    # Sane init: var(P@V) ~ 1  -> var(V) = 1/R
    scaleV = 1.0 / math.sqrt(R)
    Pq = torch.randn(B,H,M,R, device=device, dtype=dtype)
    Vq = torch.randn(B,H,R,D, device=device, dtype=dtype) * scaleV
    bq = torch.zeros (B,H,D,   device=device, dtype=dtype)

    Pk = torch.randn(B,H,M,R, device=device, dtype=dtype)
    Vk = torch.randn(B,H,R,D, device=device, dtype=dtype) * scaleV
    bk = torch.zeros (B,H,D,   device=device, dtype=dtype)

    Pv = torch.randn(B,H,M,R, device=device, dtype=dtype)
    Vv = torch.randn(B,H,R,D, device=device, dtype=dtype) * scaleV
    bv = torch.zeros (B,H,D,   device=device, dtype=dtype)

    mask = torch.ones(B,1,M, device=device, dtype=torch.bool) if not per_head_mask \
           else torch.ones(B,H,M, device=device, dtype=torch.bool)

    Q = (Pq.float() @ Vq.float()) + bq.float().unsqueeze(2)
    K = (Pk.float() @ Vk.float()) + bk.float().unsqueeze(2)
    V = (Pv.float() @ Vv.float()) + bv.float().unsqueeze(2)
    if debug:
        tensor_stats("Q", Q); tensor_stats("K", K); tensor_stats("V", V)

    Q0, K0, V0 = Q[0,0], K[0,0], V[0,0]
    base = mask if mask.ndim==4 else mask[:, :1, :].unsqueeze(2)
    s = 1.0/math.sqrt(D)

    out_stream = streaming_forward_torch(Q0, K0, V0, base, BLOCK_M, s, print_block_stats)
    out_stream_notiled = streaming_forward_torch(Q0, K0, V0, base, K0.shape[0], s, False)
    out_dense  = dense_forward(Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask)[0,0]

    print(f"[dbg] relF(stream,dense)         = {rel_fro(out_stream, out_dense):.3e}")
    print(f"[dbg] relF(stream_no_tile,dense) = {rel_fro(out_stream_notiled, out_dense):.3e}")

    out_triton = flash_svd_attention(Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask, block_m=BLOCK_M, block_r=BLOCK_R)
    fwd_rel = rel_fro(out_triton[0,0].float(), out_dense)
    print(f"[check] forward relF(triton,dense) = {fwd_rel:.3e}  (target < 1e-2)")
    print(f"[dbg]   relF(triton,stream)       = {rel_fro(out_triton[0,0].float(), out_stream):.3e}")

    with torch.enable_grad():
        Pq_r, Vq_r, bq_r = [x.detach().clone().requires_grad_(True) for x in (Pq,Vq,bq)]
        Pk_r, Vk_r, bk_r = [x.detach().clone().requires_grad_(True) for x in (Pk,Vk,bk)]
        Pv_r, Vv_r, bv_r = [x.detach().clone().requires_grad_(True) for x in (Pv,Vv,bv)]
        ref_out = dense_forward(Pq_r,Vq_r,bq_r, Pk_r,Vk_r,bk_r, Pv_r,Vv_r,bv_r, mask)
        dOut = torch.randn_like(ref_out)
        (dPq_ref, dVq_ref, dbq_ref,
         dPk_ref, dVk_ref, dbk_ref,
         dPv_ref, dVv_ref, dbv_ref) = torch.autograd.grad(
            outputs=ref_out,
            inputs=[Pq_r,Vq_r,bq_r, Pk_r,Vk_r,bk_r, Pv_r,Vv_r,bv_r],
            grad_outputs=dOut
        )

    bwd_kwargs = {
        "tune": tune,
    }
    if not tune:
        bwd_kwargs.update(block_m=BLOCK_M, block_r=BLOCK_R)

    (dPq_tt, dVq_tt, dbq_tt,
     dPk_tt, dVk_tt, dbk_tt,
     dPv_tt, dVv_tt, dbv_tt) = flash_svd_attention_backward(
        Pq.detach(),Vq.detach(),bq.detach(),
        Pk.detach(),Vk.detach(),bk.detach(),
        Pv.detach(),Vv.detach(),bv.detach(),
        mask, dOut.detach(),
        **bwd_kwargs,
    )

    def report(name, a, b):
        rel = rel_fro(a, b)
        diff = (a - b).abs()
        print(f"[check] {name:6s}  relF={rel:.3e}  max|Δ|={diff.max().item():.3e}  mean|Δ|={diff.mean().item():.3e}")

    report("dPq", dPq_tt, dPq_ref)
    report("dVq", dVq_tt, dVq_ref)
    report("dbq", dbq_tt, dbq_ref)
    report("dPk", dPk_tt, dPk_ref)
    report("dVk", dVk_tt, dVk_ref)
    report("dbk", dbk_tt, dbk_ref)
    report("dPv", dPv_tt, dPv_ref)
    report("dVv", dVv_tt, dVv_ref)
    report("dbv", dbv_tt, dbv_ref)

    # ---- Micro bench (optional) ----
    torch.cuda.synchronize()
    def bench_fwd():
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = flash_svd_attention(Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask)
        torch.cuda.synchronize()
        return time.time() - t0

    def bench_bwd():
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                flash_svd_attention_backward(
                    Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask, dOut,
                    **bwd_kwargs,
                )
        torch.cuda.synchronize()
        return time.time() - t0

    bench_fwd(); bench_bwd()
    if device_obj.type == 'cuda':
        torch.cuda.synchronize(device=device_obj)
    tf = bench_fwd()
    tb = bench_bwd()
    print(f"[bench] forward:  {tf/iters*1e3:.2f} ms / iter  (B={B},H={H},M={M},R={R},D={D})")
    mode = "autotuned" if tune else "fixed"
    print(f"[bench] backward: {tb/iters*1e3:.2f} ms / iter  ({mode})")

    if measure_peak_mem and device_obj.type == 'cuda':
        ckpt_tag = "ckpt" if use_checkpoint else "no-ckpt"

        def measure_peak(label, fn):
            with torch.cuda.device(device_obj):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                fn()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
            peak_mb = peak / (1024 ** 2)
            print(f"[mem] {label:<28s} peak={peak_mb:8.2f} MiB")
            return peak

        def triton_step():
            _ = flash_svd_attention(
                Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask,
                block_m=BLOCK_M, block_r=BLOCK_R,
            )
            flash_svd_attention_backward(
                Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask,
                dOut.detach(),
                **bwd_kwargs,
            )

        def dense_forward_wrapped(*params):
            return dense_forward(
                params[0], params[1], params[2],
                params[3], params[4], params[5],
                params[6], params[7], params[8],
                mask,
            )

        def dense_step():
            with torch.enable_grad():
                tensors = [
                    Pq.detach().clone().requires_grad_(True),
                    Vq.detach().clone().requires_grad_(True),
                    bq.detach().clone().requires_grad_(True),
                    Pk.detach().clone().requires_grad_(True),
                    Vk.detach().clone().requires_grad_(True),
                    bk.detach().clone().requires_grad_(True),
                    Pv.detach().clone().requires_grad_(True),
                    Vv.detach().clone().requires_grad_(True),
                    bv.detach().clone().requires_grad_(True),
                ]
                dOut_local = dOut.detach().clone()
                if use_checkpoint:
                    out = checkpoint_fn(dense_forward_wrapped, *tensors)
                else:
                    out = dense_forward_wrapped(*tensors)
                out.backward(dOut_local)
                del out, dOut_local
            for t in tensors:
                if t.grad is not None:
                    t.grad = None
            del tensors

        triton_step()
        dense_step()

        peak_triton = measure_peak(f"triton ({mode}, {ckpt_tag})", triton_step)
        peak_dense = measure_peak(f"dense_autograd ({ckpt_tag})", dense_step)
        delta_mb = (peak_dense - peak_triton) / (1024 ** 2)
        print(f"[mem] dense - triton ({ckpt_tag}): {delta_mb:8.2f} MiB")
    elif measure_peak_mem:
        print("[mem] Peak memory stats require CUDA device; skipping.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    check_correctness_and_bench(
        B=128, H=12, M=128*4, R=32, D=64, dtype=torch.float16, device='cuda',
        iters=20, per_head_mask=False, tune=True, debug=True, print_block_stats=False
    )
