#!/usr/bin/env python3
# flashsvdattn4D.py  – rank-aware Flash-SVD attention (mask-friendly)
#
# Needs: utils_mask_4D.py  (Triton kernel)

import math, torch, torch.nn as nn
import torch, triton, triton.language as tl
import math


BLOCK_M = 64
BLOCK_R = 64                      # Triton tile, not the low rank R

def _contig(t): return t.contiguous() if not t.is_contiguous() else t

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

# ───────────────────────────────────────────────────────────────
# 2) Streaming-attention kernel with true [B,1,1,N] mask
# ───────────────────────────────────────────────────────────────
@triton.jit
def _demo_attn_kernel(
    # Q factors
    Pq_ptr, Vq_ptr, bq_ptr,
    # K factors
    Pk_ptr, Vk_ptr, bk_ptr,
    # V factors
    Pv_ptr, Vv_ptr, bv_ptr,
    # output
    Out_ptr,
    # mask + its 4 strides
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
    # tile sizes
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)
    off_b   = off_bh // nheads
    off_h   = off_bh %  nheads
    row_off = start_m * BLOCK_M

    # 1) Q tile
    q = load_tiles(
        Pq_ptr, Vq_ptr, bq_ptr,
        sQb, sQh, sQm, sQr,
        sVqb, sVqh, sVqr, sVqd,
        sBqb, sBqh, sBqd,
        BLOCK_M, BLOCK_R, BLOCK_D,
        seqlen, r_dim, off_b, off_h, row_off,
    )

    # softmax accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # 2) iterate over key blocks
    for start_n in range(0, seqlen, BLOCK_M):
        start_n = tl.multiple_of(start_n, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_M)
        
        # load mask [B,1,1,N]
        # sMh and sMq are both zero, so head & query dims broadcast
        mask_ptrs = (
            mask_ptr
          + off_b   * sMb    # batch
          + off_h   * sMh    # = 0
          + 0       * sMq    # = 0 (only one query-row)
          + (start_n + offs_n) * sMk  # key positions
        )
        mask_i32  = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0

        # load K, V
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

        # QK^T → apply mask → online softmax
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None, :], qk, neg_inf)

        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:,None]), axis=1)
        acc      = acc * exp_diff[:,None] + tl.dot(tl.exp(qk - m_new[:,None]), v)
        m_i      = m_new

    # 3) finalize
    den = tl.reshape(l_i, (BLOCK_M,1))
    out = acc / den

    # 4) write back
    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = (
        Out_ptr + off_bh*sOb
                 + offs_m[:,None]*sOh
                 + offs_d[None,:]*sOm
    )
    tl.store(out_ptrs, out, mask=offs_m[:,None] < seqlen)



def flash_svd_attention(Pq,Vq,bq, Pk,Vk,bk, Pv,Vv,bv, mask,
                        *, block_m=BLOCK_M, block_r=BLOCK_R):
    B,H,M,R = Pq.shape
    D       = Vq.shape[-1]
    scale   = 1.0/math.sqrt(D)

    Pq,Vq,bq = map(_contig,(Pq,Vq,bq))
    Pk,Vk,bk = map(_contig,(Pk,Vk,bk))
    Pv,Vv,bv = map(_contig,(Pv,Vv,bv))

    # HF mask [B,1,1,M] or per-head [B,H,M]
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
    _demo_attn_kernel[grid](*args, BLOCK_M=BLOCK_M,
                            BLOCK_R=BLOCK_R, BLOCK_D=D)
    return Out.view(B,H,M,D).to(Pq.dtype)


