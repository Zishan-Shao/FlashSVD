# utils_mask_4D.py
import torch, triton, triton.language as tl

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
    offs_m = row_off + tl.arange(0, BLOCK_M)
    pad_q_ptrs = (
        mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_m * sMk
    )
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m < seqlen, other=0).to(tl.int32)
    pad_q   = pad_q_i > 0
    q = load_tiles(
        Pq_ptr, Vq_ptr, bq_ptr,
        sQb, sQh, sQm, sQr,
        sVqb, sVqh, sVqr, sVqd,
        sBqb, sBqh, sBqd,
        BLOCK_M, BLOCK_R, BLOCK_D,
        seqlen, r_dim, off_b, off_h, row_off,
    )
    q = q * pad_q[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, seqlen, BLOCK_M):
        start_n = tl.multiple_of(start_n, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_M)
        mask_ptrs = (
            mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + (start_n + offs_n) * sMk
        )
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
        qk = tl.where(mask_vals[None, :], qk, neg_inf)

        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:,None]), axis=1)
        acc      = acc * exp_diff[:,None] + tl.dot(tl.exp(qk - m_new[:,None]), v)
        m_i      = m_new

    den = tl.reshape(l_i, (BLOCK_M,1))
    out = acc / den

    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = (
        Out_ptr + off_bh*sOb + offs_m[:,None]*sOh + offs_d[None,:]*sOm
    )
    tl.store(out_ptrs, out, mask=offs_m[:,None] < seqlen)

