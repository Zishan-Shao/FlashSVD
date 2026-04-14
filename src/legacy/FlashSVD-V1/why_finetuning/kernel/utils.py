
# utils.py
import torch, triton, triton.language as tl
import math 

# ─────────────────────────────────────────────────────────────────────────────
# 1. On-chip tile loader (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. Streaming Attention Kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _demo_attn_kernel(
    # Q factors
    Pq_ptr, Vq_ptr, bias_q_ptr,
    # K factors
    Pk_ptr, Vk_ptr, bias_k_ptr,
    # V factors
    Pv_ptr, Vv_ptr, bias_v_ptr,
    # output
    Out_ptr,
    # strides Q
    sQb, sQh, sQm, sQr,
    sVqb, sVqh, sVqr, sVqd,
    sBqb, sBqh, sBqd,
    # strides K
    sKb, sKh, sKn, sKr,
    sVkb, sVkh, sVkr, sVkd,
    sBkb, sBkh, sBkd,
    # strides V
    sVb2, sVh2, sVn2, sVr2,
    sVvb, sVvh, sVvr, sVvd,
    sBvb, sBvh, sBvd,
    # strides Out
    sOb, sOh, sOm, #sOd,
    # sizes
    seqlen, r_dim, nheads, softmax_scale,
    # tile sizes
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # decode
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)
    off_b   = off_bh // nheads
    off_h   = off_bh %  nheads
    row_off = start_m * BLOCK_M

    # 1) load Q-tile once
    q = load_tiles(
        Pq_ptr, Vq_ptr, bias_q_ptr,
        sQb, sQh, sQm, sQr,
        sVqb, sVqh, sVqr, sVqd,
        sBqb, sBqh, sBqd,
        BLOCK_M, BLOCK_R, BLOCK_D,
        seqlen, r_dim, off_b, off_h, row_off,
    )  # shape [BLOCK_M, BLOCK_D]

    # allocate accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # 2) loop over K/V in blocks of BLOCK_M (use same block size for N)
    for start_n in range(0, seqlen, BLOCK_M):
        start_n = tl.multiple_of(start_n, BLOCK_M)
        k = load_tiles(
            Pk_ptr, Vk_ptr, bias_k_ptr,
            sKb, sKh, sKn, sKr,
            sVkb, sVkh, sVkr, sVkd,
            sBkb, sBkh, sBkd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )  # [BLOCK_M, BLOCK_D]
        v = load_tiles(
            Pv_ptr, Vv_ptr, bias_v_ptr,
            sVb2, sVh2, sVn2, sVr2,
            sVvb, sVvh, sVvr, sVvd,
            sBvb, sBvh, sBvd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            seqlen, r_dim, off_b, off_h, start_n
        )  # [BLOCK_M, BLOCK_D]

        # 2a) compute QK'
        # q: [M,D], k: [M,D] → qk: [M,M]
        qk = tl.dot(q, tl.trans(k)) * softmax_scale

        # 2b) stable-softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))            # new max per row
        exp_diff = tl.exp(m_i - m_new)                         # [M]
        l_i     = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:,None]), axis=1)
        # scale old accumulator to new “max” scale
        acc = acc * exp_diff[:, None]
        # accumulate V contribution
        acc += tl.dot(tl.exp(qk - m_new[:,None]), v)          # [M,D]
        m_i = m_new

    # 3) finalize: divide by l_i to complete softmax normalization
    out = acc / l_i[:, None]  # [BLOCK_M, D]

    # 4) write back
    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = (
        Out_ptr + off_bh * sOb
                 + offs_m[:, None] * sOh
                 + offs_d[None, :] * sOm
    )
    tl.store(out_ptrs, out, mask=offs_m[:, None] < seqlen)



# ─────────────────────────────────────────────────────────────────────────────
# 3. Test harness      (put this after the kernel in utils.py)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(0)
    dev = "cuda"

    # ---------- configuration ------------------------------------------------
    B, H, M, D, R = 2, 12, 128, 64, 64 
    BLOCK_M, BLOCK_R, BLOCK_D = 32, 32, 64
    softmax_scale = 1.0 / math.sqrt(D)

    # ---------- independent low-rank factors ---------------------------------
    Pq = torch.randn(B, H, M, R, device=dev, dtype=torch.float16).contiguous()
    Vq = torch.randn(B, H, R, D, device=dev, dtype=torch.float16).contiguous()
    bq = torch.randn(B, H,    D, device=dev, dtype=torch.float16).contiguous()

    Pk = torch.randn_like(Pq)
    Vk = torch.randn_like(Vq)
    bk = torch.randn_like(bq)

    Pv = torch.randn_like(Pq)
    Vv = torch.randn_like(Vq)
    bv = torch.randn_like(bq)
    
    print("Pq.shape: ", Pq.shape) # torch.Size([2, 12, 128, 64])
    print("Pk.shape: ", Pk.shape) # torch.Size([2, 12, 128, 64])
    print("Pv.shape: ", Pv.shape) # torch.Size([2, 12, 128, 64])
    print("Vq.shape: ", Vq.shape) # torch.Size([2, 12, 64, 64])
    print("Vk.shape: ", Vk.shape) # torch.Size([2, 12, 64, 64])
    print("Vv.shape: ", Vv.shape) # torch.Size([2, 12, 64, 64])
    print("bq.shape: ", bq.shape) # torch.Size([2, 12, 64])
    print("bk.shape: ", bk.shape) # torch.Size([2, 12, 64])
    print("bv.shape: ", bv.shape) # torch.Size([2, 12, 64])

    # for now, we have the result correct:
    # max-abs diff : 0.9426202774047852
    # rel-Frob err : 0.0026156927924603224

    # ---------- PyTorch “truth” ---------------------------------------------
    Q = (Pq.float().reshape(B*H, M, R) @ Vq.float().reshape(B*H, R, D)
         ).reshape(B, H, M, D) + bq.view(B, H, 1, D).float()

    K = (Pk.float().reshape(B*H, M, R) @ Vk.float().reshape(B*H, R, D)
         ).reshape(B, H, M, D) + bk.view(B, H, 1, D).float()

    V = (Pv.float().reshape(B*H, M, R) @ Vv.float().reshape(B*H, R, D)
         ).reshape(B, H, M, D) + bv.view(B, H, 1, D).float()

    logits  = torch.einsum("bhmd,bhnd->bhmn", Q, K) * softmax_scale
    weights = torch.softmax(logits, dim=-1)
    ref     = torch.einsum("bhmn,bhnd->bhmd", weights, V)

    # ---------- output buffer ------------------------------------------------
    Out = torch.empty(B*H, M, D, device=dev, dtype=torch.float32)

    # ---------- pack kernel arguments ---------------------------------------
    args = (
        # ① pointers (Q  K  V  Out) ------------------------------------------
        Pq, Vq, bq,   Pk, Vk, bk,   Pv, Vv, bv,   Out,
        # ② Q-strides (Pq, Vq, bq) -------------------------------------------
        *Pq.stride(), *Vq.stride(), *bq.stride(),
        # ③ K-strides ---------------------------------------------------------
        *Pk.stride(), *Vk.stride(), *bk.stride(),
        # ④ V-strides ---------------------------------------------------------
        *Pv.stride(), *Vv.stride(), *bv.stride(),
        # ⑤ Out-strides (three ints) -----------------------------------------
        *Out.stride(),
        # ⑥ sizes -------------------------------------------------------------
        M, R, H, softmax_scale,
    )

    grid = ((M + BLOCK_M - 1) // BLOCK_M, B * H)
    _demo_attn_kernel[grid](
        *args,
        BLOCK_M=BLOCK_M, BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D
    )

    out = Out.view(B, H, M, D)
    print("max-abs diff :", (ref - out).abs().max().item())
    print("rel-Frob err :", (torch.norm(ref - out) / torch.norm(ref)).item())


if __name__ == "__main__":
    main()


