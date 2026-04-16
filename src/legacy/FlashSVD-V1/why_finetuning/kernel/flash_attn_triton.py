import torch, math, triton, triton.language as tl

# ───────────────────────────────────────────────────────────────
# Triton kernels (as before)
# ───────────────────────────────────────────────────────────────

@triton.jit
def load_qkv_tile(
    ptr, off_b, off_h, row_off,
    sPb, sPh, sPm, sPd,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, seqlen
):
    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    ptrs = (
        ptr
        + off_b * sPb
        + off_h * sPh
        + offs_m[:, None] * sPm
        + offs_d[None, :] * sPd
    )
    mask = offs_m < seqlen
    return tl.load(ptrs, mask=mask[:, None], other=0.).to(tl.float32)


@triton.jit
def flashattn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, mask_ptr,
    sMb, sMh, sMq, sMk,
    sQb, sQh, sQm, sQd,
    sKb, sKh, sKm, sKd,
    sVb, sVh, sVm, sVd,
    sOb, sOm, sOd,
    seqlen, nheads, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bid = tl.program_id(0)
    bh  = tl.program_id(1)
    off_b = bh // nheads
    off_h = bh %  nheads
    row_off = bid * BLOCK_M

    q = load_qkv_tile(Q_ptr, off_b, off_h, row_off,
                      sQb, sQh, sQm, sQd,
                      BLOCK_M, BLOCK_D, seqlen)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for kb in range(0, seqlen, BLOCK_M):
        kb = tl.multiple_of(kb, BLOCK_M)
        offs_n = kb + tl.arange(0, BLOCK_M)
        mask_ptrs = (
            mask_ptr
            + off_b * sMb
            + off_h * sMh
            + 0      * sMq
            + offs_n * sMk
        )
        mask_i = tl.load(mask_ptrs, mask=offs_n<seqlen, other=0).to(tl.int32)
        mask_bool = mask_i > 0

        k = load_qkv_tile(K_ptr, off_b, off_h, kb,
                          sKb, sKh, sKm, sKd,
                          BLOCK_M, BLOCK_D, seqlen)
        v = load_qkv_tile(V_ptr, off_b, off_h, kb,
                          sVb, sVh, sVm, sVd,
                          BLOCK_M, BLOCK_D, seqlen)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neginf = tl.full(qk.shape, float("-inf"), tl.float32)
        qk = tl.where(mask_bool[None, :], qk, neginf)

        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        acc      = acc * exp_diff[:, None] \
                   + tl.dot(tl.exp(qk - m_new[:, None]), v)
        m_i      = m_new

    out = acc / tl.reshape(l_i, (BLOCK_M, 1))

    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = (
        Out_ptr
        + bh * sOb
        + offs_m[:, None] * sOm
        + offs_d[None, :] * sOd
    )
    tl.store(out_ptrs, out, mask=offs_m[:, None] < seqlen)


# ───────────────────────────────────────────────────────────────
# 1) Python wrapper around flashattn_kernel
# ───────────────────────────────────────────────────────────────
def flash_attn_triton(Q, K, V, mask, BLOCK_M=32):
    """
    Q, K, V: [B, H, M, D] float32/float16
    mask:    [B, H, 1, M] bool
    returns Out: [B, H, M, D] float32
    """
    B, H, M, D = Q.shape
    device = Q.device
    softmax_scale = 1.0 / math.sqrt(D)

    # allocate 3D output [B*H, M, D]
    Out = torch.empty(B * H, M, D, device=device, dtype=torch.float32)

    # pack arguments
    args = [
        Q, K, V, Out,
        mask, *mask.stride(),
        *Q.stride(),
        *K.stride(),
        *V.stride(),
        *Out.stride(),
        M, H, softmax_scale,
    ]
    grid = ((M + BLOCK_M - 1) // BLOCK_M, B * H)

    # launch kernel
    flashattn_kernel[grid](
        *args,
        BLOCK_M=BLOCK_M,
        BLOCK_D=D
    )

    return Out.view(B, H, M, D)


# ───────────────────────────────────────────────────────────────
# 2) Test harness using the wrapper
# ───────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(0)
    dev = "cuda"

    # config
    B, H, M, D, R = 16, 12, 128, 64, 32

    # create random low-rank factors & mask
    Pq = torch.randn(B, H, M, R, device=dev, dtype=torch.float16)
    Vq = torch.randn(B, H, R, D, device=dev, dtype=torch.float16)
    bq = torch.randn(B, H,   D, device=dev, dtype=torch.float16)
    Pk, Vk, bk = Pq.clone(), Vq.clone(), bq.clone()
    Pv, Vv, bv = Pq.clone(), Vq.clone(), bq.clone()

    true_lengths = torch.randint(36, M+1, (B,), device=dev)
    mask4d = torch.zeros(B, 1, 1, M, device=dev, dtype=torch.bool)
    for b in range(B):
        mask4d[b, 0, 0, : true_lengths[b]] = True
    attn_mask = mask4d.expand(B, H, 1, M)

    # compute full Q, K, V
    Q = (Pq.float().reshape(B*H, M, R) @ Vq.float().reshape(B*H, R, D)
         ).view(B, H, M, D) + bq.view(B, H, 1, D).float()
    K = (Pk.float().reshape(B*H, M, R) @ Vk.float().reshape(B*H, R, D)
         ).view(B, H, M, D) + bk.view(B, H, 1, D).float()
    V = (Pv.float().reshape(B*H, M, R) @ Vv.float().reshape(B*H, R, D)
         ).view(B, H, M, D) + bv.view(B, H, 1, D).float()
    
    # PyTorch reference
    scale = 1.0 / math.sqrt(D)
    logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale
    logits = logits.masked_fill(~attn_mask.squeeze(2).unsqueeze(2), -1e9)
    weights = torch.softmax(logits, dim=-1)
    ref = torch.einsum("bhmn,bhnd->bhmd", weights, V)

    # Triton result via wrapper
    out = flash_attn_triton(Q, K, V, attn_mask, BLOCK_M=32)

    # compare
    diff = (ref - out).abs()
    print("max-abs diff :", diff.max().item())
    print("rel-Frobenius error:",
          (torch.norm(diff) / torch.norm(ref)).item())


if __name__ == "__main__":
    main()

