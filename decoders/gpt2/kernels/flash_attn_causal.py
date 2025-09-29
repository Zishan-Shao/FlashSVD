import torch, math, triton, triton.language as tl

# ───────────────────────────────────────────────────────────────
# 1) Triton tile loader for 3-D Q/K/V [B,H,M,D]
# ───────────────────────────────────────────────────────────────

@triton.jit
def load_qkv_tile(
    ptr, off_b, off_h, row_off,
    sPb, sPh, sPm, sPd,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, seqlen
):
    offs_m = row_off + tl.arange(0, BLOCK_M)               # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                         # [BLOCK_D]
    ptrs = (
        ptr
        + off_b * sPb
        + off_h * sPh
        + offs_m[:, None] * sPm
        + offs_d[None, :] * sPd
    )
    mask = offs_m < seqlen
    return tl.load(ptrs, mask=mask[:, None], other=0.).to(tl.float32)


# ───────────────────────────────────────────────────────────────
# 2) FlashAttention‐style causal + padding‐masked kernel
#    with explicit pad_q load & zero‐out
# Q,K,V: [B,H,M,D] → Out: [B*H, M, D]
# ───────────────────────────────────────────────────────────────

@triton.jit
def flashattn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, mask_ptr,
    # mask strides [B, H, 1, M]
    sMb, sMh, sMq, sMk,
    # Q,K,V strides [B, H, M, D]
    sQb, sQh, sQm, sQd,
    sKb, sKh, sKm, sKd,
    sVb, sVh, sVm, sVd,
    # Out strides [B*H, M, D]
    sOb, sOm, sOd,
    seqlen, nheads, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # which query‐tile (in M) and which batch*head
    bid = tl.program_id(0)
    bh  = tl.program_id(1)
    off_b = bh // nheads
    off_h = bh %  nheads
    row_off = bid * BLOCK_M      # start index for this Q‐tile

    # global query indices for this tile
    offs_m = row_off + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)             # [BLOCK_D]

    # ─── load per‐query padding mask ────────────────────────────────
    # mask_ptr: [B, H, 1, M], so stride sMq=1, sMk=M
    pad_q_ptrs = (
        mask_ptr
        + off_b * sMb
        + off_h * sMh
        + 0      * sMq
        + offs_m * sMk
    )  # shape [BLOCK_M]
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m<seqlen, other=0).to(tl.int32)
    pad_q   = pad_q_i > 0                        # boolean [BLOCK_M]

    # ─── load Q‐tile and zero out padded rows ──────────────────────
    q = load_qkv_tile(Q_ptr, off_b, off_h, row_off,
                      sQb, sQh, sQm, sQd,
                      BLOCK_M, BLOCK_D, seqlen)
    q = q * pad_q[:, None]                      # zero padded queries

    # online‐softmax accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    # ─── loop over K/V tiles ────────────────────────────────────────
    for kb in range(0, seqlen, BLOCK_M):
        kb = tl.multiple_of(kb, BLOCK_M)
        offs_n = kb + tl.arange(0, BLOCK_M)      # [BLOCK_M]

        # load padding mask for keys
        mask_ptrs = (
            mask_ptr
            + off_b * sMb
            + off_h * sMh
            + 0      * sMq
            + offs_n * sMk
        )
        mask_i = tl.load(mask_ptrs, mask=offs_n<seqlen, other=0).to(tl.int32)
        pad_k = mask_i > 0                        # [BLOCK_M]

        # causal mask: only allow j ≤ i
        causal = offs_m[:, None] >= offs_n[None, :]  # [BLOCK_M, BLOCK_M]

        # load K and V
        k = load_qkv_tile(K_ptr, off_b, off_h, kb,
                          sKb, sKh, sKm, sKd,
                          BLOCK_M, BLOCK_D, seqlen)
        v = load_qkv_tile(V_ptr, off_b, off_h, kb,
                          sVb, sVh, sVm, sVd,
                          BLOCK_M, BLOCK_D, seqlen)

        # compute QKᵀ
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neginf = tl.full(qk.shape, float("-inf"), tl.float32)

        # combine padding + causal for keys
        key_mask = pad_k[None, :] & causal
        qk = tl.where(key_mask, qk, neginf)

        # online‐softmax update
        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        acc      = acc * exp_diff[:, None] \
                   + tl.dot(tl.exp(qk - m_new[:, None]), v)
        m_i      = m_new

    # ─── finalize, zero‐out padded queries again ───────────────────
    out = acc / tl.reshape(l_i, (BLOCK_M, 1))
    out = out * pad_q[:, None]

    # write back
    Out_m = offs_m
    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = (
        Out_ptr
        + bh * sOb
        + Out_m[:, None] * sOm
        + offs_d[None, :] * sOd
    )
    tl.store(out_ptrs, out, mask=pad_q[:, None])  # store only valid rows


# ───────────────────────────────────────────────────────────────
# 3) Python wrapper around flashattn_kernel
# ───────────────────────────────────────────────────────────────

def flash_attn_triton(Q, K, V, mask, BLOCK_M=32):
    """
    Q, K, V: [B, H, M, D] float16/float32
    mask:    [B, H, 1, M] bool (padding mask)
    returns Out: [B, H, M, D] same dtype as Q
    """
    B, H, M, D = Q.shape
    device = Q.device
    softmax_scale = 1.0 / math.sqrt(D)
    orig_dtype = Q.dtype
    
    # print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}, mask: {mask.shape}")
    # print("Yes, I'm here")

    # Allocate output in original dtype; kernel upcasts internally per tile
    Out = torch.empty(B * H, M, D, device=device, dtype=orig_dtype)

    args = [
        Q, K, V, Out, mask,
        *mask.stride(),
        *Q.stride(), *K.stride(), *V.stride(),
        *Out.stride(),
        M, H, softmax_scale,
    ]
    grid = ((M + BLOCK_M - 1) // BLOCK_M, B * H)
    flashattn_kernel[grid](
        *args,
        BLOCK_M=BLOCK_M,
        BLOCK_D=D
    )

    Out = Out.view(B, H, M, D)
    return Out


# ───────────────────────────────────────────────────────────────
# 5) KV-Cache enabled FlashAttention kernel
#    Q: [B, H, seq_len, D], K/V: [B, H, kv_seq_len, D]
# ───────────────────────────────────────────────────────────────

@triton.jit
def flashattn_kvcache_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, mask_ptr,
    # mask strides [B, H, 1, seq_len] for queries
    sMb, sMh, sMq, sMk,
    # Q strides [B, H, seq_len, D]
    sQb, sQh, sQm, sQd,
    # K,V strides [B, H, kv_seq_len, D]  
    sKb, sKh, sKm, sKd,
    sVb, sVh, sVm, sVd,
    # Out strides [B*H, seq_len, D]
    sOb, sOm, sOd,
    seq_len, kv_seq_len, nheads, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # which query‐tile (in seq_len) and which batch*head
    bid = tl.program_id(0)
    bh  = tl.program_id(1)
    off_b = bh // nheads
    off_h = bh %  nheads
    row_off = bid * BLOCK_M      # start index for this Q‐tile

    # global query indices for this tile
    offs_m = row_off + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)             # [BLOCK_D]

    # ─── load per‐query padding mask ────────────────────────────────
    # mask_ptr: [B, H, 1, seq_len], so stride sMq=1, sMk=seq_len
    pad_q_ptrs = (
        mask_ptr
        + off_b * sMb
        + off_h * sMh
        + 0      * sMq
        + offs_m * sMk
    )  # shape [BLOCK_M]
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m<seq_len, other=0).to(tl.int32)
    pad_q   = pad_q_i > 0                        # boolean [BLOCK_M]

    # ─── load Q‐tile and zero out padded rows ──────────────────────
    q_ptrs = (
        Q_ptr
        + off_b * sQb
        + off_h * sQh
        + offs_m[:, None] * sQm
        + offs_d[None, :] * sQd
    )
    q_mask = offs_m < seq_len
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.).to(tl.float32)
    q = q * pad_q[:, None]                      # zero padded queries

    # online‐softmax accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    # ─── loop over K/V tiles (kv_seq_len) ───────────────────────────
    for kb in range(0, kv_seq_len, BLOCK_M):
        kb = tl.multiple_of(kb, BLOCK_M)
        offs_n = kb + tl.arange(0, BLOCK_M)      # [BLOCK_M]

        # For KV-Cache: assume all K/V positions are valid (no padding in past)
        # In practice, you might want to handle padding in K/V as well
        pad_k = tl.full((BLOCK_M,), True, tl.int1)  # All K/V positions valid

        # causal mask: allow attention to all past tokens + current/past positions
        # For KV-Cache: queries at position i can attend to all kv positions j <= (past_len + i)
        past_len = kv_seq_len - seq_len
        causal = (offs_m[:, None] + past_len) >= offs_n[None, :]  # [BLOCK_M, BLOCK_M]

        # load K and V
        k_ptrs = (
            K_ptr
            + off_b * sKb
            + off_h * sKh
            + offs_n[:, None] * sKm
            + offs_d[None, :] * sKd
        )
        v_ptrs = (
            V_ptr
            + off_b * sVb
            + off_h * sVh
            + offs_n[:, None] * sVm
            + offs_d[None, :] * sVd
        )
        
        k_mask = offs_n < kv_seq_len
        v_mask = offs_n < kv_seq_len
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.).to(tl.float32)
        v = tl.load(v_ptrs, mask=v_mask[:, None], other=0.).to(tl.float32)

        # compute QKᵀ
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neginf = tl.full(qk.shape, float("-inf"), tl.float32)

        # combine padding + causal for keys
        key_mask = pad_k[None, :] & causal & k_mask[None, :]
        qk = tl.where(key_mask, qk, neginf)

        # online‐softmax update
        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        acc      = acc * exp_diff[:, None] \
                   + tl.dot(tl.exp(qk - m_new[:, None]), v)
        m_i      = m_new

    # ─── finalize, zero‐out padded queries again ───────────────────
    out = acc / tl.reshape(l_i, (BLOCK_M, 1))
    out = out * pad_q[:, None]

    # write back
    Out_m = offs_m
    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = (
        Out_ptr
        + bh * sOb
        + Out_m[:, None] * sOm
        + offs_d[None, :] * sOd
    )
    tl.store(out_ptrs, out, mask=pad_q[:, None])  # store only valid rows


# ───────────────────────────────────────────────────────────────
# 6) Python wrapper for KV-Cache FlashAttention
# ───────────────────────────────────────────────────────────────

def flash_attn_triton_kvcache(Q, K, V, mask, BLOCK_M=32):
    """
    KV-Cache enabled FlashAttention
    Q: [B, H, seq_len, D] - current queries
    K: [B, H, kv_seq_len, D] - past + current keys  
    V: [B, H, kv_seq_len, D] - past + current values
    mask: [B, H, 1, seq_len] bool (padding mask for queries)
    returns Out: [B, H, seq_len, D] same dtype as Q
    """
    B, H, seq_len, D = Q.shape
    kv_seq_len = K.shape[2]
    device = Q.device
    softmax_scale = 1.0 / math.sqrt(D)
    orig_dtype = Q.dtype

    # Allocate output in original dtype; kernel upcasts internally per tile
    Out = torch.empty(B * H, seq_len, D, device=device, dtype=orig_dtype)

    args = [
        Q, K, V, Out, mask,
        *mask.stride(),
        *Q.stride(), *K.stride(), *V.stride(),
        *Out.stride(),
        seq_len, kv_seq_len, H, softmax_scale,
    ]
    grid = ((seq_len + BLOCK_M - 1) // BLOCK_M, B * H)
    flashattn_kvcache_kernel[grid](
        *args,
        BLOCK_M=BLOCK_M,
        BLOCK_D=D
    )

    Out = Out.view(B, H, seq_len, D)
    return Out


# ───────────────────────────────────────────────────────────────
# 7) Unified wrapper that chooses the right kernel
# ───────────────────────────────────────────────────────────────

def flash_attn_triton_unified(Q, K, V, mask, BLOCK_M=32):
    """
    Unified FlashAttention wrapper that automatically chooses between
    standard and KV-Cache kernels based on sequence lengths.
    
    Q: [B, H, seq_len, D] 
    K: [B, H, kv_seq_len, D] 
    V: [B, H, kv_seq_len, D]
    mask: [B, H, 1, seq_len] bool (padding mask for queries)
    """
    seq_len = Q.shape[2]
    kv_seq_len = K.shape[2]
    
    if seq_len == kv_seq_len:
        # Standard case: no KV cache
        return flash_attn_triton(Q, K, V, mask, BLOCK_M)
    else:
        # KV-Cache case: different sequence lengths
        return flash_attn_triton_kvcache(Q, K, V, mask, BLOCK_M)


# ───────────────────────────────────────────────────────────────
# 4) Test harness
# ───────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(0)
    device = "cuda"

    # small config for fast debug
    B, H, M, D, R = 4, 2, 64, 32, 16

    # random low‐rank factors + bias
    Pq = torch.randn(B, H, M, R, device=device, dtype=torch.float16)
    Vq = torch.randn(B, H, R, D, device=device, dtype=torch.float16)
    bq = torch.randn(B, H, D,    device=device, dtype=torch.float16)
    Pk, Vk, bk = Pq.clone(), Vq.clone(), bq.clone()
    Pv, Vv, bv = Pq.clone(), Vq.clone(), bq.clone()

    # random true lengths + padding mask [B,1,1,M]
    true_lengths = torch.randint(1, M+1, (B,), device=device)
    pad4d = torch.zeros(B, 1, 1, M, device=device, dtype=torch.bool)
    for b in range(B):
        pad4d[b, 0, 0, : true_lengths[b]] = True
    attn_mask = pad4d.expand(B, H, 1, M)

    # build full Q,K,V
    Q = (Pq.float().reshape(B*H, M, R) @ Vq.float().reshape(B*H, R, D)).view(B, H, M, D) \
        + bq.view(B, H, 1, D).float()
    K = (Pk.float().reshape(B*H, M, R) @ Vk.float().reshape(B*H, R, D)).view(B, H, M, D) \
        + bk.view(B, H, 1, D).float()
    V = (Pv.float().reshape(B*H, M, R) @ Vv.float().reshape(B*H, R, D)).view(B, H, M, D) \
        + bv.view(B, H, 1, D).float()

    # PyTorch reference
    scale = 1.0 / math.sqrt(D)
    logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale

    # pad mask → [B,H,M,M]
    pad = attn_mask.squeeze(2).unsqueeze(3)      # [B,H,M,1]
    pad2 = pad & pad.transpose(-1, -2)           # [B,H,M,M]
    causal = torch.tril(torch.ones(M, M, device=device, dtype=torch.bool))
    ref_mask = pad2 & causal

    logits = logits.masked_fill(~ref_mask, float("-1e9"))
    weights = torch.softmax(logits, dim=-1)
    ref = torch.einsum("bhmn,bhnd->bhmd", weights, V)
    
    # Triton result - use unified wrapper
    out = flash_attn_triton_unified(Q, K, V, attn_mask, BLOCK_M=16)

    # zero‐out padded queries on both sides
    pad_q = attn_mask.squeeze(2).unsqueeze(-1)   # [B,H,M,1]
    ref = ref * pad_q
    out = out * pad_q

    # metrics
    diff = (ref - out).abs()
    print("max-abs diff (valid+pad zeroed) :", diff.max().item())
    print("rel-Frobenius error :", (torch.norm(diff) / torch.norm(ref)).item())

if __name__ == "__main__":
    main()


