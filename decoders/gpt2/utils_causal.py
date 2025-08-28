# utils_mask_4D.py
import torch, triton, triton.language as tl
import math

# ───────────────────────────────────────────────────────────────
# 1) On-chip tile loader (unchanged)
# ───────────────────────────────────────────────────────────────
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
# 1.5) ASVD on-chip tile loader for cached factors
# ───────────────────────────────────────────────────────────────
@triton.jit
def load_asvd_tiles(
    P_ptr, V_ptr, bias_ptr,
    sPb, sPt, sPr,  # P factor strides: [B, T, rank]
    sVr, sVd,       # V factor strides: [rank, hidden_size] 
    sBb, sBh, sBd,  # bias strides: [B, H, D]
    BLOCK_X: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
    r_dim, off_b, off_h, row_offset,
):  
    """Load tiles for ASVD cached factors with shapes [B,T,rank] and [rank,hidden_size]"""
    offs_x = tl.arange(0, BLOCK_X)
    offs_d = tl.arange(0, BLOCK_D)
    r_idx  = tl.arange(0, BLOCK_R)
    acc = tl.zeros((BLOCK_X, BLOCK_D), dtype=tl.float32)
    
    for r_start in range(0, r_dim, BLOCK_R):
        mask_r = (r_start + r_idx) < r_dim
        
        # P factor: [B, T, rank] 
        P_ptrs = (
            P_ptr + off_b*sPb 
                  + (row_offset+offs_x)[:,None]*sPt
                  + (r_start+r_idx)[None,:]*sPr
        )
        
        # V factor: [rank, hidden_size] - we need to extract the slice for this head
        # hidden_size = nheads * head_dim, so head h starts at h * head_dim
        head_d_start = off_h * BLOCK_D  # offset within hidden_size for this head
        V_ptrs = (
            V_ptr + (r_start+r_idx)[:,None]*sVr
                  + (head_d_start + offs_d)[None,:]*sVd
        )
        
        # Load with original dtype and convert to float32 for computation
        P_sub = tl.load(P_ptrs, mask=mask_r[None,:], other=0.).to(tl.float32)
        V_sub = tl.load(V_ptrs, mask=mask_r[:,None], other=0.).to(tl.float32)
        acc += tl.dot(P_sub, V_sub)
    
    # Add bias: [B, H, D]
    b_ptrs = bias_ptr + off_b*sBb + off_h*sBh + offs_d*sBd
    acc += tl.load(b_ptrs).to(tl.float32)[None,:]
    return acc

# ───────────────────────────────────────────────────────────────
# 2) ASVD attention kernel for cached low-rank factors
# ───────────────────────────────────────────────────────────────
@triton.jit
def _asvd_attn_kernel(
    # Q factors (current input only)
    Pq_ptr, Vq_ptr, bq_ptr,
    # K factors (from cache)
    Pk_ptr, Vk_ptr, bk_ptr,
    # V factors (from cache)
    Pv_ptr, Vv_ptr, bv_ptr,
    # output
    Out_ptr,
    # mask + its 4 strides
    mask_ptr, sMb, sMh, sMq, sMk,
    # Q factor strides: Pq [B,T,rank], Vq [rank,hidden], bq [B,H,D]
    sPqb, sPqt, sPqr,
    sVqr, sVqd,
    sBqb, sBqh, sBqd,
    # K factor strides: Pk [B,T,rank], Vk [rank,hidden], bk [B,H,D]  
    sPkb, sPkt, sPkr,
    sVkr, sVkd,
    sBkb, sBkh, sBkd,
    # V factor strides: Pv [B,T,rank], Vv [rank,hidden], bv [B,H,D]
    sPvb, sPvt, sPvr,
    sVvr, sVvd,
    sBvb, sBvh, sBvd,
    # Out strides
    sOb, sOh, sOm,
    # sizes
    seqlen_q, seqlen_kv, r_dim_q, r_dim_k, r_dim_v, nheads, head_dim, softmax_scale,
    # tile sizes
    BLOCK_M: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh  = tl.program_id(1)
    off_b   = off_bh // nheads
    off_h   = off_bh %  nheads
    row_off = start_m * BLOCK_M

    # 1) Load Q tile on-chip (current input only)
    q = load_asvd_tiles(
        Pq_ptr, Vq_ptr, bq_ptr,
        sPqb, sPqt, sPqr,
        sVqr, sVqd,
        sBqb, sBqh, sBqd,
        BLOCK_M, BLOCK_R, BLOCK_D,
        r_dim_q, off_b, off_h, row_off,
    )

    # softmax accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # 2) iterate over cached key/value blocks
    for start_n in range(0, seqlen_kv, BLOCK_M):
        start_n = tl.multiple_of(start_n, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_M)
        
        # Load mask [B,H,1,N] (broadcasted from [B,1,1,N])
        mask_ptrs = (
            mask_ptr
          + off_b   * sMb    # batch
          + off_h   * sMh    # head (should be 0 for broadcasting)
          + 0       * sMq    # query (should be 0 for broadcasting)
          + (start_n + offs_n) * sMk  # key positions
        )
        mask_i32  = tl.load(mask_ptrs, mask=(start_n + offs_n) < seqlen_kv, other=0).to(tl.int32)
        mask_vals = mask_i32 > 0

        # Load K tile on-chip from cache
        k = load_asvd_tiles(
            Pk_ptr, Vk_ptr, bk_ptr,
            sPkb, sPkt, sPkr,
            sVkr, sVkd,
            sBkb, sBkh, sBkd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            r_dim_k, off_b, off_h, start_n,
        )
        
        # Load V tile on-chip from cache
        v = load_asvd_tiles(
            Pv_ptr, Vv_ptr, bv_ptr,
            sPvb, sPvt, sPvr,
            sVvr, sVvd,
            sBvb, sBvh, sBvd,
            BLOCK_M, BLOCK_R, BLOCK_D,
            r_dim_v, off_b, off_h, start_n,
        )

        # QK^T → apply mask → online softmax
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neg_inf = tl.full(qk.shape, float("-inf"), dtype=tl.float32)
        qk = tl.where(mask_vals[None, :], qk, neg_inf)

        # Online softmax update
        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:,None]), axis=1)
        acc      = acc * exp_diff[:,None] + tl.dot(tl.exp(qk - m_new[:,None]), v)
        m_i      = m_new

    # 3) finalize softmax
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
    tl.store(out_ptrs, out, mask=offs_m[:,None] < seqlen_q)

# ───────────────────────────────────────────────────────────────
# 3) Python wrapper for ASVD cached factors
# ───────────────────────────────────────────────────────────────
def flash_attn_asvd_cached(
    # Current Q factors 
    Pq_current, Vq_weight, bq_bias,
    # Cached K factors
    Pk_cached, Vk_weight, bk_bias,
    # Cached V factors  
    Pv_cached, Vv_weight, bv_bias,
    # Attention mask
    attn_mask,
    # Config
    softmax_scale=None
):
    """
    FlashAttention with ASVD cached low-rank factors
    
    Args:
        Pq_current: [B, T_q, rank_q] - Current input P factor for Q
        Vq_weight:  [rank_q, hidden_size] - Q projection weight 
        bq_bias:    [B, H, head_dim] - Q bias
        
        Pk_cached:  [B, T_kv, rank_k] - Cached P factor for K
        Vk_weight:  [rank_k, hidden_size] - K projection weight
        bk_bias:    [B, H, head_dim] - K bias
        
        Pv_cached:  [B, T_kv, rank_v] - Cached P factor for V  
        Vv_weight:  [rank_v, hidden_size] - V projection weight
        bv_bias:    [B, H, head_dim] - V bias
        
        attn_mask:  [B, H, 1, T_kv] - Attention mask
        softmax_scale: float - Scaling for attention scores
    
    Returns:
        out: [B, H, T_q, head_dim] - Attention output
    """
    B, T_q, rank_q = Pq_current.shape
    B, T_kv, rank_k = Pk_cached.shape
    B, T_kv, rank_v = Pv_cached.shape
    
    # Extract dimensions
    hidden_size = Vq_weight.shape[1]
    nheads = bq_bias.shape[1]
    head_dim = bq_bias.shape[2]
    
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Tile sizes
    BLOCK_M = min(64, T_q)
    BLOCK_R = min(32, max(rank_q, rank_k, rank_v))
    BLOCK_D = head_dim
    
    # Output tensor
    Out = torch.empty(B*nheads, T_q, head_dim, device=Pq_current.device, dtype=torch.float32)
    
    # Prepare arguments for kernel
    args = [
        # Q factors
        Pq_current, Vq_weight, bq_bias,
        # K factors  
        Pk_cached, Vk_weight, bk_bias,
        # V factors
        Pv_cached, Vv_weight, bv_bias,
        # Output
        Out,
        # Mask
        attn_mask, *attn_mask.stride(),
        # Q strides
        *Pq_current.stride(), *Vq_weight.stride(), *bq_bias.stride(),
        # K strides
        *Pk_cached.stride(), *Vk_weight.stride(), *bk_bias.stride(),
        # V strides  
        *Pv_cached.stride(), *Vv_weight.stride(), *bv_bias.stride(),
        # Output strides
        *Out.stride(),
        # Sizes
        T_q, T_kv, rank_q, rank_k, rank_v, nheads, head_dim, softmax_scale,
    ]
    
    # Launch kernel
    grid = ((T_q + BLOCK_M - 1) // BLOCK_M, B * nheads)
    _asvd_attn_kernel[grid](*args,
                            BLOCK_M=BLOCK_M,
                            BLOCK_R=BLOCK_R, 
                            BLOCK_D=BLOCK_D)
    
    # Reshape output to [B, H, T_q, head_dim]
    return Out.view(B, nheads, T_q, head_dim)


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
        
        # —— load mask [B,1,1,N]:
        #    sMh and sMq are both zero, so head & query dims broadcast
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



# ───────────────────────────────────────────────────────────────
# 3) Test harness with [B,H,1,M] broadcasted mask
# ───────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(0)
    dev = "cuda"

    # config - similar to ASVD model
    B, H, M, D, R = 1, 12, 128, 64, 32  # Single batch, 12 heads, 128 seq len, 64 head dim, 32 rank
    BLOCK_M, BLOCK_R, BLOCK_D = 32, 32, 64
    softmax_scale = 1.0 / math.sqrt(D)

    print("=== Testing ASVD Mixed-Precision Kernel ===")
    print(f"Config: B={B}, H={H}, M={M}, D={D}, R={R}")
    print(f"Using float16 precision for all inputs")

    # Create ASVD-style low-rank factors in float16 (like the model)
    # P factors: [B, T, rank] - current input projections
    Pq_current = torch.randn(B, M, R, device=dev, dtype=torch.float16).contiguous()
    Pk_cached = torch.randn(B, M, R, device=dev, dtype=torch.float16).contiguous()
    Pv_cached = torch.randn(B, M, R, device=dev, dtype=torch.float16).contiguous()
    
    # V factors: [rank, hidden_size] - weight matrices (like BLinear weights)
    Vq_weight = torch.randn(R, H * D, device=dev, dtype=torch.float16).contiguous()
    Vk_weight = torch.randn(R, H * D, device=dev, dtype=torch.float16).contiguous()
    Vv_weight = torch.randn(R, H * D, device=dev, dtype=torch.float16).contiguous()
    
    # Bias: [B, H, head_dim] - per-head biases
    bq_bias = torch.randn(B, H, D, device=dev, dtype=torch.float16).contiguous()
    bk_bias = torch.randn(B, H, D, device=dev, dtype=torch.float16).contiguous()
    bv_bias = torch.randn(B, H, D, device=dev, dtype=torch.float16).contiguous()

    # Create attention mask for full sequence
    attn_mask = torch.ones(B, H, 1, M, device=dev, dtype=torch.bool)
    print(f"Attention mask shape: {attn_mask.shape}")

    # Test 1: PyTorch reference implementation (reconstruct full Q, K, V)
    print("\n=== PyTorch Reference Implementation ===")
    
    # Reconstruct Q: Pq @ Vq + bias
    Q_full = torch.zeros(B, H, M, D, device=dev, dtype=torch.float32)
    for h in range(H):
        Vq_h = Vq_weight[:, h*D:(h+1)*D]  # [R, D] for this head
        Q_full[:, h, :, :] = (Pq_current.to(torch.float32) @ Vq_h.to(torch.float32)) + bq_bias[:, h:h+1, :].to(torch.float32)
    
    # Reconstruct K: Pk @ Vk + bias  
    K_full = torch.zeros(B, H, M, D, device=dev, dtype=torch.float32)
    for h in range(H):
        Vk_h = Vk_weight[:, h*D:(h+1)*D]  # [R, D] for this head
        K_full[:, h, :, :] = (Pk_cached.to(torch.float32) @ Vk_h.to(torch.float32)) + bk_bias[:, h:h+1, :].to(torch.float32)
    
    # Reconstruct V: Pv @ Vv + bias
    V_full = torch.zeros(B, H, M, D, device=dev, dtype=torch.float32)
    for h in range(H):
        Vv_h = Vv_weight[:, h*D:(h+1)*D]  # [R, D] for this head
        V_full[:, h, :, :] = (Pv_cached.to(torch.float32) @ Vv_h.to(torch.float32)) + bv_bias[:, h:h+1, :].to(torch.float32)

    # Standard attention computation
    logits = torch.einsum("bhmd,bhnd->bhmn", Q_full, K_full) * softmax_scale
    logits = logits.masked_fill(~attn_mask.squeeze(2).unsqueeze(2), float("-1e9"))
    weights = torch.softmax(logits, dim=-1)
    ref_output = torch.einsum("bhmn,bhnd->bhmd", weights, V_full)
    
    print(f"Reference output shape: {ref_output.shape}")

    # Test 2: ASVD Triton kernel implementation
    print("\n=== ASVD Triton Kernel Implementation ===")
    
    try:
        # Call our ASVD kernel
        asvd_output = flash_attn_asvd_cached(
            Pq_current, Vq_weight, bq_bias,
            Pk_cached, Vk_weight, bk_bias, 
            Pv_cached, Vv_weight, bv_bias,
            attn_mask,
            softmax_scale=softmax_scale
        )
        
        print(f"ASVD kernel output shape: {asvd_output.shape}")
        
        # Compare results
        diff = ref_output - asvd_output
        max_diff = diff.abs().max().item()
        rel_error = (torch.norm(diff) / torch.norm(ref_output)).item()
        
        print(f"✅ ASVD kernel test results:")
        print(f"  Max absolute difference: {max_diff:.6f}")
        print(f"  Relative Frobenius error: {rel_error:.6f}")
        
        if max_diff < 1e-3 and rel_error < 1e-3:
            print("✅ ASVD kernel working correctly!")
        else:
            print("⚠️  ASVD kernel may have issues")
            
    except Exception as e:
        print(f"❌ ASVD kernel failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
