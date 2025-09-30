import os, math, time, statistics
import numpy as np
import torch
import triton
import triton.language as tl

# ───────────────────────────────────────────────────────────────
# Tunables (expand if you want a wider search space)
# ───────────────────────────────────────────────────────────────
BM_CANDS = [16, 32, 64, 128]       # query/kv tile size along sequence
NW_CANDS = [2, 4, 8]          # Triton num_warps
NS_CANDS = [2, 3, 4]          # Triton pipeline stages

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
#    with autotuning
# ───────────────────────────────────────────────────────────────

def _fa_configs():
    cfgs = []
    for bm in BM_CANDS:
        for nw in NW_CANDS:
            for ns in NS_CANDS:
                cfgs.append(triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns))
    return cfgs

@triton.autotune(
    configs=_fa_configs(),
    key=['seqlen', 'BLOCK_D'],   # choose best per (sequence length, head dim)
)
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

    # load per‐query padding mask [B,H,1,M]
    pad_q_ptrs = (
        mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_m * sMk
    )
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m < seqlen, other=0).to(tl.int32)
    pad_q   = pad_q_i > 0                        # boolean [BLOCK_M]

    # load Q‐tile and zero out padded rows
    q = load_qkv_tile(Q_ptr, off_b, off_h, row_off,
                      sQb, sQh, sQm, sQd,
                      BLOCK_M, BLOCK_D, seqlen)
    q = q * pad_q[:, None]                      # zero padded queries

    # online‐softmax accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    # loop over K/V tiles
    for kb in range(0, seqlen, BLOCK_M):
        kb = tl.multiple_of(kb, BLOCK_M)
        offs_n = kb + tl.arange(0, BLOCK_M)      # [BLOCK_M]

        # padding mask for keys
        mask_ptrs = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_n * sMk
        mask_i = tl.load(mask_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
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

        # QKᵀ
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neginf = tl.full(qk.shape, float("-inf"), tl.float32)

        # combine padding + causal
        key_mask = pad_k[None, :] & causal
        qk = tl.where(key_mask, qk, neginf)

        # online softmax
        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        acc      = acc * exp_diff[:, None] + tl.dot(tl.exp(qk - m_new[:, None]), v)
        m_i      = m_new

    out = acc / tl.reshape(l_i, (BLOCK_M, 1))
    out = out * pad_q[:, None]  # zero padded queries again

    # write back
    Out_m = offs_m
    out_ptrs = Out_ptr + bh * sOb + Out_m[:, None] * sOm + offs_d[None, :] * sOd
    tl.store(out_ptrs, out, mask=pad_q[:, None])  # store only valid rows

# ───────────────────────────────────────────────────────────────
# 5) KV-Cache enabled FlashAttention kernel (autotuned)
# ───────────────────────────────────────────────────────────────

@triton.autotune(
    configs=_fa_configs(),
    key=['seq_len', 'kv_seq_len', 'BLOCK_D'],   # KV length impacts choice
)
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
    bid = tl.program_id(0)
    bh  = tl.program_id(1)
    off_b = bh // nheads
    off_h = bh %  nheads
    row_off = bid * BLOCK_M

    offs_m = row_off + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # query pad mask [B,H,1,seq_len]
    pad_q_ptrs = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_m * sMk
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m < seq_len, other=0).to(tl.int32)
    pad_q   = pad_q_i > 0

    # load Q
    q_ptrs = (
        Q_ptr + off_b * sQb + off_h * sQh
        + offs_m[:, None] * sQm + offs_d[None, :] * sQd
    )
    q_mask = offs_m < seq_len
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.).to(tl.float32)
    q = q * pad_q[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    past_len = kv_seq_len - seq_len

    for kb in range(0, kv_seq_len, BLOCK_M):
        kb = tl.multiple_of(kb, BLOCK_M)
        offs_n = kb + tl.arange(0, BLOCK_M)

        # all K/V valid up to kv_seq_len (if you need K/V padding, wire it here)
        k_mask = offs_n < kv_seq_len

        # causal: (query_index + past_len) >= key_index
        causal = (offs_m[:, None] + past_len) >= offs_n[None, :]

        k_ptrs = K_ptr + off_b * sKb + off_h * sKh + offs_n[:, None] * sKm + offs_d[None, :] * sKd
        v_ptrs = V_ptr + off_b * sVb + off_h * sVh + offs_n[:, None] * sVm + offs_d[None, :] * sVd
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.).to(tl.float32)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        neginf = tl.full(qk.shape, float("-inf"), tl.float32)

        key_mask = causal & k_mask[None, :]
        qk = tl.where(key_mask, qk, neginf)

        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        acc      = acc * exp_diff[:, None] + tl.dot(tl.exp(qk - m_new[:, None]), v)
        m_i      = m_new

    out = acc / tl.reshape(l_i, (BLOCK_M, 1))
    out = out * pad_q[:, None]

    out_ptrs = Out_ptr + bh * sOb + offs_m[:, None] * sOm + offs_d[None, :] * sOd
    tl.store(out_ptrs, out, mask=pad_q[:, None])

# ───────────────────────────────────────────────────────────────
# 3) Python wrappers (unchanged API)
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

    # upcast to float32 for stability
    Qf = Q.float()
    Kf = K.float()
    Vf = V.float()
    Out = torch.empty(B * H, M, D, device=device, dtype=torch.float32)

    args = [
        Qf, Kf, Vf, Out, mask,
        *mask.stride(),
        *Qf.stride(), *Kf.stride(), *Vf.stride(),
        *Out.stride(),
        M, H, softmax_scale,
    ]
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    flashattn_kernel[grid](
        *args,
        BLOCK_D=D,
    )

    Out = Out.view(B, H, M, D)
    return Out.to(orig_dtype)

def flash_attn_triton_kvcache(Q, K, V, mask, BLOCK_M=32):
    """
    KV-Cache enabled FlashAttention
    Q: [B, H, seq_len, D] - current queries
    K: [B, H, kv_seq_len, D] - past + current keys
    V: [B, H, kv_seq_len, D] - past + current values
    mask: [B, H, 1, seq_len] bool (padding mask for queries)
    returns Out: [B, H, seq_len, D]
    """
    B, H, seq_len, D = Q.shape
    kv_seq_len = K.shape[2]
    device = Q.device
    softmax_scale = 1.0 / math.sqrt(D)
    orig_dtype = Q.dtype

    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    Out = torch.empty(B * H, seq_len, D, device=device, dtype=torch.float32)

    args = [
        Qf, Kf, Vf, Out, mask,
        *mask.stride(),
        *Qf.stride(), *Kf.stride(), *Vf.stride(),
        *Out.stride(),
        seq_len, kv_seq_len, H, softmax_scale,
    ]
    grid = (triton.cdiv(seq_len, BLOCK_M), B * H)
    flashattn_kvcache_kernel[grid](
        *args,
        BLOCK_D=D,
    )

    Out = Out.view(B, H, seq_len, D)
    return Out.to(orig_dtype)

def flash_attn_triton_unified(Q, K, V, mask, BLOCK_M=32):
    """
    Chooses between standard or KV-Cache kernel based on sequence lengths.
    """
    seq_len = Q.shape[2]
    kv_seq_len = K.shape[2]
    if seq_len == kv_seq_len:
        return flash_attn_triton(Q, K, V, mask, BLOCK_M)
    else:
        return flash_attn_triton_kvcache(Q, K, V, mask, BLOCK_M)

# ───────────────────────────────────────────────────────────────
# 4) Profiling helpers (latency + peak memory)
# ───────────────────────────────────────────────────────────────

def _cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def bench_gpu(callable_fn, warmup=20, iters=100):
    """
    Returns (times_ms_list, mean_ms, p50_ms, p90_ms)
    """
    _cuda_synchronize()
    for _ in range(warmup):
        callable_fn()
    _cuda_synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start.record()
        callable_fn()
        end.record()
        _cuda_synchronize()
        times.append(start.elapsed_time(end))  # ms

    mean_ms = float(sum(times) / len(times))
    p50_ms  = float(np.percentile(times, 50))
    p90_ms  = float(np.percentile(times, 90))
    return times, mean_ms, p50_ms, p90_ms

def peak_memory_for(callable_fn):
    """
    Runs callable once and reports peak allocated & reserved (MiB).
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    _cuda_synchronize()
    callable_fn()
    _cuda_synchronize()
    alloc = torch.cuda.max_memory_allocated() / (1024**2)
    reserv = torch.cuda.max_memory_reserved() / (1024**2)
    return alloc, reserv

# ───────────────────────────────────────────────────────────────
# 5) Test harness: correctness + profiling (std & KV-cache)
# ───────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(0)
    device = "cuda"

    # small config for fast debug
    B, H, M, D, R = 8, 16, 512, 128, 16

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

    # ===================== Correctness (standard) =====================
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

    out = flash_attn_triton_unified(Q, K, V, attn_mask, BLOCK_M=32)

    # zero‐out padded queries on both sides
    pad_q = attn_mask.squeeze(2).unsqueeze(-1)   # [B,H,M,1]
    ref = ref * pad_q
    out = out * pad_q

    diff = (ref - out).abs()
    print(f"[Correctness/std] max-abs: {diff.max().item():.3e}  rel-Fro: {(torch.norm(diff)/torch.norm(ref)).item():.3e}")

    # ===================== Profiling (standard) =======================
    def _call_std():
        flash_attn_triton_unified(Q, K, V, attn_mask, BLOCK_M=32)

    std_alloc, std_resv = peak_memory_for(_call_std)
    _, std_mean, std_p50, std_p90 = bench_gpu(_call_std, warmup=20, iters=100)

    # ===================== KV-Cache Smoke + Profiling =================
    past = M // 2
    seq_len = M
    kv_seq_len = M + past

    # queries are last 'seq_len' chunk; K/V include past + current
    Q_kv = Q.clone()                              # [B,H,seq_len,D]
    K_kv = torch.cat([K[:, :, :past, :], K], dim=2)  # [B,H,kv_seq_len,D]
    V_kv = torch.cat([V[:, :, :past, :], V], dim=2)
    mask_kv = attn_mask.clone()                   # [B,H,1,seq_len]

    out_kv = flash_attn_triton_unified(Q_kv, K_kv, V_kv, mask_kv, BLOCK_M=32)
    assert torch.isfinite(out_kv).all(), "KV-cache output has NaNs/inf!"

    def _call_kv():
        flash_attn_triton_unified(Q_kv, K_kv, V_kv, mask_kv, BLOCK_M=32)

    kv_alloc, kv_resv = peak_memory_for(_call_kv)
    _, kv_mean, kv_p50, kv_p90 = bench_gpu(_call_kv, warmup=20, iters=100)

    # ===================== Torch SDPA reference profiling =============
    causal_full = torch.tril(torch.ones(M, M, device=device, dtype=torch.bool))
    # mask [B,H,M,M] as True=keep; convert to additive mask for SDPA
    keep = ref_mask
    add_mask = (~keep) * (-1e9)

    def _call_sdpa():
        # Torch SDPA in fp32 for fair comparison (we upcast in kernel)
        Qf, Kf, Vf = Q.float(), K.float(), V.float()
        torch.nn.functional.scaled_dot_product_attention(
            Qf, Kf, Vf,
            attn_mask=add_mask, dropout_p=0.0, is_causal=False
        )

    sdpa_alloc, sdpa_resv = peak_memory_for(_call_sdpa)
    _, sdpa_mean, sdpa_p50, sdpa_p90 = bench_gpu(_call_sdpa, warmup=10, iters=30)

    # ===================== Report =====================
    print("\n--- Peak memory (MiB) ---")
    print(f" Triton std    alloc={std_alloc:7.1f}   reserved={std_resv:7.1f}")
    print(f" Triton KV     alloc={kv_alloc:7.1f}   reserved={kv_resv:7.1f}")
    print(f" Torch SDPA    alloc={sdpa_alloc:7.1f}   reserved={sdpa_resv:7.1f}")

    print("\n--- Latency (ms) over CUDA events ---")
    print(f" Triton std    mean={std_mean:6.3f}   p50={std_p50:6.3f}   p90={std_p90:6.3f}")
    print(f" Triton KV     mean={kv_mean:6.3f}   p50={kv_p50:6.3f}   p90={kv_p90:6.3f}")
    print(f" Torch SDPA    mean={sdpa_mean:6.3f}   p50={sdpa_p50:6.3f}   p90={sdpa_p90:6.3f}")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required"
    # Optional: make Triton print its autotuning decisions
    # os.environ["TRITON_LOG_AUTOTUNING"] = "1"
    main()
