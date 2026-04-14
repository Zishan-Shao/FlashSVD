# flash_attn_causal.py
# Causal + padding-masked Triton FlashAttention (with optional KV-cache)
# API:
#   flash_attn_triton(Q, K, V, mask, BLOCK_M=32)
#   flash_attn_triton_kvcache(Q, K, V, mask, BLOCK_M=32)
#   flash_attn_triton_unified(Q, K, V, mask, BLOCK_M=32)

import math
from typing import Tuple, Optional

import torch
import triton
import triton.language as tl

# ───────────────────────────────────────────────────────────────
# Tunables for autotuning (expand as needed)
# ───────────────────────────────────────────────────────────────
BM_CANDS = (16, 32, 64, 128)   # query/kv tile in sequence dimension
NW_CANDS = (2, 4, 8)           # Triton num_warps
NS_CANDS = (2, 3, 4)           # Triton pipeline stages


def _fa_configs():
    cfgs = []
    for bm in BM_CANDS:
        for nw in NW_CANDS:
            for ns in NS_CANDS:
                cfgs.append(triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns))
    return cfgs


@triton.jit
def _load_tile(ptr, off_b, off_h, row_off,
               sPb, sPh, sPm, sPd,
               BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, seqlen):
    """
    Load a (BLOCK_M x BLOCK_D) tile from a 4D tensor [B,H,M,D] with arbitrary strides.
    """
    offs_m = row_off + tl.arange(0, BLOCK_M)         # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                   # [BLOCK_D]
    ptrs = (ptr + off_b * sPb + off_h * sPh
                 + offs_m[:, None] * sPm + offs_d[None, :] * sPd)
    mask = offs_m < seqlen
    return tl.load(ptrs, mask=mask[:, None], other=0.)


@triton.autotune(configs=_fa_configs(), key=['seqlen', 'BLOCK_D'])
@triton.jit
def _flashattn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, mask_ptr,
    # mask strides [B, H, 1, M] (True=valid)
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
    """
    Causal + padding-masked attention for a full sequence (no KV-cache).
    Expects:
      - Q,K,V: [B,H,M,D]
      - mask:  [B,H,1,M] bool (True=valid)
      - Out:   [B*H,M,D] (row-major in M,D for coalesced writes)
    """
    # program ids: (tile id along M, flattened batch*head id)
    bid = tl.program_id(0)
    bh  = tl.program_id(1)
    off_b = bh // nheads
    off_h = bh %  nheads
    row_off = bid * BLOCK_M

    offs_m = row_off + tl.arange(0, BLOCK_M)    # query indices
    offs_d = tl.arange(0, BLOCK_D)

    # per-query padding [B,H,1,M] → [BLOCK_M]
    pad_q_ptrs = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_m * sMk
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m < seqlen, other=0).to(tl.int32)
    pad_q   = pad_q_i > 0

    # load Q tile (fp32) and zero out padded queries
    q = _load_tile(Q_ptr, off_b, off_h, row_off,
                   sQb, sQh, sQm, sQd,
                   BLOCK_M, BLOCK_D, seqlen)
    q = q * pad_q.to(q.dtype)[:, None]

    # online softmax accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    # loop over K/V tiles
    for kb in range(0, seqlen, BLOCK_M):
        kb = tl.multiple_of(kb, BLOCK_M)
        offs_n = kb + tl.arange(0, BLOCK_M)           # key indices

        # key padding mask
        pad_k_ptrs = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_n * sMk
        pad_k_i = tl.load(pad_k_ptrs, mask=offs_n < seqlen, other=0).to(tl.int32)
        pad_k   = pad_k_i > 0

        # causal mask j ≤ i
        causal = offs_m[:, None] >= offs_n[None, :]

        # load K,V tiles
        k = _load_tile(K_ptr, off_b, off_h, kb, sKb, sKh, sKm, sKd, BLOCK_M, BLOCK_D, seqlen)
        v = _load_tile(V_ptr, off_b, off_h, kb, sVb, sVh, sVm, sVd, BLOCK_M, BLOCK_D, seqlen).to(tl.float32)

        # compute scores
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * softmax_scale

        # combine masks; also invalidate out-of-range columns in last tile
        valid_col = offs_n < seqlen
        keep = (pad_k[None, :] & causal) & valid_col[None, :]
        neginf = tl.full(qk.shape, -float("inf"), tl.float32)
        qk = tl.where(keep, qk, neginf)

        # online softmax
        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        acc      = acc * exp_diff[:, None] + tl.dot(tl.exp(qk - m_new[:, None]), v)
        m_i      = m_new

    out = acc / tl.reshape(l_i, (BLOCK_M, 1))
    out = out * pad_q[:, None]   # re-apply padding on queries

    # write back
    out_ptrs = Out_ptr + bh * sOb + offs_m[:, None] * sOm + offs_d[None, :] * sOd
    tl.store(out_ptrs, out, mask=pad_q[:, None])


@triton.autotune(configs=_fa_configs(), key=['seq_len', 'kv_seq_len', 'BLOCK_D'])
@triton.jit
def _flashattn_kvcache_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, mask_ptr,
    # mask strides [B, H, 1, seq_len] for queries (True=valid)
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
    """
    Causal + padding-masked attention with KV-cache.
    Expects:
      - Q: [B,H,seq_len,D]
      - K,V: [B,H,kv_seq_len,D] (past + current)
      - mask: [B,H,1,seq_len] bool (True=valid queries)
      - Out: [B*H,seq_len,D]
    """
    bid = tl.program_id(0)
    bh  = tl.program_id(1)
    off_b = bh // nheads
    off_h = bh %  nheads
    row_off = bid * BLOCK_M

    offs_m = row_off + tl.arange(0, BLOCK_M)  # query
    offs_d = tl.arange(0, BLOCK_D)            # head dim

    # query padding mask
    pad_q_ptrs = mask_ptr + off_b * sMb + off_h * sMh + 0 * sMq + offs_m * sMk
    pad_q_i = tl.load(pad_q_ptrs, mask=offs_m < seq_len, other=0).to(tl.int32)
    pad_q   = pad_q_i > 0

    # load Q
    q_ptrs = Q_ptr + off_b * sQb + off_h * sQh + offs_m[:, None] * sQm + offs_d[None, :] * sQd
    q = tl.load(q_ptrs, mask=(offs_m < seq_len)[:, None], other=0.).to(tl.float32)
    q = q * pad_q[:, None]

    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    past_len = kv_seq_len - seq_len

    for kb in range(0, kv_seq_len, BLOCK_M):
        kb = tl.multiple_of(kb, BLOCK_M)
        offs_n = kb + tl.arange(0, BLOCK_M)

        # all keys valid up to kv_seq_len
        k_valid = offs_n < kv_seq_len

        # causal with offset: (query_index + past_len) >= key_index
        causal = (offs_m[:, None] + past_len) >= offs_n[None, :]

        k_ptrs = K_ptr + off_b * sKb + off_h * sKh + offs_n[:, None] * sKm + offs_d[None, :] * sKd
        v_ptrs = V_ptr + off_b * sVb + off_h * sVh + offs_n[:, None] * sVm + offs_d[None, :] * sVd
        k = tl.load(k_ptrs, mask=k_valid[:, None], other=0.).to(tl.float32)
        v = tl.load(v_ptrs, mask=k_valid[:, None], other=0.).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale

        keep = causal & k_valid[None, :]
        neginf = tl.full(qk.shape, -float("inf"), tl.float32)
        qk = tl.where(keep, qk, neginf)

        m_new    = tl.maximum(m_i, tl.max(qk, axis=1))
        exp_diff = tl.exp(m_i - m_new)
        l_i      = l_i * exp_diff + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        acc      = acc * exp_diff[:, None] + tl.dot(tl.exp(qk - m_new[:, None]), v)
        m_i      = m_new

    # Guard against zero denominator (can happen with extreme masks / underflow)
    den = tl.where(l_i > 0, l_i, 1.0)
    out = acc / tl.reshape(den, (BLOCK_M, 1))
    out = out * pad_q[:, None]

    out_ptrs = Out_ptr + bh * sOb + offs_m[:, None] * sOm + offs_d[None, :] * sOd
    tl.store(out_ptrs, out, mask=pad_q[:, None])


def _ensure_mask_bh1m(mask: torch.Tensor, B: int, H: int, M: int) -> torch.Tensor:
    """
    Accepts [B,H,1,M] or [B,1,1,M] or [B,M] and returns bool [B,H,1,M] (True=valid).
    """
    if mask is None:
        return torch.ones(B, H, 1, M, device='cuda', dtype=torch.bool)

    if mask.dtype != torch.bool:
        # Non-bool masks are treated as "1/0 validity"
        mask = mask != 0

    if mask.dim() == 2 and mask.shape == (B, M):
        mask = mask[:, None, None, :]            # [B,1,1,M]
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.expand(B, H, 1, M)           # [B,H,1,M]
    assert mask.shape == (B, H, 1, M), f"mask must be [B,H,1,M], got {tuple(mask.shape)}"
    return mask


@torch.no_grad()
def flash_attn_triton(Q: torch.Tensor,
                      K: torch.Tensor,
                      V: torch.Tensor,
                      mask: Optional[torch.Tensor],
                      BLOCK_M: int = 32) -> torch.Tensor:
    """
    Causal FlashAttention with padding.
      Q,K,V: [B,H,M,D] (fp16/bf16/fp32), arbitrary strides ok.
      mask : [B,H,1,M] bool (True=valid) or compatible (see _ensure_mask_bh1m).
    Returns:
      Out: [B,H,M,D] (same dtype as Q)
    Notes:
      - Computation is in fp32; final output is cast back to input dtype.
      - BLOCK_M is autotuned; value here is a hint and part of autotune key.
    """
    assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4, "Q,K,V must be [B,H,M,D]"
    B, H, M, D = Q.shape
    assert K.shape == (B, H, M, D) and V.shape == (B, H, M, D), "Q,K,V must share [B,H,M,D]"
    device = Q.device
    orig_dtype = Q.dtype
    softmax_scale = 1.0 / math.sqrt(D)

    mask_bh1m = _ensure_mask_bh1m(mask, B, H, M).contiguous()

    # Output is [B*H, M, D]; keep in input dtype to reduce peak mem
    Out = torch.empty(B * H, M, D, device=device, dtype=Q.dtype)

    # Kernel args
    args = [
        Q, K, V, Out, mask_bh1m,
        *mask_bh1m.stride(),
        *Q.stride(), *K.stride(), *V.stride(),
        *Out.stride(),
        M, H, softmax_scale,
    ]
    grid = (triton.cdiv(M, BLOCK_M), B * H)
    _flashattn_kernel[grid](*args, BLOCK_D=D)

    # Reshape back
    Out = Out.view(B, H, M, D)
    return Out.to(orig_dtype)


@torch.no_grad()
def flash_attn_triton_kvcache(Q: torch.Tensor,
                              K: torch.Tensor,
                              V: torch.Tensor,
                              mask: Optional[torch.Tensor],
                              BLOCK_M: int = 32) -> torch.Tensor:
    """
    Causal FlashAttention with KV-cache.
      Q: [B,H,seq_len,D]
      K: [B,H,kv_seq_len,D]
      V: [B,H,kv_seq_len,D]
      mask: [B,H,1,seq_len] bool (True=valid queries) or compatible
    Returns:
      Out: [B,H,seq_len,D]
    """
    assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4
    B, H, seq_len, D = Q.shape
    kv_seq_len = K.shape[2]
    assert K.shape == (B, H, kv_seq_len, D) and V.shape == (B, H, kv_seq_len, D)

    device = Q.device
    orig_dtype = Q.dtype
    softmax_scale = 1.0 / math.sqrt(D)

    # normalize mask to [B,H,1,seq_len]
    if mask is None:
        mask_bh1m = torch.ones(B, H, 1, seq_len, device=device, dtype=torch.bool)
    else:
        if mask.dtype != torch.bool:
            mask = mask != 0
        if mask.dim() == 2 and mask.shape == (B, seq_len):
            mask = mask[:, None, None, :]
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.expand(B, H, 1, seq_len)
        assert mask.shape == (B, H, 1, seq_len)
        mask_bh1m = mask.contiguous()

    # Keep output in input dtype to avoid large fp32 buffer
    Out = torch.empty(B * H, seq_len, D, device=device, dtype=Q.dtype)

    args = [
        Q, K, V, Out, mask_bh1m,
        *mask_bh1m.stride(),
        *Q.stride(), *K.stride(), *V.stride(),
        *Out.stride(),
        seq_len, kv_seq_len, H, softmax_scale,
    ]
    grid = (triton.cdiv(seq_len, BLOCK_M), B * H)
    _flashattn_kvcache_kernel[grid](*args, BLOCK_D=D)

    Out = Out.view(B, H, seq_len, D)
    return Out.to(orig_dtype)


@torch.no_grad()
def flash_attn_triton_unified(Q: torch.Tensor,
                              K: torch.Tensor,
                              V: torch.Tensor,
                              mask: Optional[torch.Tensor],
                              BLOCK_M: int = 32) -> torch.Tensor:
    """
    Convenience wrapper to choose standard vs KV-cache by sequence lengths.
    """
    if Q.shape[2] == K.shape[2] == V.shape[2]:
        return flash_attn_triton(Q, K, V, mask, BLOCK_M=BLOCK_M)
    return flash_attn_triton_kvcache(Q, K, V, mask, BLOCK_M=BLOCK_M)


__all__ = [
    "flash_attn_triton",
    "flash_attn_triton_kvcache",
    "flash_attn_triton_unified",
]
