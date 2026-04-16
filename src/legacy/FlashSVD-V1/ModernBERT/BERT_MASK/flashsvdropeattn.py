#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# python3 flashsvdropeattn.py --use_autotune

# python3 flashsvdropeattn.py --profile

# python3 flashsvdropeattn.py --bm 32 --bn 64


# ----------------------------
# Utilities (same RoPE helpers you used)
# ----------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


# ----------------------------
# Low-rank projection holder
# ----------------------------
@dataclass
class QKVFactors:
    # Rank-space inputs [B, M, R]
    Pq: torch.Tensor
    Pk: torch.Tensor
    Pv: torch.Tensor
    # Factors to lift back to head-space [R, H*dh]
    Vq: torch.Tensor
    Vk: torch.Tensor
    Vv: torch.Tensor
    # Optional biases on head-space (split per-head outside)
    bq: Optional[torch.Tensor] = None  # [H*dh] or None
    bk: Optional[torch.Tensor] = None
    bv: Optional[torch.Tensor] = None


# ----------------------------
# Triton kernel: FlashSVD + RoPE + online softmax
# ----------------------------
@triton.jit
def flashsvd_rope_sdpa(
    # Pq,Pk,Pv per-head activations [B,M,H,dh]
    Pq_ptr, Pk_ptr, Pv_ptr,
    # Vq,Vk,Vv per-head transformation matrices [H, dh, dh]
    Vq_ptr, Vk_ptr, Vv_ptr,
    # Optional biases per-head [H, dh] or 0
    bq_ptr, bk_ptr, bv_ptr,
    # RoPE cos/sin [B,H,M,dh]
    COS_ptr, SIN_ptr,
    # Output pointer O [B,H,M,dh]
    O_ptr,
    # Optional masks
    pad_mask_ptr,   # [B,M] (1 valid, 0 pad) or nullptr
    add_mask_ptr,   # [B,1,M,M] additive or nullptr
    # Shapes / strides
    B, H, M, dh,
    sPq_b, sPq_m, sPq_h, sPq_dh,  # P is [B,M,H,dh]
    sPk_b, sPk_m, sPk_h, sPk_dh,
    sPv_b, sPv_m, sPv_h, sPv_dh,
    sVq_h, sVq_dh_in, sVq_dh_out,  # V is [H, dh, dh]
    sVk_h, sVk_dh_in, sVk_dh_out,
    sVv_h, sVv_dh_in, sVv_dh_out,
    sbq_h, sbq_dh,  # bias is [H, dh]
    sbk_h, sbk_dh,
    sbv_h, sbv_dh,
    sCOS_b, sCOS_h, sCOS_m, sCOS_dh,
    sSIN_b, sSIN_h, sSIN_m, sSIN_dh,
    sO_b, sO_h, sO_m, sO_dh,
    sPM_b, sPM_m,           # pad mask strides
    sAM_b, sAM_mq, sAM_mk,  # additive mask strides (B, Mq, Mk)
    # Tiling
    BM: tl.constexpr, BN: tl.constexpr, BDH: tl.constexpr,
    # flags
    HAS_PAD: tl.constexpr, HAS_ADD: tl.constexpr,
    USE_TANH: tl.constexpr,
):
    # program ids
    bh   = tl.program_id(0)              # 0..B*H-1
    bid  = bh // H
    hid  = bh % H
    m_blk = tl.program_id(1)             # sequence block id

    # tile offsets
    offs_m = m_blk * BM + tl.arange(0, BM)       # query positions
    offs_d = tl.arange(0, BDH)                   # dh tile
    # head offset in the flattened [H*dh] dimension
    head_offset = hid * dh

    # --- online softmax state per (BM, dh) row ---
    m_i = tl.full((BM,), -float("inf"), dtype=tl.float32)   # running max over K
    l_i = tl.zeros((BM,), dtype=tl.float32)                 # running lSE
    acc = tl.zeros((BM, BDH), dtype=tl.float32)             # running output accumulator

    # precompute scale = 1/sqrt(dh) safely as a tensor
    dh_f = tl.full((1,), dh, dtype=tl.float32)
    scale = 1.0 / tl.sqrt(dh_f)

    # ----- MAIN LOOP over K/V sequence in blocks BN -----
    for nk in range(0, M, BN):
        offs_n = nk + tl.arange(0, BN)  # key positions
        valid_n = offs_n < M

        # Per-BN block logits accumulator across the whole dh (to be scaled later)
        scores = tl.zeros((BM, BN), dtype=tl.float32)

        # --- compute scores = QK^T over dh in tiles ---
        # We require BDH == dh (caller enforces), so this loop runs once.
        for d0 in range(0, dh, BDH):
            tl.static_assert(BDH % 2 == 0, "RoPE half–half requires even BDH")

            offs0 = tl.arange(0, BDH // 2)        # first half
            offs1 = offs0 + (BDH // 2)            # second half

            # Load cos/sin for first half only (pairwise rotation)
            cos_q0 = tl.load(COS_ptr + bid*sCOS_b + hid*sCOS_h + offs_m[:,None]*sCOS_m + offs0[None,:]*sCOS_dh,
                             mask=(offs_m[:,None] < M), other=0.0)
            sin_q0 = tl.load(SIN_ptr + bid*sSIN_b + hid*sSIN_h + offs_m[:,None]*sSIN_m + offs0[None,:]*sSIN_dh,
                             mask=(offs_m[:,None] < M), other=0.0)
            cos_k0 = tl.load(COS_ptr + bid*sCOS_b + hid*sCOS_h + offs_n[:,None]*sCOS_m + offs0[None,:]*sCOS_dh,
                             mask=(offs_n[:,None] < M), other=0.0)
            sin_k0 = tl.load(SIN_ptr + bid*sSIN_b + hid*sSIN_h + offs_n[:,None]*sSIN_m + offs0[None,:]*sSIN_dh,
                             mask=(offs_n[:,None] < M), other=0.0)

            # q/k half-buffers (fp32 math)
            # Load per-head P factors [B,M,H,dh] -> [BM, dh] and [BN, dh] for current head
            Pq_blk = tl.load(Pq_ptr + bid*sPq_b + offs_m[:,None]*sPq_m + hid*sPq_h + offs_d[None,:]*sPq_dh,
                             mask=(offs_m[:,None] < M), other=0.0)
            Pk_blk = tl.load(Pk_ptr + bid*sPk_b + offs_n[:,None]*sPk_m + hid*sPk_h + offs_d[None,:]*sPk_dh,
                             mask=(offs_n[:,None] < M), other=0.0)

            # Load per-head V transformation matrices [H, dh, dh] for current head
            # For Q: need V[hid, :, :]
            Vq_matrix = tl.load(Vq_ptr + hid*sVq_h + offs_d[:,None]*sVq_dh_in + offs_d[None,:]*sVq_dh_out,
                                mask=True, other=0.0)  # [dh, dh]
            Vk_matrix = tl.load(Vk_ptr + hid*sVk_h + offs_d[:,None]*sVk_dh_in + offs_d[None,:]*sVk_dh_out,
                                mask=True, other=0.0)  # [dh, dh]

            # Compute full Q, K via per-head matrix multiplication
            # Pq_blk: [BM, dh] @ Vq_matrix: [dh, dh] -> q_full: [BM, dh]
            q_full = tl.dot(Pq_blk, Vq_matrix.to(Pq_blk.dtype)).to(tl.float32)
            k_full = tl.dot(Pk_blk, Vk_matrix.to(Pk_blk.dtype)).to(tl.float32)

            # Split into halves for RoPE
            q0 = q_full[:, :BDH // 2]  # [BM, BDH//2] 
            q1 = q_full[:, BDH // 2:]  # [BM, BDH//2]
            k0 = k_full[:, :BDH // 2]  # [BN, BDH//2]
            k1 = k_full[:, BDH // 2:]  # [BN, BDH//2]

            # Add bias if present (per-head format [H, dh])
            if sbq_dh != 0:
                bq_head = tl.load(bq_ptr + hid*sbq_h + offs_d*sbq_dh, mask=True, other=0.0)
                bq0 = bq_head[:BDH // 2]
                bq1 = bq_head[BDH // 2:]
                q0 += bq0[None, :]; q1 += bq1[None, :]
            if sbk_dh != 0:
                bk_head = tl.load(bk_ptr + hid*sbk_h + offs_d*sbk_dh, mask=True, other=0.0)
                bk0 = bk_head[:BDH // 2]
                bk1 = bk_head[BDH // 2:]
                k0 += bk0[None, :]; k1 += bk1[None, :]

            # RoPE (pair i with i + BDH//2) using first-half angles
            q0r = q0 * cos_q0 - q1 * sin_q0
            q1r = q0 * sin_q0 + q1 * cos_q0
            k0r = k0 * cos_k0 - k1 * sin_k0
            k1r = k0 * sin_k0 + k1 * cos_k0

            scores += tl.dot(q0r, tl.trans(k0r))
            scores += tl.dot(q1r, tl.trans(k1r))

        # scale by 1/sqrt(dh)
        scores *= scale

        # ---- Masks ----
        if HAS_PAD:
            pm_k = tl.load(pad_mask_ptr + bid * sPM_b + offs_n * sPM_m, mask=valid_n, other=0).to(tl.int1)
            scores = tl.where(pm_k[None, :], scores, -float("inf"))

        if HAS_ADD:
            add = tl.load(add_mask_ptr + bid*sAM_b + offs_m[:, None]*sAM_mq + offs_n[None, :]*sAM_mk,
                          mask=(offs_m[:, None] < M) & valid_n[None, :], other=0.0)
            scores += add

        # ---- online softmax merge with running (m_i,l_i,acc) ----
        m_curr = tl.max(scores, 1)
        m_new = tl.maximum(m_i, m_curr)

        l_i *= tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        # Need V for acc update; compute v_blk using per-head transformation
        # Load per-head P_v factors [B,M,H,dh] -> [BN, dh] for current head
        Pv_blk = tl.load(Pv_ptr + bid*sPv_b + offs_n[:,None]*sPv_m + hid*sPv_h + offs_d[None,:]*sPv_dh,
                         mask=(offs_n[:,None] < M), other=0.0)
        
        # Load per-head V_v transformation matrix [H, dh, dh] for current head
        Vv_matrix = tl.load(Vv_ptr + hid*sVv_h + offs_d[:,None]*sVv_dh_in + offs_d[None,:]*sVv_dh_out,
                            mask=True, other=0.0)  # [dh, dh]

        # Compute V via per-head matrix multiplication
        # Pv_blk: [BN, dh] @ Vv_matrix: [dh, dh] -> v_blk: [BN, dh]
        v_blk = tl.dot(Pv_blk, Vv_matrix.to(Pv_blk.dtype)).to(tl.float32)

        # Add bias if present (per-head format [H, dh])
        if sbv_dh != 0:
            bv_head = tl.load(bv_ptr + hid*sbv_h + offs_d*sbv_dh, mask=True, other=0.0)
            v_blk += bv_head[None, :]

        # Update accumulator
        acc *= tl.exp(m_i[:, None] - m_new[:, None])
        acc += tl.dot(p, v_blk)

        l_i += tl.sum(p, 1)
        m_i = m_new

    # finalize output; guard against fully-masked rows
    den = tl.where(l_i > 0, l_i, 1.0)       # [BM]
    O_tile = acc / den[:, None]             # [BM, BDH]
    O_tile = tl.where(l_i[:, None] > 0, O_tile, 0.0)
    tl.store(O_ptr + bid*sO_b + hid*sO_h + offs_m[:, None]*sO_m + tl.arange(0, BDH)[None, :]*sO_dh,
             O_tile, mask=offs_m[:, None] < M)



# ----------------------------
# Autotuned variant (BM/BN/BR tuned; BDH passed by caller == dh)
# ----------------------------
# Optimized configs based on profiling - BM=64, BN=64, BR=32 is the best performer
TUNE_CONFIGS_ROPE: List[triton.Config] = [
    triton.Config({'BM': 64,  'BN': 64,  'BR': 32},  num_warps=4, num_stages=2),  # OPTIMAL CONFIG
    triton.Config({'BM': 32,  'BN': 64,  'BR': 32},  num_warps=4, num_stages=2),
    triton.Config({'BM': 64,  'BN': 128, 'BR': 32},  num_warps=4, num_stages=2),
]

@triton.autotune(configs=TUNE_CONFIGS_ROPE, key=['M', 'R', 'dh'])
@triton.jit
def flashsvd_rope_sdpa_auto(
    Pq_ptr, Pk_ptr, Pv_ptr,
    Vq_ptr, Vk_ptr, Vv_ptr,
    bq_ptr, bk_ptr, bv_ptr,
    COS_ptr, SIN_ptr,
    O_ptr,
    pad_mask_ptr, add_mask_ptr,
    B, H, M, R, dh,
    sPq_b, sPq_m, sPq_r,
    sPk_b, sPk_m, sPk_r,
    sPv_b, sPv_m, sPv_r,
    sVq_r, sVq_hd,
    sVk_r, sVk_hd,
    sVv_r, sVv_hd,
    sbq_hd, sbk_hd, sbv_hd,
    sCOS_b, sCOS_h, sCOS_m, sCOS_dh,
    sSIN_b, sSIN_h, sSIN_m, sSIN_dh,
    sO_b, sO_h, sO_m, sO_dh,
    sPM_b, sPM_m,
    sAM_b, sAM_mq, sAM_mk,
    # We still require BDH==dh; pass it explicitly (not part of the autotune configs)
    BDH: tl.constexpr,
    HAS_PAD: tl.constexpr, HAS_ADD: tl.constexpr,
    USE_TANH: tl.constexpr,
    # BM/BN/BR are provided by the chosen config via tl.meta
):
    # Same body via call to the base kernel (ensures single source of truth)
    flashsvd_rope_sdpa(
        Pq_ptr, Pk_ptr, Pv_ptr,
        Vq_ptr, Vk_ptr, Vv_ptr,
        bq_ptr, bk_ptr, bv_ptr,
        COS_ptr, SIN_ptr,
        O_ptr,
        pad_mask_ptr, add_mask_ptr,
        B, H, M, R, dh,
        sPq_b, sPq_m, sPq_r,
        sPk_b, sPk_m, sPk_r,
        sPv_b, sPv_m, sPv_r,
        sVq_r, sVq_hd,
        sVk_r, sVk_hd,
        sVv_r, sVv_hd,
        sbq_hd, sbk_hd, sbv_hd,
        sCOS_b, sCOS_h, sCOS_m, sCOS_dh,
        sSIN_b, sSIN_h, sSIN_m, sSIN_dh,
        sO_b, sO_h, sO_m, sO_dh,
        sPM_b, sPM_m,
        sAM_b, sAM_mq, sAM_mk,
        BM=tl.meta['BM'], BN=tl.meta['BN'], BDH=BDH, BR=tl.meta['BR'],
        HAS_PAD=HAS_PAD, HAS_ADD=HAS_ADD, USE_TANH=USE_TANH,
    )


# ----------------------------
# Public module
# ----------------------------
class FlashSVDRoPEAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, rotary_emb, *,
                 bm=64, bn=64, bdh=None, br=32,
                 use_autotune: bool = True,
                 pinned_cfg: Optional[dict] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.bm = bm
        self.bn = bn
        self.bdh = head_dim if bdh is None else bdh
        self.br = br
        self.use_autotune = bool(use_autotune)
        self.pinned_cfg = pinned_cfg or {}
        # IMPORTANT: correctness requires BDH == dh with current RoPE indexing
        assert self.bdh == head_dim, "Kernel currently expects BDH == dh (one dh stripe)."

    @staticmethod
    def _padding_mask_bool(attention_mask_2d: torch.Tensor) -> torch.Tensor:
        return ~(attention_mask_2d.to(torch.bool))[:, None, None, :]

    @torch.no_grad()
    def forward(self,
        qkv_factors: QKVFactors,
        attention_mask: Optional[torch.Tensor],      # 2D padding or 4D additive
        position_ids: torch.Tensor,                  # [B,M]
        sliding_window_mask: Optional[torch.Tensor] = None,  # (optional) 4D additive
    ) -> torch.Tensor:
        Pq, Pk, Pv = qkv_factors.Pq, qkv_factors.Pk, qkv_factors.Pv
        Vq, Vk, Vv = qkv_factors.Vq, qkv_factors.Vk, qkv_factors.Vv
        bq, bk, bv = qkv_factors.bq, qkv_factors.bk, qkv_factors.bv

        B, M, R = Pq.shape
        H, dh = self.num_heads, self.head_dim
        device = Pq.device
        dtype = Pq.dtype
        assert dh == self.bdh, "BDH must equal head_dim."

        # RoPE cos/sin for (B,H,M,dh)
        dummy = torch.empty((B * H, M, dh), device=device, dtype=dtype)
        posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
        cos, sin = self.rotary_emb(dummy, position_ids=posf)  # [(B*H), M, dh]
        cos = cos.view(B, H, M, dh).contiguous()
        sin = sin.view(B, H, M, dh).contiguous()

        # Prepare masks
        pad_mask_ptr = None
        add_mask_ptr = None
        has_pad = 0
        has_add = 0
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask_ptr = attention_mask.contiguous()
                has_pad = 1
            elif attention_mask.dim() == 4:
                add_mask_ptr = attention_mask.contiguous()
                has_add = 1
            else:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")
        if sliding_window_mask is not None:
            add_mask_ptr = sliding_window_mask.contiguous()
            has_add = 1

        # Output buffer [B,H,M,dh]
        O = torch.empty((B, H, M, dh), device=device, dtype=dtype)

        # Strides
        sPq_b, sPq_m, sPq_r = Pq.stride()
        sPk_b, sPk_m, sPk_r = Pk.stride()
        sPv_b, sPv_m, sPv_r = Pv.stride()
        sVq_r, sVq_hd = Vq.stride()
        sVk_r, sVk_hd = Vk.stride()
        sVv_r, sVv_hd = Vv.stride()
        sbq_hd = bq.stride(0) if bq is not None else 0
        sbk_hd = bk.stride(0) if bk is not None else 0
        sbv_hd = bv.stride(0) if bv is not None else 0
        sCOS_b, sCOS_h, sCOS_m, sCOS_dh = cos.stride()
        sSIN_b, sSIN_h, sSIN_m, sSIN_dh = sin.stride()
        sO_b, sO_h, sO_m, sO_dh = O.stride()
        if has_pad:
            sPM_b, sPM_m = pad_mask_ptr.stride()
        else:
            sPM_b = sPM_m = 0
        if has_add:
            sAM_b, sAM_1, sAM_mq, sAM_mk = add_mask_ptr.stride()
        else:
            sAM_b = sAM_mq = sAM_mk = 0

        # Launch
        if self.use_autotune:
            # Try autotune first, fall back to manual config if it fails
            try:
                grid = lambda meta: (B * H, triton.cdiv(M, meta['BM']))
                flashsvd_rope_sdpa_auto[grid](
                    Pq, Pk, Pv,
                    Vq, Vk, Vv,
                    bq if bq is not None else O,
                    bk if bk is not None else O,
                    bv if bv is not None else O,
                    cos, sin,
                    O,
                    pad_mask_ptr if has_pad else O,
                    add_mask_ptr if has_add else O,
                    B, H, M, R, dh,
                    sPq_b, sPq_m, sPq_r,
                    sPk_b, sPk_m, sPk_r,
                    sPv_b, sPv_m, sPv_r,
                    sVq_r, sVq_hd,
                    sVk_r, sVk_hd,
                    sVv_r, sVv_hd,
                    sbq_hd, sbk_hd, sbv_hd,
                    sCOS_b, sCOS_h, sCOS_m, sCOS_dh,
                    sSIN_b, sSIN_h, sSIN_m, sSIN_dh,
                    sO_b, sO_h, sO_m, sO_dh,
                    sPM_b, sPM_m,
                    sAM_b, sAM_mq, sAM_mk,
                    BDH=self.bdh,
                    HAS_PAD=has_pad, HAS_ADD=has_add,
                    USE_TANH=1,
                )
            except Exception as e:
                # Fall back to manual config if autotune fails
                if not hasattr(self, '_autotune_fallback_warned'):
                    print(f"Warning: Autotune failed ({type(e).__name__}), falling back to manual config")
                    self._autotune_fallback_warned = True
                
                BM = 64
                BN = 64  
                BR = 32
                num_warps = 4
                num_stages = 2
                grid = (B * H, triton.cdiv(M, BM))
                flashsvd_rope_sdpa[grid](
                Pq, Pk, Pv,
                Vq, Vk, Vv,
                bq if bq is not None else O,
                bk if bk is not None else O,
                bv if bv is not None else O,
                cos, sin,
                O,
                pad_mask_ptr if has_pad else O,
                add_mask_ptr if has_add else O,
                B, H, M, R, dh,
                sPq_b, sPq_m, sPq_r,
                sPk_b, sPk_m, sPk_r,
                sPv_b, sPv_m, sPv_r,
                sVq_r, sVq_hd,
                sVk_r, sVk_hd,
                sVv_r, sVv_hd,
                sbq_hd, sbk_hd, sbv_hd,
                sCOS_b, sCOS_h, sCOS_m, sCOS_dh,
                sSIN_b, sSIN_h, sSIN_m, sSIN_dh,
                sO_b, sO_h, sO_m, sO_dh,
                sPM_b, sPM_m,
                sAM_b, sAM_mq, sAM_mk,
                BM=BM, BN=BN, BDH=self.bdh, BR=BR,
                HAS_PAD=has_pad, HAS_ADD=has_add,
                USE_TANH=1,
                num_warps=num_warps, num_stages=num_stages
            )
        else:
            # Pinned (manual) config path
            BM = int(self.pinned_cfg.get('BM', self.bm))
            BN = int(self.pinned_cfg.get('BN', self.bn))
            BR = int(self.pinned_cfg.get('BR', self.br))
            num_warps  = int(self.pinned_cfg.get('num_warps', 4))
            num_stages = int(self.pinned_cfg.get('num_stages', 2))
            grid = (B * H, triton.cdiv(M, BM))
            flashsvd_rope_sdpa[grid](
                Pq, Pk, Pv,
                Vq, Vk, Vv,
                bq if bq is not None else O,
                bk if bk is not None else O,
                bv if bv is not None else O,
                cos, sin,
                O,
                pad_mask_ptr if has_pad else O,
                add_mask_ptr if has_add else O,
                B, H, M, R, dh,
                sPq_b, sPq_m, sPq_r,
                sPk_b, sPk_m, sPk_r,
                sPv_b, sPv_m, sPv_r,
                sVq_r, sVq_hd,
                sVk_r, sVk_hd,
                sVv_r, sVv_hd,
                sbq_hd, sbk_hd, sbv_hd,
                sCOS_b, sCOS_h, sCOS_m, sCOS_dh,
                sSIN_b, sSIN_h, sSIN_m, sSIN_dh,
                sO_b, sO_h, sO_m, sO_dh,
                sPM_b, sPM_m,
                sAM_b, sAM_mq, sAM_mk,
                BM=BM, BN=BN, BDH=self.bdh, BR=BR,
                HAS_PAD=has_pad, HAS_ADD=has_add,
                USE_TANH=1,
                num_warps=num_warps, num_stages=num_stages,
            )
        return O  # [B,H,M,dh]


# ----------------------------
# Minimal test harness + profiling
# ----------------------------
def _rotary_emb_make(seq_len, dim, base=10000.0, device="cuda", dtype=torch.float32):
    assert dim % 2 == 0, "RoPE dim must be even."
    half = dim // 2
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    freqs = torch.einsum("m,d->md", pos, inv_freq)  # [M, half]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, dim)
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, dim)
    return cos, sin

class _SimpleRotary:
    def __init__(self, base=10000.0):
        self.base = base
    def __call__(self, q_like: torch.Tensor, *, position_ids: torch.Tensor):
        BH, M, dh = q_like.shape
        device = q_like.device
        dtype  = q_like.dtype
        cos_tab, sin_tab = _rotary_emb_make(M, dh, base=self.base, device=device, dtype=dtype)
        cos = cos_tab.index_select(0, position_ids[0].to(torch.long)).unsqueeze(0).expand(BH, -1, -1).contiguous()
        sin = sin_tab.index_select(0, position_ids[0].to(torch.long)).unsqueeze(0).expand(BH, -1, -1).contiguous()
        return cos, sin

def _apply_rope_torch(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x1.shape[-1]],
                      x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x1.shape[-1]]], dim=-1)

@torch.no_grad()
def _reference_sdpa_with_rope(Pq, Pk, Pv, Vq, Vk, Vv, bq, bk, bv, cos, sin, attention_mask_2d=None):
    B, M, R = Pq.shape
    Hdh = Vq.shape[1]
    H = cos.shape[1]
    dh = Hdh // H

    Q = torch.matmul(Pq, Vq)
    K = torch.matmul(Pk, Vk)
    V = torch.matmul(Pv, Vv)
    if bq is not None: Q = Q + bq
    if bk is not None: K = K + bk
    if bv is not None: V = V + bv
    Q = Q.view(B, M, H, dh).permute(0, 2, 1, 3).contiguous()
    K = K.view(B, M, H, dh).permute(0, 2, 1, 3).contiguous()
    V = V.view(B, M, H, dh).permute(0, 2, 1, 3).contiguous()

    Qr = _apply_rope_torch(Q, cos, sin)
    Kr = _apply_rope_torch(K, cos, sin)

    scale = 1.0 / math.sqrt(dh)
    scores = torch.einsum("bhmd,bhnd->bhmn", Qr, Kr) * scale

    if attention_mask_2d is not None:
        am = attention_mask_2d.to(torch.bool)
        qv = am[:, None, :, None]
        kv = am[:, None, None, :]
        mask = qv & kv
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    Oref = torch.einsum("bhmn,bhnd->bhmd", attn, V)
    return Oref

def _pretty_mem(bytes_val: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.0f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.0f} PB"

# ---- Manual profiler over pinned configs ----
def profile_flashsvd_rope(
    flash_mod: FlashSVDRoPEAttention,
    qkv: QKVFactors,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    configs: List[dict],
    warmup: int = 5,
    iters: int = 20,
):
    device = qkv.Pq.device
    B, M, R = qkv.Pq.shape
    Hdh = qkv.Vq.shape[1]
    H = flash_mod.num_heads
    dh = flash_mod.head_dim
    assert Hdh == H * dh

    results = []
    # temp instance that we switch to pinned mode for each config
    runner = FlashSVDRoPEAttention(
        num_heads=H, head_dim=dh, rotary_emb=flash_mod.rotary_emb,
        bdh=dh, use_autotune=False
    ).to(device)

    # Warmup kernel compile once (first config)
    runner.pinned_cfg = configs[0]
    for _ in range(max(1, warmup)):
        _ = runner(qkv, attention_mask, position_ids)

    # Measure per config
    for cfg in configs:
        runner.pinned_cfg = cfg
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(iters):
            start_evt.record()
            _ = runner(qkv, attention_mask, position_ids)
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))  # ms
        mean_ms = float(sum(times) / len(times))
        results.append((mean_ms, cfg))

    results.sort(key=lambda x: x[0])
    print("\n=== Manual profile over FlashSVD RoPE configs ===")
    for mean_ms, cfg in results:
        print(f"BM={cfg['BM']:>3} BN={cfg['BN']:>3} BR={cfg['BR']:>3} | "
              f"warps={cfg.get('num_warps',4)} stages={cfg.get('num_stages',2)} | "
              f"mean={mean_ms:.3f} ms")
    best = results[0]
    print(f"\nBest config: BM={best[1]['BM']}, BN={best[1]['BN']}, BR={best[1]['BR']}, "
          f"warps={best[1].get('num_warps',4)}, stages={best[1].get('num_stages',2)} | "
          f"mean={best[0]:.3f} ms")
    return best


if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser("FlashSVD+RoPE kernel test (with autotune + profiling)")
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--M", type=int, default=256*4)
    parser.add_argument("--dh", type=int, default=128)
    parser.add_argument("--R", type=int, default=32, help="rank-space dimension")
    parser.add_argument("--bm", type=int, default=64)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--mask", action="store_true", help="enable 2D padding mask with random pads")
    parser.add_argument("--profile", action="store_true", help="run manual profiling sweep")
    parser.add_argument("--use_autotune", action="store_true", help="use autotuned kernel instead of pinned bm/bn")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this test.")
        raise SystemExit(1)

    # Clean memory at start for accurate measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(123)
    device = "cuda"
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    B, H, M, dh, R = args.B, args.H, args.M, args.dh, args.R
    D = H * dh

    # Random rank-space activations and factors
    Pq = torch.randn(B, M, R, device=device, dtype=dtype)
    Pk = torch.randn(B, M, R, device=device, dtype=dtype)
    Pv = torch.randn(B, M, R, device=device, dtype=dtype)

    Vq = torch.randn(R, D, device=device, dtype=dtype).contiguous()
    Vk = torch.randn(R, D, device=device, dtype=dtype).contiguous()
    Vv = torch.randn(R, D, device=device, dtype=dtype).contiguous()

    bq = torch.randn(D, device=device, dtype=dtype).contiguous()
    bk = torch.randn(D, device=device, dtype=dtype).contiguous()
    bv = torch.randn(D, device=device, dtype=dtype).contiguous()

    # simple position ids and rotary
    position_ids = torch.arange(M, device=device)[None, :].expand(B, -1)  # [B,M]
    rotary = _SimpleRotary(base=10000.0)

    # Prepare cos/sin for reference (shape [B,H,M,dh])
    dummy = torch.empty((B*H, M, dh), device=device, dtype=dtype)
    posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
    cos, sin = rotary(dummy, position_ids=posf)
    cos = cos.view(B, H, M, dh).contiguous()
    sin = sin.view(B, H, M, dh).contiguous()

    # Optional 2D padding mask
    attention_mask = None
    if args.mask:
        valid_len = int(0.9 * M)
        attention_mask = torch.zeros(B, M, device=device, dtype=torch.int32)
        attention_mask[:, :valid_len] = 1

    # Build QKVFactors struct
    qkv = QKVFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

    # Create module
    flash = FlashSVDRoPEAttention(
        num_heads=H, head_dim=dh, rotary_emb=rotary,
        bm=args.bm, bn=args.bn, bdh=dh, br=32,
        use_autotune=args.use_autotune,
        pinned_cfg={'BM': args.bm, 'BN': args.bn, 'BR': 32, 'num_warps': 4, 'num_stages': 2}
    ).to(device)

    # Optional: manual profiling sweep (powers-of-two only)
    if args.profile:
        sweep = [
            {'BM': 16,  'BN': 16,  'BR': 16, 'num_warps': 4, 'num_stages': 2},
            {'BM': 32,  'BN': 32,  'BR': 32, 'num_warps': 4, 'num_stages': 2},
            {'BM': 64,  'BN': 64,  'BR': 64, 'num_warps': 8, 'num_stages': 2},
            {'BM': 32,  'BN': 64,  'BR': 32, 'num_warps': 4, 'num_stages': 2},
            {'BM': 64,  'BN': 64,  'BR': 32, 'num_warps': 4, 'num_stages': 2},
            {'BM': 64,  'BN': 128, 'BR': 32, 'num_warps': 4, 'num_stages': 2},
            {'BM': 64,  'BN': 128, 'BR': 64, 'num_warps': 8, 'num_stages': 2},
            {'BM': 128, 'BN': 64,  'BR': 32, 'num_warps': 8, 'num_stages': 2},
            # Commented out: shared memory exceeds hardware limit (131584 > 101376)
            #{'BM': 128, 'BN': 128, 'BR': 32, 'num_warps': 8, 'num_stages': 2},
            #{'BM': 128, 'BN': 128, 'BR': 64, 'num_warps': 8, 'num_stages': 1},
        ]
        best = profile_flashsvd_rope(flash, qkv, position_ids, attention_mask, sweep, warmup=5, iters=20)
        # Pin best for the rest of the run
        flash.use_autotune = False
        flash.pinned_cfg = best[1]

    # ---- Warmup (compile) ----
    for _ in range(max(1, args.warmup)):
        _ = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()

    # ---- Measure latency (kernel) ----
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(args.iters):
        start.record()
        O_kernel = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    ms_kernel = float(sum(times) / len(times))

    # ---- Peak memory for a single forward ----
    torch.cuda.reset_peak_memory_stats()
    _ = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    # ---- Reference (fp32) ----
    Pq32, Pk32, Pv32 = Pq.float(), Pk.float(), Pv.float()
    Vq32, Vk32, Vv32 = Vq.float(), Vk.float(), Vv.float()
    bq32, bk32, bv32 = bq.float(), bk.float(), bv.float()
    cos32, sin32 = cos.float(), sin.float()
    am32 = attention_mask if attention_mask is None else attention_mask.int()

    # warmup ref
    for _ in range(3):
        _ = _reference_sdpa_with_rope(Pq32, Pk32, Pv32, Vq32, Vk32, Vv32, bq32, bk32, bv32, cos32, sin32, am32)

    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    tms = []
    for _ in range(max(1, args.iters // 5)):
        t0.record()
        O_ref = _reference_sdpa_with_rope(Pq32, Pk32, Pv32, Vq32, Vk32, Vv32, bq32, bk32, bv32, cos32, sin32, am32)
        t1.record()
        torch.cuda.synchronize()
        tms.append(t0.elapsed_time(t1))
    ms_ref = float(sum(tms) / len(tms))

    # ---- Compare ----
    O_k32 = O_kernel.float()
    diff = O_k32 - O_ref
    num = torch.linalg.norm(diff.reshape(B, -1), ord='fro')
    den = torch.linalg.norm(O_ref.reshape(B, -1), ord='fro')
    rel_fro = (num / (den + 1e-12)).item()
    max_abs = diff.abs().max().item()
    finite_kernel = torch.isfinite(O_kernel).all().item()
    finite_ref = torch.isfinite(O_ref).all().item()
    finite_diff = torch.isfinite(diff).all().item()

    def bytes_to_mb(x): return x / (1024**2)

    print("===== FlashSVD+RoPE Triton Kernel Test =====")
    print(f"Shapes: B={B}, H={H}, M={M}, dh={dh}, R={R}, dtype={dtype}")
    if attention_mask is not None:
        valid_len = int(attention_mask[0].sum().item())
        print(f"Pad mask enabled: valid_len={valid_len}/{M}")
    print(f"Finite(kernel): {finite_kernel}  Finite(ref): {finite_ref}  Finite(diff): {finite_diff}")
    print(f"Max abs error: {max_abs:.3e}")
    print(f"Rel Fro error: {rel_fro:.3e}")
    print(f"Latency (kernel): {ms_kernel:.3f} ms/iter over {args.iters} iters")
    print(f"Latency (reference): {ms_ref:.3f} ms/iter over {max(1, args.iters//5)} iters")
    print(f"Peak CUDA allocated: {_pretty_mem(peak_alloc)}   reserved: {_pretty_mem(peak_reserved)}")

    # --- peak comparison (kernel vs ref) ---
    def measure_peak_bytes(fn, *fargs, **fkwargs):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Clear cached memory
        torch.cuda.reset_peak_memory_stats()
        out = fn(*fargs, **fkwargs)
        torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    _, kern_alloc, kern_res = measure_peak_bytes(
        flash, qkv, attention_mask=attention_mask, position_ids=position_ids
    )

    def run_ref():
        return _reference_sdpa_with_rope(
            Pq.float(), Pk.float(), Pv.float(),
            Vq.float(), Vk.float(), Vv.float(),
            bq.float(), bk.float(), bv.float(),
            cos.float(), sin.float(),
            attention_mask if attention_mask is None else attention_mask.int()
        )

    torch.cuda.empty_cache()
    _, ref_alloc, ref_res = measure_peak_bytes(run_ref)

    print(f"\n--- Peak memory comparison ---")
    print(f"Kernel  peak allocated: {bytes_to_mb(kern_alloc):.1f} MB   reserved: {bytes_to_mb(kern_res):.1f} MB")
    print(f"Ref SDPA peak allocated: {bytes_to_mb(ref_alloc):.1f} MB   reserved: {bytes_to_mb(ref_res):.1f} MB")
    print(f"Savings (alloc): {bytes_to_mb(ref_alloc - kern_alloc):.1f} MB")
    print(f"Savings (reserv): {bytes_to_mb(ref_res  - kern_res ):.1f} MB")



