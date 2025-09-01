#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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
# CHANGED: add H axis to P’s; reshape V’s/bias per-head.
@dataclass
class QKVFactors:
    # Rank-space inputs [B, H, M, R]
    Pq: torch.Tensor
    Pk: torch.Tensor
    Pv: torch.Tensor
    # Per-head lifting factors [H, R, dh]
    Vq: torch.Tensor
    Vk: torch.Tensor
    Vv: torch.Tensor
    # Optional per-head biases [H, dh]
    bq: Optional[torch.Tensor] = None
    bk: Optional[torch.Tensor] = None
    bv: Optional[torch.Tensor] = None


# ----------------------------
# Triton kernel: FlashSVD + RoPE + online softmax
#   Computes: O[b,h,m0:m0+BM, dh] = softmax(QK^T + mask) @ V
#   Q,K,V are *not* stored: we form them as (P@V) tiles inside the kernel.
#
#   Grid: (B*H, ceil_div(M, BM))  -- one head per program on M-tiles
#   Loops:
#     - over dh in BDH (project tiles)
#     - over N (sequence) in BN for streaming softmax
#     - online softmax keep (m_i, l_i, acc_i)
# ----------------------------
@triton.jit
def flashsvd_rope_sdpa(
    # P rank-space
    Pq_ptr, Pk_ptr, Pv_ptr,
    # V factors [H,R,dh]
    Vq_ptr, Vk_ptr, Vv_ptr,
    # Biases [H,dh] or nullptr
    bq_ptr, bk_ptr, bv_ptr,
    # RoPE tables [B,H,M,dh]
    COS_ptr, SIN_ptr,
    O_ptr,
    pad_mask_ptr, add_mask_ptr,
    # Shapes
    B, H, M, R, dh,
    # CHANGED: include head strides for P
    sPq_b, sPq_h, sPq_m, sPq_r,
    sPk_b, sPk_h, sPk_m, sPk_r,
    sPv_b, sPv_h, sPv_m, sPv_r,
    # CHANGED: V strides [H,R,dh] → (h,r,dh)
    sVq_h, sVq_r, sVq_dh,
    sVk_h, sVk_r, sVk_dh,
    sVv_h, sVv_r, sVv_dh,
    # CHANGED: bias strides [H,dh] → (h,dh)
    sbq_h, sbq_dh, sbk_h, sbk_dh, sbv_h, sbv_dh,
    # unchanged cos/sin/O/pad/add strides...
    sCOS_b, sCOS_h, sCOS_m, sCOS_dh,
    sSIN_b, sSIN_h, sSIN_m, sSIN_dh,
    sO_b, sO_h, sO_m, sO_dh,
    sPM_b, sPM_m,
    sAM_b, sAM_mq, sAM_mk,
    BM: tl.constexpr, BN: tl.constexpr, BDH: tl.constexpr, BR: tl.constexpr,
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

    # --- online softmax state per (BM, dh) row ---
    m_i = tl.full((BM,), -float("inf"), dtype=tl.float32)   # running max over K
    l_i = tl.zeros((BM,), dtype=tl.float32)                 # running lSE
    acc = tl.zeros((BM, BDH), dtype=tl.float32)             # running output accumulator

    # Pre-load pad mask slice for queries (broadcast over dh)
    if HAS_PAD:
        pm_q = tl.load(pad_mask_ptr + bid * sPM_b + offs_m * sPM_m,
                       mask=offs_m < M, other=0).to(tl.int1)
    else:
        pm_q = tl.full((BM,), 1, dtype=tl.int1)

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
        for d0 in range(0, dh, BDH):
            # compile-time checks
            tl.static_assert(BDH % 2 == 0, "RoPE half–half requires even BDH")

            # constexpr ranges
            offs0 = tl.arange(0, BDH // 2)        # first half
            offs1 = offs0 + (BDH // 2)            # second half

            # Load cos/sin for first half only
            cos_q0 = tl.load(COS_ptr + bid*sCOS_b + hid*sCOS_h + offs_m[:,None]*sCOS_m + offs0[None,:]*sCOS_dh,
                            mask=(offs_m[:,None] < M), other=0.0)
            sin_q0 = tl.load(SIN_ptr + bid*sSIN_b + hid*sSIN_h + offs_m[:,None]*sSIN_m + offs0[None,:]*sSIN_dh,
                            mask=(offs_m[:,None] < M), other=0.0)
            cos_k0 = tl.load(COS_ptr + bid*sCOS_b + hid*sCOS_h + offs_n[:,None]*sCOS_m + offs0[None,:]*sCOS_dh,
                            mask=(offs_n[:,None] < M), other=0.0)
            sin_k0 = tl.load(SIN_ptr + bid*sSIN_b + hid*sSIN_h + offs_n[:,None]*sSIN_m + offs0[None,:]*sSIN_dh,
                            mask=(offs_n[:,None] < M), other=0.0)

            # Use constexpr sizes in shapes
            q0 = tl.zeros((BM, BDH // 2), dtype=tl.float32)
            q1 = tl.zeros((BM, BDH // 2), dtype=tl.float32)
            k0 = tl.zeros((BN, BDH // 2), dtype=tl.float32)
            k1 = tl.zeros((BN, BDH // 2), dtype=tl.float32)

            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R

                Pq_blk = tl.load(Pq_ptr + bid*sPq_b + hid*sPq_h + offs_m[:,None]*sPq_m + r[None,:]*sPq_r,
                                mask=(offs_m[:,None] < M) & mask_r[None,:], other=0.0)
                Pk_blk = tl.load(Pk_ptr + bid*sPk_b + hid*sPk_h + offs_n[:,None]*sPk_m + r[None,:]*sPk_r,
                                mask=(offs_n[:,None] < M) & mask_r[None,:], other=0.0)

                Vq0 = tl.load(Vq_ptr + hid*sVq_h + r[:,None]*sVq_r + offs0[None,:]*sVq_dh,
                            mask=mask_r[:,None], other=0.0)
                Vq1 = tl.load(Vq_ptr + hid*sVq_h + r[:,None]*sVq_r + offs1[None,:]*sVq_dh,
                            mask=mask_r[:,None], other=0.0)
                q0 += tl.dot(Pq_blk, Vq0.to(Pq_blk.dtype)).to(tl.float32)
                q1 += tl.dot(Pq_blk, Vq1.to(Pq_blk.dtype)).to(tl.float32)

                Vk0 = tl.load(Vk_ptr + hid*sVk_h + r[:,None]*sVk_r + offs0[None,:]*sVk_dh,
                            mask=mask_r[:,None], other=0.0)
                Vk1 = tl.load(Vk_ptr + hid*sVk_h + r[:,None]*sVk_r + offs1[None,:]*sVk_dh,
                            mask=mask_r[:,None], other=0.0)
                k0 += tl.dot(Pk_blk, Vk0.to(Pk_blk.dtype)).to(tl.float32)
                k1 += tl.dot(Pk_blk, Vk1.to(Pk_blk.dtype)).to(tl.float32)

            if sbq_dh != 0:
                bq0 = tl.load(bq_ptr + hid*sbq_h + offs0 * sbq_dh)
                bq1 = tl.load(bq_ptr + hid*sbq_h + offs1 * sbq_dh)
                q0 += bq0[None, :]; q1 += bq1[None, :]
            if sbk_dh != 0:
                bk0 = tl.load(bk_ptr + hid*sbk_h + offs0 * sbk_dh)
                bk1 = tl.load(bk_ptr + hid*sbk_h + offs1 * sbk_dh)
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

        # apply masks for this (BM x BN) block
        if HAS_PAD:
            pm_k = tl.load(pad_mask_ptr + bid * sPM_b + offs_n * sPM_m, mask=valid_n, other=0).to(tl.int1)
            mask_pad = (pm_q[:, None] & pm_k[None, :])
            scores = tl.where(mask_pad, scores, -float("inf"))

        if HAS_ADD:
            add = tl.load(add_mask_ptr + bid*sAM_b + offs_m[:, None]*sAM_mq + offs_n[None, :]*sAM_mk,
                          mask=(offs_m[:, None] < M) & valid_n[None, :], other=0.0)
            scores += add

        # ---- online softmax merge with running (m_i,l_i,acc) ----
        m_curr = tl.max(scores, 1)
        m_new = tl.maximum(m_i, m_curr)

        l_i *= tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        # Need V(d) for acc update; compute v_blk tiled over R
        for d0 in range(0, dh, BDH):
            d = d0 + offs_d
            mask_d = d < dh

            v_blk = tl.zeros((BN, BDH), dtype=tl.float32)
            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R

                v_blk_src = tl.load(Pv_ptr + bid*sPv_b + hid*sPv_h + offs_n[:, None]*sPv_m + r[None, :]*sPv_r,
                                    mask=(offs_n[:, None] < M) & mask_r[None, :], other=0.0)
                Vv_sub = tl.load(Vv_ptr + hid*sVv_h + r[:, None]*sVv_r + d[None, :]*sVv_dh,
                                 mask=mask_r[:, None] & mask_d[None, :], other=0.0)
                v_blk += tl.dot(v_blk_src, Vv_sub.to(v_blk_src.dtype)).to(tl.float32)

            if sbv_dh != 0:
                bv_sub = tl.load(bv_ptr + hid*sbv_h + d * sbv_dh, mask=mask_d, other=0.0)
                v_blk += bv_sub[None, :]

            if d0 == 0:
                acc *= tl.exp(m_i[:, None] - m_new[:, None])
            acc += tl.dot(p, v_blk)

        l_i += tl.sum(p, 1)
        m_i = m_new

    # finalize output: O = acc / l_i[:,None]
    O_tile = acc / l_i[:, None]
    tl.store(O_ptr + bid*sO_b + hid*sO_h + offs_m[:, None]*sO_m + tl.arange(0, BDH)[None, :]*sO_dh,
             O_tile, mask=offs_m[:, None] < M)


# ----------------------------
# Public module
# ----------------------------
class FlashSVDRoPEAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, rotary_emb, *,
                 bm=64, bn=64, bdh=None, br=64):   # <-- add br
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.bm = bm
        self.bn = bn
        self.bdh = head_dim if bdh is None else bdh
        self.br = br                                  # <-- save br
        assert self.bdh == head_dim, "Kernel currently expects BDH == dh (one dh stripe)."


    @staticmethod
    def _padding_mask_bool(attention_mask_2d: torch.Tensor) -> torch.Tensor:
        # [B,L] with 1=valid,0=pad -> [B,1,1,L] boolean; True = MASK (we'll convert to additive)
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

        H, dh = self.num_heads, self.head_dim

        # Normalize P shapes to [B,H,M,R]
        if Pq.dim() == 3:
            B, M, R = Pq.shape
            Pq = Pq.unsqueeze(1).expand(B, H, M, R)
            Pk = Pk.unsqueeze(1).expand(B, H, M, R)
            Pv = Pv.unsqueeze(1).expand(B, H, M, R)
        elif Pq.dim() == 4:
            B, H_in, M, R = Pq.shape
            assert H_in == H, f"P factors H={H_in} mismatch num_heads={H}"
        else:
            raise ValueError(f"Unsupported P shape: {Pq.shape}")

        device = Pq.device
        dtype = Pq.dtype

        # Normalize V to [H,R,dh]
        if Vq.dim() == 2:
            # [R, H*dh] -> [H,R,dh]
            Rv, Dv = Vq.shape
            assert Rv == R and Dv == H * dh, f"Vq shape {Vq.shape} incompatible with R={R},H={H},dh={dh}"
            Vq = Vq.view(R, H, dh).permute(1, 0, 2).contiguous()
            Vk = Vk.view(R, H, dh).permute(1, 0, 2).contiguous()
            Vv = Vv.view(R, H, dh).permute(1, 0, 2).contiguous()
        elif Vq.dim() == 3:
            assert Vq.shape == (H, R, dh), f"Expected Vq [H,R,dh], got {Vq.shape}"
            assert Vk.shape == (H, R, dh), f"Expected Vk [H,R,dh], got {Vk.shape}"
            assert Vv.shape == (H, R, dh), f"Expected Vv [H,R,dh], got {Vv.shape}"
        else:
            raise ValueError(f"Unsupported V shape: {Vq.shape}")

        # Normalize bias to [H,dh]
        if bq is not None:
            if bq.dim() == 1:
                assert bq.numel() == H * dh
                bq = bq.view(H, dh).contiguous()
            else:
                assert bq.shape == (H, dh)
        if bk is not None:
            if bk.dim() == 1:
                assert bk.numel() == H * dh
                bk = bk.view(H, dh).contiguous()
            else:
                assert bk.shape == (H, dh)
        if bv is not None:
            if bv.dim() == 1:
                assert bv.numel() == H * dh
                bv = bv.view(H, dh).contiguous()
            else:
                assert bv.shape == (H, dh)

        # RoPE cos/sin for (B,H,M,dh)
        # Create dummy q (B*H, M, dh) just to get shapes for rotary_emb; we only need cos/sin
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
                # 2D padding mask [B,M] with 1 valid, 0 pad
                pad_mask = attention_mask.contiguous()
                pad_mask_ptr = pad_mask
                has_pad = 1
            elif attention_mask.dim() == 4:
                add_mask = attention_mask.contiguous()
                add_mask_ptr = add_mask
                has_add = 1
            else:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        if sliding_window_mask is not None:
            # treat as additive
            add_mask_ptr = sliding_window_mask.contiguous()
            has_add = 1

        # Output buffer [B,H,M,dh]
        O = torch.empty((B, H, M, dh), device=device, dtype=dtype)

        # Strides
        sPq_b, sPq_h, sPq_m, sPq_r = Pq.stride()
        sPk_b, sPk_h, sPk_m, sPk_r = Pk.stride()
        sPv_b, sPv_h, sPv_m, sPv_r = Pv.stride()
        sVq_h, sVq_r, sVq_dh = Vq.stride()
        sVk_h, sVk_r, sVk_dh = Vk.stride()
        sVv_h, sVv_r, sVv_dh = Vv.stride()
        if bq is not None:
            sbq_h, sbq_dh = bq.stride()
        else:
            sbq_h = sbq_dh = 0
        if bk is not None:
            sbk_h, sbk_dh = bk.stride()
        else:
            sbk_h = sbk_dh = 0
        if bv is not None:
            sbv_h, sbv_dh = bv.stride()
        else:
            sbv_h = sbv_dh = 0
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
        grid = (B * H, triton.cdiv(M, self.bm))
        flashsvd_rope_sdpa[grid](
            Pq, Pk, Pv,
            Vq, Vk, Vv,
            bq if bq is not None else O,  # harmless ptr if no bias
            bk if bk is not None else O,
            bv if bv is not None else O,
            cos, sin,
            O,
            pad_mask_ptr if has_pad else O,
            add_mask_ptr if has_add else O,
            B, H, M, R, dh,
            sPq_b, sPq_h, sPq_m, sPq_r,
            sPk_b, sPk_h, sPk_m, sPk_r,
            sPv_b, sPv_h, sPv_m, sPv_r,
            sVq_h, sVq_r, sVq_dh,
            sVk_h, sVk_r, sVk_dh,
            sVv_h, sVv_r, sVv_dh,
            sbq_h, sbq_dh, sbk_h, sbk_dh, sbv_h, sbv_dh,
            sCOS_b, sCOS_h, sCOS_m, sCOS_dh,
            sSIN_b, sSIN_h, sSIN_m, sSIN_dh,
            sO_b, sO_h, sO_m, sO_dh,
            sPM_b, sPM_m,
            sAM_b, sO_m, sO_m,  # reuse strides for [Mq,Mk]
            #sAM_b, sAM_mq, sAM_mk,
            BM=self.bm, BN=self.bn, BDH=self.bdh, BR=self.br,   # <-- here
            HAS_PAD=has_pad, HAS_ADD=has_add,
            USE_TANH=1,
            num_warps=4, num_stages=2,
        )
        return O  # [B,H,M,dh]


# ----------------------------
# Example integration: replace your attn block
# ----------------------------
class ExplicitSVDWithRoPEKernelBlock(nn.Module):
    """
    Drop-in replacement for your attention part that:
      - uses rank-space Pq,Pk,Pv + factors Vq,Vk,Vv
      - applies RoPE in-kernel
      - does streaming softmax attention (no Q/K/V tensors)
    MLP remains as you implemented earlier.
    """
    def __init__(self, hf_layer, cfg, *, rank_attn: Optional[int] = None, bm=128, bn=128):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_attention_heads
        self.hidden_size = hf_layer.attn.Wo.in_features
        self.head_dim = self.hidden_size // self.num_heads
        self.attn_norm = nn.LayerNorm(self.hidden_size, eps=hf_layer.attn_norm.eps)
        self.rotary_emb = hf_layer.attn.rotary_emb
        self.Wo_attn = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wo_attn.load_state_dict(hf_layer.attn.Wo.state_dict())

        # Split original fused Wqkv -> Wq, Wk, Wv and SVD-factorize or just keep dense factors
        with torch.no_grad():
            Wqkv = hf_layer.attn.Wqkv
            Wq, Wk, Wv = torch.chunk(Wqkv.weight, 3, dim=0)  # [dm,dm] each
            bq, bk, bv = torch.chunk(Wqkv.bias, 3, dim=0) if Wqkv.bias is not None else (None, None, None)

        # Make rank-space factors: for now full-rank to validate parity.
        # You can drop to low-rank later by SVD on W^T as you did.
        self.rank = self.hidden_size if rank_attn is None else int(rank_attn)
        R = self.rank
        dm = self.hidden_size

        # Factor as W^T = U @ V  → W = V^T @ U^T
        # Build P = X @ U   and V_factor = V^T (so that P @ V_factor ≈ X @ W)
        def factor(W):
            U, S, Vh = torch.linalg.svd(W.t(), full_matrices=False)  # [dm,dm]
            r = min(R, S.shape[0])
            U_r = (U[:, :r] * S[:r])          # [dm,r]
            V_r = Vh[:r, :]                   # [r,dm]
            U_factor = nn.Linear(dm, r, bias=False)
            V_factor = nn.Linear(r, dm, bias=False)
            U_factor.weight.copy_(U_r.t())    # [r,dm]
            V_factor.weight.copy_(V_r)        # [dm,r]
            return U_factor, V_factor

        self.Pq_proj, self.Vq_proj = factor(Wq)
        self.Pk_proj, self.Vk_proj = factor(Wk)
        self.Pv_proj, self.Vv_proj = factor(Wv)

        if bq is not None:
            self.bq = nn.Parameter(bq.clone())
            self.bk = nn.Parameter(bk.clone())
            self.bv = nn.Parameter(bv.clone())
        else:
            self.bq = self.bk = self.bv = None

        # Triton attention
        self.flash = FlashSVDRoPEAttention(self.num_heads, self.head_dim, self.rotary_emb, bm=bm, bn=bn)

        # MLP path (reuse your explicit MLP code from earlier to keep parity)
        self.mlp_norm = hf_layer.mlp_norm
        self.Wi = hf_layer.mlp.Wi
        self.Wo_ffn = hf_layer.mlp.Wo
        self.act = getattr(hf_layer.mlp, "act", nn.GELU())
        self.ffn_is_geglu = (self.Wi.out_features == 2 * self.Wo_ffn.in_features)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        B, M, D = hidden_states.shape
        H, dh = self.num_heads, self.head_dim

        # pre-norm
        x = hidden_states
        xn = self.attn_norm(x)

        # --- in ExplicitSVDWithRoPEKernelBlock.forward() ---
        # rank-space [B, M, R] from your U_factor projectors
        Pq = self.Pq_proj(xn)      # [B,M,R]
        Pk = self.Pk_proj(xn)
        Pv = self.Pv_proj(xn)

        B, M, R = Pq.shape
        H, dh = self.num_heads, self.head_dim

        # No need to expand P again; already normalized earlier
        Pq_bhmr, Pk_bhmr, Pv_bhmr = Pq, Pk, Pv

        # CHANGED: turn [R, H*dh] → [H, R, dh] (contiguous for clean strides) if needed
        Vq_w = self.Vq_proj.weight.t().contiguous()
        Vk_w = self.Vk_proj.weight.t().contiguous()
        Vv_w = self.Vv_proj.weight.t().contiguous()
        Vq_hrd = Vq_w.reshape(R, H, dh).permute(1, 0, 2).contiguous()
        Vk_hrd = Vk_w.reshape(R, H, dh).permute(1, 0, 2).contiguous()
        Vv_hrd = Vv_w.reshape(R, H, dh).permute(1, 0, 2).contiguous()

        # CHANGED: [H*dh] → [H, dh]
        bq_hd = None if self.bq is None else self.bq.view(H, dh).contiguous()
        bk_hd = None if self.bk is None else self.bk.view(H, dh).contiguous()
        bv_hd = None if self.bv is None else self.bv.view(H, dh).contiguous()

        # position ids
        if position_ids is None:
            position_ids = torch.arange(M, device=hidden_states.device)[None, :].expand(B, -1)

        # in-kernel attention with RoPE
        O = self.flash(
            # Build struct for the kernel
            QKVFactors(
                Pq=Pq_bhmr, Pk=Pk_bhmr, Pv=Pv_bhmr,
                Vq=Vq_hrd,  Vk=Vk_hrd,  Vv=Vv_hrd,
                bq=bq_hd,   bk=bk_hd,   bv=bv_hd
            ),
            attention_mask=attention_mask,
            position_ids=position_ids,
            sliding_window_mask=sliding_window_mask,
        )  # [B,H,M,dh]

        attn = O.transpose(1, 2).contiguous().view(B, M, D)  # [B,M,D]
        x = x + self.Wo_attn(attn)

        # MLP (explicit)
        xn2 = self.mlp_norm(x)
        z = self.Wi(xn2)
        if self.ffn_is_geglu:
            u, v = z.chunk(2, dim=-1)
            h = self.act(u) * v
        else:
            h = self.act(z)
        x = x + self.Wo_ffn(h)
        return (x,)


# ----------------------------
# Minimal test harness
# ----------------------------
def _rotary_emb_make(seq_len, dim, base=10000.0, device="cuda", dtype=torch.float32):
    """Return (cos, sin) of shape [seq_len, dim] for standard RoPE (pairwise dims)."""
    assert dim % 2 == 0, "RoPE dim must be even."
    half = dim // 2
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    freqs = torch.einsum("m,d->md", pos, inv_freq)  # [M, half]
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # interleave to [M, dim] as [cos, cos, ...] / [sin, sin, ...] per pair
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, dim)
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, dim)
    return cos, sin

class _SimpleRotary:
    """Mimics HF rotary_emb interface used above: returns (cos, sin) for input q."""
    def __init__(self, base=10000.0):
        self.base = base
    def __call__(self, q_like: torch.Tensor, *, position_ids: torch.Tensor):
        # q_like: [(B*H), M, dh]; position_ids: [(B*H), M] or [M]
        BH, M, dh = q_like.shape
        device = q_like.device
        dtype  = q_like.dtype
        # build full table then index by position_ids
        cos_tab, sin_tab = _rotary_emb_make(M, dh, base=self.base, device=device, dtype=dtype)
        # broadcast to [(B*H), M, dh]
        cos = cos_tab.index_select(0, position_ids[0].to(torch.long)).unsqueeze(0).expand(BH, -1, -1).contiguous()
        sin = sin_tab.index_select(0, position_ids[0].to(torch.long)).unsqueeze(0).expand(BH, -1, -1).contiguous()
        return cos, sin

def _apply_rope_torch(x, cos, sin):
    # x: [B,H,M,dh], cos/sin: [B,H,M,dh]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos[..., :x1.shape[-1]] - x2 * sin[..., :x1.shape[-1]],
                      x1 * sin[..., :x1.shape[-1]] + x2 * cos[..., :x1.shape[-1]]], dim=-1)

@torch.no_grad()
def _reference_sdpa_with_rope(Pq, Pk, Pv, Vq, Vk, Vv, bq, bk, bv, cos, sin,
                              attention_mask_2d=None):
    """
    Pure-PyTorch reference that explicitly materializes Q,K,V, applies RoPE, then SDPA.
    Returns O_ref with shape [B,H,M,dh].
    """
    B, M, R = Pq.shape
    Hdh = Vq.shape[1]
    H = cos.shape[1]
    dh = Hdh // H

    # build Q,K,V per head
    # P@V -> [B,M,H*dh] then reshape to [B,H,M,dh]
    Q = torch.matmul(Pq, Vq)  # [B,M,Hdh]
    K = torch.matmul(Pk, Vk)
    V = torch.matmul(Pv, Vv)
    if bq is not None: Q = Q + bq
    if bk is not None: K = K + bk
    if bv is not None: V = V + bv
    Q = Q.view(B, M, H, dh).permute(0, 2, 1, 3).contiguous()  # [B,H,M,dh]
    K = K.view(B, M, H, dh).permute(0, 2, 1, 3).contiguous()
    V = V.view(B, M, H, dh).permute(0, 2, 1, 3).contiguous()

    # apply RoPE
    Qr = _apply_rope_torch(Q, cos, sin)
    Kr = _apply_rope_torch(K, cos, sin)

    # Use PyTorch fused SDPA (FlashAttention when available)
    attn_mask = None
    if attention_mask_2d is not None:
        am = attention_mask_2d.to(torch.bool)           # [B,M]
        qv = am[:, None, :, None]                       # [B,1,M,1]
        kv = am[:, None, None, :]                       # [B,1,1,M]
        mask_bool = ~(qv & kv)                          # True = masked
        attn_mask = mask_bool.expand(B, H, M, M).contiguous()

    # Prefer FlashAttention backend when available
    try:
        ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
    except Exception:
        ctx = torch.backends.cuda.sdp_kernel()
    with ctx:
        # Oref = F.scaled_dot_product_attention(
        #     Qr, Kr, V,
        #     attn_mask=attn_mask,
        #     dropout_p=0.0,
        #     is_causal=False,
        # )  # [B,H,M,dh]
        # OLD (disables math)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            Oref = F.scaled_dot_product_attention(Qr, Kr, V, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    return Oref



def _pretty_mem(bytes_val: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.0f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.0f} PB"

if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser("FlashSVD+RoPE kernel test")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--M", type=int, default=256*4)
    parser.add_argument("--dh", type=int, default=128)
    parser.add_argument("--R", type=int, default=32, help="rank-space dimension")
    parser.add_argument("--bm", type=int, default=64)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mask", action="store_true", help="enable 2D padding mask with random pads")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this test.")
        raise SystemExit(1)

    torch.manual_seed(args.seed)
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
        # make last ~10% positions padded for each batch
        valid_len = int(0.9 * M)
        attention_mask = torch.zeros(B, M, device=device, dtype=torch.int32)
        attention_mask[:, :valid_len] = 1

    # Run kernel
    torch.cuda.synchronize()
    flash = FlashSVDRoPEAttention(num_heads=H, head_dim=dh, rotary_emb=rotary, bm=args.bm, bn=args.bn, bdh=dh).to(device)

    # Build QKVFactors struct
    qkv = QKVFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

    # Warmup
    for _ in range(max(1, args.warmup)):
        _ = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)

    torch.cuda.synchronize()

    # Measure latency (kernel)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        O_kernel = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_kernel = (t1 - t0) * 1000.0 / max(1, args.iters)

    # Peak memory for a single forward
    torch.cuda.reset_peak_memory_stats()
    _ = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    # Reference (compute in fp32 for accuracy, then cast)
    Pq32, Pk32, Pv32 = Pq.float(), Pk.float(), Pv.float()
    Vq32, Vk32, Vv32 = Vq.float(), Vk.float(), Vv.float()
    bq32, bk32, bv32 = bq.float(), bk.float(), bv.float()
    cos32, sin32 = cos.float(), sin.float()
    am32 = attention_mask if attention_mask is None else attention_mask.int()

    # Warmup ref
    for _ in range(3):
        _ = _reference_sdpa_with_rope(Pq32, Pk32, Pv32, Vq32, Vk32, Vv32, bq32, bk32, bv32, cos32, sin32, am32)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max(1, args.iters // 5)):  # reference is slower; fewer iters is fine
        O_ref = _reference_sdpa_with_rope(Pq32, Pk32, Pv32, Vq32, Vk32, Vv32, bq32, bk32, bv32, cos32, sin32, am32)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_ref = (t1 - t0) * 1000.0 / max(1, args.iters // 5)

    # Compare
    O_k32 = O_kernel.float()
    diff = O_k32 - O_ref
    num = torch.linalg.norm(diff.reshape(B, -1), ord='fro')
    den = torch.linalg.norm(O_ref.reshape(B, -1), ord='fro')
    rel_fro = (num / (den + 1e-12)).item()
    max_abs = diff.abs().max().item()
    finite_kernel = torch.isfinite(O_kernel).all().item()
    finite_ref = torch.isfinite(O_ref).all().item()
    finite_diff = torch.isfinite(diff).all().item()

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

    
    def measure_peak_bytes(fn, *args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    def bytes_to_mb(x): return x / (1024**2)

    # --- measure kernel peak ---
    _, kern_alloc, kern_res = measure_peak_bytes(
        flash, qkv, attention_mask=attention_mask, position_ids=position_ids
    )

    # --- measure reference peak (do once; it builds [B,H,M,M]) ---
    def run_ref():
        return _reference_sdpa_with_rope(
            Pq.float(), Pk.float(), Pv.float(),
            Vq.float(), Vk.float(), Vv.float(),
            bq.float(), bk.float(), bv.float(),
            cos.float(), sin.float(),
            attention_mask if attention_mask is None else attention_mask.int()
        )

    # free kernel output so it doesn't hold memory while we profile ref
    torch.cuda.empty_cache()
    _, ref_alloc, ref_res = measure_peak_bytes(run_ref)

    print(f"\n--- Peak memory comparison ---")
    print(f"Kernel  peak allocated: {bytes_to_mb(kern_alloc):.1f} MB   reserved: {bytes_to_mb(kern_res):.1f} MB")
    print(f"Ref SDPA peak allocated: {bytes_to_mb(ref_alloc):.1f} MB   reserved: {bytes_to_mb(ref_res):.1f} MB")
    print(f"Savings (alloc): {bytes_to_mb(ref_alloc - kern_alloc):.1f} MB")
    print(f"Savings (reserv): {bytes_to_mb(ref_res  - kern_res ):.1f} MB")
