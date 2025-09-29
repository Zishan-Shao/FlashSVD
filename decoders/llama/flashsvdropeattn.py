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
# Utilities
# ----------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_bmhd(x_bmhd: torch.Tensor, cos_bmhd: torch.Tensor, sin_bmhd: torch.Tensor) -> torch.Tensor:
    # All BMHd; rotate last dim in half-pairs
    x1, x2 = x_bmhd.chunk(2, dim=-1)
    return torch.cat([x1 * cos_bmhd[..., :x1.shape[-1]] - x2 * sin_bmhd[..., :x1.shape[-1]],
                      x1 * sin_bmhd[..., :x1.shape[-1]] + x2 * cos_bmhd[..., :x1.shape[-1]]], dim=-1)


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
    # Optional biases on head-space flattened [H*dh]
    bq: Optional[torch.Tensor] = None
    bk: Optional[torch.Tensor] = None
    bv: Optional[torch.Tensor] = None


# ----------------------------
# Triton kernel: FlashSVD + RoPE + online softmax
#   Computes: O[b, m0:m0+BM, h, dh] in BMHd layout (via strides)
#   Q,K,V are formed as tiles (P@V) in-kernel; nothing materialized.
# ----------------------------
@triton.jit
def flashsvd_rope_sdpa(
    # Pq,Pk,Pv rank-space activations [B, M, R]
    Pq_ptr, Pk_ptr, Pv_ptr,
    # Vq,Vk,Vv factors [R, H*dh]
    Vq_ptr, Vk_ptr, Vv_ptr,
    # Optional biases over H*dh (0-stride if absent)
    bq_ptr, bk_ptr, bv_ptr,
    # RoPE cos/sin [B, M, H, dh] (BMHd)
    COS_ptr, SIN_ptr,
    # Output pointer O [B, M, H, dh] (BMHd)
    O_ptr,
    # Optional masks
    pad_mask_ptr,   # [B, M] (1 valid, 0 pad) or nullptr
    add_mask_ptr,   # [B, 1, M, M] additive or nullptr
    # Shapes / strides
    B, H, M, R, dh,
    sPq_b, sPq_m, sPq_r,
    sPk_b, sPk_m, sPk_r,
    sPv_b, sPv_m, sPv_r,
    sVq_r, sVq_hd,
    sVk_r, sVk_hd,
    sVv_r, sVv_hd,
    sbq_hd, sbk_hd, sbv_hd,
    sCOS_b, sCOS_m, sCOS_h, sCOS_dh,   # NOTE: BMHd order for cos/sin
    sSIN_b, sSIN_m, sSIN_h, sSIN_dh,
    sO_b, sO_m, sO_h, sO_dh,           # NOTE: BMHd order for output
    sPM_b, sPM_m,
    sAM_b, sAM_mq, sAM_mk,             # additive mask [B, 1, Mq, Mk]
    # Tiling
    BM: tl.constexpr, BN: tl.constexpr, BDH: tl.constexpr, BR: tl.constexpr,
    # flags
    HAS_PAD: tl.constexpr, HAS_ADD: tl.constexpr,
):
    # program ids: one (B, H) head per program across M-tiles
    bh   = tl.program_id(0)   # 0..B*H-1
    bid  = bh // H
    hid  = bh % H
    m_blk = tl.program_id(1)

    # tile offsets (queries)
    offs_m = m_blk * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BDH)
    head_offset = hid * dh

    # running softmax state per (BM, dh)
    m_i = tl.full((BM,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    acc = tl.zeros((BM, BDH), dtype=tl.float32)

    dh_f = tl.full((1,), dh, dtype=tl.float32)
    scale = 1.0 / tl.sqrt(dh_f)

    # stream over keys BN
    for nk in range(0, M, BN):
        offs_n = nk + tl.arange(0, BN)
        valid_n = offs_n < M

        scores = tl.zeros((BM, BN), dtype=tl.float32)

        # accumulate scores over dh in stripes
        for d0 in range(0, dh, BDH):
            tl.static_assert(BDH % 2 == 0, "BDH must be even for half-rotations.")

            offs0 = tl.arange(0, BDH // 2)
            offs1 = offs0 + (BDH // 2)

            # Load RoPE angles for queries/keys from BMHd layout
            cos_q0 = tl.load(COS_ptr + bid*sCOS_b + offs_m[:,None]*sCOS_m + hid*sCOS_h + offs0[None,:]*sCOS_dh,
                             mask=(offs_m[:,None] < M), other=0.0)
            sin_q0 = tl.load(SIN_ptr + bid*sSIN_b + offs_m[:,None]*sSIN_m + hid*sSIN_h + offs0[None,:]*sSIN_dh,
                             mask=(offs_m[:,None] < M), other=0.0)
            cos_k0 = tl.load(COS_ptr + bid*sCOS_b + offs_n[:,None]*sCOS_m + hid*sCOS_h + offs0[None,:]*sCOS_dh,
                             mask=(offs_n[:,None] < M), other=0.0)
            sin_k0 = tl.load(SIN_ptr + bid*sSIN_b + offs_n[:,None]*sSIN_m + hid*sSIN_h + offs0[None,:]*sSIN_dh,
                             mask=(offs_n[:,None] < M), other=0.0)

            # q/k halves (fp32)
            q0 = tl.zeros((BM, BDH // 2), dtype=tl.float32)
            q1 = tl.zeros((BM, BDH // 2), dtype=tl.float32)
            k0 = tl.zeros((BN, BDH // 2), dtype=tl.float32)
            k1 = tl.zeros((BN, BDH // 2), dtype=tl.float32)

            # form Q/K tiles: P@V over rank stripes
            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R

                Pq_blk = tl.load(Pq_ptr + bid*sPq_b + offs_m[:,None]*sPq_m + r[None,:]*sPq_r,
                                 mask=(offs_m[:,None] < M) & mask_r[None,:], other=0.0)
                Pk_blk = tl.load(Pk_ptr + bid*sPk_b + offs_n[:,None]*sPk_m + r[None,:]*sPk_r,
                                 mask=(offs_n[:,None] < M) & mask_r[None,:], other=0.0)

                Vq0 = tl.load(Vq_ptr + r[:,None]*sVq_r + (head_offset + offs0)[None,:]*sVq_hd,
                              mask=mask_r[:,None], other=0.0)
                Vq1 = tl.load(Vq_ptr + r[:,None]*sVq_r + (head_offset + offs1)[None,:]*sVq_hd,
                              mask=mask_r[:,None], other=0.0)
                q0 += tl.dot(Pq_blk, Vq0.to(Pq_blk.dtype)).to(tl.float32)
                q1 += tl.dot(Pq_blk, Vq1.to(Pq_blk.dtype)).to(tl.float32)

                Vk0 = tl.load(Vk_ptr + r[:,None]*sVk_r + (head_offset + offs0)[None,:]*sVk_hd,
                              mask=mask_r[:,None], other=0.0)
                Vk1 = tl.load(Vk_ptr + r[:,None]*sVk_r + (head_offset + offs1)[None,:]*sVk_hd,
                              mask=mask_r[:,None], other=0.0)
                k0 += tl.dot(Pk_blk, Vk0.to(Pk_blk.dtype)).to(tl.float32)
                k1 += tl.dot(Pk_blk, Vk1.to(Pk_blk.dtype)).to(tl.float32)

            if sbq_hd != 0:
                bq0 = tl.load(bq_ptr + (head_offset + offs0) * sbq_hd)
                bq1 = tl.load(bq_ptr + (head_offset + offs1) * sbq_hd)
                q0 += bq0[None, :]; q1 += bq1[None, :]
            if sbk_hd != 0:
                bk0 = tl.load(bk_ptr + (head_offset + offs0) * sbk_hd)
                bk1 = tl.load(bk_ptr + (head_offset + offs1) * sbk_hd)
                k0 += bk0[None, :]; k1 += bk1[None, :]

            # RoPE rotate halves using first-half angles
            q0r = q0 * cos_q0 - q1 * sin_q0
            q1r = q0 * sin_q0 + q1 * cos_q0
            k0r = k0 * cos_k0 - k1 * sin_k0
            k1r = k0 * sin_k0 + k1 * cos_k0

            scores += tl.dot(q0r, tl.trans(k0r))
            scores += tl.dot(q1r, tl.trans(k1r))

        # scale
        scores *= scale

        # masks
        if HAS_PAD:
            pm_k = tl.load(pad_mask_ptr + bid*sPM_b + offs_n*sPM_m, mask=valid_n, other=0).to(tl.int1)
            scores = tl.where(pm_k[None, :], scores, -float("inf"))
        if HAS_ADD:
            add = tl.load(add_mask_ptr + bid*sAM_b + offs_m[:,None]*sAM_mq + offs_n[None,:]*sAM_mk,
                          mask=(offs_m[:,None] < M) & valid_n[None,:], other=0.0)
            scores += add

        # online softmax merge
        m_curr = tl.max(scores, 1)
        m_new  = tl.maximum(m_i, m_curr)
        l_i *= tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        # V(d) stripes & accumulate output
        for d0 in range(0, dh, BDH):
            d = d0 + offs_d
            mask_d = d < dh

            v_blk = tl.zeros((BN, BDH), dtype=tl.float32)
            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R
                Pv_blk = tl.load(Pv_ptr + bid*sPv_b + offs_n[:,None]*sPv_m + r[None,:]*sPv_r,
                                 mask=(offs_n[:,None] < M) & mask_r[None,:], other=0.0)
                Vv_sub = tl.load(Vv_ptr + r[:,None]*sVv_r + (head_offset + d)[None,:]*sVv_hd,
                                 mask=mask_r[:,None] & mask_d[None,:], other=0.0)
                v_blk += tl.dot(Pv_blk, Vv_sub.to(Pv_blk.dtype)).to(tl.float32)

            if sbv_hd != 0:
                bv_sub = tl.load(bv_ptr + (head_offset + d) * sbv_hd, mask=mask_d, other=0.0)
                v_blk += bv_sub[None, :]

            if d0 == 0:
                acc *= tl.exp(m_i[:, None] - m_new[:, None])
            acc += tl.dot(p, v_blk)

        l_i += tl.sum(p, 1)
        m_i = m_new

    # finalize to BMHd
    den = tl.where(l_i > 0, l_i, 1.0)
    O_tile = acc / den[:, None]
    O_tile = tl.where(l_i[:, None] > 0, O_tile, 0.0)
    tl.store(O_ptr + bid*sO_b + offs_m[:,None]*sO_m + hid*sO_h + tl.arange(0, BDH)[None,:]*sO_dh,
             O_tile, mask=offs_m[:,None] < M)


# ----------------------------
# Public module (BMHd)
# ----------------------------
class FlashSVDRoPEAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, rotary_emb, *,
                 bm=64, bn=64, bdh=None, br=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.bm = bm
        self.bn = bn
        self.bdh = head_dim if bdh is None else bdh
        self.br = br
        assert self.bdh == head_dim, "Kernel currently expects BDH == dh."

    @torch.no_grad()
    def forward(
        self,
        qkv_factors: QKVFactors,
        attention_mask: Optional[torch.Tensor],      # 2D padding or 4D additive
        position_ids: torch.Tensor,                  # [B, M]
        sliding_window_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Pq, Pk, Pv = qkv_factors.Pq, qkv_factors.Pk, qkv_factors.Pv
        Vq, Vk, Vv = qkv_factors.Vq, qkv_factors.Vk, qkv_factors.Vv
        bq, bk, bv = qkv_factors.bq, qkv_factors.bk, qkv_factors.bv

        B, M, R = Pq.shape
        H, dh = self.num_heads, self.head_dim
        device = Pq.device
        dtype = Pq.dtype

        # ---- RoPE cos/sin as BMHd (convert once) ----
        dummy = torch.empty((B * H, M, dh), device=device, dtype=dtype)
        posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
        cos, sin = self.rotary_emb(dummy, position_ids=posf)  # [(B*H), M, dh]
        cos = cos.view(B, H, M, dh).permute(0, 2, 1, 3).contiguous()  # [B, M, H, dh]
        sin = sin.view(B, H, M, dh).permute(0, 2, 1, 3).contiguous()  # [B, M, H, dh]

        # ---- masks ----
        pad_mask_ptr = None
        add_mask_ptr = None
        has_pad = 0
        has_add = 0

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = attention_mask.contiguous()   # [B, M]
                pad_mask_ptr = pad_mask
                has_pad = 1
            elif attention_mask.dim() == 4:
                add_mask = attention_mask.contiguous()   # [B, 1, M, M]
                add_mask_ptr = add_mask
                has_add = 1
            else:
                raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        if sliding_window_mask is not None:
            add_mask_ptr = sliding_window_mask.contiguous()
            has_add = 1

        # ---- Output buffer in BMHd ----
        O = torch.empty((B, M, H, dh), device=device, dtype=dtype)

        # Strides for rank-space and factors
        sPq_b, sPq_m, sPq_r = Pq.stride()
        sPk_b, sPk_m, sPk_r = Pk.stride()
        sPv_b, sPv_m, sPv_r = Pv.stride()
        sVq_r, sVq_hd = Vq.stride()
        sVk_r, sVk_hd = Vk.stride()
        sVv_r, sVv_hd = Vv.stride()
        sbq_hd = bq.stride(0) if bq is not None else 0
        sbk_hd = bk.stride(0) if bk is not None else 0
        sbv_hd = bv.stride(0) if bv is not None else 0

        # BMHd strides (note the order we’ll pass to kernel)
        sCOS_b, sCOS_m, sCOS_h, sCOS_dh = cos.stride()
        sSIN_b, sSIN_m, sSIN_h, sSIN_dh = sin.stride()
        sO_b,   sO_m,   sO_h,   sO_dh   = O.stride()

        if has_pad:
            sPM_b, sPM_m = pad_mask_ptr.stride()
        else:
            sPM_b = sPM_m = 0
        if has_add:
            sAM_b, sAM_1, sAM_mq, sAM_mk = add_mask_ptr.stride()
        else:
            sAM_b = sAM_mq = sAM_mk = 0

        # Launch: (B*H, ceil(M/BM))
        grid = (B * H, triton.cdiv(M, self.bm))
        flashsvd_rope_sdpa[grid](
            Pq, Pk, Pv,
            Vq, Vk, Vv,
            bq if bq is not None else O,  # harmless ptr if unused
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
            sCOS_b, sCOS_m, sCOS_h, sCOS_dh,
            sSIN_b, sSIN_m, sSIN_h, sSIN_dh,
            sO_b,   sO_m,   sO_h,   sO_dh,
            sPM_b, sPM_m,
            sAM_b, sAM_mq, sAM_mk,
            BM=self.bm, BN=self.bn, BDH=self.bdh, BR=self.br,
            HAS_PAD=has_pad, HAS_ADD=has_add,
            num_warps=4, num_stages=2,
        )
        return O  # [B, M, H, dh] BMHd


# ----------------------------
# Example integration: BMHd throughout
# ----------------------------
class ExplicitSVDWithRoPEKernelBlock(nn.Module):
    """
    Same as before but consumes BMHd directly from the kernel:
      - output from attention is BMHd -> reshape to [B, M, D] with no transpose/contiguous
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

        with torch.no_grad():
            Wqkv = hf_layer.attn.Wqkv
            Wq, Wk, Wv = torch.chunk(Wqkv.weight, 3, dim=0)
            bq, bk, bv = torch.chunk(Wqkv.bias, 3, dim=0) if Wqkv.bias is not None else (None, None, None)

        self.rank = self.hidden_size if rank_attn is None else int(rank_attn)
        R = self.rank
        dm = self.hidden_size

        def factor(W):
            U, S, Vh = torch.linalg.svd(W.t(), full_matrices=False)
            r = min(R, S.shape[0])
            U_r = (U[:, :r] * S[:r])          # [dm, r]
            V_r = Vh[:r, :]                   # [r, dm]
            U_factor = nn.Linear(dm, r, bias=False)
            V_factor = nn.Linear(r, dm, bias=False)
            U_factor.weight.copy_(U_r.t())
            V_factor.weight.copy_(V_r)
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

        self.flash = FlashSVDRoPEAttention(self.num_heads, self.head_dim, self.rotary_emb, bm=bm, bn=bn)

        # MLP as before
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

        x = hidden_states
        xn = self.attn_norm(x)

        Pq = self.Pq_proj(xn)  # [B, M, R]
        Pk = self.Pk_proj(xn)
        Pv = self.Pv_proj(xn)

        if position_ids is None:
            position_ids = torch.arange(M, device=hidden_states.device)[None, :].expand(B, -1)

        O_bmhd = self.flash(
            QKVFactors(
                Pq=Pq, Pk=Pk, Pv=Pv,
                Vq=self.Vq_proj.weight.t().contiguous(),
                Vk=self.Vk_proj.weight.t().contiguous(),
                Vv=self.Vv_proj.weight.t().contiguous(),
                bq=self.bq, bk=self.bk, bv=self.bv
            ),
            attention_mask=attention_mask,
            position_ids=position_ids,
            sliding_window_mask=sliding_window_mask,
        )  # [B, M, H, dh]

        attn_out = O_bmhd.view(B, M, D)  # no transpose/contig
        x = x + self.Wo_attn(attn_out)

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
# Minimal test harness (BMHd)
# ----------------------------
def _rotary_emb_make(seq_len, dim, base=10000.0, device="cuda", dtype=torch.float32):
    assert dim % 2 == 0
    half = dim // 2
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    freqs = torch.einsum("m,d->md", pos, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos = torch.stack([cos, cos], dim=-1).reshape(seq_len, dim)
    sin = torch.stack([sin, sin], dim=-1).reshape(seq_len, dim)
    return cos, sin

class _SimpleRotary:
    def __init__(self, base=10000.0): self.base = base
    def __call__(self, q_like: torch.Tensor, *, position_ids: torch.Tensor):
        BH, M, dh = q_like.shape
        device, dtype = q_like.device, q_like.dtype
        cos_tab, sin_tab = _rotary_emb_make(M, dh, base=self.base, device=device, dtype=dtype)
        cos = cos_tab.index_select(0, position_ids[0].to(torch.long)).unsqueeze(0).expand(BH, -1, -1).contiguous()
        sin = sin_tab.index_select(0, position_ids[0].to(torch.long)).unsqueeze(0).expand(BH, -1, -1).contiguous()
        return cos, sin

@torch.no_grad()
def _reference_sdpa_with_rope_bmhd(Pq, Pk, Pv, Vq, Vk, Vv, bq, bk, bv, cos_bmhd, sin_bmhd, attention_mask_2d=None):
    """
    Reference in BMHd layout throughout. Returns O_ref [B, M, H, dh].
    """
    B, M, R = Pq.shape
    Hdh = Vq.shape[1]
    # H inferred from cos_bmhd
    H = cos_bmhd.shape[2]
    dh = Hdh // H

    Q = torch.matmul(Pq, Vq)  # [B, M, H*dh]
    K = torch.matmul(Pk, Vk)
    V = torch.matmul(Pv, Vv)
    if bq is not None: Q = Q + bq
    if bk is not None: K = K + bk
    if bv is not None: V = V + bv
    Q = Q.view(B, M, H, dh)
    K = K.view(B, M, H, dh)
    V = V.view(B, M, H, dh)

    Qr = apply_rotary_bmhd(Q, cos_bmhd, sin_bmhd)
    Kr = apply_rotary_bmhd(K, cos_bmhd, sin_bmhd)

    scale = 1.0 / math.sqrt(dh)
    # scores [B, H, M, M]
    scores = torch.einsum("bmhd,bnhd->bhmn", Qr, Kr) * scale

    if attention_mask_2d is not None:
        am = attention_mask_2d.to(torch.bool)  # [B, M]
        qv = am[:, None, :, None]  # [B,1,M,1]
        kv = am[:, None, None, :]  # [B,1,1,M]
        scores = scores.masked_fill(~(qv & kv), float("-inf"))

    attn = torch.softmax(scores, dim=-1)  # [B,H,M,M]
    Oref = torch.einsum("bhmn,bnhd->bmhd", attn, V)  # [B,M,H,dh]
    return Oref


if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser("FlashSVD+RoPE kernel test (BMHd)")
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--dh", type=int, default=128)
    parser.add_argument("--R", type=int, default=32, help="rank-space dimension")
    parser.add_argument("--bm", type=int, default=64)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mask", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this test."); raise SystemExit(1)

    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    B, H, M, dh, R = args.B, args.H, args.M, args.dh, args.R
    D = H * dh

    # Rank-space activations and factors
    Pq = torch.randn(B, M, R, device=device, dtype=dtype)
    Pk = torch.randn(B, M, R, device=device, dtype=dtype)
    Pv = torch.randn(B, M, R, device=device, dtype=dtype)

    Vq = torch.randn(R, D, device=device, dtype=dtype).contiguous()
    Vk = torch.randn(R, D, device=device, dtype=dtype).contiguous()
    Vv = torch.randn(R, D, device=device, dtype=dtype).contiguous()

    bq = torch.randn(D, device=device, dtype=dtype).contiguous()
    bk = torch.randn(D, device=device, dtype=dtype).contiguous()
    bv = torch.randn(D, device=device, dtype=dtype).contiguous()

    # position ids & rotary
    position_ids = torch.arange(M, device=device)[None, :].expand(B, -1)
    rotary = _SimpleRotary(base=10000.0)

    # Make BMHd cos/sin for reference
    dummy = torch.empty((B * H, M, dh), device=device, dtype=dtype)
    posf = position_ids.unsqueeze(1).expand(B, H, M).reshape(B * H, M)
    cos, sin = rotary(dummy, position_ids=posf)
    cos_bmhd = cos.view(B, H, M, dh).permute(0, 2, 1, 3).contiguous()
    sin_bmhd = sin.view(B, H, M, dh).permute(0, 2, 1, 3).contiguous()

    # Optional 2D padding mask
    attention_mask = None
    if args.mask:
        valid_len = int(0.9 * M)
        attention_mask = torch.zeros(B, M, device=device, dtype=torch.int32)
        attention_mask[:, :valid_len] = 1

    # Kernel wrapper
    torch.cuda.synchronize()
    flash = FlashSVDRoPEAttention(num_heads=H, head_dim=dh, rotary_emb=rotary, bm=args.bm, bn=args.bn, bdh=dh).to(device)

    qkv = QKVFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

    # Warmup
    for _ in range(max(1, args.warmup)):
        _ = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()

    # Latency (kernel)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        O_kernel = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_kernel = (t1 - t0) * 1000.0 / max(1, args.iters)

    # Peak memory for one forward
    torch.cuda.reset_peak_memory_stats()
    _ = flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    # Reference (BMHd)
    Pq32, Pk32, Pv32 = Pq.float(), Pk.float(), Pv.float()
    Vq32, Vk32, Vv32 = Vq.float(), Vk.float(), Vv.float()
    bq32, bk32, bv32 = bq.float(), bk.float(), bv.float()
    cos32, sin32 = cos_bmhd.float(), sin_bmhd.float()
    am32 = attention_mask if attention_mask is None else attention_mask.int()

    for _ in range(3):
        _ = _reference_sdpa_with_rope_bmhd(Pq32, Pk32, Pv32, Vq32, Vk32, Vv32, bq32, bk32, bv32, cos32, sin32, am32)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(max(1, args.iters // 5)):
        O_ref = _reference_sdpa_with_rope_bmhd(Pq32, Pk32, Pv32, Vq32, Vk32, Vv32, bq32, bk32, bv32, cos32, sin32, am32)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ms_ref = (t1 - t0) * 1000.0 / max(1, args.iters // 5)

    # Compare
    O_k32 = O_kernel.float()  # BMHd
    diff = O_k32 - O_ref
    num = torch.linalg.norm(diff.reshape(B, -1), ord='fro')
    den = torch.linalg.norm(O_ref.reshape(B, -1), ord='fro')
    rel_fro = (num / (den + 1e-12)).item()
    max_abs = diff.abs().max().item()
    finite_kernel = torch.isfinite(O_kernel).all().item()
    finite_ref = torch.isfinite(O_ref).all().item()
    finite_diff = torch.isfinite(diff).all().item()

    def _pretty_mem(x):
        for u in ["B","KB","MB","GB","TB"]:
            if x < 1024: return f"{x:.0f} {u}"
            x /= 1024.0
        return f"{x:.0f} PB"

    print("===== FlashSVD+RoPE Triton Kernel Test (BMHd) =====")
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

    # Peak comparison helper
    def measure_peak_bytes(fn, *a, **k):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        return out, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    def bytes_to_mb(x): return x / (1024**2)

    _, kern_alloc, kern_res = measure_peak_bytes(
        flash, qkv, attention_mask=attention_mask, position_ids=position_ids
    )

    def run_ref():
        return _reference_sdpa_with_rope_bmhd(
            Pq.float(), Pk.float(), Pv.float(),
            Vq.float(), Vk.float(), Vv.float(),
            bq.float(), bk.float(), bv.float(),
            cos_bmhd.float(), sin_bmhd.float(),
            attention_mask if attention_mask is None else attention_mask.int()
        )

    torch.cuda.empty_cache()
    _, ref_alloc, ref_res = measure_peak_bytes(run_ref)

    print(f"\n--- Peak memory comparison ---")
    print(f"Kernel  peak allocated: {bytes_to_mb(kern_alloc):.1f} MB   reserved: {bytes_to_mb(kern_res):.1f} MB")
    print(f"Ref SDPA peak allocated: {bytes_to_mb(ref_alloc):.1f} MB   reserved: {bytes_to_mb(ref_res):.1f} MB")
    print(f"Savings (alloc): {bytes_to_mb(ref_alloc - kern_alloc):.1f} MB")
    print(f"Savings (reserv): {bytes_to_mb(ref_res  - kern_res ):.1f} MB")
