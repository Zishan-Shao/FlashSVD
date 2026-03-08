#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FlashSVD (low-rank) + RoPE + Online Softmax
FA-aligned mask semantics:

（这个是优化过的！！！）

- packed: like flash_attn_func (no attention_mask tensor)
  supports: causal, window_size

- varlen: like flash_attn_varlen_func (mask via cu_seqlens + max_seqlen)
  supports: causal, window_size

Natural LLaMA/HF-friendly layout:
Packed:
  Pq: [B, S, H,  R]
  Pk: [B, S, Hk, R]
  Pv: [B, S, Hk, R]
  Vq: [H,  R, Dh]
  Vk: [Hk, R, Dh]
  Vv: [Hk, R, Dh]
  bq: [H,  Dh] (optional)
  bk: [Hk, Dh] (optional)
  bv: [Hk, Dh] (optional)
  rotary cos/sin: [S, Dh/2]
  O: [B, S, H, Dh]

Varlen:
  Pq: [T, H,  R]
  Pk: [T, Hk, R]
  Pv: [T, Hk, R]
  O:  [T, H, Dh]
  cu_seqlens: [B+1] int32
  rotary cos/sin: [max_seqlen, Dh/2]


CUDA_VISIBLE_DEVICES=1 python flashsvdropeattn_v1.5.py --mode packed --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64   --dtype bf16 --bm 64 --bn 64 --br 64 --warps 8 --stages 3   --causal --warmup 50 --iters 1000
请帮我测试一下，全方位衡量我们的效果相比之前在哪方面好了多少？和他对齐的是
  
"""

import math
import time
import gc
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import triton
import triton.language as tl


# ----------------------------
# Factor containers
# ----------------------------
@dataclass
class PackedFactors:
    Pq: torch.Tensor  # [B, S, H,  R]
    Pk: torch.Tensor  # [B, S, Hk, R]
    Pv: torch.Tensor  # [B, S, Hk, R]
    Vq: torch.Tensor  # [H,  R, Dh]
    Vk: torch.Tensor  # [Hk, R, Dh]
    Vv: torch.Tensor  # [Hk, R, Dh]
    bq: Optional[torch.Tensor] = None  # [H,  Dh]
    bk: Optional[torch.Tensor] = None  # [Hk, Dh]
    bv: Optional[torch.Tensor] = None  # [Hk, Dh]


@dataclass
class VarlenFactors:
    Pq: torch.Tensor  # [T, H,  R]
    Pk: torch.Tensor  # [T, Hk, R]
    Pv: torch.Tensor  # [T, Hk, R]
    Vq: torch.Tensor  # [H,  R, Dh]
    Vk: torch.Tensor  # [Hk, R, Dh]
    Vv: torch.Tensor  # [Hk, R, Dh]
    bq: Optional[torch.Tensor] = None  # [H,  Dh]
    bk: Optional[torch.Tensor] = None  # [Hk, Dh]
    bv: Optional[torch.Tensor] = None  # [Hk, Dh]


# ----------------------------
# RoPE table builder (head-shared, half-dim only)
# ----------------------------
@torch.no_grad()
def build_rope_tables(
    seqlen: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0
    half = head_dim // 2
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.einsum("m,d->md", pos, inv_freq)  # [seqlen, half]
    cos = torch.cos(ang).to(dtype)
    sin = torch.sin(ang).to(dtype)
    return cos.contiguous(), sin.contiguous()


# ----------------------------
# Triton kernels
#   Key fixes for compatibility:
#   - tl.arange upper bound must be tl.constexpr -> pass HALF explicitly
#   - avoid tl.static_assert on dtype/expressions (older Triton can crash)
#   - select dtype via IS_BF16 tl.constexpr passed from wrapper
# ----------------------------

@triton.jit
def flashsvd_rope_fwd_packed_R(
    Pq_ptr, Pk_ptr, Pv_ptr,
    Vq_ptr, Vk_ptr, Vv_ptr,
    bq_ptr, bk_ptr, bv_ptr,
    COS_ptr, SIN_ptr,   # [S, HALF]
    O_ptr,
    # specialize
    S: tl.constexpr,
    R: tl.constexpr,
    # strides
    sPq_b, sPq_s, sPq_h, sPq_r,
    sPk_b, sPk_s, sPk_h, sPk_r,
    sPv_b, sPv_s, sPv_h, sPv_r,
    sVq_h, sVq_r, sVq_d,
    sVk_h, sVk_r, sVk_d,
    sVv_h, sVv_r, sVv_d,
    sbq_h, sbq_d,
    sbk_h, sbk_d,
    sbv_h, sbv_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sO_b, sO_s, sO_h, sO_d,
    # params
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_BQ: tl.constexpr,
    HAS_BK: tl.constexpr,
    HAS_BV: tl.constexpr,
    # tiling / shapes
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid0 = tl.program_id(0)  # 0..B*Hk-1
    pid1 = tl.program_id(1)  # query block

    bid = pid0 // Hk
    hid_k = pid0 % Hk

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    offs_m = pid1 * BM + tl.arange(0, BM)
    mask_m = offs_m < S

    offs_half = tl.arange(0, HALF)
    offs_d = tl.arange(0, DH)

    # RoPE for queries (head-shared)
    cos_q = tl.load(COS_ptr + offs_m[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
                    mask=mask_m[:, None], other=0.0).to(in_dtype)
    sin_q = tl.load(SIN_ptr + offs_m[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
                    mask=mask_m[:, None], other=0.0).to(in_dtype)

    # ---- Process each Q head separately to avoid list/tuple mutations ----
    for g in tl.static_range(0, REP):
        hid_q = hid_k * REP + g

        q0 = tl.zeros((BM, HALF), dtype=tl.float32)
        q1 = tl.zeros((BM, HALF), dtype=tl.float32)

        for r0 in range(0, R, BR):
            r = r0 + tl.arange(0, BR)
            mask_r = r < R

            Pq_blk = tl.load(
                Pq_ptr + bid * sPq_b + offs_m[:, None] * sPq_s + hid_q * sPq_h + r[None, :] * sPq_r,
                mask=mask_m[:, None] & mask_r[None, :],
                other=0.0
            ).to(in_dtype)

            Vq0 = tl.load(
                Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + offs_half[None, :] * sVq_d,
                mask=mask_r[:, None],
                other=0.0
            ).to(in_dtype)
            Vq1 = tl.load(
                Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + (offs_half + HALF)[None, :] * sVq_d,
                mask=mask_r[:, None],
                other=0.0
            ).to(in_dtype)

            q0 += tl.dot(Pq_blk, Vq0).to(tl.float32)
            q1 += tl.dot(Pq_blk, Vq1).to(tl.float32)

        if HAS_BQ:
            bq0 = tl.load(bq_ptr + hid_q * sbq_h + offs_half * sbq_d).to(tl.float32)
            bq1 = tl.load(bq_ptr + hid_q * sbq_h + (offs_half + HALF) * sbq_d).to(tl.float32)
            q0 += bq0[None, :]
            q1 += bq1[None, :]

        q0 = q0.to(in_dtype)
        q1 = q1.to(in_dtype)

        q0r = q0 * cos_q - q1 * sin_q
        q1r = q0 * sin_q + q1 * cos_q

        m_i = tl.full((BM,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, DH), dtype=tl.float32)

        for nk in range(0, S, BN):
            offs_n = nk + tl.arange(0, BN)
            valid_n = offs_n < S
            if CAUSAL:
                valid_n = valid_n & (offs_n <= (pid1 * BM + (BM - 1)))

            cos_k = tl.load(COS_ptr + offs_n[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
                            mask=valid_n[:, None], other=0.0).to(in_dtype)
            sin_k = tl.load(SIN_ptr + offs_n[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
                            mask=valid_n[:, None], other=0.0).to(in_dtype)

            # Build K/V for this kv head (recomputed per q head)
            k0 = tl.zeros((BN, HALF), dtype=tl.float32)
            k1 = tl.zeros((BN, HALF), dtype=tl.float32)
            v  = tl.zeros((BN, DH),   dtype=tl.float32)

            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R

                Pk_blk = tl.load(
                    Pk_ptr + bid * sPk_b + offs_n[:, None] * sPk_s + hid_k * sPk_h + r[None, :] * sPk_r,
                    mask=valid_n[:, None] & mask_r[None, :],
                    other=0.0
                ).to(in_dtype)
                Pv_blk = tl.load(
                    Pv_ptr + bid * sPv_b + offs_n[:, None] * sPv_s + hid_k * sPv_h + r[None, :] * sPv_r,
                    mask=valid_n[:, None] & mask_r[None, :],
                    other=0.0
                ).to(in_dtype)

                Vk0 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
                    mask=mask_r[:, None], other=0.0
                ).to(in_dtype)
                Vk1 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
                    mask=mask_r[:, None], other=0.0
                ).to(in_dtype)
                Vv_sub = tl.load(
                    Vv_ptr + hid_k * sVv_h + r[:, None] * sVv_r + offs_d[None, :] * sVv_d,
                    mask=mask_r[:, None], other=0.0
                ).to(in_dtype)

                k0 += tl.dot(Pk_blk, Vk0).to(tl.float32)
                k1 += tl.dot(Pk_blk, Vk1).to(tl.float32)
                v  += tl.dot(Pv_blk, Vv_sub).to(tl.float32)

            if HAS_BK:
                bk0 = tl.load(bk_ptr + hid_k * sbk_h + offs_half * sbk_d).to(tl.float32)
                bk1 = tl.load(bk_ptr + hid_k * sbk_h + (offs_half + HALF) * sbk_d).to(tl.float32)
                k0 += bk0[None, :]
                k1 += bk1[None, :]
            if HAS_BV:
                bv_sub = tl.load(bv_ptr + hid_k * sbv_h + offs_d * sbv_d).to(tl.float32)
                v += bv_sub[None, :]

            k0 = k0.to(in_dtype)
            k1 = k1.to(in_dtype)
            v  = v.to(in_dtype)

            k0r = k0 * cos_k - k1 * sin_k
            k1r = k0 * sin_k + k1 * cos_k

            scores = tl.dot(q0r, tl.trans(k0r)).to(tl.float32)
            scores += tl.dot(q1r, tl.trans(k1r)).to(tl.float32)
            scores *= SOFTMAX_SCALE

            if CAUSAL:
                causal_mask = offs_n[None, :] <= offs_m[:, None]
                scores = tl.where(causal_mask, scores, -float("inf"))

            if WINDOW_LEFT != -1 or WINDOW_RIGHT != -1:
                left_ok  = True
                right_ok = True
                if WINDOW_LEFT != -1:
                    left_ok = offs_n[None, :] >= (offs_m[:, None] - WINDOW_LEFT)
                if WINDOW_RIGHT != -1:
                    right_ok = offs_n[None, :] <= (offs_m[:, None] + WINDOW_RIGHT)
                scores = tl.where(left_ok & right_ok, scores, -float("inf"))

            scores = tl.where(valid_n[None, :], scores, -float("inf"))
            scores = tl.where(mask_m[:, None], scores, -float("inf"))

            m_curr = tl.max(scores, axis=1)
            m_new  = tl.maximum(m_i, m_curr)
            alpha  = tl.exp(m_i - m_new)
            p      = tl.exp(scores - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)

            p_tc = p.to(in_dtype)
            acc = acc * alpha[:, None] + tl.dot(p_tc, v).to(tl.float32)
            m_i = m_new

        if hid_q < H:
            den = tl.where(l_i > 0, l_i, 1.0)
            out = acc / den[:, None]
            out = tl.where(l_i[:, None] > 0, out, 0.0)

            tl.store(
                O_ptr + bid * sO_b + offs_m[:, None] * sO_s + hid_q * sO_h + offs_d[None, :] * sO_d,
                out.to(in_dtype),
                mask=mask_m[:, None]
            )


@triton.jit
def flashsvd_rope_fwd_packed_R_value_in_rank(
    Pq_ptr, Pk_ptr, Pv_ptr,
    Vq_ptr, Vk_ptr, Vv_ptr,
    bq_ptr, bk_ptr, bv_ptr,
    COS_ptr, SIN_ptr,   # [S, HALF]
    O_ptr,
    # specialize
    S: tl.constexpr,
    R: tl.constexpr,
    # strides
    sPq_b, sPq_s, sPq_h, sPq_r,
    sPk_b, sPk_s, sPk_h, sPk_r,
    sPv_b, sPv_s, sPv_h, sPv_r,
    sVq_h, sVq_r, sVq_d,
    sVk_h, sVk_r, sVk_d,
    sVv_h, sVv_r, sVv_d,
    sbq_h, sbq_d,
    sbk_h, sbk_d,
    sbv_h, sbv_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sO_b, sO_s, sO_h, sO_d,
    # params
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_BQ: tl.constexpr,
    HAS_BK: tl.constexpr,
    HAS_BV: tl.constexpr,
    # tiling / shapes
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    Same semantics as `flashsvd_rope_fwd_packed_R`, but **accumulates values in rank-space**:
      acc_r = sum_j softmax(score_ij) * Pv[j, :]
      out   = (acc_r @ Vv) + bv

    This avoids reconstructing dense V tiles and the (BMxBN)@(BNxDH) matmul in every key block.
    """
    pid0 = tl.program_id(0)  # 0..B*Hk-1
    pid1 = tl.program_id(1)  # query block

    bid = pid0 // Hk
    hid_k = pid0 % Hk

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    offs_m = pid1 * BM + tl.arange(0, BM)
    mask_m = offs_m < S

    offs_half = tl.arange(0, HALF)
    offs_r = tl.arange(0, R)

    # RoPE for queries (head-shared)
    cos_q = tl.load(
        COS_ptr + offs_m[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
        mask=mask_m[:, None],
        other=0.0,
    ).to(in_dtype)
    sin_q = tl.load(
        SIN_ptr + offs_m[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
        mask=mask_m[:, None],
        other=0.0,
    ).to(in_dtype)

    # ---- Process each Q head separately ----
    for g in tl.static_range(0, REP):
        hid_q = hid_k * REP + g

        q0 = tl.zeros((BM, HALF), dtype=tl.float32)
        q1 = tl.zeros((BM, HALF), dtype=tl.float32)

        for r0 in range(0, R, BR):
            r = r0 + tl.arange(0, BR)
            mask_r = r < R

            Pq_blk = tl.load(
                Pq_ptr + bid * sPq_b + offs_m[:, None] * sPq_s + hid_q * sPq_h + r[None, :] * sPq_r,
                mask=mask_m[:, None] & mask_r[None, :],
                other=0.0,
            ).to(in_dtype)

            Vq0 = tl.load(
                Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + offs_half[None, :] * sVq_d,
                mask=mask_r[:, None],
                other=0.0,
            ).to(in_dtype)
            Vq1 = tl.load(
                Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + (offs_half + HALF)[None, :] * sVq_d,
                mask=mask_r[:, None],
                other=0.0,
            ).to(in_dtype)

            q0 += tl.dot(Pq_blk, Vq0).to(tl.float32)
            q1 += tl.dot(Pq_blk, Vq1).to(tl.float32)

        if HAS_BQ:
            bq0 = tl.load(bq_ptr + hid_q * sbq_h + offs_half * sbq_d).to(tl.float32)
            bq1 = tl.load(bq_ptr + hid_q * sbq_h + (offs_half + HALF) * sbq_d).to(tl.float32)
            q0 += bq0[None, :]
            q1 += bq1[None, :]

        q0 = q0.to(in_dtype)
        q1 = q1.to(in_dtype)

        q0r = q0 * cos_q - q1 * sin_q
        q1r = q0 * sin_q + q1 * cos_q

        m_i = tl.full((BM,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acc_r = tl.zeros((BM, R), dtype=tl.float32)  # rank-space accumulator

        for nk in range(0, S, BN):
            offs_n = nk + tl.arange(0, BN)
            valid_n = offs_n < S
            if CAUSAL:
                valid_n = valid_n & (offs_n <= (pid1 * BM + (BM - 1)))

            cos_k = tl.load(
                COS_ptr + offs_n[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
                mask=valid_n[:, None],
                other=0.0,
            ).to(in_dtype)
            sin_k = tl.load(
                SIN_ptr + offs_n[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
                mask=valid_n[:, None],
                other=0.0,
            ).to(in_dtype)

            # Build K for this kv head (recomputed per q head)
            k0 = tl.zeros((BN, HALF), dtype=tl.float32)
            k1 = tl.zeros((BN, HALF), dtype=tl.float32)

            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R

                Pk_blk = tl.load(
                    Pk_ptr + bid * sPk_b + offs_n[:, None] * sPk_s + hid_k * sPk_h + r[None, :] * sPk_r,
                    mask=valid_n[:, None] & mask_r[None, :],
                    other=0.0,
                ).to(in_dtype)

                Vk0 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
                    mask=mask_r[:, None],
                    other=0.0,
                ).to(in_dtype)
                Vk1 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
                    mask=mask_r[:, None],
                    other=0.0,
                ).to(in_dtype)

                k0 += tl.dot(Pk_blk, Vk0).to(tl.float32)
                k1 += tl.dot(Pk_blk, Vk1).to(tl.float32)

            if HAS_BK:
                bk0 = tl.load(bk_ptr + hid_k * sbk_h + offs_half * sbk_d).to(tl.float32)
                bk1 = tl.load(bk_ptr + hid_k * sbk_h + (offs_half + HALF) * sbk_d).to(tl.float32)
                k0 += bk0[None, :]
                k1 += bk1[None, :]

            k0 = k0.to(in_dtype)
            k1 = k1.to(in_dtype)

            k0r = k0 * cos_k - k1 * sin_k
            k1r = k0 * sin_k + k1 * cos_k

            scores = tl.dot(q0r, tl.trans(k0r)).to(tl.float32)
            scores += tl.dot(q1r, tl.trans(k1r)).to(tl.float32)
            scores *= SOFTMAX_SCALE

            if CAUSAL:
                causal_mask = offs_n[None, :] <= offs_m[:, None]
                scores = tl.where(causal_mask, scores, -float("inf"))

            if WINDOW_LEFT != -1 or WINDOW_RIGHT != -1:
                left_ok = True
                right_ok = True
                if WINDOW_LEFT != -1:
                    left_ok = offs_n[None, :] >= (offs_m[:, None] - WINDOW_LEFT)
                if WINDOW_RIGHT != -1:
                    right_ok = offs_n[None, :] <= (offs_m[:, None] + WINDOW_RIGHT)
                scores = tl.where(left_ok & right_ok, scores, -float("inf"))

            scores = tl.where(valid_n[None, :], scores, -float("inf"))
            scores = tl.where(mask_m[:, None], scores, -float("inf"))

            m_curr = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_curr)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)

            p_tc = p.to(in_dtype)
            Pv_blk = tl.load(
                Pv_ptr + bid * sPv_b + offs_n[:, None] * sPv_s + hid_k * sPv_h + offs_r[None, :] * sPv_r,
                mask=valid_n[:, None] & (offs_r[None, :] < R),
                other=0.0,
            ).to(in_dtype)
            acc_r = acc_r * alpha[:, None] + tl.dot(p_tc, Pv_blk).to(tl.float32)
            m_i = m_new

        if hid_q < H:
            den = tl.where(l_i > 0, l_i, 1.0)
            w_r = acc_r / den[:, None]
            w_r = tl.where(l_i[:, None] > 0, w_r, 0.0)
            w_tc = w_r.to(in_dtype)

            # Lift back to head dim in two halves to reduce peak register pressure.
            Vv0 = tl.load(
                Vv_ptr + hid_k * sVv_h + offs_r[:, None] * sVv_r + offs_half[None, :] * sVv_d,
                mask=True,
                other=0.0,
            ).to(in_dtype)
            Vv1 = tl.load(
                Vv_ptr + hid_k * sVv_h + offs_r[:, None] * sVv_r + (offs_half + HALF)[None, :] * sVv_d,
                mask=True,
                other=0.0,
            ).to(in_dtype)

            out0 = tl.dot(w_tc, Vv0).to(tl.float32)
            out1 = tl.dot(w_tc, Vv1).to(tl.float32)

            if HAS_BV:
                bv0 = tl.load(bv_ptr + hid_k * sbv_h + offs_half * sbv_d).to(tl.float32)
                bv1 = tl.load(bv_ptr + hid_k * sbv_h + (offs_half + HALF) * sbv_d).to(tl.float32)
                out0 = tl.where(l_i[:, None] > 0, out0 + bv0[None, :], 0.0)
                out1 = tl.where(l_i[:, None] > 0, out1 + bv1[None, :], 0.0)
            else:
                out0 = tl.where(l_i[:, None] > 0, out0, 0.0)
                out1 = tl.where(l_i[:, None] > 0, out1, 0.0)

            tl.store(
                O_ptr + bid * sO_b + offs_m[:, None] * sO_s + hid_q * sO_h + offs_half[None, :] * sO_d,
                out0.to(in_dtype),
                mask=mask_m[:, None],
            )
            tl.store(
                O_ptr + bid * sO_b + offs_m[:, None] * sO_s + hid_q * sO_h + (offs_half + HALF)[None, :] * sO_d,
                out1.to(in_dtype),
                mask=mask_m[:, None],
            )


@triton.jit
def flashsvd_rope_fwd_varlen_R(
    Pq_ptr, Pk_ptr, Pv_ptr,
    Vq_ptr, Vk_ptr, Vv_ptr,
    bq_ptr, bk_ptr, bv_ptr,
    COS_ptr, SIN_ptr,        # [max_seqlen, HALF]
    O_ptr,
    cu_seqlens_ptr,          # [B+1] int32
    # specialize
    max_seqlen: tl.constexpr,
    R: tl.constexpr,
    # strides
    sPq_t, sPq_h, sPq_r,
    sPk_t, sPk_h, sPk_r,
    sPv_t, sPv_h, sPv_r,
    sVq_h, sVq_r, sVq_d,
    sVk_h, sVk_r, sVk_d,
    sVv_h, sVv_r, sVv_d,
    sbq_h, sbq_d,
    sbk_h, sbk_d,
    sbv_h, sbv_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sO_t, sO_h, sO_d,
    # params
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_BQ: tl.constexpr,
    HAS_BK: tl.constexpr,
    HAS_BV: tl.constexpr,
    # tiling / shapes
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    bid = pid0 // Hk
    hid_k = pid0 % Hk

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    start = tl.load(cu_seqlens_ptr + bid, mask=True).to(tl.int32)
    end   = tl.load(cu_seqlens_ptr + bid + 1, mask=True).to(tl.int32)
    seqlen = end - start

    offs_m = pid1 * BM + tl.arange(0, BM)
    mask_m = offs_m < seqlen
    q_idx  = start + offs_m

    offs_half = tl.arange(0, HALF)
    offs_d = tl.arange(0, DH)

    cos_q = tl.load(COS_ptr + offs_m[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
                    mask=mask_m[:, None], other=0.0).to(in_dtype)
    sin_q = tl.load(SIN_ptr + offs_m[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
                    mask=mask_m[:, None], other=0.0).to(in_dtype)
    for g in tl.static_range(0, REP):
        hid_q = hid_k * REP + g

        q0 = tl.zeros((BM, HALF), dtype=tl.float32)
        q1 = tl.zeros((BM, HALF), dtype=tl.float32)

        for r0 in range(0, R, BR):
            r = r0 + tl.arange(0, BR)
            mask_r = r < R

            Pq_blk = tl.load(
                Pq_ptr + q_idx[:, None] * sPq_t + hid_q * sPq_h + r[None, :] * sPq_r,
                mask=mask_m[:, None] & mask_r[None, :],
                other=0.0
            ).to(in_dtype)

            Vq0 = tl.load(Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + offs_half[None, :] * sVq_d,
                          mask=mask_r[:, None], other=0.0).to(in_dtype)
            Vq1 = tl.load(Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + (offs_half + HALF)[None, :] * sVq_d,
                          mask=mask_r[:, None], other=0.0).to(in_dtype)

            q0 += tl.dot(Pq_blk, Vq0).to(tl.float32)
            q1 += tl.dot(Pq_blk, Vq1).to(tl.float32)

        if HAS_BQ:
            bq0 = tl.load(bq_ptr + hid_q * sbq_h + offs_half * sbq_d).to(tl.float32)
            bq1 = tl.load(bq_ptr + hid_q * sbq_h + (offs_half + HALF) * sbq_d).to(tl.float32)
            q0 += bq0[None, :]
            q1 += bq1[None, :]

        q0 = q0.to(in_dtype)
        q1 = q1.to(in_dtype)
        q0r = q0 * cos_q - q1 * sin_q
        q1r = q0 * sin_q + q1 * cos_q

        m_i = tl.full((BM,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BM,), dtype=tl.float32)
        acc = tl.zeros((BM, DH), dtype=tl.float32)

        for nk in range(0, max_seqlen, BN):
            offs_n = nk + tl.arange(0, BN)
            mask_n = offs_n < seqlen
            k_idx  = start + offs_n

            cos_k = tl.load(COS_ptr + offs_n[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
                            mask=mask_n[:, None], other=0.0).to(in_dtype)
            sin_k = tl.load(SIN_ptr + offs_n[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
                            mask=mask_n[:, None], other=0.0).to(in_dtype)

            k0 = tl.zeros((BN, HALF), dtype=tl.float32)
            k1 = tl.zeros((BN, HALF), dtype=tl.float32)
            v  = tl.zeros((BN, DH),   dtype=tl.float32)

            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R

                Pk_blk = tl.load(
                    Pk_ptr + k_idx[:, None] * sPk_t + hid_k * sPk_h + r[None, :] * sPk_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0
                ).to(in_dtype)
                Pv_blk = tl.load(
                    Pv_ptr + k_idx[:, None] * sPv_t + hid_k * sPv_h + r[None, :] * sPv_r,
                    mask=mask_n[:, None] & mask_r[None, :],
                    other=0.0
                ).to(in_dtype)

                Vk0 = tl.load(Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
                              mask=mask_r[:, None], other=0.0).to(in_dtype)
                Vk1 = tl.load(Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
                              mask=mask_r[:, None], other=0.0).to(in_dtype)
                Vv_sub = tl.load(Vv_ptr + hid_k * sVv_h + r[:, None] * sVv_r + offs_d[None, :] * sVv_d,
                                 mask=mask_r[:, None], other=0.0).to(in_dtype)

                k0 += tl.dot(Pk_blk, Vk0).to(tl.float32)
                k1 += tl.dot(Pk_blk, Vk1).to(tl.float32)
                v  += tl.dot(Pv_blk, Vv_sub).to(tl.float32)

            if HAS_BK:
                bk0 = tl.load(bk_ptr + hid_k * sbk_h + offs_half * sbk_d).to(tl.float32)
                bk1 = tl.load(bk_ptr + hid_k * sbk_h + (offs_half + HALF) * sbk_d).to(tl.float32)
                k0 += bk0[None, :]
                k1 += bk1[None, :]
            if HAS_BV:
                bv_sub = tl.load(bv_ptr + hid_k * sbv_h + offs_d * sbv_d).to(tl.float32)
                v += bv_sub[None, :]

            k0 = k0.to(in_dtype)
            k1 = k1.to(in_dtype)
            v  = v.to(in_dtype)

            k0r = k0 * cos_k - k1 * sin_k
            k1r = k0 * sin_k + k1 * cos_k

            scores = tl.dot(q0r, tl.trans(k0r)).to(tl.float32)
            scores += tl.dot(q1r, tl.trans(k1r)).to(tl.float32)
            scores *= SOFTMAX_SCALE

            if CAUSAL:
                scores = tl.where(offs_n[None, :] <= offs_m[:, None], scores, -float("inf"))

            if WINDOW_LEFT != -1 or WINDOW_RIGHT != -1:
                left_ok  = True
                right_ok = True
                if WINDOW_LEFT != -1:
                    left_ok = offs_n[None, :] >= (offs_m[:, None] - WINDOW_LEFT)
                if WINDOW_RIGHT != -1:
                    right_ok = offs_n[None, :] <= (offs_m[:, None] + WINDOW_RIGHT)
                scores = tl.where(left_ok & right_ok, scores, -float("inf"))

            scores = tl.where(mask_n[None, :], scores, -float("inf"))
            scores = tl.where(mask_m[:, None], scores, -float("inf"))

            m_curr = tl.max(scores, axis=1)
            m_new  = tl.maximum(m_i, m_curr)
            alpha  = tl.exp(m_i - m_new)
            p      = tl.exp(scores - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(in_dtype), v).to(tl.float32)
            m_i = m_new

        if hid_q < H:
            den = tl.where(l_i > 0, l_i, 1.0)
            out = acc / den[:, None]
            out = tl.where(l_i[:, None] > 0, out, 0.0)
            tl.store(
                O_ptr + q_idx[:, None] * sO_t + hid_q * sO_h + offs_d[None, :] * sO_d,
                out.to(in_dtype),
                mask=mask_m[:, None]
            )




# ----------------------------
# Decode (Q=1) split-KV kernels (Flash-Decoding style) for low-rank KV
#   - Keeps K in low-rank (Pk @ Vk) and applies RoPE on-the-fly
#   - Accumulates values in rank-space (value-in-rank) and lifts once at the end
#   - Uses Split-KV (sequence splits) to increase parallelism in decoding
#   - Fuses GQA head-group (REP) inside one program so K/Pv are decompressed once per kv head
# ----------------------------

@dataclass
class DecodePackedFactors:
    # Query low-rank factor (single token per sequence)
    Pq: torch.Tensor            # [B, H,  R]
    # KV caches in low-rank factor form (allocated to max_seqlen)
    Pk: torch.Tensor            # [B, Smax, Hk, R]
    Pv: torch.Tensor            # [B, Smax, Hk, R]
    # Projection bases / weights
    Vq: torch.Tensor            # [H,  R, Dh]
    Vk: torch.Tensor            # [Hk, R, Dh]
    Vv: torch.Tensor            # [Hk, R, Dh]
    # Optional biases
    bq: Optional[torch.Tensor] = None  # [H,  Dh]
    bk: Optional[torch.Tensor] = None  # [Hk, Dh]
    bv: Optional[torch.Tensor] = None  # [Hk, Dh]


@dataclass
class DecodeVarlenFactors:
    # Query low-rank factor (single token per sequence)
    Pq: torch.Tensor            # [B, H,  R]
    # KV caches in low-rank factor form (ragged batch stored contiguously)
    Pk: torch.Tensor            # [T, Hk, R]
    Pv: torch.Tensor            # [T, Hk, R]
    # Projection bases / weights
    Vq: torch.Tensor            # [H,  R, Dh]
    Vk: torch.Tensor            # [Hk, R, Dh]
    Vv: torch.Tensor            # [Hk, R, Dh]
    # Optional biases
    bq: Optional[torch.Tensor] = None  # [H,  Dh]
    bk: Optional[torch.Tensor] = None  # [Hk, Dh]
    bv: Optional[torch.Tensor] = None  # [Hk, Dh]


@triton.jit
def flashsvd_rope_decode_splitk_stage1_packed(
    # query
    Pq_q_ptr,                 # [B, H, R]
    # kv caches
    Pk_ptr, Pv_ptr,           # [B, Smax, Hk, R]
    # bases
    Vq_ptr, Vk_ptr,           # Vv not needed in stage1
    # biases
    bq_ptr, bk_ptr,
    # rope tables
    COS_ptr, SIN_ptr,         # [Smax, HALF]
    # outputs (partial states)
    M_ptr, L_ptr, Acc_ptr,    # M/L: [B, H, NSPLIT], Acc: [B, H, NSPLIT, R]
    # runtime lengths
    seqlen_k,                 # int32, current cache length (<= Smax)
    # strides
    sPq_b, sPq_h, sPq_r,
    sPk_b, sPk_s, sPk_h, sPk_r,
    sPv_b, sPv_s, sPv_h, sPv_r,
    sVq_h, sVq_r, sVq_d,
    sVk_h, sVk_r, sVk_d,
    sbq_h, sbq_d,
    sbk_h, sbk_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sM_b, sM_h, sM_s,
    sL_b, sL_h, sL_s,
    sA_b, sA_h, sA_s, sA_r,
    # params
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_BQ: tl.constexpr,
    HAS_BK: tl.constexpr,
    # shapes
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    # tiling
    BN: tl.constexpr,
    BR: tl.constexpr,
    SPLIT_K: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    Stage1 (per split): compute partial softmax states in rank space for decode (Q_len=1).

    Program mapping:
      pid0 -> (bid, hid_k)  in [0, B*Hk)
      pid1 -> split_id      in [0, NSPLIT)

    This program computes REP query heads that share the same kv head (GQA head-group fusion).
    """
    pid0 = tl.program_id(0)
    split_id = tl.program_id(1)

    bid = pid0 // Hk
    hid_k = pid0 % Hk

    # current q position (assume query attends up to seqlen_k-1)
    # (clamp to 0 to avoid OOB loads if seqlen_k==0; seqlen_k==0 should be rare in practice)
    q_pos = tl.maximum(seqlen_k - 1, 0)

    # split range in key positions (0..seqlen_k)
    split_start = split_id * SPLIT_K

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16
    offs_half = tl.arange(0, HALF)
    offs_r_full = tl.arange(0, R)

    # --- Load RoPE table for query position ---
    cos_q = tl.load(COS_ptr + q_pos * sCOS_s + offs_half * sCOS_d).to(in_dtype)
    sin_q = tl.load(SIN_ptr + q_pos * sSIN_s + offs_half * sSIN_d).to(in_dtype)

    # --- Build Q for REP heads (dense, roped) ---
    q0 = tl.zeros((REP, HALF), dtype=tl.float32)
    q1 = tl.zeros((REP, HALF), dtype=tl.float32)

    hid_qs = hid_k * REP + tl.arange(0, REP)
    mask_hq = hid_qs < H

    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        mask_r = r < R

        Pq_blk = tl.load(
            Pq_q_ptr + bid * sPq_b + hid_qs[:, None] * sPq_h + r[None, :] * sPq_r,
            mask=mask_hq[:, None] & mask_r[None, :],
            other=0.0,
        ).to(in_dtype)

        Vq0 = tl.load(
            Vq_ptr + hid_qs[:, None, None] * sVq_h + r[None, :, None] * sVq_r + offs_half[None, None, :] * sVq_d,
            mask=mask_hq[:, None, None] & mask_r[None, :, None],
            other=0.0,
        ).to(in_dtype)
        Vq1 = tl.load(
            Vq_ptr + hid_qs[:, None, None] * sVq_h + r[None, :, None] * sVq_r + (offs_half + HALF)[None, None, :] * sVq_d,
            mask=mask_hq[:, None, None] & mask_r[None, :, None],
            other=0.0,
        ).to(in_dtype)

        q0 += tl.sum(Pq_blk[:, :, None].to(tl.float32) * Vq0.to(tl.float32), axis=1)
        q1 += tl.sum(Pq_blk[:, :, None].to(tl.float32) * Vq1.to(tl.float32), axis=1)

    if HAS_BQ:
        bq0 = tl.load(
            bq_ptr + hid_qs[:, None] * sbq_h + offs_half[None, :] * sbq_d,
            mask=mask_hq[:, None],
            other=0.0,
        ).to(tl.float32)
        bq1 = tl.load(
            bq_ptr + hid_qs[:, None] * sbq_h + (offs_half + HALF)[None, :] * sbq_d,
            mask=mask_hq[:, None],
            other=0.0,
        ).to(tl.float32)
        q0 += bq0
        q1 += bq1

    q0 = q0.to(in_dtype)
    q1 = q1.to(in_dtype)

    q0r = q0 * cos_q[None, :] - q1 * sin_q[None, :]
    q1r = q0 * sin_q[None, :] + q1 * cos_q[None, :]

    # --- Initialize per-head partial softmax state ---
    m_i = tl.full((REP,), -float("inf"), tl.float32)
    l_i = tl.zeros((REP,), tl.float32)
    acc_r = tl.zeros((REP, R), tl.float32)

    # --- Key blocks within the split ---
    for nk_off in range(0, SPLIT_K, BN):
        nk = split_start + nk_off
        offs_n = nk + tl.arange(0, BN)
        valid_n = offs_n < seqlen_k

        if CAUSAL:
            valid_n = valid_n & (offs_n <= q_pos)
        if WINDOW_LEFT != -1:
            valid_n = valid_n & (offs_n >= (q_pos - WINDOW_LEFT))
        if WINDOW_RIGHT != -1:
            valid_n = valid_n & (offs_n <= (q_pos + WINDOW_RIGHT))

        # RoPE tables for keys
        cos_k = tl.load(
            COS_ptr + offs_n[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)
        sin_k = tl.load(
            SIN_ptr + offs_n[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)

        # Decompress K for this kv head
        k0 = tl.zeros((BN, HALF), dtype=tl.float32)
        k1 = tl.zeros((BN, HALF), dtype=tl.float32)

        for r0 in range(0, R, BR):
            r = r0 + tl.arange(0, BR)
            mask_r = r < R

            Pk_blk = tl.load(
                Pk_ptr + bid * sPk_b + offs_n[:, None] * sPk_s + hid_k * sPk_h + r[None, :] * sPk_r,
                mask=valid_n[:, None] & mask_r[None, :],
                other=0.0,
            ).to(in_dtype)

            Vk0 = tl.load(
                Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
                mask=mask_r[:, None],
                other=0.0,
            ).to(in_dtype)
            Vk1 = tl.load(
                Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
                mask=mask_r[:, None],
                other=0.0,
            ).to(in_dtype)

            k0 += tl.dot(Pk_blk, Vk0).to(tl.float32)
            k1 += tl.dot(Pk_blk, Vk1).to(tl.float32)

        if HAS_BK:
            bk0 = tl.load(bk_ptr + hid_k * sbk_h + offs_half * sbk_d).to(tl.float32)
            bk1 = tl.load(bk_ptr + hid_k * sbk_h + (offs_half + HALF) * sbk_d).to(tl.float32)
            k0 += bk0[None, :]
            k1 += bk1[None, :]

        k0 = k0.to(in_dtype)
        k1 = k1.to(in_dtype)

        k0r = k0 * cos_k - k1 * sin_k
        k1r = k0 * sin_k + k1 * cos_k

        # NOTE: REP can be < 16 (GQA). tl.dot enforces M/N/K >= 16 in newer Triton,
        # so use explicit mul+reduce here to keep the same math.
        scores = tl.sum(q0r.to(tl.float32)[:, None, :] * k0r.to(tl.float32)[None, :, :], axis=2)
        scores += tl.sum(q1r.to(tl.float32)[:, None, :] * k1r.to(tl.float32)[None, :, :], axis=2)
        scores *= SOFTMAX_SCALE

        scores = tl.where(valid_n[None, :], scores, -float("inf"))
        scores = tl.where(mask_hq[:, None], scores, -float("inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        # Guard the empty/all-masked case: exp(-inf - -inf) -> NaNs.
        is_neg_inf = m_new == -float("inf")
        alpha = tl.where(is_neg_inf, 1.0, tl.exp(m_i - m_new))
        p = tl.where(is_neg_inf[:, None], 0.0, tl.exp(scores - m_new[:, None]))

        l_i = l_i * alpha + tl.sum(p, axis=1)

        Pv_blk = tl.load(
            Pv_ptr + bid * sPv_b + offs_n[:, None] * sPv_s + hid_k * sPv_h + offs_r_full[None, :] * sPv_r,
            mask=valid_n[:, None] & (offs_r_full[None, :] < R),
            other=0.0,
        ).to(in_dtype)

        acc_r = acc_r * alpha[:, None] + tl.sum(p[:, :, None] * Pv_blk.to(tl.float32)[None, :, :], axis=1)
        m_i = tl.where(is_neg_inf, m_i, m_new)

    # --- Write partial states ---
    # Avoid scalar indexing like m_i[g] (can be rejected by some Triton versions).
    tl.store(
        M_ptr + bid * sM_b + hid_qs * sM_h + split_id * sM_s,
        m_i,
        mask=mask_hq,
    )
    tl.store(
        L_ptr + bid * sL_b + hid_qs * sL_h + split_id * sL_s,
        l_i,
        mask=mask_hq,
    )
    tl.store(
        Acc_ptr + bid * sA_b + hid_qs[:, None] * sA_h + split_id * sA_s + offs_r_full[None, :] * sA_r,
        acc_r,
        mask=mask_hq[:, None] & (offs_r_full[None, :] < R),
    )


@triton.jit
def flashsvd_rope_decode_splitk_reduce_packed(
    # partial states
    M_ptr, L_ptr, Acc_ptr,     # [B,H,NSPLIT] and [B,H,NSPLIT,R]
    # lift
    Vv_ptr, bv_ptr,
    # output
    O_ptr,                     # [B,H,Dh]
    # strides
    sM_b, sM_h, sM_s,
    sL_b, sL_h, sL_s,
    sA_b, sA_h, sA_s, sA_r,
    sVv_h, sVv_r, sVv_d,
    sbv_h, sbv_d,
    sO_b, sO_h, sO_d,
    # params
    HAS_BV: tl.constexpr,
    # shapes
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    # split
    NSPLIT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    Stage2 reduction over splits for decode: merges (m,l,acc_r) and produces final output.

    Program mapping:
      pid0 -> (bid, hid_q) in [0, B*H)
    """
    pid0 = tl.program_id(0)
    bid = pid0 // H
    hid_q = pid0 % H
    hid_k = hid_q // REP

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16
    offs_half = tl.arange(0, HALF)
    offs_r = tl.arange(0, R)

    m = -float("inf")
    l = 0.0
    acc = tl.zeros((R,), dtype=tl.float32)

    for s in range(0, NSPLIT):
        m_s = tl.load(M_ptr + bid * sM_b + hid_q * sM_h + s * sM_s)
        l_s = tl.load(L_ptr + bid * sL_b + hid_q * sL_h + s * sL_s)
        a_s = tl.load(
            Acc_ptr + bid * sA_b + hid_q * sA_h + s * sA_s + offs_r * sA_r,
            mask=offs_r < R,
            other=0.0,
        ).to(tl.float32)

        m_new = tl.maximum(m, m_s)
        # Guard the all-masked case where m_new=-inf.
        is_neg_inf = m_new == -float("inf")
        alpha = tl.where(is_neg_inf, 1.0, tl.exp(m - m_new))
        beta = tl.where(is_neg_inf, 0.0, tl.exp(m_s - m_new))
        l = l * alpha + l_s * beta
        acc = acc * alpha + a_s * beta
        m = tl.where(is_neg_inf, m, m_new)

    den = tl.where(l > 0, l, 1.0)
    w_r = acc / den
    w_r = tl.where(l > 0, w_r, 0.0)

    Vv0 = tl.load(
        Vv_ptr + hid_k * sVv_h + offs_r[:, None] * sVv_r + offs_half[None, :] * sVv_d,
        mask=offs_r[:, None] < R,
        other=0.0,
    ).to(in_dtype)
    Vv1 = tl.load(
        Vv_ptr + hid_k * sVv_h + offs_r[:, None] * sVv_r + (offs_half + HALF)[None, :] * sVv_d,
        mask=offs_r[:, None] < R,
        other=0.0,
    ).to(in_dtype)

    # M=1 here; avoid tl.dot constraints for small matrices.
    out0 = tl.sum(w_r[:, None] * Vv0.to(tl.float32), axis=0)
    out1 = tl.sum(w_r[:, None] * Vv1.to(tl.float32), axis=0)

    if HAS_BV:
        bv0 = tl.load(bv_ptr + hid_k * sbv_h + offs_half * sbv_d).to(tl.float32)
        bv1 = tl.load(bv_ptr + hid_k * sbv_h + (offs_half + HALF) * sbv_d).to(tl.float32)
        out0 = out0 + bv0
        out1 = out1 + bv1

    tl.store(
        O_ptr + bid * sO_b + hid_q * sO_h + offs_half * sO_d,
        out0.to(in_dtype),
        mask=True,
    )
    tl.store(
        O_ptr + bid * sO_b + hid_q * sO_h + (offs_half + HALF) * sO_d,
        out1.to(in_dtype),
        mask=True,
    )


@triton.jit
def flashsvd_rope_decode_splitk_stage1_varlen(
    # query
    Pq_q_ptr,                 # [B, H, R]
    # kv caches (ragged)
    Pk_ptr, Pv_ptr,           # [T, Hk, R]
    # bases
    Vq_ptr, Vk_ptr,
    # biases
    bq_ptr, bk_ptr,
    # rope tables
    COS_ptr, SIN_ptr,         # [max_seqlen, HALF]
    # outputs
    M_ptr, L_ptr, Acc_ptr,    # [B, H, NSPLIT] and [B, H, NSPLIT, R]
    cu_seqlens_ptr,           # [B+1] int32
    # strides
    sPq_b, sPq_h, sPq_r,
    sPk_t, sPk_h, sPk_r,
    sPv_t, sPv_h, sPv_r,
    sVq_h, sVq_r, sVq_d,
    sVk_h, sVk_r, sVk_d,
    sbq_h, sbq_d,
    sbk_h, sbk_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sM_b, sM_h, sM_s,
    sL_b, sL_h, sL_s,
    sA_b, sA_h, sA_s, sA_r,
    # params
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_BQ: tl.constexpr,
    HAS_BK: tl.constexpr,
    # shapes
    max_seqlen: tl.constexpr,
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    # tiling
    BN: tl.constexpr,
    BR: tl.constexpr,
    SPLIT_K: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid0 = tl.program_id(0)
    split_id = tl.program_id(1)

    bid = pid0 // Hk
    hid_k = pid0 % Hk

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16
    offs_half = tl.arange(0, HALF)
    offs_r_full = tl.arange(0, R)

    start = tl.load(cu_seqlens_ptr + bid).to(tl.int32)
    end = tl.load(cu_seqlens_ptr + bid + 1).to(tl.int32)
    seqlen_k = end - start

    q_pos = tl.maximum(seqlen_k - 1, 0)
    split_start = split_id * SPLIT_K

    cos_q = tl.load(COS_ptr + q_pos * sCOS_s + offs_half * sCOS_d).to(in_dtype)
    sin_q = tl.load(SIN_ptr + q_pos * sSIN_s + offs_half * sSIN_d).to(in_dtype)

    # Build Q for REP heads
    q0 = tl.zeros((REP, HALF), dtype=tl.float32)
    q1 = tl.zeros((REP, HALF), dtype=tl.float32)

    hid_qs = hid_k * REP + tl.arange(0, REP)
    mask_hq = hid_qs < H

    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        mask_r = r < R

        Pq_blk = tl.load(
            Pq_q_ptr + bid * sPq_b + hid_qs[:, None] * sPq_h + r[None, :] * sPq_r,
            mask=mask_hq[:, None] & mask_r[None, :],
            other=0.0,
        ).to(in_dtype)

        Vq0 = tl.load(
            Vq_ptr + hid_qs[:, None, None] * sVq_h + r[None, :, None] * sVq_r + offs_half[None, None, :] * sVq_d,
            mask=mask_hq[:, None, None] & mask_r[None, :, None],
            other=0.0,
        ).to(in_dtype)
        Vq1 = tl.load(
            Vq_ptr + hid_qs[:, None, None] * sVq_h + r[None, :, None] * sVq_r + (offs_half + HALF)[None, None, :] * sVq_d,
            mask=mask_hq[:, None, None] & mask_r[None, :, None],
            other=0.0,
        ).to(in_dtype)

        q0 += tl.sum(Pq_blk[:, :, None].to(tl.float32) * Vq0.to(tl.float32), axis=1)
        q1 += tl.sum(Pq_blk[:, :, None].to(tl.float32) * Vq1.to(tl.float32), axis=1)

    if HAS_BQ:
        bq0 = tl.load(
            bq_ptr + hid_qs[:, None] * sbq_h + offs_half[None, :] * sbq_d,
            mask=mask_hq[:, None],
            other=0.0,
        ).to(tl.float32)
        bq1 = tl.load(
            bq_ptr + hid_qs[:, None] * sbq_h + (offs_half + HALF)[None, :] * sbq_d,
            mask=mask_hq[:, None],
            other=0.0,
        ).to(tl.float32)
        q0 += bq0
        q1 += bq1

    q0 = q0.to(in_dtype)
    q1 = q1.to(in_dtype)
    q0r = q0 * cos_q[None, :] - q1 * sin_q[None, :]
    q1r = q0 * sin_q[None, :] + q1 * cos_q[None, :]

    m_i = tl.full((REP,), -float("inf"), tl.float32)
    l_i = tl.zeros((REP,), tl.float32)
    acc_r = tl.zeros((REP, R), tl.float32)

    for nk_off in range(0, SPLIT_K, BN):
        nk = split_start + nk_off
        offs_n = nk + tl.arange(0, BN)
        valid_n = offs_n < seqlen_k

        if CAUSAL:
            valid_n = valid_n & (offs_n <= q_pos)
        if WINDOW_LEFT != -1:
            valid_n = valid_n & (offs_n >= (q_pos - WINDOW_LEFT))
        if WINDOW_RIGHT != -1:
            valid_n = valid_n & (offs_n <= (q_pos + WINDOW_RIGHT))

        cos_k = tl.load(
            COS_ptr + offs_n[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)
        sin_k = tl.load(
            SIN_ptr + offs_n[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)

        k0 = tl.zeros((BN, HALF), dtype=tl.float32)
        k1 = tl.zeros((BN, HALF), dtype=tl.float32)

        for r0 in range(0, R, BR):
            r = r0 + tl.arange(0, BR)
            mask_r = r < R

            Pk_blk = tl.load(
                Pk_ptr + (start + offs_n)[:, None] * sPk_t + hid_k * sPk_h + r[None, :] * sPk_r,
                mask=valid_n[:, None] & mask_r[None, :],
                other=0.0,
            ).to(in_dtype)

            Vk0 = tl.load(
                Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
                mask=mask_r[:, None],
                other=0.0,
            ).to(in_dtype)
            Vk1 = tl.load(
                Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
                mask=mask_r[:, None],
                other=0.0,
            ).to(in_dtype)

            k0 += tl.dot(Pk_blk, Vk0).to(tl.float32)
            k1 += tl.dot(Pk_blk, Vk1).to(tl.float32)

        if HAS_BK:
            bk0 = tl.load(bk_ptr + hid_k * sbk_h + offs_half * sbk_d).to(tl.float32)
            bk1 = tl.load(bk_ptr + hid_k * sbk_h + (offs_half + HALF) * sbk_d).to(tl.float32)
            k0 += bk0[None, :]
            k1 += bk1[None, :]

        k0 = k0.to(in_dtype)
        k1 = k1.to(in_dtype)
        k0r = k0 * cos_k - k1 * sin_k
        k1r = k0 * sin_k + k1 * cos_k

        # REP can be < 16 under GQA; tl.dot requires M >= 16 on newer Triton.
        scores = tl.sum(q0r.to(tl.float32)[:, None, :] * k0r.to(tl.float32)[None, :, :], axis=2)
        scores += tl.sum(q1r.to(tl.float32)[:, None, :] * k1r.to(tl.float32)[None, :, :], axis=2)
        scores *= SOFTMAX_SCALE

        scores = tl.where(valid_n[None, :], scores, -float("inf"))
        scores = tl.where(mask_hq[:, None], scores, -float("inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        is_neg_inf = m_new == -float("inf")
        alpha = tl.where(is_neg_inf, 1.0, tl.exp(m_i - m_new))
        p = tl.where(is_neg_inf[:, None], 0.0, tl.exp(scores - m_new[:, None]))

        l_i = l_i * alpha + tl.sum(p, axis=1)

        Pv_blk = tl.load(
            Pv_ptr + (start + offs_n)[:, None] * sPv_t + hid_k * sPv_h + offs_r_full[None, :] * sPv_r,
            mask=valid_n[:, None] & (offs_r_full[None, :] < R),
            other=0.0,
        ).to(in_dtype)

        acc_r = acc_r * alpha[:, None] + tl.sum(p[:, :, None] * Pv_blk.to(tl.float32)[None, :, :], axis=1)
        m_i = tl.where(is_neg_inf, m_i, m_new)

    # Write partial states (vectorized stores to avoid scalar indexing).
    tl.store(
        M_ptr + bid * sM_b + hid_qs * sM_h + split_id * sM_s,
        m_i,
        mask=mask_hq,
    )
    tl.store(
        L_ptr + bid * sL_b + hid_qs * sL_h + split_id * sL_s,
        l_i,
        mask=mask_hq,
    )
    tl.store(
        Acc_ptr + bid * sA_b + hid_qs[:, None] * sA_h + split_id * sA_s + offs_r_full[None, :] * sA_r,
        acc_r,
        mask=mask_hq[:, None] & (offs_r_full[None, :] < R),
    )


# ----------------------------
# Decode wrappers
# ----------------------------
@torch.no_grad()
def flashsvd_attn_decode_packed(
    f: DecodePackedFactors,
    rotary_cos: torch.Tensor,  # [Smax, Dh/2]
    rotary_sin: torch.Tensor,  # [Smax, Dh/2]
    *,
    seqlen_k: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[int, int] = (-1, -1),
    # split-K parameters
    split_k: int = 512,
    bn: int = 64,
    br: int = 64,
    num_warps_stage1: int = 4,
    num_stages_stage1: int = 2,
    num_warps_stage2: int = 4,
    num_stages_stage2: int = 1,
) -> torch.Tensor:
    """
    Decode (Q_len=1) attention with low-rank KV + RoPE using a Flash-Decoding style split-K pipeline.

    Returns: O [B, H, Dh] for the single query token in each sequence.
    """
    Pq_q = f.Pq
    Pk, Pv = f.Pk, f.Pv
    Vq, Vk, Vv = f.Vq, f.Vk, f.Vv
    bq, bk, bv = f.bq, f.bk, f.bv

    assert Pq_q.dim() == 3, f"Pq query must be [B,H,R], got {tuple(Pq_q.shape)}"
    B, H, R = Pq_q.shape
    assert Pk.dim() == 4 and Pv.dim() == 4
    B2, Smax, Hk, R2 = Pk.shape
    assert B2 == B and R2 == R
    assert Pv.shape == (B, Smax, Hk, R)
    assert H % Hk == 0
    rep = H // Hk

    br = min(br, R)
    assert split_k % bn == 0, "split_k must be a multiple of bn"
    num_splits = triton.cdiv(Smax, split_k)

    dtype = Pq_q.dtype
    assert dtype in (torch.float16, torch.bfloat16)
    is_bf16 = int(dtype == torch.bfloat16)

    Dh = Vq.shape[-1]
    assert Dh % 2 == 0
    half = Dh // 2
    assert Vq.shape == (H, R, Dh)
    assert Vk.shape == (Hk, R, Dh)
    assert Vv.shape == (Hk, R, Dh)
    assert rotary_cos.shape == (Smax, half) and rotary_sin.shape == (Smax, half)

    _assert_last_dim_contig(Pq_q, "Pq_q")
    _assert_last_dim_contig(Pk, "Pk")
    _assert_last_dim_contig(Pv, "Pv")
    _assert_last_dim_contig(Vq, "Vq")
    _assert_last_dim_contig(Vk, "Vk")
    _assert_last_dim_contig(Vv, "Vv")
    _assert_last_dim_contig(rotary_cos, "rotary_cos")
    _assert_last_dim_contig(rotary_sin, "rotary_sin")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dh)

    has_bq = int(bq is not None)
    has_bk = int(bk is not None)
    has_bv = int(bv is not None)
    if has_bq:
        assert bq.shape == (H, Dh)
        _assert_last_dim_contig(bq, "bq")
    if has_bk:
        assert bk.shape == (Hk, Dh)
        _assert_last_dim_contig(bk, "bk")
    if has_bv:
        assert bv.shape == (Hk, Dh)
        _assert_last_dim_contig(bv, "bv")

    # workspace: float32 states
    M = torch.empty((B, H, num_splits), device=Pq_q.device, dtype=torch.float32)
    L = torch.empty((B, H, num_splits), device=Pq_q.device, dtype=torch.float32)
    Acc = torch.empty((B, H, num_splits, R), device=Pq_q.device, dtype=torch.float32)

    # output: [B, H, Dh]
    O = torch.empty((B, H, Dh), device=Pq_q.device, dtype=dtype)

    # strides
    sPq_b, sPq_h, sPq_r = Pq_q.stride()
    sPk_b, sPk_s, sPk_h, sPk_r = Pk.stride()
    sPv_b, sPv_s, sPv_h, sPv_r = Pv.stride()
    sVq_h, sVq_r, sVq_d = Vq.stride()
    sVk_h, sVk_r, sVk_d = Vk.stride()
    sVv_h, sVv_r, sVv_d = Vv.stride()
    sCOS_s, sCOS_d = rotary_cos.stride()
    sSIN_s, sSIN_d = rotary_sin.stride()

    sM_b, sM_h, sM_s = M.stride()
    sL_b, sL_h, sL_s = L.stride()
    sA_b, sA_h, sA_s, sA_r = Acc.stride()
    sO_b, sO_h, sO_d = O.stride()

    if has_bq:
        sbq_h, sbq_d = bq.stride()
    else:
        sbq_h = sbq_d = 0
    if has_bk:
        sbk_h, sbk_d = bk.stride()
    else:
        sbk_h = sbk_d = 0
    if has_bv:
        sbv_h, sbv_d = bv.stride()
    else:
        sbv_h = sbv_d = 0

    grid1 = (B * Hk, num_splits)
    flashsvd_rope_decode_splitk_stage1_packed[grid1](
        Pq_q,
        Pk, Pv,
        Vq, Vk,
        bq if has_bq else O,
        bk if has_bk else O,
        rotary_cos, rotary_sin,
        M, L, Acc,
        seqlen_k,
        sPq_b=sPq_b, sPq_h=sPq_h, sPq_r=sPq_r,
        sPk_b=sPk_b, sPk_s=sPk_s, sPk_h=sPk_h, sPk_r=sPk_r,
        sPv_b=sPv_b, sPv_s=sPv_s, sPv_h=sPv_h, sPv_r=sPv_r,
        sVq_h=sVq_h, sVq_r=sVq_r, sVq_d=sVq_d,
        sVk_h=sVk_h, sVk_r=sVk_r, sVk_d=sVk_d,
        sbq_h=sbq_h, sbq_d=sbq_d,
        sbk_h=sbk_h, sbk_d=sbk_d,
        sCOS_s=sCOS_s, sCOS_d=sCOS_d,
        sSIN_s=sSIN_s, sSIN_d=sSIN_d,
        sM_b=sM_b, sM_h=sM_h, sM_s=sM_s,
        sL_b=sL_b, sL_h=sL_h, sL_s=sL_s,
        sA_b=sA_b, sA_h=sA_h, sA_s=sA_s, sA_r=sA_r,
        SOFTMAX_SCALE=softmax_scale,
        CAUSAL=int(causal),
        WINDOW_LEFT=window_size[0],
        WINDOW_RIGHT=window_size[1],
        HAS_BQ=has_bq,
        HAS_BK=has_bk,
        R=R,
        DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        BN=bn, BR=br, SPLIT_K=split_k,
        IS_BF16=is_bf16,
        num_warps=num_warps_stage1,
        num_stages=num_stages_stage1,
    )

    grid2 = (B * H,)
    flashsvd_rope_decode_splitk_reduce_packed[grid2](
        M, L, Acc,
        Vv,
        bv if has_bv else O,
        O,
        sM_b=sM_b, sM_h=sM_h, sM_s=sM_s,
        sL_b=sL_b, sL_h=sL_h, sL_s=sL_s,
        sA_b=sA_b, sA_h=sA_h, sA_s=sA_s, sA_r=sA_r,
        sVv_h=sVv_h, sVv_r=sVv_r, sVv_d=sVv_d,
        sbv_h=sbv_h, sbv_d=sbv_d,
        sO_b=sO_b, sO_h=sO_h, sO_d=sO_d,
        HAS_BV=has_bv,
        R=R, DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        NSPLIT=num_splits,
        IS_BF16=is_bf16,
        num_warps=num_warps_stage2,
        num_stages=num_stages_stage2,
    )
    return O


@torch.no_grad()
def flashsvd_attn_decode_varlen(
    f: DecodeVarlenFactors,
    cu_seqlens: torch.Tensor,   # [B+1] int32
    max_seqlen: int,
    rotary_cos: torch.Tensor,   # [max_seqlen, Dh/2]
    rotary_sin: torch.Tensor,   # [max_seqlen, Dh/2]
    *,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[int, int] = (-1, -1),
    split_k: int = 512,
    bn: int = 64,
    br: int = 64,
    num_warps_stage1: int = 4,
    num_stages_stage1: int = 2,
    num_warps_stage2: int = 4,
    num_stages_stage2: int = 1,
) -> torch.Tensor:
    """
    Varlen decode (Q_len=1 per sequence) using split-K low-rank KV kernel.

    Returns: O [B, H, Dh]
    """
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_cuda

    Pq_q = f.Pq
    Pk, Pv = f.Pk, f.Pv
    Vq, Vk, Vv = f.Vq, f.Vk, f.Vv
    bq, bk, bv = f.bq, f.bk, f.bv

    assert Pq_q.dim() == 3
    B, H, R = Pq_q.shape
    T, Hk, R2 = Pk.shape
    assert R2 == R and Pv.shape == (T, Hk, R)
    assert H % Hk == 0
    rep = H // Hk

    br = min(br, R)
    assert split_k % bn == 0
    num_splits = triton.cdiv(max_seqlen, split_k)

    dtype = Pq_q.dtype
    assert dtype in (torch.float16, torch.bfloat16)
    is_bf16 = int(dtype == torch.bfloat16)

    Dh = Vq.shape[-1]
    assert Dh % 2 == 0
    half = Dh // 2
    assert Vq.shape == (H, R, Dh)
    assert Vk.shape == (Hk, R, Dh)
    assert Vv.shape == (Hk, R, Dh)
    assert rotary_cos.shape == (max_seqlen, half) and rotary_sin.shape == (max_seqlen, half)

    _assert_last_dim_contig(Pq_q, "Pq_q")
    _assert_last_dim_contig(Pk, "Pk")
    _assert_last_dim_contig(Pv, "Pv")
    _assert_last_dim_contig(Vq, "Vq")
    _assert_last_dim_contig(Vk, "Vk")
    _assert_last_dim_contig(Vv, "Vv")
    _assert_last_dim_contig(rotary_cos, "rotary_cos")
    _assert_last_dim_contig(rotary_sin, "rotary_sin")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dh)

    has_bq = int(bq is not None)
    has_bk = int(bk is not None)
    has_bv = int(bv is not None)
    if has_bq:
        assert bq.shape == (H, Dh)
        _assert_last_dim_contig(bq, "bq")
    if has_bk:
        assert bk.shape == (Hk, Dh)
        _assert_last_dim_contig(bk, "bk")
    if has_bv:
        assert bv.shape == (Hk, Dh)
        _assert_last_dim_contig(bv, "bv")

    M = torch.empty((B, H, num_splits), device=Pq_q.device, dtype=torch.float32)
    L = torch.empty((B, H, num_splits), device=Pq_q.device, dtype=torch.float32)
    Acc = torch.empty((B, H, num_splits, R), device=Pq_q.device, dtype=torch.float32)
    O = torch.empty((B, H, Dh), device=Pq_q.device, dtype=dtype)

    sPq_b, sPq_h, sPq_r = Pq_q.stride()
    sPk_t, sPk_h, sPk_r = Pk.stride()
    sPv_t, sPv_h, sPv_r = Pv.stride()
    sVq_h, sVq_r, sVq_d = Vq.stride()
    sVk_h, sVk_r, sVk_d = Vk.stride()
    sVv_h, sVv_r, sVv_d = Vv.stride()
    sCOS_s, sCOS_d = rotary_cos.stride()
    sSIN_s, sSIN_d = rotary_sin.stride()
    sM_b, sM_h, sM_s = M.stride()
    sL_b, sL_h, sL_s = L.stride()
    sA_b, sA_h, sA_s, sA_r = Acc.stride()
    sO_b, sO_h, sO_d = O.stride()

    if has_bq:
        sbq_h, sbq_d = bq.stride()
    else:
        sbq_h = sbq_d = 0
    if has_bk:
        sbk_h, sbk_d = bk.stride()
    else:
        sbk_h = sbk_d = 0
    if has_bv:
        sbv_h, sbv_d = bv.stride()
    else:
        sbv_h = sbv_d = 0

    grid1 = (B * Hk, num_splits)
    flashsvd_rope_decode_splitk_stage1_varlen[grid1](
        Pq_q,
        Pk, Pv,
        Vq, Vk,
        bq if has_bq else O,
        bk if has_bk else O,
        rotary_cos, rotary_sin,
        M, L, Acc,
        cu_seqlens,
        sPq_b=sPq_b, sPq_h=sPq_h, sPq_r=sPq_r,
        sPk_t=sPk_t, sPk_h=sPk_h, sPk_r=sPk_r,
        sPv_t=sPv_t, sPv_h=sPv_h, sPv_r=sPv_r,
        sVq_h=sVq_h, sVq_r=sVq_r, sVq_d=sVq_d,
        sVk_h=sVk_h, sVk_r=sVk_r, sVk_d=sVk_d,
        sbq_h=sbq_h, sbq_d=sbq_d,
        sbk_h=sbk_h, sbk_d=sbk_d,
        sCOS_s=sCOS_s, sCOS_d=sCOS_d,
        sSIN_s=sSIN_s, sSIN_d=sSIN_d,
        sM_b=sM_b, sM_h=sM_h, sM_s=sM_s,
        sL_b=sL_b, sL_h=sL_h, sL_s=sL_s,
        sA_b=sA_b, sA_h=sA_h, sA_s=sA_s, sA_r=sA_r,
        SOFTMAX_SCALE=softmax_scale,
        CAUSAL=int(causal),
        WINDOW_LEFT=window_size[0],
        WINDOW_RIGHT=window_size[1],
        HAS_BQ=has_bq,
        HAS_BK=has_bk,
        max_seqlen=max_seqlen,
        R=R, DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        BN=bn, BR=br, SPLIT_K=split_k,
        IS_BF16=is_bf16,
        num_warps=num_warps_stage1,
        num_stages=num_stages_stage1,
    )

    grid2 = (B * H,)
    flashsvd_rope_decode_splitk_reduce_packed[grid2](
        M, L, Acc,
        Vv,
        bv if has_bv else O,
        O,
        sM_b=sM_b, sM_h=sM_h, sM_s=sM_s,
        sL_b=sL_b, sL_h=sL_h, sL_s=sL_s,
        sA_b=sA_b, sA_h=sA_h, sA_s=sA_s, sA_r=sA_r,
        sVv_h=sVv_h, sVv_r=sVv_r, sVv_d=sVv_d,
        sbv_h=sbv_h, sbv_d=sbv_d,
        sO_b=sO_b, sO_h=sO_h, sO_d=sO_d,
        HAS_BV=has_bv,
        R=R, DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        NSPLIT=num_splits,
        IS_BF16=is_bf16,
        num_warps=num_warps_stage2,
        num_stages=num_stages_stage2,
    )
    return O



# ----------------------------
# Wrapper helpers
# ----------------------------
def _assert_last_dim_contig(x: torch.Tensor, name: str):
    if x.stride(-1) != 1:
        raise ValueError(f"{name} last dim must be contiguous (stride(-1)==1). Got stride={x.stride()} shape={tuple(x.shape)}")


@torch.no_grad()
def flashsvd_attn_packed(
    f: PackedFactors,
    rotary_cos: torch.Tensor,  # [S, Dh/2]
    rotary_sin: torch.Tensor,  # [S, Dh/2]
    *,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[int, int] = (-1, -1),
    bm: int = 64,
    bn: int = 64,
    br: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
    value_in_rank: bool = False,
) -> torch.Tensor:
    Pq, Pk, Pv = f.Pq, f.Pk, f.Pv
    Vq, Vk, Vv = f.Vq, f.Vk, f.Vv
    bq, bk, bv = f.bq, f.bk, f.bv

    assert Pq.dim() == 4 and Pk.dim() == 4 and Pv.dim() == 4
    B, S, H, R = Pq.shape
    Hk = Pk.shape[2]
    assert Pk.shape == (B, S, Hk, R)
    assert Pv.shape == (B, S, Hk, R)
    assert H % Hk == 0
    rep = H // Hk

    # Avoid masked rank tiles when user passes a larger BR than rank.
    br = min(br, R)

    dtype = Pq.dtype
    assert dtype in (torch.float16, torch.bfloat16)
    is_bf16 = int(dtype == torch.bfloat16)

    Dh = Vq.shape[-1]
    assert Dh % 2 == 0
    half = Dh // 2

    assert Vq.shape == (H, R, Dh)
    assert Vk.shape == (Hk, R, Dh)
    assert Vv.shape == (Hk, R, Dh)
    assert rotary_cos.shape == (S, half) and rotary_sin.shape == (S, half)

    _assert_last_dim_contig(Pq, "Pq")
    _assert_last_dim_contig(Pk, "Pk")
    _assert_last_dim_contig(Pv, "Pv")
    _assert_last_dim_contig(Vq, "Vq")
    _assert_last_dim_contig(Vk, "Vk")
    _assert_last_dim_contig(Vv, "Vv")
    _assert_last_dim_contig(rotary_cos, "rotary_cos")
    _assert_last_dim_contig(rotary_sin, "rotary_sin")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dh)

    has_bq = int(bq is not None)
    has_bk = int(bk is not None)
    has_bv = int(bv is not None)
    if has_bq:
        assert bq.shape == (H, Dh)
        _assert_last_dim_contig(bq, "bq")
    if has_bk:
        assert bk.shape == (Hk, Dh)
        _assert_last_dim_contig(bk, "bk")
    if has_bv:
        assert bv.shape == (Hk, Dh)
        _assert_last_dim_contig(bv, "bv")

    O = torch.empty((B, S, H, Dh), device=Pq.device, dtype=dtype)

    sPq_b, sPq_s, sPq_h, sPq_r = Pq.stride()
    sPk_b, sPk_s, sPk_h, sPk_r = Pk.stride()
    sPv_b, sPv_s, sPv_h, sPv_r = Pv.stride()
    sVq_h, sVq_r, sVq_d = Vq.stride()
    sVk_h, sVk_r, sVk_d = Vk.stride()
    sVv_h, sVv_r, sVv_d = Vv.stride()
    sCOS_s, sCOS_d = rotary_cos.stride()
    sSIN_s, sSIN_d = rotary_sin.stride()
    sO_b, sO_s, sO_h, sO_d = O.stride()

    if has_bq:
        sbq_h, sbq_d = bq.stride()
    else:
        sbq_h = sbq_d = 0
    if has_bk:
        sbk_h, sbk_d = bk.stride()
    else:
        sbk_h = sbk_d = 0
    if has_bv:
        sbv_h, sbv_d = bv.stride()
    else:
        sbv_h = sbv_d = 0

    grid = (B * Hk, triton.cdiv(S, bm))
    kernel = flashsvd_rope_fwd_packed_R_value_in_rank if value_in_rank else flashsvd_rope_fwd_packed_R
    kernel[grid](
        Pq, Pk, Pv,
        Vq, Vk, Vv,
        bq if has_bq else O,
        bk if has_bk else O,
        bv if has_bv else O,
        rotary_cos, rotary_sin,
        O,
        S=S, R=R,
        sPq_b=sPq_b, sPq_s=sPq_s, sPq_h=sPq_h, sPq_r=sPq_r,
        sPk_b=sPk_b, sPk_s=sPk_s, sPk_h=sPk_h, sPk_r=sPk_r,
        sPv_b=sPv_b, sPv_s=sPv_s, sPv_h=sPv_h, sPv_r=sPv_r,
        sVq_h=sVq_h, sVq_r=sVq_r, sVq_d=sVq_d,
        sVk_h=sVk_h, sVk_r=sVk_r, sVk_d=sVk_d,
        sVv_h=sVv_h, sVv_r=sVv_r, sVv_d=sVv_d,
        sbq_h=sbq_h, sbq_d=sbq_d,
        sbk_h=sbk_h, sbk_d=sbk_d,
        sbv_h=sbv_h, sbv_d=sbv_d,
        sCOS_s=sCOS_s, sCOS_d=sCOS_d,
        sSIN_s=sSIN_s, sSIN_d=sSIN_d,
        sO_b=sO_b, sO_s=sO_s, sO_h=sO_h, sO_d=sO_d,
        SOFTMAX_SCALE=softmax_scale,
        CAUSAL=int(causal),
        WINDOW_LEFT=window_size[0],
        WINDOW_RIGHT=window_size[1],
        HAS_BQ=has_bq,
        HAS_BK=has_bk,
        HAS_BV=has_bv,
        BM=bm, BN=bn, BR=br,
        DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        IS_BF16=is_bf16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return O


@torch.no_grad()
def flashsvd_attn_varlen(
    f: VarlenFactors,
    cu_seqlens: torch.Tensor,   # [B+1] int32
    max_seqlen: int,
    rotary_cos: torch.Tensor,   # [max_seqlen, Dh/2]
    rotary_sin: torch.Tensor,   # [max_seqlen, Dh/2]
    *,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[int, int] = (-1, -1),
    bm: int = 64,
    bn: int = 64,
    br: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    Pq, Pk, Pv = f.Pq, f.Pk, f.Pv
    Vq, Vk, Vv = f.Vq, f.Vk, f.Vv
    bq, bk, bv = f.bq, f.bk, f.bv

    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_cuda
    T, H, R = Pq.shape
    Hk = Pk.shape[1]
    assert Pk.shape == (T, Hk, R) and Pv.shape == (T, Hk, R)
    assert H % Hk == 0
    rep = H // Hk

    # Avoid masked rank tiles when user passes a larger BR than rank.
    br = min(br, R)

    dtype = Pq.dtype
    assert dtype in (torch.float16, torch.bfloat16)
    is_bf16 = int(dtype == torch.bfloat16)

    Dh = Vq.shape[-1]
    assert Dh % 2 == 0
    half = Dh // 2
    assert Vq.shape == (H, R, Dh)
    assert Vk.shape == (Hk, R, Dh)
    assert Vv.shape == (Hk, R, Dh)
    assert rotary_cos.shape == (max_seqlen, half) and rotary_sin.shape == (max_seqlen, half)
    _assert_last_dim_contig(rotary_cos, "rotary_cos")
    _assert_last_dim_contig(rotary_sin, "rotary_sin")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dh)

    has_bq = int(bq is not None)
    has_bk = int(bk is not None)
    has_bv = int(bv is not None)
    if has_bq:
        assert bq.shape == (H, Dh)
        _assert_last_dim_contig(bq, "bq")
    if has_bk:
        assert bk.shape == (Hk, Dh)
        _assert_last_dim_contig(bk, "bk")
    if has_bv:
        assert bv.shape == (Hk, Dh)
        _assert_last_dim_contig(bv, "bv")

    O = torch.empty((T, H, Dh), device=Pq.device, dtype=dtype)

    sPq_t, sPq_h, sPq_r = Pq.stride()
    sPk_t, sPk_h, sPk_r = Pk.stride()
    sPv_t, sPv_h, sPv_r = Pv.stride()
    sVq_h, sVq_r, sVq_d = Vq.stride()
    sVk_h, sVk_r, sVk_d = Vk.stride()
    sVv_h, sVv_r, sVv_d = Vv.stride()
    sCOS_s, sCOS_d = rotary_cos.stride()
    sSIN_s, sSIN_d = rotary_sin.stride()
    sO_t, sO_h, sO_d = O.stride()

    if has_bq:
        sbq_h, sbq_d = bq.stride()
    else:
        sbq_h = sbq_d = 0
    if has_bk:
        sbk_h, sbk_d = bk.stride()
    else:
        sbk_h = sbk_d = 0
    if has_bv:
        sbv_h, sbv_d = bv.stride()
    else:
        sbv_h = sbv_d = 0

    B = cu_seqlens.numel() - 1
    grid = (B * Hk, triton.cdiv(max_seqlen, bm))

    flashsvd_rope_fwd_varlen_R[grid](
        Pq, Pk, Pv,
        Vq, Vk, Vv,
        bq if has_bq else O,
        bk if has_bk else O,
        bv if has_bv else O,
        rotary_cos, rotary_sin,
        O,
        cu_seqlens,
        max_seqlen=max_seqlen, R=R,
        sPq_t=sPq_t, sPq_h=sPq_h, sPq_r=sPq_r,
        sPk_t=sPk_t, sPk_h=sPk_h, sPk_r=sPk_r,
        sPv_t=sPv_t, sPv_h=sPv_h, sPv_r=sPv_r,
        sVq_h=sVq_h, sVq_r=sVq_r, sVq_d=sVq_d,
        sVk_h=sVk_h, sVk_r=sVk_r, sVk_d=sVk_d,
        sVv_h=sVv_h, sVv_r=sVv_r, sVv_d=sVv_d,
        sbq_h=sbq_h, sbq_d=sbq_d,
        sbk_h=sbk_h, sbk_d=sbk_d,
        sbv_h=sbv_h, sbv_d=sbv_d,
        sCOS_s=sCOS_s, sCOS_d=sCOS_d,
        sSIN_s=sSIN_s, sSIN_d=sSIN_d,
        sO_t=sO_t, sO_h=sO_h, sO_d=sO_d,
        SOFTMAX_SCALE=softmax_scale,
        CAUSAL=int(causal),
        WINDOW_LEFT=window_size[0],
        WINDOW_RIGHT=window_size[1],
        HAS_BQ=has_bq,
        HAS_BK=has_bk,
        HAS_BV=has_bv,
        BM=bm, BN=bn, BR=br,
        DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        IS_BF16=is_bf16,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return O


# ----------------------------
# Reference correctness (fp32) for packed
# ----------------------------
@torch.no_grad()
def rope_apply_bshd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B,S,H,Dh], cos/sin: [S, Dh/2]
    B,S,H,Dh = x.shape
    half = Dh // 2
    x0 = x[..., :half]
    x1 = x[..., half:]
    cos_ = cos[None, :, None, :]
    sin_ = sin[None, :, None, :]
    y0 = x0 * cos_ - x1 * sin_
    y1 = x0 * sin_ + x1 * cos_
    return torch.cat([y0, y1], dim=-1)


@torch.no_grad()
def reference_packed_fp32(
    f: PackedFactors,
    cos: torch.Tensor, sin: torch.Tensor,
    causal: bool,
    window_left: int,
    window_right: int,
) -> torch.Tensor:
    Pq,Pk,Pv = f.Pq.float(), f.Pk.float(), f.Pv.float()
    Vq,Vk,Vv = f.Vq.float(), f.Vk.float(), f.Vv.float()
    bq = f.bq.float() if f.bq is not None else None
    bk = f.bk.float() if f.bk is not None else None
    bv = f.bv.float() if f.bv is not None else None

    B,S,H,R = Pq.shape
    Hk = Pk.shape[2]
    Dh = Vq.shape[-1]
    rep = H // Hk
    scale = 1.0 / math.sqrt(Dh)

    Q = torch.einsum("bshr,hrd->bshd", Pq, Vq)
    K = torch.einsum("bskr,krd->bskd", Pk, Vk)
    V = torch.einsum("bskr,krd->bskd", Pv, Vv)

    if bq is not None: Q = Q + bq[None, None, :, :]
    if bk is not None: K = K + bk[None, None, :, :]
    if bv is not None: V = V + bv[None, None, :, :]

    Q = rope_apply_bshd(Q, cos.float(), sin.float())
    K = rope_apply_bshd(K, cos.float(), sin.float())

    K_full = K.repeat_interleave(rep, dim=2)
    V_full = V.repeat_interleave(rep, dim=2)

    scores = torch.einsum("bshd,bthd->bhst", Q, K_full) * scale  # [B,H,S,S]

    idx = torch.arange(S, device=scores.device)
    qpos = idx[:, None]
    kpos = idx[None, :]

    if causal:
        scores = scores.masked_fill(kpos > qpos, float("-inf"))

    if window_left != -1 or window_right != -1:
        left_ok  = True if window_left  == -1 else (kpos >= (qpos - window_left))
        right_ok = True if window_right == -1 else (kpos <= (qpos + window_right))
        scores = scores.masked_fill(~(left_ok & right_ok), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bthd->bshd", attn, V_full)
    return out


# ----------------------------
# Profiling helpers
# ----------------------------
def pretty_bytes(n: int) -> str:
    x = float(n)
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"

def do_bench_ms(fn, warmup=50, rep=200) -> float:
    try:
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    except Exception:
        torch.cuda.synchronize()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(rep):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / rep

def isolated_peak(fn, *a, **k):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = fn(*a, **k)
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

def estimate_flops_attention(B, H, S, Dh, causal=True):
    pairs = S * (S + 1) // 2 if causal else S * S
    return 4 * Dh * pairs * B * H  # QK(2Dh)+PV(2Dh)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("FlashSVD FA-aligned packed/varlen benchmark")
    ap.add_argument("--mode", type=str, default="packed", choices=["packed", "varlen"])
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--Hk", type=int, default=16)
    ap.add_argument("--Dh", type=int, default=128)
    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--br", type=int, default=64)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument(
        "--value_in_rank",
        action="store_true",
        help="accumulate values in rank-space (Pv) and lift once with Vv at the end (can reduce compute/register pressure)",
    )
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--window_left", type=int, default=-1)
    ap.add_argument("--window_right", type=int, default=-1)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--trace", type=str, default="trace_flashsvd.json")
    ap.add_argument("--check", action="store_true", help="run fp32 reference check (recommend small S/max_seqlen)")
    ap.add_argument("--stress", action="store_true", help="run stability stress (scale Q/K factors)")
    ap.add_argument("--stress_scales", type=str, default="1,3,10,30")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    B, S, H, Hk, Dh, R = args.B, args.S, args.H, args.Hk, args.Dh, args.R
    assert H % Hk == 0
    assert Dh % 2 == 0
    assert args.bm > 0 and args.bn > 0 and args.br > 0

    torch.manual_seed(0)

    if args.mode == "packed":
        cos, sin = build_rope_tables(S, Dh, base=10000.0, device=device, dtype=dtype)

        Pq = torch.randn(B, S, H,  R, device=device, dtype=dtype).contiguous()
        Pk = torch.randn(B, S, Hk, R, device=device, dtype=dtype).contiguous()
        Pv = torch.randn(B, S, Hk, R, device=device, dtype=dtype).contiguous()
        Vq = torch.randn(H,  R, Dh, device=device, dtype=dtype).contiguous()
        Vk = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
        Vv = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
        bq = torch.randn(H,  Dh, device=device, dtype=dtype).contiguous()
        bk = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()
        bv = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()

        f = PackedFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

        # warmup
        for _ in range(args.warmup):
            _ = flashsvd_attn_packed(
                f, cos, sin,
                causal=args.causal,
                window_size=(args.window_left, args.window_right),
                bm=args.bm, bn=args.bn, br=args.br,
                num_warps=args.warps, num_stages=args.stages,
                value_in_rank=args.value_in_rank,
            )
        torch.cuda.synchronize()

        # peak memory
        _, peak_alloc, peak_res = isolated_peak(
            flashsvd_attn_packed,
            f, cos, sin,
            causal=args.causal,
            window_size=(args.window_left, args.window_right),
            bm=args.bm, bn=args.bn, br=args.br,
            num_warps=args.warps, num_stages=args.stages,
            value_in_rank=args.value_in_rank,
        )

        # perf
        ms = do_bench_ms(
            lambda: flashsvd_attn_packed(
                f, cos, sin,
                causal=args.causal,
                window_size=(args.window_left, args.window_right),
                bm=args.bm, bn=args.bn, br=args.br,
                num_warps=args.warps, num_stages=args.stages,
                value_in_rank=args.value_in_rank,
            ),
            warmup=max(10, args.warmup // 2),
            rep=args.iters
        )

        flops = estimate_flops_attention(B, H, S, Dh, causal=args.causal)
        tflops = flops / (ms / 1e3) / 1e12
        tok_s = (B * S) / (ms / 1e3)

        print("==== FlashSVD (packed, FA-aligned) ====")
        print(f"Shape: B={B}, S={S}, H={H}, Hk={Hk}, Dh={Dh}, R={R}, dtype={dtype}")
        print(f"Mask: packed (no attention_mask tensor), causal={args.causal}, window=({args.window_left},{args.window_right})")
        print(
            f"tile: BM={args.bm}, BN={args.bn}, BR={args.br}, warps={args.warps}, stages={args.stages}, "
            f"value_in_rank={args.value_in_rank}"
        )
        print(f"latency: {ms:.4f} ms | tokens/s: {tok_s:,.0f} | eff TFLOPs(QK+PV): {tflops:.2f}")
        print(f"peak alloc: {pretty_bytes(peak_alloc)} | peak reserved: {pretty_bytes(peak_res)}")

        if args.profile:
            from torch.profiler import profile, ProfilerActivity
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                for _ in range(30):
                    _ = flashsvd_attn_packed(
                        f, cos, sin,
                        causal=args.causal,
                        window_size=(args.window_left, args.window_right),
                        bm=args.bm, bn=args.bn, br=args.br,
                        num_warps=args.warps, num_stages=args.stages,
                        value_in_rank=args.value_in_rank,
                    )
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
            prof.export_chrome_trace(args.trace)
            print(f"trace exported: {args.trace}")

        if args.check:
            if S > 256:
                print("[check] S too large for O(S^2) reference; re-run with --S <= 256")
            else:
                O = flashsvd_attn_packed(
                    f, cos, sin,
                    causal=args.causal,
                    window_size=(args.window_left, args.window_right),
                    bm=args.bm, bn=args.bn, br=args.br,
                    num_warps=args.warps, num_stages=args.stages,
                    value_in_rank=args.value_in_rank,
                )
                Oref = reference_packed_fp32(
                    f, cos, sin,
                    causal=args.causal,
                    window_left=args.window_left,
                    window_right=args.window_right
                ).to(torch.float32)
                diff = (O.float() - Oref)
                max_abs = diff.abs().max().item()
                rel_fro = (torch.linalg.norm(diff) / (torch.linalg.norm(Oref) + 1e-12)).item()
                finite = torch.isfinite(O).all().item()
                print(f"[check] finite={finite} max_abs={max_abs:.3e} rel_fro={rel_fro:.3e}")

        if args.stress:
            scales = [float(x) for x in args.stress_scales.split(",") if x.strip()]
            for sc in scales:
                f2 = PackedFactors(
                    Pq=f.Pq * sc, Pk=f.Pk * sc, Pv=f.Pv,
                    Vq=f.Vq, Vk=f.Vk, Vv=f.Vv,
                    bq=f.bq, bk=f.bk, bv=f.bv
                )
                O = flashsvd_attn_packed(
                    f2, cos, sin,
                    causal=args.causal,
                    window_size=(args.window_left, args.window_right),
                    bm=args.bm, bn=args.bn, br=args.br,
                    num_warps=args.warps, num_stages=args.stages
                )
                print(f"[stress] scale={sc:g} finite={torch.isfinite(O).all().item()} max|O|={O.abs().max().item():.3e}")

    else:
        # varlen: random lengths in [S/2, S]
        lens = torch.randint(low=max(1, S // 2), high=S + 1, size=(B,), device=device, dtype=torch.int32)
        cu = torch.zeros(B + 1, device=device, dtype=torch.int32)
        cu[1:] = torch.cumsum(lens, dim=0)
        T = int(cu[-1].item())
        max_seqlen = int(lens.max().item())

        cos, sin = build_rope_tables(max_seqlen, Dh, base=10000.0, device=device, dtype=dtype)

        Pq = torch.randn(T, H,  R, device=device, dtype=dtype).contiguous()
        Pk = torch.randn(T, Hk, R, device=device, dtype=dtype).contiguous()
        Pv = torch.randn(T, Hk, R, device=device, dtype=dtype).contiguous()
        Vq = torch.randn(H,  R, Dh, device=device, dtype=dtype).contiguous()
        Vk = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
        Vv = torch.randn(Hk, R, Dh, device=device, dtype=dtype).contiguous()
        bq = torch.randn(H,  Dh, device=device, dtype=dtype).contiguous()
        bk = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()
        bv = torch.randn(Hk, Dh, device=device, dtype=dtype).contiguous()

        f = VarlenFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

        for _ in range(args.warmup):
            _ = flashsvd_attn_varlen(
                f, cu, max_seqlen, cos, sin,
                causal=args.causal,
                window_size=(args.window_left, args.window_right),
                bm=args.bm, bn=args.bn, br=args.br,
                num_warps=args.warps, num_stages=args.stages
            )
        torch.cuda.synchronize()

        _, peak_alloc, peak_res = isolated_peak(
            flashsvd_attn_varlen,
            f, cu, max_seqlen, cos, sin,
            causal=args.causal,
            window_size=(args.window_left, args.window_right),
            bm=args.bm, bn=args.bn, br=args.br,
            num_warps=args.warps, num_stages=args.stages
        )

        ms = do_bench_ms(
            lambda: flashsvd_attn_varlen(
                f, cu, max_seqlen, cos, sin,
                causal=args.causal,
                window_size=(args.window_left, args.window_right),
                bm=args.bm, bn=args.bn, br=args.br,
                num_warps=args.warps, num_stages=args.stages
            ),
            warmup=max(10, args.warmup // 2),
            rep=args.iters
        )

        tok_s = T / (ms / 1e3)
        print("==== FlashSVD (varlen, FA-aligned) ====")
        print(f"B={B}, T={T}, max_seqlen={max_seqlen}, H={H}, Hk={Hk}, Dh={Dh}, R={R}, dtype={dtype}")
        print(f"Mask: varlen via cu_seqlens, causal={args.causal}, window=({args.window_left},{args.window_right})")
        print(f"tile: BM={args.bm}, BN={args.bn}, BR={args.br}, warps={args.warps}, stages={args.stages}")
        print(f"latency: {ms:.4f} ms | tokens/s: {tok_s:,.0f} (packed tokens)")
        print(f"peak alloc: {pretty_bytes(peak_alloc)} | peak reserved: {pretty_bytes(peak_res)}")

        if args.profile:
            from torch.profiler import profile, ProfilerActivity
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                for _ in range(30):
                    _ = flashsvd_attn_varlen(
                        f, cu, max_seqlen, cos, sin,
                        causal=args.causal,
                        window_size=(args.window_left, args.window_right),
                        bm=args.bm, bn=args.bn, br=args.br,
                        num_warps=args.warps, num_stages=args.stages
                    )
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
            prof.export_chrome_trace(args.trace)
            print(f"trace exported: {args.trace}")

        if args.check:
            if max_seqlen > 256 or B > 8:
                print("[check] varlen reference is O(L^2) per seq; re-run with smaller B/max_seqlen (<=256)")
            else:
                # compute kernel output once
                O = flashsvd_attn_varlen(
                    f, cu, max_seqlen, cos, sin,
                    causal=args.causal,
                    window_size=(args.window_left, args.window_right),
                    bm=args.bm, bn=args.bn, br=args.br,
                    num_warps=args.warps, num_stages=args.stages
                ).float()

                # per-sequence reference using packed fp32
                max_abs_all = 0.0
                rels: List[float] = []
                finite = torch.isfinite(O).all().item()

                for b in range(B):
                    s0 = int(cu[b].item())
                    s1 = int(cu[b+1].item())
                    L = s1 - s0
                    if L == 0:
                        continue

                    # slice & reshape to packed B=1
                    Pq_b = f.Pq[s0:s1].view(1, L, H, R)
                    Pk_b = f.Pk[s0:s1].view(1, L, Hk, R)
                    Pv_b = f.Pv[s0:s1].view(1, L, Hk, R)
                    cos_b = cos[:L]
                    sin_b = sin[:L]
                    ff = PackedFactors(
                        Pq=Pq_b.to(f.Pq.dtype),
                        Pk=Pk_b.to(f.Pk.dtype),
                        Pv=Pv_b.to(f.Pv.dtype),
                        Vq=f.Vq, Vk=f.Vk, Vv=f.Vv,
                        bq=f.bq, bk=f.bk, bv=f.bv
                    )
                    Oref = reference_packed_fp32(
                        ff, cos_b, sin_b,
                        causal=args.causal,
                        window_left=args.window_left,
                        window_right=args.window_right
                    ).squeeze(0).float()  # [L,H,Dh]

                    diff = O[s0:s1] - Oref
                    max_abs_all = max(max_abs_all, diff.abs().max().item())
                    rel = (torch.linalg.norm(diff) / (torch.linalg.norm(Oref) + 1e-12)).item()
                    rels.append(rel)

                print(f"[check] finite={finite} max_abs={max_abs_all:.3e} rel_fro_mean={sum(rels)/max(1,len(rels)):.3e}")

        if args.stress:
            scales = [float(x) for x in args.stress_scales.split(",") if x.strip()]
            for sc in scales:
                f2 = VarlenFactors(
                    Pq=f.Pq * sc, Pk=f.Pk * sc, Pv=f.Pv,
                    Vq=f.Vq, Vk=f.Vk, Vv=f.Vv,
                    bq=f.bq, bk=f.bk, bv=f.bv
                )
                O2 = flashsvd_attn_varlen(
                    f2, cu, max_seqlen, cos, sin,
                    causal=args.causal,
                    window_size=(args.window_left, args.window_right),
                    bm=args.bm, bn=args.bn, br=args.br,
                    num_warps=args.warps, num_stages=args.stages
                )
                print(f"[stress] scale={sc:g} finite={torch.isfinite(O2).all().item()} max|O|={O2.abs().max().item():.3e}")




# ============================================================================
# Decode v2 (decode-first): FlashInfer-aligned scheduling + low-rank-specific tricks
# ----------------------------------------------------------------------------
# Key upgrades vs v1:
#   1) num_splits computed from *seqlen_k* (not Smax) to avoid empty-split work and NaNs.
#   2) optional "writethrough" (NSPLIT==1) fused kernel: stage1 + reduce in one launch.
#   3) optional precompute of RoPE'd dense Q once per step (reused across splits).
#   4) pad head-group (REP) up to multiple-of-16 (GROUP_M) to unlock tensor-cores for
#      score GEMM and Pv accumulation GEMM (rep is often 4/8 under GQA).
#   5) Vk-tile "resident" inside each program (per split): load Vk0/Vk1 once and reuse
#      across all BN blocks in the split when BR==R.
#   6) robust all-masked guards in stage1/reduce to avoid exp(-inf - -inf) -> NaNs.
#
# We keep the original decode APIs as *_v1 for easy A/B benchmarking.
# ============================================================================

# ---- Preserve v1 wrappers for A/B benchmarking ----
flashsvd_attn_decode_packed_v1 = flashsvd_attn_decode_packed
flashsvd_attn_decode_varlen_v1 = flashsvd_attn_decode_varlen


@triton.jit
def flashsvd_rope_decode_build_q_packed_v2(
    Pq_q_ptr,                 # [B, H, R]
    Vq_ptr,                   # [H, R, Dh]
    bq_ptr,                   # [H, Dh] (optional)
    COS_ptr, SIN_ptr,         # [Smax, HALF]
    Q0_ptr, Q1_ptr,           # [B, H, HALF] (roped halves)
    seqlen_k,                 # int32
    # strides
    sPq_b, sPq_h, sPq_r,
    sVq_h, sVq_r, sVq_d,
    sbq_h, sbq_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sQ_b, sQ_h, sQ_d,
    # params
    HAS_BQ: tl.constexpr,
    # shapes
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    # tiling
    BR: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    Build RoPE'd dense Q halves (q0r/q1r) once per decode step.

    Grid: pid0 in [0, B*H)
      bid = pid0 // H
      hid_q = pid0 % H
    """
    pid0 = tl.program_id(0)
    bid = pid0 // H
    hid_q = pid0 % H

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    # q_pos = seqlen_k - 1 (clamp to 0)
    q_pos = tl.maximum(seqlen_k - 1, 0)

    offs_half = tl.arange(0, HALF)

    cos_q = tl.load(COS_ptr + q_pos * sCOS_s + offs_half * sCOS_d).to(in_dtype)
    sin_q = tl.load(SIN_ptr + q_pos * sSIN_s + offs_half * sSIN_d).to(in_dtype)

    q0 = tl.zeros((HALF,), dtype=tl.float32)
    q1 = tl.zeros((HALF,), dtype=tl.float32)

    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        mask_r = r < R

        Pq_blk = tl.load(
            Pq_q_ptr + bid * sPq_b + hid_q * sPq_h + r * sPq_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)

        Vq0 = tl.load(
            Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + offs_half[None, :] * sVq_d,
            mask=mask_r[:, None],
            other=0.0,
        ).to(in_dtype)
        Vq1 = tl.load(
            Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + (offs_half + HALF)[None, :] * sVq_d,
            mask=mask_r[:, None],
            other=0.0,
        ).to(in_dtype)

        # (BR) @ (BR, HALF) -> (HALF)
        q0 += tl.sum(Pq_blk[:, None].to(tl.float32) * Vq0.to(tl.float32), axis=0)
        q1 += tl.sum(Pq_blk[:, None].to(tl.float32) * Vq1.to(tl.float32), axis=0)

    if HAS_BQ:
        bq0 = tl.load(bq_ptr + hid_q * sbq_h + offs_half * sbq_d).to(tl.float32)
        bq1 = tl.load(bq_ptr + hid_q * sbq_h + (offs_half + HALF) * sbq_d).to(tl.float32)
        q0 += bq0
        q1 += bq1

    # RoPE
    q0t = q0.to(in_dtype)
    q1t = q1.to(in_dtype)
    q0r = q0t * cos_q - q1t * sin_q
    q1r = q0t * sin_q + q1t * cos_q

    tl.store(Q0_ptr + bid * sQ_b + hid_q * sQ_h + offs_half * sQ_d, q0r, mask=True)
    tl.store(Q1_ptr + bid * sQ_b + hid_q * sQ_h + offs_half * sQ_d, q1r, mask=True)


@triton.jit
def flashsvd_rope_decode_build_q_varlen_v2(
    Pq_q_ptr,                 # [B, H, R]
    Vq_ptr,                   # [H, R, Dh]
    bq_ptr,                   # [H, Dh] (optional)
    COS_ptr, SIN_ptr,         # [max_seqlen, HALF]
    Q0_ptr, Q1_ptr,           # [B, H, HALF]
    cu_seqlens_ptr,           # [B+1] int32
    # strides
    sPq_b, sPq_h, sPq_r,
    sVq_h, sVq_r, sVq_d,
    sbq_h, sbq_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sQ_b, sQ_h, sQ_d,
    # params
    HAS_BQ: tl.constexpr,
    # shapes
    max_seqlen: tl.constexpr,
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    # tiling
    BR: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    Varlen version of Q build: q_pos depends on each sequence length from cu_seqlens.
    Grid: pid0 in [0, B*H)
    """
    pid0 = tl.program_id(0)
    bid = pid0 // H
    hid_q = pid0 % H

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    start = tl.load(cu_seqlens_ptr + bid).to(tl.int32)
    end = tl.load(cu_seqlens_ptr + bid + 1).to(tl.int32)
    seqlen_k = end - start
    q_pos = tl.maximum(seqlen_k - 1, 0)

    offs_half = tl.arange(0, HALF)
    cos_q = tl.load(COS_ptr + q_pos * sCOS_s + offs_half * sCOS_d).to(in_dtype)
    sin_q = tl.load(SIN_ptr + q_pos * sSIN_s + offs_half * sSIN_d).to(in_dtype)

    q0 = tl.zeros((HALF,), dtype=tl.float32)
    q1 = tl.zeros((HALF,), dtype=tl.float32)

    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        mask_r = r < R

        Pq_blk = tl.load(
            Pq_q_ptr + bid * sPq_b + hid_q * sPq_h + r * sPq_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)

        Vq0 = tl.load(
            Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + offs_half[None, :] * sVq_d,
            mask=mask_r[:, None],
            other=0.0,
        ).to(in_dtype)
        Vq1 = tl.load(
            Vq_ptr + hid_q * sVq_h + r[:, None] * sVq_r + (offs_half + HALF)[None, :] * sVq_d,
            mask=mask_r[:, None],
            other=0.0,
        ).to(in_dtype)

        q0 += tl.sum(Pq_blk[:, None].to(tl.float32) * Vq0.to(tl.float32), axis=0)
        q1 += tl.sum(Pq_blk[:, None].to(tl.float32) * Vq1.to(tl.float32), axis=0)

    if HAS_BQ:
        bq0 = tl.load(bq_ptr + hid_q * sbq_h + offs_half * sbq_d).to(tl.float32)
        bq1 = tl.load(bq_ptr + hid_q * sbq_h + (offs_half + HALF) * sbq_d).to(tl.float32)
        q0 += bq0
        q1 += bq1

    q0t = q0.to(in_dtype)
    q1t = q1.to(in_dtype)
    q0r = q0t * cos_q - q1t * sin_q
    q1r = q0t * sin_q + q1t * cos_q

    tl.store(Q0_ptr + bid * sQ_b + hid_q * sQ_h + offs_half * sQ_d, q0r, mask=True)
    tl.store(Q1_ptr + bid * sQ_b + hid_q * sQ_h + offs_half * sQ_d, q1r, mask=True)


@triton.jit
def flashsvd_rope_decode_splitk_stage1_packed_v2(
    # precomputed roped Q halves
    Q0_ptr, Q1_ptr,           # [B, H, HALF]
    # kv caches
    Pk_ptr, Pv_ptr,           # [B, Smax, Hk, R] (or [B, Hk, Smax, R] if strides set accordingly)
    # bases
    Vk_ptr,                   # [Hk, R, Dh]
    # bias
    bk_ptr,
    # rope tables
    COS_ptr, SIN_ptr,         # [Smax, HALF]
    # outputs
    M_ptr, L_ptr, Acc_ptr,    # [B, H, NSPLIT] and [B, H, NSPLIT, R]
    # runtime len
    seqlen_k,                 # int32
    # strides
    sQ_b, sQ_h, sQ_d,
    sPk_b, sPk_s, sPk_h, sPk_r,
    sPv_b, sPv_s, sPv_h, sPv_r,
    sVk_h, sVk_r, sVk_d,
    sbk_h, sbk_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sM_b, sM_h, sM_s,
    sL_b, sL_h, sL_s,
    sA_b, sA_h, sA_s, sA_r,
    # params
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_BK: tl.constexpr,
    # shapes
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    # tiling
    BN: tl.constexpr,
    BR: tl.constexpr,          # NOTE: for VK_RESIDENT, we require BR == R
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,     # padded head-group, multiple of 16
    VK_RESIDENT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    Stage1 (per split): compute partial softmax states in rank space for decode (Q_len=1).

    Improvements:
      - consumes precomputed RoPE'd Q halves (Q0/Q1)
      - pads REP -> GROUP_M to unlock tensor cores on score / Pv GEMMs
      - optional Vk residency (load Vk once per program when BR==R)
      - robust all-masked guard
    """
    pid0 = tl.program_id(0)
    split_id = tl.program_id(1)

    bid = pid0 // Hk
    hid_k = pid0 % Hk

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    q_pos = tl.maximum(seqlen_k - 1, 0)
    split_start = split_id * SPLIT_K

    offs_half = tl.arange(0, HALF)
    offs_r = tl.arange(0, R)
    g = tl.arange(0, GROUP_M)  # padded head-group lanes

    # Heads belonging to this kv head: g < REP
    mask_g = g < REP
    hid_qs = hid_k * REP + g  # (GROUP_M,)
    # We must not spill into next kv-head group; mask_g enforces that.

    # Load precomputed roped Q halves
    q0r = tl.load(
        Q0_ptr + bid * sQ_b + hid_qs[:, None] * sQ_h + offs_half[None, :] * sQ_d,
        mask=mask_g[:, None],
        other=0.0,
    ).to(in_dtype)
    q1r = tl.load(
        Q1_ptr + bid * sQ_b + hid_qs[:, None] * sQ_h + offs_half[None, :] * sQ_d,
        mask=mask_g[:, None],
        other=0.0,
    ).to(in_dtype)

    # Optionally keep Vk tiles resident across BN blocks in this split.
    # Only supported efficiently when BR == R (single rank tile).
    if VK_RESIDENT:
        Vk0_full = tl.load(
            Vk_ptr + hid_k * sVk_h + offs_r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
            mask=offs_r[:, None] < R,
            other=0.0,
        ).to(in_dtype)
        Vk1_full = tl.load(
            Vk_ptr + hid_k * sVk_h + offs_r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
            mask=offs_r[:, None] < R,
            other=0.0,
        ).to(in_dtype)

    # Online softmax partial state (padded heads)
    m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((GROUP_M,), tl.float32)
    acc_r = tl.zeros((GROUP_M, R), tl.float32)

    for nk_off in range(0, SPLIT_K, BN):
        nk = split_start + nk_off
        offs_n = nk + tl.arange(0, BN)
        valid_n = offs_n < seqlen_k

        if CAUSAL:
            valid_n = valid_n & (offs_n <= q_pos)
        if WINDOW_LEFT != -1:
            valid_n = valid_n & (offs_n >= (q_pos - WINDOW_LEFT))
        if WINDOW_RIGHT != -1:
            valid_n = valid_n & (offs_n <= (q_pos + WINDOW_RIGHT))

        # RoPE tables for keys
        cos_k = tl.load(
            COS_ptr + offs_n[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)
        sin_k = tl.load(
            SIN_ptr + offs_n[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)

        # Decompress K for this kv head
        if VK_RESIDENT:
            # BR must equal R here (single-tile rank)
            Pk_blk = tl.load(
                Pk_ptr + bid * sPk_b + offs_n[:, None] * sPk_s + hid_k * sPk_h + offs_r[None, :] * sPk_r,
                mask=valid_n[:, None] & (offs_r[None, :] < R),
                other=0.0,
            ).to(in_dtype)

            k0 = tl.dot(Pk_blk, Vk0_full).to(tl.float32)
            k1 = tl.dot(Pk_blk, Vk1_full).to(tl.float32)
        else:
            k0 = tl.zeros((BN, HALF), dtype=tl.float32)
            k1 = tl.zeros((BN, HALF), dtype=tl.float32)
            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R
                Pk_blk = tl.load(
                    Pk_ptr + bid * sPk_b + offs_n[:, None] * sPk_s + hid_k * sPk_h + r[None, :] * sPk_r,
                    mask=valid_n[:, None] & mask_r[None, :],
                    other=0.0,
                ).to(in_dtype)
                Vk0 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
                    mask=mask_r[:, None],
                    other=0.0,
                ).to(in_dtype)
                Vk1 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
                    mask=mask_r[:, None],
                    other=0.0,
                ).to(in_dtype)
                k0 += tl.dot(Pk_blk, Vk0).to(tl.float32)
                k1 += tl.dot(Pk_blk, Vk1).to(tl.float32)

        if HAS_BK:
            bk0 = tl.load(bk_ptr + hid_k * sbk_h + offs_half * sbk_d).to(tl.float32)
            bk1 = tl.load(bk_ptr + hid_k * sbk_h + (offs_half + HALF) * sbk_d).to(tl.float32)
            k0 += bk0[None, :]
            k1 += bk1[None, :]

        k0t = k0.to(in_dtype)
        k1t = k1.to(in_dtype)

        # Apply RoPE to K
        k0r = k0t * cos_k - k1t * sin_k
        k1r = k0t * sin_k + k1t * cos_k

        # Scores: (GROUP_M, HALF) x (HALF, BN) -> (GROUP_M, BN)
        # NOTE: tl.dot may require M/N/K >= 16 on some Triton versions for fp16/bf16.
        # Provide a safe fallback for small GROUP_M (e.g., REP=1 when Hk==H).
        if GROUP_M < 16:
            scores = tl.sum(q0r.to(tl.float32)[:, None, :] * k0r.to(tl.float32)[None, :, :], axis=2)
            scores += tl.sum(q1r.to(tl.float32)[:, None, :] * k1r.to(tl.float32)[None, :, :], axis=2)
        else:
            scores = tl.dot(q0r, tl.trans(k0r)).to(tl.float32)
            scores += tl.dot(q1r, tl.trans(k1r)).to(tl.float32)
        scores *= SOFTMAX_SCALE

        scores = tl.where(valid_n[None, :], scores, -float("inf"))
        scores = tl.where(mask_g[:, None], scores, -float("inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)

        # all-masked guard: m_new == -inf => keep state, p=0
        is_neg_inf = m_new == -float("inf")
        alpha = tl.where(is_neg_inf, 1.0, tl.exp(m_i - m_new))
        p = tl.where(is_neg_inf[:, None], 0.0, tl.exp(scores - m_new[:, None]))

        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Accumulate value in rank-space: (GROUP_M, BN) x (BN, R) -> (GROUP_M, R)
        Pv_blk = tl.load(
            Pv_ptr + bid * sPv_b + offs_n[:, None] * sPv_s + hid_k * sPv_h + offs_r[None, :] * sPv_r,
            mask=valid_n[:, None] & (offs_r[None, :] < R),
            other=0.0,
        ).to(in_dtype)

        if GROUP_M < 16:
            acc_r = acc_r * alpha[:, None] + tl.sum(
                p.to(tl.float32)[:, :, None] * Pv_blk.to(tl.float32)[None, :, :],
                axis=1,
            )
        else:
            acc_r = acc_r * alpha[:, None] + tl.dot(p.to(in_dtype), Pv_blk).to(tl.float32)
        m_i = tl.where(is_neg_inf, m_i, m_new)

    # Store partials for valid heads only (g < REP)
    tl.store(
        M_ptr + bid * sM_b + hid_qs * sM_h + split_id * sM_s,
        m_i,
        mask=mask_g,
    )
    tl.store(
        L_ptr + bid * sL_b + hid_qs * sL_h + split_id * sL_s,
        l_i,
        mask=mask_g,
    )
    tl.store(
        Acc_ptr + bid * sA_b + hid_qs[:, None] * sA_h + split_id * sA_s + offs_r[None, :] * sA_r,
        acc_r,
        mask=mask_g[:, None] & (offs_r[None, :] < R),
    )


@triton.jit
def flashsvd_rope_decode_splitk_stage1_varlen_v2(
    # precomputed roped Q halves
    Q0_ptr, Q1_ptr,           # [B, H, HALF]
    # kv caches (ragged)
    Pk_ptr, Pv_ptr,           # [T, Hk, R]
    # bases
    Vk_ptr,                   # [Hk, R, Dh]
    # bias
    bk_ptr,
    # rope tables
    COS_ptr, SIN_ptr,         # [max_seqlen, HALF]
    # outputs
    M_ptr, L_ptr, Acc_ptr,    # [B, H, NSPLIT] and [B, H, NSPLIT, R]
    cu_seqlens_ptr,           # [B+1] int32
    # strides
    sQ_b, sQ_h, sQ_d,
    sPk_t, sPk_h, sPk_r,
    sPv_t, sPv_h, sPv_r,
    sVk_h, sVk_r, sVk_d,
    sbk_h, sbk_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sM_b, sM_h, sM_s,
    sL_b, sL_h, sL_s,
    sA_b, sA_h, sA_s, sA_r,
    # params
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_BK: tl.constexpr,
    # shapes
    max_seqlen: tl.constexpr,
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    # tiling
    BN: tl.constexpr,
    BR: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    VK_RESIDENT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid0 = tl.program_id(0)
    split_id = tl.program_id(1)

    bid = pid0 // Hk
    hid_k = pid0 % Hk

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16

    start = tl.load(cu_seqlens_ptr + bid).to(tl.int32)
    end = tl.load(cu_seqlens_ptr + bid + 1).to(tl.int32)
    seqlen_k = end - start
    q_pos = tl.maximum(seqlen_k - 1, 0)

    split_start = split_id * SPLIT_K

    offs_half = tl.arange(0, HALF)
    offs_r = tl.arange(0, R)
    g = tl.arange(0, GROUP_M)
    mask_g = g < REP
    hid_qs = hid_k * REP + g

    # Load precomputed Q halves
    q0r = tl.load(
        Q0_ptr + bid * sQ_b + hid_qs[:, None] * sQ_h + offs_half[None, :] * sQ_d,
        mask=mask_g[:, None],
        other=0.0,
    ).to(in_dtype)
    q1r = tl.load(
        Q1_ptr + bid * sQ_b + hid_qs[:, None] * sQ_h + offs_half[None, :] * sQ_d,
        mask=mask_g[:, None],
        other=0.0,
    ).to(in_dtype)

    if VK_RESIDENT:
        Vk0_full = tl.load(
            Vk_ptr + hid_k * sVk_h + offs_r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
            mask=offs_r[:, None] < R,
            other=0.0,
        ).to(in_dtype)
        Vk1_full = tl.load(
            Vk_ptr + hid_k * sVk_h + offs_r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
            mask=offs_r[:, None] < R,
            other=0.0,
        ).to(in_dtype)

    m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((GROUP_M,), tl.float32)
    acc_r = tl.zeros((GROUP_M, R), tl.float32)

    for nk_off in range(0, SPLIT_K, BN):
        nk = split_start + nk_off
        offs_n = nk + tl.arange(0, BN)
        valid_n = offs_n < seqlen_k

        if CAUSAL:
            valid_n = valid_n & (offs_n <= q_pos)
        if WINDOW_LEFT != -1:
            valid_n = valid_n & (offs_n >= (q_pos - WINDOW_LEFT))
        if WINDOW_RIGHT != -1:
            valid_n = valid_n & (offs_n <= (q_pos + WINDOW_RIGHT))

        cos_k = tl.load(
            COS_ptr + offs_n[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)
        sin_k = tl.load(
            SIN_ptr + offs_n[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)

        if VK_RESIDENT:
            Pk_blk = tl.load(
                Pk_ptr + (start + offs_n)[:, None] * sPk_t + hid_k * sPk_h + offs_r[None, :] * sPk_r,
                mask=valid_n[:, None] & (offs_r[None, :] < R),
                other=0.0,
            ).to(in_dtype)
            k0 = tl.dot(Pk_blk, Vk0_full).to(tl.float32)
            k1 = tl.dot(Pk_blk, Vk1_full).to(tl.float32)
        else:
            k0 = tl.zeros((BN, HALF), dtype=tl.float32)
            k1 = tl.zeros((BN, HALF), dtype=tl.float32)
            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R
                Pk_blk = tl.load(
                    Pk_ptr + (start + offs_n)[:, None] * sPk_t + hid_k * sPk_h + r[None, :] * sPk_r,
                    mask=valid_n[:, None] & mask_r[None, :],
                    other=0.0,
                ).to(in_dtype)
                Vk0 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
                    mask=mask_r[:, None],
                    other=0.0,
                ).to(in_dtype)
                Vk1 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
                    mask=mask_r[:, None],
                    other=0.0,
                ).to(in_dtype)
                k0 += tl.dot(Pk_blk, Vk0).to(tl.float32)
                k1 += tl.dot(Pk_blk, Vk1).to(tl.float32)

        if HAS_BK:
            bk0 = tl.load(bk_ptr + hid_k * sbk_h + offs_half * sbk_d).to(tl.float32)
            bk1 = tl.load(bk_ptr + hid_k * sbk_h + (offs_half + HALF) * sbk_d).to(tl.float32)
            k0 += bk0[None, :]
            k1 += bk1[None, :]

        k0t = k0.to(in_dtype)
        k1t = k1.to(in_dtype)

        k0r = k0t * cos_k - k1t * sin_k
        k1r = k0t * sin_k + k1t * cos_k

        scores = tl.dot(q0r, tl.trans(k0r)).to(tl.float32)
        scores += tl.dot(q1r, tl.trans(k1r)).to(tl.float32)
        scores *= SOFTMAX_SCALE

        scores = tl.where(valid_n[None, :], scores, -float("inf"))
        scores = tl.where(mask_g[:, None], scores, -float("inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)

        is_neg_inf = m_new == -float("inf")
        alpha = tl.where(is_neg_inf, 1.0, tl.exp(m_i - m_new))
        p = tl.where(is_neg_inf[:, None], 0.0, tl.exp(scores - m_new[:, None]))

        l_i = l_i * alpha + tl.sum(p, axis=1)

        Pv_blk = tl.load(
            Pv_ptr + (start + offs_n)[:, None] * sPv_t + hid_k * sPv_h + offs_r[None, :] * sPv_r,
            mask=valid_n[:, None] & (offs_r[None, :] < R),
            other=0.0,
        ).to(in_dtype)

        acc_r = acc_r * alpha[:, None] + tl.dot(p.to(in_dtype), Pv_blk).to(tl.float32)
        m_i = tl.where(is_neg_inf, m_i, m_new)

    tl.store(
        M_ptr + bid * sM_b + hid_qs * sM_h + split_id * sM_s,
        m_i,
        mask=mask_g,
    )
    tl.store(
        L_ptr + bid * sL_b + hid_qs * sL_h + split_id * sL_s,
        l_i,
        mask=mask_g,
    )
    tl.store(
        Acc_ptr + bid * sA_b + hid_qs[:, None] * sA_h + split_id * sA_s + offs_r[None, :] * sA_r,
        acc_r,
        mask=mask_g[:, None] & (offs_r[None, :] < R),
    )


@triton.jit
def flashsvd_rope_decode_splitk_reduce_packed_v2(
    # partial states
    M_ptr, L_ptr, Acc_ptr,     # [B,H,NSPLIT] and [B,H,NSPLIT,R]
    # lift
    Vv_ptr, bv_ptr,
    # output
    O_ptr,                     # [B,H,Dh]
    # strides
    sM_b, sM_h, sM_s,
    sL_b, sL_h, sL_s,
    sA_b, sA_h, sA_s, sA_r,
    sVv_h, sVv_r, sVv_d,
    sbv_h, sbv_d,
    sO_b, sO_h, sO_d,
    # params
    HAS_BV: tl.constexpr,
    # shapes
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    # split
    NSPLIT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    Robust split reduction (guards all-masked cases).
    """
    pid0 = tl.program_id(0)
    bid = pid0 // H
    hid_q = pid0 % H
    hid_k = hid_q // REP

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16
    offs_half = tl.arange(0, HALF)
    offs_r = tl.arange(0, R)

    m = -float("inf")
    l = 0.0
    acc = tl.zeros((R,), dtype=tl.float32)

    for s in range(0, NSPLIT):
        m_s = tl.load(M_ptr + bid * sM_b + hid_q * sM_h + s * sM_s)
        l_s = tl.load(L_ptr + bid * sL_b + hid_q * sL_h + s * sL_s)
        a_s = tl.load(
            Acc_ptr + bid * sA_b + hid_q * sA_h + s * sA_s + offs_r * sA_r,
            mask=offs_r < R,
            other=0.0,
        ).to(tl.float32)

        m_new = tl.maximum(m, m_s)
        is_neg_inf = m_new == -float("inf")
        alpha = tl.where(is_neg_inf, 1.0, tl.exp(m - m_new))
        beta = tl.where(is_neg_inf, 0.0, tl.exp(m_s - m_new))
        l = l * alpha + l_s * beta
        acc = acc * alpha + a_s * beta
        m = tl.where(is_neg_inf, m, m_new)

    den = tl.where(l > 0, l, 1.0)
    w_r = acc / den
    w_r = tl.where(l > 0, w_r, 0.0)

    Vv0 = tl.load(
        Vv_ptr + hid_k * sVv_h + offs_r[:, None] * sVv_r + offs_half[None, :] * sVv_d,
        mask=offs_r[:, None] < R,
        other=0.0,
    ).to(in_dtype)
    Vv1 = tl.load(
        Vv_ptr + hid_k * sVv_h + offs_r[:, None] * sVv_r + (offs_half + HALF)[None, :] * sVv_d,
        mask=offs_r[:, None] < R,
        other=0.0,
    ).to(in_dtype)

    # (1,R) x (R,HALF) -- small M; avoid tl.dot shape constraints.
    out0 = tl.sum(w_r[:, None] * Vv0.to(tl.float32), axis=0)
    out1 = tl.sum(w_r[:, None] * Vv1.to(tl.float32), axis=0)

    if HAS_BV:
        bv0 = tl.load(bv_ptr + hid_k * sbv_h + offs_half * sbv_d).to(tl.float32)
        bv1 = tl.load(bv_ptr + hid_k * sbv_h + (offs_half + HALF) * sbv_d).to(tl.float32)
        out0 = out0 + bv0
        out1 = out1 + bv1

    tl.store(
        O_ptr + bid * sO_b + hid_q * sO_h + offs_half * sO_d,
        out0.to(in_dtype),
        mask=True,
    )
    tl.store(
        O_ptr + bid * sO_b + hid_q * sO_h + (offs_half + HALF) * sO_d,
        out1.to(in_dtype),
        mask=True,
    )


@triton.jit
def flashsvd_rope_decode_writethrough_packed_v2(
    # query (low-rank)
    Pq_q_ptr,                 # [B, H, R]
    # kv caches
    Pk_ptr, Pv_ptr,           # [B, Smax, Hk, R]
    # bases
    Vq_ptr, Vk_ptr, Vv_ptr,   # [H,R,Dh], [Hk,R,Dh], [Hk,R,Dh]
    # biases
    bq_ptr, bk_ptr, bv_ptr,
    # rope
    COS_ptr, SIN_ptr,         # [Smax, HALF]
    # output
    O_ptr,                    # [B, H, Dh]
    # runtime len
    seqlen_k,                 # int32 (<= SPLIT_K)
    # strides
    sPq_b, sPq_h, sPq_r,
    sPk_b, sPk_s, sPk_h, sPk_r,
    sPv_b, sPv_s, sPv_h, sPv_r,
    sVq_h, sVq_r, sVq_d,
    sVk_h, sVk_r, sVk_d,
    sVv_h, sVv_r, sVv_d,
    sbq_h, sbq_d,
    sbk_h, sbk_d,
    sbv_h, sbv_d,
    sCOS_s, sCOS_d,
    sSIN_s, sSIN_d,
    sO_b, sO_h, sO_d,
    # params
    SOFTMAX_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    HAS_BQ: tl.constexpr,
    HAS_BK: tl.constexpr,
    HAS_BV: tl.constexpr,
    # shapes
    R: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    H: tl.constexpr,
    Hk: tl.constexpr,
    REP: tl.constexpr,
    # tiling
    BN: tl.constexpr,
    BR: tl.constexpr,          # recommended BR == R
    SPLIT_K: tl.constexpr,     # upper bound for seqlen_k in writethrough
    GROUP_M: tl.constexpr,
    VK_RESIDENT: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """
    Writethrough decode (NSPLIT==1): compute final output in a single kernel.
    """
    pid0 = tl.program_id(0)
    bid = pid0 // Hk
    hid_k = pid0 % Hk

    in_dtype = tl.bfloat16 if IS_BF16 else tl.float16
    q_pos = tl.maximum(seqlen_k - 1, 0)

    offs_half = tl.arange(0, HALF)
    offs_r = tl.arange(0, R)
    g = tl.arange(0, GROUP_M)
    mask_g = g < REP
    hid_qs = hid_k * REP + g

    # RoPE for query position
    cos_q = tl.load(COS_ptr + q_pos * sCOS_s + offs_half * sCOS_d).to(in_dtype)
    sin_q = tl.load(SIN_ptr + q_pos * sSIN_s + offs_half * sSIN_d).to(in_dtype)

    # Build dense Q halves for this head-group
    q0 = tl.zeros((GROUP_M, HALF), dtype=tl.float32)
    q1 = tl.zeros((GROUP_M, HALF), dtype=tl.float32)

    for r0 in range(0, R, BR):
        r = r0 + tl.arange(0, BR)
        mask_r = r < R

        Pq_blk = tl.load(
            Pq_q_ptr + bid * sPq_b + hid_qs[:, None] * sPq_h + r[None, :] * sPq_r,
            mask=mask_g[:, None] & mask_r[None, :],
            other=0.0,
        ).to(in_dtype)

        Vq0 = tl.load(
            Vq_ptr + hid_qs[:, None, None] * sVq_h + r[None, :, None] * sVq_r + offs_half[None, None, :] * sVq_d,
            mask=mask_g[:, None, None] & mask_r[None, :, None],
            other=0.0,
        ).to(in_dtype)
        Vq1 = tl.load(
            Vq_ptr + hid_qs[:, None, None] * sVq_h + r[None, :, None] * sVq_r + (offs_half + HALF)[None, None, :] * sVq_d,
            mask=mask_g[:, None, None] & mask_r[None, :, None],
            other=0.0,
        ).to(in_dtype)

        q0 += tl.sum(Pq_blk[:, :, None].to(tl.float32) * Vq0.to(tl.float32), axis=1)
        q1 += tl.sum(Pq_blk[:, :, None].to(tl.float32) * Vq1.to(tl.float32), axis=1)

    if HAS_BQ:
        bq0 = tl.load(
            bq_ptr + hid_qs[:, None] * sbq_h + offs_half[None, :] * sbq_d,
            mask=mask_g[:, None],
            other=0.0,
        ).to(tl.float32)
        bq1 = tl.load(
            bq_ptr + hid_qs[:, None] * sbq_h + (offs_half + HALF)[None, :] * sbq_d,
            mask=mask_g[:, None],
            other=0.0,
        ).to(tl.float32)
        q0 += bq0
        q1 += bq1

    q0t = q0.to(in_dtype)
    q1t = q1.to(in_dtype)
    q0r = q0t * cos_q[None, :] - q1t * sin_q[None, :]
    q1r = q0t * sin_q[None, :] + q1t * cos_q[None, :]

    # Vk resident
    if VK_RESIDENT:
        Vk0_full = tl.load(
            Vk_ptr + hid_k * sVk_h + offs_r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
            mask=offs_r[:, None] < R,
            other=0.0,
        ).to(in_dtype)
        Vk1_full = tl.load(
            Vk_ptr + hid_k * sVk_h + offs_r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
            mask=offs_r[:, None] < R,
            other=0.0,
        ).to(in_dtype)

    m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((GROUP_M,), tl.float32)
    acc_r = tl.zeros((GROUP_M, R), tl.float32)

    # Key blocks (0..SPLIT_K)
    for nk in range(0, SPLIT_K, BN):
        offs_n = nk + tl.arange(0, BN)
        valid_n = offs_n < seqlen_k

        if CAUSAL:
            valid_n = valid_n & (offs_n <= q_pos)
        if WINDOW_LEFT != -1:
            valid_n = valid_n & (offs_n >= (q_pos - WINDOW_LEFT))
        if WINDOW_RIGHT != -1:
            valid_n = valid_n & (offs_n <= (q_pos + WINDOW_RIGHT))

        cos_k = tl.load(
            COS_ptr + offs_n[:, None] * sCOS_s + offs_half[None, :] * sCOS_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)
        sin_k = tl.load(
            SIN_ptr + offs_n[:, None] * sSIN_s + offs_half[None, :] * sSIN_d,
            mask=valid_n[:, None],
            other=0.0,
        ).to(in_dtype)

        if VK_RESIDENT:
            Pk_blk = tl.load(
                Pk_ptr + bid * sPk_b + offs_n[:, None] * sPk_s + hid_k * sPk_h + offs_r[None, :] * sPk_r,
                mask=valid_n[:, None] & (offs_r[None, :] < R),
                other=0.0,
            ).to(in_dtype)
            k0 = tl.dot(Pk_blk, Vk0_full).to(tl.float32)
            k1 = tl.dot(Pk_blk, Vk1_full).to(tl.float32)
        else:
            k0 = tl.zeros((BN, HALF), dtype=tl.float32)
            k1 = tl.zeros((BN, HALF), dtype=tl.float32)
            for r0 in range(0, R, BR):
                r = r0 + tl.arange(0, BR)
                mask_r = r < R
                Pk_blk = tl.load(
                    Pk_ptr + bid * sPk_b + offs_n[:, None] * sPk_s + hid_k * sPk_h + r[None, :] * sPk_r,
                    mask=valid_n[:, None] & mask_r[None, :],
                    other=0.0,
                ).to(in_dtype)
                Vk0 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + offs_half[None, :] * sVk_d,
                    mask=mask_r[:, None],
                    other=0.0,
                ).to(in_dtype)
                Vk1 = tl.load(
                    Vk_ptr + hid_k * sVk_h + r[:, None] * sVk_r + (offs_half + HALF)[None, :] * sVk_d,
                    mask=mask_r[:, None],
                    other=0.0,
                ).to(in_dtype)
                k0 += tl.dot(Pk_blk, Vk0).to(tl.float32)
                k1 += tl.dot(Pk_blk, Vk1).to(tl.float32)

        if HAS_BK:
            bk0 = tl.load(bk_ptr + hid_k * sbk_h + offs_half * sbk_d).to(tl.float32)
            bk1 = tl.load(bk_ptr + hid_k * sbk_h + (offs_half + HALF) * sbk_d).to(tl.float32)
            k0 += bk0[None, :]
            k1 += bk1[None, :]

        k0t = k0.to(in_dtype)
        k1t = k1.to(in_dtype)
        k0r = k0t * cos_k - k1t * sin_k
        k1r = k0t * sin_k + k1t * cos_k

        scores = tl.dot(q0r, tl.trans(k0r)).to(tl.float32)
        scores += tl.dot(q1r, tl.trans(k1r)).to(tl.float32)
        scores *= SOFTMAX_SCALE

        scores = tl.where(valid_n[None, :], scores, -float("inf"))
        scores = tl.where(mask_g[:, None], scores, -float("inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)

        is_neg_inf = m_new == -float("inf")
        alpha = tl.where(is_neg_inf, 1.0, tl.exp(m_i - m_new))
        p = tl.where(is_neg_inf[:, None], 0.0, tl.exp(scores - m_new[:, None]))

        l_i = l_i * alpha + tl.sum(p, axis=1)

        Pv_blk = tl.load(
            Pv_ptr + bid * sPv_b + offs_n[:, None] * sPv_s + hid_k * sPv_h + offs_r[None, :] * sPv_r,
            mask=valid_n[:, None] & (offs_r[None, :] < R),
            other=0.0,
        ).to(in_dtype)

        acc_r = acc_r * alpha[:, None] + tl.dot(p.to(in_dtype), Pv_blk).to(tl.float32)
        m_i = tl.where(is_neg_inf, m_i, m_new)

    # Normalize in rank-space
    den = tl.where(l_i > 0, l_i, 1.0)
    w_r = acc_r / den[:, None]
    w_r = tl.where(l_i[:, None] > 0, w_r, 0.0)
    w_tc = w_r.to(in_dtype)

    # Lift once: (GROUP_M, R) x (R, HALF) -> (GROUP_M, HALF)
    Vv0 = tl.load(
        Vv_ptr + hid_k * sVv_h + offs_r[:, None] * sVv_r + offs_half[None, :] * sVv_d,
        mask=offs_r[:, None] < R,
        other=0.0,
    ).to(in_dtype)
    Vv1 = tl.load(
        Vv_ptr + hid_k * sVv_h + offs_r[:, None] * sVv_r + (offs_half + HALF)[None, :] * sVv_d,
        mask=offs_r[:, None] < R,
        other=0.0,
    ).to(in_dtype)

    out0 = tl.dot(w_tc, Vv0).to(tl.float32)
    out1 = tl.dot(w_tc, Vv1).to(tl.float32)

    if HAS_BV:
        bv0 = tl.load(
            bv_ptr + hid_k * sbv_h + offs_half * sbv_d,
            mask=True,
            other=0.0,
        ).to(tl.float32)
        bv1 = tl.load(
            bv_ptr + hid_k * sbv_h + (offs_half + HALF) * sbv_d,
            mask=True,
            other=0.0,
        ).to(tl.float32)
        out0 = out0 + bv0[None, :]
        out1 = out1 + bv1[None, :]

    # Store only real heads (g < REP)
    tl.store(
        O_ptr + bid * sO_b + hid_qs[:, None] * sO_h + offs_half[None, :] * sO_d,
        out0.to(in_dtype),
        mask=mask_g[:, None],
    )
    tl.store(
        O_ptr + bid * sO_b + hid_qs[:, None] * sO_h + (offs_half + HALF)[None, :] * sO_d,
        out1.to(in_dtype),
        mask=mask_g[:, None],
    )


# ----------------------------
# Decode wrappers (override default with v2 pipeline)
# ----------------------------
@torch.no_grad()
def flashsvd_attn_decode_packed(
    f: DecodePackedFactors,
    rotary_cos: torch.Tensor,  # [Smax, Dh/2]
    rotary_sin: torch.Tensor,  # [Smax, Dh/2]
    *,
    seqlen_k: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[int, int] = (-1, -1),
    # split-K parameters
    split_k: int = 512,
    bn: int = 64,
    br: int = 64,
    num_warps_stage1: int = 4,
    num_stages_stage1: int = 2,
    num_warps_stage2: int = 4,
    num_stages_stage2: int = 1,
    workspace: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    # v2 knobs
    q_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,   # (Q0, Q1) buffers [B,H,HALF]
    precompute_q: bool = True,
    writethrough: bool = True,
    pad_to_16: bool = True,
    vk_resident: bool = True,
) -> torch.Tensor:
    """
    v2 decode (packed): split-K Flash-Decoding style for low-rank KV.

    Notes:
      - num_splits is computed from seqlen_k (not Smax) to avoid wasted splits.
      - When num_splits==1 and writethrough=True, runs a single fused kernel (no workspace).
      - When pad_to_16=True, we pad REP to GROUP_M (multiple of 16) to use tensor cores.
      - vk_resident=True only enables full-Vk residency when BR==R and resident payload is small;
        otherwise it falls back to BR-tiled Vk loading to avoid SMEM overflow.
    """
    Pq_q = f.Pq
    Pk, Pv = f.Pk, f.Pv
    Vq, Vk, Vv = f.Vq, f.Vk, f.Vv
    bq, bk, bv = f.bq, f.bk, f.bv

    assert Pq_q.dim() == 3, f"Pq query must be [B,H,R], got {tuple(Pq_q.shape)}"
    B, H, R = Pq_q.shape
    assert Pk.dim() == 4 and Pv.dim() == 4
    B2, Smax, Hk, R2 = Pk.shape
    assert B2 == B and R2 == R
    assert Pv.shape == (B, Smax, Hk, R)
    assert H % Hk == 0
    rep = H // Hk

    assert seqlen_k >= 0 and seqlen_k <= Smax
    if seqlen_k == 0:
        # no keys: output zeros
        Dh = Vq.shape[-1]
        if out is None:
            return torch.zeros((B, H, Dh), device=Pq_q.device, dtype=Pq_q.dtype)
        out.zero_()
        return out

    # tiles
    bn = int(bn)
    split_k = int(split_k)
    assert split_k % bn == 0, "split_k must be a multiple of bn"
    # IMPORTANT: compute num_splits from *seqlen_k* to avoid empty splits.
    num_splits = max(1, triton.cdiv(seqlen_k, split_k))

    dtype = Pq_q.dtype
    assert dtype in (torch.float16, torch.bfloat16)
    is_bf16 = int(dtype == torch.bfloat16)

    Dh = Vq.shape[-1]
    assert Dh % 2 == 0
    half = Dh // 2

    assert Vq.shape == (H, R, Dh)
    assert Vk.shape == (Hk, R, Dh)
    assert Vv.shape == (Hk, R, Dh)
    assert rotary_cos.shape == (Smax, half) and rotary_sin.shape == (Smax, half)

    _assert_last_dim_contig(Pq_q, "Pq_q")
    _assert_last_dim_contig(Pk, "Pk")
    _assert_last_dim_contig(Pv, "Pv")
    _assert_last_dim_contig(Vq, "Vq")
    _assert_last_dim_contig(Vk, "Vk")
    _assert_last_dim_contig(Vv, "Vv")
    _assert_last_dim_contig(rotary_cos, "rotary_cos")
    _assert_last_dim_contig(rotary_sin, "rotary_sin")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dh)

    has_bq = int(bq is not None)
    has_bk = int(bk is not None)
    has_bv = int(bv is not None)
    if has_bq:
        assert bq.shape == (H, Dh)
        _assert_last_dim_contig(bq, "bq")
    if has_bk:
        assert bk.shape == (Hk, Dh)
        _assert_last_dim_contig(bk, "bk")
    if has_bv:
        assert bv.shape == (Hk, Dh)
        _assert_last_dim_contig(bv, "bv")

    # output: [B, H, Dh]
    if out is None:
        O = torch.empty((B, H, Dh), device=Pq_q.device, dtype=dtype)
    else:
        O = out
        if O.shape != (B, H, Dh) or O.dtype != dtype or O.device != Pq_q.device:
            raise ValueError(f"out must be {dtype}[{B},{H},{Dh}] on {Pq_q.device}")
        _assert_last_dim_contig(O, "out")

    # choose GROUP_M (pad heads) for tensorcore path
    if pad_to_16:
        group_m = 16 * ((rep + 15) // 16)
    else:
        group_m = rep  # may disable tensorcores for REP<16
    # Do not force BR=R blindly: full-Vk residency can exceed SMEM for large R (e.g. R=1024).
    # We keep small-tile BR by default, and only enable full residency when BR covers all ranks
    # and the Vk resident payload is small enough.
    br_eff = min(br, R)
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    full_vk_resident_bytes = 2 * R * half * elem_bytes  # Vk0 + Vk1
    allow_full_vk_resident = full_vk_resident_bytes <= 64 * 1024
    if vk_resident and br_eff == R and allow_full_vk_resident:
        vk_res = 1
    else:
        vk_res = 0

    # Fast path: single split -> writethrough fused kernel (no workspace)
    if writethrough and num_splits == 1:
        # strides
        sPq_b, sPq_h, sPq_r = Pq_q.stride()
        sPk_b, sPk_s, sPk_h, sPk_r = Pk.stride()
        sPv_b, sPv_s, sPv_h, sPv_r = Pv.stride()
        sVq_h, sVq_r, sVq_d = Vq.stride()
        sVk_h, sVk_r, sVk_d = Vk.stride()
        sVv_h, sVv_r, sVv_d = Vv.stride()
        sCOS_s, sCOS_d = rotary_cos.stride()
        sSIN_s, sSIN_d = rotary_sin.stride()
        sO_b, sO_h, sO_d = O.stride()
        if has_bq:
            sbq_h, sbq_d = bq.stride()
        else:
            sbq_h = sbq_d = 0
        if has_bk:
            sbk_h, sbk_d = bk.stride()
        else:
            sbk_h = sbk_d = 0
        if has_bv:
            sbv_h, sbv_d = bv.stride()
        else:
            sbv_h = sbv_d = 0

        grid = (B * Hk,)
        flashsvd_rope_decode_writethrough_packed_v2[grid](
            Pq_q,
            Pk, Pv,
            Vq, Vk, Vv,
            bq if has_bq else O,
            bk if has_bk else O,
            bv if has_bv else O,
            rotary_cos, rotary_sin,
            O,
            seqlen_k,
            sPq_b=sPq_b, sPq_h=sPq_h, sPq_r=sPq_r,
            sPk_b=sPk_b, sPk_s=sPk_s, sPk_h=sPk_h, sPk_r=sPk_r,
            sPv_b=sPv_b, sPv_s=sPv_s, sPv_h=sPv_h, sPv_r=sPv_r,
            sVq_h=sVq_h, sVq_r=sVq_r, sVq_d=sVq_d,
            sVk_h=sVk_h, sVk_r=sVk_r, sVk_d=sVk_d,
            sVv_h=sVv_h, sVv_r=sVv_r, sVv_d=sVv_d,
            sbq_h=sbq_h, sbq_d=sbq_d,
            sbk_h=sbk_h, sbk_d=sbk_d,
            sbv_h=sbv_h, sbv_d=sbv_d,
            sCOS_s=sCOS_s, sCOS_d=sCOS_d,
            sSIN_s=sSIN_s, sSIN_d=sSIN_d,
            sO_b=sO_b, sO_h=sO_h, sO_d=sO_d,
            SOFTMAX_SCALE=softmax_scale,
            CAUSAL=int(causal),
            WINDOW_LEFT=window_size[0],
            WINDOW_RIGHT=window_size[1],
            HAS_BQ=has_bq,
            HAS_BK=has_bk,
            HAS_BV=has_bv,
            R=R, DH=Dh, HALF=half,
            H=H, Hk=Hk, REP=rep,
            BN=bn, BR=br_eff, SPLIT_K=split_k,
            GROUP_M=group_m,
            VK_RESIDENT=vk_res,
            IS_BF16=is_bf16,
            num_warps=num_warps_stage1,
            num_stages=num_stages_stage1,
        )
        return O

    # ---------- Multi-split path (stage1 + reduce) ----------
    # Workspace: float32 states (optionally preallocated)
    if workspace is None:
        M = torch.empty((B, H, num_splits), device=Pq_q.device, dtype=torch.float32)
        L = torch.empty((B, H, num_splits), device=Pq_q.device, dtype=torch.float32)
        Acc = torch.empty((B, H, num_splits, R), device=Pq_q.device, dtype=torch.float32)
    else:
        if not (isinstance(workspace, tuple) and len(workspace) == 3):
            raise TypeError("workspace must be a tuple (M, L, Acc)")
        M, L, Acc = workspace
        if M.shape != (B, H, num_splits) or M.dtype != torch.float32 or M.device != Pq_q.device:
            raise ValueError(f"workspace M must be float32[{B},{H},{num_splits}] on {Pq_q.device}")
        if L.shape != (B, H, num_splits) or L.dtype != torch.float32 or L.device != Pq_q.device:
            raise ValueError(f"workspace L must be float32[{B},{H},{num_splits}] on {Pq_q.device}")
        if Acc.shape != (B, H, num_splits, R) or Acc.dtype != torch.float32 or Acc.device != Pq_q.device:
            raise ValueError(f"workspace Acc must be float32[{B},{H},{num_splits},{R}] on {Pq_q.device}")
        _assert_last_dim_contig(M, "workspace.M")
        _assert_last_dim_contig(L, "workspace.L")
        _assert_last_dim_contig(Acc, "workspace.Acc")

    # Q buffers: [B,H,HALF] (in dtype)
    if not precompute_q:
        raise ValueError("v2 decode currently requires precompute_q=True for multi-split (set writethrough=True for single-split).")

    if q_buffers is None:
        Q0 = torch.empty((B, H, half), device=Pq_q.device, dtype=dtype)
        Q1 = torch.empty((B, H, half), device=Pq_q.device, dtype=dtype)
    else:
        if not (isinstance(q_buffers, tuple) and len(q_buffers) == 2):
            raise TypeError("q_buffers must be a tuple (Q0, Q1)")
        Q0, Q1 = q_buffers
        if Q0.shape != (B, H, half) or Q0.dtype != dtype or Q0.device != Pq_q.device:
            raise ValueError(f"Q0 must be {dtype}[{B},{H},{half}] on {Pq_q.device}")
        if Q1.shape != (B, H, half) or Q1.dtype != dtype or Q1.device != Pq_q.device:
            raise ValueError(f"Q1 must be {dtype}[{B},{H},{half}] on {Pq_q.device}")
        _assert_last_dim_contig(Q0, "q_buffers.Q0")
        _assert_last_dim_contig(Q1, "q_buffers.Q1")

    # strides
    sPq_b, sPq_h, sPq_r = Pq_q.stride()
    sPk_b, sPk_s, sPk_h, sPk_r = Pk.stride()
    sPv_b, sPv_s, sPv_h, sPv_r = Pv.stride()
    sVq_h, sVq_r, sVq_d = Vq.stride()
    sVk_h, sVk_r, sVk_d = Vk.stride()
    sVv_h, sVv_r, sVv_d = Vv.stride()
    sCOS_s, sCOS_d = rotary_cos.stride()
    sSIN_s, sSIN_d = rotary_sin.stride()

    sQ_b, sQ_h, sQ_d = Q0.stride()
    sM_b, sM_h, sM_s = M.stride()
    sL_b, sL_h, sL_s = L.stride()
    sA_b, sA_h, sA_s, sA_r = Acc.stride()
    sO_b, sO_h, sO_d = O.stride()

    if has_bq:
        sbq_h, sbq_d = bq.stride()
    else:
        sbq_h = sbq_d = 0
    if has_bk:
        sbk_h, sbk_d = bk.stride()
    else:
        sbk_h = sbk_d = 0
    if has_bv:
        sbv_h, sbv_d = bv.stride()
    else:
        sbv_h = sbv_d = 0

    # (1) build Q halves once
    grid_q = (B * H,)
    flashsvd_rope_decode_build_q_packed_v2[grid_q](
        Pq_q,
        Vq,
        bq if has_bq else O,
        rotary_cos, rotary_sin,
        Q0, Q1,
        seqlen_k,
        sPq_b=sPq_b, sPq_h=sPq_h, sPq_r=sPq_r,
        sVq_h=sVq_h, sVq_r=sVq_r, sVq_d=sVq_d,
        sbq_h=sbq_h, sbq_d=sbq_d,
        sCOS_s=sCOS_s, sCOS_d=sCOS_d,
        sSIN_s=sSIN_s, sSIN_d=sSIN_d,
        sQ_b=sQ_b, sQ_h=sQ_h, sQ_d=sQ_d,
        HAS_BQ=has_bq,
        R=R, DH=Dh, HALF=half, H=H,
        BR=min(br_eff, R),
        IS_BF16=is_bf16,
        num_warps=4,
        num_stages=1,
    )

    # (2) stage1 split-K
    grid1 = (B * Hk, num_splits)
    flashsvd_rope_decode_splitk_stage1_packed_v2[grid1](
        Q0, Q1,
        Pk, Pv,
        Vk,
        bk if has_bk else O,
        rotary_cos, rotary_sin,
        M, L, Acc,
        seqlen_k,
        sQ_b=sQ_b, sQ_h=sQ_h, sQ_d=sQ_d,
        sPk_b=sPk_b, sPk_s=sPk_s, sPk_h=sPk_h, sPk_r=sPk_r,
        sPv_b=sPv_b, sPv_s=sPv_s, sPv_h=sPv_h, sPv_r=sPv_r,
        sVk_h=sVk_h, sVk_r=sVk_r, sVk_d=sVk_d,
        sbk_h=sbk_h, sbk_d=sbk_d,
        sCOS_s=sCOS_s, sCOS_d=sCOS_d,
        sSIN_s=sSIN_s, sSIN_d=sSIN_d,
        sM_b=sM_b, sM_h=sM_h, sM_s=sM_s,
        sL_b=sL_b, sL_h=sL_h, sL_s=sL_s,
        sA_b=sA_b, sA_h=sA_h, sA_s=sA_s, sA_r=sA_r,
        SOFTMAX_SCALE=softmax_scale,
        CAUSAL=int(causal),
        WINDOW_LEFT=window_size[0],
        WINDOW_RIGHT=window_size[1],
        HAS_BK=has_bk,
        R=R, DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        BN=bn, BR=br_eff, SPLIT_K=split_k,
        GROUP_M=group_m,
        VK_RESIDENT=vk_res,
        IS_BF16=is_bf16,
        num_warps=num_warps_stage1,
        num_stages=num_stages_stage1,
    )

    # (3) reduce
    grid2 = (B * H,)
    flashsvd_rope_decode_splitk_reduce_packed_v2[grid2](
        M, L, Acc,
        Vv,
        bv if has_bv else O,
        O,
        sM_b=sM_b, sM_h=sM_h, sM_s=sM_s,
        sL_b=sL_b, sL_h=sL_h, sL_s=sL_s,
        sA_b=sA_b, sA_h=sA_h, sA_s=sA_s, sA_r=sA_r,
        sVv_h=sVv_h, sVv_r=sVv_r, sVv_d=sVv_d,
        sbv_h=sbv_h, sbv_d=sbv_d,
        sO_b=sO_b, sO_h=sO_h, sO_d=sO_d,
        HAS_BV=has_bv,
        R=R, DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        NSPLIT=num_splits,
        IS_BF16=is_bf16,
        num_warps=num_warps_stage2,
        num_stages=num_stages_stage2,
    )
    return O


@torch.no_grad()
def flashsvd_attn_decode_varlen(
    f: DecodeVarlenFactors,
    cu_seqlens: torch.Tensor,   # [B+1] int32
    max_seqlen: int,
    rotary_cos: torch.Tensor,   # [max_seqlen, Dh/2]
    rotary_sin: torch.Tensor,   # [max_seqlen, Dh/2]
    *,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    window_size: Tuple[int, int] = (-1, -1),
    split_k: int = 512,
    bn: int = 64,
    br: int = 64,
    num_warps_stage1: int = 4,
    num_stages_stage1: int = 2,
    num_warps_stage2: int = 4,
    num_stages_stage2: int = 1,
    workspace: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    out: Optional[torch.Tensor] = None,
    # v2 knobs
    q_buffers: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    precompute_q: bool = True,
    pad_to_16: bool = True,
    vk_resident: bool = True,
) -> torch.Tensor:
    """
    v2 varlen decode: same algorithmic upgrades as packed, but q_pos is per-sequence from cu_seqlens.
    vk_resident follows the same safety policy as packed decode (auto fallback to BR-tiled path).
    """
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_cuda

    Pq_q = f.Pq
    Pk, Pv = f.Pk, f.Pv
    Vq, Vk, Vv = f.Vq, f.Vk, f.Vv
    bq, bk, bv = f.bq, f.bk, f.bv

    assert Pq_q.dim() == 3
    B, H, R = Pq_q.shape
    T, Hk, R2 = Pk.shape
    assert R2 == R and Pv.shape == (T, Hk, R)
    assert H % Hk == 0
    rep = H // Hk

    bn = int(bn)
    split_k = int(split_k)
    assert split_k % bn == 0
    num_splits = max(1, triton.cdiv(max_seqlen, split_k))

    dtype = Pq_q.dtype
    assert dtype in (torch.float16, torch.bfloat16)
    is_bf16 = int(dtype == torch.bfloat16)

    Dh = Vq.shape[-1]
    assert Dh % 2 == 0
    half = Dh // 2

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Dh)

    has_bq = int(bq is not None)
    has_bk = int(bk is not None)
    has_bv = int(bv is not None)

    if pad_to_16:
        group_m = 16 * ((rep + 15) // 16)
    else:
        group_m = rep

    br_eff = min(br, R)
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    full_vk_resident_bytes = 2 * R * half * elem_bytes  # Vk0 + Vk1
    allow_full_vk_resident = full_vk_resident_bytes <= 64 * 1024
    if vk_resident and br_eff == R and allow_full_vk_resident:
        vk_res = 1
    else:
        vk_res = 0

    # output: [B, H, Dh]
    if out is None:
        O = torch.empty((B, H, Dh), device=Pq_q.device, dtype=dtype)
    else:
        O = out
        if O.shape != (B, H, Dh) or O.dtype != dtype or O.device != Pq_q.device:
            raise ValueError(f"out must be {dtype}[{B},{H},{Dh}] on {Pq_q.device}")
        _assert_last_dim_contig(O, "out")

    # Workspace
    if workspace is None:
        M = torch.empty((B, H, num_splits), device=Pq_q.device, dtype=torch.float32)
        L = torch.empty((B, H, num_splits), device=Pq_q.device, dtype=torch.float32)
        Acc = torch.empty((B, H, num_splits, R), device=Pq_q.device, dtype=torch.float32)
    else:
        if not (isinstance(workspace, tuple) and len(workspace) == 3):
            raise TypeError("workspace must be a tuple (M, L, Acc)")
        M, L, Acc = workspace

    # Q buffers
    if not precompute_q:
        raise ValueError("v2 varlen decode currently requires precompute_q=True.")
    if q_buffers is None:
        Q0 = torch.empty((B, H, half), device=Pq_q.device, dtype=dtype)
        Q1 = torch.empty((B, H, half), device=Pq_q.device, dtype=dtype)
    else:
        Q0, Q1 = q_buffers

    # strides
    sPq_b, sPq_h, sPq_r = Pq_q.stride()
    sPk_t, sPk_h, sPk_r = Pk.stride()
    sPv_t, sPv_h, sPv_r = Pv.stride()
    sVq_h, sVq_r, sVq_d = Vq.stride()
    sVk_h, sVk_r, sVk_d = Vk.stride()
    sVv_h, sVv_r, sVv_d = Vv.stride()
    sCOS_s, sCOS_d = rotary_cos.stride()
    sSIN_s, sSIN_d = rotary_sin.stride()
    sQ_b, sQ_h, sQ_d = Q0.stride()
    sM_b, sM_h, sM_s = M.stride()
    sL_b, sL_h, sL_s = L.stride()
    sA_b, sA_h, sA_s, sA_r = Acc.stride()
    sO_b, sO_h, sO_d = O.stride()

    if has_bq:
        sbq_h, sbq_d = bq.stride()
    else:
        sbq_h = sbq_d = 0
    if has_bk:
        sbk_h, sbk_d = bk.stride()
    else:
        sbk_h = sbk_d = 0
    if has_bv:
        sbv_h, sbv_d = bv.stride()
    else:
        sbv_h = sbv_d = 0

    # (1) build Q halves (per sequence q_pos)
    grid_q = (B * H,)
    flashsvd_rope_decode_build_q_varlen_v2[grid_q](
        Pq_q,
        Vq,
        bq if has_bq else O,
        rotary_cos, rotary_sin,
        Q0, Q1,
        cu_seqlens,
        sPq_b=sPq_b, sPq_h=sPq_h, sPq_r=sPq_r,
        sVq_h=sVq_h, sVq_r=sVq_r, sVq_d=sVq_d,
        sbq_h=sbq_h, sbq_d=sbq_d,
        sCOS_s=sCOS_s, sCOS_d=sCOS_d,
        sSIN_s=sSIN_s, sSIN_d=sSIN_d,
        sQ_b=sQ_b, sQ_h=sQ_h, sQ_d=sQ_d,
        HAS_BQ=has_bq,
        max_seqlen=max_seqlen,
        R=R, DH=Dh, HALF=half, H=H,
        BR=min(br_eff, R),
        IS_BF16=is_bf16,
        num_warps=4,
        num_stages=1,
    )

    # (2) stage1
    grid1 = (B * Hk, num_splits)
    flashsvd_rope_decode_splitk_stage1_varlen_v2[grid1](
        Q0, Q1,
        Pk, Pv,
        Vk,
        bk if has_bk else O,
        rotary_cos, rotary_sin,
        M, L, Acc,
        cu_seqlens,
        sQ_b=sQ_b, sQ_h=sQ_h, sQ_d=sQ_d,
        sPk_t=sPk_t, sPk_h=sPk_h, sPk_r=sPk_r,
        sPv_t=sPv_t, sPv_h=sPv_h, sPv_r=sPv_r,
        sVk_h=sVk_h, sVk_r=sVk_r, sVk_d=sVk_d,
        sbk_h=sbk_h, sbk_d=sbk_d,
        sCOS_s=sCOS_s, sCOS_d=sCOS_d,
        sSIN_s=sSIN_s, sSIN_d=sSIN_d,
        sM_b=sM_b, sM_h=sM_h, sM_s=sM_s,
        sL_b=sL_b, sL_h=sL_h, sL_s=sL_s,
        sA_b=sA_b, sA_h=sA_h, sA_s=sA_s, sA_r=sA_r,
        SOFTMAX_SCALE=softmax_scale,
        CAUSAL=int(causal),
        WINDOW_LEFT=window_size[0],
        WINDOW_RIGHT=window_size[1],
        HAS_BK=has_bk,
        max_seqlen=max_seqlen,
        R=R, DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        BN=bn, BR=br_eff, SPLIT_K=split_k,
        GROUP_M=group_m,
        VK_RESIDENT=vk_res,
        IS_BF16=is_bf16,
        num_warps=num_warps_stage1,
        num_stages=num_stages_stage1,
    )

    # (3) reduce (reuse packed reduce)
    grid2 = (B * H,)
    flashsvd_rope_decode_splitk_reduce_packed_v2[grid2](
        M, L, Acc,
        Vv,
        bv if has_bv else O,
        O,
        sM_b=sM_b, sM_h=sM_h, sM_s=sM_s,
        sL_b=sL_b, sL_h=sL_h, sL_s=sL_s,
        sA_b=sA_b, sA_h=sA_h, sA_s=sA_s, sA_r=sA_r,
        sVv_h=sVv_h, sVv_r=sVv_r, sVv_d=sVv_d,
        sbv_h=sbv_h, sbv_d=sbv_d,
        sO_b=sO_b, sO_h=sO_h, sO_d=sO_d,
        HAS_BV=has_bv,
        R=R, DH=Dh, HALF=half,
        H=H, Hk=Hk, REP=rep,
        NSPLIT=num_splits,
        IS_BF16=is_bf16,
        num_warps=num_warps_stage2,
        num_stages=num_stages_stage2,
    )
    return O



if __name__ == "__main__":
    main()
