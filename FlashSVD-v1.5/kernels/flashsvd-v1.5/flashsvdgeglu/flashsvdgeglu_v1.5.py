#!/usr/bin/env python3
"""FlashSVD GEGLU v1.5

Redesigned for encoder throughput:
- default fast path fuses phase-1 and phase-2 (no materialized S buffer)
- keeps two-stage path as compatibility fallback
- adds lightweight runtime scheduler to reduce pointless autotune/recompile overhead
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_PRECOMPUTED_FFN_G_CACHE: Dict[tuple, torch.Tensor] = {}


# -----------------------------
# Triton kernels
# -----------------------------
@triton.jit
def fused_ffn_phase1_geglu_kernel(
    P_ptr,
    V1_ptr,
    U2_ptr,
    S_ptr,
    b1_ptr,
    B,
    L,
    D,
    R1,
    R2,
    sP_b,
    sP_l,
    sP_r1,
    sV1_r1,
    sV1_d,
    sU2_d,
    sU2_r2,
    sb1,
    sS_b,
    sS_l,
    sS_r2,
    BL: tl.constexpr,
    BD: tl.constexpr,
    BR1: tl.constexpr,
    BR2: tl.constexpr,
    USE_TANH: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_r2 = tl.program_id(2)

    offs_l = pid_l * BL + tl.arange(0, BL)
    offs_r2 = pid_r2 * BR2 + tl.arange(0, BR2)

    acc = tl.zeros((BL, BR2), dtype=tl.float32)

    c0 = 0.7978845608028654
    c1 = 0.044715
    inv_sqrt2 = 0.7071067811865476

    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m_d = d < D

        tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
        tv_acc = tl.zeros((BL, BD), dtype=tl.float32)

        for r1_0 in range(0, R1, BR1):
            r1 = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1

            p_blk = tl.load(
                P_ptr + pid_b * sP_b + offs_l[:, None] * sP_l + r1[None, :] * sP_r1,
                mask=(offs_l[:, None] < L) & m_r1[None, :],
                other=0.0,
            )
            v1u_blk = tl.load(
                V1_ptr + r1[:, None] * sV1_r1 + d[None, :] * sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0,
            )
            v1v_blk = tl.load(
                V1_ptr + r1[:, None] * sV1_r1 + (d[None, :] + D) * sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0,
            )

            tu_acc += tl.dot(p_blk.to(tl.float32), v1u_blk.to(tl.float32))
            tv_acc += tl.dot(p_blk.to(tl.float32), v1v_blk.to(tl.float32))

        b1u = tl.load(b1_ptr + d * sb1, mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D) * sb1, mask=m_d, other=0.0).to(tl.float32)
        tu = tu_acc + b1u[None, :]
        tv = tv_acc + b1v[None, :]

        if USE_TANH:
            z = c0 * (tu + c1 * tu * tu * tu)
            z2 = 2.0 * z
            sig_2z = tl.where(z2 >= 0, 1.0 / (1.0 + tl.exp(-z2)), tl.exp(z2) / (1.0 + tl.exp(z2)))
            tanh_z = 2.0 * sig_2z - 1.0
            hu = 0.5 * tu * (1.0 + tanh_z)
        else:
            hu = 0.5 * tu * (1.0 + tl.erf(tu * inv_sqrt2))

        h_blk = hu * tv

        u2_blk = tl.load(
            U2_ptr + d[:, None] * sU2_d + offs_r2[None, :] * sU2_r2,
            mask=m_d[:, None] & (offs_r2[None, :] < R2),
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(h_blk, u2_blk)

    mask = (offs_l[:, None] < L) & (offs_r2[None, :] < R2)
    tl.store(S_ptr + pid_b * sS_b + offs_l[:, None] * sS_l + offs_r2[None, :] * sS_r2, acc, mask=mask)


@triton.jit
def fused_ffn_phase12_geglu_kernel(
    P_ptr,
    V1_ptr,
    U2_ptr,
    V2_ptr,
    Y_ptr,
    b1_ptr,
    b2_ptr,
    B,
    L,
    D,
    R1,
    R2,
    Hdim,
    sP_b,
    sP_l,
    sP_r1,
    sV1_r1,
    sV1_d,
    sU2_d,
    sU2_r2,
    sV2_r2,
    sV2_h,
    sY_b,
    sY_l,
    sY_h,
    sb1,
    sb2,
    BL: tl.constexpr,
    BD: tl.constexpr,
    BR1: tl.constexpr,
    BH: tl.constexpr,
    BR2: tl.constexpr,
    USE_TANH: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_l = pid_l * BL + tl.arange(0, BL)
    offs_h = pid_h * BH + tl.arange(0, BH)

    acc_y = tl.zeros((BL, BH), dtype=tl.float32)

    c0 = 0.7978845608028654
    c1 = 0.044715
    inv_sqrt2 = 0.7071067811865476

    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m_d = d < D

        tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
        tv_acc = tl.zeros((BL, BD), dtype=tl.float32)

        for r1_0 in range(0, R1, BR1):
            r1 = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1

            p_blk = tl.load(
                P_ptr + pid_b * sP_b + offs_l[:, None] * sP_l + r1[None, :] * sP_r1,
                mask=(offs_l[:, None] < L) & m_r1[None, :],
                other=0.0,
            )
            v1u_blk = tl.load(
                V1_ptr + r1[:, None] * sV1_r1 + d[None, :] * sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0,
            )
            v1v_blk = tl.load(
                V1_ptr + r1[:, None] * sV1_r1 + (d[None, :] + D) * sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0,
            )

            tu_acc += tl.dot(p_blk.to(tl.float32), v1u_blk.to(tl.float32))
            tv_acc += tl.dot(p_blk.to(tl.float32), v1v_blk.to(tl.float32))

        b1u = tl.load(b1_ptr + d * sb1, mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D) * sb1, mask=m_d, other=0.0).to(tl.float32)
        tu = tu_acc + b1u[None, :]
        tv = tv_acc + b1v[None, :]

        if USE_TANH:
            z = c0 * (tu + c1 * tu * tu * tu)
            z2 = 2.0 * z
            sig_2z = tl.where(z2 >= 0, 1.0 / (1.0 + tl.exp(-z2)), tl.exp(z2) / (1.0 + tl.exp(z2)))
            tanh_z = 2.0 * sig_2z - 1.0
            hu = 0.5 * tu * (1.0 + tanh_z)
        else:
            hu = 0.5 * tu * (1.0 + tl.erf(tu * inv_sqrt2))

        h_blk = hu * tv

        g_acc = tl.zeros((BD, BH), dtype=tl.float32)
        for r2_0 in range(0, R2, BR2):
            r2 = r2_0 + tl.arange(0, BR2)
            m_r2 = r2 < R2

            u2_blk = tl.load(
                U2_ptr + d[:, None] * sU2_d + r2[None, :] * sU2_r2,
                mask=m_d[:, None] & m_r2[None, :],
                other=0.0,
            )
            v2_blk = tl.load(
                V2_ptr + r2[:, None] * sV2_r2 + offs_h[None, :] * sV2_h,
                mask=m_r2[:, None] & (offs_h[None, :] < Hdim),
                other=0.0,
            )

            g_acc += tl.dot(u2_blk.to(tl.float32), v2_blk.to(tl.float32))

        acc_y += tl.dot(h_blk, g_acc)

    b2_blk = tl.load(b2_ptr + offs_h * sb2, mask=offs_h < Hdim, other=0.0).to(tl.float32)
    mask = (offs_l[:, None] < L) & (offs_h[None, :] < Hdim)
    tl.store(
        Y_ptr + pid_b * sY_b + offs_l[:, None] * sY_l + offs_h[None, :] * sY_h,
        acc_y + b2_blk[None, :],
        mask=mask,
    )


@triton.jit
def fused_ffn_phase13_geglu_preg_kernel(
    P_ptr,
    V1_ptr,
    G_ptr,
    Y_ptr,
    b1_ptr,
    b2_ptr,
    B,
    L,
    D,
    R1,
    Hdim,
    sP_b,
    sP_l,
    sP_r1,
    sV1_r1,
    sV1_d,
    sG_d,
    sG_h,
    sY_b,
    sY_l,
    sY_h,
    sb1,
    sb2,
    BL: tl.constexpr,
    BD: tl.constexpr,
    BR1: tl.constexpr,
    BH: tl.constexpr,
    USE_TANH: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_l = pid_l * BL + tl.arange(0, BL)
    offs_h = pid_h * BH + tl.arange(0, BH)
    mask_h = offs_h < Hdim

    acc_y = tl.zeros((BL, BH), dtype=tl.float32)

    c0 = 0.7978845608028654
    c1 = 0.044715
    inv_sqrt2 = 0.7071067811865476

    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m_d = d < D

        tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
        tv_acc = tl.zeros((BL, BD), dtype=tl.float32)

        for r1_0 in range(0, R1, BR1):
            r1 = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1

            p_blk = tl.load(
                P_ptr + pid_b * sP_b + offs_l[:, None] * sP_l + r1[None, :] * sP_r1,
                mask=(offs_l[:, None] < L) & m_r1[None, :],
                other=0.0,
            )
            v1u_blk = tl.load(
                V1_ptr + r1[:, None] * sV1_r1 + d[None, :] * sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0,
            )
            v1v_blk = tl.load(
                V1_ptr + r1[:, None] * sV1_r1 + (d[None, :] + D) * sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0,
            )

            tu_acc += tl.dot(p_blk.to(tl.float32), v1u_blk.to(tl.float32))
            tv_acc += tl.dot(p_blk.to(tl.float32), v1v_blk.to(tl.float32))

        b1u = tl.load(b1_ptr + d * sb1, mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D) * sb1, mask=m_d, other=0.0).to(tl.float32)
        tu = tu_acc + b1u[None, :]
        tv = tv_acc + b1v[None, :]

        if USE_TANH:
            z = c0 * (tu + c1 * tu * tu * tu)
            z2 = 2.0 * z
            sig_2z = tl.where(z2 >= 0, 1.0 / (1.0 + tl.exp(-z2)), tl.exp(z2) / (1.0 + tl.exp(z2)))
            tanh_z = 2.0 * sig_2z - 1.0
            hu = 0.5 * tu * (1.0 + tanh_z)
        else:
            hu = 0.5 * tu * (1.0 + tl.erf(tu * inv_sqrt2))

        h_blk = hu * tv
        g_blk = tl.load(
            G_ptr + d[:, None] * sG_d + offs_h[None, :] * sG_h,
            mask=m_d[:, None] & mask_h[None, :],
            other=0.0,
        ).to(tl.float32)
        acc_y += tl.dot(h_blk, g_blk)

    b2_blk = tl.load(b2_ptr + offs_h * sb2, mask=mask_h, other=0.0).to(tl.float32)
    mask = (offs_l[:, None] < L) & mask_h[None, :]
    tl.store(
        Y_ptr + pid_b * sY_b + offs_l[:, None] * sY_l + offs_h[None, :] * sY_h,
        acc_y + b2_blk[None, :],
        mask=mask,
    )


# -----------------------------
# Config selection
# -----------------------------
_TWO_STAGE_DEFAULT = {
    "BL": 64,
    "BD": 128,
    "BR1": 64,
    "BR2": 128,
    "num_warps": 8,
    "num_stages": 2,
}

_FUSED_DEFAULT = {
    "BL": 64,
    "BD": 128,
    "BR1": 64,
    "BH": 128,
    "BR2": 64,
    "num_warps": 8,
    "num_stages": 2,
}


def _tensor_version(t: torch.Tensor) -> int:
    try:
        return int(getattr(t, "_version", -1))
    except Exception:
        return -1


def _precompute_g_enabled() -> bool:
    raw = os.getenv("FLASH_SVD_GEGLU_PRECOMPUTE_G", "").strip().lower()
    if raw in {"0", "false", "off", "no"}:
        return False
    if raw in {"1", "true", "on", "yes"}:
        return True
    return True


def _precompute_g_cache_key(U2: torch.Tensor, V2: torch.Tensor) -> tuple:
    return (
        U2.device.type,
        U2.device.index,
        str(U2.dtype),
        int(U2.shape[0]),
        int(U2.shape[1]),
        int(V2.shape[1]),
        int(U2.data_ptr()),
        int(V2.data_ptr()),
        _tensor_version(U2),
        _tensor_version(V2),
    )


def precompute_ffn_g(U2: torch.Tensor, V2: torch.Tensor) -> torch.Tensor:
    return U2.matmul(V2).contiguous()


def _get_precomputed_ffn_g(U2: torch.Tensor, V2: torch.Tensor) -> torch.Tensor:
    key = _precompute_g_cache_key(U2, V2)
    cached = _PRECOMPUTED_FFN_G_CACHE.get(key)
    if cached is not None:
        return cached
    G = precompute_ffn_g(U2, V2)
    _PRECOMPUTED_FFN_G_CACHE[key] = G
    return G


def _pick_two_stage_cfg(B: int, L: int, R2: int) -> Dict[str, int]:
    cfg = dict(_TWO_STAGE_DEFAULT)
    if L <= 512:
        cfg.update(BL=32, BR2=64, num_warps=4)
    if R2 <= 128:
        cfg.update(BR2=64)
    if B <= 2 and L <= 256:
        cfg.update(BL=32, BD=64, BR1=64, BR2=64, num_warps=4)
    return cfg


def _pick_fused_cfg(B: int, L: int, H: int, R1: int, R2: int) -> Dict[str, int]:
    cfg = dict(_FUSED_DEFAULT)
    if L <= 512:
        cfg.update(BL=32, BH=64, num_warps=4)
    if H <= 768:
        cfg.update(BH=64)
    if R2 >= 1024:
        cfg.update(BR2=128)
    if B <= 2 and L <= 256:
        cfg.update(BL=32, BD=64, BR1=64, BH=64, BR2=64, num_warps=4)
    if R1 <= 128:
        cfg.update(BR1=32)
    return cfg


def _effective_variant(kernel_variant: str, prefer_fused: Optional[bool], P: torch.Tensor, V2: torch.Tensor) -> str:
    env_variant = os.getenv(
        "FLASH_SVD_GEGLU_KERNEL_VARIANT",
        os.getenv("FLASHSVD_GEGLU_KERNEL_VARIANT", ""),
    ).strip().lower()
    if env_variant in {"preg", "preg_fused", "fused_preg", "preg-fused"}:
        return "preg"
    if env_variant in {"fused", "two_stage", "two-stage"}:
        return "fused" if env_variant == "fused" else "two_stage"

    kv = kernel_variant.strip().lower()
    if kv in {"preg", "preg_fused", "fused_preg", "preg-fused"}:
        return "preg"
    if kv in {"fused", "two_stage", "two-stage"}:
        return "fused" if kv == "fused" else "two_stage"

    if prefer_fused is not None:
        return "fused" if prefer_fused else "two_stage"

    # Auto policy: fused wins for encoder-like sizes because it avoids materializing S.
    B, L, _ = P.shape
    H = V2.shape[1]
    if P.is_cuda and L >= 128 and H >= 256 and P.dtype in (torch.float16, torch.bfloat16):
        return "preg" if _precompute_g_enabled() else "fused"
    return "two_stage"


# -----------------------------
# Public wrappers
# -----------------------------
def flashsvd_ffn_geglu_two_stage(
    P: torch.Tensor,
    V1: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    *,
    BL: int = 64,
    BD: int = 128,
    BR1: int = 64,
    BR2: int = 128,
    gelu_approx: str = "tanh",
    store_s_fp32: bool = False,
    num_warps: int = 8,
    num_stages: int = 2,
) -> torch.Tensor:
    B, L, R1 = P.shape
    R1_v1, two_d = V1.shape
    D = two_d // 2
    D_u2, R2 = U2.shape
    R2_v2, H = V2.shape

    assert R1_v1 == R1 and two_d == 2 * D
    assert D_u2 == D and R2_v2 == R2
    assert b1.numel() == 2 * D and b2.numel() == H

    s_dtype = torch.float32 if store_s_fp32 else P.dtype
    S = torch.empty((B, L, R2), device=P.device, dtype=s_dtype)

    grid = (B, triton.cdiv(L, BL), triton.cdiv(R2, BR2))
    use_tanh = 1 if gelu_approx == "tanh" else 0

    fused_ffn_phase1_geglu_kernel[grid](
        P,
        V1,
        U2,
        S,
        b1,
        B,
        L,
        D,
        R1,
        R2,
        P.stride(0),
        P.stride(1),
        P.stride(2),
        V1.stride(0),
        V1.stride(1),
        U2.stride(0),
        U2.stride(1),
        b1.stride(0),
        S.stride(0),
        S.stride(1),
        S.stride(2),
        BL=BL,
        BD=BD,
        BR1=BR1,
        BR2=BR2,
        USE_TANH=use_tanh,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if S.dtype == V2.dtype:
        y = S.matmul(V2)
    else:
        y = S.matmul(V2.to(S.dtype))
    if y.dtype != P.dtype:
        y = y.to(P.dtype)
    return y + b2.to(y.dtype).view(1, 1, -1)


def flashsvd_ffn_geglu_fused(
    P: torch.Tensor,
    V1: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    *,
    BL: int = 64,
    BD: int = 128,
    BR1: int = 64,
    BH: int = 128,
    BR2: int = 64,
    gelu_approx: str = "tanh",
    num_warps: int = 8,
    num_stages: int = 2,
) -> torch.Tensor:
    B, L, R1 = P.shape
    R1_v1, two_d = V1.shape
    D = two_d // 2
    D_u2, R2 = U2.shape
    R2_v2, H = V2.shape

    assert R1_v1 == R1 and two_d == 2 * D
    assert D_u2 == D and R2_v2 == R2
    assert b1.numel() == 2 * D and b2.numel() == H

    Y = torch.empty((B, L, H), device=P.device, dtype=P.dtype)
    grid = (B, triton.cdiv(L, BL), triton.cdiv(H, BH))
    use_tanh = 1 if gelu_approx == "tanh" else 0

    fused_ffn_phase12_geglu_kernel[grid](
        P,
        V1,
        U2,
        V2,
        Y,
        b1,
        b2,
        B,
        L,
        D,
        R1,
        R2,
        H,
        P.stride(0),
        P.stride(1),
        P.stride(2),
        V1.stride(0),
        V1.stride(1),
        U2.stride(0),
        U2.stride(1),
        V2.stride(0),
        V2.stride(1),
        Y.stride(0),
        Y.stride(1),
        Y.stride(2),
        b1.stride(0),
        b2.stride(0),
        BL=BL,
        BD=BD,
        BR1=BR1,
        BH=BH,
        BR2=BR2,
        USE_TANH=use_tanh,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return Y


def flashsvd_ffn_geglu_fused_preg(
    P: torch.Tensor,
    V1: torch.Tensor,
    G: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    *,
    BL: int = 64,
    BD: int = 128,
    BR1: int = 64,
    BH: int = 128,
    gelu_approx: str = "tanh",
    num_warps: int = 8,
    num_stages: int = 2,
) -> torch.Tensor:
    B, L, R1 = P.shape
    R1_v1, two_d = V1.shape
    D = two_d // 2
    D_g, H = G.shape

    assert R1_v1 == R1 and two_d == 2 * D
    assert D_g == D
    assert b1.numel() == 2 * D and b2.numel() == H

    Y = torch.empty((B, L, H), device=P.device, dtype=P.dtype)
    grid = (B, triton.cdiv(L, BL), triton.cdiv(H, BH))
    use_tanh = 1 if gelu_approx == "tanh" else 0

    fused_ffn_phase13_geglu_preg_kernel[grid](
        P,
        V1,
        G,
        Y,
        b1,
        b2,
        B,
        L,
        D,
        R1,
        H,
        P.stride(0),
        P.stride(1),
        P.stride(2),
        V1.stride(0),
        V1.stride(1),
        G.stride(0),
        G.stride(1),
        Y.stride(0),
        Y.stride(1),
        Y.stride(2),
        b1.stride(0),
        b2.stride(0),
        BL=BL,
        BD=BD,
        BR1=BR1,
        BH=BH,
        USE_TANH=use_tanh,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return Y


def flashsvd_ffn_geglu_autotuned(
    P: torch.Tensor,
    V1: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    gelu_approx: str = "tanh",
    store_s_fp32: bool = False,
    *,
    kernel_variant: str = "auto",
    prefer_fused: Optional[bool] = None,
    use_precomputed_g: Optional[bool] = None,
    precomputed_g: Optional[torch.Tensor] = None,
    fused_cfg: Optional[Dict[str, int]] = None,
    two_stage_cfg: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Main entry used by ModernBERT path.

    Compatibility notes:
    - keeps old signature (`gelu_approx`, `store_s_fp32`)
    - new knobs are optional and keyword-only
    """
    assert P.is_cuda and V1.is_cuda and U2.is_cuda and V2.is_cuda, "FlashSVD GEGLU requires CUDA tensors"

    B, L, R1 = P.shape
    _ = R1
    _, H = V2.shape

    variant = _effective_variant(kernel_variant, prefer_fused, P, V2)
    want_preg = bool(use_precomputed_g) if use_precomputed_g is not None else (variant == "preg")
    if want_preg:
        cfg = _pick_fused_cfg(B=B, L=L, H=H, R1=V1.shape[0], R2=U2.shape[1])
        if fused_cfg:
            cfg.update({k: int(v) for k, v in fused_cfg.items()})
        G = precomputed_g if precomputed_g is not None else _get_precomputed_ffn_g(U2, V2)
        return flashsvd_ffn_geglu_fused_preg(
            P,
            V1,
            G,
            b1,
            b2,
            BL=cfg["BL"],
            BD=cfg["BD"],
            BR1=cfg["BR1"],
            BH=cfg["BH"],
            gelu_approx=gelu_approx,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    if variant == "fused":
        cfg = _pick_fused_cfg(B=B, L=L, H=H, R1=V1.shape[0], R2=U2.shape[1])
        if fused_cfg:
            cfg.update({k: int(v) for k, v in fused_cfg.items()})
        return flashsvd_ffn_geglu_fused(
            P,
            V1,
            U2,
            V2,
            b1,
            b2,
            BL=cfg["BL"],
            BD=cfg["BD"],
            BR1=cfg["BR1"],
            BH=cfg["BH"],
            BR2=cfg["BR2"],
            gelu_approx=gelu_approx,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    cfg = _pick_two_stage_cfg(B=B, L=L, R2=U2.shape[1])
    if two_stage_cfg:
        cfg.update({k: int(v) for k, v in two_stage_cfg.items()})
    return flashsvd_ffn_geglu_two_stage(
        P,
        V1,
        U2,
        V2,
        b1,
        b2,
        BL=cfg["BL"],
        BD=cfg["BD"],
        BR1=cfg["BR1"],
        BR2=cfg["BR2"],
        gelu_approx=gelu_approx,
        store_s_fp32=store_s_fp32,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def flashsvd_ffn_geglu_configured(
    P: torch.Tensor,
    V1: torch.Tensor,
    U2: torch.Tensor,
    V2: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    BL: int = 64,
    BD: int = 128,
    BR1: int = 64,
    BR2: int = 128,
    gelu_approx: str = "tanh",
    store_s_fp32: bool = False,
    num_warps: int = 8,
    num_stages: int = 2,
    *,
    kernel_variant: str = "two_stage",
    BH: int = 128,
    precomputed_g: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Manual, pinned launch. Keeps compatibility with previous API."""
    kv = kernel_variant.strip().lower()
    if kv in {"preg", "preg_fused", "fused_preg", "preg-fused"}:
        G = precomputed_g if precomputed_g is not None else _get_precomputed_ffn_g(U2, V2)
        return flashsvd_ffn_geglu_fused_preg(
            P,
            V1,
            G,
            b1,
            b2,
            BL=BL,
            BD=BD,
            BR1=BR1,
            BH=BH,
            gelu_approx=gelu_approx,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    if kv in {"fused", "phase12"}:
        return flashsvd_ffn_geglu_fused(
            P,
            V1,
            U2,
            V2,
            b1,
            b2,
            BL=BL,
            BD=BD,
            BR1=BR1,
            BH=BH,
            BR2=BR2,
            gelu_approx=gelu_approx,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return flashsvd_ffn_geglu_two_stage(
        P,
        V1,
        U2,
        V2,
        b1,
        b2,
        BL=BL,
        BD=BD,
        BR1=BR1,
        BR2=BR2,
        gelu_approx=gelu_approx,
        store_s_fp32=store_s_fp32,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# Convenience alias used by some local scripts.
def flashsvd_ffn_geglu(*args, **kwargs):
    return flashsvd_ffn_geglu_autotuned(*args, **kwargs)


def _pt_baseline(P, V1, U2, V2, b1, b2, gelu_approx="tanh"):
    z = P.matmul(V1) + b1.view(1, 1, -1)
    zu, zv = z.split(z.shape[-1] // 2, dim=-1)
    h = F.gelu(zu, approximate=gelu_approx) * zv
    s = h.matmul(U2)
    y = s.matmul(V2) + b2.view(1, 1, -1)
    return y


if __name__ == "__main__":
    # quick smoke test
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(0)
    device = "cuda"

    B, L, H = 2, 128, 768
    D = 1152
    R1, R2 = 192, 192
    dtype = torch.bfloat16

    P = torch.randn(B, L, R1, device=device, dtype=dtype)
    V1 = torch.randn(R1, 2 * D, device=device, dtype=dtype) * 0.02
    U2 = torch.randn(D, R2, device=device, dtype=dtype) * 0.02
    V2 = torch.randn(R2, H, device=device, dtype=dtype) * 0.02
    b1 = torch.zeros(2 * D, device=device, dtype=dtype)
    b2 = torch.zeros(H, device=device, dtype=dtype)

    y_fast = flashsvd_ffn_geglu_autotuned(P, V1, U2, V2, b1, b2, kernel_variant="fused")
    y_ref = _pt_baseline(P, V1, U2, V2, b1, b2)

    diff = (y_fast - y_ref).float()
    rel = diff.norm() / (y_ref.float().norm() + 1e-12)
    print("[flashsvd_geglu_v1.5] rel_err=%.3e max_abs=%.3e" % (rel.item(), diff.abs().max().item()))
