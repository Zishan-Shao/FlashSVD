#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _reconstruct_qkv_token_kernel(
    Pq_ptr, Pk_ptr, Pv_ptr,
    Vq_ptr, Vk_ptr, Vv_ptr,
    bq_ptr, bk_ptr, bv_ptr,
    Q_ptr, K_ptr, V_ptr,
    B, H, Hk, Dh, R,
    sPq_b, sPq_h, sPq_r,
    sPk_b, sPk_h, sPk_r,
    sPv_b, sPv_h, sPv_r,
    sVq_h, sVq_r, sVq_d,
    sVk_h, sVk_r, sVk_d,
    sVv_h, sVv_r, sVv_d,
    sbq_h, sbq_d,
    sbk_h, sbk_d,
    sbv_h, sbv_d,
    sQ_b, sQ_h, sQ_d,
    sK_b, sK_h, sK_d,
    sV_b, sV_h, sV_d,
    BD: tl.constexpr,
    BR: tl.constexpr,
    HAS_BQ: tl.constexpr,
    HAS_BK: tl.constexpr,
    HAS_BV: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_d = offs_d < Dh
    in_dtype = tl.float32 if USE_FP32 else (tl.bfloat16 if USE_BF16 else tl.float16)

    if pid_h < H:
        acc_q = tl.zeros((BD,), dtype=tl.float32)
        for r0 in range(0, R, BR):
            offs_r = r0 + tl.arange(0, BR)
            mask_r = offs_r < R
            pq = tl.load(
                Pq_ptr + pid_b * sPq_b + pid_h * sPq_h + offs_r * sPq_r,
                mask=mask_r,
                other=0.0,
            ).to(in_dtype)
            vq = tl.load(
                Vq_ptr + pid_h * sVq_h + offs_r[:, None] * sVq_r + offs_d[None, :] * sVq_d,
                mask=mask_r[:, None] & mask_d[None, :],
                other=0.0,
            ).to(in_dtype)
            acc_q += tl.sum(pq[:, None].to(tl.float32) * vq.to(tl.float32), axis=0)
        if HAS_BQ:
            acc_q += tl.load(
                bq_ptr + pid_h * sbq_h + offs_d * sbq_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            Q_ptr + pid_b * sQ_b + pid_h * sQ_h + offs_d * sQ_d,
            acc_q.to(in_dtype),
            mask=mask_d,
        )

    if pid_h < Hk:
        acc_k = tl.zeros((BD,), dtype=tl.float32)
        acc_v = tl.zeros((BD,), dtype=tl.float32)
        for r0 in range(0, R, BR):
            offs_r = r0 + tl.arange(0, BR)
            mask_r = offs_r < R
            pk = tl.load(
                Pk_ptr + pid_b * sPk_b + pid_h * sPk_h + offs_r * sPk_r,
                mask=mask_r,
                other=0.0,
            ).to(in_dtype)
            pv = tl.load(
                Pv_ptr + pid_b * sPv_b + pid_h * sPv_h + offs_r * sPv_r,
                mask=mask_r,
                other=0.0,
            ).to(in_dtype)
            vk = tl.load(
                Vk_ptr + pid_h * sVk_h + offs_r[:, None] * sVk_r + offs_d[None, :] * sVk_d,
                mask=mask_r[:, None] & mask_d[None, :],
                other=0.0,
            ).to(in_dtype)
            vv = tl.load(
                Vv_ptr + pid_h * sVv_h + offs_r[:, None] * sVv_r + offs_d[None, :] * sVv_d,
                mask=mask_r[:, None] & mask_d[None, :],
                other=0.0,
            ).to(in_dtype)
            acc_k += tl.sum(pk[:, None].to(tl.float32) * vk.to(tl.float32), axis=0)
            acc_v += tl.sum(pv[:, None].to(tl.float32) * vv.to(tl.float32), axis=0)
        if HAS_BK:
            acc_k += tl.load(
                bk_ptr + pid_h * sbk_h + offs_d * sbk_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
        if HAS_BV:
            acc_v += tl.load(
                bv_ptr + pid_h * sbv_h + offs_d * sbv_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            K_ptr + pid_b * sK_b + pid_h * sK_h + offs_d * sK_d,
            acc_k.to(in_dtype),
            mask=mask_d,
        )
        tl.store(
            V_ptr + pid_b * sV_b + pid_h * sV_h + offs_d * sV_d,
            acc_v.to(in_dtype),
            mask=mask_d,
        )


@triton.jit
def _reconstruct_qkv_token_shared_kernel(
    Pq_ptr, Pk_ptr, Pv_ptr,
    Vq_ptr, Vk_ptr, Vv_ptr,
    bq_ptr, bk_ptr, bv_ptr,
    Q_ptr, K_ptr, V_ptr,
    B, H, Hk, Dh, R,
    sPq_b, sPq_r,
    sPk_b, sPk_r,
    sPv_b, sPv_r,
    sVq_h, sVq_r, sVq_d,
    sVk_h, sVk_r, sVk_d,
    sVv_h, sVv_r, sVv_d,
    sbq_h, sbq_d,
    sbk_h, sbk_d,
    sbv_h, sbv_d,
    sQ_b, sQ_h, sQ_d,
    sK_b, sK_h, sK_d,
    sV_b, sV_h, sV_d,
    BD: tl.constexpr,
    BR: tl.constexpr,
    HAS_BQ: tl.constexpr,
    HAS_BK: tl.constexpr,
    HAS_BV: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_d = offs_d < Dh
    in_dtype = tl.float32 if USE_FP32 else (tl.bfloat16 if USE_BF16 else tl.float16)

    if pid_h < H:
        acc_q = tl.zeros((BD,), dtype=tl.float32)
        for r0 in range(0, R, BR):
            offs_r = r0 + tl.arange(0, BR)
            mask_r = offs_r < R
            pq = tl.load(
                Pq_ptr + pid_b * sPq_b + offs_r * sPq_r,
                mask=mask_r,
                other=0.0,
            ).to(in_dtype)
            vq = tl.load(
                Vq_ptr + pid_h * sVq_h + offs_r[:, None] * sVq_r + offs_d[None, :] * sVq_d,
                mask=mask_r[:, None] & mask_d[None, :],
                other=0.0,
            ).to(in_dtype)
            acc_q += tl.sum(pq[:, None].to(tl.float32) * vq.to(tl.float32), axis=0)
        if HAS_BQ:
            acc_q += tl.load(
                bq_ptr + pid_h * sbq_h + offs_d * sbq_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            Q_ptr + pid_b * sQ_b + pid_h * sQ_h + offs_d * sQ_d,
            acc_q.to(in_dtype),
            mask=mask_d,
        )

    if pid_h < Hk:
        acc_k = tl.zeros((BD,), dtype=tl.float32)
        acc_v = tl.zeros((BD,), dtype=tl.float32)
        for r0 in range(0, R, BR):
            offs_r = r0 + tl.arange(0, BR)
            mask_r = offs_r < R
            pk = tl.load(
                Pk_ptr + pid_b * sPk_b + offs_r * sPk_r,
                mask=mask_r,
                other=0.0,
            ).to(in_dtype)
            pv = tl.load(
                Pv_ptr + pid_b * sPv_b + offs_r * sPv_r,
                mask=mask_r,
                other=0.0,
            ).to(in_dtype)
            vk = tl.load(
                Vk_ptr + pid_h * sVk_h + offs_r[:, None] * sVk_r + offs_d[None, :] * sVk_d,
                mask=mask_r[:, None] & mask_d[None, :],
                other=0.0,
            ).to(in_dtype)
            vv = tl.load(
                Vv_ptr + pid_h * sVv_h + offs_r[:, None] * sVv_r + offs_d[None, :] * sVv_d,
                mask=mask_r[:, None] & mask_d[None, :],
                other=0.0,
            ).to(in_dtype)
            acc_k += tl.sum(pk[:, None].to(tl.float32) * vk.to(tl.float32), axis=0)
            acc_v += tl.sum(pv[:, None].to(tl.float32) * vv.to(tl.float32), axis=0)
        if HAS_BK:
            acc_k += tl.load(
                bk_ptr + pid_h * sbk_h + offs_d * sbk_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
        if HAS_BV:
            acc_v += tl.load(
                bv_ptr + pid_h * sbv_h + offs_d * sbv_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            K_ptr + pid_b * sK_b + pid_h * sK_h + offs_d * sK_d,
            acc_k.to(in_dtype),
            mask=mask_d,
        )
        tl.store(
            V_ptr + pid_b * sV_b + pid_h * sV_h + offs_d * sV_d,
            acc_v.to(in_dtype),
            mask=mask_d,
        )


def reconstruct_qkv_token(
    Pq: torch.Tensor,
    Pk: torch.Tensor,
    Pv: torch.Tensor,
    Vq: torch.Tensor,
    Vk: torch.Tensor,
    Vv: torch.Tensor,
    *,
    bq: torch.Tensor | None = None,
    bk: torch.Tensor | None = None,
    bv: torch.Tensor | None = None,
    BD: int = 64,
    BR: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode-only token reconstruction.

    Inputs:
      Pq: [B, H, R]
      Pk: [B, Hk, R]
      Pv: [B, Hk, R]
      Vq: [H,  R, Dh]
      Vk: [Hk, R, Dh]
      Vv: [Hk, R, Dh]

    Returns:
      Q: [B, H, Dh]
      K: [B, Hk, Dh]
      V: [B, Hk, Dh]
    """
    if not (Pq.is_cuda and Pk.is_cuda and Pv.is_cuda and Vq.is_cuda and Vk.is_cuda and Vv.is_cuda):
        raise ValueError("reconstruct_qkv_token expects all tensors on CUDA.")
    if Pq.ndim != 3 or Pk.ndim != 3 or Pv.ndim != 3:
        raise ValueError(
            f"Expected Pq/Pk/Pv to be 3D [B,H|Hk,R], got {tuple(Pq.shape)}/{tuple(Pk.shape)}/{tuple(Pv.shape)}"
        )
    if Vq.ndim != 3 or Vk.ndim != 3 or Vv.ndim != 3:
        raise ValueError(
            f"Expected Vq/Vk/Vv to be 3D [H|Hk,R,Dh], got {tuple(Vq.shape)}/{tuple(Vk.shape)}/{tuple(Vv.shape)}"
        )

    B, H, R = Pq.shape
    Bk, Hk, Rk = Pk.shape
    Bv, Hkv, Rv = Pv.shape
    if Bk != B or Bv != B or Hkv != Hk or Rk != R or Rv != R:
        raise ValueError("Mismatched token-factor shapes.")
    if Vq.shape[0] != H or Vk.shape[0] != Hk or Vv.shape[0] != Hk:
        raise ValueError("Mismatched basis head dimensions.")
    if Vq.shape[1] != R or Vk.shape[1] != R or Vv.shape[1] != R:
        raise ValueError("Mismatched rank dimension between P* and V*.")
    Dh = int(Vq.shape[2])
    if Vk.shape[2] != Dh or Vv.shape[2] != Dh:
        raise ValueError("All bases must share the same head_dim.")

    if bq is not None and tuple(bq.shape) != (H, Dh):
        raise ValueError(f"Expected bq shape {(H, Dh)}, got {tuple(bq.shape)}")
    if bk is not None and tuple(bk.shape) != (Hk, Dh):
        raise ValueError(f"Expected bk shape {(Hk, Dh)}, got {tuple(bk.shape)}")
    if bv is not None and tuple(bv.shape) != (Hk, Dh):
        raise ValueError(f"Expected bv shape {(Hk, Dh)}, got {tuple(bv.shape)}")

    Q = torch.empty((B, H, Dh), device=Pq.device, dtype=Pq.dtype)
    K = torch.empty((B, Hk, Dh), device=Pq.device, dtype=Pq.dtype)
    V = torch.empty((B, Hk, Dh), device=Pq.device, dtype=Pq.dtype)

    use_fp32 = int(Pq.dtype == torch.float32)
    use_bf16 = int(Pq.dtype == torch.bfloat16)
    grid = (B, max(H, Hk), triton.cdiv(Dh, BD))

    _reconstruct_qkv_token_kernel[grid](
        Pq, Pk, Pv,
        Vq, Vk, Vv,
        bq if bq is not None else Q,
        bk if bk is not None else K,
        bv if bv is not None else V,
        Q, K, V,
        B, H, Hk, Dh, R,
        *Pq.stride(),
        *Pk.stride(),
        *Pv.stride(),
        *Vq.stride(),
        *Vk.stride(),
        *Vv.stride(),
        *(bq.stride() if bq is not None else (0, 0)),
        *(bk.stride() if bk is not None else (0, 0)),
        *(bv.stride() if bv is not None else (0, 0)),
        *Q.stride(),
        *K.stride(),
        *V.stride(),
        BD=BD,
        BR=BR,
        HAS_BQ=bq is not None,
        HAS_BK=bk is not None,
        HAS_BV=bv is not None,
        USE_BF16=use_bf16,
        USE_FP32=use_fp32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return Q, K, V


def reconstruct_qkv_token_shared(
    Pq: torch.Tensor,
    Pk: torch.Tensor,
    Pv: torch.Tensor,
    Vq: torch.Tensor,
    Vk: torch.Tensor,
    Vv: torch.Tensor,
    *,
    bq: torch.Tensor | None = None,
    bk: torch.Tensor | None = None,
    bv: torch.Tensor | None = None,
    BD: int = 64,
    BR: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode-only token reconstruction for shared global-rank factors.

    Inputs:
      Pq/Pk/Pv: [B, R]
      Vq: [H,  R, Dh]
      Vk: [Hk, R, Dh]
      Vv: [Hk, R, Dh]
    """
    if not (Pq.is_cuda and Pk.is_cuda and Pv.is_cuda and Vq.is_cuda and Vk.is_cuda and Vv.is_cuda):
        raise ValueError("reconstruct_qkv_token_shared expects all tensors on CUDA.")
    if Pq.ndim != 2 or Pk.ndim != 2 or Pv.ndim != 2:
        raise ValueError(
            f"Expected shared Pq/Pk/Pv to be 2D [B,R], got {tuple(Pq.shape)}/{tuple(Pk.shape)}/{tuple(Pv.shape)}"
        )
    if Vq.ndim != 3 or Vk.ndim != 3 or Vv.ndim != 3:
        raise ValueError(
            f"Expected Vq/Vk/Vv to be 3D [H|Hk,R,Dh], got {tuple(Vq.shape)}/{tuple(Vk.shape)}/{tuple(Vv.shape)}"
        )

    B, R = Pq.shape
    Bk, Rk = Pk.shape
    Bv, Rv = Pv.shape
    if Bk != B or Bv != B or Rk != R or Rv != R:
        raise ValueError("Mismatched shared factor shapes.")
    H = int(Vq.shape[0])
    Hk = int(Vk.shape[0])
    Dh = int(Vq.shape[2])
    if Vk.shape[1] != R or Vv.shape[1] != R or Vk.shape[2] != Dh or Vv.shape[2] != Dh:
        raise ValueError("Mismatched basis shapes for shared factor reconstruction.")

    if bq is not None and tuple(bq.shape) != (H, Dh):
        raise ValueError(f"Expected bq shape {(H, Dh)}, got {tuple(bq.shape)}")
    if bk is not None and tuple(bk.shape) != (Hk, Dh):
        raise ValueError(f"Expected bk shape {(Hk, Dh)}, got {tuple(bk.shape)}")
    if bv is not None and tuple(bv.shape) != (Hk, Dh):
        raise ValueError(f"Expected bv shape {(Hk, Dh)}, got {tuple(bv.shape)}")

    Q = torch.empty((B, H, Dh), device=Pq.device, dtype=Pq.dtype)
    K = torch.empty((B, Hk, Dh), device=Pq.device, dtype=Pq.dtype)
    V = torch.empty((B, Hk, Dh), device=Pq.device, dtype=Pq.dtype)

    use_fp32 = int(Pq.dtype == torch.float32)
    use_bf16 = int(Pq.dtype == torch.bfloat16)
    grid = (B, max(H, Hk), triton.cdiv(Dh, BD))

    _reconstruct_qkv_token_shared_kernel[grid](
        Pq, Pk, Pv,
        Vq, Vk, Vv,
        bq if bq is not None else Q,
        bk if bk is not None else K,
        bv if bv is not None else V,
        Q, K, V,
        B, H, Hk, Dh, R,
        *Pq.stride(),
        *Pk.stride(),
        *Pv.stride(),
        *Vq.stride(),
        *Vk.stride(),
        *Vv.stride(),
        *(bq.stride() if bq is not None else (0, 0)),
        *(bk.stride() if bk is not None else (0, 0)),
        *(bv.stride() if bv is not None else (0, 0)),
        *Q.stride(),
        *K.stride(),
        *V.stride(),
        BD=BD,
        BR=BR,
        HAS_BQ=bq is not None,
        HAS_BK=bk is not None,
        HAS_BV=bv is not None,
        USE_BF16=use_bf16,
        USE_FP32=use_fp32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return Q, K, V


def pack_qkv_shared_bases(
    Vq: torch.Tensor,
    Vk: torch.Tensor,
    Vv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepack shared-rank bases to GEMM-friendly [R, H*Dh] / [R, Hk*Dh] layout.

    Intended for decode loops where Vq/Vk/Vv are reused across many tokens.
    """
    if not (Vq.is_cuda and Vk.is_cuda and Vv.is_cuda):
        raise ValueError("pack_qkv_shared_bases expects CUDA tensors.")
    if Vq.ndim != 3 or Vk.ndim != 3 or Vv.ndim != 3:
        raise ValueError(
            f"Expected Vq/Vk/Vv to be 3D [H|Hk,R,Dh], got {tuple(Vq.shape)}/{tuple(Vk.shape)}/{tuple(Vv.shape)}"
        )
    R = int(Vq.shape[1])
    Dh = int(Vq.shape[2])
    H = int(Vq.shape[0])
    Hk = int(Vk.shape[0])
    if Vk.shape[1] != R or Vv.shape[1] != R or Vk.shape[2] != Dh or Vv.shape[2] != Dh:
        raise ValueError("Mismatched basis shapes for shared basis packing.")

    Vq_flat = Vq.permute(1, 0, 2).reshape(R, H * Dh).contiguous()
    Vk_flat = Vk.permute(1, 0, 2).reshape(R, Hk * Dh).contiguous()
    Vv_flat = Vv.permute(1, 0, 2).reshape(R, Hk * Dh).contiguous()
    return Vq_flat, Vk_flat, Vv_flat


def reconstruct_qkv_token_shared_prepacked(
    Pq: torch.Tensor,
    Pk: torch.Tensor,
    Pv: torch.Tensor,
    Vq_flat: torch.Tensor,
    Vk_flat: torch.Tensor,
    Vv_flat: torch.Tensor,
    *,
    H: int,
    Hk: int,
    Dh: int,
    bq: torch.Tensor | None = None,
    bk: torch.Tensor | None = None,
    bv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared-rank token reconstruction using GEMM/GEMV-friendly prepacked bases.

    This trades the per-head Triton reduction for three small vendor matmuls:
      Q = Pq @ Vq_flat
      K = Pk @ Vk_flat
      V = Pv @ Vv_flat
    """
    if not (Pq.is_cuda and Pk.is_cuda and Pv.is_cuda and Vq_flat.is_cuda and Vk_flat.is_cuda and Vv_flat.is_cuda):
        raise ValueError("reconstruct_qkv_token_shared_prepacked expects all tensors on CUDA.")
    if Pq.ndim != 2 or Pk.ndim != 2 or Pv.ndim != 2:
        raise ValueError(
            f"Expected shared Pq/Pk/Pv to be 2D [B,R], got {tuple(Pq.shape)}/{tuple(Pk.shape)}/{tuple(Pv.shape)}"
        )
    if Vq_flat.ndim != 2 or Vk_flat.ndim != 2 or Vv_flat.ndim != 2:
        raise ValueError(
            f"Expected prepacked bases to be 2D [R,N], got {tuple(Vq_flat.shape)}/{tuple(Vk_flat.shape)}/{tuple(Vv_flat.shape)}"
        )

    B, R = Pq.shape
    if Pk.shape != (B, R) or Pv.shape != (B, R):
        raise ValueError("Mismatched shared factor shapes.")
    if Vq_flat.shape != (R, H * Dh):
        raise ValueError(f"Expected Vq_flat shape {(R, H * Dh)}, got {tuple(Vq_flat.shape)}")
    if Vk_flat.shape != (R, Hk * Dh):
        raise ValueError(f"Expected Vk_flat shape {(R, Hk * Dh)}, got {tuple(Vk_flat.shape)}")
    if Vv_flat.shape != (R, Hk * Dh):
        raise ValueError(f"Expected Vv_flat shape {(R, Hk * Dh)}, got {tuple(Vv_flat.shape)}")

    q_flat = torch.matmul(Pq, Vq_flat)
    k_flat = torch.matmul(Pk, Vk_flat)
    v_flat = torch.matmul(Pv, Vv_flat)

    Q = q_flat.reshape(B, H, Dh)
    K = k_flat.reshape(B, Hk, Dh)
    V = v_flat.reshape(B, Hk, Dh)

    if bq is not None:
        if tuple(bq.shape) != (H, Dh):
            raise ValueError(f"Expected bq shape {(H, Dh)}, got {tuple(bq.shape)}")
        Q = Q + bq.unsqueeze(0)
    if bk is not None:
        if tuple(bk.shape) != (Hk, Dh):
            raise ValueError(f"Expected bk shape {(Hk, Dh)}, got {tuple(bk.shape)}")
        K = K + bk.unsqueeze(0)
    if bv is not None:
        if tuple(bv.shape) != (Hk, Dh):
            raise ValueError(f"Expected bv shape {(Hk, Dh)}, got {tuple(bv.shape)}")
        V = V + bv.unsqueeze(0)

    return Q.contiguous(), K.contiguous(), V.contiguous()
