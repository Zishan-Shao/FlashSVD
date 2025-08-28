#!/usr/bin/env python3
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# -----------------------------
# Triton kernel (SwiGLU in rank space)
# -----------------------------
@triton.jit
def fused_ffn_phase1_swiglu(
    P_ptr, V1_ptr, U2_ptr, S_ptr, b1_ptr,
    B, L, D, R1, R2,
    sP_b, sP_l, sP_r1,
    sV1_r1, sV1_d,
    sU2_d, sU2_r2,
    sb1,
    sS_b, sS_l, sS_r2,
    BL: tl.constexpr, BD: tl.constexpr, BR1: tl.constexpr, BR2: tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_l  = tl.program_id(1)
    pid_r2 = tl.program_id(2)

    offs_l  = pid_l * BL + tl.arange(0, BL)
    offs_r2 = pid_r2 * BR2 + tl.arange(0, BR2)

    Tu_acc = tl.zeros((BL, BD), dtype=tl.float32)
    Tv_acc = tl.zeros((BL, BD), dtype=tl.float32)
    acc    = tl.zeros((BL, BR2), dtype=tl.float32)

    for d0 in range(0, D, BD):
        d   = d0 + tl.arange(0, BD)
        m_d = d < D

        Tu_acc *= 0.0
        Tv_acc *= 0.0

        for r1_0 in range(0, R1, BR1):
            r1   = r1_0 + tl.arange(0, BR1)
            m_r1 = r1 < R1

            P_blk = tl.load(
                P_ptr + pid_b * sP_b + offs_l[:, None]*sP_l + r1[None, :]*sP_r1,
                mask=(offs_l[:, None] < L) & m_r1[None, :],
                other=0.0
            )

            V1u_blk = tl.load(
                V1_ptr + r1[:, None]*sV1_r1 + d[None, :]*sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0
            )

            V1v_blk = tl.load(
                V1_ptr + r1[:, None]*sV1_r1 + (d[None, :] + D)*sV1_d,
                mask=m_r1[:, None] & m_d[None, :],
                other=0.0
            )

            Tu_acc += tl.dot(P_blk, V1u_blk)
            Tv_acc += tl.dot(P_blk, V1v_blk)

        b1u = tl.load(b1_ptr + d*sb1,        mask=m_d, other=0.0).to(tl.float32)
        b1v = tl.load(b1_ptr + (d + D)*sb1,  mask=m_d, other=0.0).to(tl.float32)
        Tu  = Tu_acc + b1u[None, :]
        Tv  = Tv_acc + b1v[None, :]

        # SwiGLU: silu(Tu) * Tv, where silu(z) = z * sigmoid(z)
        Hu = Tu * (1.0 / (1.0 + tl.exp(-Tu)))
        H  = Hu * Tv

        U2_blk = tl.load(
            U2_ptr + d[:, None]*sU2_d + offs_r2[None, :]*sU2_r2,
            mask=m_d[:, None] & (offs_r2[None, :] < R2),
            other=0.0
        ).to(tl.float32)
        acc += tl.dot(H, U2_blk)

    mask = (offs_l[:, None] < L) & (offs_r2[None, :] < R2)
    tl.store(
        S_ptr + pid_b*sS_b + offs_l[:, None]*sS_l + offs_r2[None, :]*sS_r2,
        acc, mask=mask
    )


# -----------------------------
# Wrapper
# -----------------------------
def flashsvd_ffn_swiglu(
    P, V1, U2, V2, b1, b2,
    BL=64, BD=64, BR1=64, BR2=64,
    *, store_s_fp32: bool = False,
):
    assert P.is_cuda and V1.is_cuda and U2.is_cuda and V2.is_cuda
    B, L, R1 = P.shape
    R1_v1, twoD = V1.shape
    D = twoD // 2
    assert R1_v1 == R1 and twoD == 2 * D
    D_u2, R2 = U2.shape
    assert D_u2 == D
    R2_v2, H = V2.shape
    assert R2_v2 == R2
    assert b1.shape[0] == 2*D and b2.shape[0] == H

    S_dtype = torch.float32 if store_s_fp32 else P.dtype
    S = torch.empty((B, L, R2), device=P.device, dtype=S_dtype)

    strides = dict(
        sP_b=P.stride(0), sP_l=P.stride(1), sP_r1=P.stride(2),
        sV1_r1=V1.stride(0), sV1_d=V1.stride(1),
        sU2_d=U2.stride(0), sU2_r2=U2.stride(1),
        sb1=b1.stride(0),
        sS_b=S.stride(0), sS_l=S.stride(1), sS_r2=S.stride(2),
    )

    grid = (B, triton.cdiv(L, BL), triton.cdiv(R2, BR2))
    fused_ffn_phase1_swiglu[grid](
        P, V1, U2, S, b1,
        B, L, D, R1, R2,
        *strides.values(),
        BL, BD, BR1, BR2,
    )

    Y = S.matmul(V2)
    Y = Y + b2.view(1, 1, -1)
    return Y


# -----------------------------
# PyTorch reference
# -----------------------------
def _pt_baseline_swiglu(P, V1, U2, V2, b1, b2):
    Z  = P.matmul(V1) + b1.view(1, 1, -1)
    Zu, Zv = Z.split(Z.shape[-1] // 2, dim=-1)
    H  = F.silu(Zu) * Zv
    S  = H.matmul(U2)
    Y  = S.matmul(V2) + b2.view(1, 1, -1)
    return Y


