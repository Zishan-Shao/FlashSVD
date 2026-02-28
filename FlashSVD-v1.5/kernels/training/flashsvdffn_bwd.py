#!/usr/bin/env python3
# flashsvdffn_bwd.py — FlashSVD FFN forward + backward (Triton 3.3.1)
#   y = GELU(x @ (P1 V1) + b1) @ (P2 V2) + b2
#
# Two backward kernels (stage-2 then stage-1), each with autotune + fixed variants.
# Relative Frobenius checks vs dense PyTorch autograd are included.

import math, time
import torch
import triton
import triton.language as tl

# Tunables / defaults
BLOCK_M = 32     # rows (sequence/tokens)
BLOCK_N = 32     # columns (D_h or D_out tiles)
BLOCK_K = 32     # input features tile (D_in)
BLOCK_R = 32     # rank tile (for both R1 and R2)

def _contig(t): return t.contiguous() if not t.is_contiguous() else t

# -----------------------------------
# Forward kernel
#   Streams over M (rows) and N (output cols), loops over Dh tiles,
#   rank tiles (R1,R2), and K tiles to compute:
#     Z = X P1 V1 + b1
#     H = GELU(Z)
#     Y = H P2 V2 + b2
# -----------------------------------
@triton.jit
def _ffn_fwd_kernel(
    # X: [B, M, Din]
    X_ptr, sXb, sXm, sXd,
    # W1: P1 [Din, R1], V1 [R1, Dh], b1 [Dh]
    P1_ptr, sP1k, sP1r,
    V1_ptr, sV1r, sV1d,
    b1_ptr, sB1d,
    # W2: P2 [Dh, R2], V2 [R2, Dout], b2 [Dout]
    P2_ptr, sP2d, sP2r,
    V2_ptr, sV2r, sV2o,
    b2_ptr, sB2o,
    # Out: [B, M, Dout]
    Out_ptr, sYb, sYm, sYo,
    # sizes
    B, M, Din, Dh, Dout, R1, R2,
    # meta
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)          # row block
    pid_b = tl.program_id(1)          # batch index
    pid_n = tl.program_id(2)          # output-col block

    off_b  = pid_b
    row0   = pid_m * BLOCK_M
    col0   = pid_n * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = (row0 + offs_m) < M
    mask_n = (col0 + offs_n) < Dout

    # Accumulator for Y tile [M_blk, N_blk]
    Y = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Constants for GELU (erf formulation)
    INV_SQRT2 = 0.7071067811865475

    # Loop over Dh tiles
    for dh0 in range(0, Dh, BLOCK_N):
        offs_h = tl.arange(0, BLOCK_N)
        mask_h = (dh0 + offs_h) < Dh

        # Compute Z tile: Z = X P1 V1 + b1  -> [M_blk, H_blk] ----
        Z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # over R1
        for r10 in range(0, R1, BLOCK_R):
            # S = X @ P1[:, r1_chunk]  -> [M_blk, R_blk]
            S = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
            for k0 in range(0, Din, BLOCK_K):
                X_ptrs = X_ptr + off_b*sXb + (row0 + offs_m)[:, None]*sXm + (k0 + offs_k)[None, :]*sXd
                P1_ptrs = P1_ptr + (k0 + offs_k)[:, None]*sP1k + (r10 + offs_r)[None, :]*sP1r
                X_sub  = tl.load(X_ptrs, mask=mask_m[:, None] & ((k0 + offs_k)[None, :] < Din), other=0.).to(tl.float32)
                P1_sub = tl.load(P1_ptrs, mask=((k0 + offs_k)[:, None] < Din) & ((r10 + offs_r)[None, :] < R1), other=0.).to(tl.float32)
                S += tl.dot(X_sub, P1_sub)  # [M_blk, R_blk]

            # V1 sub: [R_blk, H_blk]
            V1_ptrs = V1_ptr + (r10 + offs_r)[:, None]*sV1r + (dh0 + offs_h)[None, :]*sV1d
            V1_sub = tl.load(V1_ptrs, mask=((r10 + offs_r)[:, None] < R1) & mask_h[None, :], other=0.).to(tl.float32)
            Z += tl.dot(S, V1_sub)  # [M_blk, H_blk]

        # + b1
        b1_ptrs = b1_ptr + (dh0 + offs_h) * sB1d
        b1_sub  = tl.load(b1_ptrs, mask=mask_h, other=0.).to(tl.float32)
        Z += b1_sub[None, :]

        # ---- GELU (exact via erf)
        # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        x = Z
        erf_val = tl.erf(x * INV_SQRT2)
        H = 0.5 * x * (1.0 + erf_val)

        # ---- Accumulate Y tile via W2 = P2 V2
        for r20 in range(0, R2, BLOCK_R):
            # T = H @ P2[dh0:dh0+H_blk, r2_chunk]  -> [M_blk, R_blk]
            P2_ptrs = P2_ptr + (dh0 + offs_h)[:, None]*sP2d + (r20 + offs_r)[None, :]*sP2r
            P2_sub  = tl.load(P2_ptrs, mask=mask_h[:, None] & ((r20 + offs_r)[None, :] < R2), other=0.).to(tl.float32)
            T = tl.dot(H, P2_sub)  # [M_blk, R_blk]

            # V2 sub: [R_blk, N_blk]
            V2_ptrs = V2_ptr + (r20 + offs_r)[:, None]*sV2r + (col0 + offs_n)[None, :]*sV2o
            V2_sub  = tl.load(V2_ptrs, mask=((r20 + offs_r)[:, None] < R2) & mask_n[None, :], other=0.).to(tl.float32)
            Y += tl.dot(T, V2_sub)

    # + b2
    b2_ptrs = b2_ptr + (col0 + offs_n) * sB2o
    b2_sub  = tl.load(b2_ptrs, mask=mask_n, other=0.).to(tl.float32)
    Y += b2_sub[None, :]

    # store
    Out_ptrs = Out_ptr + off_b*sYb + (row0 + offs_m)[:, None]*sYm + (col0 + offs_n)[None, :]*sYo
    tl.store(Out_ptrs, Y, mask=mask_m[:, None] & mask_n[None, :])

def flash_svd_ffn_forward(X, P1, V1, b1, P2, V2, b2,
                          *, block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K, block_r=BLOCK_R):
    """
    X  : [B, M, Din]
    P1 : [Din, R1], V1: [R1, Dh], b1: [Dh]
    P2 : [Dh, R2], V2: [R2, Dout], b2: [Dout]
    Returns Y: [B, M, Dout] (same dtype as X)
    """
    B, M, Din = X.shape
    R1, Dh    = V1.shape
    R2, Dout  = V2.shape
    assert P1.shape == (Din, R1) and P2.shape == (Dh, R2)
    assert b1.shape == (Dh,) and b2.shape == (Dout,)

    X, P1, V1, b1, P2, V2, b2 = map(_contig, (X, P1, V1, b1, P2, V2, b2))
    Y = torch.empty(B, M, Dout, device=X.device, dtype=torch.float32)

    grid = ((M + block_m - 1)//block_m, B, (Dout + block_n - 1)//block_n)
    _ffn_fwd_kernel[grid](
        X, *X.stride(),
        P1, *P1.stride(),
        V1, *V1.stride(),
        b1, b1.stride(0),
        P2, *P2.stride(),
        V2, *V2.stride(),
        b2, b2.stride(0),
        Y, *Y.stride(),
        B, M, Din, Dh, Dout, R1, R2,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=block_r
    )
    return Y.to(X.dtype)

# -----------------------------------
# Backward – Stage 2 (top): d(P2,V2,b2) and dZ buffer
# -----------------------------------
_BWD2_CONFIGS = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'BLOCK_R': 32}, num_warps=4, num_stages=2),
    #triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_R': 32}, num_warps=4, num_stages=1),
    #triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'BLOCK_R': 32}, num_warps=4, num_stages=1),
]

@triton.autotune(configs=_BWD2_CONFIGS, key=['M','Dh','Dout','Din'])
@triton.jit
def _ffn_bwd_stage2_auto(
    # X
    X_ptr, sXb, sXm, sXd,
    # W1
    P1_ptr, sP1k, sP1r,
    V1_ptr, sV1r, sV1d,
    b1_ptr, sB1d,
    # W2
    P2_ptr, sP2d, sP2r,
    V2_ptr, sV2r, sV2o,
    # dY and grads (top)
    dY_ptr, sdb, sdm, sdo,
    dP2_ptr, s_dP2d, s_dP2r,
    dV2_ptr, s_dV2r, s_dV2o,
    db2_ptr, s_db2o,
    # dZ buffer
    dZ_ptr, s_dZb, s_dZm, s_dZd,
    # sizes
    B, M, Din, Dh, Dout, R1, R2,
    # meta
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_n = tl.program_id(2)

    off_b = pid_b
    row0  = pid_m * BLOCK_M
    col0  = pid_n * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (row0 + offs_m) < M
    mask_o = (col0 + offs_n) < Dout

    # GELU constants
    SQRT_2_OVER_PI = 0.7978845608028654
    INV_SQRT2 = 0.7071067811865475

    # Load dY tile
    dY_ptrs = dY_ptr + off_b*sdb + (row0 + offs_m)[:,None]*sdm + (col0 + offs_n)[None,:]*sdo
    dY = tl.load(dY_ptrs, mask=mask_m[:,None] & mask_o[None,:], other=0.).to(tl.float32)

    # Accumulate db2 (sum over rows)
    db2_vec = tl.sum(dY, axis=0)  # [N_blk]
    db2_ptrs = db2_ptr + (col0 + offs_n)*s_db2o
    tl.atomic_add(db2_ptrs, db2_vec, mask=mask_o)

    # dZ buffer accumulation: we need dH first aggregated over all output cols.
    # We'll recompute Z,H per Dh tile and accumulate dH->dZ via GELU' into dZ buffer.
    for dh0 in range(0, Dh, BLOCK_N):
        offs_h = tl.arange(0, BLOCK_N)
        mask_h = (dh0 + offs_h) < Dh

        # ---- Recompute Z ----
        Z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for r10 in range(0, R1, BLOCK_R):
            S = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
            for k0 in range(0, Din, BLOCK_K):
                X_ptrs = X_ptr + off_b*sXb + (row0+offs_m)[:,None]*sXm + (k0+offs_k)[None,:]*sXd
                P1_ptrs = P1_ptr + (k0+offs_k)[:,None]*sP1k + (r10+offs_r)[None,:]*sP1r
                X_sub  = tl.load(X_ptrs, mask=mask_m[:,None] & ((k0+offs_k)[None,:] < Din), other=0.).to(tl.float32)
                P1_sub = tl.load(P1_ptrs, mask=((k0+offs_k)[:,None] < Din) & ((r10+offs_r)[None,:] < R1), other=0.).to(tl.float32)
                S += tl.dot(X_sub, P1_sub)
            V1_ptrs = V1_ptr + (r10+offs_r)[:,None]*sV1r + (dh0+offs_h)[None,:]*sV1d
            V1_sub  = tl.load(V1_ptrs, mask=((r10+offs_r)[:,None] < R1) & mask_h[None,:], other=0.).to(tl.float32)
            Z += tl.dot(S, V1_sub)
        b1_ptrs = b1_ptr + (dh0 + offs_h)*sB1d
        b1_sub  = tl.load(b1_ptrs, mask=mask_h, other=0.).to(tl.float32)
        Z += b1_sub[None, :]

        # H and dH accumulator for this H tile
        x = Z
        erf_val = tl.erf(x * INV_SQRT2)
        H = 0.5 * x * (1.0 + erf_val)
        dH = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ---- Accumulate top param grads and dH using dY
        for r20 in range(0, R2, BLOCK_R):
            # P2_sub: [H_blk, R_blk], V2_sub: [R_blk, N_blk]
            P2_ptrs = P2_ptr + (dh0+offs_h)[:,None]*sP2d + (r20+offs_r)[None,:]*sP2r
            P2_sub  = tl.load(P2_ptrs, mask=mask_h[:,None] & ((r20+offs_r)[None,:] < R2), other=0.).to(tl.float32)
            V2_ptrs = V2_ptr + (r20+offs_r)[:,None]*sV2r + (col0+offs_n)[None,:]*sV2o
            V2_sub  = tl.load(V2_ptrs, mask=((r20+offs_r)[:,None] < R2) & mask_o[None,:], other=0.).to(tl.float32)

            # t3 = dY @ V2^T      [M_blk, R_blk]
            t3 = tl.dot(dY, tl.trans(V2_sub))

            # dH += t3 @ P2^T     [M_blk, H_blk]
            dH += tl.dot(t3, tl.trans(P2_sub))

            # dV2 += (H @ P2) ^T @ dY
            T = tl.dot(H, P2_sub)  # [M_blk, R_blk]
            dV2_sub = tl.dot(tl.trans(T), dY)  # [R_blk, N_blk]
            dV2_ptrs = dV2_ptr + (r20+offs_r)[:,None]*s_dV2r + (col0+offs_n)[None,:]*s_dV2o
            tl.atomic_add(dV2_ptrs, dV2_sub, mask=((r20+offs_r)[:,None] < R2) & mask_o[None,:])

            # dP2 += H^T @ t3     [H_blk, R_blk]
            dP2_sub = tl.dot(tl.trans(H), t3)
            dP2_ptrs = dP2_ptr + (dh0+offs_h)[:,None]*s_dP2d + (r20+offs_r)[None,:]*s_dP2r
            tl.atomic_add(dP2_ptrs, dP2_sub, mask=mask_h[:,None] & ((r20+offs_r)[None,:] < R2))

        # ---- dZ = dH * gelu'(Z)
        # gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + 0.5*x*sqrt(2/pi)*exp(-0.5*x^2)
        exp_term = tl.exp(-0.5 * x * x)
        gelu_prime = 0.5 * (1.0 + erf_val) + 0.5 * x * SQRT_2_OVER_PI * exp_term
        dZ = dH * gelu_prime

        # accumulate into global dZ buffer
        dZ_ptrs = dZ_ptr + off_b*s_dZb + (row0 + offs_m)[:,None]*s_dZm + (dh0 + offs_h)[None,:]*s_dZd
        tl.atomic_add(dZ_ptrs, dZ, mask=mask_m[:,None] & mask_h[None,:])

@triton.jit
def _ffn_bwd_stage2_fixed(
    X_ptr, sXb, sXm, sXd,
    P1_ptr, sP1k, sP1r,
    V1_ptr, sV1r, sV1d,
    b1_ptr, sB1d,
    P2_ptr, sP2d, sP2r,
    V2_ptr, sV2r, sV2o,
    dY_ptr, sdb, sdm, sdo,
    dP2_ptr, s_dP2d, s_dP2r,
    dV2_ptr, s_dV2r, s_dV2o,
    db2_ptr, s_db2o,
    dZ_ptr, s_dZb, s_dZm, s_dZd,
    B, M, Din, Dh, Dout, R1, R2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_n = tl.program_id(2)

    off_b = pid_b
    row0  = pid_m * BLOCK_M
    col0  = pid_n * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (row0 + offs_m) < M
    mask_o = (col0 + offs_n) < Dout

    # GELU constants
    SQRT_2_OVER_PI = 0.7978845608028654
    INV_SQRT2 = 0.7071067811865475

    # Load dY tile
    dY_ptrs = dY_ptr + off_b*sdb + (row0 + offs_m)[:,None]*sdm + (col0 + offs_n)[None,:]*sdo
    dY = tl.load(dY_ptrs, mask=mask_m[:,None] & mask_o[None,:], other=0.).to(tl.float32)

    # Accumulate db2 (sum over rows)
    db2_vec = tl.sum(dY, axis=0)  # [N_blk]
    db2_ptrs = db2_ptr + (col0 + offs_n)*s_db2o
    tl.atomic_add(db2_ptrs, db2_vec, mask=mask_o)

    # dZ buffer accumulation: we need dH first aggregated over all output cols.
    # We'll recompute Z,H per Dh tile and accumulate dH->dZ via GELU' into dZ buffer.
    for dh0 in range(0, Dh, BLOCK_N):
        offs_h = tl.arange(0, BLOCK_N)
        mask_h = (dh0 + offs_h) < Dh

        # ---- Recompute Z ----
        Z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for r10 in range(0, R1, BLOCK_R):
            S = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
            for k0 in range(0, Din, BLOCK_K):
                X_ptrs = X_ptr + off_b*sXb + (row0+offs_m)[:,None]*sXm + (k0+offs_k)[None,:]*sXd
                P1_ptrs = P1_ptr + (k0+offs_k)[:,None]*sP1k + (r10+offs_r)[None,:]*sP1r
                X_sub  = tl.load(X_ptrs, mask=mask_m[:,None] & ((k0+offs_k)[None,:] < Din), other=0.).to(tl.float32)
                P1_sub = tl.load(P1_ptrs, mask=((k0+offs_k)[:,None] < Din) & ((r10+offs_r)[None,:] < R1), other=0.).to(tl.float32)
                S += tl.dot(X_sub, P1_sub)
            V1_ptrs = V1_ptr + (r10+offs_r)[:,None]*sV1r + (dh0+offs_h)[None,:]*sV1d
            V1_sub  = tl.load(V1_ptrs, mask=((r10+offs_r)[:,None] < R1) & mask_h[None,:], other=0.).to(tl.float32)
            Z += tl.dot(S, V1_sub)
        b1_ptrs = b1_ptr + (dh0 + offs_h)*sB1d
        b1_sub  = tl.load(b1_ptrs, mask=mask_h, other=0.).to(tl.float32)
        Z += b1_sub[None, :]

        # H and dH accumulator for this H tile
        x = Z
        erf_val = tl.erf(x * INV_SQRT2)
        H = 0.5 * x * (1.0 + erf_val)
        dH = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ---- Accumulate top param grads and dH using dY
        for r20 in range(0, R2, BLOCK_R):
            # P2_sub: [H_blk, R_blk], V2_sub: [R_blk, N_blk]
            P2_ptrs = P2_ptr + (dh0+offs_h)[:,None]*sP2d + (r20+offs_r)[None,:]*sP2r
            P2_sub  = tl.load(P2_ptrs, mask=mask_h[:,None] & ((r20+offs_r)[None,:] < R2), other=0.).to(tl.float32)
            V2_ptrs = V2_ptr + (r20+offs_r)[:,None]*sV2r + (col0+offs_n)[None,:]*sV2o
            V2_sub  = tl.load(V2_ptrs, mask=((r20+offs_r)[:,None] < R2) & mask_o[None,:], other=0.).to(tl.float32)

            # t3 = dY @ V2^T      [M_blk, R_blk]
            t3 = tl.dot(dY, tl.trans(V2_sub))

            # dH += t3 @ P2^T     [M_blk, H_blk]
            dH += tl.dot(t3, tl.trans(P2_sub))

            # dV2 += (H @ P2) ^T @ dY
            T = tl.dot(H, P2_sub)  # [M_blk, R_blk]
            dV2_sub = tl.dot(tl.trans(T), dY)  # [R_blk, N_blk]
            dV2_ptrs = dV2_ptr + (r20+offs_r)[:,None]*s_dV2r + (col0+offs_n)[None,:]*s_dV2o
            tl.atomic_add(dV2_ptrs, dV2_sub, mask=((r20+offs_r)[:,None] < R2) & mask_o[None,:])

            # dP2 += H^T @ t3     [H_blk, R_blk]
            dP2_sub = tl.dot(tl.trans(H), t3)
            dP2_ptrs = dP2_ptr + (dh0+offs_h)[:,None]*s_dP2d + (r20+offs_r)[None,:]*s_dP2r
            tl.atomic_add(dP2_ptrs, dP2_sub, mask=mask_h[:,None] & ((r20+offs_r)[None,:] < R2))

        # ---- dZ = dH * gelu'(Z)
        # gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + 0.5*x*sqrt(2/pi)*exp(-0.5*x^2)
        exp_term = tl.exp(-0.5 * x * x)
        gelu_prime = 0.5 * (1.0 + erf_val) + 0.5 * x * SQRT_2_OVER_PI * exp_term
        dZ = dH * gelu_prime

        # accumulate into global dZ buffer
        dZ_ptrs = dZ_ptr + off_b*s_dZb + (row0 + offs_m)[:,None]*s_dZm + (dh0 + offs_h)[None,:]*s_dZd
        tl.atomic_add(dZ_ptrs, dZ, mask=mask_m[:,None] & mask_h[None,:])

# -----------------------------------
# Backward – Stage 1 (bottom): d(P1,V1,b1,X) using dZ buffer
# -----------------------------------
_BWD1_CONFIGS = [
    triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'BLOCK_R': 32}, num_warps=4, num_stages=2),
    #triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'BLOCK_R': 32}, num_warps=4, num_stages=1),
    #triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'BLOCK_R': 32}, num_warps=4, num_stages=1),
]

@triton.autotune(configs=_BWD1_CONFIGS, key=['M','Dh','Din'])
@triton.jit
def _ffn_bwd_stage1_auto(
    # X
    X_ptr, sXb, sXm, sXd,
    # W1
    P1_ptr, sP1k, sP1r,
    V1_ptr, sV1r, sV1d,
    # dZ buffer [B,M,Dh]
    dZ_ptr, s_dZb, s_dZm, s_dZd,
    # grads to write: dP1 [Din,R1], dV1 [R1,Dh], db1 [Dh], dX [B,M,Din]
    dP1_ptr, s_dP1k, s_dP1r,
    dV1_ptr, s_dV1r, s_dV1d,
    db1_ptr, s_db1d,
    dX_ptr,  s_dXb,  s_dXm,  s_dXd,
    # sizes
    B, M, Din, Dh, R1,
    # meta
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)  # Dh tile

    off_b = pid_b
    row0  = pid_m * BLOCK_M
    dh0   = pid_h * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_N)

    mask_m = (row0 + offs_m) < M
    mask_h = (dh0 + offs_h) < Dh

    # Load dZ tile
    dZ_ptrs = dZ_ptr + off_b*s_dZb + (row0 + offs_m)[:,None]*s_dZm + (dh0 + offs_h)[None,:]*s_dZd
    dZ = tl.load(dZ_ptrs, mask=mask_m[:,None] & mask_h[None,:], other=0.).to(tl.float32)

    # db1 += sum_rows(dZ)
    db1_vec = tl.sum(dZ, axis=0)
    db1_ptrs = db1_ptr + (dh0 + offs_h)*s_db1d
    tl.atomic_add(db1_ptrs, db1_vec, mask=mask_h)

    # dV1 and dP1 need S = X @ P1 (for current r1-chunk) and
    # T = dZ @ V1^T (for current r1-chunk)
    for r10 in range(0, R1, BLOCK_R):
        # S = X @ P1[:, r1_chunk]  -> [M_blk, R_blk]
        S = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k0 in range(0, Din, BLOCK_K):
            X_ptrs = X_ptr + off_b*sXb + (row0+offs_m)[:,None]*sXm + (k0+offs_k)[None,:]*sXd
            P1_ptrs = P1_ptr + (k0+offs_k)[:,None]*sP1k + (r10+offs_r)[None,:]*sP1r
            X_sub  = tl.load(X_ptrs, mask=mask_m[:,None] & ((k0+offs_k)[None,:] < Din), other=0.).to(tl.float32)
            P1_sub = tl.load(P1_ptrs, mask=((k0+offs_k)[:,None] < Din) & ((r10+offs_r)[None,:] < R1), other=0.).to(tl.float32)
            S += tl.dot(X_sub, P1_sub)  # [M_blk, R_blk]

        # dV1 += S^T @ dZ  -> [R_blk, H_blk]
        dV1_sub = tl.dot(tl.trans(S), dZ)
        dV1_ptrs = dV1_ptr + (r10+offs_r)[:,None]*s_dV1r + (dh0+offs_h)[None,:]*s_dV1d
        tl.atomic_add(dV1_ptrs, dV1_sub, mask=((r10+offs_r)[:,None] < R1) & mask_h[None,:])

        # T = dZ @ V1[r1_chunk, H_blk]^T  -> [M_blk, R_blk]
        V1_ptrs = V1_ptr + (r10+offs_r)[:,None]*sV1r + (dh0+offs_h)[None,:]*sV1d
        V1_sub  = tl.load(V1_ptrs, mask=((r10+offs_r)[:,None] < R1) & mask_h[None,:], other=0.).to(tl.float32)
        T = tl.dot(dZ, tl.trans(V1_sub))  # [M_blk, R_blk]

        # dP1 += sum_k X[:,k]^T @ T  (block over K)
        for k0 in range(0, Din, BLOCK_K):
            X_ptrs = X_ptr + off_b*sXb + (row0+offs_m)[:,None]*sXm + (k0+offs_k)[None,:]*sXd
            X_sub = tl.load(X_ptrs, mask=mask_m[:,None] & ((k0+offs_k)[None,:] < Din), other=0.).to(tl.float32)
            dP1_sub = tl.dot(tl.trans(X_sub), T)  # [K_blk, R_blk]
            dP1_ptrs = dP1_ptr + (k0+offs_k)[:,None]*s_dP1k + (r10+offs_r)[None,:]*s_dP1r
            tl.atomic_add(dP1_ptrs, dP1_sub, mask=((k0+offs_k)[:,None] < Din) & ((r10+offs_r)[None,:] < R1))

        # dX += T @ P1[k, r1_chunk]^T  (block over K)
        for k0 in range(0, Din, BLOCK_K):
            P1_ptrs = P1_ptr + (k0+offs_k)[:,None]*sP1k + (r10+offs_r)[None,:]*sP1r
            P1_sub  = tl.load(P1_ptrs, mask=((k0+offs_k)[:,None] < Din) & ((r10+offs_r)[None,:] < R1), other=0.).to(tl.float32)
            dX_sub = tl.dot(T, tl.trans(P1_sub))  # [M_blk, K_blk]
            dX_ptrs = dX_ptr + off_b*s_dXb + (row0+offs_m)[:,None]*s_dXm + (k0+offs_k)[None,:]*s_dXd
            tl.atomic_add(dX_ptrs, dX_sub, mask=mask_m[:,None] & ((k0+offs_k)[None,:] < Din))

@triton.jit
def _ffn_bwd_stage1_fixed(
    X_ptr, sXb, sXm, sXd,
    P1_ptr, sP1k, sP1r,
    V1_ptr, sV1r, sV1d,
    dZ_ptr, s_dZb, s_dZm, s_dZd,
    dP1_ptr, s_dP1k, s_dP1r,
    dV1_ptr, s_dV1r, s_dV1d,
    db1_ptr, s_db1d,
    dX_ptr,  s_dXb,  s_dXm,  s_dXd,
    B, M, Din, Dh, R1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)  # Dh tile

    off_b = pid_b
    row0  = pid_m * BLOCK_M
    dh0   = pid_h * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_r = tl.arange(0, BLOCK_R)
    offs_h = tl.arange(0, BLOCK_N)

    mask_m = (row0 + offs_m) < M
    mask_h = (dh0 + offs_h) < Dh

    # Load dZ tile
    dZ_ptrs = dZ_ptr + off_b*s_dZb + (row0 + offs_m)[:,None]*s_dZm + (dh0 + offs_h)[None,:]*s_dZd
    dZ = tl.load(dZ_ptrs, mask=mask_m[:,None] & mask_h[None,:], other=0.).to(tl.float32)

    # db1 += sum_rows(dZ)
    db1_vec = tl.sum(dZ, axis=0)
    db1_ptrs = db1_ptr + (dh0 + offs_h)*s_db1d
    tl.atomic_add(db1_ptrs, db1_vec, mask=mask_h)

    # dV1 and dP1 need S = X @ P1 (for current r1-chunk) and
    # T = dZ @ V1^T (for current r1-chunk)
    for r10 in range(0, R1, BLOCK_R):
        # S = X @ P1[:, r1_chunk]  -> [M_blk, R_blk]
        S = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for k0 in range(0, Din, BLOCK_K):
            X_ptrs = X_ptr + off_b*sXb + (row0+offs_m)[:,None]*sXm + (k0+offs_k)[None,:]*sXd
            P1_ptrs = P1_ptr + (k0+offs_k)[:,None]*sP1k + (r10+offs_r)[None,:]*sP1r
            X_sub  = tl.load(X_ptrs, mask=mask_m[:,None] & ((k0+offs_k)[None,:] < Din), other=0.).to(tl.float32)
            P1_sub = tl.load(P1_ptrs, mask=((k0+offs_k)[:,None] < Din) & ((r10+offs_r)[None,:] < R1), other=0.).to(tl.float32)
            S += tl.dot(X_sub, P1_sub)  # [M_blk, R_blk]

        # dV1 += S^T @ dZ  -> [R_blk, H_blk]
        dV1_sub = tl.dot(tl.trans(S), dZ)
        dV1_ptrs = dV1_ptr + (r10+offs_r)[:,None]*s_dV1r + (dh0+offs_h)[None,:]*s_dV1d
        tl.atomic_add(dV1_ptrs, dV1_sub, mask=((r10+offs_r)[:,None] < R1) & mask_h[None,:])

        # T = dZ @ V1[r1_chunk, H_blk]^T  -> [M_blk, R_blk]
        V1_ptrs = V1_ptr + (r10+offs_r)[:,None]*sV1r + (dh0+offs_h)[None,:]*sV1d
        V1_sub  = tl.load(V1_ptrs, mask=((r10+offs_r)[:,None] < R1) & mask_h[None,:], other=0.).to(tl.float32)
        T = tl.dot(dZ, tl.trans(V1_sub))  # [M_blk, R_blk]

        # dP1 += sum_k X[:,k]^T @ T  (block over K)
        for k0 in range(0, Din, BLOCK_K):
            X_ptrs = X_ptr + off_b*sXb + (row0+offs_m)[:,None]*sXm + (k0+offs_k)[None,:]*sXd
            X_sub = tl.load(X_ptrs, mask=mask_m[:,None] & ((k0+offs_k)[None,:] < Din), other=0.).to(tl.float32)
            dP1_sub = tl.dot(tl.trans(X_sub), T)  # [K_blk, R_blk]
            dP1_ptrs = dP1_ptr + (k0+offs_k)[:,None]*s_dP1k + (r10+offs_r)[None,:]*s_dP1r
            tl.atomic_add(dP1_ptrs, dP1_sub, mask=((k0+offs_k)[:,None] < Din) & ((r10+offs_r)[None,:] < R1))

        # dX += T @ P1[k, r1_chunk]^T  (block over K)
        for k0 in range(0, Din, BLOCK_K):
            P1_ptrs = P1_ptr + (k0+offs_k)[:,None]*sP1k + (r10+offs_r)[None,:]*sP1r
            P1_sub  = tl.load(P1_ptrs, mask=((k0+offs_k)[:,None] < Din) & ((r10+offs_r)[None,:] < R1), other=0.).to(tl.float32)
            dX_sub = tl.dot(T, tl.trans(P1_sub))  # [M_blk, K_blk]
            dX_ptrs = dX_ptr + off_b*s_dXb + (row0+offs_m)[:,None]*s_dXm + (k0+offs_k)[None,:]*s_dXd
            tl.atomic_add(dX_ptrs, dX_sub, mask=mask_m[:,None] & ((k0+offs_k)[None,:] < Din))

# -----------------------------------
# Backward wrapper
# -----------------------------------
def flash_svd_ffn_backward(
    X, P1, V1, b1, P2, V2, b2, dY,
    *, tune=False, block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K, block_r=BLOCK_R
):
    B, M, Din = X.shape
    R1, Dh    = V1.shape
    R2, Dout  = V2.shape

    X, P1, V1, b1, P2, V2, b2, dY = map(_contig, (X,P1,V1,b1,P2,V2,b2,dY))

    # Allocate grads (fp32)
    dP1 = torch.zeros_like(P1, dtype=torch.float32)
    dV1 = torch.zeros_like(V1, dtype=torch.float32)
    db1 = torch.zeros(Dh, device=X.device, dtype=torch.float32)
    dX  = torch.zeros_like(X,  dtype=torch.float32)

    dP2 = torch.zeros_like(P2, dtype=torch.float32)
    dV2 = torch.zeros_like(V2, dtype=torch.float32)
    db2 = torch.zeros(Dout, device=X.device, dtype=torch.float32)

    # Temporary dZ buffer [B,M,Dh]
    dZ = torch.zeros(B, M, Dh, device=X.device, dtype=torch.float32)

    # ---- Stage 2 (top)
    grid2 = ( (M + block_m - 1)//block_m, B, (Dout + block_n - 1)//block_n )
    if tune:
        _ffn_bwd_stage2_auto[grid2](
            X, *X.stride(),
            P1, *P1.stride(),
            V1, *V1.stride(),
            b1, b1.stride(0),
            P2, *P2.stride(),
            V2, *V2.stride(),
            dY, *dY.stride(),
            dP2, *dP2.stride(),
            dV2, *dV2.stride(),
            db2, db2.stride(0),
            dZ, *dZ.stride(),
            B, M, Din, Dh, Dout, R1, R2,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=block_r
        )
    else:
        _ffn_bwd_stage2_fixed[grid2](
            X, *X.stride(),
            P1, *P1.stride(),
            V1, *V1.stride(),
            b1, b1.stride(0),
            P2, *P2.stride(),
            V2, *V2.stride(),
            dY, *dY.stride(),
            dP2, *dP2.stride(),
            dV2, *dV2.stride(),
            db2, db2.stride(0),
            dZ, *dZ.stride(),
            B, M, Din, Dh, Dout, R1, R2,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=block_r
        )

    # ---- Stage 1 (bottom)
    grid1 = ( (M + block_m - 1)//block_m, B, (Dh + block_n - 1)//block_n )
    if tune:
        _ffn_bwd_stage1_auto[grid1](
            X, *X.stride(),
            P1, *P1.stride(),
            V1, *V1.stride(),
            dZ, *dZ.stride(),
            dP1, *dP1.stride(),
            dV1, *dV1.stride(),
            db1, db1.stride(0),
            dX,  *dX.stride(),
            B, M, Din, Dh, R1,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=block_r
        )
    else:
        _ffn_bwd_stage1_fixed[grid1](
            X, *X.stride(),
            P1, *P1.stride(),
            V1, *V1.stride(),
            dZ, *dZ.stride(),
            dP1, *dP1.stride(),
            dV1, *dV1.stride(),
            db1, db1.stride(0),
            dX,  *dX.stride(),
            B, M, Din, Dh, R1,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_R=block_r
        )

    in_dtype = X.dtype
    return dX.to(in_dtype), dP1.to(in_dtype), dV1.to(in_dtype), db1.to(in_dtype), dP2.to(in_dtype), dV2.to(in_dtype), db2.to(in_dtype)

# -----------------------------------
# Dense reference (PyTorch) for checks
# -----------------------------------
def gelu_erf(x: torch.Tensor):
    inv_sqrt2 = 0.7071067811865475
    erf_val = torch.erf(x * inv_sqrt2)
    return 0.5 * x * (1 + erf_val)

def dense_ffn_forward(X, P1, V1, b1, P2, V2, b2):
    Z = X @ (P1 @ V1) + b1
    H = gelu_erf(Z)
    Y = H @ (P2 @ V2) + b2
    return Y

def rel_fro(a: torch.Tensor, b: torch.Tensor, eps=1e-12) -> float:
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    num = torch.linalg.vector_norm(a32 - b32)
    den = torch.linalg.vector_norm(b32).clamp_min(eps)
    return (num / den).item()

# -----------------------------------
# Simple correctness check + bench
# -----------------------------------
def check_correctness_and_bench(
    B=1, M=128, Din=64, Dh=128, Dout=64, R1=64, R2=64,
    dtype=torch.float16, device='cuda', iters=20, tune=False,
    measure_peak_mem=True
):
    device_obj = torch.device(device)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _ = torch.randn(1, device=device_obj)  # init CUDA ctx

    # Init factors with variance ~ 1
    scale1 = 1.0 / math.sqrt(R1)
    scale2 = 1.0 / math.sqrt(R2)
    X  = torch.randn(B, M, Din, device=device_obj, dtype=dtype)

    P1 = torch.randn(Din, R1, device=device_obj, dtype=dtype)
    V1 = torch.randn(R1, Dh,  device=device_obj, dtype=dtype) * scale1
    b1 = torch.zeros(Dh,      device=device_obj, dtype=dtype)

    P2 = torch.randn(Dh,  R2,   device=device_obj, dtype=dtype)
    V2 = torch.randn(R2,  Dout, device=device_obj, dtype=dtype) * scale2
    b2 = torch.zeros(Dout,      device=device_obj, dtype=dtype)

    # Forward check
    Y_tt = flash_svd_ffn_forward(X, P1, V1, b1, P2, V2, b2,
                                 block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K, block_r=BLOCK_R)
    Y_ref = dense_ffn_forward(
        X.float(),
        P1.float(),
        V1.float(),
        b1.float(),
        P2.float(),
        V2.float(),
        b2.float(),
    )
    fwd_rel = rel_fro(Y_tt.float(), Y_ref)
    print(f"[check] forward relF(triton,dense) = {fwd_rel:.3e}  (target < 1e-2)")

    # Backward check vs autograd
    X_r  = X.detach().clone().requires_grad_(True)
    P1_r = P1.detach().clone().requires_grad_(True)
    V1_r = V1.detach().clone().requires_grad_(True)
    b1_r = b1.detach().clone().requires_grad_(True)
    P2_r = P2.detach().clone().requires_grad_(True)
    V2_r = V2.detach().clone().requires_grad_(True)
    b2_r = b2.detach().clone().requires_grad_(True)

    Y_ref = dense_ffn_forward(X_r, P1_r, V1_r, b1_r, P2_r, V2_r, b2_r)
    dY = torch.randn_like(Y_ref)
    # Factor-space autograd reference
    X_r2  = X.detach().clone().requires_grad_(True)
    P1_r2 = P1.detach().clone().requires_grad_(True)
    V1_r2 = V1.detach().clone().requires_grad_(True)
    b1_r2 = b1.detach().clone().requires_grad_(True)
    P2_r2 = P2.detach().clone().requires_grad_(True)
    V2_r2 = V2.detach().clone().requires_grad_(True)
    b2_r2 = b2.detach().clone().requires_grad_(True)
    Y_ref2 = dense_ffn_forward(X_r2, P1_r2, V1_r2, b1_r2, P2_r2, V2_r2, b2_r2)
    (dX_ref2, dP1_ref, dV1_ref, db1_ref2, dP2_ref, dV2_ref, db2_ref2) = torch.autograd.grad(
        outputs=Y_ref2,
        inputs=[X_r2, P1_r2, V1_r2, b1_r2, P2_r2, V2_r2, b2_r2],
        grad_outputs=dY
    )

    (dX_tt, dP1_tt, dV1_tt, db1_tt, dP2_tt, dV2_tt, db2_tt) = flash_svd_ffn_backward(
        X, P1, V1, b1, P2, V2, b2, dY,
        tune=tune, block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K, block_r=BLOCK_R
    )

    def report(name, a, b):
        rel = rel_fro(a, b)
        diff = (a - b).abs()
        print(f"[check] {name:6s}  relF={rel:.3e}  max|Δ|={diff.max().item():.3e}  mean|Δ|={diff.mean().item():.3e}")

    report("dX",  dX_tt,  dX_ref2)
    report("dP1", dP1_tt, dP1_ref)
    report("dV1", dV1_tt, dV1_ref)
    report("db1", db1_tt, db1_ref2)
    report("dP2", dP2_tt, dP2_ref)
    report("dV2", dV2_tt, dV2_ref)
    report("db2", db2_tt, db2_ref2)

    # Bench (small dummy)
    if device_obj.type == 'cuda':
        torch.cuda.synchronize(device_obj)
    def bench_fwd():
        if device_obj.type == 'cuda':
            torch.cuda.synchronize(device_obj)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = flash_svd_ffn_forward(X, P1, V1, b1, P2, V2, b2)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize(device_obj)
        return time.time() - t0

    def bench_bwd():
        if device_obj.type == 'cuda':
            torch.cuda.synchronize(device_obj)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(iters):
                flash_svd_ffn_backward(X, P1, V1, b1, P2, V2, b2, dY, tune=tune)
        if device_obj.type == 'cuda':
            torch.cuda.synchronize(device_obj)
        return time.time() - t0

    bench_fwd(); bench_bwd()
    tf = bench_fwd()
    tb = bench_bwd()
    print(f"[bench] forward:  {tf/iters*1e3:.2f} ms / iter")
    print(f"[bench] backward: {tb/iters*1e3:.2f} ms / iter  ({'autotuned' if tune else 'fixed'})")

    if measure_peak_mem and device_obj.type == 'cuda':
        def measure_peak(label, fn):
            with torch.cuda.device(device_obj):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(device_obj)
                fn()
                torch.cuda.synchronize(device_obj)
                peak = torch.cuda.max_memory_allocated(device_obj)
            peak_mb = peak / (1024 ** 2)
            print(f"[mem] {label:<28s} peak={peak_mb:8.2f} MiB")
            return peak

        def triton_step():
            with torch.no_grad():
                _ = flash_svd_ffn_forward(
                    X, P1, V1, b1, P2, V2, b2,
                    block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K, block_r=BLOCK_R,
                )
                flash_svd_ffn_backward(
                    X, P1, V1, b1, P2, V2, b2, dY,
                    tune=tune, block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K, block_r=BLOCK_R,
                )

        def dense_step():
            with torch.enable_grad():
                X_d = X.detach().clone().requires_grad_(True)
                P1_d = P1.detach().clone().requires_grad_(True)
                V1_d = V1.detach().clone().requires_grad_(True)
                b1_d = b1.detach().clone().requires_grad_(True)
                P2_d = P2.detach().clone().requires_grad_(True)
                V2_d = V2.detach().clone().requires_grad_(True)
                b2_d = b2.detach().clone().requires_grad_(True)
                out = dense_ffn_forward(X_d, P1_d, V1_d, b1_d, P2_d, V2_d, b2_d)
                out.backward(dY.detach().clone())
            for t in (X_d, P1_d, V1_d, b1_d, P2_d, V2_d, b2_d):
                if t.grad is not None:
                    t.grad = None
            del out
            del X_d, P1_d, V1_d, b1_d, P2_d, V2_d, b2_d

        peak_triton = measure_peak("triton (fixed)", triton_step)
        peak_dense = measure_peak("dense_autograd", dense_step)
        delta_mb = (peak_dense - peak_triton) / (1024 ** 2)
        print(f"[mem] dense - triton:            {delta_mb:8.2f} MiB")
    elif measure_peak_mem:
        print("[mem] Peak memory stats require CUDA device; skipping.")

# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    check_correctness_and_bench(
        B=128, M=512, Din=768, Dh=768*4, Dout=768, R1=384, R2=384,
        dtype=torch.float16, device='cuda', iters=20, tune=False
    )
