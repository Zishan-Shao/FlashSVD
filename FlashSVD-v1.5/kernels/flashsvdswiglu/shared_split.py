from __future__ import annotations

import torch
import triton
import triton.language as tl

@triton.jit
def _shared_split_phase1_partials_token(
    P_ptr,
    GateU_ptr,
    UpU_ptr,
    DownV_ptr,
    Work_ptr,
    T,
    ND,
    D,
    R,
    sP_t,
    sP_r,
    sGate_r,
    sGate_d,
    sUp_r,
    sUp_d,
    sDownV_d,
    sDownV_r,
    sWork_t,
    sWork_n,
    sWork_r,
    BR: tl.constexpr,
    BD: tl.constexpr,
    BR2: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_t >= T or pid_n >= ND:
        return

    offs_d = pid_n * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    in_dtype = tl.float32 if USE_FP32 else (tl.bfloat16 if USE_BF16 else tl.float16)

    gate_acc = tl.zeros((1, BD), dtype=tl.float32)
    up_acc = tl.zeros((1, BD), dtype=tl.float32)

    for r0 in range(0, R, BR):
        offs_r = r0 + tl.arange(0, BR)
        mask_r = offs_r < R

        p_blk = tl.load(
            P_ptr + pid_t * sP_t + offs_r * sP_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_blk = tl.reshape(p_blk, (1, BR))

        gate_blk = tl.load(
            GateU_ptr + offs_r[:, None] * sGate_r + offs_d[None, :] * sGate_d,
            mask=mask_r[:, None] & mask_d[None, :],
            other=0.0,
        ).to(in_dtype)
        up_blk = tl.load(
            UpU_ptr + offs_r[:, None] * sUp_r + offs_d[None, :] * sUp_d,
            mask=mask_r[:, None] & mask_d[None, :],
            other=0.0,
        ).to(in_dtype)

        gate_acc = tl.dot(p_blk, gate_blk, acc=gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(p_blk, up_blk, acc=up_acc, out_dtype=tl.float32)

    h = ((gate_acc * tl.sigmoid(gate_acc)) * up_acc).to(in_dtype)

    work_base = Work_ptr + pid_t * sWork_t + pid_n * sWork_n
    for r2_0 in range(0, R, BR2):
        offs_r2 = r2_0 + tl.arange(0, BR2)
        mask_r2 = offs_r2 < R
        downv_blk = tl.load(
            DownV_ptr + offs_d[:, None] * sDownV_d + offs_r2[None, :] * sDownV_r,
            mask=mask_d[:, None] & mask_r2[None, :],
            other=0.0,
        ).to(in_dtype)
        acc = tl.zeros((1, BR2), dtype=tl.float32)
        acc = tl.dot(h, downv_blk, acc=acc, out_dtype=tl.float32)
        acc = tl.reshape(acc, (BR2,))
        tl.store(work_base + offs_r2 * sWork_r, acc, mask=mask_r2)


@triton.jit
def _shared_split_phase1_atomic_token(
    P_ptr,
    GateU_ptr,
    UpU_ptr,
    DownV_ptr,
    S_ptr,
    T,
    D,
    R,
    sP_t,
    sP_r,
    sGate_r,
    sGate_d,
    sUp_r,
    sUp_d,
    sDownV_d,
    sDownV_r,
    sS_t,
    sS_r,
    BR: tl.constexpr,
    BD: tl.constexpr,
    BR2: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_t >= T:
        return

    offs_d = pid_n * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    in_dtype = tl.float32 if USE_FP32 else (tl.bfloat16 if USE_BF16 else tl.float16)

    gate_acc = tl.zeros((1, BD), dtype=tl.float32)
    up_acc = tl.zeros((1, BD), dtype=tl.float32)

    for r0 in range(0, R, BR):
        offs_r = r0 + tl.arange(0, BR)
        mask_r = offs_r < R

        p_blk = tl.load(
            P_ptr + pid_t * sP_t + offs_r * sP_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_blk = tl.reshape(p_blk, (1, BR))

        gate_blk = tl.load(
            GateU_ptr + offs_r[:, None] * sGate_r + offs_d[None, :] * sGate_d,
            mask=mask_r[:, None] & mask_d[None, :],
            other=0.0,
        ).to(in_dtype)
        up_blk = tl.load(
            UpU_ptr + offs_r[:, None] * sUp_r + offs_d[None, :] * sUp_d,
            mask=mask_r[:, None] & mask_d[None, :],
            other=0.0,
        ).to(in_dtype)

        gate_acc = tl.dot(p_blk, gate_blk, acc=gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(p_blk, up_blk, acc=up_acc, out_dtype=tl.float32)

    h = ((gate_acc * tl.sigmoid(gate_acc)) * up_acc).to(in_dtype)
    s_base = S_ptr + pid_t * sS_t

    for r2_0 in range(0, R, BR2):
        offs_r2 = r2_0 + tl.arange(0, BR2)
        mask_r2 = offs_r2 < R
        downv_blk = tl.load(
            DownV_ptr + offs_d[:, None] * sDownV_d + offs_r2[None, :] * sDownV_r,
            mask=mask_d[:, None] & mask_r2[None, :],
            other=0.0,
        ).to(in_dtype)
        acc = tl.zeros((1, BR2), dtype=tl.float32)
        acc = tl.dot(h, downv_blk, acc=acc, out_dtype=tl.float32)
        acc = tl.reshape(acc, (BR2,))
        tl.atomic_add(s_base + offs_r2 * sS_r, acc, mask=mask_r2)


def _pick_shared_split_token_config(R: int, D: int) -> dict[str, int]:
    if max(R, D) >= 8192:
        return {'BR': 128, 'BD': 128, 'BR2': 128, 'warps': 8, 'stages': 2}
    if R >= 1024:
        return {'BR': 128, 'BD': 128, 'BR2': 128, 'warps': 8, 'stages': 2}
    if R >= 512:
        return {'BR': 64, 'BD': 128, 'BR2': 128, 'warps': 4, 'stages': 2}
    return {'BR': 64, 'BD': 128, 'BR2': 256, 'warps': 4, 'stages': 2}

def flashsvd_ffn_shared_split_token(
    P,
    GateU,
    UpU,
    DownV,
    DownU,
    b2=None,
    *,
    BR: int | None = None,
    BD: int | None = None,
    BR2: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    store_partials_fp32: bool = True,
    use_atomic_accum: bool = False,
):
    """
    Decode-only shared-P backend for exact low-rank SwiGLU checkpoints where
    gate_v_proj == up_v_proj.

    Shapes:
      P      : [B, L, R]
      GateU  : [R, D]  = gate_u_proj.weight.T
      UpU    : [R, D]  = up_u_proj.weight.T
      DownV  : [D, R]  = down_v_proj.weight.T
      DownU  : [R, H]  = down_u_proj.weight.T
    """
    assert P.is_cuda and GateU.is_cuda and UpU.is_cuda and DownV.is_cuda and DownU.is_cuda
    if P.ndim != 3:
        raise ValueError(f"P must be [B, L, R], got {tuple(P.shape)}")

    B, L, R = P.shape
    if GateU.shape[0] != R:
        raise ValueError(f"GateU must be [R, D] with R={R}, got {tuple(GateU.shape)}")
    if UpU.shape != GateU.shape:
        raise ValueError(f"UpU shape {tuple(UpU.shape)} must match GateU shape {tuple(GateU.shape)}")
    D = int(GateU.shape[1])
    if DownV.shape != (D, R):
        raise ValueError(f"DownV must be [D, R]=[{D}, {R}], got {tuple(DownV.shape)}")
    if DownU.shape[0] != R:
        raise ValueError(f"DownU must be [R, H] with R={R}, got {tuple(DownU.shape)}")

    cfg = _pick_shared_split_token_config(R, D)
    BR = cfg['BR'] if BR is None else BR
    BD = cfg['BD'] if BD is None else BD
    BR2 = cfg['BR2'] if BR2 is None else BR2
    num_warps = cfg['warps'] if num_warps is None else num_warps
    num_stages = cfg['stages'] if num_stages is None else num_stages

    T = int(B * L)
    p2d = P.contiguous().reshape(T, R)
    gate_u = GateU.contiguous()
    up_u = UpU.contiguous()
    down_v = DownV.contiguous()
    down_u = DownU.contiguous()

    use_fp32 = int(P.dtype == torch.float32)
    use_bf16 = int(P.dtype == torch.bfloat16)
    nd = triton.cdiv(D, BD)
    if use_atomic_accum:
        s2d = torch.zeros((T, R), device=P.device, dtype=torch.float32)
        _shared_split_phase1_atomic_token[(T, nd)](
            p2d,
            gate_u,
            up_u,
            down_v,
            s2d,
            T,
            D,
            R,
            p2d.stride(0),
            p2d.stride(1),
            gate_u.stride(0),
            gate_u.stride(1),
            up_u.stride(0),
            up_u.stride(1),
            down_v.stride(0),
            down_v.stride(1),
            s2d.stride(0),
            s2d.stride(1),
            BR=BR,
            BD=BD,
            BR2=BR2,
            USE_BF16=use_bf16,
            USE_FP32=use_fp32,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        work_dtype = torch.float32 if store_partials_fp32 else P.dtype
        work = torch.empty((T, nd, R), device=P.device, dtype=work_dtype)
        _shared_split_phase1_partials_token[(T, nd)](
            p2d,
            gate_u,
            up_u,
            down_v,
            work,
            T,
            nd,
            D,
            R,
            p2d.stride(0),
            p2d.stride(1),
            gate_u.stride(0),
            gate_u.stride(1),
            up_u.stride(0),
            up_u.stride(1),
            down_v.stride(0),
            down_v.stride(1),
            work.stride(0),
            work.stride(1),
            work.stride(2),
            BR=BR,
            BD=BD,
            BR2=BR2,
            USE_BF16=use_bf16,
            USE_FP32=use_fp32,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        s2d = work.sum(dim=1)

    if s2d.dtype != P.dtype:
        s2d = s2d.to(P.dtype)
    y2d = torch.matmul(s2d, down_u) if b2 is None else torch.addmm(b2, s2d, down_u)
    return y2d.reshape(B, L, down_u.shape[1])


__all__ = ["flashsvd_ffn_shared_split_token"]
