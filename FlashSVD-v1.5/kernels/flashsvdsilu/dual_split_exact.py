from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _pick_dual_split_token_config(R: int, D: int) -> dict[str, int]:
    if max(R, D) >= 8192:
        cfg = {"BR": 128, "BD": 128, "BR2": 256, "warps": 8, "stages": 2}
    elif R >= 1024:
        cfg = {"BR": 128, "BD": 128, "BR2": 256, "warps": 8, "stages": 2}
    elif R >= 512:
        cfg = {"BR": 64, "BD": 128, "BR2": 128, "warps": 4, "stages": 2}
    else:
        cfg = {"BR": 64, "BD": 128, "BR2": 256, "warps": 4, "stages": 2}

    for key, env_name in (
        ("BR", "FLASH_SVD_DUAL_SPLIT_BR"),
        ("BD", "FLASH_SVD_DUAL_SPLIT_BD"),
        ("BR2", "FLASH_SVD_DUAL_SPLIT_BR2"),
        ("warps", "FLASH_SVD_DUAL_SPLIT_WARPS"),
        ("stages", "FLASH_SVD_DUAL_SPLIT_STAGES"),
    ):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except Exception:
            continue
        if value > 0:
            cfg[key] = value
    return cfg


def _pick_dual_split_token_v2_config(R: int, D: int) -> dict[str, int]:
    if max(R, D) >= 8192:
        cfg = {"BR": 128, "BD": 128, "BR2": 128, "warps": 8, "stages": 2}
    elif R >= 1024:
        cfg = {"BR": 128, "BD": 128, "BR2": 128, "warps": 8, "stages": 2}
    elif R >= 512:
        cfg = {"BR": 64, "BD": 128, "BR2": 128, "warps": 4, "stages": 2}
    else:
        cfg = {"BR": 64, "BD": 128, "BR2": 128, "warps": 4, "stages": 2}
    for key, env_name in (
        ("BR", "FLASH_SVD_DUAL_SPLIT_V2_BR"),
        ("BD", "FLASH_SVD_DUAL_SPLIT_V2_BD"),
        ("BR2", "FLASH_SVD_DUAL_SPLIT_V2_BR2"),
        ("warps", "FLASH_SVD_DUAL_SPLIT_V2_WARPS"),
        ("stages", "FLASH_SVD_DUAL_SPLIT_V2_STAGES"),
    ):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except Exception:
            continue
        if value > 0:
            cfg[key] = value
    return cfg


def _is_sm80_device(device: torch.device) -> bool:
    try:
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        return False
    return int(major) == 8 and int(minor) == 0


def _pick_dual_split_token_v2_sm80_config(R: int, D: int, *, is_sm80: bool) -> dict[str, int]:
    if max(R, D) >= 8192:
        cfg = {"BR": 128, "BD": 64, "BR2": 128, "GD": 2, "warps": 8, "stages": 2 if is_sm80 else 1}
    elif R >= 1024:
        cfg = {"BR": 128, "BD": 64, "BR2": 128, "GD": 2, "warps": 8, "stages": 2 if is_sm80 else 1}
    elif R >= 512:
        cfg = {"BR": 64, "BD": 64, "BR2": 128, "GD": 2, "warps": 8 if is_sm80 else 4, "stages": 2}
    else:
        cfg = {"BR": 64, "BD": 128, "BR2": 128, "GD": 1, "warps": 4, "stages": 2}
    for key, env_name in (
        ("BR", "FLASH_SVD_DUAL_SPLIT_V2_SM80_BR"),
        ("BD", "FLASH_SVD_DUAL_SPLIT_V2_SM80_BD"),
        ("BR2", "FLASH_SVD_DUAL_SPLIT_V2_SM80_BR2"),
        ("GD", "FLASH_SVD_DUAL_SPLIT_V2_SM80_GD"),
        ("warps", "FLASH_SVD_DUAL_SPLIT_V2_SM80_WARPS"),
        ("stages", "FLASH_SVD_DUAL_SPLIT_V2_SM80_STAGES"),
    ):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except Exception:
            continue
        if value > 0:
            cfg[key] = value
    return cfg


def _pick_dual_split_token_v3_config(R: int, D: int, H: int) -> dict[str, int]:
    if max(R, D, H) >= 8192:
        cfg = {"BR": 128, "BD": 128, "BR2": 128, "BH": 128, "GH": 4, "warps": 8, "stages": 2}
    elif max(R, H) >= 1024:
        cfg = {"BR": 128, "BD": 128, "BR2": 128, "BH": 128, "GH": 4, "warps": 8, "stages": 2}
    else:
        cfg = {"BR": 64, "BD": 128, "BR2": 128, "BH": 128, "GH": 2, "warps": 4, "stages": 2}
    for key, env_name in (
        ("BR", "FLASH_SVD_DUAL_SPLIT_V3_BR"),
        ("BD", "FLASH_SVD_DUAL_SPLIT_V3_BD"),
        ("BR2", "FLASH_SVD_DUAL_SPLIT_V3_BR2"),
        ("BH", "FLASH_SVD_DUAL_SPLIT_V3_BH"),
        ("GH", "FLASH_SVD_DUAL_SPLIT_V3_GH"),
        ("warps", "FLASH_SVD_DUAL_SPLIT_V3_WARPS"),
        ("stages", "FLASH_SVD_DUAL_SPLIT_V3_STAGES"),
    ):
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except Exception:
            continue
        if value > 0:
            cfg[key] = value
    return cfg


@triton.jit
def _dual_split_phase1_partials_token(
    PUp_ptr,
    PGate_ptr,
    GateU_ptr,
    UpU_ptr,
    DownV_ptr,
    Work_ptr,
    T,
    ND,
    D,
    R,
    sPUp_t,
    sPUp_r,
    sPGate_t,
    sPGate_r,
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

        p_up_blk = tl.load(
            PUp_ptr + pid_t * sPUp_t + offs_r * sPUp_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_gate_blk = tl.load(
            PGate_ptr + pid_t * sPGate_t + offs_r * sPGate_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_up_blk = tl.reshape(p_up_blk, (1, BR))
        p_gate_blk = tl.reshape(p_gate_blk, (1, BR))

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

        gate_acc = tl.dot(p_gate_blk, gate_blk, acc=gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(p_up_blk, up_blk, acc=up_acc, out_dtype=tl.float32)

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
def _dual_split_phase1_atomic_token(
    PUp_ptr,
    PGate_ptr,
    GateU_ptr,
    UpU_ptr,
    DownV_ptr,
    S_ptr,
    T,
    D,
    R,
    sPUp_t,
    sPUp_r,
    sPGate_t,
    sPGate_r,
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

        p_up_blk = tl.load(
            PUp_ptr + pid_t * sPUp_t + offs_r * sPUp_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_gate_blk = tl.load(
            PGate_ptr + pid_t * sPGate_t + offs_r * sPGate_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_up_blk = tl.reshape(p_up_blk, (1, BR))
        p_gate_blk = tl.reshape(p_gate_blk, (1, BR))

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

        gate_acc = tl.dot(p_gate_blk, gate_blk, acc=gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(p_up_blk, up_blk, acc=up_acc, out_dtype=tl.float32)

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


@triton.jit
def _dual_split_phase1_atomic_token_gd2(
    PUp_ptr,
    PGate_ptr,
    GateU_ptr,
    UpU_ptr,
    DownV_ptr,
    S_ptr,
    T,
    D,
    R,
    sPUp_t,
    sPUp_r,
    sPGate_t,
    sPGate_r,
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

    offs_d0 = pid_n * (2 * BD) + tl.arange(0, BD)
    offs_d1 = offs_d0 + BD
    mask_d0 = offs_d0 < D
    mask_d1 = offs_d1 < D
    in_dtype = tl.float32 if USE_FP32 else (tl.bfloat16 if USE_BF16 else tl.float16)

    gate_acc0 = tl.zeros((1, BD), dtype=tl.float32)
    up_acc0 = tl.zeros((1, BD), dtype=tl.float32)
    gate_acc1 = tl.zeros((1, BD), dtype=tl.float32)
    up_acc1 = tl.zeros((1, BD), dtype=tl.float32)

    for r0 in range(0, R, BR):
        offs_r = r0 + tl.arange(0, BR)
        mask_r = offs_r < R

        p_up_blk = tl.load(
            PUp_ptr + pid_t * sPUp_t + offs_r * sPUp_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_gate_blk = tl.load(
            PGate_ptr + pid_t * sPGate_t + offs_r * sPGate_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_up_blk = tl.reshape(p_up_blk, (1, BR))
        p_gate_blk = tl.reshape(p_gate_blk, (1, BR))

        gate_blk0 = tl.load(
            GateU_ptr + offs_r[:, None] * sGate_r + offs_d0[None, :] * sGate_d,
            mask=mask_r[:, None] & mask_d0[None, :],
            other=0.0,
        ).to(in_dtype)
        up_blk0 = tl.load(
            UpU_ptr + offs_r[:, None] * sUp_r + offs_d0[None, :] * sUp_d,
            mask=mask_r[:, None] & mask_d0[None, :],
            other=0.0,
        ).to(in_dtype)
        gate_acc0 = tl.dot(p_gate_blk, gate_blk0, acc=gate_acc0, out_dtype=tl.float32)
        up_acc0 = tl.dot(p_up_blk, up_blk0, acc=up_acc0, out_dtype=tl.float32)

        gate_blk1 = tl.load(
            GateU_ptr + offs_r[:, None] * sGate_r + offs_d1[None, :] * sGate_d,
            mask=mask_r[:, None] & mask_d1[None, :],
            other=0.0,
        ).to(in_dtype)
        up_blk1 = tl.load(
            UpU_ptr + offs_r[:, None] * sUp_r + offs_d1[None, :] * sUp_d,
            mask=mask_r[:, None] & mask_d1[None, :],
            other=0.0,
        ).to(in_dtype)
        gate_acc1 = tl.dot(p_gate_blk, gate_blk1, acc=gate_acc1, out_dtype=tl.float32)
        up_acc1 = tl.dot(p_up_blk, up_blk1, acc=up_acc1, out_dtype=tl.float32)

    h0 = ((gate_acc0 * tl.sigmoid(gate_acc0)) * up_acc0).to(in_dtype)
    h1 = ((gate_acc1 * tl.sigmoid(gate_acc1)) * up_acc1).to(in_dtype)
    s_base = S_ptr + pid_t * sS_t

    for r2_0 in range(0, R, BR2):
        offs_r2 = r2_0 + tl.arange(0, BR2)
        mask_r2 = offs_r2 < R
        downv_blk0 = tl.load(
            DownV_ptr + offs_d0[:, None] * sDownV_d + offs_r2[None, :] * sDownV_r,
            mask=mask_d0[:, None] & mask_r2[None, :],
            other=0.0,
        ).to(in_dtype)
        acc = tl.zeros((1, BR2), dtype=tl.float32)
        acc = tl.dot(h0, downv_blk0, acc=acc, out_dtype=tl.float32)
        downv_blk1 = tl.load(
            DownV_ptr + offs_d1[:, None] * sDownV_d + offs_r2[None, :] * sDownV_r,
            mask=mask_d1[:, None] & mask_r2[None, :],
            other=0.0,
        ).to(in_dtype)
        acc = tl.dot(h1, downv_blk1, acc=acc, out_dtype=tl.float32)
        acc = tl.reshape(acc, (BR2,))
        tl.atomic_add(s_base + offs_r2 * sS_r, acc, mask=mask_r2)


@triton.jit
def _dual_split_phase1_to_y_atomic_token(
    PUp_ptr,
    PGate_ptr,
    GateU_ptr,
    UpU_ptr,
    DownV_ptr,
    DownU_ptr,
    Y_ptr,
    T,
    D,
    R,
    H,
    sPUp_t,
    sPUp_r,
    sPGate_t,
    sPGate_r,
    sGate_r,
    sGate_d,
    sUp_r,
    sUp_d,
    sDownV_d,
    sDownV_r,
    sDownU_r,
    sDownU_h,
    sY_t,
    sY_h,
    BR: tl.constexpr,
    BD: tl.constexpr,
    BR2: tl.constexpr,
    BH: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32: tl.constexpr,
    GH: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

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

        p_up_blk = tl.load(
            PUp_ptr + pid_t * sPUp_t + offs_r * sPUp_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_gate_blk = tl.load(
            PGate_ptr + pid_t * sPGate_t + offs_r * sPGate_r,
            mask=mask_r,
            other=0.0,
        ).to(in_dtype)
        p_up_blk = tl.reshape(p_up_blk, (1, BR))
        p_gate_blk = tl.reshape(p_gate_blk, (1, BR))

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

        gate_acc = tl.dot(p_gate_blk, gate_blk, acc=gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(p_up_blk, up_blk, acc=up_acc, out_dtype=tl.float32)

    h = ((gate_acc * tl.sigmoid(gate_acc)) * up_acc).to(in_dtype)
    y_base = Y_ptr + pid_t * sY_t

    for h0 in range(pid_g * BH, H, GH * BH):
        offs_h = h0 + tl.arange(0, BH)
        mask_h = offs_h < H
        y_acc = tl.zeros((1, BH), dtype=tl.float32)
        for r2_0 in range(0, R, BR2):
            offs_r2 = r2_0 + tl.arange(0, BR2)
            mask_r2 = offs_r2 < R
            downv_blk = tl.load(
                DownV_ptr + offs_d[:, None] * sDownV_d + offs_r2[None, :] * sDownV_r,
                mask=mask_d[:, None] & mask_r2[None, :],
                other=0.0,
            ).to(in_dtype)
            s_acc = tl.zeros((1, BR2), dtype=tl.float32)
            s_acc = tl.dot(h, downv_blk, acc=s_acc, out_dtype=tl.float32)
            s_acc = s_acc.to(in_dtype)
            downu_blk = tl.load(
                DownU_ptr + offs_r2[:, None] * sDownU_r + offs_h[None, :] * sDownU_h,
                mask=mask_r2[:, None] & mask_h[None, :],
                other=0.0,
            ).to(in_dtype)
            y_acc = tl.dot(s_acc, downu_blk, acc=y_acc, out_dtype=tl.float32)
        y_acc = tl.reshape(y_acc, (BH,))
        tl.atomic_add(y_base + offs_h * sY_h, y_acc, mask=mask_h)


def flashsvd_ffn_dual_split_token(
    PUp,
    PGate,
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
    store_partials_fp32: bool = False,
):
    assert PUp.is_cuda and PGate.is_cuda and GateU.is_cuda and UpU.is_cuda and DownV.is_cuda and DownU.is_cuda
    if PUp.ndim != 3 or PGate.ndim != 3:
        raise ValueError(f"PUp/PGate must be [B, L, R], got {tuple(PUp.shape)} and {tuple(PGate.shape)}")
    if PUp.shape != PGate.shape:
        raise ValueError(f"PUp and PGate shapes must match, got {tuple(PUp.shape)} and {tuple(PGate.shape)}")

    B, L, R = PUp.shape
    if GateU.shape[0] != R:
        raise ValueError(f"GateU must be [R, D] with R={R}, got {tuple(GateU.shape)}")
    if UpU.shape != GateU.shape:
        raise ValueError(f"UpU shape {tuple(UpU.shape)} must match GateU shape {tuple(GateU.shape)}")
    D = int(GateU.shape[1])
    if DownV.shape != (D, R):
        raise ValueError(f"DownV must be [D, R]=[{D}, {R}], got {tuple(DownV.shape)}")
    if DownU.shape[0] != R:
        raise ValueError(f"DownU must be [R, H] with R={R}, got {tuple(DownU.shape)}")

    cfg = _pick_dual_split_token_config(R, D)
    BR = cfg["BR"] if BR is None else BR
    BD = cfg["BD"] if BD is None else BD
    BR2 = cfg["BR2"] if BR2 is None else BR2
    num_warps = cfg["warps"] if num_warps is None else num_warps
    num_stages = cfg["stages"] if num_stages is None else num_stages

    T = int(B * L)
    p_up_2d = PUp.contiguous().reshape(T, R)
    p_gate_2d = PGate.contiguous().reshape(T, R)
    gate_u = GateU.contiguous()
    up_u = UpU.contiguous()
    down_v = DownV.contiguous()
    down_u = DownU.contiguous()

    use_fp32 = int(PUp.dtype == torch.float32)
    use_bf16 = int(PUp.dtype == torch.bfloat16)
    nd = triton.cdiv(D, BD)
    work_dtype = torch.float32 if store_partials_fp32 else PUp.dtype
    work = torch.empty((T, nd, R), device=PUp.device, dtype=work_dtype)
    _dual_split_phase1_partials_token[(T, nd)](
        p_up_2d,
        p_gate_2d,
        gate_u,
        up_u,
        down_v,
        work,
        T,
        nd,
        D,
        R,
        p_up_2d.stride(0),
        p_up_2d.stride(1),
        p_gate_2d.stride(0),
        p_gate_2d.stride(1),
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
    if s2d.dtype != PUp.dtype:
        s2d = s2d.to(PUp.dtype)
    y2d = torch.matmul(s2d, down_u) if b2 is None else torch.addmm(b2, s2d, down_u)
    return y2d.reshape(B, L, down_u.shape[1])


def flashsvd_ffn_dual_split_token_v2(
    PUp,
    PGate,
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
    workspace_s2d: torch.Tensor | None = None,
):
    assert PUp.is_cuda and PGate.is_cuda and GateU.is_cuda and UpU.is_cuda and DownV.is_cuda and DownU.is_cuda
    if PUp.ndim != 3 or PGate.ndim != 3:
        raise ValueError(f"PUp/PGate must be [B, L, R], got {tuple(PUp.shape)} and {tuple(PGate.shape)}")
    if PUp.shape != PGate.shape:
        raise ValueError(f"PUp and PGate shapes must match, got {tuple(PUp.shape)} and {tuple(PGate.shape)}")

    B, L, R = PUp.shape
    if GateU.shape[0] != R:
        raise ValueError(f"GateU must be [R, D] with R={R}, got {tuple(GateU.shape)}")
    if UpU.shape != GateU.shape:
        raise ValueError(f"UpU shape {tuple(UpU.shape)} must match GateU shape {tuple(GateU.shape)}")
    D = int(GateU.shape[1])
    if DownV.shape != (D, R):
        raise ValueError(f"DownV must be [D, R]=[{D}, {R}], got {tuple(DownV.shape)}")
    if DownU.shape[0] != R:
        raise ValueError(f"DownU must be [R, H] with R={R}, got {tuple(DownU.shape)}")

    cfg = _pick_dual_split_token_v2_config(R, D)
    BR = cfg["BR"] if BR is None else BR
    BD = cfg["BD"] if BD is None else BD
    BR2 = cfg["BR2"] if BR2 is None else BR2
    num_warps = cfg["warps"] if num_warps is None else num_warps
    num_stages = cfg["stages"] if num_stages is None else num_stages

    T = int(B * L)
    p_up_2d = PUp.contiguous().reshape(T, R)
    p_gate_2d = PGate.contiguous().reshape(T, R)
    gate_u = GateU.contiguous()
    up_u = UpU.contiguous()
    down_v = DownV.contiguous()
    down_u = DownU.contiguous()

    use_fp32 = int(PUp.dtype == torch.float32)
    use_bf16 = int(PUp.dtype == torch.bfloat16)
    nd = triton.cdiv(D, BD)
    if workspace_s2d is not None:
        if workspace_s2d.shape != (T, R):
            raise ValueError(f"workspace_s2d must be {(T, R)}, got {tuple(workspace_s2d.shape)}")
        if workspace_s2d.device != PUp.device:
            raise ValueError("workspace_s2d must be on the same device as inputs")
        if workspace_s2d.dtype != torch.float32:
            raise ValueError("workspace_s2d must have dtype float32")
        s2d = workspace_s2d
        s2d.zero_()
    else:
        s2d = torch.zeros((T, R), device=PUp.device, dtype=torch.float32)
    _dual_split_phase1_atomic_token[(T, nd)](
        p_up_2d,
        p_gate_2d,
        gate_u,
        up_u,
        down_v,
        s2d,
        T,
        D,
        R,
        p_up_2d.stride(0),
        p_up_2d.stride(1),
        p_gate_2d.stride(0),
        p_gate_2d.stride(1),
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
    if s2d.dtype != PUp.dtype:
        s2d = s2d.to(PUp.dtype)
    y2d = torch.matmul(s2d, down_u) if b2 is None else torch.addmm(b2, s2d, down_u)
    return y2d.reshape(B, L, down_u.shape[1])


def flashsvd_ffn_dual_split_token_v2_sm80(
    PUp,
    PGate,
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
    workspace_s2d: torch.Tensor | None = None,
    GD: int | None = None,
):
    assert PUp.is_cuda and PGate.is_cuda and GateU.is_cuda and UpU.is_cuda and DownV.is_cuda and DownU.is_cuda
    if PUp.ndim != 3 or PGate.ndim != 3:
        raise ValueError(f"PUp/PGate must be [B, L, R], got {tuple(PUp.shape)} and {tuple(PGate.shape)}")
    if PUp.shape != PGate.shape:
        raise ValueError(f"PUp and PGate shapes must match, got {tuple(PUp.shape)} and {tuple(PGate.shape)}")

    B, L, R = PUp.shape
    if GateU.shape[0] != R:
        raise ValueError(f"GateU must be [R, D] with R={R}, got {tuple(GateU.shape)}")
    if UpU.shape != GateU.shape:
        raise ValueError(f"UpU shape {tuple(UpU.shape)} must match GateU shape {tuple(GateU.shape)}")
    D = int(GateU.shape[1])
    if DownV.shape != (D, R):
        raise ValueError(f"DownV must be [D, R]=[{D}, {R}], got {tuple(DownV.shape)}")
    if DownU.shape[0] != R:
        raise ValueError(f"DownU must be [R, H] with R={R}, got {tuple(DownU.shape)}")

    cfg = _pick_dual_split_token_v2_sm80_config(R, D, is_sm80=_is_sm80_device(PUp.device))
    BR = cfg["BR"] if BR is None else BR
    BD = cfg["BD"] if BD is None else BD
    BR2 = cfg["BR2"] if BR2 is None else BR2
    GD = cfg["GD"] if GD is None else GD
    num_warps = cfg["warps"] if num_warps is None else num_warps
    num_stages = cfg["stages"] if num_stages is None else num_stages

    T = int(B * L)
    p_up_2d = PUp.contiguous().reshape(T, R)
    p_gate_2d = PGate.contiguous().reshape(T, R)
    gate_u = GateU.contiguous()
    up_u = UpU.contiguous()
    down_v = DownV.contiguous()
    down_u = DownU.contiguous()

    use_fp32 = int(PUp.dtype == torch.float32)
    use_bf16 = int(PUp.dtype == torch.bfloat16)
    nd = triton.cdiv(D, BD)
    if workspace_s2d is not None:
        if workspace_s2d.shape != (T, R):
            raise ValueError(f"workspace_s2d must be {(T, R)}, got {tuple(workspace_s2d.shape)}")
        if workspace_s2d.device != PUp.device:
            raise ValueError("workspace_s2d must be on the same device as inputs")
        if workspace_s2d.dtype != torch.float32:
            raise ValueError("workspace_s2d must have dtype float32")
        s2d = workspace_s2d
        s2d.zero_()
    else:
        s2d = torch.zeros((T, R), device=PUp.device, dtype=torch.float32)

    if int(GD) == 2:
        _dual_split_phase1_atomic_token_gd2[(T, triton.cdiv(nd, 2))](
            p_up_2d,
            p_gate_2d,
            gate_u,
            up_u,
            down_v,
            s2d,
            T,
            D,
            R,
            p_up_2d.stride(0),
            p_up_2d.stride(1),
            p_gate_2d.stride(0),
            p_gate_2d.stride(1),
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
        _dual_split_phase1_atomic_token[(T, nd)](
            p_up_2d,
            p_gate_2d,
            gate_u,
            up_u,
            down_v,
            s2d,
            T,
            D,
            R,
            p_up_2d.stride(0),
            p_up_2d.stride(1),
            p_gate_2d.stride(0),
            p_gate_2d.stride(1),
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
    if s2d.dtype != PUp.dtype:
        s2d = s2d.to(PUp.dtype)
    y2d = torch.matmul(s2d, down_u) if b2 is None else torch.addmm(b2, s2d, down_u)
    return y2d.reshape(B, L, down_u.shape[1])


def flashsvd_ffn_dual_split_token_v3(
    PUp,
    PGate,
    GateU,
    UpU,
    DownV,
    DownU,
    b2=None,
    *,
    BR: int | None = None,
    BD: int | None = None,
    BR2: int | None = None,
    BH: int | None = None,
    GH: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    workspace_y: torch.Tensor | None = None,
):
    assert PUp.is_cuda and PGate.is_cuda and GateU.is_cuda and UpU.is_cuda and DownV.is_cuda and DownU.is_cuda
    if PUp.ndim != 3 or PGate.ndim != 3:
        raise ValueError(f"PUp/PGate must be [B, L, R], got {tuple(PUp.shape)} and {tuple(PGate.shape)}")
    if PUp.shape != PGate.shape:
        raise ValueError(f"PUp and PGate shapes must match, got {tuple(PUp.shape)} and {tuple(PGate.shape)}")

    B, L, R = PUp.shape
    if GateU.shape[0] != R:
        raise ValueError(f"GateU must be [R, D] with R={R}, got {tuple(GateU.shape)}")
    if UpU.shape != GateU.shape:
        raise ValueError(f"UpU shape {tuple(UpU.shape)} must match GateU shape {tuple(GateU.shape)}")
    D = int(GateU.shape[1])
    if DownV.shape != (D, R):
        raise ValueError(f"DownV must be [D, R]=[{D}, {R}], got {tuple(DownV.shape)}")
    if DownU.ndim != 2 or DownU.shape[0] != R:
        raise ValueError(f"DownU must be [R, H] with R={R}, got {tuple(DownU.shape)}")
    H = int(DownU.shape[1])

    cfg = _pick_dual_split_token_v3_config(R, D, H)
    BR = cfg["BR"] if BR is None else BR
    BD = cfg["BD"] if BD is None else BD
    BR2 = cfg["BR2"] if BR2 is None else BR2
    BH = cfg["BH"] if BH is None else BH
    GH = cfg["GH"] if GH is None else GH
    num_warps = cfg["warps"] if num_warps is None else num_warps
    num_stages = cfg["stages"] if num_stages is None else num_stages

    T = int(B * L)
    p_up_2d = PUp.contiguous().reshape(T, R)
    p_gate_2d = PGate.contiguous().reshape(T, R)
    gate_u = GateU.contiguous()
    up_u = UpU.contiguous()
    down_v = DownV.contiguous()
    down_u = DownU.contiguous()

    use_fp32 = int(PUp.dtype == torch.float32)
    use_bf16 = int(PUp.dtype == torch.bfloat16)
    nd = triton.cdiv(D, BD)
    if workspace_y is not None:
        if workspace_y.shape != (T, H):
            raise ValueError(f"workspace_y must be {(T, H)}, got {tuple(workspace_y.shape)}")
        if workspace_y.device != PUp.device:
            raise ValueError("workspace_y must be on the same device as inputs")
        if workspace_y.dtype != torch.float32:
            raise ValueError("workspace_y must have dtype float32")
        y2d = workspace_y
        y2d.zero_()
    else:
        y2d = torch.zeros((T, H), device=PUp.device, dtype=torch.float32)

    _dual_split_phase1_to_y_atomic_token[(T, nd, GH)](
        p_up_2d,
        p_gate_2d,
        gate_u,
        up_u,
        down_v,
        down_u,
        y2d,
        T,
        D,
        R,
        H,
        p_up_2d.stride(0),
        p_up_2d.stride(1),
        p_gate_2d.stride(0),
        p_gate_2d.stride(1),
        gate_u.stride(0),
        gate_u.stride(1),
        up_u.stride(0),
        up_u.stride(1),
        down_v.stride(0),
        down_v.stride(1),
        down_u.stride(0),
        down_u.stride(1),
        y2d.stride(0),
        y2d.stride(1),
        BR=BR,
        BD=BD,
        BR2=BR2,
        BH=BH,
        USE_BF16=use_bf16,
        USE_FP32=use_fp32,
        GH=GH,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if y2d.dtype != PUp.dtype:
        y2d = y2d.to(PUp.dtype)
    if b2 is not None:
        y2d = y2d + b2
    return y2d.reshape(B, L, H)


__all__ = [
    "flashsvd_ffn_dual_split_token",
    "flashsvd_ffn_dual_split_token_v2",
    "flashsvd_ffn_dual_split_token_v2_sm80",
    "flashsvd_ffn_dual_split_token_v3",
]
