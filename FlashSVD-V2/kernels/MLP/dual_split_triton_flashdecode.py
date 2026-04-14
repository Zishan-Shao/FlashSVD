from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _pick_dual_split_flashdecode_config(R: int, D: int, H: int) -> dict[str, int]:
    if max(R, D, H) >= 8192:
        cfg = {"BT": 16, "BR": 64, "BD": 64, "BR2": 64, "BH": 128, "GH": 4, "warps": 8, "stages": 2}
    elif max(R, H) >= 4096:
        cfg = {"BT": 16, "BR": 64, "BD": 64, "BR2": 64, "BH": 128, "GH": 4, "warps": 8, "stages": 2}
    else:
        cfg = {"BT": 16, "BR": 32, "BD": 64, "BR2": 64, "BH": 128, "GH": 2, "warps": 4, "stages": 2}
    for key, env_name in (
        ("BT", "FLASH_SVD_DUAL_SPLIT_FLASHDECODE_BT"),
        ("BR", "FLASH_SVD_DUAL_SPLIT_FLASHDECODE_BR"),
        ("BD", "FLASH_SVD_DUAL_SPLIT_FLASHDECODE_BD"),
        ("BR2", "FLASH_SVD_DUAL_SPLIT_FLASHDECODE_BR2"),
        ("BH", "FLASH_SVD_DUAL_SPLIT_FLASHDECODE_BH"),
        ("GH", "FLASH_SVD_DUAL_SPLIT_FLASHDECODE_GH"),
        ("warps", "FLASH_SVD_DUAL_SPLIT_FLASHDECODE_WARPS"),
        ("stages", "FLASH_SVD_DUAL_SPLIT_FLASHDECODE_STAGES"),
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
def _dual_split_flashdecode_to_y_atomic_token(
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
    BT: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
    BR2: tl.constexpr,
    BH: tl.constexpr,
    GH: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32: tl.constexpr,
):
    pid_tb = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_t = pid_tb * BT + tl.arange(0, BT)
    mask_t = offs_t < T

    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    in_dtype = tl.float32 if USE_FP32 else (tl.bfloat16 if USE_BF16 else tl.float16)
    gate_acc = tl.zeros((BT, BD), dtype=tl.float32)
    up_acc = tl.zeros((BT, BD), dtype=tl.float32)

    for r0 in range(0, R, BR):
        offs_r = r0 + tl.arange(0, BR)
        mask_r = offs_r < R

        p_up = tl.load(
            PUp_ptr + offs_t[:, None] * sPUp_t + offs_r[None, :] * sPUp_r,
            mask=mask_t[:, None] & mask_r[None, :],
            other=0.0,
        ).to(in_dtype)
        p_gate = tl.load(
            PGate_ptr + offs_t[:, None] * sPGate_t + offs_r[None, :] * sPGate_r,
            mask=mask_t[:, None] & mask_r[None, :],
            other=0.0,
        ).to(in_dtype)

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

        gate_acc = tl.dot(p_gate, gate_blk, acc=gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(p_up, up_blk, acc=up_acc, out_dtype=tl.float32)

    h = ((gate_acc * tl.sigmoid(gate_acc)) * up_acc).to(in_dtype)
    for h0 in range(pid_g * BH, H, GH * BH):
        offs_h = h0 + tl.arange(0, BH)
        mask_h = offs_h < H
        y_acc = tl.zeros((BT, BH), dtype=tl.float32)

        for r2_0 in range(0, R, BR2):
            offs_r2 = r2_0 + tl.arange(0, BR2)
            mask_r2 = offs_r2 < R

            downv_blk = tl.load(
                DownV_ptr + offs_d[:, None] * sDownV_d + offs_r2[None, :] * sDownV_r,
                mask=mask_d[:, None] & mask_r2[None, :],
                other=0.0,
            ).to(in_dtype)
            s_acc = tl.zeros((BT, BR2), dtype=tl.float32)
            s_acc = tl.dot(h, downv_blk, acc=s_acc, out_dtype=tl.float32)
            s_acc = s_acc.to(in_dtype)

            downu_blk = tl.load(
                DownU_ptr + offs_r2[:, None] * sDownU_r + offs_h[None, :] * sDownU_h,
                mask=mask_r2[:, None] & mask_h[None, :],
                other=0.0,
            ).to(in_dtype)
            y_acc = tl.dot(s_acc, downu_blk, acc=y_acc, out_dtype=tl.float32)

        tl.atomic_add(
            Y_ptr + offs_t[:, None] * sY_t + offs_h[None, :] * sY_h,
            y_acc,
            mask=mask_t[:, None] & mask_h[None, :],
        )


def flashsvd_mlp_dual_split_triton_flashdecode(
    PUp,
    PGate,
    GateU,
    UpU,
    DownV,
    DownU,
    b2=None,
    *,
    BT: int | None = None,
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

    cfg = _pick_dual_split_flashdecode_config(int(R), int(D), int(H))
    BT = cfg["BT"] if BT is None else BT
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

    nd = triton.cdiv(D, BD)
    grid = (triton.cdiv(T, BT), nd, GH)
    _dual_split_flashdecode_to_y_atomic_token[grid](
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
        BT=BT,
        BR=BR,
        BD=BD,
        BR2=BR2,
        BH=BH,
        GH=GH,
        USE_BF16=use_bf16,
        USE_FP32=use_fp32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if y2d.dtype != PUp.dtype:
        y2d = y2d.to(PUp.dtype)
    if b2 is not None:
        y2d = y2d + b2
    return y2d.reshape(B, L, H)


__all__ = [
    "flashsvd_mlp_dual_split_triton_flashdecode",
]
