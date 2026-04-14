from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


def _pick_dual_split_pcat_s_config(R: int, D: int) -> dict[str, int]:
    if max(R, D) >= 8192:
        cfg = {"BT": 16, "BR": 64, "BD": 64, "BR2": 128, "warps": 4, "stages": 2}
    elif R >= 1024:
        cfg = {"BT": 16, "BR": 64, "BD": 64, "BR2": 128, "warps": 4, "stages": 2}
    elif R >= 512:
        cfg = {"BT": 16, "BR": 64, "BD": 64, "BR2": 128, "warps": 4, "stages": 2}
    else:
        cfg = {"BT": 16, "BR": 32, "BD": 64, "BR2": 64, "warps": 4, "stages": 2}
    for key, env_name in (
        ("BT", "FLASH_SVD_DUAL_SPLIT_PCAT_S_BT"),
        ("BR", "FLASH_SVD_DUAL_SPLIT_PCAT_S_BR"),
        ("BD", "FLASH_SVD_DUAL_SPLIT_PCAT_S_BD"),
        ("BR2", "FLASH_SVD_DUAL_SPLIT_PCAT_S_BR2"),
        ("warps", "FLASH_SVD_DUAL_SPLIT_PCAT_S_WARPS"),
        ("stages", "FLASH_SVD_DUAL_SPLIT_PCAT_S_STAGES"),
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
def _dual_split_pcat_to_s_atomic_token(
    PCat_ptr,
    GateU_ptr,
    UpU_ptr,
    DownV_ptr,
    S_ptr,
    T,
    D,
    R,
    sPCat_t,
    sPCat_r,
    sGate_r,
    sGate_d,
    sUp_r,
    sUp_d,
    sDownV_d,
    sDownV_r,
    sS_t,
    sS_r,
    BT: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
    BR2: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32: tl.constexpr,
):
    pid_tb = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_t = pid_tb * BT + tl.arange(0, BT)
    mask_t = offs_t < T
    offs_d = pid_n * BD + tl.arange(0, BD)
    mask_d = offs_d < D
    in_dtype = tl.float32 if USE_FP32 else (tl.bfloat16 if USE_BF16 else tl.float16)

    gate_acc = tl.zeros((BT, BD), dtype=tl.float32)
    up_acc = tl.zeros((BT, BD), dtype=tl.float32)

    for r0 in range(0, R, BR):
        offs_r = r0 + tl.arange(0, BR)
        mask_r = offs_r < R
        p_up = tl.load(
            PCat_ptr + offs_t[:, None] * sPCat_t + offs_r[None, :] * sPCat_r,
            mask=mask_t[:, None] & mask_r[None, :],
            other=0.0,
        ).to(in_dtype)
        p_gate = tl.load(
            PCat_ptr + offs_t[:, None] * sPCat_t + (R + offs_r)[None, :] * sPCat_r,
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
        tl.atomic_add(
            S_ptr + offs_t[:, None] * sS_t + offs_r2[None, :] * sS_r,
            s_acc,
            mask=mask_t[:, None] & mask_r2[None, :],
        )


def flashsvd_mlp_dual_split_triton_pcat_s(
    PCat,
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
    num_warps: int | None = None,
    num_stages: int | None = None,
    workspace_s2d: torch.Tensor | None = None,
):
    assert PCat.is_cuda and GateU.is_cuda and UpU.is_cuda and DownV.is_cuda and DownU.is_cuda
    if PCat.ndim != 3:
        raise ValueError(f"PCat must be [B, L, 2R], got {tuple(PCat.shape)}")

    B, L, two_r = PCat.shape
    if two_r % 2 != 0:
        raise ValueError(f"PCat last dim must be even, got {two_r}")
    R = int(two_r // 2)
    if GateU.shape[0] != R:
        raise ValueError(f"GateU must be [R, D] with R={R}, got {tuple(GateU.shape)}")
    if UpU.shape != GateU.shape:
        raise ValueError(f"UpU shape {tuple(UpU.shape)} must match GateU shape {tuple(GateU.shape)}")
    D = int(GateU.shape[1])
    if DownV.shape != (D, R):
        raise ValueError(f"DownV must be [D, R]=[{D}, {R}], got {tuple(DownV.shape)}")
    if DownU.ndim != 2 or DownU.shape[0] != R:
        raise ValueError(f"DownU must be [R, H] with R={R}, got {tuple(DownU.shape)}")

    cfg = _pick_dual_split_pcat_s_config(R, D)
    BT = cfg["BT"] if BT is None else BT
    BR = cfg["BR"] if BR is None else BR
    BD = cfg["BD"] if BD is None else BD
    BR2 = cfg["BR2"] if BR2 is None else BR2
    num_warps = cfg["warps"] if num_warps is None else num_warps
    num_stages = cfg["stages"] if num_stages is None else num_stages

    T = int(B * L)
    p_cat_2d = PCat.contiguous().reshape(T, 2 * R)
    gate_u = GateU.contiguous()
    up_u = UpU.contiguous()
    down_v = DownV.contiguous()
    down_u = DownU.contiguous()
    use_fp32 = int(PCat.dtype == torch.float32)
    use_bf16 = int(PCat.dtype == torch.bfloat16)

    if workspace_s2d is not None:
        if workspace_s2d.shape != (T, R):
            raise ValueError(f"workspace_s2d must be {(T, R)}, got {tuple(workspace_s2d.shape)}")
        if workspace_s2d.device != PCat.device:
            raise ValueError("workspace_s2d must be on the same device as inputs")
        if workspace_s2d.dtype != torch.float32:
            raise ValueError("workspace_s2d must have dtype float32")
        s2d = workspace_s2d
        s2d.zero_()
    else:
        s2d = torch.zeros((T, R), device=PCat.device, dtype=torch.float32)

    nd = triton.cdiv(D, BD)
    grid = (triton.cdiv(T, BT), nd)
    _dual_split_pcat_to_s_atomic_token[grid](
        p_cat_2d,
        gate_u,
        up_u,
        down_v,
        s2d,
        T,
        D,
        R,
        p_cat_2d.stride(0),
        p_cat_2d.stride(1),
        gate_u.stride(0),
        gate_u.stride(1),
        up_u.stride(0),
        up_u.stride(1),
        down_v.stride(0),
        down_v.stride(1),
        s2d.stride(0),
        s2d.stride(1),
        BT=BT,
        BR=BR,
        BD=BD,
        BR2=BR2,
        USE_BF16=use_bf16,
        USE_FP32=use_fp32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    s2d_in = s2d if s2d.dtype == PCat.dtype else s2d.to(PCat.dtype)
    y2d = torch.matmul(s2d_in, down_u) if b2 is None else torch.addmm(b2, s2d_in, down_u)
    return y2d.reshape(B, L, down_u.shape[1])


__all__ = [
    "flashsvd_mlp_dual_split_triton_pcat_s",
]
