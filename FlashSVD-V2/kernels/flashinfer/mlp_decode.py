from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


_FLASHINFER_TINYGEMM_BF16 = None
_FLASHINFER_TINYGEMM_BF16_RESOLVED = False


def get_flashinfer_tinygemm_bf16():
    global _FLASHINFER_TINYGEMM_BF16
    global _FLASHINFER_TINYGEMM_BF16_RESOLVED
    if _FLASHINFER_TINYGEMM_BF16_RESOLVED:
        return _FLASHINFER_TINYGEMM_BF16

    fn = None
    try:
        import flashinfer  # type: ignore

        gemm_mod = getattr(flashinfer, "gemm", None)
        if gemm_mod is not None:
            fn = getattr(gemm_mod, "tinygemm_bf16", None)
    except Exception:
        fn = None

    _FLASHINFER_TINYGEMM_BF16 = fn
    _FLASHINFER_TINYGEMM_BF16_RESOLVED = True
    return _FLASHINFER_TINYGEMM_BF16


def has_flashinfer_mlp_tinygemm() -> bool:
    return get_flashinfer_tinygemm_bf16() is not None


def _pad_multiple(value: int, multiple: int) -> int:
    value = int(value)
    multiple = int(multiple)
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _cuda_sm_major(device: torch.device) -> int:
    try:
        major, _minor = torch.cuda.get_device_capability(device)
    except Exception:
        return 0
    return int(major)


def _clone_weight_t(weight: torch.Tensor) -> torch.Tensor:
    return weight.t().contiguous()


@dataclass
class _TinyGemmWeightPack:
    weight_row_major: torch.Tensor
    in_features: int
    out_features: int
    padded_in_features: int
    padded_out_features: int


@dataclass
class FlashInferMLPWorkspace:
    rows: int
    backend: str
    p_cat: torch.Tensor
    up: torch.Tensor
    hidden: torch.Tensor
    down_mid: torch.Tensor
    out: torch.Tensor
    p_up: Optional[torch.Tensor] = None
    p_gate: Optional[torch.Tensor] = None


@dataclass
class FlashInferMLPDecodePlan:
    v_cat_t: torch.Tensor
    gate_u_t: torch.Tensor
    up_u_t: torch.Tensor
    down_v_t: torch.Tensor
    down_u_t: torch.Tensor
    up_rank: int
    gate_rank: int
    down_rank: int
    hidden_size: int
    intermediate_size: int
    backend_hint: str = "auto"
    tinygemm_gate_u: Optional[_TinyGemmWeightPack] = None
    tinygemm_up_u: Optional[_TinyGemmWeightPack] = None
    tinygemm_down_v: Optional[_TinyGemmWeightPack] = None
    tinygemm_down_u: Optional[_TinyGemmWeightPack] = None
    metadata: dict[str, object] = field(default_factory=dict)
    workspace_cache: dict[tuple[str, int], FlashInferMLPWorkspace] = field(default_factory=dict, repr=False)


def _build_tinygemm_weight_pack(weight_row_major: torch.Tensor) -> _TinyGemmWeightPack:
    out_features, in_features = map(int, weight_row_major.shape)
    padded_in_features = _pad_multiple(in_features, 64)
    padded_out_features = _pad_multiple(out_features, 16)
    packed = torch.zeros(
        (padded_out_features, padded_in_features),
        device=weight_row_major.device,
        dtype=weight_row_major.dtype,
    )
    packed[:out_features, :in_features].copy_(weight_row_major)
    return _TinyGemmWeightPack(
        weight_row_major=packed.contiguous(),
        in_features=in_features,
        out_features=out_features,
        padded_in_features=padded_in_features,
        padded_out_features=padded_out_features,
    )


def build_flashinfer_mlp_decode_plan(
    *,
    up_v_weight: torch.Tensor,
    gate_v_weight: torch.Tensor,
    gate_u_weight: torch.Tensor,
    up_u_weight: torch.Tensor,
    down_v_weight: torch.Tensor,
    down_u_weight: torch.Tensor,
    backend_hint: str = "auto",
    metadata: Optional[dict[str, object]] = None,
) -> FlashInferMLPDecodePlan:
    v_cat = torch.cat((up_v_weight, gate_v_weight), dim=0).contiguous()
    plan = FlashInferMLPDecodePlan(
        v_cat_t=_clone_weight_t(v_cat),
        gate_u_t=_clone_weight_t(gate_u_weight),
        up_u_t=_clone_weight_t(up_u_weight),
        down_v_t=_clone_weight_t(down_v_weight),
        down_u_t=_clone_weight_t(down_u_weight),
        up_rank=int(up_v_weight.shape[0]),
        gate_rank=int(gate_v_weight.shape[0]),
        down_rank=int(down_v_weight.shape[0]),
        hidden_size=int(up_v_weight.shape[1]),
        intermediate_size=int(gate_u_weight.shape[0]),
        backend_hint=str(backend_hint or "auto"),
        metadata=dict(metadata or {}),
    )
    if plan.v_cat_t.is_cuda and plan.v_cat_t.dtype == torch.bfloat16 and has_flashinfer_mlp_tinygemm():
        plan.tinygemm_gate_u = _build_tinygemm_weight_pack(gate_u_weight)
        plan.tinygemm_up_u = _build_tinygemm_weight_pack(up_u_weight)
        plan.tinygemm_down_v = _build_tinygemm_weight_pack(down_v_weight)
        plan.tinygemm_down_u = _build_tinygemm_weight_pack(down_u_weight)
    return plan


def _can_use_tinygemm_tail(plan: FlashInferMLPDecodePlan, x_2d: torch.Tensor) -> bool:
    return bool(
        x_2d.is_cuda
        and x_2d.dtype == torch.bfloat16
        and int(x_2d.shape[0]) <= 8
        and _cuda_sm_major(x_2d.device) >= 9
        and plan.tinygemm_gate_u is not None
        and plan.tinygemm_up_u is not None
        and plan.tinygemm_down_v is not None
        and plan.tinygemm_down_u is not None
        and plan.intermediate_size % 64 == 0
        and plan.hidden_size % 16 == 0
    )


def select_flashinfer_mlp_backend(
    plan: FlashInferMLPDecodePlan,
    x: torch.Tensor,
    *,
    backend: str | None = None,
) -> str:
    raw = str(backend or plan.backend_hint or "auto").strip().lower().replace("-", "_")
    x_2d = x.reshape(-1, x.shape[-1])
    if raw in {"tinygemm", "tinygemm_tail"}:
        return "tinygemm_tail" if _can_use_tinygemm_tail(plan, x_2d) else "torch_mm"
    if raw in {"torch", "torch_mm", "mm"}:
        return "torch_mm"
    return "tinygemm_tail" if _can_use_tinygemm_tail(plan, x_2d) else "torch_mm"


def _alloc_workspace(
    plan: FlashInferMLPDecodePlan,
    *,
    rows: int,
    backend: str,
) -> FlashInferMLPWorkspace:
    device = plan.v_cat_t.device
    dtype = plan.v_cat_t.dtype
    p_up = None
    p_gate = None
    if backend == "tinygemm_tail":
        up_cols = int(plan.tinygemm_up_u.padded_in_features)  # type: ignore[union-attr]
        gate_cols = int(plan.tinygemm_gate_u.padded_in_features)  # type: ignore[union-attr]
        down_cols = int(plan.tinygemm_down_v.padded_out_features)  # type: ignore[union-attr]
        p_up = torch.empty((rows, up_cols), device=device, dtype=dtype)
        p_gate = torch.empty((rows, gate_cols), device=device, dtype=dtype)
    else:
        down_cols = int(plan.down_rank)
    return FlashInferMLPWorkspace(
        rows=int(rows),
        backend=str(backend),
        p_cat=torch.empty((rows, int(plan.up_rank + plan.gate_rank)), device=device, dtype=dtype),
        up=torch.empty((rows, int(plan.intermediate_size)), device=device, dtype=dtype),
        hidden=torch.empty((rows, int(plan.intermediate_size)), device=device, dtype=dtype),
        down_mid=torch.empty((rows, down_cols), device=device, dtype=dtype),
        out=torch.empty((rows, int(plan.hidden_size)), device=device, dtype=dtype),
        p_up=p_up,
        p_gate=p_gate,
    )


def _get_workspace(
    plan: FlashInferMLPDecodePlan,
    *,
    rows: int,
    backend: str,
) -> FlashInferMLPWorkspace:
    key = (str(backend), int(rows))
    cached = plan.workspace_cache.get(key)
    if cached is not None:
        return cached
    workspace = _alloc_workspace(plan, rows=int(rows), backend=str(backend))
    plan.workspace_cache[key] = workspace
    return workspace


def _run_tinygemm(
    inp: torch.Tensor,
    pack: _TinyGemmWeightPack,
    out: torch.Tensor,
) -> torch.Tensor:
    tinygemm = get_flashinfer_tinygemm_bf16()
    if tinygemm is None:
        raise RuntimeError("flashinfer.gemm.tinygemm_bf16 is not available.")
    tinygemm(inp, pack.weight_row_major, out)
    return out


def _run_torch_linear(
    inp: torch.Tensor,
    weight_t: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.mm(inp, weight_t, out=out)
    return out


def _run_silu_mul_(
    *,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    torch.nn.functional.silu(gate, inplace=True)
    gate.mul_(up)
    return gate


def flashsvd_flashinfer_mlp_decode(
    x: torch.Tensor,
    plan: FlashInferMLPDecodePlan,
    *,
    backend: str | None = None,
    alias_output: bool = False,
) -> torch.Tensor:
    if x.dim() < 2:
        raise ValueError(f"Expected x to have at least 2 dims, got {tuple(x.shape)}")
    if int(x.shape[-1]) != int(plan.hidden_size):
        raise ValueError(f"Expected hidden_size={plan.hidden_size}, got {int(x.shape[-1])}")

    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    selected_backend = select_flashinfer_mlp_backend(plan, x_2d, backend=backend)
    workspace = _get_workspace(plan, rows=int(x_2d.shape[0]), backend=selected_backend)

    _run_torch_linear(x_2d, plan.v_cat_t, workspace.p_cat)
    if selected_backend == "tinygemm_tail":
        if workspace.p_up is None or workspace.p_gate is None:
            raise RuntimeError("tinygemm_tail workspace is missing packed rank buffers")
        workspace.p_up.zero_()
        workspace.p_gate.zero_()
        workspace.p_up[:, : int(plan.up_rank)].copy_(workspace.p_cat[:, : int(plan.up_rank)])
        workspace.p_gate[:, : int(plan.gate_rank)].copy_(
            workspace.p_cat[:, int(plan.up_rank) : int(plan.up_rank + plan.gate_rank)]
        )
        _run_tinygemm(workspace.p_gate, plan.tinygemm_gate_u, workspace.hidden)  # type: ignore[arg-type]
        _run_tinygemm(workspace.p_up, plan.tinygemm_up_u, workspace.up)  # type: ignore[arg-type]
        _run_silu_mul_(gate=workspace.hidden, up=workspace.up)
        _run_tinygemm(workspace.hidden, plan.tinygemm_down_v, workspace.down_mid)  # type: ignore[arg-type]
        _run_tinygemm(workspace.down_mid, plan.tinygemm_down_u, workspace.out)  # type: ignore[arg-type]
    else:
        p_up = workspace.p_cat.narrow(1, 0, int(plan.up_rank))
        p_gate = workspace.p_cat.narrow(1, int(plan.up_rank), int(plan.gate_rank))
        _run_torch_linear(p_gate, plan.gate_u_t, workspace.hidden)
        _run_torch_linear(p_up, plan.up_u_t, workspace.up)
        _run_silu_mul_(gate=workspace.hidden, up=workspace.up)
        _run_torch_linear(workspace.hidden, plan.down_v_t, workspace.down_mid)
        _run_torch_linear(workspace.down_mid, plan.down_u_t, workspace.out)

    out = workspace.out.view(*x.shape[:-1], int(plan.hidden_size))
    return out if alias_output else out.clone()


__all__ = [
    "FlashInferMLPDecodePlan",
    "FlashInferMLPWorkspace",
    "build_flashinfer_mlp_decode_plan",
    "flashsvd_flashinfer_mlp_decode",
    "get_flashinfer_tinygemm_bf16",
    "has_flashinfer_mlp_tinygemm",
    "select_flashinfer_mlp_backend",
]
