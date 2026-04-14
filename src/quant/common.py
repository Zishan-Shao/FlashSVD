from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from quant.packed_quant_linear import PackedQuantLinear


def proj_out_features(proj) -> int:
    if proj is None:
        return 0
    try:
        out = getattr(proj, "out_features", None)
        if out is not None:
            return int(out)
    except Exception:
        pass
    try:
        weight = getattr(proj, "weight", None)
        if weight is not None and torch.is_tensor(weight):
            return int(weight.shape[0])
    except Exception:
        pass
    return 0


def _fuse_weight_lora_wrapper(module: nn.Module) -> nn.Linear | None:
    base = getattr(module, "base", None)
    lora_down = getattr(module, "lora_down", None)
    lora_up = getattr(module, "lora_up", None)
    scaling = float(getattr(module, "scaling", 1.0))

    if not isinstance(base, nn.Linear) or not isinstance(lora_down, nn.Linear) or not isinstance(lora_up, nn.Linear):
        return None

    device = base.weight.device
    dtype = base.weight.dtype
    fused = nn.Linear(
        int(base.in_features),
        int(base.out_features),
        bias=(base.bias is not None),
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        if int(lora_down.in_features) == int(base.in_features) and int(lora_up.out_features) == int(base.out_features):
            # Standard weight-space LoRA: y = W x + U(Dx) * s
            delta = torch.matmul(lora_up.weight.to(dtype), lora_down.weight.to(dtype))
            fused.weight.copy_(base.weight.to(dtype) + delta * scaling)
            if base.bias is not None:
                fused.bias.copy_(base.bias.to(dtype))
        elif int(lora_down.in_features) == int(base.out_features) and int(lora_up.out_features) == int(base.out_features):
            # Activation-space LoRA: y = z + U(Dz) * s, z = W x + b
            mix = torch.eye(int(base.out_features), device=device, dtype=dtype)
            mix = mix + torch.matmul(lora_up.weight.to(dtype), lora_down.weight.to(dtype)) * scaling
            fused.weight.copy_(torch.matmul(mix, base.weight.to(dtype)))
            if base.bias is not None:
                fused.bias.copy_(torch.matmul(mix, base.bias.to(dtype)))
        else:
            return None

    return fused


def fuse_lora_wrappers_inplace(module: nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        fused = _fuse_weight_lora_wrapper(child)
        if fused is not None:
            setattr(module, name, fused)
            replaced += 1
            continue
        replaced += fuse_lora_wrappers_inplace(child)
    return replaced


def extract_flashsvd_rank_map(model: nn.Module) -> Dict[str, Dict[str, Dict[str, int]]]:
    rank_map: Dict[str, Dict[str, Dict[str, int]]] = {}
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return rank_map

    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if attn is None or mlp is None:
            continue
        rank_map[str(int(layer_idx))] = {
            "self_attn": {
                "q_rank": proj_out_features(getattr(attn, "q_v_proj", None)),
                "k_rank": proj_out_features(getattr(attn, "k_v_proj", None)),
                "v_rank": proj_out_features(getattr(attn, "v_v_proj", None)),
                "o_rank": proj_out_features(getattr(attn, "o_v_proj", None)),
            },
            "mlp": {
                "gate_rank": proj_out_features(getattr(mlp, "gate_v_proj", None)),
                "up_rank": proj_out_features(getattr(mlp, "up_v_proj", None)),
                "down_rank": proj_out_features(getattr(mlp, "down_v_proj", None)),
            },
        }
    return rank_map


def get_submodule(root: nn.Module, name: str) -> nn.Module:
    cur: nn.Module = root
    for part in name.split("."):
        cur = getattr(cur, part)
    return cur


def set_submodule(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def replace_linears_with_packed_quant_inplace(
    model: nn.Module,
    *,
    module_names: Iterable[str],
    bits: int,
    group_size: int,
    sym: bool,
    compute_dtype: Optional[torch.dtype] = None,
) -> int:
    replaced = 0
    seen = set()
    for name in module_names:
        if not name or name in seen:
            continue
        seen.add(name)
        module = get_submodule(model, name)
        if not isinstance(module, nn.Linear):
            continue
        packed = PackedQuantLinear.from_linear(
            module,
            bits=int(bits),
            group_size=int(group_size),
            sym=bool(sym),
            compute_dtype=compute_dtype or module.weight.dtype,
        )
        set_submodule(model, name, packed)
        replaced += 1
    return replaced
