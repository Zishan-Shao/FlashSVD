#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.model_utils import ensure_transformers_config_compat


def _load_checkpoint(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _empty_rope_cache_like(tensor: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    if tensor.dim() == 4:
        shape = (tensor.shape[0], tensor.shape[1], 0, tensor.shape[3])
    elif tensor.dim() == 3:
        shape = (tensor.shape[0], 0, tensor.shape[2])
    elif tensor.dim() == 2:
        shape = (0, tensor.shape[1])
    else:
        shape = (0, head_dim)
    return torch.empty(shape, dtype=tensor.dtype, device="cpu")


def _replace_buffer(module, name: str, value: torch.Tensor) -> None:
    if hasattr(module, "_buffers") and name in module._buffers:
        module._buffers[name] = value
    setattr(module, name, value)


def _strip_model_rope_cache(model) -> int:
    stripped = 0
    for module in model.modules():
        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            continue

        head_dim = int(getattr(rotary, "dim", 0) or 0)
        if head_dim <= 0:
            inv_freq = getattr(rotary, "inv_freq", None)
            if isinstance(inv_freq, torch.Tensor):
                head_dim = int(inv_freq.numel() * 2)

        for name in ("cos_cached", "sin_cached"):
            cache = getattr(rotary, name, None)
            if isinstance(cache, torch.Tensor):
                empty = _empty_rope_cache_like(cache, head_dim=head_dim)
            else:
                empty = torch.empty((0, max(1, head_dim)), dtype=torch.float32, device="cpu")
            _replace_buffer(rotary, name, empty)
            stripped += 1

        if hasattr(rotary, "max_seq_len_cached"):
            rotary.max_seq_len_cached = 0
    return stripped


def _extract_model(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    if hasattr(obj, "modules"):
        return obj
    return None


def _dtype_from_name(name: str | None):
    if name is None:
        return None
    raw = str(name).strip().lower()
    if raw in {"none", "keep", "original"}:
        return None
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> int:
    ap = argparse.ArgumentParser("Strip RoPE caches from pickled model checkpoints")
    ap.add_argument("--input", required=True, type=str)
    ap.add_argument("--output", required=True, type=str)
    ap.add_argument("--dtype", default="fp16", type=str)
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.is_file():
        raise FileNotFoundError(src)
    if src.resolve() == dst.resolve():
        raise ValueError("Input and output paths must differ.")

    obj = _load_checkpoint(src)
    model = _extract_model(obj)
    if model is None:
        raise ValueError(f"Checkpoint does not contain a model object: {src}")

    target_dtype = _dtype_from_name(args.dtype)
    if target_dtype is not None:
        model = model.to(dtype=target_dtype)
    ensure_transformers_config_compat(model)
    stripped = _strip_model_rope_cache(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(dst))

    src_bytes = int(src.stat().st_size)
    dst_bytes = int(dst.stat().st_size)
    print(f"input={src}")
    print(f"output={dst}")
    print(f"stripped_buffers={stripped}")
    print(f"input_bytes={src_bytes}")
    print(f"output_bytes={dst_bytes}")
    print(f"dtype={args.dtype}")
    if src_bytes > 0:
        print(f"shrink_ratio={dst_bytes / src_bytes:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
