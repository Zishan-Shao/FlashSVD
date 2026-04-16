#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUANT_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_QUANT_ROOT)
for _path in (_REPO_ROOT, _QUANT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # pragma: no cover
    import __main__ as _main

    if not hasattr(_main, "ActivationSpaceLoRAWrapper"):
        from expressivity.non_leak.svd_act_lora_aligned_adarank import ActivationSpaceLoRAWrapper as _ASLoRA

        setattr(_main, "ActivationSpaceLoRAWrapper", _ASLoRA)
except Exception:
    pass

from quant.common import fuse_lora_wrappers_inplace
from utils.model_utils import get_model_from_local


def _pick_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    key = str(name).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "none": None,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fuse LoRA / ActLoRA wrappers into base Linear layers for a repo-native checkpoint.")
    ap.add_argument("--in_ckpt", type=str, required=True)
    ap.add_argument("--out_ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def fuse_checkpoint(args: argparse.Namespace) -> dict:
    if not os.path.exists(args.in_ckpt):
        raise FileNotFoundError(args.in_ckpt)
    if os.path.exists(args.out_ckpt) and not args.overwrite:
        raise ValueError(f"--out_ckpt exists: {args.out_ckpt} (use --overwrite)")
    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)

    print(f"[Load] {args.in_ckpt}")
    model, tokenizer = get_model_from_local(args.in_ckpt)
    model.eval()
    dtype = _pick_dtype(args.dtype)
    if dtype is not None:
        model = model.to(dtype=dtype)
    if str(args.device).strip().lower() != "cpu":
        model = model.to(str(args.device))
    print("[Load] done.")

    tick = time.time()
    merged = fuse_lora_wrappers_inplace(model)
    wall = time.time() - tick
    print(f"[Fuse] merged_wrappers={merged} wall={wall:.2f}s")

    print(f"[Save] {args.out_ckpt}")
    torch.save({"model": model.cpu(), "tokenizer": tokenizer}, args.out_ckpt)
    print("[Save] done.")
    return {
        "in_ckpt": str(args.in_ckpt),
        "out_ckpt": str(args.out_ckpt),
        "merged_wrappers": int(merged),
        "wall_time_sec": float(wall),
    }


def main() -> None:
    args = build_parser().parse_args()
    fuse_checkpoint(args)


if __name__ == "__main__":
    main()
