#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_real_checkpoint_mlp import (  # noqa: E402
    _bench_backend_modes,
    _dtype_from_name,
    _load_model,
    _print_all_layer_report,
    _print_single_layer_report,
    _set_env,
)


DEFAULT_BACKENDS = (
    "baseline,"
    "dual_split_kernel,"
    "dual_split_kernel_v2,"
    "dual_split_kernel_v2_sm80,"
    "dual_split_cublas_legacy,"
    "dual_split_cublas"
)


def main() -> int:
    ap = argparse.ArgumentParser("Compare exact FlashSVD MLP backends: baseline vs v1 vs latest")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--all_layers", action="store_true")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--cuda_graph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--backends", type=str, default=DEFAULT_BACKENDS)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype = _dtype_from_name(args.dtype)
    model, _tokenizer = _load_model(args.checkpoint, hf_token=args.hf_token)
    model.eval().to(args.device)
    model = model.to(dtype=dtype)

    prev_env = {
        "SVDLLM_FLASH_FALLBACK": os.getenv("SVDLLM_FLASH_FALLBACK"),
        "FLASH_SVD_DISABLE_FFN": os.getenv("FLASH_SVD_DISABLE_FFN"),
        "FLASH_SVD_FFN_BACKEND": os.getenv("FLASH_SVD_FFN_BACKEND"),
        "FLASH_SVD_ENABLE_EXPERIMENTAL_FFN": os.getenv("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN"),
        "FLASH_SVD_MLP_CUDA_GRAPH": os.getenv("FLASH_SVD_MLP_CUDA_GRAPH"),
        "FLASH_SVD_MLP_CUDA_GRAPH_SCOPE": os.getenv("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE"),
    }
    try:
        backend_names = [item.strip() for item in str(args.backends).split(",") if item.strip()]
        if args.all_layers:
            per_layer_results = []
            for layer_idx, layer in enumerate(model.model.layers):
                mlp = layer.mlp
                hidden = int(layer.hidden_size)
                x = torch.randn(
                    int(args.batch_size),
                    int(args.seq_len),
                    hidden,
                    device=args.device,
                    dtype=dtype,
                )
                results = _bench_backend_modes(
                    mlp,
                    x,
                    backend_names=backend_names,
                    compare_graph=bool(args.cuda_graph),
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                )
                per_layer_results.append((layer_idx, results))
                del x
            _print_all_layer_report(args=args, backend_names=backend_names, per_layer_results=per_layer_results)
        else:
            layer = model.model.layers[int(args.layer)]
            mlp = layer.mlp
            hidden = int(layer.hidden_size)
            x = torch.randn(
                int(args.batch_size),
                int(args.seq_len),
                hidden,
                device=args.device,
                dtype=dtype,
            )
            results = _bench_backend_modes(
                mlp,
                x,
                backend_names=backend_names,
                compare_graph=bool(args.cuda_graph),
                warmup=int(args.warmup),
                iters=int(args.iters),
            )
            _print_single_layer_report(args=args, backend_names=backend_names, results=results)
            del x
    finally:
        for name, value in prev_env.items():
            _set_env(name, value)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
