#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_flashsvd_vs_svd_decode import _bench_one_mode  # noqa: E402


DEFAULT_BACKENDS = (
    "dual_split_kernel,"
    "dual_split_kernel_v2,"
    "dual_split_kernel_v2_sm80,"
    "dual_split_cublas_legacy,"
    "dual_split_cublas,"
    "auto"
)


def main() -> int:
    ap = argparse.ArgumentParser("Compare exact FlashSVD decode backends against the normal SVD baseline")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_cache_len", type=int, default=0)
    ap.add_argument("--mlp_cuda_graph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mlp_cuda_graph_scope", type=str, default="mlp", choices=["auto", "mlp", "layer_tail"])
    ap.add_argument("--backends", type=str, default=DEFAULT_BACKENDS)
    args = ap.parse_args()

    backends = [item.strip() for item in str(args.backends).split(",") if item.strip()]

    print("==== Exact Decode Backend Compare ====")
    print(
        f"Config: prompt_len={args.prompt_len} new_tokens={args.new_tokens} warmup={args.warmup} "
        f"batch={args.batch_size} dtype={args.dtype} device={args.device} "
        f"graph={int(args.mlp_cuda_graph)} scope={args.mlp_cuda_graph_scope}"
    )

    print("\n[SVD baseline]")
    baseline = _bench_one_mode(
        mode="svd",
        source=args.checkpoint,
        hf_token=args.hf_token,
        dtype_name=args.dtype,
        device=args.device,
        prompt_len=args.prompt_len,
        new_tokens=args.new_tokens,
        warmup=args.warmup,
        batch_size=args.batch_size,
        max_cache_len=args.max_cache_len,
        ffn_backend="dual_split_cublas_legacy",
        enable_mlp_graph=False,
        mlp_graph_scope="mlp",
        graph_alias_output=False,
        enable_flash_dense_attn=False,
    )

    print("\n[FlashSVD variants]")
    print("Rows: backend | decode ms/token | tok/s | speedup vs baseline")
    for backend in backends:
        result = _bench_one_mode(
            mode="flashsvd",
            source=args.checkpoint,
            hf_token=args.hf_token,
            dtype_name=args.dtype,
            device=args.device,
            prompt_len=args.prompt_len,
            new_tokens=args.new_tokens,
            warmup=args.warmup,
            batch_size=args.batch_size,
            max_cache_len=args.max_cache_len,
            ffn_backend=backend,
            enable_mlp_graph=bool(args.mlp_cuda_graph),
            mlp_graph_scope=args.mlp_cuda_graph_scope,
            graph_alias_output=False,
            enable_flash_dense_attn=False,
        )
        speedup = float(baseline["decode_ms_per_token"]) / max(float(result["decode_ms_per_token"]), 1e-9)
        print(
            f"- {backend:<24} | "
            f"{float(result['decode_ms_per_token']):8.3f} ms/token | "
            f"{float(result['decode_tok_s']):6.1f} tok/s | "
            f"{speedup:5.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
