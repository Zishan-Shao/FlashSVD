#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluator import decode_kvcache_eval
from utils.model_utils import get_model_from_source


def _set_env(name: str, value: bool) -> None:
    if value:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)


def _backend_needs_experimental_ffn(backend: str) -> bool:
    raw = str(backend).strip().lower()
    return raw in {
        "flashsvd_mlp_dual_split_prod",
        "flashsvd_mlp_dual_split_cutlass",
        "flashsvd_mlp_dual_split_triton_v1",
        "flashsvd_mlp_dual_split_triton_v2",
        "flashsvd_mlp_dual_split_triton_v2_sm80",
        "flashsvd_mlp_dual_split_triton_v3",
        "dual_split_cublas",
        "dual_split_cutlass",
        "dual_split_kernel",
        "dual_split_kernel_v2",
        "dual_split_kernel_v2_sm80",
        "dual_split_kernel_v3",
    }


def _dtype_from_name(name: str) -> torch.dtype | None:
    raw = str(name).strip().lower()
    if raw == "auto":
        return None
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _cast_model_for_eval(model, dtype_name: str):
    raw = str(dtype_name).strip().lower()
    if raw == "fp32":
        return model.float()
    if raw == "fp16":
        return model.half()
    if raw == "bf16":
        return model.to(dtype=torch.bfloat16)
    if raw == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return model.to(dtype=torch.bfloat16)
        return model.half()
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_model_and_tokenizer(source: str, *, hf_token: str | None):
    return get_model_from_source(source, hf_token=hf_token)


def _configure_mode(
    mode: str,
    *,
    ffn_backend: str,
    enable_mlp_graph: bool,
    mlp_graph_scope: str,
    graph_alias_output: bool,
    enable_flash_dense_attn: bool,
    enable_cutlass_rope_attn: bool,
    enable_baseline_dense_kvcache: bool,
) -> tuple[bool, bool, bool]:
    mode = str(mode).strip().lower()
    if mode not in {"flashsvd", "svd"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "flashsvd":
        _set_env("SVDLLM_FLASH_FALLBACK", False)
        _set_env("FLASH_SVD_DISABLE_FFN", False)
        _set_env("FLASH_SVD_BASELINE_LR_KVCACHE", False)
        _set_env("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", enable_flash_dense_attn)
        _set_env("FLASH_SVD_USE_CUTLASS_ROPEATTN", enable_cutlass_rope_attn)
        _set_env("FLASH_SVD_BASELINE_DENSE_KVCACHE", False)
        _set_env("FLASH_SVD_REFERENCE_DENSE_ATTN", False)
        _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", _backend_needs_experimental_ffn(ffn_backend))
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH", enable_mlp_graph)
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH_ALIAS_OUTPUT", graph_alias_output)
        os.environ["FLASH_SVD_MLP_CUDA_GRAPH_SCOPE"] = str(mlp_graph_scope)
        os.environ["FLASH_SVD_FFN_BACKEND"] = str(ffn_backend)
        return False, bool(enable_flash_dense_attn), False

    _set_env("SVDLLM_FLASH_FALLBACK", True)
    _set_env("FLASH_SVD_DISABLE_FFN", True)
    _set_env("FLASH_SVD_BASELINE_LR_KVCACHE", False)
    _set_env("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", False)
    _set_env("FLASH_SVD_USE_CUTLASS_ROPEATTN", False)
    _set_env("FLASH_SVD_BASELINE_DENSE_KVCACHE", enable_baseline_dense_kvcache)
    _set_env("FLASH_SVD_REFERENCE_DENSE_ATTN", enable_baseline_dense_kvcache)
    _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", False)
    _set_env("FLASH_SVD_MLP_CUDA_GRAPH", False)
    _set_env("FLASH_SVD_MLP_CUDA_GRAPH_ALIAS_OUTPUT", False)
    os.environ["FLASH_SVD_FFN_BACKEND"] = "flashsvd_mlp_dual_split_exact_legacy"
    return False, False, bool(enable_baseline_dense_kvcache)


def _bench_one_mode(
    *,
    mode: str,
    source: str,
    hf_token: str | None,
    dtype_name: str,
    device: str,
    prompt_len: int,
    new_tokens: int,
    warmup: int,
    batch_size: int,
    max_cache_len: int,
    ffn_backend: str,
    enable_mlp_graph: bool,
    mlp_graph_scope: str,
    graph_alias_output: bool,
    enable_flash_dense_attn: bool,
    enable_cutlass_rope_attn: bool,
    enable_baseline_dense_kvcache: bool,
):
    lowrank_cache, flashsvd_dense_cache, baseline_dense_kvcache = _configure_mode(
        mode,
        ffn_backend=ffn_backend,
        enable_mlp_graph=enable_mlp_graph,
        mlp_graph_scope=mlp_graph_scope,
        graph_alias_output=graph_alias_output,
        enable_flash_dense_attn=enable_flash_dense_attn,
        enable_cutlass_rope_attn=enable_cutlass_rope_attn,
        enable_baseline_dense_kvcache=enable_baseline_dense_kvcache,
    )
    model, tokenizer = _load_model_and_tokenizer(source, hf_token=hf_token)
    model.eval()
    model = _cast_model_for_eval(model, dtype_name)
    model = model.to(device)
    max_len = int(max_cache_len) if int(max_cache_len) > 0 else None
    try:
        result = decode_kvcache_eval(
            model,
            prompt_len=int(prompt_len),
            new_tokens=int(new_tokens),
            warmup=int(warmup),
            max_cache_len=max_len,
            batch_size=int(batch_size),
            device=device,
            lowrank_cache=bool(lowrank_cache),
            flashsvd_dense_cache=bool(flashsvd_dense_cache),
            baseline_dense_kvcache=bool(baseline_dense_kvcache),
            profile_decode=False,
        )
    finally:
        del tokenizer
        del model
        gc.collect()
        if torch.cuda.is_available() and "cuda" in str(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return result


def main() -> int:
    ap = argparse.ArgumentParser("Compare decode speed: FlashSVD fast path vs low-rank baseline")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Single checkpoint source used for both modes. Supports a local path, a Hugging Face repo id, "
            "or a Hugging Face repo subfolder like "
            "'namespace/repo/path/to/export' or 'namespace/repo::path/to/export'."
        ),
    )
    ap.add_argument(
        "--flashsvd_checkpoint",
        type=str,
        default=None,
        help="Checkpoint source for FlashSVD mode; same accepted formats as --checkpoint.",
    )
    ap.add_argument(
        "--svd_checkpoint",
        type=str,
        default=None,
        help="Checkpoint source for normal SVD mode; same accepted formats as --checkpoint.",
    )
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--prompt_len", type=int, default=2048)
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_cache_len", type=int, default=0)
    ap.add_argument(
        "--flashsvd_ffn_backend",
        type=str,
        default="auto",
        help="Canonical MLP backend name; old aliases remain accepted.",
    )
    ap.add_argument("--experimental_flash_dense_attn", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--cutlass_ropeattn", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--baseline_dense_kvcache", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mlp_cuda_graph", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mlp_cuda_graph_scope", type=str, default="mlp", choices=["auto", "mlp", "layer_tail"])
    ap.add_argument("--mlp_cuda_graph_alias_output", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--skip_svd", action="store_true")
    ap.add_argument("--skip_flashsvd", action="store_true")
    args = ap.parse_args()

    shared_source = args.checkpoint
    flashsvd_source = args.flashsvd_checkpoint or shared_source
    svd_source = args.svd_checkpoint or shared_source
    if not args.skip_flashsvd and not flashsvd_source:
        raise ValueError("Need --checkpoint or --flashsvd_checkpoint for FlashSVD mode.")
    if not args.skip_svd and not svd_source:
        raise ValueError("Need --checkpoint or --svd_checkpoint for SVD mode.")

    print("==== FlashSVD vs SVD Decode Benchmark ====")
    print(
        f"Config: prompt_len={args.prompt_len} new_tokens={args.new_tokens} warmup={args.warmup} "
        f"batch={args.batch_size} dtype={args.dtype} device={args.device} "
        f"ffn_backend={args.flashsvd_ffn_backend} mlp_cuda_graph={int(args.mlp_cuda_graph)} "
        f"scope={args.mlp_cuda_graph_scope} baseline_dense_kvcache={int(args.baseline_dense_kvcache)} "
        f"cutlass_ropeattn={int(args.cutlass_ropeattn)}"
    )

    results: dict[str, dict[str, float | int | bool]] = {}
    if not args.skip_svd:
        print("\n[Low-Rank baseline]")
        results["svd"] = _bench_one_mode(
            mode="svd",
            source=svd_source,
            hf_token=args.hf_token,
            dtype_name=args.dtype,
            device=args.device,
            prompt_len=args.prompt_len,
            new_tokens=args.new_tokens,
            warmup=args.warmup,
            batch_size=args.batch_size,
            max_cache_len=args.max_cache_len,
            ffn_backend=args.flashsvd_ffn_backend,
            enable_mlp_graph=args.mlp_cuda_graph,
            mlp_graph_scope=args.mlp_cuda_graph_scope,
            graph_alias_output=False,
            enable_flash_dense_attn=False,
            enable_cutlass_rope_attn=False,
            enable_baseline_dense_kvcache=bool(args.baseline_dense_kvcache),
        )

    if not args.skip_flashsvd:
        print("\n[FlashSVD]")
        results["flashsvd"] = _bench_one_mode(
            mode="flashsvd",
            source=flashsvd_source,
            hf_token=args.hf_token,
            dtype_name=args.dtype,
            device=args.device,
            prompt_len=args.prompt_len,
            new_tokens=args.new_tokens,
            warmup=args.warmup,
            batch_size=args.batch_size,
            max_cache_len=args.max_cache_len,
            ffn_backend=args.flashsvd_ffn_backend,
            enable_mlp_graph=args.mlp_cuda_graph,
            mlp_graph_scope=args.mlp_cuda_graph_scope,
            graph_alias_output=args.mlp_cuda_graph_alias_output,
            enable_flash_dense_attn=bool(args.experimental_flash_dense_attn),
            enable_cutlass_rope_attn=bool(args.cutlass_ropeattn),
            enable_baseline_dense_kvcache=False,
        )

    if "svd" in results and "flashsvd" in results:
        svd = results["svd"]
        flash = results["flashsvd"]
        speedup = float(svd["decode_ms_per_token"]) / max(float(flash["decode_ms_per_token"]), 1e-9)
        print("\n---- Summary ----")
        print(
            f"SVD decode: {float(svd['decode_ms_per_token']):.3f} ms/token | {float(svd['decode_tok_s']):,.0f} tok/s"
        )
        print(
            f"FlashSVD decode: {float(flash['decode_ms_per_token']):.3f} ms/token | {float(flash['decode_tok_s']):,.0f} tok/s"
        )
        print(f"FlashSVD speedup vs SVD: {speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
