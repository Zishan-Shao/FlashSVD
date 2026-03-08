#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch

from evaluater import decode_kvcache_eval
from utils.model_utils import get_model_from_huggingface, get_model_from_local


def _set_env(name: str, value: bool) -> None:
    if value:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)


def _backend_needs_experimental_ffn(backend: str) -> bool:
    raw = str(backend).strip().lower()
    return raw in {
        "dual_split_cublas",
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
    path = Path(source)
    if path.exists():
        return get_model_from_local(str(path))
    return get_model_from_huggingface(source, hf_token=hf_token)


def _configure_mode(
    mode: str,
    *,
    ffn_backend: str,
    enable_mlp_graph: bool,
    mlp_graph_scope: str,
    graph_alias_output: bool,
    enable_flash_dense_attn: bool,
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
    _set_env("FLASH_SVD_BASELINE_DENSE_KVCACHE", enable_baseline_dense_kvcache)
    _set_env("FLASH_SVD_REFERENCE_DENSE_ATTN", enable_baseline_dense_kvcache)
    _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", False)
    _set_env("FLASH_SVD_MLP_CUDA_GRAPH", False)
    _set_env("FLASH_SVD_MLP_CUDA_GRAPH_ALIAS_OUTPUT", False)
    os.environ["FLASH_SVD_FFN_BACKEND"] = "dual_split_cublas_legacy"
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
    enable_baseline_dense_kvcache: bool,
):
    lowrank_cache, flashsvd_dense_cache, baseline_dense_kvcache = _configure_mode(
        mode,
        ffn_backend=ffn_backend,
        enable_mlp_graph=enable_mlp_graph,
        mlp_graph_scope=mlp_graph_scope,
        graph_alias_output=graph_alias_output,
        enable_flash_dense_attn=enable_flash_dense_attn,
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
    ap.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint path/model id used for both modes.")
    ap.add_argument("--flashsvd_checkpoint", type=str, default=None, help="Checkpoint path/model id for FlashSVD mode.")
    ap.add_argument("--svd_checkpoint", type=str, default=None, help="Checkpoint path/model id for normal SVD mode.")
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--prompt_len", type=int, default=2048)
    ap.add_argument("--new_tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_cache_len", type=int, default=0)
    ap.add_argument("--flashsvd_ffn_backend", type=str, default="auto", choices=["auto", "dual_split_cublas", "dual_split_cublas_legacy", "dual_split_kernel", "dual_split_kernel_v2", "dual_split_kernel_v2_sm80", "dual_split_kernel_v3"])
    ap.add_argument("--experimental_flash_dense_attn", action=argparse.BooleanOptionalAction, default=False)
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
        f"scope={args.mlp_cuda_graph_scope} baseline_dense_kvcache={int(args.baseline_dense_kvcache)}"
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
