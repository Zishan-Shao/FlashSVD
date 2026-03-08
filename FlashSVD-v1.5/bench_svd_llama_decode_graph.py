#!/usr/bin/env python3
import argparse
import contextlib
import os
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.flashsvdswiglu import flashsvd_ffn_swiglu
from flashsvd_component.svd_llama import (
    SVD_LlamaMLP,
    enable_flashsvd_llama_layer_tail_cuda_graph,
)
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FlashSVD Llama decode graph scopes.")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    return parser.parse_args()


@contextlib.contextmanager
def temporary_env(updates: dict[str, str | None]):
    old = {}
    for key, value in updates.items():
        old[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def bench_ms(fn, warmup: int, runs: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        out = fn()
        _ = out.reshape(-1)[0]
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
        _ = out.reshape(-1)[0]
    times.sort()
    return times[len(times) // 2]


def llama_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + float(eps))
    return weight * hidden_states.to(input_dtype)


class DummySelfAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states), None


def build_layer(hidden_size: int, intermediate_size: int, ratio: float, num_heads: int, num_kv_heads: int, dtype: torch.dtype, device: torch.device):
    cfg = LlamaConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        num_hidden_layers=1,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
    )
    layer = LlamaDecoderLayer(cfg, layer_idx=0).to(device=device, dtype=dtype).eval()
    layer.self_attn = DummySelfAttention().to(device=device)
    layer.mlp = SVD_LlamaMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        hidden_act="silu",
        ratio=ratio,
    ).to(device=device, dtype=dtype).eval()
    return layer


def run_kernel_bench(layer, hidden_states: torch.Tensor, warmup: int, runs: int):
    mlp = layer.mlp
    norm = layer.post_attention_layernorm
    norm_eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
    normed = llama_rms_norm(hidden_states, norm.weight, norm_eps)
    V1, U2, V2, b1, b2 = mlp._get_flashsvd_cached_factors(hidden_states.device, hidden_states.dtype)
    P = mlp.up_v_proj(normed)

    def old_kernel():
        return flashsvd_ffn_swiglu(
            P,
            V1,
            U2,
            V2,
            b1,
            b2,
            BL=64,
            BD=128,
            BR1=64,
            BR2=64,
            use_autotune=False,
            num_warps=4,
            num_stages=2,
            use_decode_specialized=False,
        )

    def new_kernel():
        return flashsvd_ffn_swiglu(
            P,
            V1,
            U2,
            V2,
            b1,
            b2,
            use_autotune=True,
        )

    ref = old_kernel()
    cur = new_kernel()
    diff = float((ref - cur).abs().max().item())
    old_ms = bench_ms(old_kernel, warmup=warmup, runs=runs)
    new_ms = bench_ms(new_kernel, warmup=warmup, runs=runs)
    return {
        "old_ms": old_ms,
        "new_ms": new_ms,
        "speedup": old_ms / new_ms,
        "diff": diff,
    }


def run_layer_mode(layer, hidden_states: torch.Tensor, mode: str):
    if mode == "eager":
        env = {
            "FLASH_SVD_MLP_CUDA_GRAPH": "0",
            "FLASH_SVD_MLP_CUDA_GRAPH_SCOPE": None,
        }
    elif mode == "mlp_graph":
        env = {
            "FLASH_SVD_MLP_CUDA_GRAPH": "1",
            "FLASH_SVD_MLP_CUDA_GRAPH_SCOPE": "mlp",
        }
    elif mode == "layer_tail_graph":
        env = {
            "FLASH_SVD_MLP_CUDA_GRAPH": "1",
            "FLASH_SVD_MLP_CUDA_GRAPH_SCOPE": "layer_tail",
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    def fn():
        with temporary_env(env):
            return layer(hidden_states, use_cache=False, output_attentions=False)[0]

    return fn


def run_layer_bench(layer, hidden_states: torch.Tensor, warmup: int, runs: int):
    outputs = {}
    medians = {}
    for mode in ("eager", "mlp_graph", "layer_tail_graph"):
        fn = run_layer_mode(layer, hidden_states, mode)
        outputs[mode] = fn()
        medians[mode] = bench_ms(fn, warmup=warmup, runs=runs)

    return {
        "eager_ms": medians["eager"],
        "mlp_graph_ms": medians["mlp_graph"],
        "layer_tail_graph_ms": medians["layer_tail_graph"],
        "mlp_graph_speedup": medians["eager"] / medians["mlp_graph"],
        "layer_tail_graph_speedup": medians["eager"] / medians["layer_tail_graph"],
        "eager_vs_mlp_graph": float((outputs["eager"] - outputs["mlp_graph"]).abs().max().item()),
        "eager_vs_layer_tail_graph": float((outputs["eager"] - outputs["layer_tail_graph"]).abs().max().item()),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    enable_flashsvd_llama_layer_tail_cuda_graph()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    print(
        f"Config: H={args.hidden_size} D={args.intermediate_size} ratio={args.ratio} "
        f"dtype={args.dtype} heads={args.num_heads}/{args.num_kv_heads} seq={args.seq_len}"
    )

    for batch_size in args.batches:
        torch.manual_seed(0)
        layer = build_layer(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            ratio=args.ratio,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            dtype=dtype,
            device=device,
        )
        hidden_states = torch.randn(
            batch_size,
            args.seq_len,
            args.hidden_size,
            device=device,
            dtype=dtype,
        )
        rank = int(layer.mlp.up_v_proj.out_features)

        kernel_stats = run_kernel_bench(layer, hidden_states, warmup=args.warmup, runs=args.runs)
        layer_stats = run_layer_bench(layer, hidden_states, warmup=args.warmup, runs=args.runs)

        print(f"\n[B={batch_size}, L={args.seq_len}, rank={rank}]")
        print(
            "kernel: "
            f"old={kernel_stats['old_ms']:.4f} ms "
            f"new={kernel_stats['new_ms']:.4f} ms "
            f"speedup={kernel_stats['speedup']:.3f}x "
            f"max|old-new|={kernel_stats['diff']:.6g}"
        )
        print(
            "layer: "
            f"eager={layer_stats['eager_ms']:.4f} ms "
            f"mlp_graph={layer_stats['mlp_graph_ms']:.4f} ms "
            f"layer_tail_graph={layer_stats['layer_tail_graph_ms']:.4f} ms"
        )
        print(
            "layer speedup: "
            f"mlp_graph={layer_stats['mlp_graph_speedup']:.3f}x "
            f"layer_tail_graph={layer_stats['layer_tail_graph_speedup']:.3f}x"
        )
        print(
            "layer diff: "
            f"max|eager-mlp_graph|={layer_stats['eager_vs_mlp_graph']:.6g} "
            f"max|eager-layer_tail_graph|={layer_stats['eager_vs_layer_tail_graph']:.6g}"
        )


if __name__ == "__main__":
    main()
