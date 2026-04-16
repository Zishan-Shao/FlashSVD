#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import statistics as stats
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.llama import get_flashsvd_v15_attn_densekv_decode_mod

try:
    from kernels.CUTLASS.loader import cutlass_build_diagnostics, load_cutlass_ops
    from kernels.CUTLASS.ops import reconstruct_qkv_token_shared_prepacked_cutlass
except Exception:  # pragma: no cover
    def cutlass_build_diagnostics():
        return {"available": False, "reason": "kernels.CUTLASS is not available in this checkout"}

    def load_cutlass_ops():
        return None

    def reconstruct_qkv_token_shared_prepacked_cutlass(*args, **kwargs):
        raise RuntimeError("CUTLASS decode attention kernel is not available in this checkout.")
from utils.model_utils import get_model_from_source


def _dtype_from_name(name: str) -> torch.dtype:
    raw = str(name).strip().lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_model(source: str, hf_token: str | None):
    return get_model_from_source(source, hf_token=hf_token)


def _bench_ms(fn: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]], *, warmup: int, iters: int) -> float:
    for _ in range(int(warmup)):
        out = fn()
        _ = out[0].reshape(-1)[0]
    torch.cuda.synchronize()
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(int(iters)):
        start.record()
        out = fn()
        _ = out[0].reshape(-1)[0]
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return float(stats.median(samples))


def _clear_attn_runtime_caches(attn) -> None:
    for name in ("_decode_Vq", "_decode_Vk", "_decode_Vv", "_decode_ptrs"):
        setattr(attn, name, None)
    cache = getattr(attn, "_dense_decode_prepacked_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _extract_attn_layers(model) -> list:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected a decoder-only model with model.layers.")
    out = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise ValueError("Layer is missing self_attn.")
        out.append(attn)
    return out


def _benchmark_attn_layer(
    attn,
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    use_cutlass: bool,
    use_hetero_fused: bool,
) -> dict[str, float | int | bool]:
    attn.eval()
    _clear_attn_runtime_caches(attn)
    attn._ensure_runtime_state()

    dense_decode_tensors = attn._get_dense_decode_tensors(device=x.device, dtype=x.dtype)
    packed_qkv_rank, vq_flat, vk_flat, vv_flat = dense_decode_tensors[:4]
    split_sizes = dense_decode_tensors[-1]
    actual_rq, actual_rk, actual_rv = attn._qkv_rank_tuple()
    rq, rk, rv = [int(v) for v in split_sizes]
    can_use_linear = (int(rq), int(rk), int(rv)) == (int(actual_rq), int(actual_rk), int(actual_rv))
    hidden_flat = x[:, 0, :].contiguous()
    can_use_hetero_fused = bool(
        use_hetero_fused
        and len({int(actual_rq), int(actual_rk), int(actual_rv)}) != 1
        and (int(rq), int(rk), int(rv)) == (int(actual_rq), int(actual_rk), int(actual_rv))
    )

    def exact_fn():
        p_q = attn.q_v_proj(x)
        p_k = attn.k_v_proj(x)
        p_v = attn.v_v_proj(x)
        q = attn.q_u_proj(p_q).view(x.shape[0], x.shape[1], attn.num_heads, attn.head_dim)[:, 0]
        k = attn.k_u_proj(p_k).view(x.shape[0], x.shape[1], attn.num_key_value_heads, attn.head_dim)[:, 0]
        v = attn.v_u_proj(p_v).view(x.shape[0], x.shape[1], attn.num_key_value_heads, attn.head_dim)[:, 0]
        return q, k, v

    def packed_torch_fn():
        packed_rank = torch.matmul(hidden_flat, packed_qkv_rank)
        p_q, p_k, p_v = torch.split(packed_rank, (rq, rk, rv), dim=1)
        return get_flashsvd_v15_attn_densekv_decode_mod().reconstruct_qkv_token_shared_prepacked(
            p_q,
            p_k,
            p_v,
            vq_flat,
            vk_flat,
            vv_flat,
            H=attn.num_heads,
            Hk=attn.num_key_value_heads,
            Dh=attn.head_dim,
        )

    def packed_linear_fn():
        if not can_use_linear:
            raise RuntimeError("packed_linear requires unpadded split sizes.")
        packed_rank = torch.matmul(hidden_flat, packed_qkv_rank)
        p_q, p_k, p_v = torch.split(packed_rank, (rq, rk, rv), dim=1)
        q = F.linear(p_q, attn.q_u_proj.weight, attn.q_u_proj.bias).view(hidden_flat.shape[0], attn.num_heads, attn.head_dim)
        k = F.linear(p_k, attn.k_u_proj.weight, attn.k_u_proj.bias).view(
            hidden_flat.shape[0], attn.num_key_value_heads, attn.head_dim
        )
        v = F.linear(p_v, attn.v_u_proj.weight, attn.v_u_proj.bias).view(
            hidden_flat.shape[0], attn.num_key_value_heads, attn.head_dim
        )
        return q, k, v

    def packed_cutlass_fn():
        packed_rank = torch.matmul(hidden_flat, packed_qkv_rank)
        p_q, p_k, p_v = torch.split(packed_rank, (rq, rk, rv), dim=1)
        return reconstruct_qkv_token_shared_prepacked_cutlass(
            p_q,
            p_k,
            p_v,
            vq_flat,
            vk_flat,
            vv_flat,
            H=attn.num_heads,
            Hk=attn.num_key_value_heads,
            Dh=attn.head_dim,
        )

    def packed_hetero_fused_fn():
        packed_rank = torch.matmul(hidden_flat, packed_qkv_rank)
        p_q, p_k, p_v = torch.split(packed_rank, (actual_rq, actual_rk, actual_rv), dim=1)
        vq, vk, vv = attn._get_decode_factors()
        return get_flashsvd_v15_attn_densekv_decode_mod().reconstruct_qkv_token_shared_hetero(
            p_q,
            p_k,
            p_v,
            vq.to(device=x.device, dtype=x.dtype).contiguous(),
            vk.to(device=x.device, dtype=x.dtype).contiguous(),
            vv.to(device=x.device, dtype=x.dtype).contiguous(),
        )

    with torch.inference_mode():
        q_ref, k_ref, v_ref = exact_fn()
        q_torch, k_torch, v_torch = packed_torch_fn()
        torch_diff = max(
            float((q_torch - q_ref).abs().max().item()),
            float((k_torch - k_ref).abs().max().item()),
            float((v_torch - v_ref).abs().max().item()),
        )
        linear_diff = 0.0
        if can_use_linear:
            q_linear, k_linear, v_linear = packed_linear_fn()
            linear_diff = max(
                float((q_linear - q_ref).abs().max().item()),
                float((k_linear - k_ref).abs().max().item()),
                float((v_linear - v_ref).abs().max().item()),
            )

        cutlass_diff = 0.0
        if use_cutlass:
            q_cut, k_cut, v_cut = packed_cutlass_fn()
            cutlass_diff = max(
                float((q_cut - q_ref).abs().max().item()),
                float((k_cut - k_ref).abs().max().item()),
                float((v_cut - v_ref).abs().max().item()),
            )

        hetero_fused_diff = 0.0
        if can_use_hetero_fused:
            q_fused, k_fused, v_fused = packed_hetero_fused_fn()
            hetero_fused_diff = max(
                float((q_fused - q_ref).abs().max().item()),
                float((k_fused - k_ref).abs().max().item()),
                float((v_fused - v_ref).abs().max().item()),
            )

    exact_ms = _bench_ms(exact_fn, warmup=warmup, iters=iters)
    packed_torch_ms = _bench_ms(packed_torch_fn, warmup=warmup, iters=iters)
    packed_linear_ms = float("nan")
    if can_use_linear:
        packed_linear_ms = _bench_ms(packed_linear_fn, warmup=warmup, iters=iters)
    packed_cutlass_ms = float("nan")
    if use_cutlass:
        packed_cutlass_ms = _bench_ms(packed_cutlass_fn, warmup=warmup, iters=iters)
    packed_hetero_fused_ms = float("nan")
    if can_use_hetero_fused:
        packed_hetero_fused_ms = _bench_ms(packed_hetero_fused_fn, warmup=warmup, iters=iters)

    return {
        "rq": int(rq),
        "rk": int(rk),
        "rv": int(rv),
        "hidden": int(attn.hidden_size),
        "heads": int(attn.num_heads),
        "kv_heads": int(attn.num_key_value_heads),
        "head_dim": int(attn.head_dim),
        "exact_ms": float(exact_ms),
        "packed_torch_ms": float(packed_torch_ms),
        "packed_torch_speedup": float(exact_ms / max(packed_torch_ms, 1e-9)),
        "packed_torch_max_diff": float(torch_diff),
        "packed_linear_ms": float(packed_linear_ms),
        "packed_linear_speedup": float(exact_ms / max(packed_linear_ms, 1e-9)) if can_use_linear else float("nan"),
        "packed_linear_max_diff": float(linear_diff),
        "packed_cutlass_ms": float(packed_cutlass_ms),
        "packed_cutlass_speedup": float(exact_ms / max(packed_cutlass_ms, 1e-9)) if use_cutlass else float("nan"),
        "packed_cutlass_max_diff": float(cutlass_diff),
        "packed_hetero_fused_ms": float(packed_hetero_fused_ms),
        "packed_hetero_fused_speedup": float(exact_ms / max(packed_hetero_fused_ms, 1e-9)) if can_use_hetero_fused else float("nan"),
        "packed_hetero_fused_max_diff": float(hetero_fused_diff) if can_use_hetero_fused else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser("Benchmark real-checkpoint decode QKV reconstruction on current-token path")
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Checkpoint source. Supports a local path, a Hugging Face repo id, or a Hugging Face repo "
            "subfolder like 'namespace/repo/path/to/export'."
        ),
    )
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--layer", type=int, default=-1, help="Single layer index. -1 benchmarks all layers.")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--cutlass_root", type=str, default="")
    ap.add_argument("--rank_pad_multiple", type=int, default=1, help="Pad each of (Rq,Rk,Rv) independently to this multiple. Pass 1 to disable padding.")
    ap.add_argument("--allow_approx", action=argparse.BooleanOptionalAction, default=False, help="Allow padded split_sizes that can change low-precision outputs slightly.")
    ap.add_argument("--pad_mode", type=str, default="independent", choices=["independent", "uniform_layer"], help="independent pads each of Q/K/V separately; uniform_layer pads all three to the same per-layer rank.")
    ap.add_argument("--hetero_fused", action=argparse.BooleanOptionalAction, default=False, help="Benchmark fused Triton hetero-rank reconstruct path when applicable.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    if args.cutlass_root:
        os.environ["FLASH_SVD_CUTLASS_ROOT"] = str(args.cutlass_root)
    os.environ["FLASH_SVD_DENSE_DECODE_RANK_PAD_MULTIPLE"] = str(max(1, int(args.rank_pad_multiple)))
    os.environ["FLASH_SVD_DENSE_DECODE_ALLOW_APPROX"] = "1" if args.allow_approx else "0"
    os.environ["FLASH_SVD_DENSE_DECODE_PAD_MODE"] = str(args.pad_mode)
    ext = load_cutlass_ops()
    use_cutlass = ext is not None
    print("CUTLASS:", {"enabled": use_cutlass, **cutlass_build_diagnostics()})

    dtype = _dtype_from_name(args.dtype)
    model, _tok = _load_model(args.checkpoint, args.hf_token)
    model.eval().to(args.device)
    model = model.to(dtype=dtype)
    attn_layers = _extract_attn_layers(model)

    if args.layer >= 0:
        layer_indices = [int(args.layer)]
    else:
        layer_indices = list(range(len(attn_layers)))

    print("==== Real Checkpoint Decode Reconstruct Benchmark ====")
    print(
        f"checkpoint={args.checkpoint} dtype={args.dtype} device={args.device} "
        f"batch={args.batch_size} warmup={args.warmup} iters={args.iters} "
        f"layers={len(layer_indices)} rank_pad_multiple={max(1, int(args.rank_pad_multiple))} "
        f"allow_approx={bool(args.allow_approx)} pad_mode={args.pad_mode}"
    )

    per_layer: list[tuple[int, dict[str, float | int | bool]]] = []
    for layer_idx in layer_indices:
        attn = attn_layers[layer_idx]
        x = torch.randn(int(args.batch_size), 1, int(attn.hidden_size), device=args.device, dtype=dtype)
        result = _benchmark_attn_layer(
            attn,
            x,
            warmup=int(args.warmup),
            iters=int(args.iters),
            use_cutlass=use_cutlass,
            use_hetero_fused=bool(args.hetero_fused),
        )
        per_layer.append((layer_idx, result))
        actual_ranks = (
            int(attn.q_v_proj.out_features),
            int(attn.k_v_proj.out_features),
            int(attn.v_v_proj.out_features),
        )
        split_ranks = (int(result["rq"]), int(result["rk"]), int(result["rv"]))
        rank_label = (
            f"ranks={actual_ranks[0]}/{actual_ranks[1]}/{actual_ranks[2]}"
            if split_ranks == actual_ranks
            else (
                f"actual={actual_ranks[0]}/{actual_ranks[1]}/{actual_ranks[2]} "
                f"split={split_ranks[0]}/{split_ranks[1]}/{split_ranks[2]}"
            )
        )
        line = f"L{layer_idx:02d} {rank_label} exact={result['exact_ms']:.4f} ms "
        if result["packed_linear_ms"] == result["packed_linear_ms"]:
            line += (
                f"packed_linear={result['packed_linear_ms']:.4f} ms "
                f"({result['packed_linear_speedup']:.3f}x, diff={result['packed_linear_max_diff']:.3g}) | "
            )
        line += (
            f"packed_flat={result['packed_torch_ms']:.4f} ms "
            f"({result['packed_torch_speedup']:.3f}x, diff={result['packed_torch_max_diff']:.3g})"
        )
        if use_cutlass:
            line += (
                f" | cutlass_packed={result['packed_cutlass_ms']:.4f} ms "
                f"({result['packed_cutlass_speedup']:.3f}x, diff={result['packed_cutlass_max_diff']:.3g})"
            )
        if args.hetero_fused and result["packed_hetero_fused_ms"] == result["packed_hetero_fused_ms"]:
            line += (
                f" | hetero_fused={result['packed_hetero_fused_ms']:.4f} ms "
                f"({result['packed_hetero_fused_speedup']:.3f}x, diff={result['packed_hetero_fused_max_diff']:.3g})"
            )
        print(line)

    if len(per_layer) > 1:
        linear_speedups = [
            float(res["packed_linear_speedup"])
            for _, res in per_layer
            if float(res["packed_linear_speedup"]) == float(res["packed_linear_speedup"])
        ]
        torch_speedups = [float(res["packed_torch_speedup"]) for _, res in per_layer]
        exact_total = sum(float(res["exact_ms"]) for _, res in per_layer)
        linear_total = sum(
            float(res["packed_linear_ms"])
            for _, res in per_layer
            if float(res["packed_linear_ms"]) == float(res["packed_linear_ms"])
        )
        flat_total = sum(float(res["packed_torch_ms"]) for _, res in per_layer)
        auto_linear_total = sum(
            min(float(res["exact_ms"]), float(res["packed_linear_ms"]))
            if float(res["packed_linear_ms"]) == float(res["packed_linear_ms"])
            else float(res["exact_ms"])
            for _, res in per_layer
        )
        auto_flat_total = sum(min(float(res["exact_ms"]), float(res["packed_torch_ms"])) for _, res in per_layer)
        linear_wins = sum(
            1
            for _, res in per_layer
            if float(res["packed_linear_ms"]) == float(res["packed_linear_ms"])
            and float(res["packed_linear_ms"]) < float(res["exact_ms"])
        )
        flat_wins = sum(1 for _, res in per_layer if float(res["packed_torch_ms"]) < float(res["exact_ms"]))
        line = ""
        if linear_speedups:
            line += (
                f"packed_linear median={stats.median(linear_speedups):.3f}x "
                f"mean={stats.mean(linear_speedups):.3f}x "
                f"min={min(linear_speedups):.3f}x max={max(linear_speedups):.3f}x "
                f"total={exact_total / max(linear_total, 1e-9):.3f}x "
                f"auto_total={exact_total / max(auto_linear_total, 1e-9):.3f}x "
                f"wins={linear_wins}/{len(per_layer)} | "
            )
        line += (
            f"packed_flat median={stats.median(torch_speedups):.3f}x "
            f"mean={stats.mean(torch_speedups):.3f}x "
            f"min={min(torch_speedups):.3f}x max={max(torch_speedups):.3f}x "
            f"total={exact_total / max(flat_total, 1e-9):.3f}x "
            f"auto_total={exact_total / max(auto_flat_total, 1e-9):.3f}x "
            f"wins={flat_wins}/{len(per_layer)}"
        )
        if use_cutlass:
            cutlass_speedups = [float(res["packed_cutlass_speedup"]) for _, res in per_layer]
            line += (
                f" | cutlass_packed median={stats.median(cutlass_speedups):.3f}x "
                f"mean={stats.mean(cutlass_speedups):.3f}x "
                f"min={min(cutlass_speedups):.3f}x max={max(cutlass_speedups):.3f}x"
            )
        if args.hetero_fused:
            fused_speedups = [
                float(res["packed_hetero_fused_speedup"])
                for _, res in per_layer
                if float(res["packed_hetero_fused_speedup"]) == float(res["packed_hetero_fused_speedup"])
            ]
            if fused_speedups:
                line += (
                    f" | hetero_fused median={stats.median(fused_speedups):.3f}x "
                    f"mean={stats.mean(fused_speedups):.3f}x "
                    f"min={min(fused_speedups):.3f}x max={max(fused_speedups):.3f}x"
                )
        print("Aggregate:", line)

    del attn_layers
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
