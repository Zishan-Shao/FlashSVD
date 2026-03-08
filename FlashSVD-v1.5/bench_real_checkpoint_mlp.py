#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import statistics as stats
from pathlib import Path

import torch

from utils.model_utils import get_model_from_huggingface, get_model_from_local


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
    path = Path(source)
    if path.exists():
        return get_model_from_local(str(path))
    return get_model_from_huggingface(source, hf_token=hf_token)


def _bench_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(int(warmup)):
        out = fn()
        _ = out.reshape(-1)[0]
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(int(iters)):
        start.record()
        out = fn()
        _ = out.reshape(-1)[0]
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return float(stats.median(samples))


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _configure_backend_env(backend: str, *, use_graph: bool) -> None:
    if backend == "baseline":
        _set_env("SVDLLM_FLASH_FALLBACK", "1")
        _set_env("FLASH_SVD_DISABLE_FFN", "1")
        _set_env("FLASH_SVD_FFN_BACKEND", "dual_split_cublas_legacy")
        _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", "0")
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH", "0")
        _set_env("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE", "mlp")
        return

    _set_env("SVDLLM_FLASH_FALLBACK", "0")
    _set_env("FLASH_SVD_DISABLE_FFN", "0")
    _set_env("FLASH_SVD_FFN_BACKEND", backend)
    _set_env("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", "1")
    _set_env("FLASH_SVD_MLP_CUDA_GRAPH", "1" if use_graph else "0")
    _set_env("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE", "mlp")


def _clear_mlp_runtime_caches(mlp) -> None:
    for name in (
        "_flashsvd_graph_cache",
        "_flashsvd_layer_graph_cache",
    ):
        cache = getattr(mlp, name, None)
        if isinstance(cache, dict):
            cache.clear()


def _bench_backend_modes(
    mlp,
    x: torch.Tensor,
    *,
    backend_names: list[str],
    compare_graph: bool,
    warmup: int,
    iters: int,
) -> dict[str, dict[str, tuple[float, torch.Tensor]]]:
    results: dict[str, dict[str, tuple[float, torch.Tensor]]] = {}
    for backend in backend_names:
        per_backend: dict[str, tuple[float, torch.Tensor]] = {}
        mode_names = ["eager"]
        if backend != "baseline" and compare_graph:
            mode_names.append("graph")
        for mode in mode_names:
            _configure_backend_env(backend, use_graph=(mode == "graph"))
            _clear_mlp_runtime_caches(mlp)

            def _run():
                with torch.inference_mode():
                    return mlp(x)

            out = _run()
            ms = _bench_ms(_run, warmup=warmup, iters=iters)
            per_backend[mode] = (ms, out.detach().clone())
        results[backend] = per_backend
    return results


def _format_backend_line(
    name: str,
    *,
    ref_ms: float,
    ref_out: torch.Tensor,
    results: dict[str, dict[str, tuple[float, torch.Tensor]]],
) -> str:
    eager_ms, eager_out = results[name]["eager"]
    eager_diff = float((eager_out - ref_out).abs().max().item())
    eager_speedup = ref_ms / max(eager_ms, 1e-9)
    line = (
        f"- {name}: eager={eager_ms:.4f} ms "
        f"(speedup_vs_ref={eager_speedup:.3f}x, max_diff={eager_diff:.6g})"
    )
    graph_pair = results[name].get("graph")
    if graph_pair is not None:
        graph_ms, graph_out = graph_pair
        graph_diff = float((graph_out - ref_out).abs().max().item())
        graph_speedup = ref_ms / max(graph_ms, 1e-9)
        graph_gain = eager_ms / max(graph_ms, 1e-9)
        line += (
            f" | graph={graph_ms:.4f} ms "
            f"(speedup_vs_ref={graph_speedup:.3f}x, max_diff={graph_diff:.6g}, "
            f"graph_gain={graph_gain:.3f}x)"
        )
    return line


def _print_single_layer_report(
    *,
    args,
    backend_names: list[str],
    results: dict[str, dict[str, tuple[float, torch.Tensor]]],
) -> None:
    ref_name = backend_names[0]
    ref_ms, ref_out = results[ref_name]["eager"]
    print("==== Real Checkpoint MLP Benchmark ====")
    print(
        f"Config: layer={args.layer} B={args.batch_size} L={args.seq_len} "
        f"dtype={args.dtype} device={args.device} compare_graph={int(args.cuda_graph)}"
    )
    print(f"Reference: {ref_name}/eager ({ref_ms:.4f} ms)")
    for name in backend_names:
        print(_format_backend_line(name, ref_ms=ref_ms, ref_out=ref_out, results=results))


def _print_all_layer_report(
    *,
    args,
    backend_names: list[str],
    per_layer_results: list[tuple[int, dict[str, dict[str, tuple[float, torch.Tensor]]]]],
) -> None:
    print("==== Real Checkpoint MLP All-Layer Sweep ====")
    print(
        f"Config: layers={len(per_layer_results)} B={args.batch_size} L={args.seq_len} "
        f"dtype={args.dtype} device={args.device} compare_graph={int(args.cuda_graph)}"
    )
    print("Per-layer summary:")
    summary: dict[str, dict[str, list[float]]] = {
        name: {"eager_ms": [], "eager_speedup": [], "graph_ms": [], "graph_speedup": [], "graph_gain": []}
        for name in backend_names
    }
    for layer_idx, results in per_layer_results:
        ref_name = backend_names[0]
        ref_ms, ref_out = results[ref_name]["eager"]
        parts = [f"L{layer_idx:02d} ref={ref_ms:.4f}"]
        for name in backend_names:
            eager_ms, eager_out = results[name]["eager"]
            eager_speedup = ref_ms / max(eager_ms, 1e-9)
            _ = eager_out
            summary[name]["eager_ms"].append(eager_ms)
            summary[name]["eager_speedup"].append(eager_speedup)
            part = f"{name}:e={eager_ms:.4f}({eager_speedup:.3f}x)"
            graph_pair = results[name].get("graph")
            if graph_pair is not None:
                graph_ms, _graph_out = graph_pair
                graph_speedup = ref_ms / max(graph_ms, 1e-9)
                graph_gain = eager_ms / max(graph_ms, 1e-9)
                summary[name]["graph_ms"].append(graph_ms)
                summary[name]["graph_speedup"].append(graph_speedup)
                summary[name]["graph_gain"].append(graph_gain)
                part += f"/g={graph_ms:.4f}({graph_speedup:.3f}x,+{(graph_gain - 1.0) * 100.0:.1f}%)"
            parts.append(part)
        print(" | ".join(parts))

    print("All-layer aggregate:")
    for name in backend_names:
        eager_ms = summary[name]["eager_ms"]
        eager_speedup = summary[name]["eager_speedup"]
        line = (
            f"- {name}: eager_median={stats.median(eager_ms):.4f} ms "
            f"eager_mean={stats.mean(eager_ms):.4f} ms "
            f"median_speedup={stats.median(eager_speedup):.3f}x "
            f"mean_speedup={stats.mean(eager_speedup):.3f}x"
        )
        if summary[name]["graph_ms"]:
            graph_ms = summary[name]["graph_ms"]
            graph_speedup = summary[name]["graph_speedup"]
            graph_gain = summary[name]["graph_gain"]
            line += (
                f" | graph_median={stats.median(graph_ms):.4f} ms "
                f"graph_mean={stats.mean(graph_ms):.4f} ms "
                f"median_graph_speedup={stats.median(graph_speedup):.3f}x "
                f"mean_graph_speedup={stats.mean(graph_speedup):.3f}x "
                f"median_graph_gain={stats.median(graph_gain):.3f}x "
                f"mean_graph_gain={stats.mean(graph_gain):.3f}x"
            )
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser("Benchmark exact SVD-Llama MLP backends on a real checkpoint")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--all_layers", action="store_true", help="Benchmark all decoder MLP layers.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--backends",
        type=str,
        default="baseline,auto,dual_split_cublas,dual_split_cublas_legacy,dual_split_kernel,dual_split_kernel_v2,dual_split_kernel_v2_sm80,dual_split_kernel_v3",
        help="Comma-separated list of MLP backends to test.",
    )
    ap.add_argument("--cuda_graph", action=argparse.BooleanOptionalAction, default=True)
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
            per_layer_results: list[tuple[int, dict[str, dict[str, tuple[float, torch.Tensor]]]]] = []
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
