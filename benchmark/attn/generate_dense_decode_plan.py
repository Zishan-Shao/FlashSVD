#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from collections import Counter
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.attn.bench_real_checkpoint_decode_reconstruct import (
    _clear_attn_runtime_caches,
    _benchmark_attn_layer,
    _dtype_from_name,
    _extract_attn_layers,
    _load_model,
)
from runtime.cache.attn_dense_kv import FlashSVDV15DenseKVCache


def _parse_pad_multiples(raw: str) -> list[int]:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = max(1, int(part))
        if value not in out:
            out.append(value)
    if not out:
        return [1, 8, 16, 32, 64]
    return out


def _parse_int_list(raw: str, *, default: list[int]) -> list[int]:
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in out:
            out.append(value)
    return out or list(default)


def _is_finite_number(value) -> bool:
    try:
        value = float(value)
    except Exception:
        return False
    return value == value and value not in {float("inf"), float("-inf")}


def _build_candidates(
    *,
    layer_idx: int,
    actual_ranks: tuple[int, int, int],
    result: dict[str, float | int | bool],
    pad_multiple: int,
    include_exact: bool,
):
    candidates = []
    exact_ms = float(result["exact_ms"])
    if include_exact:
        candidates.append(
            {
                "layer": int(layer_idx),
                "backend": "exact",
                "impl": "exact",
                "split_sizes": list(actual_ranks),
                "pad_multiple": 1,
                "ms": exact_ms,
                "speedup_vs_exact": 1.0,
            }
        )

    split_sizes = (int(result["rq"]), int(result["rk"]), int(result["rv"]))
    packed_linear_ms = float(result["packed_linear_ms"])
    packed_flat_ms = float(result["packed_torch_ms"])

    # Runtime uses packed-linear whenever split_sizes match the actual ranks.
    if split_sizes == actual_ranks and _is_finite_number(packed_linear_ms):
        candidates.append(
            {
                "layer": int(layer_idx),
                "backend": "packed",
                "impl": "packed_linear",
                "split_sizes": list(split_sizes),
                "pad_multiple": int(pad_multiple),
                "ms": packed_linear_ms,
                "speedup_vs_exact": exact_ms / max(packed_linear_ms, 1e-9),
            }
        )

    # Packed-flat is only a distinct runtime choice once padding changes the split sizes.
    if (split_sizes != actual_ranks or not _is_finite_number(packed_linear_ms)) and _is_finite_number(packed_flat_ms):
        candidates.append(
            {
                "layer": int(layer_idx),
                "backend": "packed",
                "impl": "packed_flat",
                "split_sizes": list(split_sizes),
                "pad_multiple": int(pad_multiple),
                "ms": packed_flat_ms,
                "speedup_vs_exact": exact_ms / max(packed_flat_ms, 1e-9),
            }
        )
    return candidates


def _build_attention_candidates(
    *,
    layer_idx: int,
    actual_ranks: tuple[int, int, int],
    split_sizes: tuple[int, int, int],
    exact_ms: float,
    packed_ms: float,
    pad_multiple: int,
):
    candidates = [
        {
            "layer": int(layer_idx),
            "backend": "exact",
            "impl": "exact_attention",
            "split_sizes": list(actual_ranks),
            "pad_multiple": 1,
            "ms": float(exact_ms),
            "speedup_vs_exact": 1.0,
        }
    ]
    candidates.append(
        {
            "layer": int(layer_idx),
            "backend": "packed",
            "impl": "packed_attention",
            "split_sizes": list(split_sizes),
            "pad_multiple": int(pad_multiple),
            "ms": float(packed_ms),
            "speedup_vs_exact": float(exact_ms) / max(float(packed_ms), 1e-9),
        }
    )
    return candidates


def _make_dense_cache(
    model,
    attn,
    *,
    batch_size: int,
    past_len: int,
    device: str,
    dtype: torch.dtype,
) -> FlashSVDV15DenseKVCache:
    cache = FlashSVDV15DenseKVCache(
        model.config,
        max_batch_size=int(batch_size),
        max_cache_len=max(int(past_len) + 16, 16),
        device=device,
        dtype=dtype,
    )
    cache._ensure_layer_spec(int(attn.layer_idx), h=int(attn.num_key_value_heads), dh=int(attn.head_dim))
    if int(past_len) > 0:
        cache.key_cache[int(attn.layer_idx)][: int(batch_size), : int(past_len)].normal_()
        cache.value_cache[int(attn.layer_idx)][: int(batch_size), : int(past_len)].normal_()
    cache._seen_tokens = int(past_len)
    return cache


def _bench_full_attention_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(int(warmup)):
        out = fn()
        _ = out.reshape(-1)[0]
    torch.cuda.synchronize()
    samples: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(int(iters)):
        start.record()
        out = fn()
        _ = out.reshape(-1)[0]
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return float(samples[len(samples) // 2]) if samples else float("inf")


def _benchmark_attn_layer_attention(
    model,
    attn,
    x: torch.Tensor,
    *,
    past_len: int,
    warmup: int,
    iters: int,
) -> dict[str, float | int]:
    attn.eval()
    attn._ensure_runtime_state()
    _clear_attn_runtime_caches(attn)
    pos = torch.tensor([int(past_len)], dtype=torch.long)
    base_cache = _make_dense_cache(
        model,
        attn,
        batch_size=int(x.shape[0]),
        past_len=int(past_len),
        device=str(x.device),
        dtype=x.dtype,
    )
    dense_decode_tensors = attn._get_dense_decode_tensors(
        device=x.device,
        dtype=x.dtype,
        past_key_value=base_cache,
        cache_position=pos,
    )
    packed_qkv_rank, vq_flat, vk_flat, vv_flat = dense_decode_tensors[:4]
    split_sizes = dense_decode_tensors[-1]
    actual_ranks = tuple(int(v) for v in attn._qkv_rank_tuple())

    def _make_backend_runner(backend: str):
        os.environ["FLASH_SVD_DENSE_DECODE_BACKEND"] = str(backend)
        attn._dense_decode_backend_cache.clear()
        cache = base_cache.clone_for_autotune(layer_idx=int(attn.layer_idx), batch_size=int(x.shape[0]))
        if cache is None:
            raise RuntimeError("Failed to clone dense cache for attention benchmark.")

        def _call() -> torch.Tensor:
            with torch.inference_mode():
                return attn._flashsvd_dense_decode_token_from_hidden(
                    x,
                    cache,
                    cache_position=pos,
                    cache_bindings=None,
                    advance_cache=False,
                )
        return _call

    exact_ms = _bench_full_attention_ms(_make_backend_runner("exact"), warmup=warmup, iters=iters)
    packed_ms = _bench_full_attention_ms(_make_backend_runner("packed"), warmup=warmup, iters=iters)
    return {
        "exact_ms": float(exact_ms),
        "packed_ms": float(packed_ms),
        "rq": int(split_sizes[0]),
        "rk": int(split_sizes[1]),
        "rv": int(split_sizes[2]),
        "actual_rq": int(actual_ranks[0]),
        "actual_rk": int(actual_ranks[1]),
        "actual_rv": int(actual_ranks[2]),
    }


def main() -> int:
    ap = argparse.ArgumentParser("Generate a per-layer FlashSVD dense-decode plan from checkpoint measurements")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--hf_token", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--layer", type=int, default=-1, help="Single layer index. -1 generates the full plan.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mode", type=str, default="attention", choices=["reconstruct", "attention"])
    ap.add_argument("--past_buckets", type=str, default="0,512", help="Comma-separated past lengths for attention-mode plan generation.")
    ap.add_argument("--pad_multiples", type=str, default="1,8,16,32,64")
    ap.add_argument("--pad_mode", type=str, default=os.getenv("FLASH_SVD_DENSE_DECODE_PAD_MODE", "independent"))
    ap.add_argument("--rank_map_path", type=str, default="", help="Optional rank-map metadata to embed into the plan.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for plan generation.")

    dtype = _dtype_from_name(args.dtype)
    model, _tok = _load_model(args.checkpoint, args.hf_token)
    model.eval()
    attn_layers = _extract_attn_layers(model)
    layer_indices = [int(args.layer)] if int(args.layer) >= 0 else list(range(len(attn_layers)))
    pad_multiples = _parse_pad_multiples(args.pad_multiples)
    past_buckets = _parse_int_list(args.past_buckets, default=[0, 512])
    os.environ["FLASH_SVD_DENSE_DECODE_PAD_MODE"] = str(args.pad_mode)

    if args.rank_map_path:
        rank_map_path = str(Path(args.rank_map_path).resolve())
        try:
            rank_map_payload = json.loads(Path(rank_map_path).read_text())
        except Exception:
            rank_map_payload = None
    else:
        rank_map_path = ""
        rank_map_payload = None

    layer_plan = {}
    best_backend_counter = Counter()
    best_pad_counter = Counter()
    best_bucket_counter = Counter()
    best_past_bucket_counter = Counter()

    print("==== FlashSVD Dense Decode Plan Generation ====")
    print(
        f"checkpoint={args.checkpoint} dtype={args.dtype} device={args.device} batch={args.batch_size} "
        f"warmup={args.warmup} iters={args.iters} layers={len(layer_indices)} mode={args.mode} "
        f"pad_multiples={pad_multiples} pad_mode={args.pad_mode} past_buckets={past_buckets}"
    )

    for layer_idx in layer_indices:
        attn = attn_layers[int(layer_idx)].to(args.device, dtype=dtype)
        attn._ensure_runtime_state()
        actual_ranks = tuple(int(v) for v in attn._qkv_rank_tuple())
        bucket_plan = {}
        layer_best = []
        for past_bucket in (past_buckets if args.mode == "attention" else [-1]):
            candidates = []
            for pad_multiple in pad_multiples:
                os.environ["FLASH_SVD_DENSE_DECODE_RANK_PAD_MULTIPLE"] = str(int(pad_multiple))
                os.environ["FLASH_SVD_DENSE_DECODE_ALLOW_APPROX"] = "1" if int(pad_multiple) > 1 else "0"
                x = torch.randn(int(args.batch_size), 1, int(attn.hidden_size), device=args.device, dtype=dtype)
                if args.mode == "attention":
                    result = _benchmark_attn_layer_attention(
                        model,
                        attn,
                        x,
                        past_len=int(past_bucket),
                        warmup=int(args.warmup),
                        iters=int(args.iters),
                    )
                    candidates.extend(
                        _build_attention_candidates(
                            layer_idx=int(layer_idx),
                            actual_ranks=actual_ranks,
                            split_sizes=(int(result["rq"]), int(result["rk"]), int(result["rv"])),
                            exact_ms=float(result["exact_ms"]),
                            packed_ms=float(result["packed_ms"]),
                            pad_multiple=int(pad_multiple),
                        )
                    )
                else:
                    result = _benchmark_attn_layer(
                        attn,
                        x,
                        warmup=int(args.warmup),
                        iters=int(args.iters),
                        use_cutlass=False,
                        use_hetero_fused=False,
                    )
                    candidates.extend(
                        _build_candidates(
                            layer_idx=int(layer_idx),
                            actual_ranks=actual_ranks,
                            result=result,
                            pad_multiple=int(pad_multiple),
                            include_exact=(int(pad_multiple) == int(pad_multiples[0])),
                        )
                    )
                del x
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            best = min(candidates, key=lambda item: (float(item["ms"]), 0 if str(item["backend"]) == "packed" else 1))
            bucket = "_".join(str(int(v)) for v in best["split_sizes"])
            best_backend_counter[str(best["backend"])] += 1
            best_pad_counter[int(best["pad_multiple"])] += 1
            best_bucket_counter[bucket] += 1
            if args.mode == "attention":
                best_past_bucket_counter[int(past_bucket)] += 1

            bucket_plan[str(int(past_bucket))] = {
                "backend": str(best["backend"]),
                "impl": str(best["impl"]),
                "pad_multiple": int(best["pad_multiple"]),
                "split_sizes": [int(v) for v in best["split_sizes"]],
                "bucket": bucket,
                "best_ms": float(best["ms"]),
                "best_speedup_vs_exact": float(best["speedup_vs_exact"]),
                "candidates": [
                    {
                        "backend": str(candidate["backend"]),
                        "impl": str(candidate["impl"]),
                        "pad_multiple": int(candidate["pad_multiple"]),
                        "split_sizes": [int(v) for v in candidate["split_sizes"]],
                        "ms": float(candidate["ms"]),
                        "speedup_vs_exact": float(candidate["speedup_vs_exact"]),
                    }
                    for candidate in sorted(candidates, key=lambda item: float(item["ms"]))
                ],
            }
            layer_best.append((int(past_bucket), best))
            tag = f"L{int(layer_idx):02d}" if args.mode != "attention" else f"L{int(layer_idx):02d} P{int(past_bucket):04d}"
            print(
                f"{tag} actual={actual_ranks[0]}/{actual_ranks[1]}/{actual_ranks[2]} "
                f"best={best['backend']}:{best['impl']} split={tuple(best['split_sizes'])} "
                f"pad={int(best['pad_multiple'])} ms={float(best['ms']):.4f} "
                f"speedup={float(best['speedup_vs_exact']):.3f}x"
            )

        layer_default_bucket, layer_default = min(layer_best, key=lambda item: float(item[1]["ms"]))
        layer_entry = {
            "layer_idx": int(layer_idx),
            "backend": str(layer_default["backend"]),
            "impl": str(layer_default["impl"]),
            "pad_multiple": int(layer_default["pad_multiple"]),
            "qkv_ranks": [int(v) for v in actual_ranks],
            "split_sizes": [int(v) for v in layer_default["split_sizes"]],
            "bucket": "_".join(str(int(v)) for v in layer_default["split_sizes"]),
            "best_ms": float(layer_default["ms"]),
            "best_speedup_vs_exact": float(layer_default["speedup_vs_exact"]),
        }
        if args.mode == "attention":
            layer_entry["buckets"] = bucket_plan
            layer_entry["default_past_bucket"] = int(layer_default_bucket)
        else:
            layer_entry["candidates"] = bucket_plan[str(-1)]["candidates"]
        layer_plan[str(int(layer_idx))] = layer_entry

        attn.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    default_backend = best_backend_counter.most_common(1)[0][0] if best_backend_counter else "packed"
    default_pad = int(best_pad_counter.most_common(1)[0][0]) if best_pad_counter else 16
    plan = {
        "format": "flashsvd_dense_decode_plan_v2" if args.mode == "attention" else "flashsvd_dense_decode_plan_v1",
        "source": {
            "checkpoint": str(args.checkpoint),
            "dtype": str(args.dtype),
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "warmup": int(args.warmup),
            "iters": int(args.iters),
            "mode": str(args.mode),
            "pad_mode": str(args.pad_mode),
            "pad_multiples": [int(v) for v in pad_multiples],
            "past_buckets": [int(v) for v in past_buckets],
            "rank_map_path": rank_map_path or None,
            "rank_map_loaded": rank_map_payload is not None,
        },
        "defaults": {
            "backend": default_backend,
            "pad_multiple": default_pad,
        },
        "summary": {
            "backend_counts": dict(best_backend_counter),
            "pad_counts": {str(k): int(v) for k, v in best_pad_counter.items()},
            "bucket_counts": dict(best_bucket_counter),
            "past_bucket_counts": {str(k): int(v) for k, v in best_past_bucket_counter.items()},
        },
        "layers": layer_plan,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"Saved plan to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
