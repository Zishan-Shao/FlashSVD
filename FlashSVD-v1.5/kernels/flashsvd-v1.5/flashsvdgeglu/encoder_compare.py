#!/usr/bin/env python3
"""Legacy synthetic encoder compare for ModernBERT-style workloads.

Compares dense vs FlashSVD on:
- attention-only
- GEGLU FFN-only
- combined encoder compute (attn + ffn)

The attention path is sliding-window aware and uses ModernBERT RoPE.
This script is useful for synthetic kernel exploration, but it is not the
recommended low-rank-model-vs-low-rank-model inference benchmark.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding


def _load_module(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


@dataclass
class ModelShape:
    hidden_size: int
    num_heads: int
    intermediate_size: int
    local_attention: int


@dataclass
class BenchResult:
    mean_ms: float
    median_ms: float
    p95_ms: float
    toks_per_s: float
    peak_alloc_mib: float
    peak_reserved_mib: float


def _round_down(x: float, multiple: int, minimum: int = 8) -> int:
    if multiple <= 1:
        return max(int(x), minimum)
    v = int(x) // multiple * multiple
    return max(v, minimum)


def pick_ranks(shape: ModelShape, target_ratio: float, round_multiple: int) -> Dict[str, int]:
    d = shape.hidden_size
    f = shape.intermediate_size

    # Attention: dense D*D, low-rank ~ 2*D*R
    raw_attn = target_ratio * d / 2.0
    r_attn = _round_down(raw_attn, round_multiple, minimum=8)

    # FFN Wi: Dx(2F), low-rank R1*(D+2F)
    raw_r1 = target_ratio * (d * (2 * f)) / (d + 2 * f)
    r1 = _round_down(raw_r1, round_multiple, minimum=8)

    # FFN Wo: FxD, low-rank R2*(F+D)
    raw_r2 = target_ratio * (f * d) / (f + d)
    r2 = _round_down(raw_r2, round_multiple, minimum=8)

    return {"r_attn": r_attn, "r1": r1, "r2": r2}


def build_sliding_additive_mask(
    B: int,
    L: int,
    window_radius: int,
    dtype: torch.dtype,
    device: torch.device,
    pad_mask_2d: Optional[torch.Tensor],
) -> torch.Tensor:
    idx = torch.arange(L, device=device)
    dist = torch.abs(idx[:, None] - idx[None, :])
    local_allow = dist <= int(window_radius)

    neg_inf = torch.finfo(dtype).min
    add = torch.zeros((B, 1, L, L), device=device, dtype=dtype)
    add.masked_fill_(~local_allow[None, None, :, :], neg_inf)

    if pad_mask_2d is not None:
        valid = pad_mask_2d.to(torch.bool)
        allow = valid[:, None, :, None] & valid[:, None, None, :]
        add.masked_fill_(~allow, neg_inf)

    return add


def make_padding_mask(B: int, L: int, pad_fraction: float, device: torch.device) -> torch.Tensor:
    if pad_fraction <= 0:
        return torch.ones((B, L), device=device, dtype=torch.int32)

    keep = max(1, int(round(L * (1.0 - pad_fraction))))
    mask = torch.zeros((B, L), device=device, dtype=torch.int32)
    for b in range(B):
        # deterministic but not identical per row
        k = max(1, keep - (b % max(1, L - keep + 1)))
        mask[b, :k] = 1
    return mask


@torch.no_grad()
def bench_cuda(name: str, fn: Callable[[], torch.Tensor], B: int, L: int, warmup: int, iters: int) -> BenchResult:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        _ = out.view(-1)[0].item()

    times_t = torch.tensor(times, device="cpu")
    mean_ms = float(times_t.mean().item())
    median_ms = float(times_t.median().item())
    p95_ms = float(torch.quantile(times_t, 0.95).item())

    toks = B * L
    toks_per_s = toks / (mean_ms / 1e3)

    return BenchResult(
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        toks_per_s=toks_per_s,
        peak_alloc_mib=torch.cuda.max_memory_allocated() / (1024**2),
        peak_reserved_mib=torch.cuda.max_memory_reserved() / (1024**2),
    )


def _fmt(res: BenchResult) -> str:
    return (
        f"{res.mean_ms:8.4f} ms | {res.toks_per_s:8.0f} tok/s | "
        f"peak_alloc={res.peak_alloc_mib:8.2f} MiB peak_res={res.peak_reserved_mib:8.2f} MiB"
    )


def load_shape(args) -> ModelShape:
    if args.model_config:
        try:
            cfg = AutoConfig.from_pretrained(
                args.model_config,
                trust_remote_code=True,
                local_files_only=args.local_files_only,
            )
            hidden_size = int(getattr(cfg, "hidden_size"))
            num_heads = int(getattr(cfg, "num_attention_heads"))
            inter = int(getattr(cfg, "intermediate_size"))
            local_attn = int(getattr(cfg, "local_attention", 128))
            return ModelShape(hidden_size, num_heads, inter, local_attn)
        except Exception as e:
            print(f"[warn] failed to load config from {args.model_config}: {type(e).__name__}: {e}")

    base = ModernBertConfig()
    hidden_size = args.hidden_size or int(base.hidden_size)
    num_heads = args.num_heads or int(base.num_attention_heads)
    inter = args.intermediate_size or int(base.intermediate_size)
    local_attn = args.local_attention or int(base.local_attention)
    return ModelShape(hidden_size, num_heads, inter, local_attn)


def make_modernbert_rotary(cfg: ModernBertConfig, dh: int, device: torch.device):
    base = float(getattr(cfg, "local_rope_theta", getattr(cfg, "global_rope_theta", 10000.0)))

    # Different transformers versions expose different ctor signatures.
    builders = [
        lambda: ModernBertRotaryEmbedding(cfg, dim=dh, base=base),
        lambda: ModernBertRotaryEmbedding(cfg, dh, base),
        lambda: ModernBertRotaryEmbedding(cfg),
        lambda: ModernBertRotaryEmbedding(dh, base),
    ]
    last_err = None
    for b in builders:
        try:
            rot = b()
            return rot.to(device) if hasattr(rot, "to") else rot
        except TypeError as e:
            last_err = e
            continue
    raise TypeError(f"Unable to construct ModernBertRotaryEmbedding for this transformers version: {last_err}")


def main():
    parser = argparse.ArgumentParser("Legacy ModernBERT encoder compare (dense vs FlashSVD)")
    parser.add_argument("--model-config", type=str, default="", help="HF config path/id (optional)")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=0)
    parser.add_argument("--intermediate-size", type=int, default=0)
    parser.add_argument("--local-attention", type=int, default=0)

    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--L", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    parser.add_argument("--target-param-ratio", type=float, default=0.5)
    parser.add_argument("--rank-round-multiple", type=int, default=32)

    parser.add_argument("--window-radius", type=int, default=-1, help="override sliding radius; -1 means auto")
    parser.add_argument("--chunk-q", type=int, default=128)
    parser.add_argument("--pad-fraction", type=float, default=0.0)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--ffn-variant", type=str, default="auto", choices=["auto", "preg", "fused", "two_stage"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    this_dir = Path(__file__).resolve().parent
    geglu_mod = _load_module(this_dir / "flashsvdgeglu_v1.5.py", "flashsvdgeglu_v1_5")
    attn_mod = _load_module(this_dir / "flashsvdropeattn_v1.5_encoder.py", "flashsvdropeattn_v1_5_encoder")

    flashsvd_ffn_geglu_autotuned = geglu_mod.flashsvd_ffn_geglu_autotuned
    FlashSVDRoPEAttention = attn_mod.FlashSVDRoPEAttention
    QKVFactors = attn_mod.QKVFactors
    project_qkv_rank_packed = attn_mod.project_qkv_rank_packed

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    shape = load_shape(args)
    D = shape.hidden_size
    H = shape.num_heads
    Dh = D // H
    Fdim = shape.intermediate_size

    assert D % H == 0, f"hidden_size ({D}) must be divisible by num_heads ({H})"

    ranks = pick_ranks(shape, args.target_param_ratio, args.rank_round_multiple)
    r_attn, r1, r2 = ranks["r_attn"], ranks["r1"], ranks["r2"]

    B, L = args.B, args.L
    window_radius = args.window_radius if args.window_radius >= 0 else max(1, shape.local_attention // 2)

    print("==== Legacy ModernBERT Encoder Compare (dense vs FlashSVD) ====")
    print(
        f"Shape: B={B}, L={L}, D={D}, H={H}, Dh={Dh}, F={Fdim}, dtype={args.dtype}, "
        f"window_radius={window_radius}, ffn_variant={args.ffn_variant}"
    )
    print(
        f"Ranks: attn={r_attn}, ffn_r1={r1}, ffn_r2={r2} | target_param_ratio={args.target_param_ratio:.3f}"
    )

    # Inputs
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    pad_mask_2d = make_padding_mask(B, L, args.pad_fraction, device)
    position_ids = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, L)

    # Masks
    sliding_mask = build_sliding_additive_mask(B, L, window_radius, dtype, device, pad_mask_2d)

    # Rotary
    cfg_rot = ModernBertConfig(hidden_size=D, num_attention_heads=H)
    rotary = make_modernbert_rotary(cfg_rot, Dh, device)

    # Dense attention weights
    wq = torch.randn(D, D, device=device, dtype=dtype) * 0.02
    wk = torch.randn(D, D, device=device, dtype=dtype) * 0.02
    wv = torch.randn(D, D, device=device, dtype=dtype) * 0.02
    wo = torch.randn(D, D, device=device, dtype=dtype) * 0.02
    bq = torch.zeros(D, device=device, dtype=dtype)
    bk = torch.zeros(D, device=device, dtype=dtype)
    bv = torch.zeros(D, device=device, dtype=dtype)
    bo = torch.zeros(D, device=device, dtype=dtype)

    # Low-rank attention factors
    uq = torch.randn(D, r_attn, device=device, dtype=dtype) * 0.02
    uk = torch.randn(D, r_attn, device=device, dtype=dtype) * 0.02
    uv = torch.randn(D, r_attn, device=device, dtype=dtype) * 0.02
    vq = torch.randn(r_attn, D, device=device, dtype=dtype) * 0.02
    vk = torch.randn(r_attn, D, device=device, dtype=dtype) * 0.02
    vv = torch.randn(r_attn, D, device=device, dtype=dtype) * 0.02
    packed_qkv_u = attn_mod.get_precomputed_qkv_u(uq, uk, uv)

    # Dense FFN weights (GEGLU)
    wi = torch.randn(D, 2 * Fdim, device=device, dtype=dtype) * 0.02
    wff = torch.randn(Fdim, D, device=device, dtype=dtype) * 0.02
    b1 = torch.zeros(2 * Fdim, device=device, dtype=dtype)
    b2 = torch.zeros(D, device=device, dtype=dtype)

    # Low-rank FFN factors
    u1 = torch.randn(D, r1, device=device, dtype=dtype) * 0.02
    v1 = torch.randn(r1, 2 * Fdim, device=device, dtype=dtype) * 0.02
    u2 = torch.randn(Fdim, r2, device=device, dtype=dtype) * 0.02
    v2 = torch.randn(r2, D, device=device, dtype=dtype) * 0.02

    flash_attn = FlashSVDRoPEAttention(
        num_heads=H,
        head_dim=Dh,
        rotary_emb=rotary,
        chunk_q=args.chunk_q,
        default_window_radius=window_radius,
        enable_sliding_chunk=True,
        auto_infer_window=True,
    ).to(device)

    def dense_attention() -> torch.Tensor:
        q = torch.matmul(x, wq) + bq
        k = torch.matmul(x, wk) + bk
        v = torch.matmul(x, wv) + bv

        q = q.view(B, L, H, Dh).transpose(1, 2).contiguous()  # [B,H,L,Dh]
        k = k.view(B, L, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, L, H, Dh).transpose(1, 2).contiguous()

        qf = q.view(B * H, L, Dh)
        kf = k.view(B * H, L, Dh)
        posf = position_ids.unsqueeze(1).expand(B, H, L).reshape(B * H, L)
        cos, sin = rotary(qf, position_ids=posf)
        qf = apply_rotary(qf, cos, sin)
        kf = apply_rotary(kf, cos, sin)
        q = qf.view(B, H, L, Dh)
        k = kf.view(B, H, L, Dh)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=sliding_mask, dropout_p=0.0, is_causal=False)
        out = attn.transpose(1, 2).reshape(B, L, D)
        return torch.matmul(out, wo) + bo

    def lowrank_attention() -> torch.Tensor:
        pq, pk, pv = project_qkv_rank_packed(x, uq, uk, uv, packed_u=packed_qkv_u)

        qkv = QKVFactors(
            Pq=pq,
            Pk=pk,
            Pv=pv,
            Vq=vq,
            Vk=vk,
            Vv=vv,
            bq=bq,
            bk=bk,
            bv=bv,
        )
        attn = flash_attn(
            qkv,
            attention_mask=pad_mask_2d,
            position_ids=position_ids,
            sliding_window_mask=sliding_mask,
        )
        out = attn.transpose(1, 2).reshape(B, L, D)
        return torch.matmul(out, wo) + bo

    def dense_ffn(inp: torch.Tensor = x) -> torch.Tensor:
        z = torch.matmul(inp, wi) + b1
        zu, zv = z.chunk(2, dim=-1)
        h = F.gelu(zu, approximate="tanh") * zv
        return torch.matmul(h, wff) + b2

    def lowrank_ffn(inp: torch.Tensor = x) -> torch.Tensor:
        p = torch.matmul(inp, u1)
        return flashsvd_ffn_geglu_autotuned(
            p,
            v1,
            u2,
            v2,
            b1,
            b2,
            gelu_approx="tanh",
            store_s_fp32=False,
            kernel_variant=args.ffn_variant,
        )

    def dense_layer() -> torch.Tensor:
        return dense_attention() + dense_ffn(x)

    def lowrank_layer() -> torch.Tensor:
        return lowrank_attention() + lowrank_ffn(x)

    results = {}
    results["dense_attn"] = bench_cuda("dense_attn", dense_attention, B, L, args.warmup, args.iters)
    results["lowrank_attn"] = bench_cuda("lowrank_attn", lowrank_attention, B, L, args.warmup, args.iters)
    results["dense_ffn"] = bench_cuda("dense_ffn", dense_ffn, B, L, args.warmup, args.iters)
    results["lowrank_ffn"] = bench_cuda("lowrank_ffn", lowrank_ffn, B, L, args.warmup, args.iters)
    results["dense_layer"] = bench_cuda("dense_layer", dense_layer, B, L, args.warmup, args.iters)
    results["lowrank_layer"] = bench_cuda("lowrank_layer", lowrank_layer, B, L, args.warmup, args.iters)

    print("\n=== Results ===")
    for k in [
        "dense_attn",
        "lowrank_attn",
        "dense_ffn",
        "lowrank_ffn",
        "dense_layer",
        "lowrank_layer",
    ]:
        print(f"- {k:12s}: {_fmt(results[k])}")

    print("\n=== Speedup (lowrank / dense) ===")
    attn_sp = results["dense_attn"].mean_ms / results["lowrank_attn"].mean_ms
    ffn_sp = results["dense_ffn"].mean_ms / results["lowrank_ffn"].mean_ms
    layer_sp = results["dense_layer"].mean_ms / results["lowrank_layer"].mean_ms
    print(f"- attention: x{attn_sp:.3f}")
    print(f"- ffn      : x{ffn_sp:.3f}")
    print(f"- combined : x{layer_sp:.3f}")


if __name__ == "__main__":
    main()
