#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LlamaPreset:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} is not divisible by num_heads={self.num_heads}"
            )
        return self.hidden_size // self.num_heads


LLAMA_PRESETS: dict[str, LlamaPreset] = {
    "llama2-7b": LlamaPreset(hidden_size=4096, intermediate_size=11008, num_heads=32, num_kv_heads=32),
    "llama2-13b": LlamaPreset(hidden_size=5120, intermediate_size=13824, num_heads=40, num_kv_heads=40),
    "llama2-70b": LlamaPreset(hidden_size=8192, intermediate_size=28672, num_heads=64, num_kv_heads=8),
    "llama3-8b": LlamaPreset(hidden_size=4096, intermediate_size=14336, num_heads=32, num_kv_heads=8),
    "llama3-70b": LlamaPreset(hidden_size=8192, intermediate_size=28672, num_heads=64, num_kv_heads=8),
    "llama3.1-8b": LlamaPreset(hidden_size=4096, intermediate_size=14336, num_heads=32, num_kv_heads=8),
    "llama3.1-70b": LlamaPreset(hidden_size=8192, intermediate_size=28672, num_heads=64, num_kv_heads=8),
}


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _dtype_from_name(name: str) -> torch.dtype:
    raw = name.strip().lower()
    if raw in {"fp16", "float16", "half"}:
        return torch.float16
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype)


def _bench_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    try:
        import triton  # type: ignore

        return float(triton.testing.do_bench(fn, warmup=warmup, rep=iters))
    except Exception:
        for _ in range(max(1, warmup)):
            _ = fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            _ = fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        normed = normed.to(weight.dtype)
    return normed * weight


def _build_rope_tables_half(
    seqlen: int,
    head_dim: int,
    base: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    pos = torch.arange(seqlen, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = torch.einsum("m,d->md", pos, inv_freq)
    return torch.cos(ang).to(dtype).contiguous(), torch.sin(ang).to(dtype).contiguous()


def _rope_apply_bh1d(x_bh1d: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    dh = int(x_bh1d.shape[-1])
    half = dh // 2
    x0 = x_bh1d[..., :half]
    x1 = x_bh1d[..., half:]
    cos = cos_half.reshape(1, 1, 1, half)
    sin = sin_half.reshape(1, 1, 1, half)
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)


def _import_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_cuda_graph_replay(
    *,
    device: torch.device,
    prep_inputs: Callable[[], None],
    run_capture: Callable[[], torch.Tensor],
    clone_output: bool = True,
) -> Callable[[], torch.Tensor]:
    prep_inputs()
    warm_stream = torch.cuda.Stream(device=device)
    current_stream = torch.cuda.current_stream(device=device)
    warm_stream.wait_stream(current_stream)
    with torch.cuda.stream(warm_stream):
        for _ in range(3):
            y = run_capture()
            _ = y.reshape(-1)[0]
    current_stream.wait_stream(warm_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_static = run_capture()

    def _run():
        prep_inputs()
        graph.replay()
        if clone_output:
            return y_static.clone()
        return y_static

    return _run


def _load_dense_decode_mod():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "flashsvdropeattn_dense_decode.py")
    return _import_from_path("flashsvdropeattn_dense_decode_stack_compare", path)


def _load_flashsvd_swiglu_mod():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("kernels.flashsvdswiglu")


def _try_load_flash_attn2_kvcache() -> Optional[Callable[..., torch.Tensor]]:
    try:
        from flash_attn import flash_attn_with_kvcache  # type: ignore

        return flash_attn_with_kvcache
    except Exception:
        pass
    try:
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # type: ignore

        return flash_attn_with_kvcache
    except Exception:
        return None


def _call_flash_attn2_kvcache(
    flash_attn_with_kvcache: Callable[..., torch.Tensor],
    q_bmhd: torch.Tensor,
    k_cache_bmhd: torch.Tensor,
    v_cache_bmhd: torch.Tensor,
    *,
    k_bmhd: torch.Tensor,
    v_bmhd: torch.Tensor,
    cache_seqlens: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    sig = inspect.signature(flash_attn_with_kvcache)
    params = sig.parameters
    kwargs: dict[str, object] = {}
    if "k" in params:
        kwargs["k"] = k_bmhd
    if "v" in params:
        kwargs["v"] = v_bmhd
    if "cache_seqlens" in params:
        kwargs["cache_seqlens"] = cache_seqlens
    if "rotary_cos" in params:
        kwargs["rotary_cos"] = rotary_cos
    if "rotary_sin" in params:
        kwargs["rotary_sin"] = rotary_sin
    if "rotary_interleaved" in params:
        kwargs["rotary_interleaved"] = False
    if "causal" in params:
        kwargs["causal"] = causal
    if "dropout_p" in params:
        kwargs["dropout_p"] = 0.0
    if "window_size" in params:
        kwargs["window_size"] = (-1, -1)
    return flash_attn_with_kvcache(q_bmhd, k_cache_bmhd, v_cache_bmhd, **kwargs)


class OriginalSVDLlamaAttention(nn.Module):
    def __init__(self, preset: LlamaPreset, ratio: float):
        super().__init__()
        self.hidden_size = int(preset.hidden_size)
        self.num_heads = int(preset.num_heads)
        self.num_kv_heads = int(preset.num_kv_heads)
        self.head_dim = int(preset.head_dim)
        self.ratio = float(ratio)
        self.low_rank = max(1, int(self.hidden_size * self.ratio / 2))
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_u_proj = nn.Linear(self.low_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, self.low_rank, bias=False)
        self.k_u_proj = nn.Linear(self.low_rank, self.num_kv_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, self.low_rank, bias=False)
        self.v_u_proj = nn.Linear(self.low_rank, self.num_kv_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, self.low_rank, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, self.low_rank, bias=False)
        self.o_u_proj = nn.Linear(self.low_rank, self.hidden_size, bias=False)

        self._prepacked_cache: dict[tuple[str, int | None, torch.dtype], tuple[torch.Tensor, ...]] = {}

    def _extract_decode_factors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = int(self.low_rank)
        h = int(self.num_heads)
        hk = int(self.num_kv_heads)
        dh = int(self.head_dim)
        vq = self.q_u_proj.weight.view(h, dh, r).permute(0, 2, 1).contiguous()
        vk = self.k_u_proj.weight.view(hk, dh, r).permute(0, 2, 1).contiguous()
        vv = self.v_u_proj.weight.view(hk, dh, r).permute(0, 2, 1).contiguous()
        return vq, vk, vv

    def get_optimized_decode_tensors(
        self,
        dense_decode_mod,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, dtype)
        cached = self._prepacked_cache.get(key)
        if cached is not None:
            return cached

        qkv_rank = torch.cat(
            [
                self.q_v_proj.weight.t(),
                self.k_v_proj.weight.t(),
                self.v_v_proj.weight.t(),
            ],
            dim=1,
        ).to(device=device, dtype=dtype).contiguous()
        vq, vk, vv = self._extract_decode_factors()
        vq = vq.to(device=device, dtype=dtype).contiguous()
        vk = vk.to(device=device, dtype=dtype).contiguous()
        vv = vv.to(device=device, dtype=dtype).contiguous()
        vq_flat, vk_flat, vv_flat = dense_decode_mod.pack_qkv_shared_bases(vq, vk, vv)
        out = (qkv_rank, vq_flat, vk_flat, vv_flat)
        self._prepacked_cache[key] = out
        return out


class OriginalSVDLlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, ratio: float, *, share_v_proj: bool):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.ratio = float(ratio)
        self.low_rank = max(
            1,
            int(self.intermediate_size * self.hidden_size * self.ratio / (self.intermediate_size + self.hidden_size)),
        )
        self.gate_u_proj = nn.Linear(self.low_rank, self.intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(self.hidden_size, self.low_rank, bias=False)
        self.down_u_proj = nn.Linear(self.low_rank, self.hidden_size, bias=False)
        self.down_v_proj = nn.Linear(self.intermediate_size, self.low_rank, bias=False)
        self.up_u_proj = nn.Linear(self.low_rank, self.intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(self.hidden_size, self.low_rank, bias=False)
        if share_v_proj:
            self.gate_v_proj.weight.data.copy_(self.up_v_proj.weight.data)
        self._flashsvd_cache: dict[tuple[str, int | None, torch.dtype], tuple[torch.Tensor, ...]] = {}
        self._shared_cublas_cache: dict[tuple[str, int | None, torch.dtype], tuple[torch.Tensor, ...]] = {}
        self._shared_split_kernel_cache: dict[tuple[str, int | None, torch.dtype], tuple[torch.Tensor, ...]] = {}

    def get_flashsvd_factors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, dtype)
        cached = self._flashsvd_cache.get(key)
        if cached is not None:
            return cached

        v1 = torch.cat(
            [self.up_u_proj.weight.t(), self.gate_u_proj.weight.t()],
            dim=1,
        ).to(device=device, dtype=dtype).contiguous()
        u2 = self.down_v_proj.weight.t().to(device=device, dtype=dtype)
        v2 = self.down_u_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        b1 = torch.zeros((2 * self.intermediate_size,), device=device, dtype=dtype)
        b2 = torch.zeros((self.hidden_size,), device=device, dtype=dtype)
        out = (v1, u2, v2, b1, b2)
        self._flashsvd_cache[key] = out
        return out

    def get_shared_cublas_factors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, dtype)
        cached = self._shared_cublas_cache.get(key)
        if cached is not None:
            return cached

        # Exact shared-P form for checkpoints where gate_v_proj == up_v_proj.
        # Keep gate first so the activation matches silu(gate) * up.
        v1 = torch.cat(
            [self.gate_u_proj.weight.t(), self.up_u_proj.weight.t()],
            dim=1,
        ).to(device=device, dtype=dtype).contiguous()
        u2 = self.down_v_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        v2 = self.down_u_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        b1 = torch.zeros((2 * self.intermediate_size,), device=device, dtype=dtype)
        b2 = torch.zeros((self.hidden_size,), device=device, dtype=dtype)
        out = (v1, u2, v2, b1, b2)
        self._shared_cublas_cache[key] = out
        return out

    def get_shared_split_kernel_factors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device.type, device.index, dtype)
        cached = self._shared_split_kernel_cache.get(key)
        if cached is not None:
            return cached

        gate_u = self.gate_u_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        up_u = self.up_u_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        down_v = self.down_v_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        down_u = self.down_u_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        b2 = torch.zeros((self.hidden_size,), device=device, dtype=dtype)
        out = (gate_u, up_u, down_v, down_u, b2)
        self._shared_split_kernel_cache[key] = out
        return out


def _original_attention_step(
    attn: OriginalSVDLlamaAttention,
    hidden_states: torch.Tensor,
    *,
    past_k_bhld: torch.Tensor,
    past_v_bhld: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    q_pos: int,
) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.shape
    if q_len != 1:
        raise ValueError(f"Only q_len=1 is supported, got {q_len}")

    h = attn.num_heads
    hk = attn.num_kv_heads
    dh = attn.head_dim
    rep = h // hk

    q = attn.q_u_proj(attn.q_v_proj(hidden_states)).view(bsz, 1, h, dh).transpose(1, 2).contiguous()
    k = attn.k_u_proj(attn.k_v_proj(hidden_states)).view(bsz, 1, hk, dh).transpose(1, 2).contiguous()
    v = attn.v_u_proj(attn.v_v_proj(hidden_states)).view(bsz, 1, hk, dh).transpose(1, 2).contiguous()

    q = _rope_apply_bh1d(q, cos_half[q_pos], sin_half[q_pos])
    k = _rope_apply_bh1d(k, cos_half[q_pos], sin_half[q_pos])

    key_states = torch.cat([past_k_bhld, k], dim=2)
    value_states = torch.cat([past_v_bhld, v], dim=2)

    if hk != h:
        key_states = key_states.repeat_interleave(rep, dim=1).contiguous()
        value_states = value_states.repeat_interleave(rep, dim=1).contiguous()

    attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * attn.scale
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, 1, h * dh).contiguous()
    return attn.o_u_proj(attn.o_v_proj(attn_output))


def _optimized_attention_step(
    attn: OriginalSVDLlamaAttention,
    hidden_states: torch.Tensor,
    *,
    dense_decode_mod,
    flash_attn_with_kvcache: Callable[..., torch.Tensor],
    packed_qkv_rank: torch.Tensor,
    vq_flat: torch.Tensor,
    vk_flat: torch.Tensor,
    vv_flat: torch.Tensor,
    k_cache_bmhd: torch.Tensor,
    v_cache_bmhd: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.shape
    if q_len != 1:
        raise ValueError(f"Only q_len=1 is supported, got {q_len}")

    hidden_flat = hidden_states[:, 0, :].contiguous()
    packed_rank = torch.matmul(hidden_flat, packed_qkv_rank)
    rank = attn.low_rank
    q_rank, k_rank, v_rank = torch.split(packed_rank, rank, dim=1)
    q_bhd, k_bkd, v_bkd = dense_decode_mod.reconstruct_qkv_token_shared_prepacked(
        q_rank,
        k_rank,
        v_rank,
        vq_flat,
        vk_flat,
        vv_flat,
        H=attn.num_heads,
        Hk=attn.num_kv_heads,
        Dh=attn.head_dim,
    )
    q_bmhd = q_bhd[:, None, :, :].contiguous()
    k_bmhd = k_bkd[:, None, :, :].contiguous()
    v_bmhd = v_bkd[:, None, :, :].contiguous()
    out = _call_flash_attn2_kvcache(
        flash_attn_with_kvcache,
        q_bmhd,
        k_cache_bmhd,
        v_cache_bmhd,
        k_bmhd=k_bmhd,
        v_bmhd=v_bmhd,
        cache_seqlens=cache_seqlens,
        rotary_cos=cos_half,
        rotary_sin=sin_half,
        causal=True,
    )
    if out.shape == (bsz, 1, attn.num_heads, attn.head_dim):
        out_dense = out.reshape(bsz, 1, attn.num_heads * attn.head_dim).contiguous()
    elif out.shape == (bsz, attn.num_heads, 1, attn.head_dim):
        out_dense = out.transpose(1, 2).reshape(bsz, 1, attn.num_heads * attn.head_dim).contiguous()
    else:
        raise ValueError(f"Unexpected flash_attn_with_kvcache output shape: {tuple(out.shape)}")
    return attn.o_u_proj(attn.o_v_proj(out_dense))


def _original_mlp_step(mlp: OriginalSVDLlamaMLP, hidden_states: torch.Tensor) -> torch.Tensor:
    up = mlp.up_u_proj(mlp.up_v_proj(hidden_states))
    gate = mlp.gate_u_proj(mlp.gate_v_proj(hidden_states))
    return mlp.down_u_proj(mlp.down_v_proj(F.silu(gate) * up))


def _flashsvd_mlp_step(
    mlp: OriginalSVDLlamaMLP,
    hidden_states: torch.Tensor,
    *,
    flashsvd_ffn_swiglu: Callable[..., torch.Tensor],
) -> torch.Tensor:
    v1, u2, v2, b1, b2 = mlp.get_flashsvd_factors(device=hidden_states.device, dtype=hidden_states.dtype)
    p = mlp.up_v_proj(hidden_states)
    return flashsvd_ffn_swiglu(p, v1, u2, v2, b1, b2, use_autotune=True)


def _shared_cublas_mlp_step(
    mlp: OriginalSVDLlamaMLP,
    hidden_states: torch.Tensor,
    *,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if factors is None:
        factors = mlp.get_shared_cublas_factors(device=hidden_states.device, dtype=hidden_states.dtype)
    v1, u2, v2, b1, b2 = factors
    n_tokens = int(hidden_states.shape[0] * hidden_states.shape[1])
    p2d = mlp.up_v_proj(hidden_states).reshape(n_tokens, mlp.low_rank)
    z2d = torch.addmm(b1, p2d, v1)
    gate2d, up2d = torch.split(z2d, mlp.intermediate_size, dim=1)
    h2d = F.silu(gate2d) * up2d
    s2d = torch.matmul(h2d, u2)
    y2d = torch.addmm(b2, s2d, v2)
    return y2d.reshape(hidden_states.shape[0], hidden_states.shape[1], mlp.hidden_size)


def _shared_split_mlp_step(
    mlp: OriginalSVDLlamaMLP,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # Exact when gate_v_proj == up_v_proj: reuse the same rank-space P, but keep the
    # rest of the computation in the same shape as the original MLP.
    p = mlp.up_v_proj(hidden_states)
    gate = mlp.gate_u_proj(p)
    up = mlp.up_u_proj(p)
    return mlp.down_u_proj(mlp.down_v_proj(F.silu(gate) * up))


def _shared_split_kernel_mlp_step(
    mlp: OriginalSVDLlamaMLP,
    hidden_states: torch.Tensor,
    *,
    flashsvd_ffn_shared_split_token: Callable[..., torch.Tensor],
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    if factors is None:
        factors = mlp.get_shared_split_kernel_factors(device=hidden_states.device, dtype=hidden_states.dtype)
    gate_u, up_u, down_v, down_u, b2 = factors
    p = mlp.up_v_proj(hidden_states)
    return flashsvd_ffn_shared_split_token(
        p,
        gate_u,
        up_u,
        down_v,
        down_u,
        b2,
        BR=128,
        BD=128,
        BR2=128,
        num_warps=8,
        num_stages=2,
        store_partials_fp32=False,
        use_atomic_accum=False,
    )


def _build_dense_kv_cache(
    *,
    batch_size: int,
    kv_len: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_len <= 0:
        shape = (batch_size, num_kv_heads, 0, head_dim)
        return (
            torch.empty(shape, device=device, dtype=dtype),
            torch.empty(shape, device=device, dtype=dtype),
        )
    past_k = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype).contiguous()
    past_v = torch.randn(batch_size, num_kv_heads, kv_len, head_dim, device=device, dtype=dtype).contiguous()
    return past_k, past_v


def _build_kvcache_buffers(
    past_k_bhld: torch.Tensor,
    past_v_bhld: torch.Tensor,
    *,
    total_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, hk, past_len, dh = past_k_bhld.shape
    k_cache = torch.empty((bsz, total_len, hk, dh), device=past_k_bhld.device, dtype=past_k_bhld.dtype)
    v_cache = torch.empty_like(k_cache)
    if past_len > 0:
        k_cache[:, :past_len].copy_(past_k_bhld.permute(0, 2, 1, 3).contiguous())
        v_cache[:, :past_len].copy_(past_v_bhld.permute(0, 2, 1, 3).contiguous())
    if total_len > past_len:
        k_cache[:, past_len:].zero_()
        v_cache[:, past_len:].zero_()
    cache_seqlens = torch.full((bsz,), past_len, device=past_k_bhld.device, dtype=torch.int32)
    return k_cache, v_cache, cache_seqlens


def _fmt_diff(x: float) -> str:
    if x == 0.0:
        return "0"
    if x < 1e-3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser("Single-layer decode comparison: original SVD-Llama vs FlashSVD attention/MLP")
    ap.add_argument("--llama", type=str, default="llama2-7b", choices=sorted(LLAMA_PRESETS.keys()))
    ap.add_argument("--ratio", type=float, default=0.5, help="Low-rank compression ratio used by the original SVD modules.")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--Ls", type=str, default="256,1024,4096,8192")
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--rope-base", type=float, default=10000.0)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--share-mlp-vproj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Synthetic benchmark option: force gate_v_proj == up_v_proj. Keep this off for real checkpoint-aligned runs.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    flash_attn_with_kvcache = _try_load_flash_attn2_kvcache()
    if flash_attn_with_kvcache is None:
        raise RuntimeError("flash_attn_with_kvcache is required for the optimized attention benchmark.")

    dtype = _dtype_from_name(args.dtype)
    preset = LLAMA_PRESETS[args.llama]
    batch_size = int(args.B)
    seq_lens = [x for x in _parse_csv_ints(args.Ls) if x > 0]
    device = torch.device("cuda")
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    dense_decode_mod = _load_dense_decode_mod()
    flashsvd_swiglu_mod = _load_flashsvd_swiglu_mod()
    flashsvd_ffn_swiglu = flashsvd_swiglu_mod.flashsvd_ffn_swiglu
    flashsvd_ffn_shared_split_token = flashsvd_swiglu_mod.flashsvd_ffn_shared_split_token

    attn = OriginalSVDLlamaAttention(preset, args.ratio).to(device=device, dtype=dtype).eval()
    mlp = OriginalSVDLlamaMLP(
        hidden_size=preset.hidden_size,
        intermediate_size=preset.intermediate_size,
        ratio=args.ratio,
        share_v_proj=bool(args.share_mlp_vproj),
    ).to(device=device, dtype=dtype).eval()

    attn_norm_weight = torch.ones((preset.hidden_size,), device=device, dtype=dtype)
    post_norm_weight = torch.ones((preset.hidden_size,), device=device, dtype=dtype)
    packed_qkv_rank, vq_flat, vk_flat, vv_flat = attn.get_optimized_decode_tensors(
        dense_decode_mod,
        device=device,
        dtype=dtype,
    )

    print("==== Decode Stack Comparison ====")
    print(
        f"Setup: llama={args.llama} | B={batch_size} | Ls={','.join(str(x) for x in seq_lens)} | "
        f"hidden={preset.hidden_size} interm={preset.intermediate_size} | "
        f"H={preset.num_heads} Hk={preset.num_kv_heads} Dh={preset.head_dim} | "
        f"attn_rank={attn.low_rank} mlp_rank={mlp.low_rank} | dtype={_dtype_name(dtype)} | "
        f"share_mlp_vproj={int(args.share_mlp_vproj)}"
    )
    print(
        "Rows: L | attn orig->opt | mlp orig/flash/shared_split_g/shared_split_kernel_g | "
        "layer orig / +attn / +split_g / best(attn+kernel_g) | best speedup | "
        "diffs(attn,+attn,+split_g,best)"
    )

    full_speedups: list[float] = []
    attn_speedups: list[float] = []
    mlp_speedups: list[float] = []
    shared_split_mlp_speedups: list[float] = []
    shared_split_kernel_mlp_speedups: list[float] = []

    with torch.inference_mode():
        for total_len in seq_lens:
            q_pos = total_len - 1
            cos_half, sin_half = _build_rope_tables_half(
                total_len,
                preset.head_dim,
                args.rope_base,
                device=device,
                dtype=dtype,
            )
            hidden_states = torch.randn(batch_size, 1, preset.hidden_size, device=device, dtype=dtype).contiguous()
            past_k_bhld, past_v_bhld = _build_dense_kv_cache(
                batch_size=batch_size,
                kv_len=max(0, total_len - 1),
                num_kv_heads=preset.num_kv_heads,
                head_dim=preset.head_dim,
                device=device,
                dtype=dtype,
            )

            attn_input = _rms_norm(hidden_states, attn_norm_weight, args.eps).contiguous()
            k_cache_bmhd, v_cache_bmhd, cache_seqlens = _build_kvcache_buffers(
                past_k_bhld,
                past_v_bhld,
                total_len=total_len,
            )

            def _orig_attn():
                return _original_attention_step(
                    attn,
                    attn_input,
                    past_k_bhld=past_k_bhld,
                    past_v_bhld=past_v_bhld,
                    cos_half=cos_half,
                    sin_half=sin_half,
                    q_pos=q_pos,
                )

            def _opt_attn():
                return _optimized_attention_step(
                    attn,
                    attn_input,
                    dense_decode_mod=dense_decode_mod,
                    flash_attn_with_kvcache=flash_attn_with_kvcache,
                    packed_qkv_rank=packed_qkv_rank,
                    vq_flat=vq_flat,
                    vk_flat=vk_flat,
                    vv_flat=vv_flat,
                    k_cache_bmhd=k_cache_bmhd,
                    v_cache_bmhd=v_cache_bmhd,
                    cache_seqlens=cache_seqlens,
                    cos_half=cos_half,
                    sin_half=sin_half,
                )

            attn_ref = _orig_attn()
            attn_opt = _opt_attn()
            attn_hidden_ref = hidden_states + attn_ref
            mlp_input = _rms_norm(attn_hidden_ref, post_norm_weight, args.eps).contiguous()
            shared_split_kernel_factors = mlp.get_shared_split_kernel_factors(device=device, dtype=dtype)

            def _orig_mlp():
                return _original_mlp_step(mlp, mlp_input)

            def _flash_mlp():
                return _flashsvd_mlp_step(
                    mlp,
                    mlp_input,
                    flashsvd_ffn_swiglu=flashsvd_ffn_swiglu,
                )

            def _shared_split_mlp():
                return _shared_split_mlp_step(mlp, mlp_input)

            def _shared_split_kernel_mlp():
                return _shared_split_kernel_mlp_step(
                    mlp,
                    mlp_input,
                    flashsvd_ffn_shared_split_token=flashsvd_ffn_shared_split_token,
                    factors=shared_split_kernel_factors,
                )

            mlp_ref = _orig_mlp()
            mlp_opt = _flash_mlp()
            mlp_shared_split = _shared_split_mlp()
            mlp_shared_split_kernel = _shared_split_kernel_mlp()

            static_mlp_input = torch.empty_like(mlp_input)

            static_split_mlp_input = torch.empty_like(mlp_input)

            def _prep_shared_split_graph_inputs():
                static_split_mlp_input.copy_(mlp_input)

            def _shared_split_graph_body():
                return _shared_split_mlp_step(mlp, static_split_mlp_input)

            _shared_split_mlp_graph = _make_cuda_graph_replay(
                device=device,
                prep_inputs=_prep_shared_split_graph_inputs,
                run_capture=_shared_split_graph_body,
                clone_output=False,
            )
            mlp_shared_split_graph = _shared_split_mlp_graph()

            static_split_kernel_mlp_input = torch.empty_like(mlp_input)

            def _prep_shared_split_kernel_graph_inputs():
                static_split_kernel_mlp_input.copy_(mlp_input)

            def _shared_split_kernel_graph_body():
                return _shared_split_kernel_mlp_step(
                    mlp,
                    static_split_kernel_mlp_input,
                    flashsvd_ffn_shared_split_token=flashsvd_ffn_shared_split_token,
                    factors=shared_split_kernel_factors,
                )

            _shared_split_kernel_mlp_graph = _make_cuda_graph_replay(
                device=device,
                prep_inputs=_prep_shared_split_kernel_graph_inputs,
                run_capture=_shared_split_kernel_graph_body,
                clone_output=False,
            )
            mlp_shared_split_kernel_graph = _shared_split_kernel_mlp_graph()

            def _orig_layer():
                attn_out = _original_attention_step(
                    attn,
                    attn_input,
                    past_k_bhld=past_k_bhld,
                    past_v_bhld=past_v_bhld,
                    cos_half=cos_half,
                    sin_half=sin_half,
                    q_pos=q_pos,
                )
                hidden_mid = hidden_states + attn_out
                mlp_in = _rms_norm(hidden_mid, post_norm_weight, args.eps)
                mlp_out = _original_mlp_step(mlp, mlp_in)
                return hidden_mid + mlp_out

            def _layer_attn_opt():
                attn_out = _optimized_attention_step(
                    attn,
                    attn_input,
                    dense_decode_mod=dense_decode_mod,
                    flash_attn_with_kvcache=flash_attn_with_kvcache,
                    packed_qkv_rank=packed_qkv_rank,
                    vq_flat=vq_flat,
                    vk_flat=vk_flat,
                    vv_flat=vv_flat,
                    k_cache_bmhd=k_cache_bmhd,
                    v_cache_bmhd=v_cache_bmhd,
                    cache_seqlens=cache_seqlens,
                    cos_half=cos_half,
                    sin_half=sin_half,
                )
                hidden_mid = hidden_states + attn_out
                mlp_in = _rms_norm(hidden_mid, post_norm_weight, args.eps)
                mlp_out = _original_mlp_step(mlp, mlp_in)
                return hidden_mid + mlp_out

            def _layer_mlp_opt():
                attn_out = _original_attention_step(
                    attn,
                    attn_input,
                    past_k_bhld=past_k_bhld,
                    past_v_bhld=past_v_bhld,
                    cos_half=cos_half,
                    sin_half=sin_half,
                    q_pos=q_pos,
                )
                hidden_mid = hidden_states + attn_out
                mlp_in = _rms_norm(hidden_mid, post_norm_weight, args.eps)
                mlp_out = _flashsvd_mlp_step(
                    mlp,
                    mlp_in,
                    flashsvd_ffn_swiglu=flashsvd_ffn_swiglu,
                )
                return hidden_mid + mlp_out

            def _layer_shared_split_opt():
                attn_out = _original_attention_step(
                    attn,
                    attn_input,
                    past_k_bhld=past_k_bhld,
                    past_v_bhld=past_v_bhld,
                    cos_half=cos_half,
                    sin_half=sin_half,
                    q_pos=q_pos,
                )
                hidden_mid = hidden_states + attn_out
                mlp_in = _rms_norm(hidden_mid, post_norm_weight, args.eps)
                mlp_out = _shared_split_mlp_step(mlp, mlp_in)
                return hidden_mid + mlp_out

            def _layer_best_opt():
                attn_out = _optimized_attention_step(
                    attn,
                    attn_input,
                    dense_decode_mod=dense_decode_mod,
                    flash_attn_with_kvcache=flash_attn_with_kvcache,
                    packed_qkv_rank=packed_qkv_rank,
                    vq_flat=vq_flat,
                    vk_flat=vk_flat,
                    vv_flat=vv_flat,
                    k_cache_bmhd=k_cache_bmhd,
                    v_cache_bmhd=v_cache_bmhd,
                    cache_seqlens=cache_seqlens,
                    cos_half=cos_half,
                    sin_half=sin_half,
                )
                hidden_mid = hidden_states + attn_out
                mlp_in = _rms_norm(hidden_mid, post_norm_weight, args.eps)
                mlp_out = _shared_split_kernel_mlp_step(
                    mlp,
                    mlp_in,
                    flashsvd_ffn_shared_split_token=flashsvd_ffn_shared_split_token,
                    factors=shared_split_kernel_factors,
                )
                return hidden_mid + mlp_out

            static_hidden_states = torch.empty_like(hidden_states)

            def _prep_layer_shared_split_graph_inputs():
                static_hidden_states.copy_(hidden_states)

            def _layer_shared_split_graph_body():
                hs = static_hidden_states
                attn_in = _rms_norm(hs, attn_norm_weight, args.eps)
                attn_out = _original_attention_step(
                    attn,
                    attn_in,
                    past_k_bhld=past_k_bhld,
                    past_v_bhld=past_v_bhld,
                    cos_half=cos_half,
                    sin_half=sin_half,
                    q_pos=q_pos,
                )
                hidden_mid = hs + attn_out
                mlp_in = _rms_norm(hidden_mid, post_norm_weight, args.eps)
                mlp_out = _shared_split_mlp_step(mlp, mlp_in)
                return hidden_mid + mlp_out

            _layer_shared_split_graph = _make_cuda_graph_replay(
                device=device,
                prep_inputs=_prep_layer_shared_split_graph_inputs,
                run_capture=_layer_shared_split_graph_body,
                clone_output=False,
            )

            def _prep_layer_shared_split_kernel_graph_inputs():
                static_hidden_states.copy_(hidden_states)

            def _layer_shared_split_kernel_graph_body():
                hs = static_hidden_states
                attn_in = _rms_norm(hs, attn_norm_weight, args.eps)
                attn_out = _original_attention_step(
                    attn,
                    attn_in,
                    past_k_bhld=past_k_bhld,
                    past_v_bhld=past_v_bhld,
                    cos_half=cos_half,
                    sin_half=sin_half,
                    q_pos=q_pos,
                )
                hidden_mid = hs + attn_out
                mlp_in = _rms_norm(hidden_mid, post_norm_weight, args.eps)
                mlp_out = _shared_split_kernel_mlp_step(
                    mlp,
                    mlp_in,
                    flashsvd_ffn_shared_split_token=flashsvd_ffn_shared_split_token,
                    factors=shared_split_kernel_factors,
                )
                return hidden_mid + mlp_out

            _layer_shared_split_kernel_graph = _make_cuda_graph_replay(
                device=device,
                prep_inputs=_prep_layer_shared_split_kernel_graph_inputs,
                run_capture=_layer_shared_split_kernel_graph_body,
                clone_output=False,
            )

            def _prep_layer_best_graph_inputs():
                static_hidden_states.copy_(hidden_states)

            def _layer_best_graph_body():
                hs = static_hidden_states
                attn_in = _rms_norm(hs, attn_norm_weight, args.eps)
                attn_out = _optimized_attention_step(
                    attn,
                    attn_in,
                    dense_decode_mod=dense_decode_mod,
                    flash_attn_with_kvcache=flash_attn_with_kvcache,
                    packed_qkv_rank=packed_qkv_rank,
                    vq_flat=vq_flat,
                    vk_flat=vk_flat,
                    vv_flat=vv_flat,
                    k_cache_bmhd=k_cache_bmhd,
                    v_cache_bmhd=v_cache_bmhd,
                    cache_seqlens=cache_seqlens,
                    cos_half=cos_half,
                    sin_half=sin_half,
                )
                hidden_mid = hs + attn_out
                mlp_in = _rms_norm(hidden_mid, post_norm_weight, args.eps)
                mlp_out = _shared_split_kernel_mlp_step(
                    mlp,
                    mlp_in,
                    flashsvd_ffn_shared_split_token=flashsvd_ffn_shared_split_token,
                    factors=shared_split_kernel_factors,
                )
                return hidden_mid + mlp_out

            _layer_best_graph = _make_cuda_graph_replay(
                device=device,
                prep_inputs=_prep_layer_best_graph_inputs,
                run_capture=_layer_best_graph_body,
                clone_output=False,
            )

            layer_ref = _orig_layer()
            layer_attn_opt = _layer_attn_opt()
            layer_mlp_opt = _layer_mlp_opt()
            layer_shared_split_opt = _layer_shared_split_opt()
            layer_shared_split_graph = _layer_shared_split_graph()
            layer_best_opt = _layer_best_opt()
            layer_best_graph = _layer_best_graph()
            layer_shared_split_kernel_graph = _layer_shared_split_kernel_graph()

            diff_attn = _max_abs_diff(attn_ref, attn_opt)
            diff_layer_attn = _max_abs_diff(layer_ref, layer_attn_opt)
            diff_layer_shared_split = _max_abs_diff(layer_ref, layer_shared_split_graph)
            diff_layer_best = _max_abs_diff(layer_ref, layer_best_graph)

            attn_orig_ms = _bench_ms(_orig_attn, warmup=args.warmup, iters=args.iters)
            attn_opt_ms = _bench_ms(_opt_attn, warmup=args.warmup, iters=args.iters)
            mlp_orig_ms = _bench_ms(_orig_mlp, warmup=args.warmup, iters=args.iters)
            mlp_opt_ms = _bench_ms(_flash_mlp, warmup=args.warmup, iters=args.iters)
            mlp_shared_split_ms = _bench_ms(_shared_split_mlp, warmup=args.warmup, iters=args.iters)
            mlp_shared_split_graph_ms = _bench_ms(_shared_split_mlp_graph, warmup=args.warmup, iters=args.iters)
            mlp_shared_split_kernel_ms = _bench_ms(_shared_split_kernel_mlp, warmup=args.warmup, iters=args.iters)
            mlp_shared_split_kernel_graph_ms = _bench_ms(_shared_split_kernel_mlp_graph, warmup=args.warmup, iters=args.iters)
            layer_orig_ms = _bench_ms(_orig_layer, warmup=args.warmup, iters=args.iters)
            layer_attn_opt_ms = _bench_ms(_layer_attn_opt, warmup=args.warmup, iters=args.iters)
            layer_shared_split_graph_ms = _bench_ms(_layer_shared_split_graph, warmup=args.warmup, iters=args.iters)
            layer_shared_split_kernel_graph_ms = _bench_ms(_layer_shared_split_kernel_graph, warmup=args.warmup, iters=args.iters)
            layer_best_graph_ms = _bench_ms(_layer_best_graph, warmup=args.warmup, iters=args.iters)

            attn_speedup = attn_orig_ms / max(attn_opt_ms, 1e-9)
            mlp_speedup = mlp_orig_ms / max(mlp_opt_ms, 1e-9)
            shared_split_mlp_speedup = mlp_orig_ms / max(mlp_shared_split_graph_ms, 1e-9)
            shared_split_kernel_mlp_speedup = mlp_orig_ms / max(mlp_shared_split_kernel_graph_ms, 1e-9)
            full_speedup = layer_orig_ms / max(layer_best_graph_ms, 1e-9)
            attn_speedups.append(attn_speedup)
            mlp_speedups.append(mlp_speedup)
            shared_split_mlp_speedups.append(shared_split_mlp_speedup)
            shared_split_kernel_mlp_speedups.append(shared_split_kernel_mlp_speedup)
            full_speedups.append(full_speedup)

            print(
                f"L={total_len:<5} | "
                f"attn {attn_orig_ms:.4f}->{attn_opt_ms:.4f} ms ({attn_speedup:.2f}x) | "
                f"mlp {mlp_orig_ms:.4f}/{mlp_opt_ms:.4f}/{mlp_shared_split_graph_ms:.4f}/{mlp_shared_split_kernel_graph_ms:.4f} ms | "
                f"layer {layer_orig_ms:.4f} / {layer_attn_opt_ms:.4f} / {layer_shared_split_graph_ms:.4f} / {layer_best_graph_ms:.4f} ms | "
                f"full {full_speedup:.2f}x"
            )
            print(
                f"         diff: attn={_fmt_diff(diff_attn)} | "
                f"+attn={_fmt_diff(diff_layer_attn)} | "
                f"+split_g={_fmt_diff(diff_layer_shared_split)} | "
                f"best={_fmt_diff(diff_layer_best)}"
            )

    if full_speedups:
        print("---- Summary ----")
        print(
            f"Average speedup: attention={sum(attn_speedups) / len(attn_speedups):.2f}x | "
            f"flash_mlp={sum(mlp_speedups) / len(mlp_speedups):.2f}x | "
            f"shared_split_g={sum(shared_split_mlp_speedups) / len(shared_split_mlp_speedups):.2f}x | "
            f"shared_split_kernel_g={sum(shared_split_kernel_mlp_speedups) / len(shared_split_kernel_mlp_speedups):.2f}x | "
            f"full={sum(full_speedups) / len(full_speedups):.2f}x"
        )
        print(
            f"Best full speedup={max(full_speedups):.2f}x | "
            f"Worst full speedup={min(full_speedups):.2f}x"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
