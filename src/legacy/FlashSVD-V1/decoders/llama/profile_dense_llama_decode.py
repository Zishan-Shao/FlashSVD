#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_dense_llama_decode.py — SVD-weights + Dense KV (HF cache) prefill+decode profiler for LLaMA

This is literally the profile_svd_kv_llama_decode.py, which in this case we use this one prefered

What it profiles (per decode length)
------------------------------------
- KV cache size (MiB), two ways:
  (1) Measured from the HF cache object ("present") by summing GPU storages across *all* layers
  (2) Allocation delta vs model-only baseline (end-to-end)
  And the theoretical KV size for cross-check.
- Prefill & decode timing (averaged over RUNS), plus per-run details.
- Prefill and decode CUDA peak memory (sanity).

Design choices
--------------
- Uses HF DynamicCache (new HF requires Cache or None at top level); we never pass a tuple to LlamaModel.
- SVD for Q/K/V (+ optional O/FF). KV is **dense**; GQA repetition expands K/V to all heads for attention.
- **No giant [B,T,V] logits**: we call `model.model(...)` and only compute *last-token* logits -> argmax, then delete.
- Absolute RoPE via `position_ids` for both prefill and decode.
- Minimal fragmentation: fixed prompt tensors, decode uses no growing attention_mask (we build an additive causal bias).
- Memory stats reset (sync + empty + reset peaks) before prefill and decode for clean measurements.

Environment knobs
-----------------
LLAMA_MODEL=meta-llama/Llama-2-7b-hf
DTYPE=float16|bfloat16|float32
PROMPT_BATCH=16
PROMPT_LEN=256              # or SEQ_LEN (fallback)
DECODE_CURVE="128,256"      # one or many decode lengths; if unset, uses MAX_GEN_TOKENS
RUNS=3

# SVD ranks (per-head for Q/K/V; full-matrix for O/FF; 0 disables O/FF low-rank)
RANK_Q=128 RANK_KV=128 RANK_O=0 RANK_FF=0

Example:
CUDA_VISIBLE_DEVICES=0 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 PROMPT_BATCH=16 PROMPT_LEN=256 DECODE_CURVE=128,256 RUNS=3 \
RANK_Q=64 RANK_KV=64 RANK_O=2048 RANK_FF=2048 \
python3 profile_dense_llama_decode.py


CUDA_VISIBLE_DEVICES=0 LLAMA_MODEL=meta-llama/Llama-2-7b-hf DTYPE=float16 PROMPT_BATCH=16 PROMPT_LEN=256 DECODE_CURVE=128,256 RUNS=1 MEM_TRACE=1 RANK_Q=64 RANK_KV=64 RANK_O=2048 RANK_FF=2048 SVD_DTYPE=bf16 SVD_COMPUTE_FP32=1 python3 profile_dense_llama_decode.py


"""

import os, time, platform, inspect, gc, statistics
from typing import Optional, Dict, Tuple, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from kernels.flash_attn_causal import flash_attn_triton_unified
from transformers import AutoTokenizer, LlamaForCausalLM

MiB = float(1024**2)

# -------------------- small utils --------------------
def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16): return 2
    if dtype == torch.float32: return 4
    return torch.finfo(dtype).bits // 8

def _expected_kv_mib(cfg, batch_size: int, seq_len: int, kv_dtype: torch.dtype) -> float:
    L  = int(getattr(cfg, 'num_hidden_layers'))
    H  = int(getattr(cfg, 'num_attention_heads'))
    Hk = int(getattr(cfg, 'num_key_value_heads', H))
    Dh = int(getattr(cfg, 'hidden_size')) // H
    bpe = _dtype_nbytes(kv_dtype)
    total_bytes = 2 * L * batch_size * seq_len * Hk * Dh * bpe  # *2 for (K,V)
    return total_bytes / MiB

def _sync_and_reset():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def _mem_stats_mib():
    if not torch.cuda.is_available():
        return dict(alloc=0.0, reserved=0.0, max_alloc=0.0, max_reserved=0.0)
    return dict(
        alloc=torch.cuda.memory_allocated() / MiB,
        reserved=torch.cuda.memory_reserved() / MiB,
        max_alloc=torch.cuda.max_memory_allocated() / MiB,
        max_reserved=torch.cuda.max_memory_reserved() / MiB,
    )

# -------------------- attention helpers --------------------
def _repeat_kv(x, n_rep: int):
    # x: [B, Hk, T, Dh] -> [B, H, T, Dh]
    if n_rep == 1:
        return x
    B, Hk, T, Dh = x.shape
    return x[:, :, None].expand(B, Hk, n_rep, T, Dh).reshape(B, Hk * n_rep, T, Dh)

def _ensure_bhtd(x: torch.Tensor, Hk: int):
    assert x.dim() == 4, f"expected 4D, got {tuple(x.shape)}"
    if x.size(1) == Hk:
        return x
    if x.size(2) == Hk:
        return x.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"Unrecognized KV layout {tuple(x.shape)} for Hk={Hk}")

def _build_causal_bias(q_len, k_len, device, dtype):
    # causal with KV offset
    i = torch.arange(q_len, device=device).view(q_len, 1)
    j = torch.arange(k_len, device=device).view(1, k_len)
    past_len = k_len - q_len
    causal = (j <= (past_len + i))
    bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    bias.masked_fill_(~causal, torch.finfo(dtype).min)
    return bias.view(1, 1, q_len, k_len)  # [1,1,Q,K]

# -------------------- SVD helpers --------------------
@torch.no_grad()
def _decompose_heads_svd(weight: torch.Tensor, n_heads: int, head_dim: int, rank: int):
    # weight: [H*Dh, D_in] -> Us: [H, Dh, r], V: [H, r, D_in]
    W = weight.detach().to(torch.float32)
    H, dh, _ = n_heads, head_dim, W.shape[1]
    Us, Vs = [], []
    for h in range(H):
        Wh = W[h*dh:(h+1)*dh, :]
        U, S, Vh = torch.linalg.svd(Wh, full_matrices=False)
        r = max(1, min(int(rank), U.shape[1], Vh.shape[0]))
        Us.append(U[:, :r] * S[:r].unsqueeze(0))
        Vs.append(Vh[:r, :])
    return torch.stack(Us, 0), torch.stack(Vs, 0)

@torch.no_grad()
def _decompose_full_svd(weight: torch.Tensor, rank: int):
    W = weight.detach().to(torch.float32)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = max(1, min(int(rank), U.shape[1], Vh.shape[0]))
    return U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :]

# -------------------- robust KV measurement --------------------
def _bytes_of_present(pkv) -> float:
    """
    Sum unique GPU storages for K and V across *all* layers (DynamicCache or legacy tuple/list).
    """
    if pkv is None:
        return 0.0

    # 1) Try explicit HF Cache layouts first (fast & exact)
    try:
        key_cache  = getattr(pkv, "key_cache",  None)
        value_cache= getattr(pkv, "value_cache",None)
        if isinstance(key_cache, (list, tuple)) and isinstance(value_cache, (list, tuple)):
            total = 0
            for t in key_cache:
                if torch.is_tensor(t) and t.is_cuda:
                    total += t.numel() * t.element_size()
            for t in value_cache:
                if torch.is_tensor(t) and t.is_cuda:
                    total += t.numel() * t.element_size()
            if total > 0:
                return total / MiB

        layers = getattr(pkv, "layers", None)
        if isinstance(layers, (list, tuple)):
            total = 0
            for lyr in layers:
                for name in ("keys","values"):
                    t = getattr(lyr, name, None)
                    if torch.is_tensor(t) and t.is_cuda:
                        total += t.numel() * t.element_size()
            if total > 0:
                return total / MiB
    except Exception:
        pass

    # 2) Fallback: generic traversal with storage dedup
    seen, total = set(), 0
    def add_tensor(x: torch.Tensor):
        nonlocal total
        if not x.is_cuda: return
        try:
            s = x.untyped_storage(); key = (s.data_ptr(), int(s.nbytes()))
        except Exception:
            s = x.storage()
            ptr = s.data_ptr() if hasattr(s, "data_ptr") else x.data_ptr()
            nbytes = s.nbytes() if hasattr(s, "nbytes") else s.size() * x.element_size()
            key = (ptr, int(nbytes))
        if key not in seen:
            seen.add(key); total += key[1]

    def rec(x):
        if torch.is_tensor(x): add_tensor(x); return
        if isinstance(x, (list, tuple, set)):
            for y in x: rec(y); return
        if isinstance(x, dict):
            for y in x.values(): rec(y); return
        for name in ("key_cache","value_cache","layers","keys","values","k_cache","v_cache"):
            if hasattr(x, name):
                try: rec(getattr(x, name))
                except Exception: pass

    rec(pkv)
    return total / MiB

# -------------------- SDPA ctx --------------------
def _sdpa_ctx():
    try:
        from torch.nn.attention import sdpa_kernel
        return sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel
            return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        except Exception:
            return nullcontext()

# -------------------- SVD LLaMA block (Dense KV) --------------------
class SVDLlamaBlock(nn.Module):
    """
    SVD for q/k/v (+ optional o/ff), SDPA attention, dense KV via HF cache.
    Uses absolute position_ids for RoPE (prefill + decode).
    """
    def __init__(self, hf_layer: nn.Module, cfg,
                 rank_q: int, rank_kv: int, rank_o: Optional[int], rank_ff: Optional[int],
                 factor_dtype: torch.dtype = torch.float32, compute_in_fp32: bool = True):
        super().__init__()
        attn, mlp = hf_layer.self_attn, hf_layer.mlp
        self.D  = int(cfg.hidden_size)
        self.H  = int(cfg.num_attention_heads)
        self.Hk = int(getattr(cfg, "num_key_value_heads", self.H))
        self.Dh = self.D // self.H
        self.rep = self.H // self.Hk

        self.compute_in_fp32 = bool(compute_in_fp32)
        self.factor_dtype = factor_dtype
        self.layer_idx = getattr(getattr(hf_layer, "self_attn", None), "layer_idx", None)

        # norms
        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm

        # RoPE (absolute pos_ids): precompute inv_freq
        rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
        half = self.Dh // 2
        inv = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("rope_inv_freq", inv, persistent=False)

        # SVD factors
        q_Us, q_V = _decompose_heads_svd(attn.q_proj.weight, self.H,  self.Dh, rank_q)
        k_Us, k_V = _decompose_heads_svd(attn.k_proj.weight, self.Hk, self.Dh, rank_kv)
        v_Us, v_V = _decompose_heads_svd(attn.v_proj.weight, self.Hk, self.Dh, rank_kv)
        self.q_Us = nn.Parameter(q_Us.to(factor_dtype), requires_grad=False)
        self.q_V  = nn.Parameter(q_V.to(factor_dtype),  requires_grad=False)
        self.k_Us = nn.Parameter(k_Us.to(factor_dtype), requires_grad=False)
        self.k_V  = nn.Parameter(k_V.to(factor_dtype),  requires_grad=False)
        self.v_Us = nn.Parameter(v_Us.to(factor_dtype), requires_grad=False)
        self.v_V  = nn.Parameter(v_V.to(factor_dtype),  requires_grad=False)

        # Output / MLP (optional low-rank)
        if rank_o is not None and rank_o > 0:
            o_Us, o_V = _decompose_full_svd(attn.o_proj.weight, rank_o)
            self.o_Us = nn.Parameter(o_Us.to(factor_dtype), requires_grad=False)
            self.o_V  = nn.Parameter(o_V.to(factor_dtype),  requires_grad=False)
            self.use_lr_o = True
        else:
            self.o = nn.Linear(self.D, self.D, bias=False, dtype=attn.o_proj.weight.dtype)
            with torch.no_grad(): self.o.weight.copy_(attn.o_proj.weight)
            self.use_lr_o = False

        inter = int(cfg.intermediate_size)
        if rank_ff is not None and rank_ff > 0:
            g_Us, g_V = _decompose_full_svd(mlp.gate_proj.weight, rank_ff)
            u_Us, u_V = _decompose_full_svd(mlp.up_proj.weight,   rank_ff)
            d_Us, d_V = _decompose_full_svd(mlp.down_proj.weight, rank_ff)
            for name, val in dict(g_Us=g_Us, g_V=g_V, u_Us=u_Us, u_V=u_V, d_Us=d_Us, d_V=d_V).items():
                setattr(self, name, nn.Parameter(val.to(factor_dtype), requires_grad=False))
            self.use_lr_ff = True
        else:
            self.gate = nn.Linear(self.D, inter, bias=False, dtype=mlp.gate_proj.weight.dtype)
            self.up   = nn.Linear(self.D, inter, bias=False, dtype=mlp.up_proj.weight.dtype)
            self.down = nn.Linear(inter, self.D, bias=False, dtype=mlp.down_proj.weight.dtype)
            with torch.no_grad():
                self.gate.weight.copy_(mlp.gate_proj.weight)
                self.up.weight.copy_(mlp.up_proj.weight)
                self.down.weight.copy_(mlp.down_proj.weight)
            self.use_lr_ff = False

        # free dense modules kept on hf_layer to reduce retention
        for obj in (attn, mlp):
            for name in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"):
                if hasattr(obj, name):
                    try: delattr(obj, name)
                    except Exception: pass

    # ----- RoPE with absolute position_ids -----
    def _apply_rope(self, x_bhtd: torch.Tensor, position_ids: Optional[torch.Tensor]):
        # x: [B,H|Hk,T,Dh]; position_ids: [B,T] absolute
        B, _, T, Dh = x_bhtd.shape
        half = Dh // 2
        if position_ids is None:
            pos = torch.arange(T, device=x_bhtd.device).unsqueeze(0).expand(B, T)
        else:
            pos = torch.clamp(position_ids.to(torch.long), min=0)
        ang = (pos.to(torch.float32)[..., None] *
               self.rope_inv_freq[None, :].to(x_bhtd.device))  # [B,T,half]
        cos = ang.cos().to(x_bhtd.dtype).unsqueeze(1)  # [B,1,T,half]
        sin = ang.sin().to(x_bhtd.dtype).unsqueeze(1)  # [B,1,T,half]
        x1, x2 = x_bhtd[..., :half], x_bhtd[..., half:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out.contiguous()

    # ----- projections -----
    def _proj_per_head(self, x: torch.Tensor, Us: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D], V: [H|Hk, r, D], Us: [H|Hk, Dh, r] -> [B, H|Hk, T, Dh]
        if self.compute_in_fp32:
            xr  = torch.einsum('b t d, h r d -> b t h r', x.float(), V.float())
            out = torch.einsum('b t h r, h d r -> b t h d', xr, Us.float())
            return out.to(x.dtype).transpose(1, 2).contiguous()
        xr  = torch.einsum('b t d, h r d -> b t h r', x.to(V.dtype), V)
        out = torch.einsum('b t h r, h d r -> b t h d', xr, Us)
        return out.to(x.dtype).transpose(1, 2).contiguous()

    # ----- robust cache readback (when update returns None) -----
    @staticmethod
    def _read_layer_kv_from_cache(cache_obj, layer_idx: int):
        k = v = None
        if hasattr(cache_obj, "layers") and 0 <= layer_idx < len(cache_obj.layers):
            lyr = cache_obj.layers[layer_idx]
            k = getattr(lyr, "keys", None); v = getattr(lyr, "values", None)
        if (k is None or v is None) and hasattr(cache_obj, "key_cache"):
            try:
                k = cache_obj.key_cache[layer_idx]; v = cache_obj.value_cache[layer_idx]
            except Exception:
                pass
        return k, v

    def _detect_cache_layout(self, past_key_value, layer_idx) -> str:
        try:
            k_prev, _ = self._read_layer_kv_from_cache(past_key_value, layer_idx)
            if isinstance(k_prev, torch.Tensor) and k_prev.dim() == 4:
                if k_prev.size(1) == self.Hk: return "bhtd"
                if k_prev.size(2) == self.Hk: return "bthd"
        except Exception:
            pass
        return "bhtd"

    def _try_update(self, pkv, k_upd, v_upd, layer_idx, cache_position):
        attempts = []
        if cache_position is not None:
            attempts.append(("kw_cachepos", lambda: pkv.update(k_upd, v_upd, layer_idx=layer_idx, cache_position=cache_position)))
        attempts.append(("kw",  lambda: pkv.update(k_upd, v_upd, layer_idx=layer_idx)))
        attempts.append(("pos", lambda: pkv.update(k_upd, v_upd, layer_idx)))
        last_exc = None
        for _, fn in attempts:
            try:
                ret = fn()
                if isinstance(ret, tuple) and len(ret) == 2: return ret
                k_seq, v_seq = self._read_layer_kv_from_cache(pkv, layer_idx)
                if k_seq is not None and v_seq is not None: return k_seq, v_seq
            except Exception as e:
                last_exc = e; continue
        raise RuntimeError(f"DynamicCache.update failed; last error: {last_exc}")

    def _cache_update(self, past_key_value, k_bhtd, v_bhtd, layer_idx, cache_position):
        layout = self._detect_cache_layout(past_key_value, layer_idx)
        if layout == "bhtd":
            cand = [(k_bhtd, v_bhtd), (k_bhtd.transpose(1, 2), v_bhtd.transpose(1, 2))]
        else:
            cand = [(k_bhtd.transpose(1, 2), v_bhtd.transpose(1, 2)), (k_bhtd, v_bhtd)]
        last_exc = None
        for (k_upd, v_upd) in cand:
            try:
                k_seq, v_seq = self._try_update(past_key_value, k_upd, v_upd, layer_idx, cache_position)
                k_seq = _ensure_bhtd(k_seq, self.Hk); v_seq = _ensure_bhtd(v_seq, self.Hk)
                return k_seq, v_seq
            except RuntimeError as e:
                last_exc = e; continue
        raise RuntimeError(f"Failed to update K/V after trying both layouts. Last error: {last_exc}")

    # ----- forward -----
    def forward(self, hidden_states, attention_mask=None, past_key_value=None,
                position_ids=None, use_cache: bool = False, **kw):
        B, T, _ = hidden_states.shape
        x = self.ln1(hidden_states)
        if os.getenv("MEM_TRACE", "0") == "1" and torch.cuda.is_available():
            ms = _mem_stats_mib(); li = getattr(self, "layer_idx", None)
            print(f"[MEM] L{li} after ln1: alloc={ms['alloc']:.1f}MiB max={ms['max_alloc']:.1f}MiB")

        T_max = T
        x_trim = x[:, :T_max, :]
        pos_ids = position_ids[:, :T_max] if position_ids is not None else None

        # Projections
        q = self._proj_per_head(x_trim, self.q_Us, self.q_V)     # [B,H,T,Dh]
        k_new = self._proj_per_head(x_trim, self.k_Us, self.k_V) # [B,Hk,T,Dh]
        v_new = self._proj_per_head(x_trim, self.v_Us, self.v_V)
        if os.getenv("MEM_TRACE", "0") == "1" and torch.cuda.is_available():
            ms = _mem_stats_mib(); li = getattr(self, "layer_idx", None)
            print(f"[MEM] L{li} after q/k/v proj: alloc={ms['alloc']:.1f}MiB max={ms['max_alloc']:.1f}MiB")

        # RoPE (absolute)
        q     = self._apply_rope(q,     pos_ids)
        k_rot = self._apply_rope(k_new, pos_ids)

        # Cache integration
        present_out = None
        cache_position = kw.get("cache_position", None)

        if hasattr(past_key_value, "update"):  # DynamicCache
            li = self.layer_idx if self.layer_idx is not None else kw.get("layer_idx", None)
            k_seq, v_seq = self._cache_update(past_key_value, k_rot, v_new, li, cache_position)
            present_out = past_key_value
        else:  # Legacy tuple/list fallback (if ever passed in)
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                k_seq = torch.cat([_ensure_bhtd(past_key_value[0], self.Hk), k_rot], dim=2)
                v_seq = torch.cat([_ensure_bhtd(past_key_value[1], self.Hk), v_new], dim=2)
            else:
                k_seq, v_seq = k_rot, v_new
            present_out = (k_seq, v_seq)
        if os.getenv("MEM_TRACE", "0") == "1" and torch.cuda.is_available():
            ms = _mem_stats_mib(); li = getattr(self, "layer_idx", None)
            print(f"[MEM] L{li} after cache update: alloc={ms['alloc']:.1f}MiB max={ms['max_alloc']:.1f}MiB")

        # Attention via Triton FlashAttention (causal, with or without KV-cache)
        k = _repeat_kv(k_seq, self.rep)   # [B,H,Tk,Dh]
        v = _repeat_kv(v_seq, self.rep)
        q_len = q.size(-2)

        # match dtypes
        if not (q.dtype == k.dtype == v.dtype):
            target = hidden_states.dtype
            q = q.to(target); k = k.to(target); v = v.to(target)

        # simple query-valid mask [B,H,1,Q]; causal handled inside the kernel
        q_mask = torch.ones(B, self.H, 1, q_len, device=q.device, dtype=torch.bool)
        y = flash_attn_triton_unified(q, k, v, q_mask)  # [B,H,Q,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T_max, self.D)
        if os.getenv("MEM_TRACE", "0") == "1" and torch.cuda.is_available():
            ms = _mem_stats_mib(); li = getattr(self, "layer_idx", None)
            print(f"[MEM] L{li} after attn: alloc={ms['alloc']:.1f}MiB max={ms['max_alloc']:.1f}MiB")

        # Out + MLP
        if getattr(self, "use_lr_o", False):
            y = ((y.to(self.o_V.dtype) @ self.o_V.t()) @ self.o_Us.t()).to(hidden_states.dtype)
        else:
            y = getattr(self, "o", None)(y) if hasattr(self, "o") else y

        h = hidden_states[:, :T_max, :] + y
        z = self.ln2(h)
        if getattr(self, "use_lr_ff", False):
            y1 = (z.to(self.g_V.dtype) @ self.g_V.t()) @ self.g_Us.t()
            y2 = (z.to(self.u_V.dtype) @ self.u_V.t()) @ self.u_Us.t()
            ff = (((F.silu(y1) * y2) @ self.d_V.t()) @ self.d_Us.t()).to(hidden_states.dtype)
        else:
            ff = getattr(self, "down")(F.silu(getattr(self, "gate")(z)) * getattr(self, "up")(z))
        out = h + ff
        if os.getenv("MEM_TRACE", "0") == "1" and torch.cuda.is_available():
            ms = _mem_stats_mib(); li = getattr(self, "layer_idx", None)
            print(f"[MEM] L{li} end layer: alloc={ms['alloc']:.1f}MiB max={ms['max_alloc']:.1f}MiB")

        return (out, present_out) if use_cache else (out,)

# -------------------- swap-in --------------------
def replace_with_svd(model, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32):
    cfg = model.config
    dev = next(model.parameters()).device
    new_layers = nn.ModuleList()
    for layer in model.model.layers:
        shim = _wrap_svd_layer(layer, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32)
        shim.to(dev); new_layers.append(shim)
    model.model.layers = new_layers

def _wrap_svd_layer(hf_layer, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32):
    class _Shim(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.layer_idx = getattr(getattr(inner, "self_attn", None), "layer_idx", None)
            self.block = SVDLlamaBlock(inner, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32)
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, use_cache=True, **kw):
            if "layer_idx" not in kw and self.layer_idx is not None:
                kw = dict(kw); kw["layer_idx"] = self.layer_idx
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            return self.block(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                              past_key_value=past_key_value, use_cache=use_cache, **kw)
    return _Shim(hf_layer)

# -------------------- helpers --------------------
def _supports_kwarg(fn, name: str) -> bool:
    try: return name in inspect.signature(fn).parameters
    except Exception: return False

@torch.no_grad()
def _next_from_last_hidden(model: LlamaForCausalLM, last_hidden_state: torch.Tensor, greedy: bool = True) -> torch.Tensor:
    # compute logits only for the last token and drop immediately
    last = last_hidden_state[:, -1, :]  # [B,D]
    logits_last = model.lm_head(last)   # [B,V]
    nxt = logits_last.argmax(dim=-1, keepdim=True) if greedy else torch.multinomial(F.softmax(logits_last.float(), -1), 1)
    del logits_last
    return nxt

# -------------------- prefill + decode (single run) --------------------
@torch.no_grad()
def profile_once(model: LlamaForCausalLM, tok: AutoTokenizer, device: str,
                 prompt_len: int, decode_tokens: int, batch_size: int,
                 model_baseline_alloc_mib: float) -> Dict[str, float]:
    # fixed inputs (no tokenization overhead)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 2
    input_ids = torch.full((batch_size, prompt_len), eos_id, dtype=torch.long, device=device)
    pos_full  = torch.arange(prompt_len, device=device).unsqueeze(0).expand(batch_size, -1)

    model.config.use_cache = True
    supports_cache_pos = _supports_kwarg(model.model.forward, "cache_position")

    # -------- PREFILL --------
    _sync_and_reset()
    kwargs = dict(input_ids=input_ids, attention_mask=None, position_ids=pos_full,
                  use_cache=True, return_dict=True)
    if supports_cache_pos:
        kwargs["cache_position"] = pos_full[:, -1]

    t0 = time.perf_counter()
    out = model.model(**kwargs)  # base model -> no full [B,T,V] logits
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    prefill_peak_mib = torch.cuda.max_memory_allocated() / MiB
    prefill_peak_reserved_mib = torch.cuda.max_memory_reserved() / MiB

    present = out.past_key_values  # DynamicCache
    next_ids = _next_from_last_hidden(model, out.last_hidden_state, greedy=True)

    # keep only KV; drop activations
    del out
    gc.collect(); torch.cuda.synchronize()

    kv_prefill_present_mib = _bytes_of_present(present)
    mem_after_prefill_mib = torch.cuda.memory_allocated() / MiB
    kv_prefill_alloc_mib = max(0.0, mem_after_prefill_mib - model_baseline_alloc_mib)
    prefill_transient_overhead_mib = max(0.0, prefill_peak_mib - (model_baseline_alloc_mib + kv_prefill_alloc_mib))

    # -------- DECODE --------
    _sync_and_reset()
    cur_len = prompt_len
    t_decode = 0.0
    for _ in range(decode_tokens):
        pos_next = torch.full((batch_size, 1), cur_len, dtype=torch.long, device=device)
        step_kwargs = dict(input_ids=next_ids, attention_mask=None, position_ids=pos_next,
                           past_key_values=present, use_cache=True, return_dict=True)
        if supports_cache_pos:
            step_kwargs["cache_position"] = pos_next.squeeze(1)
        t1 = time.perf_counter()
        step = model.model(**step_kwargs)
        torch.cuda.synchronize()
        t_decode += (time.perf_counter() - t1)

        next_ids = _next_from_last_hidden(model, step.last_hidden_state, greedy=True)
        present  = step.past_key_values
        del step
        cur_len += 1

    decode_ms_per_tok = (t_decode * 1000.0) / max(1, decode_tokens)
    decode_peak_mib = torch.cuda.max_memory_allocated() / MiB
    decode_peak_reserved_mib = torch.cuda.max_memory_reserved() / MiB

    kv_decode_present_mib = _bytes_of_present(present)
    mem_after_decode_mib = torch.cuda.memory_allocated() / MiB
    kv_decode_alloc_mib = max(0.0, mem_after_decode_mib - model_baseline_alloc_mib)
    decode_transient_overhead_mib = max(0.0, decode_peak_mib - (model_baseline_alloc_mib + kv_decode_alloc_mib))

    del present, next_ids
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()

    return {
        "prefill_ms": prefill_ms,
        "prefill_peak_mib": prefill_peak_mib,
        "prefill_peak_reserved_mib": prefill_peak_reserved_mib,
        "prefill_transient_overhead_mib": prefill_transient_overhead_mib,
        "kv_prefill_present_mib": kv_prefill_present_mib,
        "kv_prefill_alloc_mib": kv_prefill_alloc_mib,
        "decode_ms_per_tok": decode_ms_per_tok,
        "decode_peak_mib": decode_peak_mib,
        "decode_peak_reserved_mib": decode_peak_reserved_mib,
        "decode_transient_overhead_mib": decode_transient_overhead_mib,
        "kv_decode_present_mib": kv_decode_present_mib,
        "kv_decode_alloc_mib": kv_decode_alloc_mib,
    }

# -------------------- main --------------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # env
    dt = os.getenv("DTYPE", "float16").lower()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]
    MODEL_NAME = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-hf")
    PROMPT_BATCH = int(os.getenv("PROMPT_BATCH", "1"))
    PROMPT_LEN = int(os.getenv("PROMPT_LEN", os.getenv("SEQ_LEN", "256")))
    RUNS = int(os.getenv("RUNS", "3"))

    # decode lengths: DECODE_CURVE or MAX_GEN_TOKENS
    curve_env = os.getenv("DECODE_CURVE", "").strip()
    if curve_env:
        DECODE_LENS = [int(x) for x in curve_env.split(",") if x.strip()]
    else:
        DECODE_LENS = [int(os.getenv("MAX_GEN_TOKENS", "128"))]

    # SVD ranks
    RANK_Q  = int(os.getenv("RANK_Q",  "128"))
    RANK_KV = int(os.getenv("RANK_KV", "128"))
    RANK_O  = int(os.getenv("RANK_O",  "0")) or None
    RANK_FF = int(os.getenv("RANK_FF", "0")) or None
    SVD_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[os.getenv("SVD_DTYPE", "fp32").lower()]
    SVD_COMPUTE_FP32 = os.getenv("SVD_COMPUTE_FP32", "1") == "1"

    # model
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    model.config.use_cache = True
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # swap SVD blocks
    replace_with_svd(model, RANK_Q, RANK_KV, RANK_O, RANK_FF, factor_dtype=SVD_DTYPE, compute_in_fp32=SVD_COMPUTE_FP32)
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # model-only baseline
    gc.collect(); torch.cuda.synchronize()
    model_baseline_alloc_mib = torch.cuda.memory_allocated() / MiB

    print("\n================== LLaMA + SVD (Decode Mode, Dense KV) ==================")
    print(f"Python {platform.python_version()}  Torch {torch.__version__}")
    print(f"Device/dtype: {device}/{dtype}")
    print(f"Ranks: q={RANK_Q}, kv={RANK_KV}, o={RANK_O}, ff={RANK_FF}  |  SVD store dtype={SVD_DTYPE}  |  SVD compute fp32={SVD_COMPUTE_FP32}")
    print(f"Batch={PROMPT_BATCH}  PromptLen={PROMPT_LEN}  DecodeLens={DECODE_LENS}  RUNS={RUNS}")
    print(f"Model-only baseline (alloc): {model_baseline_alloc_mib:.1f} MiB\n")

    # For each decode length, run RUNS and report memory/time + theory
    for new_tokens in DECODE_LENS:
        results: List[Dict[str, float]] = []
        for _ in range(max(1, RUNS)):
            res = profile_once(model, tok, device,
                               prompt_len=PROMPT_LEN, decode_tokens=new_tokens,
                               batch_size=PROMPT_BATCH,
                               model_baseline_alloc_mib=model_baseline_alloc_mib)
            results.append(res)

        def mean_std(xs: List[float]) -> Tuple[float, float]:
            return (statistics.fmean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)

        prefill_ms_mean, prefill_ms_std = mean_std([r["prefill_ms"] for r in results])
        decode_ms_mean, decode_ms_std   = mean_std([r["decode_ms_per_tok"] for r in results])
        last = results[-1]

        # theoretical KV sizes
        kv_dtype = dtype
        expected_prefill_mib = _expected_kv_mib(model.config, PROMPT_BATCH, PROMPT_LEN, kv_dtype)
        expected_decode_mib  = _expected_kv_mib(model.config, PROMPT_BATCH, PROMPT_LEN + new_tokens, kv_dtype)

        print(f"---- Memory profile (end-to-end) — decode_len={new_tokens} ----")
        print(f"{'':<24} | {'Measured (present)':>21} | {'Alloc Δ vs model':>18} | {'Theoretical':>12}")
        print("-"*86)
        print(f"{'Prefill KV (MiB)':<24} | {last['kv_prefill_present_mib']:>21.1f} | {last['kv_prefill_alloc_mib']:>18.1f} | {expected_prefill_mib:>12.1f}")
        print(f"{'Decode  KV (MiB)':<24} | {last['kv_decode_present_mib']:>21.1f} | {last['kv_decode_alloc_mib']:>18.1f} | {expected_decode_mib:>12.1f}")
        print(f"{'(peaks, sanity)':<24} | {'prefill_peak':>21} | {'decode_peak':>18} |")
        print(f"{'':<24} | {last['prefill_peak_mib']:>21.1f} | {last['decode_peak_mib']:>18.1f} |")
        print(f"{'(reserved peaks)':<24} | {last['prefill_peak_reserved_mib']:>21.1f} | {last['decode_peak_reserved_mib']:>18.1f} |")
        print(f"{'(transient overhead)':<24} | {last['prefill_transient_overhead_mib']:>21.1f} | {last['decode_transient_overhead_mib']:>18.1f} |")
        print()

        print(f"---- Timing profile (averaged over RUNS) — decode_len={new_tokens} ----")
        print(f"{'Prefill (ms)':<18} {prefill_ms_mean:>10.1f}  ± {prefill_ms_std:<6.1f}")
        print(f"{'Decode (ms/tok)':<18} {decode_ms_mean:>10.2f}  ± {decode_ms_std:<6.2f}")
        print("\nPer-run details:")
        print(f"{'Run':<4} | {'Prefill ms':>10} | {'Decode ms/tok':>13} | {'KV pre (MiB)':>12} | {'KV dec (MiB)':>12}")
        print("-"*66)
        for i, r in enumerate(results, 1):
            print(f"{i:<4} | {r['prefill_ms']:>10.1f} | {r['decode_ms_per_tok']:>13.2f} | {r['kv_prefill_present_mib']:>12.1f} | {r['kv_decode_present_mib']:>12.1f}")
        print()
