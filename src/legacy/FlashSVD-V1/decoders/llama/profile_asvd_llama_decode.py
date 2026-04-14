#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_asvd_llama_decode.py — ASVD-only (Pk,Pv cache) prefill+decode profiler for LLaMA

What it profiles (per decode length)
------------------------------------
- ASVD KV size (MiB) two ways:
  (1) Measured from live (Pk,Pv) storages across *all* layers,
  (2) Allocation delta vs a model-only baseline,
  plus the theoretical ASVD KV footprint for cross-check.
- Clean separation of prefill vs decode; CUDA peaks shown for sanity.
- Timing averaged over RUNS, with per-run details.

Design choices
--------------
- SVD for Q/K/V (+ optional O/FF). ASVD cache stores only low-rank Pk, Pv per layer.
- Dense K,V reconstructed on the fly for attention (SDPA).
- **No big [B,T,V] logits**: we only compute last-token logits for argmax and delete them.
- Minimal fragmentation: fixed prompt tensors; decode uses internal causal bias (no growing attention_mask).
- Memory stats are synchronized and reset before prefill and decode.

Example:
CUDA_VISIBLE_DEVICES=0 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 PROMPT_BATCH=16 PROMPT_LEN=256 DECODE_CURVE=128,256 RUNS=3 \
RANK_Q=128 RANK_KV=128 RANK_O=0 RANK_FF=0 SVD_DTYPE=bf16 SVD_COMPUTE_FP32=1 \
python3 profile_asvd_llama_decode.py
"""

import os, time, platform, inspect, gc, statistics
from typing import Optional, Dict, Tuple, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from kernels.flash_attn_causal import flash_attn_triton_unified

# ---------------- basics ----------------
MiB = float(1024**2)

def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16): return 2
    if dtype == torch.float32: return 4
    return torch.finfo(dtype).bits // 8

def _sync_and_reset():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def _repeat_kv(x, n_rep: int):
    # x: [B, Hk, T, Dh] -> [B, H, T, Dh]
    if n_rep == 1:
        return x
    B, Hk, T, Dh = x.shape
    return x[:, :, None].expand(B, Hk, n_rep, T, Dh).reshape(B, Hk * n_rep, T, Dh)

def _build_causal_bias(q_len, k_len, device, dtype):
    # causal (with KV offset)
    i = torch.arange(q_len, device=device).view(q_len, 1)
    j = torch.arange(k_len, device=device).view(1, k_len)
    past_len = k_len - q_len
    causal = (j <= (past_len + i))
    bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    bias.masked_fill_(~causal, torch.finfo(dtype).min)
    return bias.view(1, 1, q_len, k_len)  # [1,1,Q,K]

@torch.no_grad()
def _decompose_heads_svd(weight: torch.Tensor, n_heads: int, head_dim: int, rank: int):
    # weight: [H*Dh, D_in] -> Us: [H, Dh, r] (UΣ), V: [H, r, D_in]
    W = weight.detach().to(torch.float32)
    H, dh, _ = n_heads, head_dim, W.shape[1]
    Us, Vs = [], []
    for h in range(H):
        W_h = W[h*dh:(h+1)*dh, :]
        U, S, Vh = torch.linalg.svd(W_h, full_matrices=False)
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

# ---- ASVD KV size (theoretical) ----
def _estimate_asvd_kv_cache_mib(cfg, batch_size: int, seq_len: int, factor_dtype: torch.dtype, rank_kv: int) -> float:
    L  = int(getattr(cfg, 'num_hidden_layers'))
    H  = int(getattr(cfg, 'num_attention_heads'))
    Hk = int(getattr(cfg, 'num_key_value_heads', H))
    bpe = _dtype_nbytes(factor_dtype)
    total_bytes = 2 * L * batch_size * seq_len * Hk * int(rank_kv) * bpe  # (Pk,Pv)
    return total_bytes / MiB

# ---- ASVD KV size (measured from live Pk,Pv across all layers) ----
def _measure_asvd_cache_mib(model) -> float:
    total = 0
    try:
        for lyr in getattr(model.model, "layers", []):
            blk = getattr(lyr, "block", None)
            for t in (getattr(blk, "_asvd_pk", None), getattr(blk, "_asvd_pv", None)):
                if torch.is_tensor(t) and t.is_cuda:
                    total += t.numel() * t.element_size()
    except Exception:
        pass
    return total / MiB

# choose SDPA context (new API if available; fallback otherwise)
def _sdpa_ctx_for(q_len: int, k_len: int):
    """Select SDPA backend: use Flash when Mq == Mk, otherwise mem-efficient for non-square causal."""
    use_flash = (q_len == k_len)
    try:
        from torch.nn.attention import sdpa_kernel
        if use_flash:
            return sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            return sdpa_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel
            if use_flash:
                return sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
            else:
                return sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
        except Exception:
            return nullcontext()

# ---------------- SVD LLaMA block (ASVD-only) ----------------
class SVDLlamaBlock(nn.Module):
    """
    SVD for q/k/v (+ optional o/ff), SDPA attention, **ASVD-only** (Pk,Pv) internal cache.
    Uses absolute position_ids for Q; K uses contiguous absolute positions [0..Tk-1].
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

        # internal ASVD cache (Pk,Pv)
        self._asvd_pk = None  # [B,Hk,T,r] in factor_dtype
        self._asvd_pv = None

        # norms
        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm

        # RoPE (absolute pos_ids): precompute inv_freq
        rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
        half = self.Dh // 2
        inv = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("rope_inv_freq", inv, persistent=False)

        # SVD factors for Q/K/V
        q_Us, q_V = _decompose_heads_svd(attn.q_proj.weight, self.H,  self.Dh, rank_q)
        k_Us, k_V = _decompose_heads_svd(attn.k_proj.weight, self.Hk, self.Dh, rank_kv)
        v_Us, v_V = _decompose_heads_svd(attn.v_proj.weight, self.Hk, self.Dh, rank_kv)
        self.q_Us = nn.Parameter(q_Us.to(factor_dtype), requires_grad=False)
        self.q_V  = nn.Parameter(q_V.to(factor_dtype),  requires_grad=False)
        self.k_Us = nn.Parameter(k_Us.to(factor_dtype), requires_grad=False)
        self.k_V  = nn.Parameter(k_V.to(factor_dtype),  requires_grad=False)
        self.v_Us = nn.Parameter(v_Us.to(factor_dtype), requires_grad=False)
        self.v_V  = nn.Parameter(v_V.to(factor_dtype),  requires_grad=False)

        # Out / MLP
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

        # free dense kernels kept on hf_layer to reduce retention
        for obj in (attn, mlp):
            for name in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"):
                if hasattr(obj, name):
                    try: delattr(obj, name)
                    except Exception: pass

    # ----- RoPE with absolute position_ids -----
    def _apply_rope(self, x_bhtd: torch.Tensor, position_ids: Optional[torch.Tensor]):
        # x: [B,H|Hk,T,Dh]; position_ids: [B,T] absolute, or None -> [0..T-1]
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

    def _proj_per_head_lowrank(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        # P = X·V -> [B,Hk,T,r] in factor_dtype
        if self.compute_in_fp32:
            P = torch.einsum('b t d, h r d -> b t h r', x.float(), V.float())
        else:
            P = torch.einsum('b t d, h r d -> b t h r', x.to(V.dtype), V)
        return P.transpose(1, 2).to(self.factor_dtype).contiguous()

    # ----- forward (ASVD-only) -----
    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache: bool = True, **kw):
        """
        hidden_states: [B, T, D]
        """
        B, T, _ = hidden_states.shape
        x = self.ln1(hidden_states)                           # [B,T,D]
        q = self._proj_per_head(x, self.q_Us, self.q_V)       # [B,H,T,Dh]

        # ASVD accumulation (Pk,Pv)
        if use_cache:
            Pk_new = self._proj_per_head_lowrank(x, self.k_V)   # [B,Hk,T,r]
            Pv_new = self._proj_per_head_lowrank(x, self.v_V)
            if self._asvd_pk is None:
                self._asvd_pk, self._asvd_pv = Pk_new, Pv_new
            else:
                self._asvd_pk = torch.cat([self._asvd_pk, Pk_new], dim=2)
                self._asvd_pv = torch.cat([self._asvd_pv, Pv_new], dim=2)
            Pk_seq, Pv_seq = self._asvd_pk, self._asvd_pv
        else:
            Pk_seq = self._proj_per_head_lowrank(x, self.k_V)
            Pv_seq = self._proj_per_head_lowrank(x, self.v_V)

        # reconstruct dense K,V for attention
        if self.compute_in_fp32:
            k_seq = torch.einsum('b h t r, h d r -> b h t d', Pk_seq.float(), self.k_Us.float()).to(q.dtype)
            v_seq = torch.einsum('b h t r, h d r -> b h t d', Pv_seq.float(), self.v_Us.float()).to(q.dtype)
        else:
            k_seq = torch.einsum('b h t r, h d r -> b h t d', Pk_seq.to(self.k_Us.dtype), self.k_Us).to(q.dtype)
            v_seq = torch.einsum('b h t r, h d r -> b h t d', Pv_seq.to(self.v_Us.dtype), self.v_Us).to(q.dtype)

        # RoPE: Q uses absolute provided positions; K uses absolute range [0..Tk-1]
        q = self._apply_rope(q, position_ids)
        Tk = k_seq.size(-2)
        pos_k = torch.arange(Tk, device=k_seq.device).unsqueeze(0).expand(B, Tk)
        k_seq = self._apply_rope(k_seq, pos_k)

        # repeat K,V to full heads
        k = _repeat_kv(k_seq, self.rep)        # [B,H,Tk,Dh]
        v = _repeat_kv(v_seq, self.rep)

        # attention via Triton FlashAttention (unified: full/prefix). No additive mask.
        # match dtypes
        if not (q.dtype == k.dtype == v.dtype):
            tgt = hidden_states.dtype
            q = q.to(tgt); k = k.to(tgt); v = v.to(tgt)

        # Build simple query-valid mask [B,H,1,T]
        q_len = q.size(-2)
        q_mask = torch.ones(B, self.H, 1, q_len, device=q.device, dtype=torch.bool)
        y = flash_attn_triton_unified(q, k, v, q_mask)  # [B,H,T,Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, self.D)

        # Out + MLP
        if getattr(self, "use_lr_o", False):
            y = ((y.to(self.o_V.dtype) @ self.o_V.t()) @ self.o_Us.t()).to(hidden_states.dtype)
        else:
            y = getattr(self, "o", None)(y) if hasattr(self, "o") else y

        h = hidden_states + y
        z = self.ln2(h)
        if getattr(self, "use_lr_ff", False):
            y1 = (z.to(self.g_V.dtype) @ self.g_V.t()) @ self.g_Us.t()
            y2 = (z.to(self.u_V.dtype) @ self.u_V.t()) @ self.u_Us.t()
            ff = (((F.silu(y1) * y2) @ self.d_V.t()) @ self.d_Us.t()).to(hidden_states.dtype)
        else:
            ff = getattr(self, "down")(F.silu(getattr(self, "gate")(z)) * getattr(self, "up")(z))
        out = h + ff

        # ASVD-only: return no HF cache (we keep Pk,Pv internally)
        return (out, None) if use_cache else (out,)

# ---------------- swap-in ----------------
def replace_with_svd(model, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32):
    cfg = model.config
    dev = next(model.parameters()).device
    new_layers = nn.ModuleList()
    for layer in model.model.layers:
        shim = _wrap_svd_layer(layer, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32)
        shim.to(dev)
        new_layers.append(shim)
    model.model.layers = new_layers

def _wrap_svd_layer(hf_layer, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32):
    class _Shim(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.layer_idx = getattr(getattr(inner, "self_attn", None), "layer_idx", None)
            self.block = SVDLlamaBlock(inner, cfg, rank_q, rank_kv, rank_o, rank_ff, factor_dtype, compute_in_fp32)
        def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=True, **kw):
            if "layer_idx" not in kw and self.layer_idx is not None:
                kw = dict(kw); kw["layer_idx"] = self.layer_idx
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            y, _ = self.block(hidden_states, attention_mask=attention_mask, position_ids=position_ids, use_cache=use_cache, **kw)
            # Return shape compatible tuple (no HF cache)
            return (y, None) if use_cache else (y,)
    return _Shim(hf_layer)

# ---------------- helpers ----------------
def _supports_kwarg(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False

def _reset_asvd_cache(model):
    """
    Zero out all per-layer ASVD caches (Pk, Pv) so that each run and each
    decode length starts from a clean slate. Also free their storages.
    """
    for lyr in getattr(model.model, "layers", []):
        blk = getattr(lyr, "block", None)
        if blk is None:
            continue
        for name in ("_asvd_pk", "_asvd_pv"):
            t = getattr(blk, name, None)
            if torch.is_tensor(t):
                # Drop the tensor to free its storage
                setattr(blk, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

@torch.no_grad()
def _next_from_last_hidden(model: LlamaForCausalLM, last_hidden_state: torch.Tensor, greedy: bool = True) -> torch.Tensor:
    # last-token logits only; delete immediately to avoid [B,T,V] residency
    last = last_hidden_state[:, -1, :]     # [B,D]
    logits_last = model.lm_head(last)      # [B,V]
    nxt = logits_last.argmax(dim=-1, keepdim=True) if greedy else torch.multinomial(F.softmax(logits_last.float(), dim=-1), 1)
    del logits_last
    return nxt

# ---------------- prefill + decode (single run) ----------------
@torch.no_grad()
def profile_once(model: LlamaForCausalLM, tok: AutoTokenizer, device: str,
                 prompt_len: int, decode_tokens: int, batch_size: int,
                 model_baseline_alloc_mib: float, rank_kv: int, factor_dtype: torch.dtype) -> Dict[str, float]:

    # fixed inputs (no tokenization overhead)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 2
    input_ids = torch.full((batch_size, prompt_len), eos_id, dtype=torch.long, device=device)
    pos_full  = torch.arange(prompt_len, device=device).unsqueeze(0).expand(batch_size, -1)

    model.config.use_cache = True
    supports_cache_pos = _supports_kwarg(model.model.forward, "cache_position")

    # ----- PREFILL -----
    _sync_and_reset()
    kwargs = dict(input_ids=input_ids, attention_mask=None, position_ids=pos_full,
                  use_cache=True, return_dict=True)
    if supports_cache_pos:
        kwargs["cache_position"] = pos_full[:, -1]

    t0 = time.perf_counter()
    out = model.model(**kwargs)   # Base model => no [B,T,V] logits
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000.0
    prefill_peak_mib = torch.cuda.max_memory_allocated() / MiB

    next_ids = _next_from_last_hidden(model, out.last_hidden_state, greedy=True)
    del out
    gc.collect(); torch.cuda.synchronize()

    kv_prefill_present_mib = _measure_asvd_cache_mib(model)
    mem_after_prefill_mib  = torch.cuda.memory_allocated() / MiB
    kv_prefill_alloc_mib   = max(0.0, mem_after_prefill_mib - model_baseline_alloc_mib)

    # ----- DECODE -----
    _sync_and_reset()
    cur_len = prompt_len
    t_decode = 0.0
    decode_poststep_peak = 0.0

    for _ in range(decode_tokens):
        pos_next = torch.full((batch_size, 1), cur_len, dtype=torch.long, device=device)
        step_kwargs = dict(input_ids=next_ids, attention_mask=None, position_ids=pos_next,
                           use_cache=True, return_dict=True)
        if supports_cache_pos:
            step_kwargs["cache_position"] = pos_next.squeeze(1)
        t1 = time.perf_counter()
        step = model.model(**step_kwargs)
        torch.cuda.synchronize()
        t_decode += (time.perf_counter() - t1)

        next_ids = _next_from_last_hidden(model, step.last_hidden_state, greedy=True)
        del step
        cur_len += 1

        alloc_now = torch.cuda.memory_allocated() / MiB
        if alloc_now > decode_poststep_peak:
            decode_poststep_peak = alloc_now

    decode_ms_per_tok = (t_decode * 1000.0) / max(1, decode_tokens)
    decode_peak_mib   = torch.cuda.max_memory_allocated() / MiB

    kv_decode_present_mib = _measure_asvd_cache_mib(model)
    mem_after_decode_mib  = torch.cuda.memory_allocated() / MiB
    kv_decode_alloc_mib   = max(0.0, mem_after_decode_mib - model_baseline_alloc_mib)

    del next_ids
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()

    # theoretical ASVD sizes (Pk,Pv)
    expected_prefill_mib = _estimate_asvd_kv_cache_mib(model.config, batch_size, prompt_len,            factor_dtype, rank_kv)
    expected_decode_mib  = _estimate_asvd_kv_cache_mib(model.config, batch_size, prompt_len+decode_tokens, factor_dtype, rank_kv)

    return {
        # timings
        "prefill_ms": prefill_ms,
        "decode_ms_per_tok": decode_ms_per_tok,
        # peaks
        "prefill_peak_mib": prefill_peak_mib,
        "decode_peak_mib":  decode_peak_mib,
        "decode_poststep_mib": decode_poststep_peak,
        # KV (measured + alloc deltas)
        "kv_prefill_present_mib": kv_prefill_present_mib,
        "kv_decode_present_mib":  kv_decode_present_mib,
        "kv_prefill_alloc_mib":   kv_prefill_alloc_mib,
        "kv_decode_alloc_mib":    kv_decode_alloc_mib,
        # theory
        "kv_prefill_theory_mib":  expected_prefill_mib,
        "kv_decode_theory_mib":   expected_decode_mib,
    }

# ---------------- main ----------------
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # env
    dt = os.getenv("DTYPE", "float16").lower()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dt]
    MODEL_NAME = os.getenv("LLAMA_MODEL", "meta-llama/Llama-2-7b-hf")
    PROMPT_BATCH = int(os.getenv("PROMPT_BATCH", "1"))
    PROMPT_LEN   = int(os.getenv("PROMPT_LEN", os.getenv("SEQ_LEN", "256")))
    RUNS         = int(os.getenv("RUNS", "3"))

    # decode lengths: DECODE_CURVE or MAX_GEN_TOKENS
    curve_env = os.getenv("DECODE_CURVE", "").strip()
    if curve_env:
        DECODE_LENS = [int(x) for x in curve_env.split(",") if x.strip()]
    else:
        DECODE_LENS = [int(os.getenv("MAX_GEN_TOKENS", "128"))]

    # SVD ranks + storage/compute dtype for factors
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

    # swap SVD blocks (ASVD-aware)
    replace_with_svd(model, RANK_Q, RANK_KV, RANK_O, RANK_FF, factor_dtype=SVD_DTYPE, compute_in_fp32=SVD_COMPUTE_FP32)
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # model-only baseline (after swap-in, before any caching)
    gc.collect(); torch.cuda.synchronize()
    model_baseline_alloc_mib = torch.cuda.memory_allocated() / MiB

    print("\n================== LLaMA + SVD (Decode Mode, ASVD KV) ==================")
    print(f"Python {platform.python_version()}  Torch {torch.__version__}")
    print(f"Device/dtype: {device}/{dtype}")
    print(f"Ranks: q={RANK_Q}, kv={RANK_KV}, o={RANK_O}, ff={RANK_FF}  |  ASVD store dtype={SVD_DTYPE}  |  ASVD compute fp32={SVD_COMPUTE_FP32}")
    print(f"Batch={PROMPT_BATCH}  PromptLen={PROMPT_LEN}  DecodeLens={DECODE_LENS}  RUNS={RUNS}")
    print(f"Model-only baseline (alloc): {model_baseline_alloc_mib:.1f} MiB\n")

    for new_tokens in DECODE_LENS:
        _reset_asvd_cache(model)
        
        results: List[Dict[str, float]] = []
        for _ in range(max(1, RUNS)):
            _reset_asvd_cache(model)

            res = profile_once(
                model, tok, device,
                prompt_len=PROMPT_LEN, decode_tokens=new_tokens,
                batch_size=PROMPT_BATCH,
                model_baseline_alloc_mib=model_baseline_alloc_mib,
                rank_kv=RANK_KV, factor_dtype=SVD_DTYPE
            )
            results.append(res)


        def mean_std(xs: List[float]) -> Tuple[float, float]:
            return (statistics.fmean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)

        prefill_ms_mean, prefill_ms_std = mean_std([r["prefill_ms"] for r in results])
        decode_ms_mean,  decode_ms_std  = mean_std([r["decode_ms_per_tok"] for r in results])
        last = results[-1]

        print(f"---- Memory profile (end-to-end) — decode_len={new_tokens} ----")
        print(f"{'':<24} | {'Measured (present)':>21} | {'Alloc Δ vs model':>18} | {'Theoretical':>12}")
        print("-"*86)
        print(f"{'Prefill KV (MiB)':<24} | {last['kv_prefill_present_mib']:>21.1f} | {last['kv_prefill_alloc_mib']:>18.1f} | {last['kv_prefill_theory_mib']:>12.1f}")
        print(f"{'Decode  KV (MiB)':<24} | {last['kv_decode_present_mib']:>21.1f} | {last['kv_decode_alloc_mib']:>18.1f} | {last['kv_decode_theory_mib']:>12.1f}")
        print(f"{'(peaks, sanity)':<24} | {'prefill_peak':>21} | {'decode_peak':>18} |")
        print(f"{'':<24} | {last['prefill_peak_mib']:>21.1f} | {last['decode_peak_mib']:>18.1f} |")
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


'''
CUDA_VISIBLE_DEVICES=0 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 PROMPT_BATCH=16 PROMPT_LEN=256 DECODE_CURVE=128,256 RUNS=3 \
RANK_Q=64 RANK_KV=64 RANK_O=2048 RANK_FF=2048 SVD_DTYPE=bf16 SVD_COMPUTE_FP32=1 \
python3 profile_asvd_llama_decode.py

'''
