#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_asvd_flashsvd_llama_decode.py — FlashSVD (RoPE) + ASVD cache profiler for LLaMA

What it measures per decode length
----------------------------------
- ASVD KV cache size (MiB) via live tensors vs allocator delta, plus theoretical estimate.
- Prefill vs decode timing (averaged across RUNS) and CUDA peaks for sanity.
- Per-run breakdown of timings and KV footprint.

Design notes
------------
- Q/K/V (and optional O/FF) are factorised once via SVD; low-rank Pk/Pv are cached per layer.
- FlashSVD RoPE attention operates directly in rank space; decode replays the full prefix on each step.
- No `[B,T,V]` logits: next tokens come from last hidden state only, immediately freed.
- CUDA allocator is reset around prefill/decode segments to keep stats clean.

Example:
CUDA_VISIBLE_DEVICES=0 \
LLAMA_MODEL=meta-llama/Llama-2-7b-hf \
DTYPE=float16 PROMPT_BATCH=16 PROMPT_LEN=256 DECODE_CURVE=128,256 RUNS=3 \
RANK_Q=64 RANK_KV=64 RANK_O=2048 RANK_FF=2048 SVD_DTYPE=bf16 SVD_COMPUTE_FP32=1 \
python3 profile_asvd_flashsvd_llama_decode.py

"""

import os, time, platform, inspect, gc, statistics, sys
from typing import Optional, Dict, Tuple, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
LLAMA_KERNELS = os.path.join(REPO_ROOT, "decoders", "llama", "kernels")
if LLAMA_KERNELS not in sys.path:
    sys.path.insert(0, LLAMA_KERNELS)

from decoders.llama.kernels.flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors
from decoders.llama.kernels.flashsvdswiglu import flashsvd_ffn_swiglu

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

def _repeat_lowrank(P: torch.Tensor, n_rep: int) -> torch.Tensor:
    # P: [B, Hk, T, R] -> [B, H, T, R]
    if n_rep == 1:
        return P
    B, Hk, T, R = P.shape
    return P[:, :, None].expand(B, Hk, n_rep, T, R).reshape(B, Hk * n_rep, T, R)

def _flatten_per_head(P: torch.Tensor) -> torch.Tensor:
    # P: [B, H, T, R] -> [B, T, H*R]
    B, H, T, R = P.shape
    return P.permute(0, 2, 1, 3).reshape(B, T, H * R)

def _block_diag_Us(Us: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Us: [Hk, Dh, R] -> block-diagonal matrix [H_total*R, H_total*Dh]
    Hk, Dh, R = Us.shape
    H = Hk * n_rep
    out = Us.new_zeros(H * R, H * Dh)
    for hk in range(Hk):
        block = Us[hk].transpose(0, 1)  # [R, Dh]
        for rep_idx in range(n_rep):
            h = hk * n_rep + rep_idx
            rs = h * R
            cs = h * Dh
            out[rs:rs + R, cs:cs + Dh] = block
    return out

def _build_causal_mask(batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
    if seq_len == 0:
        return torch.zeros(batch_size, 1, 0, 0, device=device, dtype=dtype)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask.masked_fill_(causal, torch.finfo(dtype).min)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, -1, -1, -1)

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
            for t in (getattr(blk, "_asvd_pq", None), getattr(blk, "_asvd_pk", None), getattr(blk, "_asvd_pv", None)):
                if torch.is_tensor(t) and t.is_cuda:
                    total += t.numel() * t.element_size()
    except Exception:
        pass
    return total / MiB


class _SimpleRotary:
    def __init__(self, inv_freq: torch.Tensor):
        self.inv_freq = inv_freq
        self._cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def __call__(self, x: torch.Tensor, position_ids: torch.Tensor):
        # x: [B*H, M, Dh], position_ids: [B*H, M]
        device = x.device
        dtype = x.dtype
        key = (device, torch.float32)
        if key not in self._cache:
            self._cache[key] = self.inv_freq.to(device=device, dtype=torch.float32)
        inv = self._cache[key]
        half = inv.numel()
        angles = position_ids.to(torch.float32)[..., None] * inv
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        cos = torch.stack((cos, cos), dim=-1).reshape(position_ids.size(0), position_ids.size(1), half * 2)
        sin = torch.stack((sin, sin), dim=-1).reshape(position_ids.size(0), position_ids.size(1), half * 2)
        return cos.to(dtype), sin.to(dtype)

# choose SDPA context (new API if available; fallback otherwise)
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

# ---------------- SVD LLaMA block (ASVD-only) ----------------
class SVDLlamaBlock(nn.Module):
    """SVD for q/k/v (+ optional o/ff) with FlashSVD RoPE attention and ASVD cache."""

    def __init__(self, hf_layer: nn.Module, cfg,
                 rank_q: int, rank_kv: int, rank_o: Optional[int], rank_ff: Optional[int],
                 factor_dtype: torch.dtype = torch.float32, compute_in_fp32: bool = True):
        super().__init__()
        attn, mlp = hf_layer.self_attn, hf_layer.mlp
        self.D = int(cfg.hidden_size)
        self.H = int(cfg.num_attention_heads)
        self.Hk = int(getattr(cfg, "num_key_value_heads", self.H))
        self.Dh = self.D // self.H
        self.rep = self.H // self.Hk

        if int(rank_q) != int(rank_kv):
            raise ValueError("FlashSVD decode path currently requires RANK_Q == RANK_KV")
        self.rank = int(rank_q)

        self.compute_in_fp32 = bool(compute_in_fp32)
        self.factor_dtype = factor_dtype
        self.layer_idx = getattr(getattr(hf_layer, "self_attn", None), "layer_idx", None)
        self.rotary_emb = getattr(attn, "rotary_emb", None)

        rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
        half = self.Dh // 2
        inv = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("rope_inv_freq", inv, persistent=False)
        if self.rotary_emb is None:
            self.rotary_emb = _SimpleRotary(self.rope_inv_freq)


        # internal ASVD caches (rank-space) + position ids
        self._asvd_pq_cpu: Optional[torch.Tensor] = None  # stored on CPU
        self._asvd_pk: Optional[torch.Tensor] = None      # [B,Hk,T,R]
        self._asvd_pv: Optional[torch.Tensor] = None
        self._pos_ids_cpu: Optional[torch.Tensor] = None  # [B,T]

        # norms
        self.ln1 = hf_layer.input_layernorm
        self.ln2 = hf_layer.post_attention_layernorm

        # SVD factors for Q/K/V
        q_Us, q_V = _decompose_heads_svd(attn.q_proj.weight, self.H, self.Dh, rank_q)
        k_Us, k_V = _decompose_heads_svd(attn.k_proj.weight, self.Hk, self.Dh, rank_kv)
        v_Us, v_V = _decompose_heads_svd(attn.v_proj.weight, self.Hk, self.Dh, rank_kv)
        self.q_Us = nn.Parameter(q_Us.to(factor_dtype), requires_grad=False)
        self.q_V = nn.Parameter(q_V.to(factor_dtype), requires_grad=False)
        self.k_Us = nn.Parameter(k_Us.to(factor_dtype), requires_grad=False)
        self.k_V = nn.Parameter(k_V.to(factor_dtype), requires_grad=False)
        self.v_Us = nn.Parameter(v_Us.to(factor_dtype), requires_grad=False)
        self.v_V = nn.Parameter(v_V.to(factor_dtype), requires_grad=False)

        # FlashSVD RoPE attention kernel
        self.flash_attn = FlashSVDRoPEAttention(self.H, self.Dh, self.rotary_emb)

        # Output / MLP (low-rank optional)
        if rank_o is not None and rank_o > 0:
            o_Us, o_V = _decompose_full_svd(attn.o_proj.weight, rank_o)
            self.o_Us = nn.Parameter(o_Us.to(factor_dtype), requires_grad=False)
            self.o_V = nn.Parameter(o_V.to(factor_dtype), requires_grad=False)
            self.use_lr_o = True
        else:
            self.o = nn.Linear(self.D, self.D, bias=False, dtype=attn.o_proj.weight.dtype)
            with torch.no_grad():
                self.o.weight.copy_(attn.o_proj.weight)
            self.use_lr_o = False

        inter = int(cfg.intermediate_size)
        if rank_ff is not None and rank_ff > 0:
            # Low-rank FFN (SwiGLU) with FlashSVD kernel in rank space
            g_Us, g_V = _decompose_full_svd(mlp.gate_proj.weight, rank_ff)
            u_Us, u_V = _decompose_full_svd(mlp.up_proj.weight,   rank_ff)
            d_Us, d_V = _decompose_full_svd(mlp.down_proj.weight, rank_ff)

            # Build kernel-friendly factors:
            # U1: [D, R1] (shared input low-rank). Use up-proj basis (approximate); transpose g_V/u_V shape [r,D] -> [D,r]
            U1 = u_V.t()  # [D, r]
            # V1: [r, 2*inter] = concat([g_Us^T, u_Us^T])
            V1 = torch.cat([g_Us.t(), u_Us.t()], dim=1)
            # U2: [inter, r2], V2: [r2, D]
            U2 = d_V.t()
            V2 = d_Us.t()

            self.ff_u1 = nn.Parameter(U1.to(factor_dtype), requires_grad=False)
            self.ff_v1 = nn.Parameter(V1.to(factor_dtype), requires_grad=False)
            self.ff_u2 = nn.Parameter(U2.to(factor_dtype), requires_grad=False)
            self.ff_v2 = nn.Parameter(V2.to(factor_dtype), requires_grad=False)
            self.register_buffer("ff_b1", torch.zeros(2*inter, dtype=factor_dtype), persistent=False)
            self.register_buffer("ff_b2", torch.zeros(self.D,    dtype=factor_dtype), persistent=False)

            # Drop per-op SVD factors for FFN to avoid duplication
            for name in ("g_Us","g_V","u_Us","u_V","d_Us","d_V"):
                if hasattr(self, name):
                    try: delattr(self, name)
                    except Exception: pass

            self.use_lr_ff = True
        else:
            self.gate = nn.Linear(self.D, inter, bias=False, dtype=mlp.gate_proj.weight.dtype)
            self.up = nn.Linear(self.D, inter, bias=False, dtype=mlp.up_proj.weight.dtype)
            self.down = nn.Linear(inter, self.D, bias=False, dtype=mlp.down_proj.weight.dtype)
            with torch.no_grad():
                self.gate.weight.copy_(mlp.gate_proj.weight)
                self.up.weight.copy_(mlp.up_proj.weight)
                self.down.weight.copy_(mlp.down_proj.weight)
            self.use_lr_ff = False

        # Remove heavy dense modules on the original layer for memory hygiene
        for obj in (attn, mlp):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                if hasattr(obj, name):
                    try:
                        delattr(obj, name)
                    except Exception:
                        pass

    def _proj_per_head_lowrank(self, x: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if self.compute_in_fp32:
            P = torch.einsum('b t d, h r d -> b t h r', x.float(), V.float())
        else:
            P = torch.einsum('b t d, h r d -> b t h r', x.to(V.dtype), V)
        return P.transpose(1, 2).to(self.factor_dtype).contiguous()

    def forward(self, hidden_states, attention_mask=None, position_ids=None, use_cache: bool = True, **kw):
        B, T, _ = hidden_states.shape
        x = self.ln1(hidden_states)

        if attention_mask is None:
            T_max = T
            x_chunk = x
            pos_chunk = position_ids
        else:
            keep = attention_mask[:, :T] > 0
            T_max = int(keep.sum(dim=1).max().item())
            T_max = max(T_max, 1)
            x_chunk = x[:, :T_max, :]
            pos_chunk = position_ids[:, :T_max] if position_ids is not None else None

        Pq_chunk = self._proj_per_head_lowrank(x_chunk, self.q_V)
        Pk_chunk = self._proj_per_head_lowrank(x_chunk, self.k_V)
        Pv_chunk = self._proj_per_head_lowrank(x_chunk, self.v_V)

        if use_cache:
            Pq_chunk_cpu = Pq_chunk.detach().to("cpu")
            if self._asvd_pq_cpu is None:
                self._asvd_pq_cpu = Pq_chunk_cpu
                self._asvd_pk = Pk_chunk
                self._asvd_pv = Pv_chunk
            else:
                self._asvd_pq_cpu = torch.cat([self._asvd_pq_cpu, Pq_chunk_cpu], dim=2)
                self._asvd_pk = torch.cat([self._asvd_pk, Pk_chunk], dim=2)
                self._asvd_pv = torch.cat([self._asvd_pv, Pv_chunk], dim=2)

            if pos_chunk is not None:
                pos_chunk_cpu = pos_chunk.long().cpu()
                if self._pos_ids_cpu is None:
                    self._pos_ids_cpu = pos_chunk_cpu
                else:
                    self._pos_ids_cpu = torch.cat([self._pos_ids_cpu, pos_chunk_cpu], dim=1)
            elif self._pos_ids_cpu is None:
                seq_len = self._asvd_pk.size(2)
                self._pos_ids_cpu = torch.arange(seq_len, device="cpu").unsqueeze(0).expand(B, -1)

            Pq_seq_cpu = self._asvd_pq_cpu
            Pk_seq = self._asvd_pk
            Pv_seq = self._asvd_pv
            pos_seq_dev = self._pos_ids_cpu.to(hidden_states.device)
            Pq_seq = Pq_seq_cpu.to(hidden_states.device, dtype=self.factor_dtype)
            del Pq_chunk
        else:
            Pq_seq = Pq_chunk.to(self.factor_dtype)
            Pk_seq, Pv_seq = Pk_chunk, Pv_chunk
            if pos_chunk is not None:
                pos_seq_dev = pos_chunk.long()
            else:
                seq_len = Pq_seq.size(2)
                pos_seq_dev = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        # Prepare per-head rank-space inputs without flattening; avoid block-diagonal factors
        # Pq: [B,H,T,R]; Pk,Pv: [B,Hk,T,R]
        Pq_hm = Pq_seq  # already [B,H,T,R]
        Pk_hm = Pk_seq  # [B,Hk,T,R]
        Pv_hm = Pv_seq

        # V factors per head: [H|Hk, R, Dh]
        q_V = self.q_Us.detach().to(self.factor_dtype).transpose(1, 2).contiguous().to(hidden_states.device)
        k_V = self.k_Us.detach().to(self.factor_dtype).transpose(1, 2).contiguous().to(hidden_states.device)
        v_V = self.v_Us.detach().to(self.factor_dtype).transpose(1, 2).contiguous().to(hidden_states.device)

        factors = QKVFactors(Pq=Pq_hm,
                             Pk=Pk_hm,
                             Pv=Pv_hm,
                             Vq=q_V,
                             Vk=k_V,
                             Vv=v_V,
                             bq=None, bk=None, bv=None)
        # Causal masking is handled inside the FlashSVD kernel; no additive mask needed
        attn_bmhd = self.flash_attn(factors, None, pos_seq_dev)
        attn_chunk = attn_bmhd[:, -T_max:, :, :]
        y = attn_chunk.permute(0, 2, 1, 3).contiguous().view(B, T_max, self.D).to(hidden_states.dtype)

        # Apply output projection (O) as in standard attention
        if getattr(self, "use_lr_o", False):
            y = ((y.to(self.o_V.dtype) @ self.o_V.t()) @ self.o_Us.t()).to(hidden_states.dtype)
        else:
            y = getattr(self, "o", None)(y) if hasattr(self, "o") else y

        if use_cache:
            # release GPU copies
            del Pq_seq
        del q_V, k_V, v_V

        h = hidden_states[:, :T_max, :] + y
        z = self.ln2(h)
        if getattr(self, "use_lr_ff", False):
            # Rank-space SwiGLU FFN via FlashSVD kernel
            P = z.to(self.ff_u1.dtype) @ self.ff_u1  # [B,T,R1]
            ff = flashsvd_ffn_swiglu(
                P.contiguous(),
                self.ff_v1, self.ff_u2, self.ff_v2,
                self.ff_b1, self.ff_b2,
                use_autotune=True,
            ).to(hidden_states.dtype)
        else:
            ff = self.down(F.silu(self.gate(z)) * self.up(z))
        out_chunk = h + ff

        if T_max < T:
            pad = torch.zeros(B, T - T_max, self.D, dtype=out_chunk.dtype, device=out_chunk.device)
            out = torch.cat([out_chunk, pad], dim=1)
        else:
            out = out_chunk

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
        for name in ("_asvd_pq_cpu", "_asvd_pk", "_asvd_pv", "_pos_ids_cpu"):
            if hasattr(blk, name):
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
