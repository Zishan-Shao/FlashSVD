#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_asvd_accum_flash.py  —  Minimal ASVD decode memory profiler (no big logits)

What this version does:
- ASVD-only (no dense baseline anywhere).
- **Avoids full [B,S,V] logits**: we call model.transformer() and apply lm_head
  only on the last position to choose the next token.
- Measures KV size using allocator-accurate storage bytes (includes padding).
- Reports prefill vs decode:
    * time (ms)
    * peak allocated (MiB) per phase (max_memory_allocated)
    * post-step allocated peak (MiB) — measured right after each token step
    * end-of-phase allocated (MiB) — steady memory at phase end
    * KV_end (MiB) — actual bytes occupied by KV storages
- Aggressive freeing in forward() to minimize transient pressure.


CUDA_VISIBLE_DEVICES=0 python3 profile_asvd_accum_flash.py   --decode-batch 16   --prompt-len 256   --decode-curve 128,256   --rounds 3   --rank-ratio-attn 1   --rank-ratio-mlp 1   --kv-profile

"""

import os, math, time, argparse, statistics, itertools, gc
from typing import Optional, Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer

from kernels.flash_attn_causal import flash_attn_triton_kvcache

MiB = float(1024**2)

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def svd_factor(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if W.dtype not in (torch.float32, torch.float64):
        W = W.float()
    W = W.contiguous()
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except TypeError:
        U_, S_, V_ = torch.svd(W)
        U, S, Vh = U_, S_, V_.t()
    r = min(rank, S.numel())
    U_r = U[:, :r].contiguous()
    V_r = (S[:r, None] * Vh[:r, :]).contiguous()
    return U_r, V_r

def as_linear_weight(W_raw: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    if W_raw.shape == (in_dim, out_dim):
        return W_raw.contiguous()
    if W_raw.shape == (out_dim, in_dim):
        return W_raw.t().contiguous()
    raise ValueError(f"Unexpected weight shape {tuple(W_raw.shape)}; expected ({in_dim},{out_dim}) or ({out_dim},{in_dim}).")

# -------------------------
# KV Profiler (time/bytes/mem)
# -------------------------
class KVProfiler:
    """
    Per-phase profiler: 'prefill' and 'decode'.
    Tracks wall times, KV bytes, and memory checkpoints:
      - qkv_s, attn_s, update_s, kv_new_bytes, kv_read_bytes, calls
      - poststep_peak_bytes: max(memory_allocated) right after a step
      - end_alloc_bytes: memory_allocated at end of phase
    """
    def __init__(self):
        self.enabled = False
        self.phase = "decode"
        self.reset()

    def reset(self):
        def zero():
            return dict(
                qkv_s=0.0, attn_s=0.0, update_s=0.0,
                kv_new_bytes=0, kv_read_bytes=0, calls=0,
                poststep_peak_bytes=0, end_alloc_bytes=0
            )
        self.stats = {"prefill": zero(), "decode": zero()}

    def enable(self, flag: bool = True):
        self.enabled = bool(flag)

    def set_phase(self, phase: str):
        self.phase = "prefill" if phase == "prefill" else "decode"

    def add_time(self, key: str, seconds: float):
        if self.enabled:
            self.stats[self.phase][key] += float(seconds)

    def add_bytes(self, key: str, nbytes: int):
        if self.enabled:
            self.stats[self.phase][key] += int(nbytes)

    def inc_calls(self):
        if self.enabled:
            self.stats[self.phase]["calls"] += 1

    def add_mem_poststep(self, bytes_now: int):
        if self.enabled:
            d = self.stats[self.phase]
            if bytes_now > d["poststep_peak_bytes"]:
                d["poststep_peak_bytes"] = bytes_now

    def set_end_alloc(self, bytes_now: int):
        if self.enabled:
            self.stats[self.phase]["end_alloc_bytes"] = bytes_now

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for ph, d in self.stats.items():
            out[ph] = {
                "qkv_ms": d["qkv_s"] * 1000.0,
                "attn_ms": d["attn_s"] * 1000.0,
                "update_ms": d["update_s"] * 1000.0,
                "kv_new_MiB": d["kv_new_bytes"] / MiB,
                "kv_read_MiB": d["kv_read_bytes"] / MiB,
                "poststep_peak_MiB": d["poststep_peak_bytes"] / MiB,
                "end_alloc_MiB": d["end_alloc_bytes"] / MiB,
                "calls": d["calls"],
            }
        return out

def attach_profiler(model: GPT2LMHeadModel, prof: Optional[KVProfiler]):
    if not hasattr(model, "transformer"):
        return
    for layer in model.transformer.h:
        if hasattr(layer, "profiler"):
            layer.profiler = prof
        blk = getattr(layer, "block", None)
        if blk is not None and hasattr(blk, "profiler"):
            blk.profiler = prof

# -------------------------
# ASVD cache
# -------------------------
class ASVDCache:
    """ Holds per-layer low-rank factors (Pk, Pv) with shape [B,H,T,r]. """
    def __init__(self, n_layers: int):
        self.layers: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
    def get_seq_length(self, layer_idx: int) -> int:
        entry = self.layers[layer_idx]
        return 0 if entry is None else entry[0].size(2)
    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.layers[layer_idx]
    @torch.no_grad()
    def update(self, Pk_new: torch.Tensor, Pv_new: torch.Tensor, layer_idx: int):
        assert Pk_new.dim() == 4 and Pv_new.dim() == 4, "Pk/Pv must be [B,H,S_new,r]"
        entry = self.layers[layer_idx]
        if entry is None:
            self.layers[layer_idx] = (Pk_new, Pv_new)
        else:
            Pk, Pv = entry
            self.layers[layer_idx] = (
                torch.cat([Pk, Pk_new], dim=2),
                torch.cat([Pv, Pv_new], dim=2),
            )

# -------------------------
# Low-rank GPT-2 Block (ASVD)
# -------------------------
class LowRankSVDBlock(nn.Module):
    """
    GPT-2 block with per-head low-rank factors U,V for Q/K/V and MLP.
    If asvd=True, the cache stores Pk = X @ Uk and Pv = X @ Uv, then we
    reconstruct dense K,V on the fly for attention.
    """
    def __init__(
        self,
        hf_layer: nn.Module,
        rank_ratio_attn: float = 1.0,
        rank_ratio_mlp: float = 1.0,
        preload_factors: Optional[Dict[str, torch.Tensor]] = None,
        asvd: bool = True,
    ):
        super().__init__()
        attn = hf_layer.attn
        self.ln1 = hf_layer.ln_1
        self.ln2 = hf_layer.ln_2

        D = attn.embed_dim
        H = attn.num_heads
        if D % H != 0:
            raise ValueError(f"[LowRankSVDBlock] embed_dim={D} not divisible by heads={H}")
        dh = D // H

        self.D, self.H, self.dh = D, H, dh
        self.asvd = asvd

        dev = next(hf_layer.parameters()).device
        ptdtype = next(hf_layer.parameters()).dtype

        # ---- ATTENTION (Q,K,V) ----
        Wc_lin = as_linear_weight(hf_layer.attn.c_attn.weight.data, in_dim=D, out_dim=3 * D)
        bc = hf_layer.attn.c_attn.bias.data.clone().to(device=dev, dtype=ptdtype)
        q_w = Wc_lin[:, :D].contiguous().view(D, H, dh)
        k_w = Wc_lin[:, D:2*D].contiguous().view(D, H, dh)
        v_w = Wc_lin[:, 2*D:3*D].contiguous().view(D, H, dh)
        q_b = bc[:D].view(H, dh).contiguous()
        k_b = bc[D:2*D].view(H, dh).contiguous()
        v_b = bc[2*D:3*D].view(H, dh).contiguous()

        r_attn = max(1, int(rank_ratio_attn * min(D, dh)))
        def alloc_uv(name: str):
            U = nn.Parameter(torch.empty(D, H, r_attn, device=dev, dtype=ptdtype))
            V = nn.Parameter(torch.empty(H, r_attn, dh, device=dev, dtype=ptdtype))
            self.register_parameter(f"{name}_U", U)
            self.register_parameter(f"{name}_V", V)
            return U, V
        self.q_U, self.q_V = alloc_uv("q")
        self.k_U, self.k_V = alloc_uv("k")
        self.v_U, self.v_V = alloc_uv("v")
        self.q_b = nn.Parameter(q_b.to(device=dev, dtype=ptdtype))
        self.k_b = nn.Parameter(k_b.to(device=dev, dtype=ptdtype))
        self.v_b = nn.Parameter(v_b.to(device=dev, dtype=ptdtype))

        if preload_factors is None:
            with torch.no_grad():
                for name, W_h in (("q", q_w), ("k", k_w), ("v", v_w)):
                    U_param = getattr(self, f"{name}_U")
                    V_param = getattr(self, f"{name}_V")
                    Us, Vs = [], []
                    for h in range(H):
                        Wh = W_h[:, h, :]
                        Uh, Vh = svd_factor(Wh, r_attn)
                        Us.append(Uh.to(device=dev, dtype=ptdtype))
                        Vs.append(Vh.to(device=dev, dtype=ptdtype))
                    U = torch.stack(Us, dim=1)
                    V = torch.stack(Vs, dim=0)
                    U_param.copy_(U)
                    V_param.copy_(V)
        else:
            self.load_factors_(preload_factors)

        # ---- OUT PROJ ----
        W_out_lin = as_linear_weight(hf_layer.attn.c_proj.weight.data, in_dim=D, out_dim=D)
        b_out = hf_layer.attn.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_out = max(1, int(rank_ratio_attn * min(W_out_lin.shape)))
        Uo, Vo = svd_factor(W_out_lin, r_out)
        self.out_U = nn.Parameter(Uo.to(device=dev, dtype=ptdtype))
        self.out_V = nn.Parameter(Vo.to(device=dev, dtype=ptdtype))
        self.out_b = nn.Parameter(b_out)

        # ---- MLP ----
        I = hf_layer.mlp.c_fc.bias.data.numel()
        W1_lin = as_linear_weight(hf_layer.mlp.c_fc.weight.data, in_dim=D, out_dim=I)
        b_fc1 = hf_layer.mlp.c_fc.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc1 = max(1, int(rank_ratio_mlp * min(W1_lin.shape)))
        U1, V1 = svd_factor(W1_lin, r_fc1)
        self.fc1_U = nn.Parameter(U1.to(device=dev, dtype=ptdtype))
        self.fc1_V = nn.Parameter(V1.to(device=dev, dtype=ptdtype))
        self.fc1_b = nn.Parameter(b_fc1)

        W2_lin = as_linear_weight(hf_layer.mlp.c_proj.weight.data, in_dim=I, out_dim=D)
        b_fc2 = hf_layer.mlp.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc2 = max(1, int(rank_ratio_mlp * min(W2_lin.shape)))
        U2, V2 = svd_factor(W2_lin, r_fc2)
        self.fc2_U = nn.Parameter(U2.to(device=dev, dtype=ptdtype))
        self.fc2_V = nn.Parameter(V2.to(device=dev, dtype=ptdtype))
        self.fc2_b = nn.Parameter(b_fc2)

        self.r_attn = r_attn
        self.profiler: Optional[KVProfiler] = None

    def factors_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "q_U": self.q_U, "q_V": self.q_V, "q_b": self.q_b,
            "k_U": self.k_U, "k_V": self.k_V, "k_b": self.k_b,
            "v_U": self.v_U, "v_V": self.v_V, "v_b": self.v_b,
            "out_U": self.out_U, "out_V": self.out_V, "out_b": self.out_b,
            "fc1_U": self.fc1_U, "fc1_V": self.fc1_V, "fc1_b": self.fc1_b,
            "fc2_U": self.fc2_U, "fc2_V": self.fc2_V, "fc2_b": self.fc2_b,
        }

    def load_factors_(self, tensors: Dict[str, torch.Tensor]):
        mine = self.factors_state_dict()
        for k, p in mine.items():
            if k not in tensors:
                raise KeyError(f"Missing factor '{k}' in preload_factors")
            with torch.no_grad():
                p.copy_(tensors[k].to(dtype=p.dtype, device=p.device))

    def forward(
        self,
        hidden_states: torch.Tensor,                           # [B,S,D]
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (Pk,Pv) [B,H,T_past,r]
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        B, S, D = hidden_states.shape
        dev = hidden_states.device
        H, dh, r = self.H, self.dh, self.r_attn
        prof: Optional[KVProfiler] = self.profiler

        # LN1
        x = self.ln1(hidden_states)

        # ---- Build Q, factors, and reconstruct dense new-step K,V
        t0 = time.perf_counter()
        Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]
        Pk_new = torch.einsum('bsd,dhr->bhsr', x, self.k_U)
        Pv_new = torch.einsum('bsd,dhr->bhsr', x, self.v_U)
        del x
        K_new  = torch.einsum('bhsr,hrd->bhsd', Pk_new, self.k_V) + self.k_b[None, :, None, :]
        V_new  = torch.einsum('bhsr,hrd->bhsd', Pv_new, self.v_V) + self.v_b[None, :, None, :]

        # Concat past factors (reconstruct dense K_cat/V_cat)
        if layer_past is not None and isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
            past0, past1 = layer_past
            if past0 is not None and past0.dim() == 4 and past0.size(-1) == r:
                Pk_cat = torch.cat([past0.to(Pk_new.dtype), Pk_new], dim=2)
                Pv_cat = torch.cat([past1.to(Pv_new.dtype), Pv_new], dim=2)
                K_cat = torch.einsum('bhtR,hRd->bhtd', Pk_cat, self.k_V) + self.k_b[None, :, None, :]
                V_cat = torch.einsum('bhtR,hRd->bhtd', Pv_cat, self.v_V) + self.v_b[None, :, None, :]
                del Pk_cat, Pv_cat
            else:
                K_cat = torch.cat([past0.to(K_new.dtype), K_new], dim=2)
                V_cat = torch.cat([past1.to(V_new.dtype), V_new], dim=2)
        else:
            K_cat, V_cat = K_new, V_new

        if prof:
            prof.add_time("qkv_s", time.perf_counter() - t0)
            prof.add_bytes("kv_read_bytes", (K_cat.numel() + V_cat.numel()) * K_cat.element_size())
            prof.add_bytes("kv_new_bytes", (Pk_new.numel() + Pv_new.numel()) * Pk_new.element_size())
            prof.inc_calls()

        # ---- Attention (build compact mask)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                q_mask = attention_mask[..., -S:].to(dtype=torch.bool)
                if q_mask.size(2) != 1:
                    q_mask = q_mask[..., :1, :]
            elif attention_mask.dim() == 2:
                q_mask = attention_mask[:, -S:].bool()[:, None, None, :]
            else:
                q_mask = torch.ones(B, 1, 1, S, dtype=torch.bool, device=dev)
        else:
            q_mask = torch.ones(B, 1, 1, S, dtype=torch.bool, device=dev)
        attn_mask_bh1s = q_mask.expand(B, H, 1, S).contiguous()
        del q_mask

        tA0 = time.perf_counter()
        Y_heads = flash_attn_triton_kvcache(Q, K_cat, V_cat, attn_mask_bh1s)
        if prof: prof.add_time("attn_s", time.perf_counter() - tA0)

        # Free big attention inputs
        del Q, K_cat, V_cat, attn_mask_bh1s

        # Merge heads
        Y = Y_heads.transpose(1, 2).contiguous().view(B, S, self.D)
        del Y_heads

        # Out proj + residual
        Y = torch.matmul(torch.matmul(Y, self.out_U), self.out_V)
        Y.add_(self.out_b)
        hidden_states = hidden_states.add(Y)
        del Y

        # MLP (+ residual) with aggressive frees
        z = self.ln2(hidden_states)
        t1 = torch.matmul(z, self.fc1_U); del z
        h1 = torch.matmul(t1, self.fc1_V); del t1
        h1.add_(self.fc1_b)
        h1 = F.gelu(h1)
        t2 = torch.matmul(h1, self.fc2_U); del h1
        h2 = torch.matmul(t2, self.fc2_V); del t2
        h2.add_(self.fc2_b)
        hidden_states.add_(h2); del h2

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + ((Pk_new, Pv_new),)
        else:
            del Pk_new, Pv_new

        if output_attentions:
            outputs = outputs + (None,)
        return outputs

def _attach_asvd_cache_to_shims(model, asvd_cache):
    for i, layer in enumerate(model.transformer.h):
        if isinstance(layer, LayerShim) and layer.asvd:
            setattr(layer, "_asvd_cache", asvd_cache)

class LayerShim(nn.Module):
    def __init__(self, block: LowRankSVDBlock, layer_idx: int, asvd: bool = True):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.asvd = asvd
        self._asvd_cache = None
        self.profiler: Optional[KVProfiler] = None

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        use_cache_flag = bool(kwargs.get("use_cache", False))
        prof: Optional[KVProfiler] = self.profiler
        layer_past = None

        if self.asvd and use_cache_flag:
            asvd_cache = getattr(self, "_asvd_cache", None)
            if isinstance(asvd_cache, ASVDCache):
                entry = asvd_cache.get(self.layer_idx)
                if entry is not None and asvd_cache.get_seq_length(self.layer_idx) > 0:
                    layer_past = entry

        result = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache_flag,
            output_attentions=kwargs.get("output_attentions", False),
        )

        if self.asvd and use_cache_flag:
            asvd_cache = getattr(self, "_asvd_cache", None)
            if (isinstance(asvd_cache, ASVDCache) and
                isinstance(result, tuple) and len(result) >= 2 and
                isinstance(result[1], (tuple, list)) and len(result[1]) == 2):
                Pk_new, Pv_new = result[1]
                t0 = time.perf_counter()
                asvd_cache.update(Pk_new, Pv_new, self.layer_idx)
                if prof: prof.add_time("update_s", time.perf_counter() - t0)
        return result

# -------------------------
# Build ASVD model
# -------------------------
def build_asvd_model(rank_ratio_attn: float, rank_ratio_mlp: float, device: Optional[str] = None) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if device:
        model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model.transformer.h):
        blk = LowRankSVDBlock(layer, rank_ratio_attn=rank_ratio_attn, rank_ratio_mlp=rank_ratio_mlp, asvd=True)
        shim = LayerShim(blk, layer_idx=i, asvd=True).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim
    model._uses_asvd_cache = True
    return model

# -------------------------
# Accurate KV bytes (allocator storage)
# -------------------------
@torch.no_grad()
def estimate_kv_bytes(past_key_values: ASVDCache) -> int:
    """
    Sum UNIQUE storage bytes of all Pk/Pv tensors in ASVDCache.
    Uses untyped_storage().nbytes() when available.
    """
    def storage_nbytes(t: torch.Tensor) -> Tuple[int, int]:
        try:
            s = t.untyped_storage()
            return s.data_ptr(), int(s.nbytes())
        except Exception:
            s = t.storage()
            nbytes = (s.nbytes() if hasattr(s, "nbytes") else s.size() * t.element_size())
            ptr = s.data_ptr() if hasattr(s, "data_ptr") else t.data_ptr()
            return ptr, int(nbytes)

    seen = set()
    total = 0
    for entry in past_key_values.layers:
        if entry is None:
            continue
        Pk, Pv = entry
        for t in (Pk, Pv):
            if t is None or not t.is_cuda:
                continue
            key = storage_nbytes(t)
            if key in seen:
                continue
            seen.add(key)
            total += key[1]
    return total

# -------------------------
# Helper: next token from last hidden (no full logits)
# -------------------------
@torch.no_grad()
def _next_token_from_last_hidden(model: GPT2LMHeadModel, last_hidden_state: torch.Tensor, greedy: bool = True) -> torch.Tensor:
    last = last_hidden_state[:, -1, :]           # [B, D]
    logits_last = model.lm_head(last)            # [B, V]
    if greedy:
        return logits_last.argmax(dim=-1, keepdim=True)  # [B,1]
    probs = F.softmax(logits_last.float(), dim=-1)
    return torch.multinomial(probs, 1)

# -------------------------
# Decode benchmark (ASVD only, no big logits)
# -------------------------
@torch.no_grad()
def decode_benchmark_asvd(
    model: GPT2LMHeadModel,
    prompt: torch.Tensor,
    new_tokens: int,
    device: str,
    profiler: Optional[KVProfiler] = None,
    greedy: bool = True,
) -> Dict[str, float]:
    model.eval()
    B = prompt.size(0)

    attach_profiler(model, profiler)
    if profiler is not None:
        profiler.reset()
        profiler.enable(True)

    # clear mem counters
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # ---- Prefill (cache build) ----
    if profiler: profiler.set_phase("prefill")
    past = ASVDCache(n_layers=len(model.transformer.h))
    _attach_asvd_cache_to_shims(model, past)

    t0 = time.perf_counter()
    # ONLY transformer output (no full logits)
    out = model.transformer(input_ids=prompt, use_cache=True, return_dict=True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # Choose first token using ONLY last position logits
    next_id = _next_token_from_last_hidden(model, out.last_hidden_state, greedy=greedy)

    # Memory checkpoints (prefill)
    prefill_peak_mib = torch.cuda.max_memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    prefill_end_alloc_mib = torch.cuda.memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    _ = estimate_kv_bytes(past) / MiB  # (optional, not printed here)
    if profiler:
        profiler.add_mem_poststep(int(prefill_end_alloc_mib * MiB))
        profiler.set_end_alloc(int(prefill_end_alloc_mib * MiB))

    # Free prefill outputs BEFORE decode measurement
    del out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # ---- Decode loop ----
    if profiler: profiler.set_phase("decode")

    t_dec = 0.0
    decode_poststep_peak_mib = 0.0

    for _ in range(new_tokens):
        # feed the token we just picked
        t1 = time.perf_counter()
        _attach_asvd_cache_to_shims(model, past)
        step = model.transformer(input_ids=next_id, use_cache=True, return_dict=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_dec += (time.perf_counter() - t1)

        # pick next strictly from last position (no [B,S,V])
        next_id = _next_token_from_last_hidden(model, step.last_hidden_state, greedy=greedy)

        # free step outputs promptly
        del step
        gc.collect()

        # Measure allocated *after* the step, which grows with KV
        if torch.cuda.is_available():
            alloc_now_mib = torch.cuda.memory_allocated() / MiB
            if profiler: profiler.add_mem_poststep(int(alloc_now_mib * MiB))
            if alloc_now_mib > decode_poststep_peak_mib:
                decode_poststep_peak_mib = alloc_now_mib

    decode_peak_mib = torch.cuda.max_memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    decode_end_alloc_mib = torch.cuda.memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    kv_final_mib = estimate_kv_bytes(past) / MiB
    if profiler:
        profiler.set_end_alloc(int(decode_end_alloc_mib * MiB))

    toks_per_s = (B * max(new_tokens, 1)) / max(t_dec, 1e-6)

    return {
        "prefill_ms": prefill_s * 1000.0,
        "decode_ms": t_dec * 1000.0,
        "prefill_peak_MiB": prefill_peak_mib,
        "prefill_end_alloc_MiB": prefill_end_alloc_mib,
        "decode_peak_MiB": decode_peak_mib,
        "decode_poststep_peak_MiB": decode_poststep_peak_mib,
        "decode_end_alloc_MiB": decode_end_alloc_mib,
        "kv_end_MiB": kv_final_mib,
        "toks_per_s": toks_per_s,
        "prof_snapshot": (profiler.snapshot() if profiler and profiler.enabled else None),
    }

def _fmt_mean_std(vals: List[float], width: int = None, prec: int = 2) -> str:
    if not vals:
        s = "nan"
    else:
        m = statistics.mean(vals)
        sd = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
        s = f"{m:.{prec}f}±{sd:.{prec}f}"
    return f"{s:>{width}}" if width else s

@torch.no_grad()
def decode_growth_curve_asvd(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
    prompt_len: int,
    curve_lens: List[int],
    rounds: int = 5,
    kv_profile: bool = True,
):
    print(f"\n=== Decoding-time KV-cache growth (ASVD, last-token logits only) — {rounds} rounds avg ===")
    vocab = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    prompt = torch.randint(0, min(1000, vocab), (batch_size, prompt_len), device=device)

    header = (f"{'new_T':>7} | {'t/s':>10} | {'prefill ms':>11} | {'decode ms':>10} | "
              f"{'prefill peak':>12} | {'dec peak':>9} | {'poststep':>9} | {'end_alloc':>9} | {'KV_end':>7}")
    print(header)
    print("-" * len(header))

    for idx, new_T in enumerate(curve_lens):
        tps, pre_ms, dec_ms = [], [], []
        pre_peak, dec_peak = [], []
        poststep, end_alloc, kv_end = [], [], []

        for r in range(rounds):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            prof = KVProfiler() if kv_profile else None
            res = decode_benchmark_asvd(model, prompt, new_T, device, profiler=prof, greedy=True)

            tps.append(res["toks_per_s"])
            pre_ms.append(res["prefill_ms"])
            dec_ms.append(res["decode_ms"])
            pre_peak.append(res["prefill_peak_MiB"])
            dec_peak.append(res["decode_peak_MiB"])
            poststep.append(res["decode_poststep_peak_MiB"])
            end_alloc.append(res["decode_end_alloc_MiB"])
            kv_end.append(res["kv_end_MiB"])

            if kv_profile and idx == 0 and r == 0 and res["prof_snapshot"]:
                snap = res["prof_snapshot"]
                print("\n  [KV Profiler — per-phase, aggregated across layers, round 1]")
                for ph in ("prefill", "decode"):
                    s = snap[ph]
                    calls = int(s["calls"])
                    avg_attn_ms = (s["attn_ms"]/max(calls,1)) if calls else 0.0
                    avg_qkv_ms  = (s["qkv_ms"]/max(calls,1))  if calls else 0.0
                    avg_upd_ms  = (s["update_ms"]/max(calls,1)) if calls else 0.0
                    print(f"   {ph:>7}: qkv={s['qkv_ms']:7.1f}ms  attn={s['attn_ms']:7.1f}ms  upd={s['update_ms']:7.1f}ms  "
                          f"calls={calls:4d}  kv_new={s['kv_new_MiB']:7.1f}MiB  kv_read={s['kv_read_MiB']:7.1f}MiB  "
                          f"poststep_peak={s['poststep_peak_MiB']:7.1f}MiB  end_alloc={s['end_alloc_MiB']:7.1f}MiB  "
                          f"[avg/call: qkv={avg_qkv_ms:5.2f}ms, attn={avg_attn_ms:5.2f}ms, upd={avg_upd_ms:5.2f}ms]")

        print(
            f"{new_T:7d} | "
            f"{_fmt_mean_std(tps, 10, 2)} | {_fmt_mean_std(pre_ms, 11, 1)} | {_fmt_mean_std(dec_ms, 10, 1)} | "
            f"{_fmt_mean_std(pre_peak, 12, 1)} | {_fmt_mean_std(dec_peak, 9, 1)} | "
            f"{_fmt_mean_std(poststep, 9, 1)} | {_fmt_mean_std(end_alloc, 9, 1)} | "
            f"{_fmt_mean_std(kv_end, 7, 1)}"
        )

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-ratio-attn", type=float, default=1.0)
    parser.add_argument("--rank-ratio-mlp",  type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    # Decode mem benchmark (ASVD only)
    parser.add_argument("--decode-curve", type=str, default="64,128,256,512")
    parser.add_argument("--decode-batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--kv-profile", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)

    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build ASVD model
    print("\n=== Building ASVD Model ===")
    svd_model = build_asvd_model(rank_ratio_attn=args.rank_ratio_attn,
                                 rank_ratio_mlp=args.rank_ratio_mlp,
                                 device=device)
    first_blk = svd_model.transformer.h[0].block
    print(f"QKV rank: {first_blk.r_attn}, embed_dim={first_blk.D}, heads={first_blk.H}, dh={first_blk.dh}")

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    curve = [int(x) for x in args.decode_curve.split(",") if x.strip()]
    bsz = args.decode_batch
    p_len = args.prompt_len

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    decode_growth_curve_asvd(
        svd_model, tok, device=device,
        batch_size=bsz, prompt_len=p_len, curve_lens=curve,
        rounds=args.rounds, kv_profile=args.kv_profile
    )

if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=0 \
python3 profile_asvd_accum_flash.py \
  --decode-batch 16 \
  --prompt-len 256 \
  --decode-curve 128,256 \
  --rounds 3 \
  --rank-ratio-attn 0.5 \
  --rank-ratio-mlp 0.5 \
  --kv-profile

'''