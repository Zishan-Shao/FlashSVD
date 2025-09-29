#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_svd_kv_accum.py  —  ASVD variant

Per-head low-rank SVD factorization for GPT-2 with correctness checks, profiling,
and a decoding-time memory/throughput growth benchmark.

NEW (ASVD):
- Instead of caching dense K,V, we cache low-rank factors:
    Pk = X @ Uk   [B,H,T,r]     Pv = X @ Uv   [B,H,T,r]
  Then reconstruct each step:
    K  = Pk @ Vk + kb           V  = Pv @ Vv + vb
- This reduces cache memory from O(T·dh) to O(T·r) per head per stream.
- Toggle with --asvd (only affects the SVD model; dense baseline still uses HF cache).
"""

import os, math, time, itertools, argparse
from typing import Optional, Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer

from kernels.flash_attn_causal import flash_attn_triton_kvcache
from kernels.flashsvdattn import flash_svd_attention
from kernels.flashsvdffn import flashsvd_ffn

# =========================
# Utils
# =========================
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_persistent_memory(m: nn.Module) -> float:
    total = 0
    for p in itertools.chain(m.parameters(), m.buffers()):
        total += p.numel() * p.element_size()
    return total / (1024**2)

def compute_module_param_bytes(m: nn.Module) -> int:
    total = 0
    for p in itertools.chain(m.parameters(), m.buffers()):
        total += p.numel() * p.element_size()
    return total

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

def make_causal_slice_mask(s_new: int, total_len: int, device, dtype=torch.bool) -> torch.Tensor:
    full = torch.ones(total_len, total_len, dtype=dtype, device=device).tril_()
    return full[-s_new:, :].contiguous()

def as_linear_weight(W_raw: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    if W_raw.shape == (in_dim, out_dim):
        return W_raw.contiguous()
    if W_raw.shape == (out_dim, in_dim):
        return W_raw.t().contiguous()
    raise ValueError(f"Unexpected weight shape {tuple(W_raw.shape)}; expected ({in_dim},{out_dim}) or ({out_dim},{in_dim}).")


# =========================
# ASVD cache (low-rank factors)
# =========================
class ASVDCache:
    """
    Holds per-layer low-rank factors (Pk, Pv) with shape [B,H,T,r].
    """
    def __init__(self, n_layers: int):
        self.layers: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

    def get_seq_length(self, layer_idx: int) -> int:
        entry = self.layers[layer_idx]
        if entry is None:
            return 0
        return entry[0].size(2)

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.layers[layer_idx]

    @torch.no_grad()
    def update(self, Pk_new: torch.Tensor, Pv_new: torch.Tensor, layer_idx: int):
        """Append new [B,H,S_new,r] along time dim=2."""
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


# =========================
# Low-rank GPT-2 Block (ASVD-capable)
# =========================
class LowRankSVDBlock(nn.Module):
    """
    GPT-2 block with per-head low-rank factors:
      - Q,K,V: U:[D,H,r], V:[H,r,dh], b:[H,dh]
      - out:   Uo:[D,ro], Vo:[ro,D], bo:[D]
      - FFN:   fc1 (D->I) low-rank, fc2 (I->D) low-rank

    If `asvd=True`, caching stores Pk= X@Uk and Pv= X@Uv with shape [B,H,T,r].
    We reconstruct dense K,V on the fly for attention math.
    """
    def __init__(
        self,
        hf_layer: nn.Module,
        rank_ratio_attn: float = 1.0,
        rank_ratio_mlp: float = 1.0,
        preload_factors: Optional[Dict[str, torch.Tensor]] = None,
        save_factors_to: Optional[str] = None,
        asvd: bool = False,
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
        self.scale = 1.0 / math.sqrt(dh)
        self.asvd = asvd

        dev = next(hf_layer.parameters()).device
        ptdtype = next(hf_layer.parameters()).dtype

        # ---------- ATTENTION (Q,K,V) ----------
        Wc_lin = as_linear_weight(hf_layer.attn.c_attn.weight.data, in_dim=D, out_dim=3 * D)  # [D,3D]
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

        # Initialize factors (per-head SVD) or preload
        if preload_factors is None:
            with torch.no_grad():
                for name, W_h in (("q", q_w), ("k", k_w), ("v", v_w)):
                    U_param = getattr(self, f"{name}_U")
                    V_param = getattr(self, f"{name}_V")
                    Us, Vs = [], []
                    for h in range(H):
                        Wh = W_h[:, h, :]                # [D, dh]
                        Uh, Vh = svd_factor(Wh, r_attn)  # [D,r], [r,dh]
                        Us.append(Uh.to(device=dev, dtype=ptdtype))
                        Vs.append(Vh.to(device=dev, dtype=ptdtype))
                    U = torch.stack(Us, dim=1)           # [D,H,r]
                    V = torch.stack(Vs, dim=0)           # [H,r,dh]
                    U_param.copy_(U)
                    V_param.copy_(V)
        else:
            self.load_factors_(preload_factors)

        # ---------- OUT PROJ ----------
        W_out_lin = as_linear_weight(hf_layer.attn.c_proj.weight.data, in_dim=D, out_dim=D)  # [D,D]
        b_out = hf_layer.attn.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_out = max(1, int(rank_ratio_attn * min(W_out_lin.shape)))
        Uo, Vo = svd_factor(W_out_lin, r_out)  # [D,r], [r,D]
        self.out_U = nn.Parameter(Uo.to(device=dev, dtype=ptdtype))
        self.out_V = nn.Parameter(Vo.to(device=dev, dtype=ptdtype))
        self.out_b = nn.Parameter(b_out)

        # ---------- MLP ----------
        I = hf_layer.mlp.c_fc.bias.data.numel()  # robust intermediate size

        W1_lin = as_linear_weight(hf_layer.mlp.c_fc.weight.data, in_dim=D, out_dim=I)
        b_fc1 = hf_layer.mlp.c_fc.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc1 = max(1, int(rank_ratio_mlp * min(W1_lin.shape)))
        U1, V1 = svd_factor(W1_lin, r_fc1)  # [D,r1], [r1,I]
        self.fc1_U = nn.Parameter(U1.to(device=dev, dtype=ptdtype))
        self.fc1_V = nn.Parameter(V1.to(device=dev, dtype=ptdtype))
        self.fc1_b = nn.Parameter(b_fc1)

        W2_lin = as_linear_weight(hf_layer.mlp.c_proj.weight.data, in_dim=I, out_dim=D)
        b_fc2 = hf_layer.mlp.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc2 = max(1, int(rank_ratio_mlp * min(W2_lin.shape)))
        U2, V2 = svd_factor(W2_lin, r_fc2)  # [I,r2], [r2,D]
        self.fc2_U = nn.Parameter(U2.to(device=dev, dtype=ptdtype))
        self.fc2_V = nn.Parameter(V2.to(device=dev, dtype=ptdtype))
        self.fc2_b = nn.Parameter(b_fc2)

        # Keep ranks for logs
        self.r_attn = r_attn
        self.r_out  = self.out_V.shape[0]
        self.r_fc1  = self.fc1_V.shape[0]
        self.r_fc2  = self.fc2_V.shape[0]

    # ---------------------
    # State I/O
    # ---------------------
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

    # ---------------------
    # Forward (+ ASVD or dense KV cache)
    # ---------------------
    def forward(
        self,
        hidden_states: torch.Tensor,                           # [B,S,D]
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # ASVD: (Pk,Pv) [B,H,T_past,r] ; Dense legacy: (K,V) [B,H,T_past,dh]
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        B, S, D = hidden_states.shape
        dev = hidden_states.device
        neg_inf = torch.finfo(hidden_states.dtype).min
        H, dh, r = self.H, self.dh, self.r_attn

        # LN1 + Q from low-rank
        x = self.ln1(hidden_states)  # [B,S,D]

        # Dense queries from low-rank factors
        Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]
        # Low-rank Q factors for Flash-SVD path
        Pq = torch.einsum('bsd,dhr->bhsr', x, self.q_U)  # [B,H,S,r]
        Vq = self.q_V.unsqueeze(0).expand(B, H, r, dh)   # [B,H,r,dh]
        bq = self.q_b.unsqueeze(0).expand(B, H, dh)      # [B,H,dh]
        
        # ---- K,V via ASVD factorization ----
        # New-step factors (Pk_new,Pv_new) and reconstructions (K_new,V_new)
        # Pk_new = x @ Uk ;  Pv_new = x @ Uv
        Pk_new = torch.einsum('bsd,dhr->bhsr', x, self.k_U)  # [B,H,S,r]
        Pv_new = torch.einsum('bsd,dhr->bhsr', x, self.v_U)  # [B,H,S,r]
        # Dense reconstructions (only needed for dense path)
        # K_new/V_new will be computed lazily only when required

        # Concatenate with past
        past_len = 0
        K_cat = V_cat = None
        Pk_cat = Pv_cat = None
        if layer_past is not None and isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
            past0, past1 = layer_past
            # ASVD path: past are Pk,Pv (last dim == r)
            if past0 is not None and past0.dim() == 4 and past0.size(-1) == r and self.asvd:
                Pk_cat = torch.cat([past0.to(Pk_new.dtype), Pk_new], dim=2)  # [B,H,T_total,r]
                Pv_cat = torch.cat([past1.to(Pv_new.dtype), Pv_new], dim=2)
                past_len = past0.size(2)
                # Reconstruct dense K,V for attention math
                K_cat = torch.einsum('bhtR,hRd->bhtd', Pk_cat, self.k_V) + self.k_b[None, :, None, :]
                V_cat = torch.einsum('bhtR,hRd->bhtd', Pv_cat, self.v_V) + self.v_b[None, :, None, :]
            else:
                # Legacy dense (if someone feeds it): past are K,V
                K_new  = torch.einsum('bhsr,hrd->bhsd', Pk_new, self.k_V) + self.k_b[None, :, None, :]  # [B,H,S,dh]
                V_new  = torch.einsum('bhsr,hrd->bhsd', Pv_new, self.v_V) + self.v_b[None, :, None, :]  # [B,H,S,dh]
                K_cat = torch.cat([past0.to(K_new.dtype), K_new], dim=2)
                V_cat = torch.cat([past1.to(V_new.dtype), V_new], dim=2)
                past_len = past0.size(2)
        else:
            if self.asvd:
                # No past, reconstruct dense K,V from current factors
                K_cat  = torch.einsum('bhsr,hrd->bhsd', Pk_new, self.k_V) + self.k_b[None, :, None, :]
                V_cat  = torch.einsum('bhsr,hrd->bhsd', Pv_new, self.v_V) + self.v_b[None, :, None, :]
                Pk_cat, Pv_cat = Pk_new, Pv_new
            else:
                K_cat  = torch.einsum('bhsr,hrd->bhsd', Pk_new, self.k_V) + self.k_b[None, :, None, :]
                V_cat  = torch.einsum('bhsr,hrd->bhsd', Pv_new, self.v_V) + self.v_b[None, :, None, :]

        total_len = past_len + S

        # Build per-query padding mask [B,H,1,S] expected by kernel
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

        # Use Flash-SVD attention only when there is no past (kernel assumes seq_len == kv_seq_len)
        # Otherwise fall back to dense FlashAttention with KV-cache
        if self.asvd and past_len == 0 and (Pk_cat is not None) and (Pv_cat is not None):
            Vk = self.k_V.unsqueeze(0).expand(B, H, r, dh)
            bk = self.k_b.unsqueeze(0).expand(B, H, dh)
            Vv = self.v_V.unsqueeze(0).expand(B, H, r, dh)
            bv = self.v_b.unsqueeze(0).expand(B, H, dh)
            Y_heads = flash_svd_attention(
                Pq, Vq, bq,
                Pk_cat, Vk, bk,
                Pv_cat, Vv, bv,
                mask=attn_mask_bh1s, block_r=r
            )
        else:
            Y_heads = flash_attn_triton_kvcache(Q, K_cat, V_cat, attn_mask_bh1s)
        Y = Y_heads.transpose(1, 2).contiguous().view(B, S, self.D)  # [B,S,D]
        attn_probs = None

        # Out projection
        Y = torch.matmul(torch.matmul(Y, self.out_U), self.out_V) + self.out_b
        hidden_states = hidden_states + Y

        # MLP
        z = self.ln2(hidden_states)
        t1 = torch.matmul(z, self.fc1_U)
        # FlashSVD FFN: produces final projection already
        h2 = flashsvd_ffn(t1, self.fc1_V, self.fc2_U, self.fc2_V, self.fc1_b, self.fc2_b)
        hidden_states = hidden_states + h2
        

        outputs = (hidden_states,)

        if use_cache:
            if self.asvd:
                # Return only the new-step factors for cache growth
                outputs = outputs + ((Pk_new, Pv_new),)
            else:
                # Non-ASVD mode: return dense K,V (new-step only),
                # so external caches that expect deltas can update correctly.
                outputs = outputs + ((K_new, V_new),)

        if output_attentions:
            outputs = outputs + (attn_probs,)

        return outputs


def _attach_asvd_cache_to_shims(model, asvd_cache):
    """
    Attach the same ASVDCache instance to every LayerShim so they can
    read/update it without routing through HF's past_key_values.
    """
    for layer in model.transformer.h:
        if hasattr(layer, "asvd") and getattr(layer, "asvd", False):
            # layer is a LayerShim in our build
            setattr(layer, "_asvd_cache", asvd_cache)


class LayerShim(nn.Module):
    """
    - If asvd=True: ignores top-level past_key_value; uses self._asvd_cache (ASVDCache).
    - Else: behaves like a thin wrapper passing dense deltas to DynamicCache.
    """
    def __init__(self, block: LowRankSVDBlock, layer_idx: int, asvd: bool):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.asvd = asvd
        self._asvd_cache = None  # set at runtime via _attach_asvd_cache_to_shims

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        layer_past = None

        if self.asvd:
            # Use the injected ASVDCache instead of HF past_key_values
            asvd_cache = getattr(self, "_asvd_cache", None)
            if isinstance(asvd_cache, ASVDCache):
                entry = asvd_cache.get(self.layer_idx)
                if entry is not None and asvd_cache.get_seq_length(self.layer_idx) > 0:
                    layer_past = entry  # (Pk_past, Pv_past) [B,H,T,r]
        else:
            # DynamicCache (HF) or legacy tuple
            if hasattr(past_key_value, "get_seq_length"):  # HF cache
                try:
                    seq_len = past_key_value.get_seq_length(self.layer_idx)
                except Exception:
                    seq_len = 0
                if seq_len and hasattr(past_key_value, "layers") and len(past_key_value.layers) > self.layer_idx:
                    # We let our block return (K_new,V_new); HF cache will update itself.
                    layer_past = None
            elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) == 2:
                layer_past = past_key_value  # legacy dense KV (not recommended)

        # Call block
        result = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=kwargs.get("use_cache", False),
            output_attentions=kwargs.get("output_attentions", False),
        )

        # Update caches
        if self.asvd:
            asvd_cache = getattr(self, "_asvd_cache", None)
            if (isinstance(asvd_cache, ASVDCache) and
                isinstance(result, tuple) and len(result) >= 2 and
                isinstance(result[1], (tuple, list)) and len(result[1]) == 2):
                Pk_new, Pv_new = result[1]  # [B,H,S_new,r]
                asvd_cache.update(Pk_new, Pv_new, self.layer_idx)
        else:
            if (hasattr(past_key_value, "update") and
                isinstance(result, tuple) and len(result) >= 2 and
                isinstance(result[1], (tuple, list)) and len(result[1]) == 2):
                k_new, v_new = result[1]
                past_key_value.update(k_new, v_new, self.layer_idx)

        return result


# =========================
# Dense GPT-2 Block with FlashAttention kernel
# =========================
class DenseFlashBlock(nn.Module):
    def __init__(self, hf_layer: nn.Module):
        super().__init__()
        attn = hf_layer.attn
        self.hf_attn = attn
        self.ln1 = hf_layer.ln_1
        self.ln2 = hf_layer.ln_2
        self.mlp = hf_layer.mlp

        D = attn.embed_dim
        H = attn.num_heads
        if D % H != 0:
            raise ValueError(f"[DenseFlashBlock] embed_dim={D} not divisible by heads={H}")
        dh = D // H

        self.D, self.H, self.dh = D, H, dh

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # [B,H,T_past,dh]
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        B, S, D = hidden_states.shape
        dev = hidden_states.device

        x = self.ln1(hidden_states)
        qkv = self.hf_attn.c_attn(x)  # [B,S,3D]
        q, k, v = qkv.split(self.D, dim=-1)
        # to [B,H,S,dh]
        Q = q.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous()
        K = k.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous()
        V = v.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous()

        past_len = 0
        if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2 and layer_past[0] is not None:
            past_k, past_v = layer_past
            if past_k.dtype != K.dtype: past_k = past_k.to(dtype=K.dtype)
            if past_v.dtype != V.dtype: past_v = past_v.to(dtype=V.dtype)
            if past_k.device != K.device: past_k = past_k.to(K.device)
            if past_v.device != V.device: past_v = past_v.to(V.device)
            K_cat = torch.cat([past_k, K], dim=2)
            V_cat = torch.cat([past_v, V], dim=2)
            past_len = past_k.size(2)
        else:
            K_cat, V_cat = K, V

        # Build [B,H,1,S] query mask
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
        attn_mask_bh1s = q_mask.expand(B, self.H, 1, S).contiguous()

        # FlashAttention kernel
        Y_heads = flash_attn_triton_kvcache(Q, K_cat, V_cat, attn_mask_bh1s)  # [B,H,S,dh]
        Y = Y_heads.transpose(1, 2).contiguous().view(B, S, D)

        # Output proj + residual
        Y = self.hf_attn.c_proj(Y)
        hidden_states = hidden_states + Y

        # FFN + residual
        z = self.ln2(hidden_states)
        h2 = self.mlp(z)
        hidden_states = hidden_states + h2

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + ((K, V),)
        if output_attentions:
            outputs = outputs + (None,)
        return outputs

def enable_flashattention_for_dense(model: GPT2LMHeadModel) -> None:
    for i, layer in enumerate(model.transformer.h):
        model.transformer.h[i] = LayerShim(DenseFlashBlock(layer), layer_idx=i, asvd=False)


# =========================
# Builders & Validators
# =========================
def build_svd_model(
    rank_ratio_attn: float,
    rank_ratio_mlp: float,
    save_factors_dir: Optional[str] = None,
    load_factors_dir: Optional[str] = None,
    device: Optional[str] = None,
    asvd: bool = False,
) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if device:
        model = model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    for i, layer in enumerate(model.transformer.h):
        preload = None
        save_path = None
        if load_factors_dir is not None:
            fp = os.path.join(load_factors_dir, f"gpt2_block_{i}.pt")
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"Missing factors for block {i}: {fp}")
            preload = torch.load(fp, map_location="cpu")
        elif save_factors_dir is not None:
            save_path = os.path.join(save_factors_dir, f"gpt2_block_{i}.pt")

        blk = LowRankSVDBlock(
            layer,
            rank_ratio_attn=rank_ratio_attn,
            rank_ratio_mlp=rank_ratio_mlp,
            preload_factors=preload,
            save_factors_to=save_path,
            asvd=asvd,
        )
        shim = LayerShim(blk, layer_idx=i, asvd=asvd).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim

    # attach a tiny flag for downstream checks
    model._uses_asvd_cache = asvd
    return model


@torch.no_grad()
def compare_qkv_intermediate(dense_block: nn.Module, svd_block: LowRankSVDBlock, x: torch.Tensor):
    D, H, dh = svd_block.D, svd_block.H, svd_block.dh

    Wc_lin = as_linear_weight(dense_block.attn.c_attn.weight.data, in_dim=D, out_dim=3 * D)
    bc = dense_block.attn.c_attn.bias.data.to(device=x.device, dtype=x.dtype)

    qkv = x @ Wc_lin + bc  # [B,S,3D]
    q, k, v = qkv.split(D, dim=-1)
    q = q.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()
    k = k.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()
    v = v.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()

    Q = torch.einsum('bsd,dhr,hre->bhse', x, svd_block.q_U, svd_block.q_V) + svd_block.q_b[None, :, None, :]
    # Reconstruct K,V via ASVD for fair comparison
    Pk = torch.einsum('bsd,dhr->bhsr', x, svd_block.k_U)
    Pv = torch.einsum('bsd,dhr->bhsr', x, svd_block.v_U)
    K = torch.einsum('bhsr,hrd->bhsd', Pk, svd_block.k_V) + svd_block.k_b[None, :, None, :]
    V = torch.einsum('bhsr,hrd->bhsd', Pv, svd_block.v_V) + svd_block.v_b[None, :, None, :]

    def stats(name, A, B):
        md = (A - B).abs().max().item()
        rd = (A - B).norm() / (A.norm() + 1e-12)
        print(f"{name:>3}  max|Δ|={md:.8f}  rel={rd:.8f}")

    print("=== QKV intermediate comparison (ASVD reconstruction) ===")
    stats("Q", Q, q)
    stats("K", K, k)
    stats("V", V, v)


@torch.no_grad()
def end_to_end_validation(dense: GPT2LMHeadModel, svd: GPT2LMHeadModel, device: str, use_cache: bool = False):
    test_input_ids = torch.randint(0, 1000, (2, 16), device=device)
    attn = torch.ones_like(test_input_ids, device=device)

    o1 = dense(input_ids=test_input_ids, attention_mask=attn, use_cache=use_cache).logits

    if getattr(svd, "_uses_asvd_cache", False) and use_cache:
        past = ASVDCache(n_layers=len(svd.transformer.h))
        _attach_asvd_cache_to_shims(svd, past)
        o2 = svd(input_ids=test_input_ids, attention_mask=attn, use_cache=True).logits
    else:
        o2 = svd(input_ids=test_input_ids, attention_mask=attn, use_cache=use_cache).logits

    max_diff = (o1 - o2).abs().max().item()
    rel_diff = (o1 - o2).norm() / (o1.norm() + 1e-12)
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Relative difference:     {rel_diff:.8f}")
    ok = max_diff < 1e-1
    print("✓ Validation PASSED" if ok else "✗ Validation FAILED")
    return max_diff, rel_diff


# =========================
# Perplexity + Memory + Time (evaluation mode)
# =========================
@torch.no_grad()
def perplexity_peak_time(mdl: GPT2LMHeadModel, loader, device: str, use_mask: bool = True):
    mdl.eval()
    total_loss, total_tokens = 0.0, 0
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        chunk = 4
        B = batch["input_ids"].size(0)
        for i in range(0, B, chunk):
            sl = slice(i, min(i + chunk, B))
            ids = batch["input_ids"][sl]
            mask = batch["attention_mask"][sl]

            kwargs = dict(input_ids=ids, use_cache=False)
            if use_mask:
                kwargs["attention_mask"] = mask

            out = mdl(**kwargs)

            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = ids[..., 1:].contiguous()

            if use_mask:
                m = mask[..., 1:].contiguous().bool()
                if m.any():
                    valid_logits = shift_logits[m]
                    valid_labels = shift_labels[m]
                    loss = F.cross_entropy(valid_logits, valid_labels)
                    total_loss += loss.item() * m.sum().item()
                    total_tokens += m.sum().item()
                    del valid_logits, valid_labels, m
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_loss += loss.item() * shift_labels.numel()
                total_tokens += shift_labels.numel()

            del out, shift_logits, shift_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - start) * 1000.0 / len(loader)
    peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    return ppl, peak, ms_per_batch


# =========================
# Decoding-time KV-cache growth benchmark
# =========================
def _to_legacy_kv(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return past_key_values.to_legacy_cache()
        except Exception:
            return None
    return None

@torch.no_grad()
def estimate_kv_bytes(past_key_values: Union[ASVDCache, object]) -> int:
    """
    - ASVDCache: count bytes of Pk and Pv -> 2 * B*H*T*r * sizeof(dtype)
    - HF legacy or DynamicCache: fall back to dense K,V counting if convertible.
    """
    # ASVD
    if isinstance(past_key_values, ASVDCache):
        total = 0
        for entry in past_key_values.layers:
            if entry is None:
                continue
            Pk, Pv = entry  # [B,H,T,r]
            total += Pk.numel() * Pk.element_size()
            total += Pv.numel() * Pv.element_size()
        return total

    # Dense (legacy / DynamicCache)
    pkv = _to_legacy_kv(past_key_values)
    if pkv is None:
        return 0
    total = 0
    for layer_kv in pkv:
        if not isinstance(layer_kv, (tuple, list)) or len(layer_kv) != 2 or layer_kv[0] is None:
            continue
        k, v = layer_kv
        total += k.numel() * k.element_size()
        total += v.numel() * v.element_size()
    return total


@torch.no_grad()
def decode_once_with_cache(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    new_tokens: int,
    device: str,
    greedy: bool = True,
) -> Dict[str, float]:
    model.eval()
    B = input_ids.size(0)
    n_layers = len(model.transformer.h)

    use_asvd = bool(getattr(model, "_uses_asvd_cache", False))
    past = ASVDCache(n_layers) if use_asvd else None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Attach our ASVD cache to the shims (no HF past_key_values involved)
    if use_asvd:
        _attach_asvd_cache_to_shims(model, past)
        out = model(input_ids=input_ids, use_cache=True)  # no past_key_values
    else:
        out = model(input_ids=input_ids, use_cache=True)

    logits = out.logits
    generated = input_ids

    for _ in range(new_tokens):
        next_id = (logits[:, -1, :].argmax(-1, keepdim=True) if greedy
                   else torch.multinomial(F.softmax(logits[:, -1, :], -1), 1))
        if use_asvd:
            _attach_asvd_cache_to_shims(model, past)
            out = model(input_ids=next_id, use_cache=True)  # still no past_key_values
        else:
            out = model(input_ids=next_id, use_cache=True, past_key_values=out.past_key_values)
        logits = out.logits
        generated = torch.cat([generated, next_id], dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elap = time.perf_counter() - t0

    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    kv_bytes = estimate_kv_bytes(past if use_asvd else out.past_key_values)
    kv_mib = kv_bytes / (1024**2)
    toks_per_s = (B * max(new_tokens, 1)) / max(elap, 1e-6)

    return {
        "peak_alloc_MiB": peak_alloc,
        "est_KV_MiB": kv_mib,
        "toks_per_s": toks_per_s,
        "total_tokens": int(generated.size(1)),
    }


@torch.no_grad()
def decode_growth_curve(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
    prompt_len: int,
    curve_lens: List[int],
    label: str,
):
    print(f"\n=== Decoding-time KV-cache growth ({label}) ===")
    vocab = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 50257
    safe_hi = min(1000, vocab)
    prompt = torch.randint(0, safe_hi, (batch_size, prompt_len), device=device)

    header = f"{'new_T':>8} | {'peak_alloc(MiB)':>16} | {'est_KV(MiB)':>12} | {'toks/s':>10} | {'total_T':>8} | {'param(MiB)':>11} | {'act_est(MiB)':>13} | {'act_wo_kv(MiB)':>16}"
    print(header)
    print("-" * len(header))
    for new_T in curve_lens:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        metrics = decode_once_with_cache(model, prompt, new_T, device)
        param_mib = compute_module_param_bytes(model)/(1024**2)
        act_est_mib = max(0.0, metrics['peak_alloc_MiB'] - param_mib)
        act_wo_kv_mib = max(0.0, metrics['peak_alloc_MiB'] - param_mib - metrics['est_KV_MiB'])
        print(f"{new_T:8d} | {metrics['peak_alloc_MiB']:16.1f} | {metrics['est_KV_MiB']:12.1f} | {metrics['toks_per_s']:10.2f} | {metrics['total_tokens']:8d} | {param_mib:11.1f} | {act_est_mib:13.1f} | {act_wo_kv_mib:16.1f}")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-ratio-attn", type=float, default=1.0)
    parser.add_argument("--rank-ratio-mlp",  type=float, default=1.0)
    parser.add_argument("--save-factors-dir", type=str, default=None)
    parser.add_argument("--load-factors-dir", type=str, default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--debug-attn", action="store_true")

    # Decoding mem benchmark
    parser.add_argument("--decode-mem", action="store_true")
    parser.add_argument("--decode-curve", type=str, default="64,128,256,512")
    parser.add_argument("--decode-batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--compare-dense", action="store_true")

    # ASVD
    parser.add_argument("--asvd", action="store_true", help="Use low-rank factor cache (Pk,Pv) for the SVD model")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dense baseline (reference)
    dense = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in dense.parameters(): p.requires_grad = False
    dense_mem = compute_persistent_memory(dense)
    print(f"Dense model storage: {dense_mem:6.1f} MiB")

    # SVD/ASVD model
    print("\n=== Building SVD Model ===")
    svd_model = build_svd_model(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        save_factors_dir=args.save_factors_dir,
        load_factors_dir=args.load_factors_dir,
        device=device,
        asvd=args.asvd,
    )
    for p in svd_model.parameters(): p.requires_grad = False
    print(f"SVD model built with per-head rank≈{args.rank_ratio_attn}*min(D,dh) and MLP ranks≈{args.rank_ratio_mlp}*...")

    first_blk = svd_model.transformer.h[0].block
    print(f"QKV rank: {first_blk.r_attn}, Out rank: {first_blk.r_out}")
    print(f"FC1 rank: {first_blk.r_fc1}, FC2 rank: {first_blk.r_fc2}")

    if args.debug_attn:
        blk0 = svd_model.transformer.h[0].block
        print(f"[debug-attn] D={blk0.D}, H={blk0.H}, dh={blk0.dh}, ASVD={blk0.asvd}")

    layer0 = dense.transformer.h[0]
    Wc = layer0.attn.c_attn.weight
    bc = layer0.attn.c_attn.bias
    print("weight", tuple(Wc.shape), "bias", tuple(bc.shape),
          "embed_dim", layer0.attn.embed_dim, "heads", layer0.attn.num_heads)

    # Optional validation
    if args.validate:
        print("\n=== SVD Validation ===")
        end_to_end_validation(dense, svd_model, device=device, use_cache=False)
        if args.validate_cache:
            print("\n--- With Cache ---")
            end_to_end_validation(dense, svd_model, device=device, use_cache=True)
        print("\n--- Block-0 Q/K/V check (ASVD recon) ---")
        with torch.no_grad():
            test_ids = torch.randint(0, 1000, (2, 16), device=device)
            wte = dense.transformer.wte(test_ids)
            pos = torch.arange(test_ids.size(1), device=device)[None, :]
            wpe = dense.transformer.wpe(pos)
            h0_in = dense.transformer.drop(wte + wpe)
            x0 = dense.transformer.h[0].ln_1(h0_in)
            compare_qkv_intermediate(dense.transformer.h[0], svd_model.transformer.h[0].block, x0)
        print("=== End Validation ===\n")

    # ===== Evaluation vs mem benchmark =====
    if not args.decode_mem:
        print("Preparing Wikitext-2 (test split)...")
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token

        def tok_fn(batch):
            return tok(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            )
        ds = raw.map(tok_fn, batched=True, remove_columns=["text"])
        ds.set_format("torch")
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: {
                "input_ids": torch.stack([x["input_ids"] for x in b]),
                "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            },
        )

        print("\n=== Dense baseline (FlashAttention) ===")
        # Offload SVD while profiling Dense
        svd_model = svd_model.to("cpu")
        dense = dense.to(device)
        enable_flashattention_for_dense(dense)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        ppl_m, peak_m, t_m = perplexity_peak_time(dense, loader, device, use_mask=True)
        print(f"Dense w/ mask   | ppl={ppl_m:.4f} | peak={peak_m:7.1f} MiB | {t_m:6.1f} ms/b")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        ppl_nm, peak_nm, t_nm = perplexity_peak_time(dense, loader, device, use_mask=False)
        print(f"Dense w/o mask  | ppl={ppl_nm:.4f} | peak={peak_nm:7.1f} MiB | {t_nm:6.1f} ms/b")

        print("\n=== SVD/ASVD model ===")
        # Offload Dense while profiling SVD/ASVD
        dense = dense.to("cpu")
        svd_model = svd_model.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        svd_mem = compute_persistent_memory(svd_model)
        print(f"SVD model storage: {svd_mem:6.1f} MiB "
              f"(saving {dense_mem - svd_mem:+.1f} MiB, {100*(dense_mem - svd_mem)/max(dense_mem,1e-9):.1f}%)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        ppl_m_s, peak_m_s, t_m_s = perplexity_peak_time(svd_model, loader, device, use_mask=True)
        print(f"SVD   w/ mask   | ppl={ppl_m_s:.4f} | peak={peak_m_s:7.1f} MiB | {t_m_s:6.1f} ms/b")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        ppl_nm_s, peak_nm_s, t_nm_s = perplexity_peak_time(svd_model, loader, device, use_mask=False)
        print(f"SVD   w/o mask  | ppl={ppl_nm_s:.4f} | peak={peak_nm_s:7.1f} MiB | {t_nm_s:6.1f} ms/b")

        print("\n=== Performance Summary ===")
        print(f"Storage (MiB): Dense={dense_mem:.1f} | SVD={svd_mem:.1f} | Δ={dense_mem - svd_mem:+.1f} ({100*(dense_mem - svd_mem)/max(dense_mem,1e-9):.1f}%)")
        print(f"Perplexity: dense w/={ppl_m:.4f} w/o={ppl_nm:.4f} | svd w/={ppl_m_s:.4f} w/o={ppl_nm_s:.4f}")
        print(f"Peak (MiB): dense w/={peak_m:7.1f} w/o={peak_nm:7.1f} | svd w/={peak_m_s:7.1f} w/o={peak_nm_s:7.1f}")
        print(f"Latency (ms/batch): dense w/={t_m:6.1f} w/o={t_nm:6.1f} | svd w/={t_m_s:6.1f} w/o={t_nm_s:6.1f}")

    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        curve = [int(x) for x in args.decode_curve.split(",") if x.strip()]
        bsz = args.decode_batch
        p_len = args.prompt_len

        # Offload Dense; run SVD/ASVD curve
        dense = dense.to("cpu")
        svd_model = svd_model.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        decode_growth_curve(
            svd_model, tok, device=device,
            batch_size=bsz, prompt_len=p_len, curve_lens=curve, label=("SVD-ASVD" if args.asvd else "SVD")
        )

        # Optionally compare dense baseline
        if args.compare_dense:
            # Offload SVD; run Dense curve
            svd_model = svd_model.to("cpu")
            dense = dense.to(device)
            enable_flashattention_for_dense(dense)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            decode_growth_curve(
                dense, tok, device=device,
                batch_size=bsz, prompt_len=p_len, curve_lens=curve, label="Dense"
            )


if __name__ == "__main__":
    main()

# Examples
# SVD w/ ASVD cache vs Dense:
# CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 python3 profile_asvd_accum_flashsvd.py --decode-mem --asvd --compare-dense --decode-batch 2 --prompt-len 64 --decode-curve 128,256 --rank-ratio-attn 0.5 --rank-ratio-mlp 0.5
