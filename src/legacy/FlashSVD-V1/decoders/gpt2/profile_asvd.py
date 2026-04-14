#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_asvd.py — GPT‑2 SVD blocks with ASVD-only cache

- Per-head SVD for Q/K/V (+ optional low-rank O/FF inside the block).
- ASVD cache: store only low‑rank factors (Pk, Pv) per layer and reconstruct dense K,V on the fly.
- Evaluation modes kept here focus on ASVD usage only:
  • Prefill perplexity on WikiText-2 (no cache used).
  • Decode-style perplexity on WikiText-2 (uses the ASVD cache path).

Dense baseline and decode-time token growth benchmarks are intentionally removed
from this script to keep it focused on ASVD-only evaluation.
"""

import os, math, time, itertools, argparse
from typing import Optional, Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer


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

        # Q is computed densely from low-rank factors (not cached)
        Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]  # [B,H,S,dh]

        # ---- K,V via ASVD factorization ----
        # New-step factors (Pk_new,Pv_new) and reconstructions (K_new,V_new)
        # Pk_new = x @ Uk ;  Pv_new = x @ Uv
        Pk_new = torch.einsum('bsd,dhr->bhsr', x, self.k_U)  # [B,H,S,r]
        Pv_new = torch.einsum('bsd,dhr->bhsr', x, self.v_U)  # [B,H,S,r]
        K_new  = torch.einsum('bhsr,hrd->bhsd', Pk_new, self.k_V) + self.k_b[None, :, None, :]  # [B,H,S,dh]
        V_new  = torch.einsum('bhsr,hrd->bhsd', Pv_new, self.v_V) + self.v_b[None, :, None, :]  # [B,H,S,dh]

        # Concatenate with past
        past_len = 0
        if layer_past is not None and isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
            past0, past1 = layer_past
            # ASVD path: past are Pk,Pv
            if past0 is not None and past0.dim() == 4 and past0.size(-1) == r:
                Pk_cat = torch.cat([past0.to(Pk_new.dtype), Pk_new], dim=2)  # [B,H,T_total,r]
                Pv_cat = torch.cat([past1.to(Pv_new.dtype), Pv_new], dim=2)
                past_len = past0.size(2)
                # Reconstruct dense K,V for attention math
                K_cat = torch.einsum('bhtR,hRd->bhtd', Pk_cat, self.k_V) + self.k_b[None, :, None, :]
                V_cat = torch.einsum('bhtR,hRd->bhtd', Pv_cat, self.v_V) + self.v_b[None, :, None, :]
            # Legacy dense (if someone feeds it): past are K,V
            else:
                K_cat = torch.cat([past0.to(K_new.dtype), K_new], dim=2)
                V_cat = torch.cat([past1.to(V_new.dtype), V_new], dim=2)
                past_len = past0.size(2)
        else:
            K_cat, V_cat = K_new, V_new

        total_len = past_len + S

        # Attention scores: [B,H,S,total_len]
        attn_scores = torch.matmul(Q, K_cat.transpose(-2, -1)) * self.scale

        # Causal + external mask
        causal = make_causal_slice_mask(S, total_len, device=dev, dtype=torch.bool)  # [S,total_len]
        attn_scores = attn_scores.masked_fill(~causal[None, None, :, :], neg_inf)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                am = attention_mask[..., -total_len:]
                if am.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    attn_scores = attn_scores + am.to(dtype=attn_scores.dtype)
                else:
                    key_keep = am.bool()
                    attn_scores = attn_scores.masked_fill(~key_keep, neg_inf)
            elif attention_mask.dim() == 2:
                if attention_mask.size(-1) == total_len:
                    key_keep = attention_mask[:, None, None, :].bool()
                elif attention_mask.size(-1) == S:
                    pad = torch.ones(B, past_len, dtype=attention_mask.dtype, device=dev)
                    key_keep = torch.cat([pad, attention_mask], dim=-1)[:, None, None, :].bool()
                else:
                    key_keep = torch.ones(B, 1, 1, total_len, dtype=torch.bool, device=dev)
                attn_scores = attn_scores.masked_fill(~key_keep, neg_inf)

        attn_probs = F.softmax(attn_scores, dim=-1)
        Y = torch.matmul(attn_probs, V_cat)          # [B,H,S,dh]
        Y = Y.transpose(1, 2).contiguous().view(B, S, self.D)  # [B,S,D]

        # Out projection
        Y = torch.matmul(torch.matmul(Y, self.out_U), self.out_V) + self.out_b
        hidden_states = hidden_states + Y

        # MLP
        z = self.ln2(hidden_states)
        t1 = torch.matmul(z, self.fc1_U)
        h1 = torch.matmul(t1, self.fc1_V) + self.fc1_b
        h1 = F.gelu(h1)
        t2 = torch.matmul(h1, self.fc2_U)
        h2 = torch.matmul(t2, self.fc2_V) + self.fc2_b
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
    """Attach shared ASVDCache to all shims in the stack.

    Earlier versions keyed on a boolean attribute that no longer exists after
    simplifying the shim. Now attach whenever the layer exposes `_asvd_cache`
    or is an instance of our LayerShim.
    """
    for layer in model.transformer.h:
        if isinstance(layer, LayerShim) or hasattr(layer, "_asvd_cache"):
            setattr(layer, "_asvd_cache", asvd_cache)


class LayerShim(nn.Module):
    """
    ASVD-only shim:
    - Ignores HF past_key_values; uses an injected ASVDCache shared across layers.
    """
    def __init__(self, block: LowRankSVDBlock, layer_idx: int):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self._asvd_cache = None  # set at runtime via _attach_asvd_cache_to_shims

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        layer_past = None

        # Use the injected ASVDCache instead of HF past_key_values
        asvd_cache = getattr(self, "_asvd_cache", None)
        if isinstance(asvd_cache, ASVDCache):
            entry = asvd_cache.get(self.layer_idx)
            if entry is not None and asvd_cache.get_seq_length(self.layer_idx) > 0:
                layer_past = entry  # (Pk_past, Pv_past) [B,H,T,r]

        # Call block
        result = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=kwargs.get("output_attentions", False),
        )

        # Update ASVD cache with new-step factors
        if (isinstance(asvd_cache, ASVDCache) and
            isinstance(result, tuple) and len(result) >= 2 and
            isinstance(result[1], (tuple, list)) and len(result[1]) == 2):
            Pk_new, Pv_new = result[1]  # [B,H,S_new,r]
            asvd_cache.update(Pk_new, Pv_new, self.layer_idx)

        return result



# =========================
# Builders & Validators
# =========================
def build_svd_model(
    rank_ratio_attn: float,
    rank_ratio_mlp: float,
    save_factors_dir: Optional[str] = None,
    load_factors_dir: Optional[str] = None,
    device: Optional[str] = None,
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
            asvd=True,
        )
        shim = LayerShim(blk, layer_idx=i).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim

    # attach a tiny flag for downstream checks
    model._uses_asvd_cache = True
    return model


@torch.no_grad()
def end_to_end_validation():
    """Removed: dense/SVD cross checks are out of scope for ASVD-only script."""
    raise NotImplementedError


# =========================
# Perplexity (evaluation mode)
# =========================
@torch.no_grad()
def perplexity_prefill(
    mdl: GPT2LMHeadModel,
    loader,
    device: str,
    *,
    use_mask: bool = True,
):
    """
    Prefill-style perplexity: single forward over entire sequence.
    Returns (ppl, peak_alloc_MiB, ms_per_batch).
    """
    mdl.eval()
    total_loss, total_tokens = 0.0, 0
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        ids, mask = batch["input_ids"], batch["attention_mask"]

        out = mdl(input_ids=ids, attention_mask=mask if use_mask else None, use_cache=False)
        shift_logits = out.logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()

        if use_mask:
            m = mask[..., 1:].contiguous().bool()
            if m.any():
                loss = F.cross_entropy(shift_logits[m], shift_labels[m])
                total_loss += loss.item() * int(m.sum().item())
                total_tokens += int(m.sum().item())
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item() * shift_labels.numel()
            total_tokens += int(shift_labels.numel())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - t0) * 1000.0 / max(1, len(loader))
    peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    return ppl, peak, ms_per_batch


@torch.no_grad()
def estimate_kv_bytes_asvd(cache: ASVDCache) -> int:
    """Theoretical bytes = sum(numel * element_size) for Pk,Pv across layers."""
    total = 0
    for entry in cache.layers:
        if entry is None:
            continue
        Pk, Pv = entry
        total += Pk.numel() * Pk.element_size()
        total += Pv.numel() * Pv.element_size()
    return total


@torch.no_grad()
def measure_kv_bytes_asvd(cache: ASVDCache) -> int:
    """
    Measure live bytes of ASVD cache by summing unique underlying storages.
    This captures actual allocated bytes (including padding/alignment) rather than
    only tensor numel*element_size.
    """
    seen = set()
    total = 0

    def acc_storage(t: torch.Tensor):
        nonlocal total
        if not torch.is_tensor(t):
            return
        try:
            s = t.untyped_storage()
            key = (s.data_ptr(), int(s.nbytes()))
            if key not in seen:
                seen.add(key)
                total += key[1]
        except Exception:
            total += t.numel() * t.element_size()

    for entry in cache.layers:
        if entry is None:
            continue
        Pk, Pv = entry
        acc_storage(Pk)
        acc_storage(Pv)
    return total


@torch.no_grad()
def perplexity_decode_cached(
    mdl: GPT2LMHeadModel,
    loader,
    device: str,
    *,
    max_batches: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    Compute perplexity by incremental decode using cache.
    - If the model was built with ASVD (mdl._uses_asvd_cache=True), this exercises
      the low-rank (Pk,Pv) cache path. Otherwise it uses HF dense KV cache.
    - Only uses last-token logits each step; avoids building a full [B,S,V] tensor.

    Returns: (ppl, ms_per_batch)
    """
    mdl.eval()
    total_loss, total_tokens = 0.0, 0
    use_asvd = True

    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    # Track ASVD KV footprint across batches (average)
    kv_bytes_total = 0
    kv_batches = 0

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        B, S = ids.shape

        # absolute position ids [B,S]: 0..S-1 (right padding assumed)
        pos_ids_full = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

        past = ASVDCache(n_layers=len(mdl.transformer.h)) if use_asvd else None
        out = None

        # Iterate tokens: feed y_t, score y_{t+1}
        for t in range(0, S - 1):
            inp = ids[:, t:t+1]
            pos_step = pos_ids_full[:, t:t+1]
            _attach_asvd_cache_to_shims(mdl, past)
            out = mdl(input_ids=inp, position_ids=pos_step, use_cache=True)

            logits_last = out.logits[:, -1, :]  # [B,V]
            target = ids[:, t + 1]
            m = mask[:, t + 1].bool()
            if m.any():
                loss = F.cross_entropy(logits_last[m], target[m])
                total_loss += loss.item() * int(m.sum().item())
                total_tokens += int(m.sum().item())

        # measure ASVD KV size accumulated this batch
        if isinstance(past, ASVDCache):
            kv_bytes_total += float(measure_kv_bytes_asvd(past))
            kv_batches += 1

        # small cleanup between batches
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - t0) * 1000.0 / max(1, (max_batches if max_batches is not None else len(loader)))
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    kv_mib_avg = (kv_bytes_total / max(kv_batches, 1)) / (1024**2)
    return ppl, kv_mib_avg, peak_mib, ms_per_batch


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-ratio-attn", type=float, default=1.0)
    parser.add_argument("--rank-ratio-mlp",  type=float, default=1.0)
    parser.add_argument("--save-factors-dir", type=str, default=None)
    parser.add_argument("--load-factors-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit number of evaluation samples")
    parser.add_argument("--mode", type=str, choices=["prefill", "decode", "both"], default="both",
                        help="Evaluation mode: prefill (no cache), decode (ASVD cache), or both")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SVD/ASVD model
    print("\n=== Building SVD+ASVD Model ===")
    svd_model = build_svd_model(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        save_factors_dir=args.save_factors_dir,
        load_factors_dir=args.load_factors_dir,
        device=device,
    )
    for p in svd_model.parameters(): p.requires_grad = False
    print(f"SVD model built (ASVD cache) with per-head rank≈{args.rank_ratio_attn}*min(D,dh) and MLP ranks≈{args.rank_ratio_mlp}*...")

    first_blk = svd_model.transformer.h[0].block
    print(f"QKV rank: {first_blk.r_attn}, Out rank: {first_blk.r_out}")
    print(f"FC1 rank: {first_blk.r_fc1}, FC2 rank: {first_blk.r_fc2}")

    blk0 = svd_model.transformer.h[0].block
    print(f"[info] D={blk0.D}, H={blk0.H}, dh={blk0.dh}, ASVD={blk0.asvd}")

    # ===== Evaluation (ASVD only) =====
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
    if args.max_eval_samples is not None:
        ds = ds.select(range(min(len(ds), args.max_eval_samples)))
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
    svd_mem = compute_persistent_memory(svd_model)
    print(f"SVD+ASVD model storage: {svd_mem:6.1f} MiB")

    if args.mode in ("prefill", "both"):
        print("\n=== Prefill Perplexity ===")
        ppl_mask, peak_m, t_m = perplexity_prefill(svd_model, loader, device, use_mask=True)
        print(f"Prefill w/ mask | ppl={ppl_mask:.4f} | peak={peak_m:7.1f} MiB | {t_m:6.1f} ms/b")

        ppl_nom, peak_nm, t_nm = perplexity_prefill(svd_model, loader, device, use_mask=False)
        print(f"Prefill no mask | ppl={ppl_nom:.4f} | peak={peak_nm:7.1f} MiB | {t_nm:6.1f} ms/b")

    if args.mode in ("decode", "both"):
        print("\n=== Decode Perplexity (ASVD cache) ===")
        ppl_svd_dec, kv_mib_avg, peak_mib, t_svd_dec = perplexity_decode_cached(svd_model, loader, device)
        print(f"Decode (ASVD)   | ppl={ppl_svd_dec:.4f} | KV≈{kv_mib_avg:7.1f} MiB | peak={peak_mib:7.1f} MiB | {t_svd_dec:6.1f} ms/b")


if __name__ == "__main__":
    main()

# Example
# python3 profile_asvd.py --mode both --batch-size 1 --max-length 512 --max-eval-samples 8 --rank-ratio-attn 1.0 --rank-ratio-mlp 1.0
