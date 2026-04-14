#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_dense_flash_kv.py — Dense (full) KV-cache decode memory profiler with FlashAttention (no big logits)

What this version does:
- Dense baseline ONLY (no ASVD anywhere).
- Uses a custom DenseKVCache we control (per-layer dense K,V).
- Dense forward uses our FlashAttention kernel (flash_attn_triton_kvcache).
- **Avoids full [B,S,V] logits**: we call model.transformer() and apply lm_head only on the last position.
- Memory accounting:
    * Prefill vs Decode times (ms)
    * Prefill/Decode peaks (MiB) via torch.cuda.max_memory_allocated()
    * Decode post-step allocated peak (MiB) — grows with sequence length
    * Decode end-of-phase allocated (MiB) — grows with sequence length
    * KV_end (MiB): allocator-accurate storage bytes for KV (deduped storages)
- Aggressive freeing between phases to reduce transient pressure.

Usage:
CUDA_VISIBLE_DEVICES=0 \
python3 profile_dense_accum_flash.py --decode-batch 16 --prompt-len 256 \
  --decode-curve 128,256 --rounds 1 --kv-profile
"""

import time, argparse, statistics, gc
from typing import Optional, Dict, Tuple, List

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
      - poststep_peak_bytes: max(memory_allocated) right after each token integration
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
# Dense KV cache
# -------------------------
class DenseKVCache:
    """Per-layer dense K,V with shape [B,H,T,dh]."""
    def __init__(self, n_layers: int):
        self.layers: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
    def get_seq_length(self, layer_idx: int) -> int:
        entry = self.layers[layer_idx]
        return 0 if entry is None else entry[0].size(2)
    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.layers[layer_idx]
    @torch.no_grad()
    def update(self, K_new: torch.Tensor, V_new: torch.Tensor, layer_idx: int):
        assert K_new.dim() == 4 and V_new.dim() == 4, "K/V must be [B,H,S_new,dh]"
        entry = self.layers[layer_idx]
        if entry is None:
            self.layers[layer_idx] = (K_new, V_new)
        else:
            K, V = entry
            self.layers[layer_idx] = (
                torch.cat([K, K_new], dim=2),
                torch.cat([V, V_new], dim=2),
            )

# -------------------------
# Dense GPT-2 Block with FlashAttention kernel
# -------------------------
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
        self.profiler: Optional[KVProfiler] = None

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
        prof: Optional[KVProfiler] = self.profiler

        x = self.ln1(hidden_states)

        t0 = time.perf_counter()
        qkv = self.hf_attn.c_attn(x)  # [B,S,3D]
        q, k, v = qkv.split(self.D, dim=-1)
        del qkv

        Q = q.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous(); del q
        K = k.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous(); del k
        V = v.view(B, S, self.H, self.dh).permute(0, 2, 1, 3).contiguous(); del v

        if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2 and layer_past[0] is not None:
            past_k, past_v = layer_past
            if past_k.dtype != K.dtype: past_k = past_k.to(dtype=K.dtype)
            if past_v.dtype != V.dtype: past_v = past_v.to(dtype=V.dtype)
            if past_k.device != K.device: past_k = past_k.to(K.device)
            if past_v.device != V.device: past_v = past_v.to(V.device)
            K_cat = torch.cat([past_k, K], dim=2)
            V_cat = torch.cat([past_v, V], dim=2)
        else:
            K_cat, V_cat = K, V

        if prof:
            prof.add_time("qkv_s", time.perf_counter() - t0)
            prof.add_bytes("kv_read_bytes", (K_cat.numel() + V_cat.numel()) * K_cat.element_size())
            prof.add_bytes("kv_new_bytes", (K.numel() + V.numel()) * K.element_size())
            prof.inc_calls()

        # Mask (build compact BH1S; free asap)
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
        del q_mask

        tA0 = time.perf_counter()
        Y_heads = flash_attn_triton_kvcache(Q, K_cat, V_cat, attn_mask_bh1s)
        if prof: prof.add_time("attn_s", time.perf_counter() - tA0)

        # Free big attention inputs
        del Q, K_cat, V_cat, attn_mask_bh1s

        Y = Y_heads.transpose(1, 2).contiguous().view(B, S, D); del Y_heads

        # Output proj + residual
        Y = self.hf_attn.c_proj(Y)
        hidden_states = hidden_states.add(Y); del Y

        # FFN + residual
        z = self.ln2(hidden_states)
        h2 = self.mlp(z); del z
        hidden_states = hidden_states.add(h2); del h2

        outputs = (hidden_states,)
        if use_cache:
            # return ONLY the new-step K,V for cache
            outputs = outputs + ((K, V),)
        else:
            del K, V

        if output_attentions:
            outputs = outputs + (None,)
        return outputs

class LayerShim(nn.Module):
    def __init__(self, block: DenseFlashBlock, layer_idx: int):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self._dense_cache: Optional[DenseKVCache] = None
        self.profiler: Optional[KVProfiler] = None

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        use_cache_flag = bool(kwargs.get("use_cache", False))
        prof: Optional[KVProfiler] = self.profiler
        layer_past = None

        # use our DenseKVCache when enabled
        if use_cache_flag:
            dense_cache = getattr(self, "_dense_cache", None)
            if isinstance(dense_cache, DenseKVCache):
                entry = dense_cache.get(self.layer_idx)
                if entry is not None and dense_cache.get_seq_length(self.layer_idx) > 0:
                    layer_past = entry

        result = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache_flag,
            output_attentions=kwargs.get("output_attentions", False),
        )

        if use_cache_flag:
            dense_cache = getattr(self, "_dense_cache", None)
            if (isinstance(dense_cache, DenseKVCache) and
                isinstance(result, tuple) and len(result) >= 2 and
                isinstance(result[1], (tuple, list)) and len(result[1]) == 2):
                K_new, V_new = result[1]
                t0 = time.perf_counter()
                dense_cache.update(K_new, V_new, self.layer_idx)
                if prof: prof.add_time("update_s", time.perf_counter() - t0)
        return result

def _attach_dense_cache_to_shims(model, dense_cache: DenseKVCache):
    for layer in model.transformer.h:
        if isinstance(layer, LayerShim):
            setattr(layer, "_dense_cache", dense_cache)

# -------------------------
# Build Dense+FA model
# -------------------------
def build_dense_fa_model(device: Optional[str] = None) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    if device:
        model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model.transformer.h):
        shim = LayerShim(DenseFlashBlock(layer), layer_idx=i).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim
    model._uses_dense_kv = True
    return model

# -------------------------
# Accurate KV bytes (allocator storage)
# -------------------------
@torch.no_grad()
def estimate_kv_bytes_dense(cache: DenseKVCache) -> int:
    """
    Sum UNIQUE storage bytes of all dense K/V tensors in our DenseKVCache.
    Uses untyped_storage().nbytes() when available to capture allocator padding.
    """
    def storage_key_and_nbytes(t: torch.Tensor):
        try:
            s = t.untyped_storage()
            return (s.data_ptr(), int(s.nbytes()))
        except Exception:
            s = t.storage()
            nbytes = (s.nbytes() if hasattr(s, "nbytes") else s.size() * t.element_size())
            ptr = s.data_ptr() if hasattr(s, "data_ptr") else t.data_ptr()
            return (ptr, int(nbytes))

    seen = set()
    total = 0
    for entry in cache.layers:
        if entry is None:
            continue
        k, v = entry
        for t in (k, v):
            if t is None or not t.is_cuda:
                continue
            key = storage_key_and_nbytes(t)
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
# Decode benchmark (Dense+FA only) — no big logits
# -------------------------
@torch.no_grad()
def decode_benchmark_dense(
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

    # ---- Prefill (build cache, no [B,S,V] logits) ----
    if profiler: profiler.set_phase("prefill")
    kv = DenseKVCache(n_layers=len(model.transformer.h))
    _attach_dense_cache_to_shims(model, kv)

    t0 = time.perf_counter()
    out = model.transformer(input_ids=prompt, use_cache=True, return_dict=True)  # only hidden states
    if torch.cuda.is_available(): torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # choose first token from last position only
    next_id = _next_token_from_last_hidden(model, out.last_hidden_state, greedy=greedy)

    prefill_peak_mib = torch.cuda.max_memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    prefill_end_alloc_mib = torch.cuda.memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    _ = estimate_kv_bytes_dense(kv) / MiB  # optional: not printed here
    if profiler:
        profiler.add_mem_poststep(int(prefill_end_alloc_mib * MiB))
        profiler.set_end_alloc(int(prefill_end_alloc_mib * MiB))

    # free prefill outputs before measuring decode
    del out
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # ---- Decode ----
    if profiler: profiler.set_phase("decode")

    t_dec = 0.0
    decode_poststep_peak_mib = 0.0

    for _ in range(new_tokens):
        t1 = time.perf_counter()
        _attach_dense_cache_to_shims(model, kv)
        step = model.transformer(input_ids=next_id, use_cache=True, return_dict=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t_dec += (time.perf_counter() - t1)

        # pick next strictly from last position (avoid [B,S,V])
        next_id = _next_token_from_last_hidden(model, step.last_hidden_state, greedy=greedy)

        del step
        gc.collect()

        if torch.cuda.is_available():
            alloc_now_mib = torch.cuda.memory_allocated() / MiB
            if profiler: profiler.add_mem_poststep(int(alloc_now_mib * MiB))
            if alloc_now_mib > decode_poststep_peak_mib:
                decode_poststep_peak_mib = alloc_now_mib

    decode_peak_mib = torch.cuda.max_memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    decode_end_alloc_mib = torch.cuda.memory_allocated() / MiB if torch.cuda.is_available() else 0.0
    kv_final_mib = estimate_kv_bytes_dense(kv) / MiB
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
def decode_growth_curve_dense(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    device: str,
    batch_size: int,
    prompt_len: int,
    curve_lens: List[int],
    rounds: int = 5,
    kv_profile: bool = True,
):
    print(f"\n=== Decoding-time KV-cache growth (Dense+FlashAttn, last-token logits only) — {rounds} rounds avg ===")
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
            res = decode_benchmark_dense(model, prompt, new_T, device, profiler=prof, greedy=True)

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
    parser.add_argument("--seed", type=int, default=0)

    # Decode mem benchmark (Dense+FA only)
    parser.add_argument("--decode-curve", type=str, default="64,128,256,512")
    parser.add_argument("--decode-batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--kv-profile", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)

    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== Building Dense+FlashAttention Model ===")
    model = build_dense_fa_model(device=device)
    blk0 = model.transformer.h[0].block
    print(f"embed_dim={blk0.D}, heads={blk0.H}, dh={blk0.dh}")

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    curve = [int(x) for x in args.decode_curve.split(",") if x.strip()]
    bsz = args.decode_batch
    p_len = args.prompt_len

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    decode_growth_curve_dense(
        model, tok, device=device,
        batch_size=bsz, prompt_len=p_len, curve_lens=curve,
        rounds=args.rounds, kv_profile=args.kv_profile
    )

if __name__ == "__main__":
    main()
