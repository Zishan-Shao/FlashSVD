#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_asvd_eval.py — Evaluation-only SVD variant (no KV-cache growth)

What changed vs your original:
- Removed all decoding/KV-growth benchmarking code.
- Removed ASVDCache/LayerShim and any past_key_values handling.
- Model is always invoked with use_cache=False during eval.
- Forward never returns cache deltas; no cache can accumulate.
- Kept per-head low-rank SVD for Q,K,V (reconstruct K,V on-the-fly), low-rank out proj, and low-rank MLP.

Usage examples:
  python3 profile_asvd_eval.py --rank-ratio-attn 1.0 --rank-ratio-mlp 1.0 --batch-size 8 --max-length 256
  python3 profile_asvd_eval.py --validate
"""

import os, math, time, itertools, argparse
from typing import Optional, Dict, Tuple, List

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

def as_linear_weight(W_raw: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    if W_raw.shape == (in_dim, out_dim):
        return W_raw.contiguous()
    if W_raw.shape == (out_dim, in_dim):
        return W_raw.t().contiguous()
    raise ValueError(f"Unexpected weight shape {tuple(W_raw.shape)}; expected ({in_dim},{out_dim}) or ({out_dim},{in_dim}).")


# =========================
# Low-rank GPT-2 Block (no cache support)
# =========================
class LowRankSVDBlock(nn.Module):
    """
    GPT-2 block with per-head low-rank factors:
      - Q,K,V: U:[D,H,r], V:[H,r,dh], b:[H,dh]
      - out:   Uo:[D,ro], Vo:[ro,D], bo:[D]
      - FFN:   fc1 (D->I) low-rank, fc2 (I->D) low-rank

    Evaluation-only: reconstruct K,V on the fly; no KV cache is created or returned.
    """
    def __init__(
        self,
        hf_layer: nn.Module,
        rank_ratio_attn: float = 1.0,
        rank_ratio_mlp: float = 1.0,
        preload_factors: Optional[Dict[str, torch.Tensor]] = None,
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
    # Forward (no cache)
    # ---------------------
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Eval-only block. Ignores all HF extras (past, head_mask, encoder_*, use_cache, etc.).
        We only look at `attention_mask` if provided.
        """
        attention_mask: Optional[torch.Tensor] = kwargs.get("attention_mask", None)

        B, S, D = hidden_states.shape
        dev = hidden_states.device
        H, dh = self.H, self.dh
        neg_inf = torch.finfo(hidden_states.dtype).min

        # LN1 + Q from low-rank
        x = self.ln1(hidden_states)  # [B,S,D]
        Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]  # [B,H,S,dh]

        # K,V via low-rank (reconstructed on-the-fly)
        Pk = torch.einsum('bsd,dhr->bhsr', x, self.k_U)  # [B,H,S,r]
        Pv = torch.einsum('bsd,dhr->bhsr', x, self.v_U)  # [B,H,S,r]
        K  = torch.einsum('bhsr,hrd->bhsd', Pk, self.k_V) + self.k_b[None, :, None, :]  # [B,H,S,dh]
        V  = torch.einsum('bhsr,hrd->bhsd', Pv, self.v_V) + self.v_b[None, :, None, :]  # [B,H,S,dh]

        # Attention scores: [B,H,S,S]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(dh))

        # Causal mask
        causal = torch.ones(S, S, dtype=torch.bool, device=dev).tril_()
        attn_scores = attn_scores.masked_fill(~causal[None, None, :, :], neg_inf)

        # Optional external mask (padding)
        if attention_mask is not None:
            if attention_mask.dim() == 2 and attention_mask.size(-1) == S:
                key_keep = attention_mask[:, None, None, :].bool()
                attn_scores = attn_scores.masked_fill(~key_keep, neg_inf)
            elif attention_mask.dim() == 4 and attention_mask.size(-1) == S:
                if attention_mask.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                    attn_scores = attn_scores + attention_mask.to(dtype=attn_scores.dtype)
                else:
                    attn_scores = attn_scores.masked_fill(~attention_mask.bool(), neg_inf)

        attn_probs = F.softmax(attn_scores, dim=-1)
        Y = torch.matmul(attn_probs, V)          # [B,H,S,dh]
        Y = Y.transpose(1, 2).contiguous().view(B, S, D)  # [B,S,D]

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

        return (hidden_states,)


# =========================
# Builders & Validators
# =========================
def build_svd_model(
    rank_ratio_attn: float,
    rank_ratio_mlp: float,
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
        if load_factors_dir is not None:
            fp = os.path.join(load_factors_dir, f"gpt2_block_{i}.pt")
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"Missing factors for block {i}: {fp}")
            preload = torch.load(fp, map_location="cpu")

        blk = LowRankSVDBlock(
            layer,
            rank_ratio_attn=rank_ratio_attn,
            rank_ratio_mlp=rank_ratio_mlp,
            preload_factors=preload,
        ).to(device if device is not None else next(model.parameters()).device)
        # Replace the HF block with our low-rank block (same call signature for forward pass in eval)
        model.transformer.h[i] = blk

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
    Pk = torch.einsum('bsd,dhr->bhsr', x, svd_block.k_U)
    Pv = torch.einsum('bsd,dhr->bhsr', x, svd_block.v_U)
    K = torch.einsum('bhsr,hrd->bhsd', Pk, svd_block.k_V) + svd_block.k_b[None, :, None, :]
    V = torch.einsum('bhsr,hrd->bhsd', Pv, svd_block.v_V) + svd_block.v_b[None, :, None, :]

    def stats(name, A, B):
        md = (A - B).abs().max().item()
        rd = (A - B).norm() / (A.norm() + 1e-12)
        print(f"{name:>3}  max|Δ|={md:.8f}  rel={rd:.8f}")

    print("=== QKV intermediate comparison (low-rank reconstruction) ===")
    stats("Q", Q, q)
    stats("K", K, k)
    stats("V", V, v)


@torch.no_grad()
def end_to_end_validation(dense: GPT2LMHeadModel, svd: GPT2LMHeadModel, device: str):
    test_input_ids = torch.randint(0, 1000, (2, 16), device=device)
    attn = torch.ones_like(test_input_ids, device=device)

    # Always no-cache for eval-only build
    o1 = dense(input_ids=test_input_ids, attention_mask=attn, use_cache=False).logits
    o2 = svd(input_ids=test_input_ids, attention_mask=attn, use_cache=False).logits

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

            # EVAL ONLY: never cache
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
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-ratio-attn", type=float, default=1.0)
    parser.add_argument("--rank-ratio-mlp",  type=float, default=1.0)
    parser.add_argument("--load-factors-dir", type=str, default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--debug-attn", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dense baseline (reference)
    dense = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in dense.parameters(): p.requires_grad = False
    dense_mem = compute_persistent_memory(dense)
    print(f"Dense model storage: {dense_mem:6.1f} MiB")

    # SVD model
    print("\n=== Building SVD Model (eval-only) ===")
    svd_model = build_svd_model(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        load_factors_dir=args.load_factors_dir,
        device=device,
    )
    for p in svd_model.parameters(): p.requires_grad = False
    print(f"SVD model built with per-head rank≈{args.rank_ratio_attn}*min(D,dh) and MLP ranks≈{args.rank_ratio_mlp}*...")

    first_blk = svd_model.transformer.h[0]
    print(f"QKV rank: {first_blk.r_attn}, Out rank: {first_blk.r_out}")
    print(f"FC1 rank: {first_blk.r_fc1}, FC2 rank: {first_blk.r_fc2}")

    if args.debug_attn:
        blk0 = svd_model.transformer.h[0]
        print(f"[debug-attn] D={blk0.D}, H={blk0.H}, dh={blk0.dh}")

    layer0 = dense.transformer.h[0]
    Wc = layer0.attn.c_attn.weight
    bc = layer0.attn.c_attn.bias
    print("weight", tuple(Wc.shape), "bias", tuple(bc.shape),
          "embed_dim", layer0.attn.embed_dim, "heads", layer0.attn.num_heads)

    # Optional validation (no cache)
    if args.validate:
        print("\n=== SVD Validation (no-cache) ===")
        end_to_end_validation(dense, svd_model, device=device)
        print("\n--- Block-0 Q/K/V check (reconstruction) ---")
        with torch.no_grad():
            test_ids = torch.randint(0, 1000, (2, 16), device=device)
            wte = dense.transformer.wte(test_ids)
            pos = torch.arange(test_ids.size(1), device=device)[None, :]
            wpe = dense.transformer.wpe(pos)
            h0_in = dense.transformer.drop(wte + wpe)
            x0 = dense.transformer.h[0].ln_1(h0_in)
            compare_qkv_intermediate(dense.transformer.h[0], svd_model.transformer.h[0], x0)
        print("=== End Validation ===\n")

    # ===== Evaluation on Wikitext-2 =====
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

    print("\n=== Dense baseline ===")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    ppl_m, peak_m, t_m = perplexity_peak_time(dense, loader, device, use_mask=True)
    print(f"Dense w/ mask   | ppl={ppl_m:.4f} | peak={peak_m:7.1f} MiB | {t_m:6.1f} ms/b")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    ppl_nm, peak_nm, t_nm = perplexity_peak_time(dense, loader, device, use_mask=False)
    print(f"Dense w/o mask  | ppl={ppl_nm:.4f} | peak={peak_nm:7.1f} MiB | {t_nm:6.1f} ms/b")

    print("\n=== SVD model (eval-only) ===")
    svd_mem = compute_persistent_memory(svd_model)
    print(f"SVD model storage: {svd_mem:6.1f} MiB "
          f"(saving {dense_mem - svd_mem:+.1f} MiB, {100*(dense_mem - svd_mem)/max(dense_mem,1e-9):.1f}%)")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    ppl_m_s, peak_m_s, t_m_s = perplexity_peak_time(svd_model, loader, device, use_mask=True)
    print(f"SVD   w/ mask   | ppl={ppl_m_s:.4f} | peak={peak_m_s:7.1f} MiB | {t_m_s:6.1f} ms/b")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    ppl_nm_s, peak_nm_s, t_nm_s = perplexity_peak_time(svd_model, loader, device, use_mask=False)
    print(f"SVD   w/o mask  | ppl={ppl_nm_s:.4f} | peak={peak_nm_s:7.1f} MiB | {t_nm_s:6.1f} ms/b")

    print("\n=== Performance Summary ===")
    print(f"Storage (MiB): Dense={dense_mem:.1f} | SVD={svd_mem:.1f} | Δ={dense_mem - svd_mem:+.1f} ({100*(dense_mem - svd_mem)/max(dense_mem,1e-9):.1f}%)")
    print(f"Perplexity: dense w/={ppl_m:.4f} w/o={ppl_nm:.4f} | svd w/={ppl_m_s:.4f} w/o={ppl_nm_s:.4f}")
    print(f"Peak (MiB): dense w/={peak_m:7.1f} w/o={peak_nm:7.1f} | svd w/={peak_m_s:7.1f} w/o={peak_nm_s:7.1f}")
    print(f"Latency (ms/batch): dense w/={t_m:6.1f} w/o={t_nm:6.1f} | svd w/={t_m_s:6.1f} w/o={t_nm_s:6.1f}")


if __name__ == "__main__":
    main()
