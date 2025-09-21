#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_svd_linear.py
Per-head low-rank SVD factorization for GPT-2 with correctness checks and profiling.

- Q,K,V per head: U:[D,H,r], V:[H,r,dh], b:[H,dh]
- Out proj low-rank
- FFN (c_fc, c_proj) low-rank
- Optional factor save/load (no SVD at runtime if loaded)
- Q/K/V intermediate comparison + end-to-end validation vs dense
- Perplexity + peak memory + latency on Wikitext-2

Usage:
  python3 profile_svd_linear.py --rank-ratio-attn 1.0 --rank-ratio-mlp 1.0 --validate --debug-attn
"""

import os, sys, math, json, time, itertools, argparse
from typing import Optional, Dict, Any, Tuple

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
    """
    Factor W ≈ U @ V with target rank r.
    Returns float32 factors by default; call .to(device=..., dtype=...) after.
    """
    if W.dtype not in (torch.float32, torch.float64):
        W = W.float()
    W = W.contiguous()
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except TypeError:
        U_, S_, V_ = torch.svd(W)
        U, S, Vh = U_, S_, V_.t()

    r = min(rank, S.numel())
    U_r = U[:, :r].contiguous()                    # [M, r]
    V_r = (S[:r, None] * Vh[:r, :]).contiguous()   # [r, N]
    return U_r, V_r

def make_causal_mask(S: int, device, dtype=torch.bool) -> torch.Tensor:
    return torch.ones(S, S, dtype=dtype, device=device).tril_()

def as_linear_weight(W_raw: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    """
    Normalize a weight tensor to the 'linear form' used as x @ W_lin (+ b),
    i.e., W_lin has shape [in_dim, out_dim].

    Accepts either storage layout:
      - [out_dim, in_dim] (Conv1D/Linear weight) -> returns W_raw.t()
      - [in_dim, out_dim] (column-major style)   -> returns W_raw
    """
    if W_raw.shape == (in_dim, out_dim):
        return W_raw.contiguous()
    if W_raw.shape == (out_dim, in_dim):
        return W_raw.t().contiguous()
    raise ValueError(f"Unexpected weight shape {tuple(W_raw.shape)}; expected ({in_dim},{out_dim}) or ({out_dim},{in_dim}).")


# =========================
# Low-rank GPT-2 Block
# =========================
class LowRankSVDBlock(nn.Module):
    """
    GPT-2 block with per-head low-rank factors:
      - Q,K,V: U:[D,H,r], V:[H,r,dh], b:[H,dh]
      - out:   Uo:[D,ro], Vo:[ro,D], bo:[D]
      - FFN:   fc1 (W1_lin:[D,I]): U1:[D,r1], V1:[r1,I], b1:[I]
               fc2 (W2_lin:[I,D]): U2:[I,r2], V2:[r2,D], b2:[D]
    where D=hidden_size, H=num_heads, dh=D//H, I=intermediate size.
    """
    def __init__(
        self,
        hf_layer: nn.Module,
        rank_ratio_attn: float = 1.0,
        rank_ratio_mlp: float = 1.0,
        preload_factors: Optional[Dict[str, torch.Tensor]] = None,
        save_factors_to: Optional[str] = None,
    ):
        super().__init__()
        attn = hf_layer.attn
        self.ln1 = hf_layer.ln_1
        self.ln2 = hf_layer.ln_2

        # Derive dims from the attention module
        D_cfg = attn.embed_dim
        H = attn.num_heads
        if D_cfg % H != 0:
            raise ValueError(f"[LowRankSVDBlock] embed_dim={D_cfg} not divisible by heads={H}")
        dh = D_cfg // H

        self.D, self.H, self.dh = D_cfg, H, dh
        self.scale = 1.0 / math.sqrt(dh)

        # Robust device/dtype anchor
        dev = next(hf_layer.parameters()).device
        ptdtype = next(hf_layer.parameters()).dtype

        # ---------- ATTENTION (Q,K,V) ----------
        # Normalize to linear weight Wc_lin:[D, 3D]
        Wc_raw = hf_layer.attn.c_attn.weight.data
        bc = hf_layer.attn.c_attn.bias.data.clone().to(device=dev, dtype=ptdtype)
        Wc_lin = as_linear_weight(Wc_raw, in_dim=self.D, out_dim=3 * self.D)  # [D,3D]

        # Split to [D,D] each, and reshape to [D,H,dh]
        q_w = Wc_lin[:, :self.D].contiguous().view(self.D, self.H, dh)
        k_w = Wc_lin[:, self.D:2*self.D].contiguous().view(self.D, self.H, dh)
        v_w = Wc_lin[:, 2*self.D:3*self.D].contiguous().view(self.D, self.H, dh)

        q_b = bc[:self.D].view(self.H, dh).contiguous()
        k_b = bc[self.D:2*self.D].view(self.H, dh).contiguous()
        v_b = bc[2*self.D:3*self.D].view(self.H, dh).contiguous()

        r_attn = max(1, int(rank_ratio_attn * min(self.D, dh)))

        # Alloc parameters
        def alloc_uv(name: str):
            U = nn.Parameter(torch.empty(self.D, self.H, r_attn, device=dev, dtype=ptdtype))
            V = nn.Parameter(torch.empty(self.H, r_attn, dh, device=dev, dtype=ptdtype))
            self.register_parameter(f"{name}_U", U)
            self.register_parameter(f"{name}_V", V)
            return U, V

        self.q_U, self.q_V = alloc_uv("q")
        self.k_U, self.k_V = alloc_uv("k")
        self.v_U, self.v_V = alloc_uv("v")

        self.q_b = nn.Parameter(q_b.to(device=dev, dtype=ptdtype))
        self.k_b = nn.Parameter(k_b.to(device=dev, dtype=ptdtype))
        self.v_b = nn.Parameter(v_b.to(device=dev, dtype=ptdtype))

        # Initialize factors
        if preload_factors is None:
            with torch.no_grad():
                for name, W_h in (("q", q_w), ("k", k_w), ("v", v_w)):
                    U_param = getattr(self, f"{name}_U")
                    V_param = getattr(self, f"{name}_V")
                    Us, Vs = [], []
                    for h in range(self.H):
                        Wh = W_h[:, h, :]                # [D, dh]
                        Uh, Vh = svd_factor(Wh, r_attn)  # Uh:[D,r], Vh:[r,dh]
                        Us.append(Uh.to(device=dev, dtype=ptdtype))
                        Vs.append(Vh.to(device=dev, dtype=ptdtype))
                    U = torch.stack(Us, dim=1)           # [D,H,r]
                    V = torch.stack(Vs, dim=0)           # [H,r,dh]
                    U_param.copy_(U)
                    V_param.copy_(V)
        else:
            self.load_factors_(preload_factors)

        # ---------- OUT PROJ ----------
        # Normalize to linear W_out_lin:[D,D] (used as Y @ W_out_lin + b)
        W_out_raw = hf_layer.attn.c_proj.weight.data
        W_out_lin = as_linear_weight(W_out_raw, in_dim=self.D, out_dim=self.D)  # [D,D]
        b_out = hf_layer.attn.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)

        r_out = max(1, int(rank_ratio_attn * min(W_out_lin.shape)))  # == int(rank_ratio_attn * D)
        Uo, Vo = svd_factor(W_out_lin, r_out)  # Uo:[D,r], Vo:[r,D]

        self.out_U = nn.Parameter(Uo.to(device=dev, dtype=ptdtype))
        self.out_V = nn.Parameter(Vo.to(device=dev, dtype=ptdtype))
        self.out_b = nn.Parameter(b_out)

        # ---------- MLP ----------
        # Use bias size to infer I robustly
        I = hf_layer.mlp.c_fc.bias.data.numel()  # 4*D typically

        # FC1: z:[B,S,D], W1_lin:[D,I]  => z @ W1_lin + b1
        W1_raw = hf_layer.mlp.c_fc.weight.data
        W1_lin = as_linear_weight(W1_raw, in_dim=self.D, out_dim=I)  # [D,I]
        b_fc1 = hf_layer.mlp.c_fc.bias.data.clone().to(device=dev, dtype=ptdtype)

        r_fc1 = max(1, int(rank_ratio_mlp * min(W1_lin.shape)))  # min(D,I)
        U1, V1 = svd_factor(W1_lin, r_fc1)  # U1:[D,r1], V1:[r1,I]
        self.fc1_U = nn.Parameter(U1.to(device=dev, dtype=ptdtype))
        self.fc1_V = nn.Parameter(V1.to(device=dev, dtype=ptdtype))
        self.fc1_b = nn.Parameter(b_fc1)

        # FC2: h1:[B,S,I], W2_lin:[I,D] => h1 @ W2_lin + b2
        W2_raw = hf_layer.mlp.c_proj.weight.data
        W2_lin = as_linear_weight(W2_raw, in_dim=I, out_dim=self.D)  # [I,D]
        b_fc2 = hf_layer.mlp.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)

        r_fc2 = max(1, int(rank_ratio_mlp * min(W2_lin.shape)))  # min(I,D)
        U2, V2 = svd_factor(W2_lin, r_fc2)  # U2:[I,r2], V2:[r2,D]
        self.fc2_U = nn.Parameter(U2.to(device=dev, dtype=ptdtype))
        self.fc2_V = nn.Parameter(V2.to(device=dev, dtype=ptdtype))
        self.fc2_b = nn.Parameter(b_fc2)

        # Optional save
        if preload_factors is None and save_factors_to is not None:
            os.makedirs(os.path.dirname(save_factors_to), exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in self.factors_state_dict().items()}, save_factors_to)

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
    # Forward
    # ---------------------
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, S, D = hidden_states.shape  # D == self.D

        # LN1
        x = self.ln1(hidden_states)  # [B,S,D]

        # Q,K,V via low-rank per-head
        # Q = X @ (Uq@Vq)  => einsum('b s d, d h r, h r e -> b h s e')
        Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]
        K = torch.einsum('bsd,dhr,hre->bhse', x, self.k_U, self.k_V) + self.k_b[None, :, None, :]
        V = torch.einsum('bsd,dhr,hre->bhse', x, self.v_U, self.v_V) + self.v_b[None, :, None, :]

        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,S,S]

        # causal mask
        causal = make_causal_mask(S, device=attn_scores.device)
        attn_scores = attn_scores.masked_fill(~causal, float('-inf'))

        # external mask (expects 1 for tokens to keep)
        if attention_mask is not None:
            if attention_mask.dim() == 2:          # [B,S]
                am = attention_mask[:, None, None, :].bool()
            else:                                   # already broadcasted
                am = attention_mask.bool()
                if am.shape[-2] == 1:
                    am = am.expand(-1, -1, S, -1)
            attn_scores = attn_scores.masked_fill(~am, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        Y = torch.matmul(attn_probs, V)                             # [B,H,S,dh]
        Y = Y.transpose(1, 2).contiguous().view(B, S, self.D)       # [B,S,D]

        # Out proj: Y @ (Uo@Vo)
        Y = torch.matmul(torch.matmul(Y, self.out_U), self.out_V) + self.out_b
        hidden_states = hidden_states + Y

        # MLP
        z = self.ln2(hidden_states)   # [B,S,D]

        # fc1: W1_lin ≈ U1@V1 with U1:[D,r1], V1:[r1,I]
        t1 = torch.matmul(z, self.fc1_U)                 # [B,S,r1]
        h1 = torch.matmul(t1, self.fc1_V) + self.fc1_b   # [B,S,I]
        h1 = F.gelu(h1)

        # fc2: W2_lin ≈ U2@V2 with U2:[I,r2], V2:[r2,D]
        t2 = torch.matmul(h1, self.fc2_U)                # [B,S,r2]
        h2 = torch.matmul(t2, self.fc2_V) + self.fc2_b   # [B,S,D]

        hidden_states = hidden_states + h2
        return (hidden_states,)


# =========================
# Simple shim to keep HF forward happy
# =========================
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        return self.block(hidden_states, attention_mask, **kwargs)


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
    model.config.use_cache = False
    model.eval()
    if device:
        model = model.to(device)

    # Disable grads
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
        )
        shim = LayerShim(blk).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim

    return model


@torch.no_grad()
def compare_qkv_intermediate(dense_block: nn.Module, svd_block: LowRankSVDBlock, x: torch.Tensor):
    """
    Compare dense vs low-rank Q/K/V given *post-LN* input x to the block.
    """
    D, H, dh = svd_block.D, svd_block.H, svd_block.dh

    # Normalize dense attn weight to linear form [D,3D]
    Wc_raw = dense_block.attn.c_attn.weight.data
    bc = dense_block.attn.c_attn.bias.data.to(device=x.device, dtype=x.dtype)
    W_lin = as_linear_weight(Wc_raw, in_dim=D, out_dim=3 * D)  # [D,3D]

    qkv = x @ W_lin + bc  # [B,S,3D]
    q, k, v = qkv.split(D, dim=-1)
    q = q.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()
    k = k.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()
    v = v.view(x.shape[0], x.shape[1], H, dh).permute(0, 2, 1, 3).contiguous()

    # Low-rank
    Q = torch.einsum('bsd,dhr,hre->bhse', x, svd_block.q_U, svd_block.q_V) + svd_block.q_b[None, :, None, :]
    K = torch.einsum('bsd,dhr,hre->bhse', x, svd_block.k_U, svd_block.k_V) + svd_block.k_b[None, :, None, :]
    V = torch.einsum('bsd,dhr,hre->bhse', x, svd_block.v_U, svd_block.v_V) + svd_block.v_b[None, :, None, :]

    def stats(name, A, B):
        md = (A - B).abs().max().item()
        rd = (A - B).norm() / (A.norm() + 1e-12)
        print(f"{name:>3}  max|Δ|={md:.8f}  rel={rd:.8f}")

    print("=== QKV intermediate comparison ===")
    stats("Q", Q, q)
    stats("K", K, k)
    stats("V", V, v)


@torch.no_grad()
def end_to_end_validation(dense: GPT2LMHeadModel, svd: GPT2LMHeadModel, device: str):
    test_input_ids = torch.randint(0, 1000, (2, 16), device=device)
    attn = torch.ones_like(test_input_ids, device=device)

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
# Perplexity + Memory + Time
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

        # chunk to reduce peak
        chunk = 4
        B = batch["input_ids"].size(0)
        for i in range(0, B, chunk):
            sl = slice(i, min(i + chunk, B))
            ids = batch["input_ids"][sl]
            mask = batch["attention_mask"][sl]

            if use_mask:
                out = mdl(input_ids=ids, attention_mask=mask, use_cache=False)
            else:
                out = mdl(input_ids=ids, use_cache=False)

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
    parser.add_argument("--save-factors-dir", type=str, default=None)
    parser.add_argument("--load-factors-dir", type=str, default=None)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--debug-attn", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Dense model (reference) =====
    dense = GPT2LMHeadModel.from_pretrained("gpt2")
    dense.config.use_cache = False
    dense = dense.to(device).eval()
    for p in dense.parameters():
        p.requires_grad = False
    dense_mem = compute_persistent_memory(dense)
    print(f"Dense model storage: {dense_mem:6.1f} MiB")

    # ===== Build SVD model =====
    print("\n=== Building SVD Model ===")
    svd_model = build_svd_model(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        save_factors_dir=args.save_factors_dir,
        load_factors_dir=args.load_factors_dir,
        device=device,
    )
    for p in svd_model.parameters():
        p.requires_grad = False
    print(f"SVD model built with per-head rank≈{args.rank_ratio_attn}*min(D,dh) and MLP ranks≈{args.rank_ratio_mlp}*...")

    # Print ranks from first block
    first_blk = svd_model.transformer.h[0].block
    print(f"QKV rank: {first_blk.r_attn}, Out rank: {first_blk.r_out}")
    print(f"FC1 rank: {first_blk.r_fc1}, FC2 rank: {first_blk.r_fc2}")
    
    if args.debug_attn:
        blk0 = svd_model.transformer.h[0].block
        print(f"[debug-attn] D={blk0.D}, H={blk0.H}, dh={blk0.dh}")

    layer0 = dense.transformer.h[0]
    Wc = layer0.attn.c_attn.weight
    bc = layer0.attn.c_attn.bias
    print("weight", tuple(Wc.shape), "bias", tuple(bc.shape),
          "embed_dim", layer0.attn.embed_dim, "heads", layer0.attn.num_heads)

    # ===== Optional validation =====
    if args.validate:
        print("\n=== SVD Validation ===")
        maxd, reld = end_to_end_validation(dense, svd_model, device=device)
        print("\n--- Block-0 Q/K/V check ---")
        with torch.no_grad():
            test_ids = torch.randint(0, 1000, (2, 16), device=device)
            wte = dense.transformer.wte(test_ids)
            pos = torch.arange(test_ids.size(1), device=device)[None, :]
            wpe = dense.transformer.wpe(pos)
            h0_in = wte + wpe
            h0_in = dense.transformer.drop(h0_in)
            x0 = dense.transformer.h[0].ln_1(h0_in)
            compare_qkv_intermediate(dense.transformer.h[0], svd_model.transformer.h[0].block, x0)
        print("=== End Validation ===\n")

    # ===== Data =====
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

    # ===== Baseline dense metrics =====
    print("\n=== Dense baseline ===")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ppl_m, peak_m, t_m = perplexity_peak_time(dense, loader, device, use_mask=True)
    print(f"Dense w/ mask   | ppl={ppl_m:.4f} | peak={peak_m:7.1f} MiB | {t_m:6.1f} ms/b")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ppl_nm, peak_nm, t_nm = perplexity_peak_time(dense, loader, device, use_mask=False)
    print(f"Dense w/o mask  | ppl={ppl_nm:.4f} | peak={peak_nm:7.1f} MiB | {t_nm:6.1f} ms/b")

    # ===== SVD metrics =====
    print("\n=== SVD model ===")
    svd_mem = compute_persistent_memory(svd_model)
    print(f"SVD model storage: {svd_mem:6.1f} MiB "
          f"(saving {dense_mem - svd_mem:+.1f} MiB, {100*(dense_mem - svd_mem)/max(dense_mem,1e-9):.1f}%)")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ppl_m_s, peak_m_s, t_m_s = perplexity_peak_time(svd_model, loader, device, use_mask=True)
    print(f"SVD   w/ mask   | ppl={ppl_m_s:.4f} | peak={peak_m_s:7.1f} MiB | {t_m_s:6.1f} ms/b")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ppl_nm_s, peak_nm_s, t_nm_s = perplexity_peak_time(svd_model, loader, device, use_mask=False)
    print(f"SVD   w/o mask  | ppl={ppl_nm_s:.4f} | peak={peak_nm_s:7.1f} MiB | {t_nm_s:6.1f} ms/b")

    # ===== Summary =====
    print("\n=== Performance Summary ===")
    print(f"Storage (MiB): Dense={dense_mem:.1f} | SVD={svd_mem:.1f} | Δ={dense_mem - svd_mem:+.1f} ({100*(dense_mem - svd_mem)/max(dense_mem,1e-9):.1f}%)")
    print(f"Perplexity: dense w/={ppl_m:.4f} w/o={ppl_nm:.4f} | svd w/={ppl_m_s:.4f} w/o={ppl_nm_s:.4f}")
    print(f"Peak (MiB): dense w/={peak_m:7.1f} w/o={peak_nm:7.1f} | svd w/={peak_m_s:7.1f} w/o={peak_nm_s:7.1f}")
    print(f"Latency (ms/batch): dense w/={t_m:6.1f} w/o={t_nm:6.1f} | svd w/={t_m_s:6.1f} w/o={t_nm_s:6.1f}")


if __name__ == "__main__":
    main()
