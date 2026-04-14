# lowrank_gpt2.py
# ------------------------------------------------------------
# Low-rank GPT-2 blocks + SVD init + helpers + CLI.
# ------------------------------------------------------------
import math
import os
import time
import argparse
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer


# -----------------------
# Utility
# -----------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_linear_weight(W_raw: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
    """
    Return weight in (in_dim, out_dim) layout regardless of HF Conv1D orientation.
    """
    if W_raw.shape == (in_dim, out_dim):
        return W_raw.contiguous()
    if W_raw.shape == (out_dim, in_dim):
        return W_raw.t().contiguous()
    raise ValueError(
        f"Unexpected weight shape {tuple(W_raw.shape)}; expected ({in_dim},{out_dim}) or ({out_dim},{in_dim})."
    )


@torch.no_grad()
def svd_factor(W: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Truncated SVD factorization producing U in (in,r) and V in (r,out) such that U@V ≈ W (in,out).
    We fold Σ into both sides as sqrt(Σ) for numerical stability (balanced scaling).
    """
    if W.dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        W = W.float()
    W = W.contiguous()
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except TypeError:
        U_, S_, V_ = torch.svd(W)
        U, S, Vh = U_, S_, V_.t()

    r = min(rank, S.numel())
    U = U[:, :r].contiguous()
    S = S[:r].contiguous()
    Vh = Vh[:r, :].contiguous()

    sqrtS = torch.sqrt(S).to(U.dtype)
    U_bal = U * sqrtS[None, :]
    V_bal = (sqrtS[:, None] * Vh).contiguous()
    return U_bal, V_bal  # (in,r), (r,out)


# -----------------------
# Low-rank GPT-2 block
# -----------------------
class LowRankSVDBlock(nn.Module):
    """
    A GPT-2 block whose linear layers are replaced with explicit low-rank factors (U @ V).
    Q/K/V are parameterized per-head to keep einsum cheap and cache-friendly.
    """
    def __init__(
        self,
        hf_layer: nn.Module,
        rank_ratio_attn: float = 1.0,
        rank_ratio_mlp: float = 1.0,
        preload_factors: Optional[Dict[str, torch.Tensor]] = None,
        init_from_dense: bool = True,
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
        Wc_lin = as_linear_weight(hf_layer.attn.c_attn.weight.data, in_dim=D, out_dim=3 * D)  # (D,3D)
        bc = hf_layer.attn.c_attn.bias.data.clone().to(device=dev, dtype=ptdtype)

        q_w = Wc_lin[:, :D].contiguous().view(D, H, dh)
        k_w = Wc_lin[:, D:2 * D].contiguous().view(D, H, dh)
        v_w = Wc_lin[:, 2 * D:3 * D].contiguous().view(D, H, dh)

        q_b = bc[:D].view(H, dh).contiguous()
        k_b = bc[D:2 * D].view(H, dh).contiguous()
        v_b = bc[2 * D:3 * D].view(H, dh).contiguous()

        # ----- ranks (auto-adopt from preloaded factors if provided) -----
        preload_ranks = None
        if preload_factors is not None:
            try:
                r_attn_pre = int(preload_factors["q_U"].shape[2])   # [D, H, r]
                r_out_pre  = int(preload_factors["out_V"].shape[0]) # [r, D]
                r_fc1_pre  = int(preload_factors["fc1_V"].shape[0]) # [r, I]
                r_fc2_pre  = int(preload_factors["fc2_V"].shape[0]) # [r, D]
                preload_ranks = (r_attn_pre, r_out_pre, r_fc1_pre, r_fc2_pre)
            except KeyError:
                preload_ranks = None

        r_attn = preload_ranks[0] if preload_ranks else max(1, int(rank_ratio_attn * min(D, dh)))

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

        # ---------- OUT PROJ ----------
        W_out_lin = as_linear_weight(hf_layer.attn.c_proj.weight.data, in_dim=D, out_dim=D)
        b_out = hf_layer.attn.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_out = preload_ranks[1] if preload_ranks else max(1, int(rank_ratio_attn * min(W_out_lin.shape)))
        Uo, Vo = svd_factor(W_out_lin, r_out)
        self.out_U = nn.Parameter(Uo.to(device=dev, dtype=ptdtype))
        self.out_V = nn.Parameter(Vo.to(device=dev, dtype=ptdtype))
        self.out_b = nn.Parameter(b_out)

        # ---------- MLP ----------
        I = hf_layer.mlp.c_fc.bias.data.numel()
        W1_lin = as_linear_weight(hf_layer.mlp.c_fc.weight.data, in_dim=D, out_dim=I)
        b_fc1 = hf_layer.mlp.c_fc.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc1 = preload_ranks[2] if preload_ranks else max(1, int(rank_ratio_mlp * min(W1_lin.shape)))
        U1, V1 = svd_factor(W1_lin, r_fc1)
        self.fc1_U = nn.Parameter(U1.to(device=dev, dtype=ptdtype))
        self.fc1_V = nn.Parameter(V1.to(device=dev, dtype=ptdtype))
        self.fc1_b = nn.Parameter(b_fc1)

        W2_lin = as_linear_weight(hf_layer.mlp.c_proj.weight.data, in_dim=I, out_dim=D)
        b_fc2 = hf_layer.mlp.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_fc2 = preload_ranks[3] if preload_ranks else max(1, int(rank_ratio_mlp * min(W2_lin.shape)))
        U2, V2 = svd_factor(W2_lin, r_fc2)
        self.fc2_U = nn.Parameter(U2.to(device=dev, dtype=ptdtype))
        self.fc2_V = nn.Parameter(V2.to(device=dev, dtype=ptdtype))
        self.fc2_b = nn.Parameter(b_fc2)

        # Keep ranks for logs
        self.r_attn = r_attn
        self.r_out = self.out_V.shape[0]
        self.r_fc1 = self.fc1_V.shape[0]
        self.r_fc2 = self.fc2_V.shape[0]

        # Optional preload of (whitened or vanilla) factors
        if preload_factors is not None:
            self.load_factors_(preload_factors)
        elif init_from_dense:
            # Initialize Q/K/V with vanilla per-head SVD if not preloaded
            for h in range(H):
                Uq, Vq = svd_factor(q_w[:, h, :], r_attn)
                Uk, Vk = svd_factor(k_w[:, h, :], r_attn)
                Uv, Vv = svd_factor(v_w[:, h, :], r_attn)
                with torch.no_grad():
                    self.q_U[:, h, :].copy_(Uq.to(device=dev, dtype=ptdtype))
                    self.q_V[h, :, :].copy_(Vq.to(device=dev, dtype=ptdtype))
                    self.k_U[:, h, :].copy_(Uk.to(device=dev, dtype=ptdtype))
                    self.k_V[h, :, :].copy_(Vk.to(device=dev, dtype=ptdtype))
                    self.v_U[:, h, :].copy_(Uv.to(device=dev, dtype=ptdtype))
                    self.v_V[h, :, :].copy_(Vv.to(device=dev, dtype=ptdtype))

    def factors_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "q_U": self.q_U, "q_V": self.q_V, "q_b": self.q_b,
            "k_U": self.k_U, "k_V": self.k_V, "k_b": self.k_b,
            "v_U": self.v_U, "v_V": self.v_V, "v_b": self.v_b,
            "out_U": self.out_U, "out_V": self.out_V, "out_b": self.out_b,
            "fc1_U": self.fc1_U, "fc1_V": self.fc1_V, "fc1_b": self.fc1_b,
            "fc2_U": self.fc2_U, "fc2_V": self.fc2_V, "fc2_b": self.fc2_b,
        }

    @torch.no_grad()
    def load_factors_(self, tensors: Dict[str, torch.Tensor]):
        mine = self.factors_state_dict()
        for k, p in mine.items():
            if k not in tensors:
                raise KeyError(f"Missing factor '{k}' in preload_factors")
            p.copy_(tensors[k].to(dtype=p.dtype, device=p.device))

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

        x = self.ln1(hidden_states)  # [B,S,D]

        # Q/K/V via low-rank factors
        Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]
        K = torch.einsum('bsd,dhr,hre->bhse', x, self.k_U, self.k_V) + self.k_b[None, :, None, :]
        V = torch.einsum('bsd,dhr,hre->bhse', x, self.v_U, self.v_V) + self.v_b[None, :, None, :]

        # KV cache concat if provided
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

        total_len = past_len + S

        # attention
        attn_scores = torch.matmul(Q, K_cat.transpose(-2, -1)) * self.scale  # [B,H,S,total_len]

        # causal + optional attention mask (bf16/fp16-safe large negative)
        neg = torch.full((), -1e4, dtype=hidden_states.dtype, device=dev)
        causal = torch.ones(total_len, total_len, dtype=torch.bool, device=dev).tril_()[-S:, :].contiguous()
        attn_scores = attn_scores.masked_fill(~causal[None, None, :, :], neg)

        if attention_mask is not None:
            if attention_mask.dim() == 4:  # already broadcasted by HF
                am = attention_mask[..., -total_len:]
                if am.dtype.is_floating_point:
                    attn_scores = attn_scores + am.to(dtype=attn_scores.dtype)
                else:
                    attn_scores = attn_scores.masked_fill(~am.bool(), neg)
            elif attention_mask.dim() == 2:
                if attention_mask.size(-1) == total_len:
                    key_keep = attention_mask[:, None, None, :].bool()
                elif attention_mask.size(-1) == S:
                    pad = torch.ones(B, past_len, dtype=attention_mask.dtype, device=dev)
                    key_keep = torch.cat([pad, attention_mask], dim=-1)[:, None, None, :].bool()
                else:
                    key_keep = torch.ones(B, 1, 1, total_len, dtype=torch.bool, device=dev)
                attn_scores = attn_scores.masked_fill(~key_keep, neg)

        attn_probs = F.softmax(attn_scores, dim=-1)
        Y = torch.matmul(attn_probs, V_cat)  # [B,H,S,dh]
        Y = Y.transpose(1, 2).contiguous().view(B, S, self.D)  # [B,S,D]

        # out proj low-rank
        Y = torch.matmul(torch.matmul(Y, self.out_U), self.out_V) + self.out_b
        hidden_states = hidden_states + Y

        # MLP
        z = self.ln2(hidden_states)
        h1 = torch.matmul(torch.matmul(z, self.fc1_U), self.fc1_V) + self.fc1_b
        h1 = F.gelu(h1)
        h2 = torch.matmul(torch.matmul(h1, self.fc2_U), self.fc2_V) + self.fc2_b
        hidden_states = hidden_states + h2

        outputs = (hidden_states,)
        if use_cache:
            outputs = outputs + ((K, V),)
        if output_attentions:
            outputs = outputs + (attn_probs,)
        return outputs


# -----------------------
# Cache shim to match HF
# -----------------------
def _ensure_bhtd(k: torch.Tensor, v: torch.Tensor, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert k.dim() == 4 and v.dim() == 4, "Cache tensors must be 4D"
    if k.size(1) == H:  # [B,H,T,dh]
        return k, v
    if k.size(2) == H:  # [B,T,H,dh] -> [B,H,T,dh]
        return k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"Unrecognized cache layout for K={tuple(k.shape)} V={tuple(v.shape)} (H={H})")


def _from_bhtd_to_cache_layout(k_bhtd: torch.Tensor, v_bhtd: torch.Tensor, expect_bthd: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if expect_bthd:
        return k_bhtd.permute(0, 2, 1, 3).contiguous(), v_bhtd.permute(0, 2, 1, 3).contiguous()
    return k_bhtd, v_bhtd


class LayerShim(nn.Module):
    """
    Wrap LowRankSVDBlock to satisfy GPT2Block's cache interface.
    """
    def __init__(self, block: nn.Module, layer_idx: int = None):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        layer_past = None
        expect_bthd = False

        if past_key_value is not None and self.layer_idx is not None:
            if hasattr(past_key_value, "get_seq_length"):
                try:
                    seq_len = past_key_value.get_seq_length(self.layer_idx)
                except Exception:
                    seq_len = 0
                if seq_len and hasattr(past_key_value, "layers") and len(past_key_value.layers) > self.layer_idx:
                    layer_cache = past_key_value.layers[self.layer_idx]
                    k_cache = getattr(layer_cache, "keys", None)
                    v_cache = getattr(layer_cache, "values", None)
                    if k_cache is not None and v_cache is not None and k_cache.dim() == 4:
                        expect_bthd = (k_cache.size(2) == self.block.H)
                        k_std, v_std = _ensure_bhtd(k_cache, v_cache, self.block.H)
                        layer_past = (k_std, v_std)

        result = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            **kwargs,
        )

        if (past_key_value is not None and
            hasattr(past_key_value, "update") and
            self.layer_idx is not None and
            isinstance(result, tuple) and len(result) >= 2 and
            isinstance(result[1], tuple) and len(result[1]) == 2):
            k_new_bhtd, v_new_bhtd = result[1]
            k_upd, v_upd = _from_bhtd_to_cache_layout(k_new_bhtd, v_new_bhtd, expect_bthd)
            past_key_value.update(k_upd, v_upd, self.layer_idx)

        return result


# -----------------------
# Builders / Save / Load
# -----------------------
@torch.no_grad()
def build_lowrank_gpt2(
    rank_ratio_attn: float = 0.5,
    rank_ratio_mlp: float = 0.5,
    preload_dir: Optional[str] = None,
    device: Optional[str] = None,
    trainable: bool = True,
) -> GPT2LMHeadModel:
    """
    Create a GPT-2 model with every block replaced by a low-rank block (SVD-initialized).
    Optionally preload factors from directory produced by save_lowrank_factors().
    """
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    if device:
        model.to(device)
    model.eval()

    for i, layer in enumerate(model.transformer.h):
        preload = None
        if preload_dir is not None:
            fp = os.path.join(preload_dir, f"gpt2_block_{i}.pt")
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"Missing low-rank factors for block {i}: {fp}")
            preload = torch.load(fp, map_location="cpu")

        blk = LowRankSVDBlock(
            layer,
            rank_ratio_attn=rank_ratio_attn,
            rank_ratio_mlp=rank_ratio_mlp,
            preload_factors=preload,
            init_from_dense=(preload is None),
        )
        shim = LayerShim(blk, layer_idx=i).to(next(model.parameters()).device)
        model.transformer.h[i] = shim

    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    if trainable:
        # unfreeze low-rank params only by default
        for shim in model.transformer.h:
            for name, p in shim.block.named_parameters():
                if any(name.startswith(k) for k in ["q_", "k_", "v_", "out_", "fc1_", "fc2_"]):
                    p.requires_grad = True

    return model


def save_lowrank_factors(model: GPT2LMHeadModel, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for i, shim in enumerate(model.transformer.h):
        blk: LowRankSVDBlock = shim.block  # type: ignore
        bundle = {k: v.detach().cpu() for k, v in blk.factors_state_dict().items()}
        torch.save(bundle, os.path.join(save_dir, f"gpt2_block_{i}.pt"))


def load_lowrank_factors_dir(preload_dir: str) -> Dict[int, Dict[str, torch.Tensor]]:
    factors = {}
    for fname in os.listdir(preload_dir):
        if not fname.endswith(".pt"):
            continue
        idx = int(fname.split("_")[-1].split(".")[0])
        factors[idx] = torch.load(os.path.join(preload_dir, fname), map_location="cpu")
    return factors


# -----------------------
# Trainable param groups
# -----------------------
def get_param_groups(
    model: GPT2LMHeadModel,
    update: str = "lowrank-only",  # "lowrank-only" | "lowrank+ln+bias"
    lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> List[Dict]:
    """
    Build optimizer param groups. Applies weight decay to U/V only, not to biases or LN.
    """
    decay, no_decay = [], []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_lr = any(key in n for key in ["q_U", "q_V", "k_U", "k_V", "v_U", "v_V", "out_U", "out_V", "fc1_U", "fc1_V", "fc2_U", "fc2_V"])
        is_bias = n.endswith("_b") or n.endswith(".bias")
        is_ln = (".ln_1." in n) or (".ln_2." in n) or (".ln_f." in n)

        if update == "lowrank-only":
            if is_lr:
                (decay if not is_bias else no_decay).append(p)
        elif update == "lowrank+ln+bias":
            if is_lr or is_ln or is_bias:
                (decay if (is_lr and not is_bias) else no_decay).append(p)
        else:
            raise ValueError(f"Unknown update mode: {update}")

    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


# -----------------------
# Evaluation helper
# -----------------------
@torch.no_grad()
def evaluate_perplexity(
    model: GPT2LMHeadModel,
    dataloader,
    device: str,
    mixed_precision: Optional[str] = None,  # "fp16" | "bf16" | None
) -> float:
    """
    Masked language modeling perplexity (causal) over the dataloader.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    use_amp = (mixed_precision in ("fp16", "bf16"))
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits  # [B,T,V]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            m = attn[..., 1:].contiguous().bool()

            if m.any():
                valid_logits = shift_logits[m]
                valid_labels = shift_labels[m]
                loss = F.cross_entropy(valid_logits, valid_labels)
                total_loss += loss.item() * m.sum().item()
                total_tokens += m.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    return ppl


# -----------------------
# Info & quick demo helpers
# -----------------------
def count_params_and_bytes(model: nn.Module) -> Tuple[int, float]:
    """Return (num_params, MiB) across parameters & buffers."""
    n, b = 0, 0
    for p in model.parameters():
        n += p.numel()
        b += p.numel() * p.element_size()
    for t in model.buffers():
        n += t.numel()
        b += t.numel() * t.element_size()
    return n, b / (1024 ** 2)


@torch.no_grad()
def summarize_ranks(model: GPT2LMHeadModel):
    lines = []
    for i, shim in enumerate(model.transformer.h):
        blk: LowRankSVDBlock = shim.block  # type: ignore
        lines.append(f"layer {i:02d}: attn_r={blk.r_attn:3d}  out_r={blk.r_out:3d}  fc1_r={blk.r_fc1:3d}  fc2_r={blk.r_fc2:3d}")
    print("\n".join(lines), flush=True)


@torch.no_grad()
def quick_generate(
    model: GPT2LMHeadModel,
    prompt: str,
    max_new_tokens: int = 60,
    device: str = "cpu",
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> str:
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model.eval()
    ids = tok.encode(prompt, return_tensors="pt").to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model(input_ids=ids, use_cache=True)
    past, logits = out.past_key_values, out.logits
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    generated = ids
    for _ in range(max_new_tokens):
        last = logits[:, -1, :]
        if do_sample:
            if temperature != 1.0:
                last = last / max(temperature, 1e-5)
            if top_k > 0 and top_k < last.size(-1):
                kth = torch.topk(last, top_k, dim=-1).values[..., -1, None]
                last = last.masked_fill(last < kth, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(last, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs, dim=-1)
                cutoff = cum > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = False
                sorted_logits[cutoff] = float("-inf")
                last.scatter_(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(last, dim=-1)
            nxt = torch.multinomial(probs, 1)
        else:
            nxt = torch.argmax(last, dim=-1, keepdim=True)

        generated = torch.cat([generated, nxt], dim=1)
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits

    text = tok.decode(generated[0], skip_special_tokens=True)
    print(f"[gen] prefill={prefill_ms:.1f} ms  new={max_new_tokens} toks")
    return text


# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser(description="Low-rank GPT-2: build/compress/eval/generate")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    # Compression controls
    ap.add_argument("--rank-ratio-attn", type=float, default=0.4, help="ratio for Q/K/V and attn.out")
    ap.add_argument("--rank-ratio-mlp",  type=float, default=0.4, help="ratio for MLP fc1/fc2")
    ap.add_argument("--preload-factors", type=str, default=None, help="dir with gpt2_block_*.pt (auto-adopt ranks)")
    ap.add_argument("--save-factors",    type=str, default="./factors", help="dir to save current low-rank factors")

    # Evaluation
    ap.add_argument("--eval-dataset", type=str, default=None,
                    help="If set, evaluate perplexity on validation split (wikitext2|wiki|wikitext-2|ptb)")
    ap.add_argument("--max-length",   type=int, default=256)
    ap.add_argument("--batch-size",   type=int, default=8)
    ap.add_argument("--eval-samples", type=int, default=2048)
    ap.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])

    # Generation
    ap.add_argument("--generate", action="store_true", help="Run a quick text generation demo")
    ap.add_argument("--prompt",   type=str, default="The future of artificial intelligence")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)

    # Info
    ap.add_argument("--print-ranks", action="store_true")
    ap.add_argument("--print-storage", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    # Build low-rank model (auto-adopts preload ranks if provided)
    model = build_lowrank_gpt2(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        preload_dir=args.preload_factors,
        device=args.device,
        trainable=False,
    )

    if args.print_ranks:
        print("[ranks] per-layer ranks (attn/out/fc1/fc2):")
        summarize_ranks(model)

    if args.print_storage:
        n, mib = count_params_and_bytes(model)
        dense = GPT2LMHeadModel.from_pretrained("gpt2").to(args.device).eval()
        n_d, mib_d = count_params_and_bytes(dense)
        print(f"[storage] lowrank: {n/1e6:.2f}M params, {mib:.1f} MiB | dense: {n_d/1e6:.2f}M, {mib_d:.1f} MiB | "
              f"Δ={mib_d-mib:+.1f} MiB ({(mib_d-mib)/max(mib_d,1e-9)*100:.1f}%)")
        del dense

    # Optional: save factors
    if args.save_factors:
        os.makedirs(args.save_factors, exist_ok=True)
        save_lowrank_factors(model, args.save_factors)
        print(f"[save] factors saved to: {args.save_factors}")

    # Optional: eval perplexity
    if args.eval_dataset:
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

        ds_name = args.eval_dataset.lower()
        if ds_name in ("wikitext2", "wikitext-2", "wiki"):
            dval = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            textcol = "text"
        elif ds_name in ("ptb", "penn", "ptb_text_only"):
            dval = load_dataset("ptb_text_only", "penn_treebank", split="validation")
            textcol = "sentence"
        else:
            raise ValueError(f"Unsupported eval dataset: {args.eval_dataset}")

        def tok_fn(batch):
            return tok(batch[textcol], padding="max_length", truncation=True, max_length=args.max_length)
        dval = dval.map(tok_fn, batched=True, remove_columns=[textcol])
        #if args.eval_samples:  # noqa: E701
        dval = dval.select(range(min(len(dval), args.eval_samples)))
        dval.set_format("torch")

        def collate(b):
            return {
                "input_ids": torch.stack([x["input_ids"] for x in b]),
                "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            }
        loader = torch.utils.data.DataLoader(dval, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

        mp = None if args.mixed_precision == "none" else args.mixed_precision
        ppl = evaluate_perplexity(model, loader, device=args.device, mixed_precision=mp)
        print(f"[eval] {args.eval_dataset} val perplexity: {ppl:.3f}")

    # Optional: quick generation
    if args.generate:
        text = quick_generate(
            model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("\n=== Generated ===")
        print(text)


if __name__ == "__main__":
    main()
