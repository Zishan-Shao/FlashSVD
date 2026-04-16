import os, math, time, itertools, argparse, json
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
    if W.dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
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

# ================================================================
# SVD-LLM WHITENING (Cholesky) + WHITENED SVD FACTORIZATION
#   Implements: S from Cholesky(XX^T), SVD on (W_out_in @ S),
#               W'u = U * sqrt(Σ), W'v = sqrt(Σ)*V^T @ S^{-1}
#               Return in (in_dim x r, r x out_dim) to match our U,V.
#   See SVD-LLM Fig. 2 + Alg. 1/2.  (arXiv:2403.07378)  # citations above
# ================================================================
@torch.no_grad()
def cholesky_whitening(xxT: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given X X^T (in_dim x in_dim), compute S via Cholesky(XX^T + eps I) and S^{-1}.
    """
    in_dim = xxT.size(0)
    device, dtype = xxT.device, xxT.dtype
    xxT = xxT + eps * torch.eye(in_dim, device=device, dtype=dtype)
    S = torch.linalg.cholesky(xxT)  # lower-triangular, xxT = S S^T
    # We need S^{-1}, not (S S^T)^{-1}. Use a triangular solve against the identity.
    I = torch.eye(in_dim, device=device, dtype=dtype)
    # Solve S * X = I  ->  X = S^{-1}
    S_inv = torch.linalg.solve_triangular(S, I, upper=False)
    return S, S_inv


@torch.no_grad()
def whitened_svd_factor(
    W_in_out: torch.Tensor,   # (in_dim, out_dim)
    S: torch.Tensor,          # (in_dim, in_dim) whitening matrix
    S_inv: torch.Tensor,      # (in_dim, in_dim)
    rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Produce factors (U_inxr, V_rxout) such that U @ V ≈ W under SVD-LLM whitening.
      1) Form W_out_in = W^T (out x in), and target = W_out_in @ S  (out x in)
      2) [U, Σ, V^T] = svd(target) with truncation
      3) W'_u = U * sqrt(Σ)      (out x r)     [column-wise scaling]
         W'_v = sqrt(Σ) * V^T @ S^{-1}   (r x in)
      4) Convert back to (in x out) factorization:
         W' = (W'_v)^T @ (W'_u)^T  ==> return U_final=(W'_v)^T, V_final=(W'_u)^T
    """
    in_dim, out_dim = W_in_out.shape
    W_out_in = W_in_out.t().contiguous()  # (out, in)

    # SVD on W_out_in @ S
    target = W_out_in @ S  # (out, in)  # SVD-LLM: SVD on W S  (transpose-equivalent)
    try:
        U, Svals, Vh = torch.linalg.svd(target, full_matrices=False)
    except TypeError:
        U_, Svals_, V_ = torch.svd(target)
        U, Svals, Vh = U_, Svals_, V_.t()

    r = min(rank, Svals.numel())
    U = U[:, :r]                  # (out, r)
    s = Svals[:r]                 # (r,)
    Vh = Vh[:r, :]                # (r, in)

    sqrt_s = torch.sqrt(s).to(U.dtype)
    # Scale columns of U and rows of Vh by sqrt(Σ)
    # W_u = U * sqrt_s[None, :]                 # (out, r)
    # W_v = (sqrt_s[:, None] * Vh) @ S_inv      # (r, in)
    W_u = U * sqrt_s[None, :]                 # (out, r)
    # Right-solve: find Z such that Z @ S = sqrtΣ * V^T
    T = (sqrt_s[:, None] * Vh)                # (r, in)
    # Solve (Z S)^T = S^T Z^T = T^T  ->  Z^T = solve_triangular(S^T, T^T, upper=True)
    Zt = torch.linalg.solve_triangular(S.transpose(-2, -1), T.transpose(-2, -1), upper=True)
    W_v = Zt.transpose(-2, -1)                # (r, in)

    U_final = W_v.t().contiguous()            # (in, r)
    V_final = W_u.t().contiguous()            # (r, out)
    return U_final, V_final

# ================================================================
# Calibration: collect X^T X per weight input with hooks, then Cholesky
# Targets per GPT-2 block i:
#   - attn.c_attn (in=D)          -> S_attn_in
#   - attn.c_proj (in=D)          -> S_attn_out
#   - mlp.c_fc   (in=D)           -> S_mlp_fc1
#   - mlp.c_proj (in=I=intermed.) -> S_mlp_fc2
# ================================================================
class CovAccumulator:
    def __init__(self, in_dim: Optional[int] = None, device: str = "cpu"):
        self.in_dim = in_dim
        self.device = device
        # lazy init: allocate when we see the first batch
        self.sum_xxt: Optional[torch.Tensor] = (
            torch.zeros(in_dim, in_dim, dtype=torch.float64, device=device)
            if in_dim is not None else None
        )
        self.count = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor):
        # x: (B, S, in_dim) or (N, in_dim)
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        elif x.dim() == 2:
            pass
        else:
            return
        x = x.to(device=self.device, dtype=torch.float64)
        d = x.size(-1)
        # (re)initialize if needed or if in_dim was guessed incorrectly
        if self.sum_xxt is None or self.sum_xxt.size(0) != d:
            self.in_dim = d
            self.sum_xxt = torch.zeros(d, d, dtype=torch.float64, device=self.device)
            self.count = 0
        # accumulate X^T X  (d x d)
        self.sum_xxt.add_(x.t() @ x)
        self.count += x.size(0)

    @torch.no_grad()
    def finalize(self, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        #S, S_inv = cholesky_whitening(self.sum_xxt, eps=eps)
        assert self.sum_xxt is not None, "No activations were observed for this module."
        S, S_inv = cholesky_whitening(self.sum_xxt, eps=eps)
        return S.to(torch.float32), S_inv.to(torch.float32)

def _module_linear_in_dim(mod: nn.Module) -> int:
    # # GPT-2 uses Conv1D wrapper with .weight of shape (out_dim, in_dim) under the hood.
    # W = getattr(mod, "weight", None)
    # if W is None:
    #     # HF GPT-2 Conv1D stores weight as .weight of shape (nin, nout) or (nout, nin) depending on version
    #     # Fall back to try common attributes
    #     raise RuntimeError("Module has no .weight for dimension inference.")
    # if W.dim() != 2:
    #     raise RuntimeError("Expected a 2D weight.")
    # return W.shape[1]  # (out, in) -> return in
    W = getattr(mod, "weight", None)
    if W is None or W.dim() != 2:
        raise RuntimeError("Module has no 2D .weight for dimension inference.")
    b = getattr(mod, "bias", None)
    if b is not None and b.ndim == 1:
        out_dim = b.numel()
        # If weight is (in, out), W.shape[1] == out_dim
        if W.shape[1] == out_dim:
            return W.shape[0]
        # If weight is (out, in), W.shape[0] == out_dim
        if W.shape[0] == out_dim:
            return W.shape[1]
    # Fallback: GPT-2 Conv1D commonly stores (in, out)
    return W.shape[0]

@torch.no_grad()
def collect_whitening_mats_gpt2(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    calib_samples: int,
    max_length: int,
    batch_size: int,
    device: str,
    eps: float,
    save_dir: Optional[str] = None,
) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Returns dict: block_idx -> {
        "attn_c_attn": (S, S_inv)   [in=D]
        "attn_c_proj": (S, S_inv)   [in=D]
        "mlp_c_fc":    (S, S_inv)   [in=D]
        "mlp_c_proj":  (S, S_inv)   [in=I]
    }
    Optionally saves these matrices to save_dir/whiten/.
    """
    model.eval()
    model = model.to(device)

    # ----- dataset selection (calibration) -----
    if dataset_name.lower() in ("wikitext2", "wiki", "wikitext-2"):
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        textcol = "text"
    elif dataset_name.lower() in ("ptb", "penn", "ptb_text_only"):
        raw = load_dataset("ptb_text_only", "penn_treebank", split="train")
        textcol = "sentence"
    else:
        raise ValueError(f"Unsupported --calib-dataset '{dataset_name}'. Use 'wikitext2' or 'ptb'.")

    tok = tokenizer

    def tok_fn(batch):
        return tok(batch[textcol], padding="max_length", truncation=True, max_length=max_length)
    ds = raw.map(tok_fn, batched=True, remove_columns=[textcol])
    ds.set_format("torch")
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        },
    )

    # ----- build accumulators and hooks -----
    blocks = model.transformer.h
    accs = []
    hooks = []
    for i, blk in enumerate(blocks):
        # Each blk has: attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj
        A1_in = _module_linear_in_dim(blk.attn.c_attn)
        A2_in = _module_linear_in_dim(blk.attn.c_proj)
        F1_in = _module_linear_in_dim(blk.mlp.c_fc)
        F2_in = _module_linear_in_dim(blk.mlp.c_proj)

        acc = {
            "attn_c_attn": CovAccumulator(A1_in, device="cpu"),
            "attn_c_proj": CovAccumulator(A2_in, device="cpu"),
            "mlp_c_fc":    CovAccumulator(F1_in, device="cpu"),
            "mlp_c_proj":  CovAccumulator(F2_in, device="cpu"),
        }
        accs.append(acc)

        # forward_pre_hook to capture inputs *before* matmul
        def _mk_pre(name, a):
            def pre_hook(mod, inputs):
                x = inputs[0]
                a.add(x)
            return pre_hook

        hooks.append(blk.attn.c_attn.register_forward_pre_hook(_mk_pre("attn_c_attn", acc["attn_c_attn"])))
        hooks.append(blk.attn.c_proj.register_forward_pre_hook(_mk_pre("attn_c_proj", acc["attn_c_proj"])))
        hooks.append(blk.mlp.c_fc.register_forward_pre_hook(_mk_pre("mlp_c_fc", acc["mlp_c_fc"])))
        hooks.append(blk.mlp.c_proj.register_forward_pre_hook(_mk_pre("mlp_c_proj", acc["mlp_c_proj"])))

    # ----- run calibration -----
    need = calib_samples
    seen = 0
    for batch in loader:
        if seen >= need:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            _ = model(**batch)
        seen += batch["input_ids"].size(0)

    # remove hooks
    for h in hooks:
        h.remove()

    # ----- finalize S and optionally save -----
    S_dict: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for i, acc in enumerate(accs):
        S_block = {}
        for k, a in acc.items():
            S, S_inv = a.finalize(eps=eps)
            S_block[k] = (S, S_inv)
        S_dict[i] = S_block

    if save_dir is not None:
        wdir = os.path.join(save_dir, "whiten")
        os.makedirs(wdir, exist_ok=True)
        for i, block in S_dict.items():
            for name, (S, S_inv) in block.items():
                torch.save({"S": S.cpu(), "S_inv": S_inv.cpu()},
                           os.path.join(wdir, f"blk{i}_{name}.pt"))
        # minimal config
        cfg = {
            "dataset": dataset_name,
            "calib_samples": calib_samples,
            "max_length": max_length,
            "eps": eps,
            "num_blocks": len(blocks),
        }
        with open(os.path.join(wdir, "whitening_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    return S_dict

# ================================================================
# Low-rank GPT-2 Block with dense KV cache  (unchanged forward)
# Now capable of being preloaded with *whitened* factors.
# ================================================================
class LowRankSVDBlock(nn.Module):
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

        D = attn.embed_dim
        H = attn.num_heads
        if D % H != 0:
            raise ValueError(f"[LowRankSVDBlock] embed_dim={D} not divisible by heads={H}")
        dh = D // H

        self.D, self.H, self.dh = D, H, dh
        self.scale = 1.0 / math.sqrt(dh)
        # cache for causal masks to avoid re-allocating per layer
        self._mask_cache: Dict[Tuple[str, Optional[int], int, int], torch.Tensor] = {}

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

        # ---------- OUT PROJ ----------
        W_out_lin = as_linear_weight(hf_layer.attn.c_proj.weight.data, in_dim=D, out_dim=D)
        b_out = hf_layer.attn.c_proj.bias.data.clone().to(device=dev, dtype=ptdtype)
        r_out = max(1, int(rank_ratio_attn * min(W_out_lin.shape)))
        Uo, Vo = svd_factor(W_out_lin, r_out)
        self.out_U = nn.Parameter(Uo.to(device=dev, dtype=ptdtype))
        self.out_V = nn.Parameter(Vo.to(device=dev, dtype=ptdtype))
        self.out_b = nn.Parameter(b_out)

        # ---------- MLP ----------
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

        # Keep ranks for logs
        self.r_attn = r_attn
        self.r_out  = self.out_V.shape[0]
        self.r_fc1  = self.fc1_V.shape[0]
        self.r_fc2  = self.fc2_V.shape[0]

        # Optional preload of whitened (or vanilla) factors
        # Initialize Q/K/V from vanilla SVD if we are NOT loading precomputed factors.
        if preload_factors is None:
            # q_w, k_w, v_w have shapes [D, H, dh]
            for h in range(H):
                Uq, Vq = svd_factor(q_w[:, h, :], r_attn)  # [D,r], [r,dh]
                Uk, Vk = svd_factor(k_w[:, h, :], r_attn)
                Uv, Vv = svd_factor(v_w[:, h, :], r_attn)
                with torch.no_grad():
                    self.q_U[:, h, :].copy_(Uq.to(device=dev, dtype=ptdtype))
                    self.q_V[h, :, :].copy_(Vq.to(device=dev, dtype=ptdtype))
                    self.k_U[:, h, :].copy_(Uk.to(device=dev, dtype=ptdtype))
                    self.k_V[h, :, :].copy_(Vk.to(device=dev, dtype=ptdtype))
                    self.v_U[:, h, :].copy_(Uv.to(device=dev, dtype=ptdtype))
                    self.v_V[h, :, :].copy_(Vv.to(device=dev, dtype=ptdtype))

        else:
            self.load_factors_(preload_factors)

        # Optional save
        if save_factors_to is not None:
            os.makedirs(os.path.dirname(save_factors_to), exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in self.factors_state_dict().items()}, save_factors_to)

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
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # [B,H,T_past,dh]
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        B, S, D = hidden_states.shape
        dev = hidden_states.device
        neg_inf = torch.finfo(hidden_states.dtype).min

        x = self.ln1(hidden_states)  # [B,S,D]

        Q = torch.einsum('bsd,dhr,hre->bhse', x, self.q_U, self.q_V) + self.q_b[None, :, None, :]
        K = torch.einsum('bsd,dhr,hre->bhse', x, self.k_U, self.k_V) + self.k_b[None, :, None, :]
        V = torch.einsum('bsd,dhr,hre->bhse', x, self.v_U, self.v_V) + self.v_b[None, :, None, :]

        # Concatenate with past if provided (expects [B,H,T_past,dh])
        past_len = 0
        if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2 and layer_past[0] is not None:
            past_k, past_v = layer_past
            if past_k.dtype != K.dtype: past_k = past_k.to(dtype=K.dtype)
            if past_v.dtype != V.dtype: past_v = past_v.to(dtype=V.dtype)
            if past_k.device != K.device: past_k = past_k.to(K.device)
            if past_v.device != V.device: past_v = past_v.to(V.device)

            # Sanity checks
            assert past_k.dim() == 4 and past_v.dim() == 4, "past K/V must be 4D"
            assert past_k.shape[:2] == (B, self.H) and past_v.shape[:2] == (B, self.H), \
                f"Expected past [B,H,*,dh], got K {tuple(past_k.shape)} V {tuple(past_v.shape)}"
            assert past_k.shape[-1] == self.dh and past_v.shape[-1] == self.dh, "Head dim mismatch in past cache"

            K_cat = torch.cat([past_k, K], dim=2)
            V_cat = torch.cat([past_v, V], dim=2)
            past_len = past_k.size(2)
        else:
            K_cat, V_cat = K, V

        total_len = past_len + S

        attn_scores = torch.matmul(Q, K_cat.transpose(-2, -1)) * self.scale  # [B,H,S,total_len]

        # Fast causal mask: cache per (device, S, total_len)
        key = (dev.type, dev.index, int(S), int(total_len))
        mask = self._mask_cache.get(key)
        if mask is None:
            # build [S,total_len] mask: allow j <= past_len + i
            i = torch.arange(S, device=dev).view(S, 1)
            j = torch.arange(total_len, device=dev).view(1, total_len)
            past_len = total_len - S
            causal = (j <= (past_len + i))  # [S,total_len]
            mask = causal[None, None, :, :].contiguous()  # [1,1,S,total_len]
            self._mask_cache[key] = mask
        attn_scores = attn_scores.masked_fill(~mask, neg_inf)

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

        Y = torch.matmul(torch.matmul(Y, self.out_U), self.out_V) + self.out_b
        hidden_states = hidden_states + Y

        z = self.ln2(hidden_states)
        t1 = torch.matmul(z, self.fc1_U)
        h1 = torch.matmul(t1, self.fc1_V) + self.fc1_b
        h1 = F.gelu(h1)
        t2 = torch.matmul(h1, self.fc2_U)
        h2 = torch.matmul(t2, self.fc2_V) + self.fc2_b
        hidden_states = hidden_states + h2

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + ((K, V),)

        if output_attentions:
            outputs = outputs + (attn_probs,)

        return outputs

# =========================
# Cache layout helpers & shim  (unchanged)
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

def _ensure_bhtd(k: torch.Tensor, v: torch.Tensor, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert k.dim() == 4 and v.dim() == 4, "Cache tensors must be 4D"
    if k.size(1) == H:  # [B,H,T,dh]
        return k, v
    if k.size(2) == H:  # [B,T,H,dh] -> [B,H,T,dh]
        return k.permute(0, 2, 1, 3).contiguous(), v.permute(0, 2, 1, 3).contiguous()
    raise RuntimeError(f"Unrecognized cache layout for shapes K={tuple(k.shape)} V={tuple(v.shape)} (H={H})")

def _from_bhtd_to_cache_layout(k_bhtd: torch.Tensor, v_bhtd: torch.Tensor, expect_bthd: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if expect_bthd:
        return k_bhtd.permute(0, 2, 1, 3).contiguous(), v_bhtd.permute(0, 2, 1, 3).contiguous()
    return k_bhtd, v_bhtd

class DenseKVCache:
    """Simple per-layer dense KV cache: stores (K,V) as [B,H,T,dh] and grows along T."""
    def __init__(self, n_layers: int):
        self.layers: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers
    def get(self, i: int):
        return self.layers[i]
    def get_seq_length(self, i: int) -> int:
        entry = self.layers[i]
        return 0 if entry is None else int(entry[0].size(2))
    @torch.no_grad()
    def update(self, K_new: torch.Tensor, V_new: torch.Tensor, i: int):
        assert K_new.dim() == 4 and V_new.dim() == 4
        entry = self.layers[i]
        if entry is None:
            self.layers[i] = (K_new, V_new)
        else:
            K, V = entry
            self.layers[i] = (torch.cat([K, K_new], dim=2), torch.cat([V, V_new], dim=2))


class LayerShim(nn.Module):
    def __init__(self, block: nn.Module, layer_idx: int = None):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self._dense_cache: Optional[DenseKVCache] = None

    def forward(self, hidden_states, past_key_value=None, cache_position=None, attention_mask=None, *args, **kwargs):
        # Ignore HF past_key_value; use internal dense KV cache for stable decode
        layer_past = None
        dense_cache = getattr(self, "_dense_cache", None)
        if isinstance(dense_cache, DenseKVCache):
            entry = dense_cache.get(self.layer_idx)
            if entry is not None and dense_cache.get_seq_length(self.layer_idx) > 0:
                layer_past = entry

        result = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=kwargs.get("output_attentions", False),
        )

        if (isinstance(dense_cache, DenseKVCache) and
            isinstance(result, tuple) and len(result) >= 2 and
            isinstance(result[1], (tuple, list)) and len(result[1]) == 2):
            K_new, V_new = result[1]
            dense_cache.update(K_new, V_new, self.layer_idx)
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
        )
        shim = LayerShim(blk, layer_idx=i).to(device if device is not None else next(model.parameters()).device)
        model.transformer.h[i] = shim

    return model

# ================================================================
# Factor computation from whitening (block-by-block) + saving
# ================================================================
@torch.no_grad()
def compute_and_save_whitened_factors(
    dense: GPT2LMHeadModel,
    S_dict: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
    rank_ratio_attn: float,
    rank_ratio_mlp: float,
    save_dir: str,
    device: str,
):
    os.makedirs(save_dir, exist_ok=True)
    dense = dense.to(device).eval()
    for i, layer in enumerate(dense.transformer.h):
        D = layer.attn.embed_dim
        H = layer.attn.num_heads
        dh = D // H
        I = layer.mlp.c_fc.bias.numel()

        ptdtype = next(layer.parameters()).dtype

        # Extract dense weights in (in, out) layout
        Wc_lin = as_linear_weight(layer.attn.c_attn.weight.data, in_dim=D, out_dim=3 * D)  # [D,3D]
        bc = layer.attn.c_attn.bias.data.clone()

        q_w = Wc_lin[:, :D].contiguous().view(D, H, dh)
        k_w = Wc_lin[:, D:2*D].contiguous().view(D, H, dh)
        v_w = Wc_lin[:, 2*D:3*D].contiguous().view(D, H, dh)

        W_out_lin = as_linear_weight(layer.attn.c_proj.weight.data, in_dim=D, out_dim=D)
        b_out = layer.attn.c_proj.bias.data.clone()

        W1_lin = as_linear_weight(layer.mlp.c_fc.weight.data, in_dim=D, out_dim=I)
        b_fc1 = layer.mlp.c_fc.bias.data.clone()
        W2_lin = as_linear_weight(layer.mlp.c_proj.weight.data, in_dim=I, out_dim=D)
        b_fc2 = layer.mlp.c_proj.bias.data.clone()

        # Ranks
        r_attn = max(1, int(rank_ratio_attn * min(D, dh)))
        r_out  = max(1, int(rank_ratio_attn * min(D, D)))
        r_fc1  = max(1, int(rank_ratio_mlp * min(D, I)))
        r_fc2  = max(1, int(rank_ratio_mlp * min(I, D)))

        # Whitening matrices for this block
        S_attn, S_inv_attn = S_dict[i]["attn_c_attn"]
        S_proj, S_inv_proj = S_dict[i]["attn_c_proj"]
        S_fc1,  S_inv_fc1  = S_dict[i]["mlp_c_fc"]
        S_fc2,  S_inv_fc2  = S_dict[i]["mlp_c_proj"]

        # Move to device/dtype just-in-time for numeric stability
        S_attn     = S_attn.to(device, dtype=torch.float32)
        S_inv_attn = S_inv_attn.to(device, dtype=torch.float32)
        S_proj     = S_proj.to(device, dtype=torch.float32)
        S_inv_proj = S_inv_proj.to(device, dtype=torch.float32)
        S_fc1      = S_fc1.to(device, dtype=torch.float32)
        S_inv_fc1  = S_inv_fc1.to(device, dtype=torch.float32)
        S_fc2      = S_fc2.to(device, dtype=torch.float32)
        S_inv_fc2  = S_inv_fc2.to(device, dtype=torch.float32)

        # ---- Q/K/V per head with shared S_attn ----
        qU, qV = [], []
        kU, kV = [], []
        vU, vV = [], []
        for h in range(H):
            Uq, Vq = whitened_svd_factor(q_w[:, h, :].to(device, dtype=torch.float32), S_attn, S_inv_attn, r_attn)
            Uk, Vk = whitened_svd_factor(k_w[:, h, :].to(device, dtype=torch.float32), S_attn, S_inv_attn, r_attn)
            Uv, Vv = whitened_svd_factor(v_w[:, h, :].to(device, dtype=torch.float32), S_attn, S_inv_attn, r_attn)
            qU.append(Uq.to(ptdtype).cpu()); qV.append(Vq.to(ptdtype).cpu())
            kU.append(Uk.to(ptdtype).cpu()); kV.append(Vk.to(ptdtype).cpu())
            vU.append(Uv.to(ptdtype).cpu()); vV.append(Vv.to(ptdtype).cpu())
        qU = torch.stack(qU, dim=1)  # [D,H,r]
        qV = torch.stack(qV, dim=0)  # [H,r,dh]
        kU = torch.stack(kU, dim=1)
        kV = torch.stack(kV, dim=0)
        vU = torch.stack(vU, dim=1)
        vV = torch.stack(vV, dim=0)

        # ---- Attention out projection with S_proj ----
        out_U, out_V = whitened_svd_factor(W_out_lin.to(device, dtype=torch.float32), S_proj, S_inv_proj, r_out)
        out_U = out_U.to(ptdtype).cpu()
        out_V = out_V.to(ptdtype).cpu()

        # ---- MLP ----
        fc1_U, fc1_V = whitened_svd_factor(W1_lin.to(device, dtype=torch.float32), S_fc1, S_inv_fc1, r_fc1)
        fc2_U, fc2_V = whitened_svd_factor(W2_lin.to(device, dtype=torch.float32), S_fc2, S_inv_fc2, r_fc2)
        fc1_U = fc1_U.to(ptdtype).cpu(); fc1_V = fc1_V.to(ptdtype).cpu()
        fc2_U = fc2_U.to(ptdtype).cpu(); fc2_V = fc2_V.to(ptdtype).cpu()

        # package & save
        bundle = {
            "q_U": qU, "q_V": qV, "q_b": bc[:D].view(H, dh).contiguous().to(ptdtype).cpu(),
            "k_U": kU, "k_V": kV, "k_b": bc[D:2*D].view(H, dh).contiguous().to(ptdtype).cpu(),
            "v_U": vU, "v_V": vV, "v_b": bc[2*D:3*D].view(H, dh).contiguous().to(ptdtype).cpu(),
            "out_U": out_U, "out_V": out_V, "out_b": b_out.to(ptdtype).cpu(),
            "fc1_U": fc1_U, "fc1_V": fc1_V, "fc1_b": b_fc1.to(ptdtype).cpu(),
            "fc2_U": fc2_U, "fc2_V": fc2_V, "fc2_b": b_fc2.to(ptdtype).cpu(),
        }
        torch.save(bundle, os.path.join(save_dir, f"gpt2_block_{i}.pt"))

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

@torch.no_grad()
def perplexity_decode_cached(
    mdl: GPT2LMHeadModel,
    loader,
    device: str,
    *,
    max_batches: Optional[int] = None,
):
    """
    Compute perplexity by incremental decode using HF cache; only last-token logits each step.
    Uses absolute position_ids per step for GPT-2.
    Returns (ppl, ms_per_batch).
    """
    mdl.eval()
    total_loss, total_tokens = 0.0, 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        B, S = ids.shape

        pos_ids_full = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        out = None
        # Internal dense KV cache for this batch
        dense_cache = DenseKVCache(n_layers=len(mdl.transformer.h))
        # Attach to shims
        for lyr in mdl.transformer.h:
            if isinstance(lyr, LayerShim):
                setattr(lyr, "_dense_cache", dense_cache)
        for t in range(0, S - 1):
            inp = ids[:, t:t+1]
            pos_step = pos_ids_full[:, t:t+1]
            out = mdl(input_ids=inp, position_ids=pos_step, use_cache=True)

            logits_last = out.logits[:, -1, :]
            target = ids[:, t + 1]
            m = mask[:, t + 1].bool()
            if m.any():
                loss = F.cross_entropy(logits_last[m], target[m])
                total_loss += loss.item() * int(m.sum().item())
                total_tokens += int(m.sum().item())

        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms_per_batch = (time.perf_counter() - t0) * 1000.0 / max(1, (max_batches if max_batches is not None else len(loader)))
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if total_tokens > 0 else float("inf")
    return ppl, ms_per_batch

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
    parser.add_argument("--mode", type=str, choices=["prefill","decode","both"], default="both")
    parser.add_argument("--max-eval-samples", type=int, default=None)

    # --- NEW: Whitening / calibration options ---
    parser.add_argument("--whiten", action="store_true", help="Run truncation-aware data whitening (SVD-LLM) and save factors")
    parser.add_argument("--calib-dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb"], help="Calibration dataset for whitening")
    parser.add_argument("--calib-samples", type=int, default=512, help="Number of calibration sequences")
    parser.add_argument("--calib-max-length", type=int, default=256, help="Max tokens per calibration sequence")
    parser.add_argument("--whiten-eps", type=float, default=1e-5, help="Diagonal epsilon for Cholesky(XX^T)")
    parser.add_argument("--overwrite-whiten", action="store_true", help="Recompute whitening/factors even if present")

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dense model only used for whitening (optional)
    dense = None

    # ---------- Whitening path ----------
    # Autodetect existing factors in save_factors_dir if load_factors_dir is absent
    if args.whiten:
        if args.save_factors_dir is None:
            raise ValueError("--whiten requires --save-factors-dir")
        tmp_dense = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        blocks_exist = all(os.path.isfile(os.path.join(args.save_factors_dir, f"gpt2_block_{i}.pt")) for i in range(len(tmp_dense.transformer.h)))
        if blocks_exist and not args.overwrite_whiten:
            print(f"[whiten] Factors already exist in {args.save_factors_dir}. Skipping recompute.")
        else:
            tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
            print(f"[whiten] Collecting whitening matrices on {args.calib_dataset}...")
            S_dict = collect_whitening_mats_gpt2(
                model=tmp_dense, tokenizer=tok, dataset_name=args.calib_dataset,
                calib_samples=args.calib_samples, max_length=args.calib_max_length,
                batch_size=args.batch_size, device=device, eps=args.whiten_eps,
                save_dir=args.save_factors_dir
            )
            print("[whiten] Computing whitened low-rank factors (SVD on W S, reconstruction with Σ^{1/2} and S^{-1})...")
            compute_and_save_whitened_factors(
                dense=tmp_dense, S_dict=S_dict, rank_ratio_attn=args.rank_ratio_attn,
                rank_ratio_mlp=args.rank_ratio_mlp, save_dir=args.save_factors_dir, device=device
            )
        # auto-set load dir for subsequent build
        if args.load_factors_dir is None:
            args.load_factors_dir = args.save_factors_dir

    print("\n=== Building SVD Model ===")
    svd_model = build_svd_model(
        rank_ratio_attn=args.rank_ratio_attn,
        rank_ratio_mlp=args.rank_ratio_mlp,
        save_factors_dir=(None if args.load_factors_dir else args.save_factors_dir),
        load_factors_dir=args.load_factors_dir,
        device=device,
    )
    for p in svd_model.parameters():
        p.requires_grad = False
    print(f"SVD model built with per-head rank≈{args.rank_ratio_attn}*min(D,dh) and MLP ranks≈{args.rank_ratio_mlp}*...")

    first_blk = svd_model.transformer.h[0].block
    print(f"QKV rank: {first_blk.r_attn}, Out rank: {first_blk.r_out}")
    print(f"FC1 rank: {first_blk.r_fc1}, FC2 rank: {first_blk.r_fc2}")

    # Perplexity eval (dataset arg now supports wikitext2 or ptb)
    if True:
        print("Preparing evaluation data...")
        # default: WikiText-2 test split; allow PTB via calib-dataset for symmetry
        eval_name = args.calib_dataset
        if eval_name.lower() in ("wikitext2", "wiki", "wikitext-2"):
            raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            textcol = "text"
        else:
            raw = load_dataset("ptb_text_only", "penn_treebank", split="validation")
            textcol = "sentence"

        tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
        def tok_fn(batch):
            return tok(batch[textcol], padding="max_length", truncation=True, max_length=args.max_length)
        ds = raw.map(tok_fn, batched=True, remove_columns=[textcol])
        ds.set_format("torch")
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=lambda b: {
                "input_ids": torch.stack([x["input_ids"] for x in b]),
                "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            },
        )

        print("\n=== SVD model (whitened factors if provided) ===")
        svd_mem = compute_persistent_memory(svd_model)
        print(f"SVD model storage: {svd_mem:6.1f} MiB")
        if args.mode in ("prefill","both"):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            ppl_m_s, peak_m_s, t_m_s = perplexity_peak_time(svd_model, loader, device, use_mask=True)
            print(f"Prefill w/ mask | ppl={ppl_m_s:.4f} | peak={peak_m_s:7.1f} MiB | {t_m_s:6.1f} ms/b")

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            ppl_nm_s, peak_nm_s, t_nm_s = perplexity_peak_time(svd_model, loader, device, use_mask=False)
            print(f"Prefill no mask | ppl={ppl_nm_s:.4f} | peak={peak_nm_s:7.1f} MiB | {t_nm_s:6.1f} ms/b")

        if args.mode in ("decode","both"):
            ppl_dec, t_dec = perplexity_decode_cached(svd_model, loader, device)
            print(f"Decode (HF cache) | ppl={ppl_dec:.4f} | {t_dec:6.1f} ms/b")
    

if __name__ == "__main__":
    main()


# python3 profile_svd_kv_infer_w_whiten.py --mode decode --batch-size 2 --max-length 512 --max-eval-samples 8 --rank-ratio-attn 1.0 --rank-ratio-mlp 1.0
