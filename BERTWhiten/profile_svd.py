# profile_flashsvd_full.py

import os
import sys
import time
import itertools
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
from evaluate import load as load_metric
from typing import Callable, Tuple
import math
import torch.nn.functional as F
from flash_attn_triton import flash_attn_triton


import functools
import torch



# ─── locate repo & model ─────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
task_name = "stsb" 
MODEL_DIR = os.path.join(REPO_ROOT, "models/BERT", f"bert-base-uncased-{task_name}")

# ─── 0) Helpers for SVD decomposition ─────────────────────────────────────────
def build_plain_svd_helpers(model):
    def svd_per_head(Wt: torch.Tensor, rank: int):
        d_model, _ = Wt.shape
        H          = model.config.num_attention_heads
        dh         = d_model // H
        Wt3        = Wt.view(d_model, H, dh)
        Us, Vs     = [], []
        for h in range(H):
            Wh = Wt3[:, h, :].float()  # to float32 for SVD
            U32, S32, Vh32 = torch.linalg.svd(Wh, full_matrices=False)
            U = (U32[:, :rank] * S32[:rank]).to(Wt.dtype)
            V = Vh32[:rank, :].to(Wt.dtype)
            Us.append(U)
            Vs.append(V)
        return torch.stack(Us, 0), torch.stack(Vs, 0)

    def svd_low_rank(W: torch.Tensor, rank: int):
        Wf = W.float()
        U32, S32, Vh32 = torch.linalg.svd(Wf, full_matrices=False)
        U = (U32[:, :rank] * S32[:rank]).to(W.dtype)
        V = Vh32[:rank, :].to(W.dtype)
        return U, V

    return svd_per_head, svd_low_rank


# ─── 0b) DRONE (data-aware) low-rank decomposition ────────────────────────────
def _truncated_svd(M: torch.Tensor, k: int):
    # torch.linalg.svd returns U,S,Vh with Vh is V^T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    k_eff = min(k, U.shape[1], Vh.shape[0])
    return U[:, :k_eff], S[:k_eff], Vh[:k_eff, :]

def drone_low_rank(W: torch.Tensor, X: torch.Tensor, rank: int, eps: float = 1e-6):
    """
    DRONE factorization for y = W x with calibration inputs X.
    Runs the SVDs on CPU to match X (your calibration stores X on CPU).
    Returns CPU tensors; the module .to(device) call will move them later.
    """
    # --- do all linear algebra on CPU (prevents CUDA/CPU mismatch & saves VRAM)
    Wf = W.to(torch.device('cpu'), dtype=torch.float32)
    Xf = X.to(torch.device('cpu'), dtype=torch.float32)

    # SVDs with numerical rank detection
    Uw, Sw, Vhw = torch.linalg.svd(Wf, full_matrices=False)   # W = Uw diag(Sw) Vw^T
    r = int((Sw > eps).sum().item())
    if r == 0:
        # degenerate: fall back to a tiny rank-1 to avoid crashes
        r = 1
    Uw_r = Uw[:, :r]
    Sw_r = Sw[:r]
    Vw_r = Vhw[:r, :].T

    Ux, Sx, Vhx = torch.linalg.svd(Xf, full_matrices=False)   # X = Ux diag(Sx) Vx^T
    t = int((Sx > eps).sum().item())
    if t == 0:
        t = 1
    Ux_t = Ux[:, :t]
    Sx_t = Sx[:t].clamp_min(eps)

    # Z = S_Wr * (V_Wr^T U_Xt) * S_Xt  (shape r x t)
    Z = (Vw_r.T @ Ux_t)                      # r x t (now both on CPU)
    Z = Z * Sw_r.unsqueeze(1)                # row-scale by S_Wr
    Z = Z * Sx_t.unsqueeze(0)                # col-scale by S_Xt

    Uz, Sz, Vhz = torch.linalg.svd(Z, full_matrices=False)
    k_eff = min(rank, Uz.shape[1], Vhz.shape[0])
    if k_eff == 0:
        k_eff = 1
    Uz_k = Uz[:, :k_eff]                     # r x k
    Vz_k = Vhz[:k_eff, :].T                  # t x k

    # U* = U_Wr * U_zk      (d_out x k)
    U_star = Uw_r @ Uz_k

    # V* = U_Xt * (S_Xt^{-1} V_zk)   (d_in x k)
    V_scaled = Vz_k * (1.0 / Sx_t).unsqueeze(1)
    V_star   = Ux_t @ V_scaled

    # Keep on CPU for now; caller/module .to(device) will move later.
    P = V_star.to(dtype=W.dtype)             # (d_in, k)
    V = U_star.T.to(dtype=W.dtype)           # (k, d_out)
    return P, V

def drone_per_head(Wt_head: torch.Tensor, X_in: torch.Tensor, rank: int):
    """
    DRONE per head, matching your existing 'svd_per_head' API.
    Wt_head : (d_model, dh)  == W^T for this head
    X_in    : (d_model, n_samples)
    Returns:
      P : (d_model, rank)
      V : (rank, dh)
    """
    # Our helper expects W (d_out, d_in), so give W = Wt^T
    W = Wt_head.T.contiguous()  # (dh, d_model)
    P, V = drone_low_rank(W, X_in, rank)
    return P, V


# ─── 1) Collect activations X per encoder layer ───────────────────────────────
@torch.no_grad()
def collect_calibration_X(model, loader, device, tokens_per_layer=8192, max_batches=50):
    """
    For each encoder layer i:
      - X_attn[i]: inputs to self-attention (shape d_model x N_i)
      - X_ffn[i] : inputs to intermediate.dense (shape d_model x N_i)
    """
    L = len(model.bert.encoder.layer)
    d_model = model.config.hidden_size

    attn_buf = {i: [] for i in range(L)}
    ffn_buf  = {i: [] for i in range(L)}
    attn_cnt = {i: 0 for i in range(L)}
    ffn_cnt  = {i: 0 for i in range(L)}

    hooks = []

    def make_attn_hook(idx):
        def _hook(mod, inputs, output):
            if attn_cnt[idx] >= tokens_per_layer: return
            h = inputs[0].detach()                       # (B, M, d_model)
            B, M, D = h.shape
            take = min((tokens_per_layer - attn_cnt[idx]), B*M)
            if take <= 0: return
            # random subset of tokens for diversity
            flat = h.reshape(B*M, D)
            idxs = torch.randperm(B*M, device=flat.device)[:take]
            samp = flat[idxs].cpu().float().T            # (D, take)
            attn_buf[idx].append(samp)
            attn_cnt[idx] += take
        return _hook

    def make_ffn_hook(idx):
        def _hook(mod, inputs, output):
            if ffn_cnt[idx] >= tokens_per_layer: return
            h = inputs[0].detach()                       # (B, M, d_model), input to intermediate
            B, M, D = h.shape
            take = min((tokens_per_layer - ffn_cnt[idx]), B*M)
            if take <= 0: return
            flat = h.reshape(B*M, D)
            idxs = torch.randperm(B*M, device=flat.device)[:take]
            samp = flat[idxs].cpu().float().T            # (D, take)
            ffn_buf[idx].append(samp)
            ffn_cnt[idx] += take
        return _hook

    # register hooks
    for i, layer in enumerate(model.bert.encoder.layer):
        hooks.append(layer.attention.self.register_forward_hook(make_attn_hook(i)))
        hooks.append(layer.intermediate.register_forward_hook(make_ffn_hook(i)))

    # run a few batches
    model.to(device).eval()
    seen = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(input_ids=batch["input_ids"],
                  attention_mask=batch["attention_mask"],
                  labels=batch["labels"])
        seen += 1
        if seen >= max_batches: break
        # early exit if all filled
        if all(attn_cnt[i] >= tokens_per_layer for i in range(L)) and \
           all(ffn_cnt[i]  >= tokens_per_layer for i in range(L)):
            break

    # cleanup
    for h in hooks: h.remove()

    # pack to tensors (d_model, N)
    X_attn = {i: (torch.cat(attn_buf[i], dim=1) if attn_buf[i] else torch.empty(d_model,0)) for i in range(L)}
    X_ffn  = {i: (torch.cat(ffn_buf[i],  dim=1) if ffn_buf[i]  else torch.empty(d_model,0)) for i in range(L)}
    return X_attn, X_ffn




# ─── 2) LayerShim ────────────────────────────────────────────────────────────
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        raw_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4:
            raw_mask = (attention_mask[:,0,0,:] == 0)
        return (self.block(hidden_states, raw_mask),)


# -----------------------------------------------------------------------------
# 2) Pure-PyTorch FWSVD-block
# -----------------------------------------------------------------------------
class SVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int,
                 svd_per_head: Callable, svd_low_rank: Callable,
                 rank_wo: int = 768,
                 X_attn: torch.Tensor = None,   # (d_model, N)  calibration for Q/K/V
                 X_ffn:  torch.Tensor = None):  # (d_model, N)  calibration for FFN-in
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        d_ff    = hf_layer.intermediate.dense.out_features

        # 1) grab raw weights
        WqT = hf_layer.attention.self.query.weight.data.t()   # (d_model, d_model)
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq  = hf_layer.attention.self.query.bias.data.view(1, H, 1, dh)
        bk  = hf_layer.attention.self.key.bias.data.view(1, H, 1, dh)
        bv  = hf_layer.attention.self.value.bias.data.view(1, H, 1, dh)

        # reshape heads
        WqT3, WkT3, WvT3 = WqT.view(d_model, H, dh), WkT.view(d_model, H, dh), WvT.view(d_model, H, dh)

        # 1a) DRONE per head for Q/K/V using X_attn (fall back to SVD if none)
        Us_q, Vs_q, Us_k, Vs_k, Us_v, Vs_v = [], [], [], [], [], []
        for h in range(H):
            Wh_q = WqT3[:, h, :]
            Wh_k = WkT3[:, h, :]
            Wh_v = WvT3[:, h, :]
            if X_attn is not None and X_attn.numel() > 0:
                Pq, Vq = drone_per_head(Wh_q, X_attn, rank_attn)
                Pk, Vk = drone_per_head(Wh_k, X_attn, rank_attn)
                Pv, Vv = drone_per_head(Wh_v, X_attn, rank_attn)
            else:
                # fallback to your plain SVD
                Pq, Vq = svd_low_rank(Wh_q, rank_attn)
                Pk, Vk = svd_low_rank(Wh_k, rank_attn)
                Pv, Vv = svd_low_rank(Wh_v, rank_attn)
            Us_q.append(Pq); Vs_q.append(Vq)
            Us_k.append(Pk); Vs_k.append(Vk)
            Us_v.append(Pv); Vs_v.append(Vv)

        self.Pq, self.Vq, self.bq = map(nn.Parameter, (torch.stack(Us_q, 0), torch.stack(Vs_q, 0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (torch.stack(Us_k, 0), torch.stack(Vs_k, 0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (torch.stack(Us_v, 0), torch.stack(Vs_v, 0), bv))

        # 2) FFN (intermediate) via DRONE on Wi^T : (d_model, d_ff)
        Wi   = hf_layer.intermediate.dense.weight.data.t()   # (d_model, d_ff)
        bi   = hf_layer.intermediate.dense.bias.data
        if X_ffn is not None and X_ffn.numel() > 0:
            # drone_low_rank expects W(d_out,d_in) so pass Wi^T, X_ffn(d_model, N)
            P1, V1 = drone_low_rank(Wi.T.contiguous(), X_ffn, rank_ff)  # P1:(d_model,k), V1:(k,d_ff)
            U1, V1 = P1, V1
        else:
            U1, V1 = svd_low_rank(Wi, rank_ff)   # fallback to SVD

        self.U1, self.V1, self.b1 = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)

        # 3) FFN (output) and attention-output projection: keep your SVD (stable default)
        WoT  = hf_layer.output.dense.weight.data.t()         # (d_ff, d_model)
        bo2  = hf_layer.output.dense.bias.data
        U2, V2 = svd_low_rank(WoT, rank_ff)

        Wo_full = hf_layer.attention.output.dense.weight.data  # (d_model, d_model)
        bo_attn = hf_layer.attention.output.dense.bias.data
        Uo, Vo  = svd_low_rank(Wo_full.t(), rank_wo)

        self.U2, self.V2, self.b2     = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)
        self.Uo, self.Vo, self.bo_attn= nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        self.ln1, self.ln2            = hf_layer.attention.output.LayerNorm, hf_layer.output.LayerNorm
    
    
    
    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H = self.Pq.shape[0] if self.Pq.dim()==3 else self.Pq.shape[1]
        dh = dm // H
        scale = 1.0 / math.sqrt(dh)

        # project into low-rank Q/K/V
        def project(x, P, V, b):
            # x: [B,M,dm], P: [H,dm,r], V: [H,r,dh], b: [1,H,1,dh]
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)     # [B,H,M,r]
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b  # [B,H,M,dh]

        Q = project(x, self.Pq if self.Pq.dim()==3 else self.Pq[0], 
                       self.Vq if self.Vq.dim()==3 else self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk if self.Pk.dim()==3 else self.Pk[0], 
                       self.Vk if self.Vk.dim()==3 else self.Vk[0], self.bk).contiguous()
        V = project(x, self.Pv if self.Pv.dim()==3 else self.Pv[0], 
                       self.Vv if self.Vv.dim()==3 else self.Vv[0], self.bv).contiguous()

        # attention mask -> [B,H,1,M] bool (True = keep)
        # mask: [B, M], True means "keep"; flash_attn wants pad=True → invert
        if mask is not None:
            if mask.dtype != torch.bool:
                keep = mask > 0
            else:
                keep = mask
            key_pad = ~keep                              # invert: True on PADs
            mask4d  = key_pad.view(B, 1, 1, M).expand(B, H, 1, M)
        else:
            # no pads → all False (no masking)
            mask4d = torch.zeros(B, H, 1, M, device=x.device, dtype=torch.bool)

        # FlashAttention expects [B,H,M,dh]
        #attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)  # [B,H,M,dh]
        USE_FLASH_ATTN = False
        if USE_FLASH_ATTN:
            attn = flash_attn_triton(Q, K, V, mask4d, BLOCK_M=32)  # [B,H,M,dh]
        else:
            # safe, reference attention to verify behavior
            scale = 1.0 / math.sqrt(dh)
            logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * scale
            logits = logits.masked_fill(mask4d, float("-1e9"))     # mask True = pad
            A = torch.softmax(logits, dim=-1)
            attn = torch.einsum("bhmn,bhnd->bhmd", A, V)
    
        # back to [B,M,dm]
        attn = attn.transpose(1, 2).reshape(B, M, dm)

        # attention output projection (low-rank Wo)
        x_resid = (attn @ self.Uo) @ self.Vo + self.bo_attn
        x1 = self.ln1(x + x_resid)

        # FFN (low-rank)
        mid  = x1 @ self.U1
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y    = (midA @ self.U2) @ self.V2 + self.b2
        out  = self.ln2(x1 + y)
        return out



# Updated `if __name__ == "__main__":` with absolute low-rank factors memory
if __name__ == "__main__":
    import os, time, itertools, torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import BertForSequenceClassification, AutoTokenizer
    from evaluate import load as load_metric
    from torch.profiler import profile, ProfilerActivity


    BATCH_SIZE = 32
    SEQ_LEN    = 128*2
    device     = "cuda"
    RANK_ATTN  = 40 # 
    RANK_FF    = 240 # 576 # 
    RANK_WO    = 240 # 576 # 
    
    # (60, 480, 480),   # 10%
    # (56, 384, 384),   # Conservative % 25% reduction
    # (48, 336, 336),   # 35%
    # (48, 288, 288),   # Conservative % 
    # (40, 240, 240),   # Conservative % 50% reduction
    # (32, 192, 192),   # Conservative % 60%
    
    # ─── 3) Load & tokenize GLUE ──────────────────────────────────────────────
    if task_name == "mnli":
        val_split = "validation_matched"
    else:
        val_split = "validation"
    raw = load_dataset("glue", task_name, split=val_split)
    tokz = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Which GLUE tasks take one sentence vs. two?
    single_sent_tasks = {"cola", "sst2"}
    pair_sent_tasks   = {"qqp", "mnli", "qnli", "stsb", "rte", "mrpc"}
    # map each pair task to its two fields
    field_map = {
      "qqp":  ("question1",   "question2"),
      "mnli": ("premise",     "hypothesis"),
      "qnli": ("question",    "sentence"),
      "stsb": ("sentence1",   "sentence2"),
      "rte":  ("sentence1",   "sentence2"),
      "mrpc":  ("sentence1",   "sentence2"),
    }

    def tokenize_fn(batch):
        if task_name in single_sent_tasks:
            # e.g. SST-2 / CoLA
            return tokz(
                batch["sentence"],
                padding="max_length",
                truncation=True,
                max_length=SEQ_LEN,
            )
        else:
            # QQP, MNLI, QNLI, STS-B
            f1, f2 = field_map[task_name]
            return tokz(
                batch[f1],
                batch[f2],
                padding="max_length",
                truncation=True,
                max_length=SEQ_LEN,
            )

    # drop _all_ original columns except `label`
    remove_cols = [c for c in raw.column_names if c != "label"]
    ds = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_cols,
    )
    ds.set_format("torch")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: {
            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels": torch.tensor(
                [x["label"] for x in b],
                dtype=torch.float32 if task_name == "stsb" else torch.long
            ),
        },
    )


    print(f"BATCH_SIZE: {BATCH_SIZE}  RANK_ATTN: {RANK_ATTN}  RANK_FF: {RANK_FF}  RANK_WO: {RANK_WO}")

    # 3) Load & prep model in FP32
    # Choose the right metric for the task
    if task_name == "stsb":
        metric = load_metric("pearsonr")
    else:
        metric = load_metric("accuracy")
    
    #model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
    # pick the right # of labels (and problem_type for STS-B)
    if task_name == "mnli":
        num_labels = 3
        problem_type = None
    elif task_name == "stsb":
        # STS-B is a regression task
        num_labels     = 1
        problem_type   = "regression"
    else:
        # all the binary‐classification GLUE tasks
        num_labels   = 2
        problem_type = None

    # build a config that matches the checkpoint
    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    # now load with that config
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        config=cfg,
    )
    model = model.to(device).eval()
    
    # Calibrate (collect X for DRONE). Tune tokens/batches as you like.
    TOKENS_PER_LAYER = 4096
    MAX_CALIB_BATCH  = 50

    print("Calibrating activations for DRONE ...")
    X_attn_map, X_ffn_map = collect_calibration_X(model, loader, device,
                                                tokens_per_layer=TOKENS_PER_LAYER,
                                                max_batches=MAX_CALIB_BATCH)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    CACHED_ORIG_MEM = torch.cuda.max_memory_allocated()/1024**2#torch.cuda.max_memory_reserved()/1024**2#compute_persistent_memory(model)
    print(f"Persistent model storage: {CACHED_ORIG_MEM:6.1f} MiB")

    # 3.1) Persistent memory helper
    # def compute_persistent_memory(m):
    #     total = 0
    #     for p in itertools.chain(m.parameters(), m.buffers()):
    #         total += p.numel() * p.element_size()
    #     return total / (1024**2)

    # 4) Build SVD helpers
    svd_per_head, svd_low_rank = build_plain_svd_helpers(model)
    
    
    def summarize_dense_vs_lowrank(model):
        dense_bytes, lowrank_bytes = 0, 0

        for name, p in model.named_parameters():
            size = p.numel() * p.element_size()
            # assume any param under "block." is low-rank
            if ".block." in name or name.startswith("bert.encoder.layer") and any(
                part in name for part in ("Pq","Vq","Pk","Vk","Pv","Vv","U1","V1","U2","V2","Uo","Vo")
            ):
                lowrank_bytes += size
            else:
                dense_bytes   += size

        print(f"{'Type':<12}{'MiB':>8}")
        print("----------------------")
        print(f"{'Dense':<12}{dense_bytes/1024**2:8.1f}")
        print(f"{'Low-rank':<12}{lowrank_bytes/1024**2:8.1f}")
        print("----------------------")
        print(f"{'TOTAL':<12}{(dense_bytes+lowrank_bytes)/1024**2:8.1f}")
        base_mem = (dense_bytes+lowrank_bytes)
        return base_mem 


    # 5) Cache original layers
    #orig_layers = list(model.bert.encoder.layer)

    # 6) Benchmark helper
    @torch.no_grad()
    def acc_peak_time(mdl, use_mask=True):
        mdl.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        total_acc, steps = 0.0, 0
        start = time.perf_counter()
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = mdl(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"] if use_mask else None).logits
            
            # Handle predictions based on task type
            if task_name == "stsb":
                # For regression (STS-B), use raw logits as predictions
                preds = logits.squeeze(-1)  # Remove last dimension for regression
            else:
                # For classification, use argmax
                preds = torch.argmax(logits, -1)
            
            total_acc += metric.compute(
                predictions=preds.cpu(),
                references=batch["labels"].cpu()
            )["pearsonr" if task_name == "stsb" else "accuracy"]
            steps += 1
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start)*1000/steps
        peak = torch.cuda.max_memory_allocated()/1024**2
        return total_acc/steps, peak, elapsed

    # 8) Patch in low-rank SVD blocks
    #for i, layer in enumerate(orig_layers):
    for i, layer in enumerate(model.bert.encoder.layer):
        X_attn_i = X_attn_map.get(i, None)
        X_ffn_i  = X_ffn_map.get(i,  None)
        blk = SVDBlock(layer, RANK_ATTN, RANK_FF, svd_per_head, svd_low_rank,
                        RANK_WO, X_attn=X_attn_i, X_ffn=X_ffn_i)
        model.bert.encoder.layer[i] = LayerShim(blk).to(device).eval().float()

    
    
    del layer, blk, svd_per_head, svd_low_rank
    # if you built helpers per-layer, you might also need:
    for layer in model.bert.encoder.layer:
        # suppose your block stores them as attributes:
        if hasattr(layer, 'svd_per_head'):
            del layer.svd_per_head
        if hasattr(layer, 'svd_low_rank'):
            del layer.svd_low_rank

    baseline = summarize_dense_vs_lowrank(model) / 1024**2

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # we use the profile_flashsvd_offload to validate the result of above, which turn out to be accurate. 
    with_act = torch.cuda.max_memory_allocated() / 1024**2
    print(f"low-rank model storage with GPU Redundancy: {with_act:.1f} MiB")
            
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # ----- Flash-FW ------------------------------------------------------------
    CACHED_ORIG_MEM = baseline#torch.cuda.max_memory_allocated()/1024**2#compute_persistent_memory(model)
    print(f"Persistent low-rank model storage (SVD): {CACHED_ORIG_MEM:6.1f} MiB")

    # 9) LowRank SVD inference
    metric_name = "pearson" if task_name == "stsb" else "acc"
    acc, peak_lr, t = acc_peak_time(model)
    print(f"LowRank SVD     | {metric_name}={acc:.4f} | peak ={(peak_lr):6.1f} MiB | real peak ={(peak_lr-with_act + CACHED_ORIG_MEM):6.1f} MiB | Transient={(peak_lr-with_act):6.1f} MiB | {t:6.1f} ms/b")


