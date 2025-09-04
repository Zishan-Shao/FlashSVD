# profile_flashsvd_full.py — FlashSVD build using ARS ranks (argparse + budget report)
# -----------------------------------------------------------------------------
# What this does:
#   • Loads per-op ranks from --ranks_path (ars_out/ranks.json from ARS).
#   • Optional global rescale via --ars_ratio_keep (no retraining).
#   • Builds FlashSVD attention (flash_svd_attention) + FFN (v1/v2).
#   • Prints a compact budget report (params under ARS vs scaled).
#   • Evaluates accuracy/pearson on GLUE task.
# -----------------------------------------------------------------------------

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import math
import argparse
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig
from evaluate import load as load_metric

from flashsvdattn import flash_svd_attention
from flashsvdffnv1 import flashsvd_ffn_v1
from flashsvdffnv2 import flashsvd_ffn as flashsvd_ffn_v2  # keep alias


'''

# Use ARS ranks as-is
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python profile_flashsvd.py \
  --model_dir /home/zs89/FlashSVD/models/BERT/bert-base-uncased-stsb \
  --task_name stsb \
  --ranks_path BERTAda/ars_out/ranks.json \
  --ars_ratio_keep 0.85 \
  --ffn_kernel v1

# Try a tighter budget (e.g., 85% of ARS params) with Flash kernels
CUDA_VISIBLE_DEVICES=3,4,5,6,7 python profile_flashsvd.py \
  --model_dir /home/zs89/FlashSVD/models/BERT/bert-base-uncased-stsb \
  --task_name stsb \
  --ranks_path BERTAda/ars_out/ranks.json \
  --ars_ratio_keep 0.85 \
  --ffn_kernel v2


'''

# ─── SVD helpers ─────────────────────────────────────────────────────────────
def build_plain_svd_helpers(model):
    def svd_per_head(Wt: torch.Tensor, rank: int):
        d_model, _ = Wt.shape
        H          = model.config.num_attention_heads
        dh         = d_model // H
        Wt3        = Wt.view(d_model, H, dh)
        Us, Vs     = [], []
        for h in range(H):
            Wh = Wt3[:, h, :].float()  # [dm, dh]
            U32, S32, Vh32 = torch.linalg.svd(Wh, full_matrices=False)
            k = max(1, min(rank, S32.numel()))
            U = (U32[:, :k] * S32[:k]).to(Wt.dtype)  # [dm,k]
            V = Vh32[:k, :].to(Wt.dtype)            # [k,dh]
            Us.append(U); Vs.append(V)
        return torch.stack(Us, 0), torch.stack(Vs, 0)  # [H,dm,k], [H,k,dh]

    def svd_low_rank(W: torch.Tensor, rank: int):
        Wf = W.float()
        U32, S32, Vh32 = torch.linalg.svd(Wf, full_matrices=False)
        k = max(1, min(rank, S32.numel()))
        U = (U32[:, :k] * S32[:k]).to(W.dtype)   # [in,k]
        V = Vh32[:k, :].to(W.dtype)              # [k,out]
        return U, V

    return svd_per_head, svd_low_rank

# ─── Names & accounting ──────────────────────────────────────────────────────
def attach_fullnames(mdl: nn.Module, prefix=""):
    for name, mod in mdl.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(mod, nn.Linear):
            setattr(mod, "_ars_fullname", full)
        attach_fullnames(mod, full)

def full_name(mod: nn.Module) -> str:
    return getattr(mod, "_ars_fullname", "")

def collect_linears(model: nn.Module) -> Dict[str, nn.Linear]:
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            out[name] = mod
    return out

def cap_full_rank(lin: nn.Linear) -> int:
    return min(lin.in_features, lin.out_features)

def op_cost_per_rank(lin: nn.Linear) -> int:
    # SVD factor cost per rank: (in + out)
    return lin.in_features + lin.out_features

def build_scaled_ranks(model: nn.Module, ars_ranks: Dict[str, int], ratio_keep: float) -> Dict[str, int]:
    """
    Scale ARS ranks globally by ratio_keep; greedy water-filling to meet budget after rounding.
    """
    linears = collect_linears(model)
    caps    = {n: cap_full_rank(m) for n, m in linears.items()}
    costs   = {n: op_cost_per_rank(m) for n, m in linears.items()}

    base = {n: int(ars_ranks.get(n, caps[n])) for n in linears}  # ARS or full
    T_base = sum(costs[n] * base[n] for n in linears)
    target = int(math.floor(ratio_keep * T_base + 1e-6))

    k = {n: max(1, min(caps[n], int(round(base[n] * ratio_keep)))) for n in linears}
    T_now = sum(costs[n] * k[n] for n in linears)
    if T_now <= target:
        return k

    # Greedy: decrement the op with the largest cost per rank until under target.
    import heapq
    heap = [(-costs[n], n) for n in linears]; heapq.heapify(heap)
    while T_now > target and heap:
        negc, name = heapq.heappop(heap)
        c = -negc
        if k[name] > 1:
            k[name] -= 1
            T_now -= c
            heapq.heappush(heap, (-c, name))
    return k

# ─── FlashSVD block using per-op ranks ───────────────────────────────────────
class FlashSVDBlock(nn.Module):
    def __init__(self, hf_layer, rkdict: dict, svd_per_head: Callable, svd_low_rank: Callable, ffn_kernel: str = "v1"):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H

        # Submodules
        q_lin = hf_layer.attention.self.query
        k_lin = hf_layer.attention.self.key
        v_lin = hf_layer.attention.self.value
        o_lin = hf_layer.attention.output.dense
        i_lin = hf_layer.intermediate.dense
        o2_lin= hf_layer.output.dense

        def rk(m):
            name = full_name(m)
            return max(1, int(rkdict.get(name, 1)))

        # Q/K/V factorization per head
        WqT, bq = q_lin.weight.data.t(), q_lin.bias.data.view(1,H,1,dh)
        WkT, bk = k_lin.weight.data.t(), k_lin.bias.data.view(1,H,1,dh)
        WvT, bv = v_lin.weight.data.t(), v_lin.bias.data.view(1,H,1,dh)

        Uq, Vq = svd_per_head(WqT, rk(q_lin))
        Uk, Vk = svd_per_head(WkT, rk(k_lin))
        Uv, Vv = svd_per_head(WvT, rk(v_lin))

        self.Pq, self.Vq = map(nn.Parameter, (Uq, Vq))      # [H,dm,k], [H,k,dh]
        self.Pk, self.Vk = map(nn.Parameter, (Uk, Vk))
        self.Pv, self.Vv = map(nn.Parameter, (Uv, Vv))
        self.bq, self.bk, self.bv = map(nn.Parameter, (bq, bk, bv))

        # FFN
        Wi, bi   = i_lin.weight.data.t(),  i_lin.bias.data
        WoT, bo2 = o2_lin.weight.data.t(), o2_lin.bias.data
        U1, V1 = svd_low_rank(Wi,  rk(i_lin))
        U2, V2 = svd_low_rank(WoT, rk(o2_lin))
        self.U1, self.V1, self.b1 = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2 = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        # Attn output projection
        Wo_full = o_lin.weight.data
        bo_attn = o_lin.bias.data
        Uo, Vo  = svd_low_rank(Wo_full.t(), rk(o_lin))
        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

        assert ffn_kernel in ("v1", "v2")
        self.ffn_kernel = ffn_kernel

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R     = self.Pq.shape[0], self.Pq.shape[2]
        dh       = dm // H

        # project into rank-R head spaces: x:[B,M,dm] • P:[H,dm,R] → tmp:[B,H,M,R]
        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq)
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk)
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv)

        Vq_full = self.Vq.expand(B, H, R, dh)
        Vk_full = self.Vk.expand(B, H, R, dh)
        Vv_full = self.Vv.expand(B, H, R, dh)
        bq_full = self.bq.expand(B, H, 1, dh).squeeze(2)
        bk_full = self.bk.expand(B, H, 1, dh).squeeze(2)
        bv_full = self.bv.expand(B, H, 1, dh).squeeze(2)

        mask4 = mask.view(B,1,1,M) if mask is not None else None

        # FlashSVD attention core
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()

        attn = attn_out.view(B, H, M, dh).transpose(1, 2).reshape(B, M, dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)

        # FlashSVD FFN
        mid = x1 @ self.U1
        if self.ffn_kernel == "v1":
            y = flashsvd_ffn_v1(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        else:
            y = flashsvd_ffn_v2(mid, self.V1, self.U2, self.V2, self.b1, self.b2)

        out = self.ln2(x1 + y)
        return out

# ─── Layer shim ──────────────────────────────────────────────────────────────
class LayerShim(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
    def forward(self, hidden_states, attention_mask=None, *args, **kwargs):
        raw_mask = attention_mask
        if attention_mask is not None and attention_mask.dim() == 4:
            raw_mask = (attention_mask[:,0,0,:] == 0)
        return (self.block(hidden_states, raw_mask),)

# ─── CLI + main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FlashSVD build/eval from ARS ranks.")
    parser.add_argument("--task_name", type=str, default="stsb")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to HF checkpoint dir")
    parser.add_argument("--ranks_path", type=str, required=True, help="Path to ars_out/ranks.json")
    parser.add_argument("--ars_ratio_keep", type=float, default=1.0, help="Scale ARS ranks globally (1.0 = as-is)")
    parser.add_argument("--seq_len", type=int, default=256*2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ffn_kernel", type=str, default="v1", choices=["v1","v2"])
    args = parser.parse_args()

    device = args.device if (args.device in ["cuda","cpu"]) else ("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    val_split = "validation_matched" if args.task_name == "mnli" else "validation"
    raw = load_dataset("glue", args.task_name, split=val_split)
    tokz = AutoTokenizer.from_pretrained(args.model_dir)

    single_sent = {"cola", "sst2"}
    field_map = {
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "stsb": ("sentence1", "sentence2"),
        "rte":  ("sentence1", "sentence2"),
        "mrpc": ("sentence1", "sentence2"),
    }

    def tokenize_fn(batch):
        if args.task_name in single_sent:
            return tokz(batch["sentence"], padding="max_length", truncation=True, max_length=args.seq_len)
        else:
            f1, f2 = field_map[args.task_name]
            return tokz(batch[f1], batch[f2], padding="max_length", truncation=True, max_length=args.seq_len)

    remove_cols = [c for c in raw.column_names if c != "label"]
    ds = raw.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: {
                            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                            "labels":         torch.tensor([x["label"]         for x in b]),
                        })

    # Model
    if args.task_name == "mnli":
        num_labels, problem_type = 3, None
    elif args.task_name == "stsb":
        num_labels, problem_type = 1, "regression"
    else:
        num_labels, problem_type = 2, None

    cfg = AutoConfig.from_pretrained(args.model_dir, num_labels=num_labels, problem_type=problem_type)
    model = BertForSequenceClassification.from_pretrained(args.model_dir, config=cfg).to(device).eval()

    # Attach names and load ranks
    attach_fullnames(model)
    if not os.path.exists(args.ranks_path):
        raise FileNotFoundError(f"Could not find ranks file: {args.ranks_path}")
    with open(args.ranks_path, "r") as f:
        ars_ranks = json.load(f)

    # Build scaled ranks + budget report
    linears = collect_linears(model)
    caps    = {n: cap_full_rank(m) for n, m in linears.items()}
    costs   = {n: op_cost_per_rank(m) for n, m in linears.items()}

    base = {n: int(ars_ranks.get(n, caps[n])) for n in linears}
    T_base = sum(costs[n] * base[n] for n in linears)

    rkdict = build_scaled_ranks(model, ars_ranks=ars_ranks, ratio_keep=args.ars_ratio_keep)
    T_scaled = sum(costs[n] * rkdict[n] for n in rkdict)
    achieved = (T_scaled / (T_base + 1e-12)) if T_base > 0 else 1.0

    print("\n=== FlashSVD Inference Budget Report ===")
    print(f"ops               : {len(linears)}")
    print(f"ARS base params   : {T_base:,}")
    print(f"scaled params     : {T_scaled:,}  (keep={args.ars_ratio_keep:.3f})")
    print(f"achieved ratio    : {achieved*100:.2f}%")
    print(f"saved vs ARS base : {100*(1-achieved):.2f}%\n")

    # Build SVD helpers
    svd_per_head, svd_low_rank = build_plain_svd_helpers(model)

    # Patch encoder with FlashSVD blocks (per-op ranks)
    for i, layer in enumerate(model.bert.encoder.layer):
        blk = FlashSVDBlock(layer, rkdict, svd_per_head, svd_low_rank, ffn_kernel=args.ffn_kernel)
        model.bert.encoder.layer[i] = LayerShim(blk).to(device).eval().float()

    # --- Eval ---
    @torch.no_grad()
    def acc_peak_time(mdl):
        mdl.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if args.task_name == "stsb":
            metric = load_metric("pearsonr"); key = "pearsonr"
        else:
            metric = load_metric("accuracy"); key = "accuracy"
        total, steps = 0.0, 0
        start = time.perf_counter()
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = mdl(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            preds = logits.squeeze(-1) if args.task_name == "stsb" else torch.argmax(logits, -1)
            total += metric.compute(predictions=preds.cpu(), references=batch["labels"].cpu())[key]
            steps += 1
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start)*1000/steps
        peak = torch.cuda.max_memory_allocated()/1024**2
        return total/steps, peak, elapsed

    metric_name = "pearson" if args.task_name == "stsb" else "acc"
    score, peak_mem, ms = acc_peak_time(model)
    print(f"FlashSVD (ARS) | {metric_name}={score:.4f} | peak ={peak_mem:6.1f} MiB | {ms:6.1f} ms/b")

if __name__ == "__main__":
    main()

