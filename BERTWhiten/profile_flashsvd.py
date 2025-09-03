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
from flashsvdattn import flash_svd_attention
from flashsvdffn import flashsvd_ffn
from flashsvdffnv1 import flashsvd_ffn_v1
from typing import Callable, Tuple
import math
import torch.nn.functional as F

import functools
import torch



# ─── locate repo & model ─────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.fwsvd import (
    compute_row_sum_svd_decomposition,
    estimate_fisher_weights_bert,
    estimate_fisher_weights_bert_with_attention,
)
task_name = "stsb"
MODEL_DIR = os.path.join(REPO_ROOT, "model", f"bert-base-uncased-{task_name}")

# ─── 0) Helpers for SVD decomposition ─────────────────────────────────────────
# TODO: change this method so it can perform Data Whitening
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



# ─── 1) Flash-SVD block ───────────────────────────────────────────────────────
class FlashFWSVDBlock(nn.Module):
    def __init__(self, hf_layer, rank_attn, rank_ff, svd_per_head, svd_low_rank, rank_wo):
        super().__init__()
        cfg     = hf_layer.attention.self
        d_model = cfg.all_head_size
        H       = cfg.num_attention_heads
        dh      = d_model // H
        
        # factor Q/K/V
        WqT, bq = cfg.query.weight.data.t(), cfg.query.bias.data.view(1,H,1,dh)
        WkT, bk = cfg.key.weight.data.t(),   cfg.key.bias.data.view(1,H,1,dh)
        WvT, bv = cfg.value.weight.data.t(), cfg.value.bias.data.view(1,H,1,dh)
        self.Pq, self.Vq = map(nn.Parameter, svd_per_head(WqT, rank_attn))
        self.Pk, self.Vk = map(nn.Parameter, svd_per_head(WkT, rank_attn))
        self.Pv, self.Vv = map(nn.Parameter, svd_per_head(WvT, rank_attn))
        self.bq, self.bk, self.bv = map(nn.Parameter, (bq,bk,bv))

        # factor FFN
        Wi, bi   = hf_layer.intermediate.dense.weight.data.t(), hf_layer.intermediate.dense.bias.data
        WoT, bo2 = hf_layer.output.dense.weight.data.t(),      hf_layer.output.dense.bias.data
        self.U1, self.V1 = map(nn.Parameter, svd_low_rank(Wi,   rank_ff))
        self.U2, self.V2 = map(nn.Parameter, svd_low_rank(WoT, rank_ff))
        self.b1, self.b2 = map(nn.Parameter, (bi, bo2))

        # output projection (attn)
        Wo_full  = hf_layer.attention.output.dense.weight.data
        bo_attn  = hf_layer.attention.output.dense.bias.data
        self.Uo, self.Vo = map(nn.Parameter, svd_low_rank(Wo_full.t(), rank_wo))
        self.bo_attn    = nn.Parameter(bo_attn)

        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        H, R     = self.Pq.shape[0], self.Pq.shape[2]
        dh       = dm // H

        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq)
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk)
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv)

        Vq_full = self.Vq.expand(B,H,R,dh)
        Vk_full = self.Vk.expand(B,H,R,dh)
        Vv_full = self.Vv.expand(B,H,R,dh)
        bq_full = self.bq.expand(B,H,1,dh).squeeze(2)
        bk_full = self.bk.expand(B,H,1,dh).squeeze(2)
        bv_full = self.bv.expand(B,H,1,dh).squeeze(2)

        mask4 = mask.view(B,1,1,M) if mask is not None else None
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()
        
        attn = attn_out.view(B,H,M,dh).transpose(1,2).reshape(B,M,dm)
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) # Q: what is the memory complexity of this one?

        mid = x1 @ self.U1 
        # flashsvdffn v2
        #y   = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        
        # flashsvdffn v1
        y = flashsvd_ffn_v1(
            mid,       # P
            self.V1,   # V1
            self.U2,   # U2
            self.V2,   # V2
            self.b1,   # b1
            self.b2    # b2
            # you can optionally override BL, BD, BR1, BR2 here
            # e.g. , BL=64, BD=128, BR1=32, BR2=32
        )
        
        # # If we do not use the below inference, it will slow down the process
        # midV = mid @ self.V1
        # midA = F.gelu(midV + self.b1)
        # y    = (midA @ self.U2) @ self.V2 + self.b2
        
        out = self.ln2(x1 + y)
        return out 



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
        total_acc += acc_metric.compute(
            predictions=torch.argmax(logits, -1).cpu(),
            references=batch["labels"].cpu()
        )["accuracy"]
        steps += 1
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start)*1000/steps
    peak = torch.cuda.max_memory_allocated()/1024**2#torch.cuda.max_memory_reserved()/1024**2
    return total_acc/steps, peak, elapsed



if __name__ == "__main__":
    import os, time, torch
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import BertForSequenceClassification, AutoTokenizer
    from evaluate import load as load_metric
    from flashsvdattn import flash_svd_attention
    from flashsvdffn import flashsvd_ffn
    from torch.profiler import profile, ProfilerActivity
    
    
    BATCH_SIZE = 32
    SEQ_LEN    = 128*2 
    device    = "cuda" 
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
            "labels":         torch.tensor([x["label"]         for x in b]),
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

    # def param_bytes(model):
    #     return sum(p.numel() * p.element_size() for p in model.parameters())

    # orig_params  = param_bytes(orig_model)  # before low-rank
    # lowrank_params = param_bytes(model)     # after
    # print(f"Orig params:     {orig_params/1024**2:.1f} MiB")
    # print(f"Low-rank params: {lowrank_params/1024**2:.1f} MiB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # ——— 3) Persistent & low-rank memory ———
    CACHED_ORIG_MEM = torch.cuda.max_memory_allocated()/1024**2#torch.cuda.max_memory_reserved()/1024**2#compute_persistent_memory(model)
    print(f"Persistent model storage: {CACHED_ORIG_MEM:6.1f} MiB")
    
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
    

    for i, layer in enumerate(model.bert.encoder.layer):
        blk = FlashFWSVDBlock(layer, RANK_ATTN, RANK_FF, svd_per_head, svd_low_rank, RANK_WO)
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

    # we use the profile_flashfwsvd_offload to validate the result of above, which turn out to be accurate. 
    with_act = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Flash low-rank model storage with GPU Redundancy: {with_act:.1f} MiB")
            
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # ----- Flash-FW ------------------------------------------------------------
    CACHED_ORIG_MEM = baseline#torch.cuda.max_memory_allocated()/1024**2#compute_persistent_memory(model)
    print(f"Persistent low-rank model storage (SVD): {CACHED_ORIG_MEM:6.1f} MiB")
    
    
    # del layer, blk, svd_per_head, svd_low_rank
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.synchronize()
    
    # CACHED_ORIG_MEM = torch.cuda.max_memory_allocated() / 1024**2#compute_persistent_memory(model)
    # print(f"Low-rank factors absolute storage: {CACHED_ORIG_MEM:6.1f} MiB")   
    
     # 9) LowRank SVD inference
    metric_name = "pearson" if task_name == "stsb" else "acc"
    acc, peak_lr, t = acc_peak_time(model)
    print(f"FlashSVD     | {metric_name}={acc:.4f} | peak ={(peak_lr):6.1f} MiB | real peak ={(peak_lr-with_act + CACHED_ORIG_MEM):6.1f} MiB | Transient={(peak_lr-with_act):6.1f} MiB | {t:6.1f} ms/b") 
    
    