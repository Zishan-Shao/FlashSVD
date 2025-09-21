# profile_flashsvd_full.py

import os
import sys
import time
import itertools
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer
from evaluate import load as load_metric
from flashsvdattn import flash_svd_attention
from flashsvdffn import flashsvd_ffn
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
MODEL_DIR = os.path.join(REPO_ROOT, "model", "bert-base-uncased-sst2")

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

        # print("================ New Layer ================")
        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.synchronize()
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
        # print("FlashSVD Peak:", torch.cuda.max_memory_allocated()/1024**2)

        
        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.synchronize()
        
        x1   = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn) # Q: what is the memory complexity of this one?

        mid = x1 @ self.U1 
        #y   = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
        
        # If we do not use the below inference, it will slow down the process
        midV = mid @ self.V1
        midA = F.gelu(midV + self.b1)
        y    = (midA @ self.U2) @ self.V2 + self.b2
        
        out = self.ln2(x1 + y)
        #print("FlashSVDFFN Peak:", torch.cuda.max_memory_allocated()/1024**2)
        
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
    
    
    BATCH_SIZE = 8*4*2
    SEQ_LEN    = 128*4
    device    = "cuda" 
    RANK_ATTN = 32 // 2#*2
    RANK_FF   = 384 // 2 #*2  # biggest overhead is the FFN, we can reduce the latency by: 1. store entire tile of D, or 2. store a single slice.
    RANK_WO   = 384 // 2 #*2
    
    raw = load_dataset("glue", "sst2", split="validation")
    tokz = AutoTokenizer.from_pretrained(MODEL_DIR)
    def tokenize_fn(batch):
        return tokz(batch["sentence"],
                    padding="max_length", truncation=True, max_length=SEQ_LEN)

    ds = raw.map(tokenize_fn, batched=True, remove_columns=["sentence","idx"])
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: {
                            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                            "labels":         torch.tensor([x["label"]         for x in b]),
                        })


    print(f"BATCH_SIZE: {BATCH_SIZE}  RANK_ATTN: {RANK_ATTN}  RANK_FF: {RANK_FF}  RANK_WO: {RANK_WO}")

    acc_metric = load_metric("accuracy")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
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
    for i, layer in enumerate(model.bert.encoder.layer):
        blk = FlashFWSVDBlock(layer, RANK_ATTN, RANK_FF, svd_per_head, svd_low_rank, RANK_WO)
        model.bert.encoder.layer[i] = LayerShim(blk).to(device).eval().float()
    
    del layer, blk, svd_per_head, svd_low_rank
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    CACHED_ORIG_MEM = torch.cuda.max_memory_allocated() / 1024**2#compute_persistent_memory(model)
    print(f"Low-rank factors absolute storage: {CACHED_ORIG_MEM:6.1f} MiB")    
    
    
    # ——— 4) Warm-up ———
    batch = next(iter(loader))
    inp = batch["input_ids"].to(device)
    msk = batch["attention_mask"].to(device)
    
    # warm up once so lazy allocations happen
    with torch.no_grad():
        _ = model(input_ids=inp, attention_mask=msk)

    # clear and sync
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # run your real inference
    with torch.no_grad():
        _ = model(input_ids=inp, attention_mask=msk)
    torch.cuda.synchronize()

    # peak allocated vs. reserved
    #peak_alloc = torch.cuda.max_memory_reserved()   / 1024**2
    peak_res  = torch.cuda.max_memory_allocated()   / 1024**2#torch.cuda.max_memory_reserved() / 1024**2
    scratch   = peak_res - CACHED_ORIG_MEM
    
    print(f"Peak Memory: {peak_res:.1f} MiB")
    print(f"Transient scratch: {scratch:.1f} MiB")
    
    # # ——— 4) Warm-up ———
    # batch = next(iter(loader))
    # inp = batch["input_ids"].to(device)
    # msk = batch["attention_mask"].to(device)
    # with torch.no_grad():
    #     _ = model(input_ids=inp, attention_mask=msk)
    
    # ——— 5) Memory summary + profiling ———
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # print("\n=== CUDA MEMORY SUMMARY ===")
    # print("\n".join(torch.cuda.memory_summary().splitlines()[:30]))

    # with profile(
    #     activities=[ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
    #     with torch.no_grad():
    #         _ = model(input_ids=inp, attention_mask=msk)

    # print("\n=== TOP 10 OPS BY GPU MEM USAGE ===")
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=30))

    
    # ——— 6) Transient scratch ———
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input_ids=inp, attention_mask=msk)
    torch.cuda.synchronize()

    total_peak = torch.cuda.max_memory_allocated() / 1024**2#torch.cuda.max_memory_reserved() / 1024**2
    scratch    = total_peak - CACHED_ORIG_MEM
    print(f"\nFlash-LowRank total peak:        {total_peak:6.1f} MiB")
    print(f"Flash-LowRank transient scratch: {scratch:6.1f} MiB")
    
    # ——— 7) Validation accuracy ———
    print("\nRunning full validation pass to compute accuracy…")
    total_acc = 0.0
    steps     = 0
    start = time.perf_counter()
    for batch in loader:
        inp = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=inp, attention_mask=msk).logits
        preds = torch.argmax(logits, dim=-1).cpu()
        total_acc += acc_metric.compute(
            predictions=preds,
            references=batch["labels"].cpu()
        )["accuracy"]
        steps += 1
    torch.cuda.synchronize()
    t = (time.perf_counter() - start)*1000/steps 

    print(f"Validation accuracy: {total_acc/steps:.4f}, Latency: {t:6.1f} ms/b")








        
        
    # # ——— Warm-up once (lazy allocs) ———
    # batch = next(iter(loader))
    # inp = batch["input_ids"].to(device)
    # msk = batch["attention_mask"].to(device)
    # with torch.no_grad():
    #     _ = model(input_ids=inp, attention_mask=msk)

    # # ——— Transient scratch measurement ———
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.synchronize()
    # with torch.no_grad():
    #     _ = model(input_ids=inp, attention_mask=msk)
    # torch.cuda.synchronize()

    # peak_alloc = torch.cuda.max_memory_reserved() / 1024**2
    # peak_res   = torch.cuda.max_memory_reserved()  / 1024**2
    # scratch    = peak_alloc - CACHED_ORIG_MEM
    # frag       = peak_res   - peak_alloc

    # print(f"Peak allocated:           {peak_alloc:.1f} MiB")
    # print(f"Transient scratch:        {scratch:.1f} MiB")
    # print(f"Allocator fragmentation:  {frag:.1f} MiB")

    # # ——— Profile + op-level breakdown ———
    # torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats()
    # torch.cuda.synchronize()

    # with profile(
    #     activities=[ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
    #     with torch.no_grad():
    #         _ = model(input_ids=inp, attention_mask=msk)

    # # Use the right key for your build
    # print("device_memory_usage")
    # print(prof.key_averages()
    #     .table(sort_by="device_memory_usage", row_limit=10))