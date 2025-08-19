# this file contains utility functions for accuracy/inference metrics
import torch
import os, time, math, torch, pandas as pd
from torch.utils.data import DataLoader    
from datasets import load_dataset
from evaluate import load as load_metric
import itertools
from typing import Callable


# ─── 6. accuracy + memory / time helper ────────────────────────────────────
@torch.no_grad()
def acc_peak_time(mdl, loader, metric, task_name=None, device="cuda", use_mask=True):
    mdl.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    scores, n = 0.0, 0
    start = time.perf_counter()
    for batch in loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        if use_mask:
            logits = mdl(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"]).logits
        else:
            logits = mdl(input_ids=batch["input_ids"]).logits

        # Handle predictions based on task type
        if task_name == "stsb":
            # For regression (STS-B), use raw logits as predictions
            preds = logits.squeeze(-1)  # Remove last dimension for regression
        else:
            # For classification, use argmax
            preds = torch.argmax(logits, -1)

        scores += metric.compute(predictions=preds.cpu(),
                                    references=batch["labels"].cpu())["pearsonr" if task_name == "stsb" else "accuracy"]
        n += 1
    torch.cuda.synchronize()
    t = (time.perf_counter() - start)*1000.0/n
    peak = torch.cuda.max_memory_allocated()/(1024**2)
    #peak = torch.cuda.max_memory_reserved()/(1024**2)
    return scores/n, peak, t



def compute_persistent_memory(m):
        total = 0
        for p in itertools.chain(m.parameters(), m.buffers()):
            total += p.numel() * p.element_size()
        return total / (1024**2)



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


