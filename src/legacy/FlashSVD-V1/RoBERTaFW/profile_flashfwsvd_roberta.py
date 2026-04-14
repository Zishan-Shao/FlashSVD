# profile_flashfwsvd_roberta.py

import os
import sys
import time
import itertools
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, AutoTokenizer, AutoConfig
from evaluate import load as load_metric
from typing import Callable, Tuple
import math
import torch.nn.functional as F

import functools
from torch.profiler import profile, ProfilerActivity
torch.manual_seed(1919810)

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 profile_flashfwsvd_full_roberta.py 

THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
print(REPO_ROOT)
task_name = "mnli"
MODEL_DIR = os.path.join(REPO_ROOT, "models/RoBERTa", f"roberta-base-{task_name}")
from src.utils.metrics import acc_peak_time, compute_persistent_memory, summarize_dense_vs_lowrank
from src.kernels.flash_attn_triton import flash_attn_triton
from src.kernels.flashsvdattn import flash_svd_attention
from src.kernels.flashsvdffnv2 import flashsvd_ffn
from src.kernels.flashsvdffnv1 import flashsvd_ffn_v1
from src.utils.svd_helpers import build_fwsvd_helpers
from src.utils.svd_helpers import BertFWLayerShim as LayerShim
from src.utils.FlashSVDBlocks import RobertaFlashFWSVDBlock as FlashFWSVDBlock



if __name__ == "__main__":
  
    BATCH_SIZE = 64
    SEQ_LEN    = 128
    device     = "cuda"
    RANK_ATTN  = 40 #
    RANK_FF    = 240 # 
    RANK_WO    = 240 #
    
    if task_name == "mnli":
        val_split = "validation_matched"
    else:
        val_split = "validation"
    raw = load_dataset("glue", task_name, split=val_split)
    tokz = AutoTokenizer.from_pretrained(MODEL_DIR)

    single_sent_tasks = {"cola", "sst2"}
    pair_sent_tasks   = {"qqp", "mnli", "qnli", "stsb", "rte", "mrpc"}
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

    print(f"BATCH_SIZE: {BATCH_SIZE} SEQ_LEN: {SEQ_LEN}  RANK_ATTN: {RANK_ATTN}  RANK_FF: {RANK_FF}  RANK_WO: {RANK_WO}")

    if task_name == "stsb":
        metric = load_metric("pearsonr")
    else:
        metric = load_metric("accuracy")
    
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

    cfg = AutoConfig.from_pretrained(
        MODEL_DIR,
        num_labels=num_labels,
        problem_type=problem_type,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_DIR,
        config=cfg,
    )
    model = model.to(device).eval()
    model.bert = model.roberta

    CACHED_ORIG_MEM = torch.cuda.max_memory_allocated()/1024**2
    print(f"Persistent model storage: {CACHED_ORIG_MEM:6.1f} MiB")

    fwsvd_per_head, fwsvd_low_rank = build_fwsvd_helpers(
        model, loader, device=device
    )
    for i, layer in enumerate(model.roberta.encoder.layer):
        block = FlashFWSVDBlock(
            layer,
            rank_attn       = RANK_ATTN,
            rank_ff         = RANK_FF,
            fwsvd_per_head  = fwsvd_per_head,
            fwsvd_low_rank  = fwsvd_low_rank,
            rank_wo         = RANK_WO,
        )
        model.roberta.encoder.layer[i] = LayerShim(block).to(device).eval().float()

    del layer, block, fwsvd_per_head, fwsvd_low_rank
    for layer in model.roberta.encoder.layer:
        if hasattr(layer, 'fwsvd_per_head'):
            del layer.fwsvd_per_head
        if hasattr(layer, 'fwsvd_low_rank'):
            del layer.fwsvd_low_rank
    
    baseline = summarize_dense_vs_lowrank(model) / 1024**2

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with_act = torch.cuda.max_memory_allocated() / 1024**2
    print(f"low-rank model storage with GPU Redundancy: {with_act:.1f} MiB (zero if CPU decomposed)")
            
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    

    metric_name = "pearson" if task_name == "stsb" else "acc"
    
    CACHED_ORIG_MEM = baseline
    print(f"Persistent low-rank model storage (SVD): {CACHED_ORIG_MEM:6.1f} MiB")

    acc, peak_lr, t = acc_peak_time(model, loader, metric, task_name, device, use_mask=True) 
    print(f"RoBERTa LowRank FlashFWSVD     | {metric_name}={acc:.4f} | peak ={(peak_lr):6.1f} MiB | real peak ={(peak_lr-with_act + CACHED_ORIG_MEM):6.1f} MiB | Transient={(peak_lr-with_act):6.1f} MiB | {t:6.1f} ms/b")
