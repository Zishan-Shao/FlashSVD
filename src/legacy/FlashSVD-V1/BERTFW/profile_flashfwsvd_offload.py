# profile_flashfwsvd_offload.py
# the only thing different of this script to the non-offload on is that this will compute SVD on CPU, which will lead to large latency overhead (one-time)
# however, this will lead to less fragmentation in terms of memory

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

import functools
from torch.profiler import profile, ProfilerActivity
import pandas as pd
import gc
torch.manual_seed(114514)


# we need to access this directory first
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
task_name = "stsb"
print(REPO_ROOT)
MODEL_DIR = os.path.join(REPO_ROOT, "models/BERT", f"bert-base-uncased-{task_name}")
from src.utils.metrics import acc_peak_time, compute_persistent_memory, summarize_dense_vs_lowrank
from src.kernels.flash_attn_triton import flash_attn_triton
from src.kernels.flashsvdattn import flash_svd_attention
from src.kernels.flashsvdffnv2 import flashsvd_ffn
from src.kernels.flashsvdffnv1 import flashsvd_ffn_v1
from src.utils.svd_helpers import build_fwsvd_helpers
from src.utils.svd_helpers import BertFWLayerShim as LayerShim
from src.utils.FlashSVDBlocks import BertFlashFWSVDBlock as FlashFWSVDBlock



torch.manual_seed(114514)



if __name__ == "__main__":
        
    BATCH_SIZE = 64
    SEQ_LEN    = 128
    device     = "cuda"
    RANK_ATTN  = 40 # 
    RANK_FF    = 240 #  
    RANK_WO    = 240 #  
    
    # We'll split work between CPU (prep) and GPU (inference)
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda")


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
    # instead of using the full loader, use a smaller one for faster SVD
    small_loader = DataLoader(ds.shuffle(seed=0).select(range(128)),
                            batch_size=BATCH_SIZE, 
                            collate_fn=lambda b: {
            "input_ids":      torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "labels":         torch.tensor([x["label"]     for x in b]),
        }
    )
    
    print(f"BATCH_SIZE: {BATCH_SIZE} SEQ_LEN: {SEQ_LEN}  RANK_ATTN: {RANK_ATTN}  RANK_FF: {RANK_FF}  RANK_WO: {RANK_WO}")

    # 3) Load & prep model in FP32
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
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR, config=cfg,
    ).to(cpu_device)
    model.eval()
    
    # baseline CPU memory (just for reference)
    print(f"CPU model storage: {torch.cuda.memory_allocated():6.1f} MiB "
          "(should be 0 MiB since we're on CPU)")

    # ─── 2) Build FW-SVD helpers entirely on CPU ────────────────────────────────
    fwsvd_per_head, fwsvd_low_rank = build_fwsvd_helpers(
        model, small_loader, device=cpu_device
    )

    # ─── 3) Patch each layer with a CPU-only FlashFWSVDBlock ────────────────────
    for i, hf_layer in enumerate(model.bert.encoder.layer):
        block = FlashFWSVDBlock(
            hf_layer,
            rank_attn     = RANK_ATTN,    # or RANK_ATTN
            rank_ff       = RANK_FF,   # or RANK_FF
            fwsvd_per_head= fwsvd_per_head,
            fwsvd_low_rank= fwsvd_low_rank,
            rank_wo       = RANK_WO    # or RANK_WO
        )
        # keep everything on CPU for now
        model.bert.encoder.layer[i] = LayerShim(block).to(cpu_device).eval().float()

    # drop helper refs & run GC to free any CPU‐side fisher intermediates
    del fwsvd_per_head, fwsvd_low_rank, block, hf_layer
    gc.collect()

    # ─── 4) Move finalized low-rank model to GPU ────────────────────────────────
    model = model.to(gpu_device).eval()

    # clear GPU stats so baseline is just the parameters
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with_act = torch.cuda.memory_allocated() / 1024**2
    print(f"low-rank model storage with GPU Redundancy: {with_act:.1f} MiB")
            
    # ----- Flash-FW ------------------------------------------------------------
    CACHED_ORIG_MEM = with_act#torch.cuda.max_memory_allocated()/1024**2#compute_persistent_memory(model)
    print(f"Persistent low-rank model storage (FWSVD): {CACHED_ORIG_MEM:6.1f} MiB")

    # ----- FW-PT ---------------------------------------------------------------
    acc, peak_lr, t = acc_peak_time(model, loader, metric, task_name, gpu_device, use_mask=True) 
    print(f"FlashSVD     | acc={acc:.4f} | peak ={(peak_lr-with_act+CACHED_ORIG_MEM):6.1f} MiB | Transient={(peak_lr-with_act):6.1f} MiB | {t:6.1f} ms/b")

