# bert.py should load the pre-computed low-rank factors for computation
#         it should not redo the SVD, but load it directly

# Update 6.26.2025: We have added the FWSVD along with vanilla SVD here for computation
# Issue: we should expect the activation memory of the dense be smaller than the memory of the flashsvd
#        but in here we see the activation of flashsvd be slightly larger --> 311 v.s. 274
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel

import time
import pandas as pd

from transformers import BertForSequenceClassification, AutoConfig
from typing import Callable



torch.manual_seed(0)


# we need to access this directory first
# use sys.path(), where the src.fwsvd is located at /home/zs89/FlashSVDFFN/src/fwsvd
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


def compute_persistent_memory(m):
        total = 0
        for p in itertools.chain(m.parameters(), m.buffers()):
            total += p.numel() * p.element_size()
        return total / (1024**2)


if __name__ == "__main__":
    import os, time, math, torch, pandas as pd
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import BertForSequenceClassification, AutoTokenizer
    from evaluate import load as load_metric
    import itertools
    
    # ─── 1. dataset & loader ────────────────────────────────────────────────────
    BATCH_SIZE = 8*4
    SEQ_LEN    = 128*2
    device     = "cuda"
    
    
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

    print(f"BATCH_SIZE: {BATCH_SIZE}  SEQ_LEN: {SEQ_LEN}")
    
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
    
    CACHED_ORIG_MEM = compute_persistent_memory(model)
    print(f"Low-rank factors absolute storage: {CACHED_ORIG_MEM:6.1f} MiB")

    
    # ─── 6. accuracy + memory / time helper ────────────────────────────────────
    @torch.no_grad()
    def acc_peak_time(mdl, use_mask=True):
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

    # ─── 7. Dense baseline ─────────────────────────────────────────────────────
    metric_name = "pearson" if task_name == "stsb" else "acc"
    
    acc, peak_m, t = acc_peak_time(model, use_mask=True) 
    print(f"Dense   w/ mask   | {metric_name}={acc:.4f} | peak={peak_m:6.1f} MiB | transient={peak_m - CACHED_ORIG_MEM:6.1f} MiB | {t:6.1f} ms/b")
    acc, peak_nm, t = acc_peak_time(model, use_mask=False)
    print(f"Dense   w/o mask  | {metric_name}={acc:.4f} | peak={peak_nm:6.1f} MiB | transient={peak_m - CACHED_ORIG_MEM:6.1f} MiB | {t:6.1f} ms/b")
    BASE_PEAK = max(peak_m, peak_nm)

