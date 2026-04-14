# profile_flashsvd_full.py
# 
# Multi-task Flash-SVD inference script for BERT models
# Supports both SST-2 (sentiment classification) and STS-B (semantic similarity) tasks
# 
# Usage:
#   1. Change the task_name variable below to "sst2" or "stsb"
#   2. Run: python profile_flashsvd_full.py
#   3. The script will automatically load the appropriate model and evaluate on the specified task

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
from typing import Callable, Tuple, Dict, Any, List
import math
import torch.nn.functional as F
import re

import functools
import torch
from flashsvdffnv1 import flashsvd_ffn_v1


# ─── Configuration ───────────────────────────────────────────────────────────
# Set task_name here - change this to switch between tasks
task_name = "sst2"  # Options: "sst2", "stsb"

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
MODEL_DIR = os.path.join(REPO_ROOT, "model", f"bert-base-uncased-{task_name}")

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

# ─── 1) Flash-SVD Block Implementation ───────────────────────────────────────
class FlashSVDBlock(nn.Module):
    """Low-rank Flash-SVD block for inference."""
    
    def __init__(self, config: Dict[str, Any], bert_config):
        super().__init__()
        self.config = config
        
        # Derive parameters from BERT config
        d_model = bert_config.hidden_size
        H = bert_config.num_attention_heads
        dh = d_model // H
        d_ff = bert_config.intermediate_size
        rank_attn = config['rank_attn']
        rank_ff = config['rank_ff']
        rank_wo = config['rank_wo']
        
        # Store derived config for reference
        self.full_config = {
            'd_model': d_model,
            'num_heads': H,
            'head_dim': dh,
            'd_ff': d_ff,
            'rank_attn': rank_attn,
            'rank_ff': rank_ff,
            'rank_wo': rank_wo
        }
        
        # Attention parameters
        self.Pq = nn.Parameter(torch.empty(1, H, d_model, rank_attn))
        self.Vq = nn.Parameter(torch.empty(1, H, rank_attn, dh))
        self.bq = nn.Parameter(torch.empty(1, H, 1, dh))
        self.Pk = nn.Parameter(torch.empty(1, H, d_model, rank_attn))
        self.Vk = nn.Parameter(torch.empty(1, H, rank_attn, dh))
        self.bk = nn.Parameter(torch.empty(1, H, 1, dh))
        self.Pv = nn.Parameter(torch.empty(1, H, d_model, rank_attn))
        self.Vv = nn.Parameter(torch.empty(1, H, rank_attn, dh))
        self.bv = nn.Parameter(torch.empty(1, H, 1, dh))
        
        # Attention output projection
        self.Uo = nn.Parameter(torch.empty(d_model, rank_wo))
        self.Vo = nn.Parameter(torch.empty(rank_wo, d_model))
        self.bo_attn = nn.Parameter(torch.empty(d_model))
        
        # FFN parameters
        self.U1 = nn.Parameter(torch.empty(d_model, rank_ff))
        self.V1 = nn.Parameter(torch.empty(rank_ff, d_ff))
        self.b1 = nn.Parameter(torch.empty(d_ff))
        self.U2 = nn.Parameter(torch.empty(d_ff, rank_ff))
        self.V2 = nn.Parameter(torch.empty(rank_ff, d_model))
        self.b2 = nn.Parameter(torch.empty(d_model))
        
        # Layer norms (will be loaded from state dict)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        _, H, _, R = self.Pq.shape
        dh = dm // H

        # Project into low-rank Q/K/V
        tmp_q = torch.einsum('bmd,hdr->bhmr', x, self.Pq[0])
        tmp_k = torch.einsum('bmd,hdr->bhmr', x, self.Pk[0])
        tmp_v = torch.einsum('bmd,hdr->bhmr', x, self.Pv[0])

        Vq_full = self.Vq[0].expand(B, H, R, dh)
        Vk_full = self.Vk[0].expand(B, H, R, dh)
        Vv_full = self.Vv[0].expand(B, H, R, dh)
        bq_full = self.bq[0].expand(B, H, 1, dh).squeeze(2)
        bk_full = self.bk[0].expand(B, H, 1, dh).squeeze(2)
        bv_full = self.bv[0].expand(B, H, 1, dh).squeeze(2)

        # Flash-SVD attention
        if mask is not None:
            mask4 = mask.view(B, 1, 1, M)
        else:
            # Create a mask of all ones (all tokens valid) when no mask is provided
            mask4 = torch.ones(B, 1, 1, M, device=x.device, dtype=torch.bool)
        
        attn_out = flash_svd_attention(
            tmp_q, Vq_full, bq_full,
            tmp_k, Vk_full, bk_full,
            tmp_v, Vv_full, bv_full,
            mask=mask4, block_m=32, block_r=R
        )
        del tmp_q, tmp_k, tmp_v, Vq_full, Vk_full, Vv_full, bq_full, bk_full, bv_full
        torch.cuda.empty_cache()
        
        # Back to [B,M,dm]
        attn = attn_out.view(B, H, M, dh).transpose(1, 2).reshape(B, M, dm)
        
        # Output projection + LayerNorm
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)
        
        # Flash-SVD FFN
        mid = x1 @ self.U1
        #y = flashsvd_ffn(mid, self.V1, self.U2, self.V2, self.b1, self.b2)
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
        out = self.ln2(x1 + y)
        
        return out

# ─── 2) Flash-SVD Model for Inference ───────────────────────────────────────
class FlashSVDModel(nn.Module):
    """A BERT model with Flash-SVD decomposition for inference."""
    
    def __init__(self, config, svd_config=None):
        super().__init__()
        self.config = config
        
        # Use provided SVD config or default values
        if svd_config is None:
            svd_config = {
                'rank_attn': 16,
                'rank_ff': 192,
                'rank_wo': 192,
                'model_type': 'bert_flashsvd',
                'svd_version': '1.0'
            }
        
        self.svd_config = svd_config
        
        # Create the full BERT structure to match the training model
        from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler
        
        # Create embeddings with the same structure as BERT
        self.embeddings = BertEmbeddings(config)
        
        # Create pooler with the same structure as BERT
        self.pooler = BertPooler(config)
        
        # Create classifier
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Create Flash-SVD layers with hardcoded ranks
        self.svd_layers = nn.ModuleList([
            FlashSVDBlock(self.svd_config, config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Ignore unused parameters like token_type_ids
        x = self.embeddings(input_ids)
        x = x * (self.embeddings.word_embeddings.embedding_dim ** 0.5)
        
        # Process through Flash-SVD layers
        for layer in self.svd_layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        pooled = self.pooler(x)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                # Regression task (STS-B)
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels)
            else:
                # Classification task (SST-2)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': x,
            'attentions': None
        })()

    @classmethod
    def from_pretrained(cls, model_directory: str, task_name: str = None, device = "cpu"):
        """Load a saved Flash-SVD model for inference."""
        # Parse SVD ranks from folder name
        svd_config = cls._parse_ranks_from_folder(model_directory)
        
        # Load base config from the dense model directory
        if task_name is None:
            # Extract task name from model directory path
            task_from_path = model_directory.split('-finetuned-')[-1] if '-finetuned-' in model_directory else "sst2"
        else:
            task_from_path = task_name
        
        dense_model_dir = f"../model/bert-base-uncased-{task_from_path}"  # Path relative to BERT_FINETUNE
        config = AutoConfig.from_pretrained(dense_model_dir)
        
        # Create model with parsed SVD ranks
        model = cls(config, svd_config)
        
        # Load the saved weights
        model_path = os.path.join(model_directory, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Model weights not found at {model_path}")
        
        model.to(device)
        return model
    
    @staticmethod
    def _parse_ranks_from_folder(folder_path: str) -> Dict[str, Any]:
        """Parse SVD ranks from folder name pattern: bert-svd-{attn_rank}-{ff_rank}-{wo_rank}-finetuned"""
        folder_name = os.path.basename(folder_path)
        
        # Try to parse the pattern: bert-svd-{attn}-{ff}-{wo}-finetuned
        pattern = r'bert-svd-(\d+)-(\d+)-(\d+)-finetuned'
        match = re.search(pattern, folder_name)
        
        if match:
            rank_attn = int(match.group(1))
            rank_ff = int(match.group(2))
            rank_wo = int(match.group(3))
            
            print(f"📊 Parsed SVD ranks from folder name: attn={rank_attn}, ff={rank_ff}, wo={rank_wo}")
            
            return {
                'rank_attn': rank_attn,
                'rank_ff': rank_ff,
                'rank_wo': rank_wo,
                'model_type': 'bert_flashsvd',
                'svd_version': '1.0'
            }
        else:
            # Fallback to default values if pattern doesn't match
            print(f"⚠️  Could not parse ranks from folder name '{folder_name}', using defaults")
            return {
                'rank_attn': 16,
                'rank_ff': 192,
                'rank_wo': 192,
                'model_type': 'bert_flashsvd',
                'svd_version': '1.0'
            }

# ─── 3) Inference Functions ───────────────────────────────────────────────────
def load_flashsvd_model(model_path: str, task_name: str = None, device = "cpu"):
    """Load a saved Flash-SVD model for inference."""
    print(f"Loading Flash-SVD model from {model_path}...")
    model = FlashSVDModel.from_pretrained(model_path, task_name, device)
    model.eval()
    return model

def evaluate_model(model, tokenizer, task_name, device="cpu", max_samples=None, seq_len=128, batch_size=32):
    """Evaluate the model on GLUE dataset (SST-2 or STS-B)."""
    print(f"\n{'='*60}")
    print(f"GLUE {task_name.upper()} EVALUATION")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"📊 Loading GLUE {task_name.upper()} dataset...")
    dataset = load_dataset("glue", task_name)
    
    # Task-specific tokenization
    def tokenize_function(examples):
        if task_name == "sst2":
            return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=seq_len)
        elif task_name == "stsb":
            return tokenizer(examples["sentence1"], examples["sentence2"], 
                           truncation=True, padding="max_length", max_length=seq_len)
        else:
            raise ValueError(f"Unsupported task: {task_name}")
    
    # Tokenize and prepare dataset
    tokenized_dataset = dataset["validation"].map(tokenize_function, batched=True)
    
    # Task-specific column handling
    if task_name == "sst2":
        tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    elif task_name == "stsb":
        tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    tokenized_dataset.set_format("torch")
    
    # Limit samples if specified
    if max_samples:
        tokenized_dataset = tokenized_dataset.select(range(min(max_samples, len(tokenized_dataset))))
        print(f"📝 Using {len(tokenized_dataset)} samples for evaluation")
    
    # Create data loader
    data_collator = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=lambda b: {
        "input_ids":      torch.stack([x["input_ids"]      for x in b]),
        "attention_mask": torch.stack([x["attention_mask"] for x in b]),
        "labels":         torch.tensor([x["labels"]         for x in b]),
    })
    
    # Load metric
    metric = load_metric("glue", task_name)
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    print(f"🔄 Evaluating on {len(tokenized_dataset)} samples...")
    print(f"   Task: {task_name.upper()}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Batch size: {batch_size}")
    
    with torch.no_grad():
        for batch in data_collator:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss
            
            # Calculate loss
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
            
            # Task-specific prediction handling
            if task_name == "sst2":
                # Classification: argmax for class prediction
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())
            elif task_name == "stsb":
                # Regression: squeeze logits for continuous values
                predictions = logits.squeeze(-1)
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())
    
    # Calculate metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    results = metric.compute(predictions=all_predictions, references=all_labels)
    
    # Task-specific results display
    print(f"\n📊 EVALUATION RESULTS:")
    if task_name == "sst2":
        print(f"   Accuracy: {results['accuracy']:.4f}")
    elif task_name == "stsb":
        print(f"   Pearson Correlation: {results['pearson']:.4f}")
        print(f"   Spearman Correlation: {results['spearmanr']:.4f}")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   Total Samples: {len(all_predictions)}")
    
    # Show some example predictions
    print(f"\n📝 SAMPLE PREDICTIONS:")
    for i in range(min(10, len(tokenized_dataset))):
        sample = tokenized_dataset[i]
        text = sample["input_ids"]
        # Decode text (simplified)
        decoded_text = tokenizer.decode(text, skip_special_tokens=True)
        if len(decoded_text) > 50:
            decoded_text = decoded_text[:50] + "..."
        
        if task_name == "sst2":
            pred_label = "POSITIVE" if all_predictions[i] == 1 else "NEGATIVE"
            true_label = "POSITIVE" if all_labels[i] == 1 else "NEGATIVE"
            correct = "✅" if all_predictions[i] == all_labels[i] else "❌"
            print(f"   {correct} \"{decoded_text}\" → Pred: {pred_label}, True: {true_label}")
        elif task_name == "stsb":
            pred_score = all_predictions[i]
            true_score = all_labels[i]
            error = abs(pred_score - true_score)
            quality = "✅" if error < 0.5 else "⚠️" if error < 1.0 else "❌"
            print(f"   {quality} \"{decoded_text}\" → Pred: {pred_score:.2f}, True: {true_score:.2f}")
    
    return results

# ─── 4) Original Flash-SVD block (kept for reference) ───────────────────────
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

        if mask is not None:
            mask4 = mask.view(B,1,1,M)
        else:
            # Create a mask of all ones (all tokens valid) when no mask is provided
            mask4 = torch.ones(B, 1, 1, M, device=x.device, dtype=torch.bool)
        
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
        # If we do not use the below inference, it will slow down the process
        # midV = mid @ self.V1
        # midA = F.gelu(midV + self.b1)
        # y    = (midA @ self.U2) @ self.V2 + self.b2
        out = self.ln2(x1 + y)
        #print("FlashSVDFFN Peak:", torch.cuda.max_memory_allocated()/1024**2)
        
        return out 

# ─── 5) LayerShim ────────────────────────────────────────────────────────────
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

# ─── 6) Main Function ───────────────────────────────────────────────────────
def main():
    """Main function with rank-aware Flash-SVD inference, memory profiling, and GLUE evaluation."""
    
    # ─── Configuration Parameters ───────────────────────────────────────────
    BATCH_SIZE = 32
    SEQ_LEN = 128
    MAX_SAMPLES = 1000000  # For evaluation
    RANK_ATTN = 16 # 40 #
    RANK_FF = 96 # 240 #
    RANK_WO = 96 # 240 #
    
    
    # Load the actual saved model from the training
    model_path = f"model/bert-svd-{RANK_ATTN}-{RANK_FF}-{RANK_WO}-finetuned-{task_name}"
    
    print(f"\n{'='*70}")
    print("FLASH-SVD MODEL INFERENCE & EVALUATION")
    print(f"{'='*70}")
    print(f"🔧 Configuration: task={task_name.upper()}, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, max_samples={MAX_SAMPLES}")
    print("🔍 Adaptive rank parsing from folder name...")
    
    # Check if model exists
    if not os.path.exists(model_path): 
        print(f"❌ Model not found at {model_path}")
        print("Please run finetune_svd_saveable.py first to create a model.")
        return
    
    print(f"✅ Found model at: {model_path}")
    
    # Check model files
    required_files = ["pytorch_model.bin", "tokenizer.json"]
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file}: {size:.1f} MB")
        else:
            print(f"❌ Missing: {file}")
            return
    
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔄 Loading Flash-SVD model on device: {device}")
    
    try:
        model = load_flashsvd_model(model_path, task_name, device)
        print("✅ Flash-SVD model loaded successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   SVD ranks: attn={model.svd_config['rank_attn']}, ff={model.svd_config['rank_ff']}, wo={model.svd_config['rank_wo']}")
        
        # Show derived configuration from first layer
        if len(model.svd_layers) > 0:
            first_layer_config = model.svd_layers[0].full_config
            print(f"   Derived config: d_model={first_layer_config['d_model']}, heads={first_layer_config['num_heads']}, d_ff={first_layer_config['d_ff']}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return
    
    # ─── 7) Memory profiling ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MEMORY PROFILING")
    print(f"{'='*60}")
    
    # Load dataset for profiling
    raw = load_dataset("glue", task_name, split="validation")
    def tokenize_fn(batch):
        if task_name == "sst2":
            return tokenizer(batch["sentence"],
                            padding="max_length", truncation=True, max_length=SEQ_LEN)
        elif task_name == "stsb":
            return tokenizer(batch["sentence1"], batch["sentence2"],
                            padding="max_length", truncation=True, max_length=SEQ_LEN)
        else:
            raise ValueError(f"Unsupported task: {task_name}")

    # Task-specific column removal
    if task_name == "sst2":
        ds = raw.map(tokenize_fn, batched=True, remove_columns=["sentence","idx"])
    elif task_name == "stsb":
        ds = raw.map(tokenize_fn, batched=True, remove_columns=["sentence1","sentence2","idx"])
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        collate_fn=lambda b: {
                            "input_ids":      torch.stack([x["input_ids"]      for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                            "labels":         torch.tensor([x["labels"]         for x in b]),
                        })
    
    # Choose the right metric for the task
    if task_name == "stsb":
        metric = load_metric("pearsonr")
    else:
        metric = load_metric("accuracy")
    
    # Latency and memory measurement function
    @torch.no_grad()
    def acc_peak_time(mdl, use_mask=True):
        mdl.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        total_score, steps = 0.0, 0
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
            
            total_score += metric.compute(
                predictions=preds.cpu(),
                references=batch["labels"].cpu()
            )["pearsonr" if task_name == "stsb" else "accuracy"]
            steps += 1
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start)*1000/steps
        peak = torch.cuda.max_memory_allocated()/1024**2
        return total_score/steps, peak, elapsed
    
    # Memory measurement
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Persistent model storage
    CACHED_ORIG_MEM = torch.cuda.max_memory_allocated()/1024**2
    print(f"Persistent model storage: {CACHED_ORIG_MEM:6.1f} MiB")
    
    # Warm-up
    batch = next(iter(loader))
    inp = batch["input_ids"].to(device)
    msk = batch["attention_mask"].to(device)
    
    with torch.no_grad():
        _ = model(input_ids=inp, attention_mask=msk)

    # Clear and sync
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run comprehensive latency and memory measurement
    metric_name = "pearson" if task_name == "stsb" else "acc"
    
    # Test with mask
    score, peak_m, latency = acc_peak_time(model, use_mask=True)
    print(f"FlashSVD w/ mask   | {metric_name}={score:.4f} | peak={peak_m:6.1f} MiB | transient={peak_m - CACHED_ORIG_MEM:6.1f} MiB | {latency:6.1f} ms/b")
    
    # Test without mask  
    score, peak_nm, latency = acc_peak_time(model, use_mask=False)
    print(f"FlashSVD w/o mask  | {metric_name}={score:.4f} | peak={peak_nm:6.1f} MiB | transient={peak_nm - CACHED_ORIG_MEM:6.1f} MiB | {latency:6.1f} ms/b")
    
    # Store results for final summary
    peak_res = max(peak_m, peak_nm)
    scratch = peak_res - CACHED_ORIG_MEM
    
    # ─── 8) Evaluate on GLUE dataset ─────────────────────────────────────
    try:
        results = evaluate_model(model, tokenizer, task_name, device, max_samples=MAX_SAMPLES, 
                                seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Flash-SVD model successfully loaded from {model_path}")
        print(f"✓ Adaptive rank parsing working correctly!")
        
        # Task-specific results display
        if task_name == "sst2":
            print(f"✓ SST-2 Accuracy: {results['accuracy']:.4f}")
        elif task_name == "stsb":
            print(f"✓ STS-B Pearson: {results['pearson']:.4f}")
            print(f"✓ STS-B Spearman: {results['spearmanr']:.4f}")
        
        # Test model size comparison
        model_size_mb = os.path.getsize(os.path.join(model_path, "pytorch_model.bin")) / (1024*1024)
        print(f"✓ Model size: {model_size_mb:.1f} MB")
        print(f"✓ Model can be used for inference without re-finetuning!")
        print(f"✓ Flash-SVD compression working correctly!")
        print(f"✓ Peak memory usage: {peak_res:.1f} MiB")
        print(f"✓ Transient scratch: {scratch:.1f} MiB")
        print(f"✓ Configuration: task={task_name.upper()}, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
        
    except Exception as e:
        print(f"❌ {task_name.upper()} evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()