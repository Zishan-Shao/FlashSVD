import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, AutoConfig
from evaluate import load as load_metric
from typing import Dict, Any, Optional, Tuple
import math
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# ─── 0) Setup ─────────────────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from src.fwsvd import (
    compute_row_sum_svd_decomposition,
    estimate_fisher_weights_bert,
    estimate_fisher_weights_bert_with_attention,
)
# Default task - can be overridden by command line argument
task_name = "stsb"
MODEL_DIR = os.path.join(REPO_ROOT, "model", f"bert-base-uncased-{task_name}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# python finetune_svd_saveable.py --task stsb --rank_attn 32 --rank_ff 192 --rank_wo 192 --epochs 5

# python finetune_svd_saveable.py --task sst2 --rank_attn 32 --rank_ff 192 --rank_wo 192 --epochs 5



# ─── 1) SVD Block Implementation ─────────────────────────────────────────────
class SVDBlock(nn.Module):
    """Low-rank SVD block that can be saved and loaded independently."""
    
    def __init__(self, hf_layer, rank_attn: int, rank_ff: int, 
                 svd_per_head, svd_low_rank, rank_wo: int = 768):
        super().__init__()
        cfg = hf_layer.attention.self
        d_model = cfg.all_head_size
        H = cfg.num_attention_heads
        dh = d_model // H
        d_ff = hf_layer.intermediate.dense.out_features
        
        # Store configuration for saving/loading
        self.config = {
            'rank_attn': rank_attn,
            'rank_ff': rank_ff,
            'rank_wo': rank_wo,
            'd_model': d_model,
            'num_heads': H,
            'head_dim': dh,
            'd_ff': d_ff
        }
        
        # 1) Factor Q/K/V projections
        WqT = hf_layer.attention.self.query.weight.data.t()
        WkT = hf_layer.attention.self.key.weight.data.t()
        WvT = hf_layer.attention.self.value.weight.data.t()
        bq = hf_layer.attention.self.query.bias.data.view(1, H, 1, dh)
        bk = hf_layer.attention.self.key.bias.data.view(1, H, 1, dh)
        bv = hf_layer.attention.self.value.bias.data.view(1, H, 1, dh)

        Uq, Vq = svd_per_head(WqT, rank_attn)
        Uk, Vk = svd_per_head(WkT, rank_attn)
        Uv, Vv = svd_per_head(WvT, rank_attn)

        # 2) Factor FFN projections
        Wi = hf_layer.intermediate.dense.weight.data.t()
        bi = hf_layer.intermediate.dense.bias.data
        WoT = hf_layer.output.dense.weight.data.t()
        bo2 = hf_layer.output.dense.bias.data

        U1, V1 = svd_low_rank(Wi, rank_ff)
        U2, V2 = svd_low_rank(WoT, rank_ff)
        
        # 3) Factor attention output projection
        Wo_full = hf_layer.attention.output.dense.weight.data
        bo_attn = hf_layer.attention.output.dense.bias.data
        Uo, Vo = svd_low_rank(Wo_full.t(), rank_wo)
        
        # Register all parameters
        self.Pq, self.Vq, self.bq = map(nn.Parameter, (Uq.unsqueeze(0), Vq.unsqueeze(0), bq))
        self.Pk, self.Vk, self.bk = map(nn.Parameter, (Uk.unsqueeze(0), Vk.unsqueeze(0), bk))
        self.Pv, self.Vv, self.bv = map(nn.Parameter, (Uv.unsqueeze(0), Vv.unsqueeze(0), bv))
        self.Uo, self.Vo, self.bo_attn = nn.Parameter(Uo), nn.Parameter(Vo), nn.Parameter(bo_attn)
        self.U1, self.V1, self.b1 = nn.Parameter(U1), nn.Parameter(V1), nn.Parameter(bi)
        self.U2, self.V2, self.b2 = nn.Parameter(U2), nn.Parameter(V2), nn.Parameter(bo2)

        # Copy layer norms
        self.ln1 = hf_layer.attention.output.LayerNorm
        self.ln2 = hf_layer.output.LayerNorm

    def forward(self, x, mask=None):
        B, M, dm = x.shape
        _, H, _, R = self.Pq.shape
        dh = dm // H

        # Project into low-rank Q/K/V
        def project(x, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

        Q = project(x, self.Pq[0], self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk[0], self.Vk[0], self.bk).contiguous()
        V = project(x, self.Pv[0], self.Vv[0], self.bv).contiguous()
        
        # Standard attention (can be replaced with FlashAttention)
        logits = torch.einsum("bhmd,bhnd->bhmn", Q, K) * (1.0 / math.sqrt(dh))
        
        if mask is not None:
            m = mask.view(B, 1, 1, M).to(torch.bool)
            logits = logits.masked_fill(~m, float("-1e9"))

        A = torch.softmax(logits, dim=-1)
        attn = torch.einsum("bhmn,bhnd->bhmd", A, V)
        
        # Back to [B,M,dm]
        attn = attn.transpose(1, 2).reshape(B, M, dm)
        
        # Output projection + LayerNorm
        x1 = self.ln1(x + (attn @ self.Uo) @ self.Vo + self.bo_attn)
        
        # FFN
        mid = x1 @ self.U1
        midV = mid @ self.V1
        midA = torch.nn.functional.gelu(midV + self.b1)
        y = (midA @ self.U2) @ self.V2 + self.b2
        out = self.ln2(x1 + y)
        
        return out

    def get_config(self) -> Dict[str, Any]:
        """Get the configuration for this SVD block."""
        return self.config.copy()

# ─── 2) Saveable SVD Model ───────────────────────────────────────────────────
class SaveableSVDModel(nn.Module):
    """A BERT model with SVD decomposition that can be saved and loaded."""
    
    def __init__(self, base_model: BertForSequenceClassification, 
                 rank_attn: int, rank_ff: int, rank_wo: int = 768):
        super().__init__()
        
        # Store original model config and add SVD info
        self.config = base_model.config
        self.svd_config = {
            'rank_attn': rank_attn,
            'rank_ff': rank_ff,
            'rank_wo': rank_wo,
            'model_type': 'bert_svd',
            'svd_version': '1.0'
        }
        
        # Copy non-encoder parts
        self.embeddings = base_model.bert.embeddings
        self.pooler = base_model.bert.pooler
        self.classifier = base_model.classifier
        
        # Build SVD helpers
        svd_per_head, svd_low_rank = self._build_svd_helpers(base_model)
        
        # Replace encoder layers with SVD blocks
        self.svd_layers = nn.ModuleList([
            SVDBlock(layer, rank_attn, rank_ff, svd_per_head, svd_low_rank, rank_wo)
            for layer in base_model.bert.encoder.layer
        ])
        
        # Final layer norm
        self.norm = base_model.bert.encoder.layer[-1].output.LayerNorm

    def _build_svd_helpers(self, model):
        """Build SVD decomposition helpers."""
        def svd_per_head(Wt: torch.Tensor, rank: int):
            d_model, _ = Wt.shape
            H = model.config.num_attention_heads
            dh = d_model // H
            Wt3 = Wt.view(d_model, H, dh)
            Us, Vs = [], []
            for h in range(H):
                Wh = Wt3[:, h, :].float()
                U32, S32, Vh32 = torch.linalg.svd(Wh, full_matrices=False)
                Us.append((U32[:, :rank] * S32[:rank]).to(Wt.dtype))
                Vs.append(Vh32[:rank, :].to(Wt.dtype))
            return torch.stack(Us, 0), torch.stack(Vs, 0)

        def svd_low_rank(W: torch.Tensor, rank: int):
            Wf = W.float()
            U32, S32, Vh32 = torch.linalg.svd(Wf, full_matrices=False)
            U = (U32[:, :rank] * S32[:rank]).to(W.dtype)
            V = Vh32[:rank, :].to(W.dtype)
            return U, V

        return svd_per_head, svd_low_rank

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Ignore unused parameters like token_type_ids
        x = self.embeddings(input_ids)
        x = x * (self.embeddings.word_embeddings.embedding_dim ** 0.5)
        
        # Process through SVD layers
        for layer in self.svd_layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        pooled = self.pooler(x)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            if self.config.problem_type == "regression":
                # For regression tasks like STS-B
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                # For classification tasks
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': x,
            'attentions': None
        })()

    def save_pretrained(self, save_directory: str):
        """Save the model with all necessary configurations."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the base config
        self.config.save_pretrained(save_directory)
        
        # Save SVD-specific config
        svd_config_path = os.path.join(save_directory, "svd_config.json")
        with open(svd_config_path, 'w') as f:
            json.dump(self.svd_config, f, indent=2)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        print(f"✓ Saved SVD model to {save_directory}")
        print(f"  - Base config: {os.path.join(save_directory, 'config.json')}")
        print(f"  - SVD config: {svd_config_path}")
        print(f"  - Model weights: {model_path}")

    @classmethod
    def from_pretrained(cls, model_directory: str, device = "cpu"):
        """Load a saved SVD model."""
        # Load base config
        config = AutoConfig.from_pretrained(model_directory)
        
        # Load SVD config
        svd_config_path = os.path.join(model_directory, "svd_config.json")
        if not os.path.exists(svd_config_path):
            raise ValueError(f"SVD config not found at {svd_config_path}")
        
        with open(svd_config_path, 'r') as f:
            svd_config = json.load(f)
        
        # Create a dummy base model to extract structure (but don't load weights)
        base_model = BertForSequenceClassification(config)
        
        # Create SVD model with same structure
        model = cls(
            base_model, 
            rank_attn=svd_config['rank_attn'],
            rank_ff=svd_config['rank_ff'],
            rank_wo=svd_config.get('rank_wo', 768)
        )
        
        # Load the saved weights
        model_path = os.path.join(model_directory, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Model weights not found at {model_path}")
        
        model.to(device)
        return model

# ─── 3) Training and Saving Functions ────────────────────────────────────────
def create_svd_model(base_model_path: str, rank_attn: int, rank_ff: int, 
                    rank_wo: int = 768, device = "cuda", task_name = "stsb") -> SaveableSVDModel:
    """Create a new SVD model from a base model."""
    # Configure model based on task
    if task_name == "stsb":
        # STS-B is a regression task with 1 output
        base_model = BertForSequenceClassification.from_pretrained(
            base_model_path, 
            num_labels=1,
            problem_type="regression"
        )
    else:
        # Get number of labels from the dataset
        ds = load_dataset("glue", task_name)
        num_labels = ds["train"].features["label"].num_classes
        base_model = BertForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=num_labels
        )
    
    svd_model = SaveableSVDModel(base_model, rank_attn, rank_ff, rank_wo)
    return svd_model.to(device)

def train_svd_model(model: SaveableSVDModel, train_loader: DataLoader, 
                   val_loader: DataLoader, num_epochs: int = 5, 
                   learning_rate: float = 3e-5, device = "cuda", task_name = "stsb"):
    """Train the SVD model and return training history."""
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Total train samples: {len(train_loader.dataset)}")
    print(f"Total val samples: {len(val_loader.dataset)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler with patience
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',           # Monitor validation accuracy (higher is better)
        factor=0.5,           # Reduce LR by half when plateauing
        patience=2,           # Wait 2 epochs before reducing LR
        min_lr=1e-7          # Minimum learning rate
    )
    
    # Use appropriate metric for the task
    if task_name == "stsb":
        metric = load_metric("pearsonr")  # Pearson correlation for STS-B
    elif task_name == "cola":
        metric = load_metric("matthews_correlation")  # Matthews correlation for COLA
    else:
        metric = load_metric("accuracy")  # Accuracy for other classification tasks
    
    history = {
        'train_loss': [], 
        'val_accuracy': [], 
        'val_loss': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    # Early stopping variables
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 4  # Stop if no improvement for 4 epochs
    best_model_state = None
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"Learning rate scheduler: ReduceLROnPlateau (patience=2, factor=0.5)")
    print(f"Early stopping patience: {early_stopping_patience} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        total_loss = 0.0
        batch_losses = []
        
        print(f"\n📚 EPOCH {epoch+1}/{num_epochs}")
        print(f"{'─'*40}")
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", 
                         leave=False, ncols=100)
        
        for batch_idx, batch in enumerate(train_pbar):
            batch_start_time = time.time()
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labels = batch["labels"].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Update progress bar
            avg_loss_so_far = total_loss / (batch_idx + 1)
            current_lr = optimizer.param_groups[0]["lr"]
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss_so_far:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: "
                      f"loss={loss.item():.4f}, avg_loss={avg_loss_so_far:.4f}, lr={current_lr:.2e}")

        train_time = time.time() - epoch_start_time
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        print(f"\n✅ Training completed:")
        print(f"   Average loss: {avg_train_loss:.4f}")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Loss std dev: {torch.tensor(batch_losses).std().item():.4f}")

        # Validation
        print(f"\n🔍 VALIDATION")
        print(f"{'─'*40}")
        
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                labels = batch["labels"].to(device)

                outputs = model(**inputs, labels=labels)
                logits = outputs.logits
                
                # Handle different task types
                if model.config.problem_type == "regression":
                    # For regression tasks like STS-B
                    preds = logits.view(-1)
                    loss_fct = nn.MSELoss()
                    batch_val_loss = loss_fct(logits.view(-1), labels.view(-1).float())
                else:
                    # For classification tasks
                    preds = torch.argmax(logits, dim=-1)
                    loss_fct = nn.CrossEntropyLoss()
                    batch_val_loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
                
                val_loss += batch_val_loss.item()
                val_batches += 1

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Update progress bar
                if model.config.problem_type == "regression":
                    val_pbar.set_postfix({
                        'val_loss': f'{batch_val_loss.item():.4f}',
                        'pred_range': f'[{min(preds.cpu().tolist()):.2f}, {max(preds.cpu().tolist()):.2f}]'
                    })
                else:
                    val_pbar.set_postfix({
                        'val_loss': f'{batch_val_loss.item():.4f}',
                        'correct': f'{sum(p == l for p, l in zip(preds.cpu().tolist(), labels.cpu().tolist()))}/{len(labels)}'
                    })

        val_time = time.time() - epoch_start_time - train_time
        avg_val_loss = val_loss / val_batches
        # Compute appropriate metric
        if task_name == "stsb":
            # For STS-B, compute Pearson correlation
            metric_result = metric.compute(predictions=all_preds, references=all_labels)
            acc = metric_result["pearsonr"]
            print(f"   Pearson correlation: {acc:.4f}")
        elif task_name == "cola":
            # For COLA, compute Matthews correlation
            metric_result = metric.compute(predictions=all_preds, references=all_labels)
            acc = metric_result["matthews_correlation"]
            print(f"   Matthews correlation: {acc:.4f}")
        else:
            # For classification tasks
            acc = metric.compute(predictions=all_preds, references=all_labels)["accuracy"]
            print(f"   Accuracy: {acc:.4f}")
        
        history['val_accuracy'].append(acc)
        history['val_loss'].append(avg_val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]["lr"])
        history['epoch_times'].append(time.time() - epoch_start_time)
        
        print(f"\n✅ Validation completed:")
        print(f"   Accuracy: {acc:.4f} ({sum(p == l for p, l in zip(all_preds, all_labels))}/{len(all_labels)} correct)")
        print(f"   Validation loss: {avg_val_loss:.4f}")
        print(f"   Validation time: {val_time:.2f}s")
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(acc)  # Step based on validation accuracy
        new_lr = optimizer.param_groups[0]["lr"]
        
        if new_lr != old_lr:
            print(f"🔄 Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Early stopping check
        if acc > best_val_acc:
            best_val_acc = acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"🏆 New best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs (best: {best_val_acc:.4f})")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\n📊 EPOCH {epoch+1} SUMMARY")
        print(f"{'─'*40}")
        print(f"   Train loss: {avg_train_loss:.4f}")
        print(f"   Val loss: {avg_val_loss:.4f}")
        print(f"   Val accuracy: {acc:.4f}")
        print(f"   Learning rate: {new_lr:.2e}")
        print(f"   Total epoch time: {epoch_time:.2f}s")
        
        # Show improvement
        if epoch > 0:
            loss_improvement = history['train_loss'][-2] - avg_train_loss
            acc_improvement = acc - history['val_accuracy'][-2]
            print(f"   Loss improvement: {loss_improvement:+.4f}")
            print(f"   Accuracy improvement: {acc_improvement:+.4f}")
        
        print(f"{'─'*40}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after {patience_counter} epochs without improvement")
            print(f"Restoring best model (accuracy: {best_val_acc:.4f})")
            model.load_state_dict(best_model_state)
            break
    
    # Training summary
    total_training_time = sum(history['epoch_times'])
    best_acc = max(history['val_accuracy'])
    best_acc_epoch = history['val_accuracy'].index(best_acc) + 1
    final_loss = history['train_loss'][-1]
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.1f} minutes)")
    if task_name == "stsb":
        print(f"Best validation Pearson correlation: {best_acc:.4f} (epoch {best_acc_epoch})")
        print(f"Final training loss: {final_loss:.4f}")
        print(f"Final validation Pearson correlation: {history['val_accuracy'][-1]:.4f}")
    elif task_name == "cola":
        print(f"Best validation Matthews correlation: {best_acc:.4f} (epoch {best_acc_epoch})")
        print(f"Final training loss: {final_loss:.4f}")
        print(f"Final validation Matthews correlation: {history['val_accuracy'][-1]:.4f}")
    else:
        print(f"Best validation accuracy: {best_acc:.4f} (epoch {best_acc_epoch})")
        print(f"Final training loss: {final_loss:.4f}")
        print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"Average epoch time: {total_training_time/len(history['epoch_times']):.2f}s")
    
    # Learning rate summary
    lr_changes = [i for i in range(1, len(history['learning_rates'])) 
                  if history['learning_rates'][i] != history['learning_rates'][i-1]]
    if lr_changes:
        print(f"Learning rate was reduced {len(lr_changes)} times (epochs: {[e+1 for e in lr_changes]})")
    
    # Plot training curves (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy/Correlation plot
        if task_name == "stsb":
            ax2.plot(history['val_accuracy'], label='Val Pearson Correlation', marker='s', color='green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Pearson Correlation')
            ax2.set_title('Validation Pearson Correlation')
        elif task_name == "cola":
            ax2.plot(history['val_accuracy'], label='Val Matthews Correlation', marker='s', color='green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Matthews Correlation')
            ax2.set_title('Validation Matthews Correlation')
        else:
            ax2.plot(history['val_accuracy'], label='Val Accuracy', marker='s', color='green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(history['learning_rates'], label='Learning Rate', marker='o', color='red')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print(f"Training curves saved to 'training_curves.png'")
        
    except ImportError:
        print("Matplotlib not available - skipping training curve plot")
    
    return history

# ─── 4) Main Training Script ─────────────────────────────────────────────────
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SVD model on GLUE tasks')
    parser.add_argument('--task', type=str, default='stsb', 
                       choices=['stsb', 'qnli', 'mnli', 'rte', 'mrpc', 'qqp', 'cola', 'sst2'],
                       help='GLUE task to train on')
    parser.add_argument('--rank_attn', type=int, default=16, help='Attention rank')
    parser.add_argument('--rank_ff', type=int, default=96, help='Feed-forward rank')
    parser.add_argument('--rank_wo', type=int, default=96, help='Output projection rank')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Update global task_name
    global task_name, MODEL_DIR
    task_name = args.task
    MODEL_DIR = os.path.join(REPO_ROOT, "model", f"bert-base-uncased-{task_name}")
    
    # Configuration
    RANK_ATTN = args.rank_attn
    RANK_FF = args.rank_ff
    RANK_WO = args.rank_wo
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    
    print(f"Training SVD model on {task_name.upper()} with ranks: attn={RANK_ATTN}, ff={RANK_FF}, wo={RANK_WO}")
    
    # 1. Create SVD model
    print("Creating SVD model...")
    svd_model = create_svd_model(MODEL_DIR, RANK_ATTN, RANK_FF, RANK_WO, DEVICE, task_name)
    
    # 2. Prepare data
    print("Preparing data...")
    ds = load_dataset("glue", task_name)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # Handle different GLUE tasks
    if task_name == "stsb":
        # STS-B is a regression task with sentence1, sentence2 pairs
        def tokenize_stsb(examples):
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                padding="max_length",
                max_length=256
            )
        tokenized = ds.map(
            tokenize_stsb,
            batched=True,
            remove_columns=["sentence1", "sentence2", "idx"]
        )
    elif task_name in ["qnli", "mnli", "rte", "mrpc", "qqp"]:
        # These are pair sentence classification tasks
        field_map = {
            "qnli": ("question", "sentence"),
            "mnli": ("premise", "hypothesis"),
            "rte": ("sentence1", "sentence2"),
            "mrpc": ("sentence1", "sentence2"),
            "qqp": ("question1", "question2")
        }
        first_field, second_field = field_map[task_name]
        
        def tokenize_pair(examples):
            return tokenizer(
                examples[first_field],
                examples[second_field],
                truncation=True,
                padding="max_length",
                max_length=256
            )
        tokenized = ds.map(
            tokenize_pair,
            batched=True,
            remove_columns=[first_field, second_field, "idx"]
        )
    else:
        # Single sentence tasks like SST-2, COLA
        tokenized = ds.map(
            lambda ex: tokenizer(ex["sentence"], truncation=True),
            batched=True,
            remove_columns=["sentence"]
        )
    
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    
    # Handle different validation split names
    if task_name == "mnli":
        val_split = "validation_matched"
    else:
        val_split = "validation"
    
    val_loader = DataLoader(tokenized[val_split], batch_size=64, shuffle=False, collate_fn=collator)
    
    # 3. Train the model
    print("Starting training...")
    history = train_svd_model(svd_model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE, task_name)
    
    # 4. Save the trained model
    save_dir = f"model/bert-svd-{RANK_ATTN}-{RANK_FF}-{RANK_WO}-finetuned-{task_name}"
    svd_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # 5. Test loading
    print("\nTesting model loading...")
    loaded_model = SaveableSVDModel.from_pretrained(save_dir, DEVICE)
    loaded_model.eval()
    
    # 6. Test inference
    if task_name == "stsb":
        # For STS-B, test with sentence pairs
        examples = [
            ("A wonderfully executed performance!", "This was utterly dull and uninspired."),
            ("The weather is nice today.", "It's a beautiful sunny day.")
        ]
        enc = tokenizer(
            [ex[0] for ex in examples], 
            [ex[1] for ex in examples], 
            padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = loaded_model(**enc).logits
            preds = logits.view(-1).cpu().tolist()

        print("\nInference test (STS-B):")
        for (sent1, sent2), p in zip(examples, preds):
            print(f"\"{sent1}\" vs \"{sent2}\" --> Similarity: {p:.3f}")
    elif task_name in ["qnli", "mnli", "rte", "mrpc", "qqp"]:
        # For pair sentence classification tasks
        if task_name == "qnli":
            examples = [
                ("What is the capital of France?", "Paris is the capital of France."),
                ("What is 2+2?", "The sky is blue.")
            ]
        elif task_name == "mnli":
            examples = [
                ("The cat is on the mat.", "There is a cat on the mat."),
                ("The cat is on the mat.", "The dog is running.")
            ]
        elif task_name == "rte":
            examples = [
                ("The cat is on the mat.", "There is a cat on the mat."),
                ("The cat is on the mat.", "The dog is running.")
            ]
        elif task_name == "mrpc":
            examples = [
                ("The cat is on the mat.", "There is a cat on the mat."),
                ("The cat is on the mat.", "The dog is running.")
            ]
        elif task_name == "qqp":
            examples = [
                ("What is the capital of France?", "What is France's capital?"),
                ("What is 2+2?", "What is the sky color?")
            ]
        
        enc = tokenizer(
            [ex[0] for ex in examples], 
            [ex[1] for ex in examples], 
            padding=True, truncation=True, return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = loaded_model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        print(f"\nInference test ({task_name.upper()}):")
        for (sent1, sent2), p in zip(examples, preds):
            print(f"\"{sent1}\" vs \"{sent2}\" --> {loaded_model.config.id2label[p]}")
    else:
        # For single sentence classification tasks
        examples = [
            "A wonderfully executed performance!",
            "This was utterly dull and uninspired."
        ]
        enc = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            logits = loaded_model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()

        print(f"\nInference test ({task_name.upper()}):")
        for sent, p in zip(examples, preds):
            print(f"\"{sent}\" --> {loaded_model.config.id2label[p]}")
    
    print(f"\n✓ Model successfully saved and loaded from {save_dir}")
    print("You can now use SaveableSVDModel.from_pretrained() to load this model!")

if __name__ == "__main__":
    print("="*80)
    print("SVD BERT Fine-tuning Script")
    print("="*80)
    print("Supported GLUE tasks: stsb, qnli, mnli, rte, mrpc, qqp, cola, sst2")
    print("Example usage:")
    print("  python finetune_svd_saveable.py --task stsb --epochs 3 --batch_size 16")
    print("  python finetune_svd_saveable.py --task qnli --rank_attn 32 --rank_ff 128")
    print("="*80)
    main() 