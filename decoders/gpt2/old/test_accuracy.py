import os
import sys
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import math

# Import our SVD implementation
sys.path.append(os.path.dirname(__file__))
from profile_svd_kv import LinearGPT2Block, LinearSVDBlock, LayerShim

def test_model_accuracy():
    device = "cuda"
    batch_size = 4
    seq_len = 64
    
    # Load dataset
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokz = AutoTokenizer.from_pretrained("gpt2")
    tokz.pad_token = tokz.eos_token
    
    def tokenize_fn(batch):
        return tokz(batch["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=seq_len)

    ds = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: {
                            "input_ids": torch.stack([x["input_ids"] for x in b]),
                            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
                        })
    
    # Test original model
    print("Testing original GPT-2 model...")
    original_model = GPT2LMHeadModel.from_pretrained("gpt2")
    original_model = original_model.to(device).eval()
    
    @torch.no_grad()
    def compute_perplexity(model):
        total_loss, total_tokens = 0.0, 0
        
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["input_ids"][..., 1:].contiguous()
            mask = batch["attention_mask"][..., 1:].contiguous()
            
            valid_positions = mask.bool()
            
            if valid_positions.sum() > 0:
                valid_logits = shift_logits[valid_positions]
                valid_labels = shift_labels[valid_positions]
                
                loss = F.cross_entropy(valid_logits, valid_labels)
                total_loss += loss.item() * valid_positions.sum().item()
                total_tokens += valid_positions.sum().item()
        
        if total_tokens == 0:
            return float('nan')
        
        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)
    
    original_perplexity = compute_perplexity(original_model)
    print(f"Original model perplexity: {original_perplexity:.4f}")
    
    # Test SVD model
    print("Testing SVD model...")
    svd_model = GPT2LMHeadModel.from_pretrained("gpt2")
    svd_model = svd_model.to(device).eval()
    
    # Convert to SVD (full rank)
    for i, layer in enumerate(svd_model.transformer.h):
        linear_block = LinearGPT2Block(layer)
        svd_block = LinearSVDBlock(linear_block, 64, 768, 768)  # Full rank
        svd_model.transformer.h[i] = LayerShim(svd_block).to(device).eval()
    
    svd_perplexity = compute_perplexity(svd_model)
    print(f"SVD model perplexity: {svd_perplexity:.4f}")
    
    # Compare
    print(f"Perplexity difference: {abs(svd_perplexity - original_perplexity):.4f}")
    print(f"Relative difference: {abs(svd_perplexity - original_perplexity) / original_perplexity * 100:.2f}%")
    
    if abs(svd_perplexity - original_perplexity) > 1.0:
        print("WARNING: Significant accuracy loss detected!")
    else:
        print("Accuracy is acceptable.")

if __name__ == "__main__":
    test_model_accuracy() 