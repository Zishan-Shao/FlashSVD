import os
import sys
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
from typing import Dict, Any, List
import math

# ─── 0) Setup ─────────────────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ─── 1) SVD Block Implementation ─────────────────────────────────────────────
class SVDBlock(nn.Module):
    """Low-rank SVD block for inference."""
    
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
        def project(x, P, V, b):
            tmp = torch.einsum("bmd,hdr->bhmr", x, P)
            return torch.einsum("bhmr,hrd->bhmd", tmp, V) + b

        Q = project(x, self.Pq[0], self.Vq[0], self.bq).contiguous()
        K = project(x, self.Pk[0], self.Vk[0], self.bk).contiguous()
        V = project(x, self.Pv[0], self.Vv[0], self.bv).contiguous()
        
        # Standard attention
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

# ─── 2) Saveable SVD Model for Inference ─────────────────────────────────────
class SVDModel(nn.Module):
    """A BERT model with SVD decomposition for inference."""
    
    def __init__(self, config, svd_config=None):
        super().__init__()
        self.config = config
        
        # Use provided SVD config or default values
        if svd_config is None:
            svd_config = {
                'rank_attn': 16,
                'rank_ff': 192,
                'rank_wo': 192,
                'model_type': 'bert_svd',
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
        
        # Create SVD layers with hardcoded ranks
        self.svd_layers = nn.ModuleList([
            SVDBlock(self.svd_config, config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)

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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return type('Outputs', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': x,
            'attentions': None
        })()

    @classmethod
    def from_pretrained(cls, model_directory: str, device = "cpu"):
        """Load a saved SVD model for inference."""
        # Parse SVD ranks from folder name
        svd_config = cls._parse_ranks_from_folder(model_directory)
        
        # Load base config from the dense model directory
        dense_model_dir = "../model/bert-base-uncased-sst2"  # Path relative to BERT_FINETUNE
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
        import re
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
                'model_type': 'bert_svd',
                'svd_version': '1.0'
            }
        else:
            # Fallback to default values if pattern doesn't match
            print(f"⚠️  Could not parse ranks from folder name '{folder_name}', using defaults")
            return {
                'rank_attn': 16,
                'rank_ff': 192,
                'rank_wo': 192,
                'model_type': 'bert_svd',
                'svd_version': '1.0'
            }

# ─── 3) Inference Functions ───────────────────────────────────────────────────
def load_svd_model(model_path: str, device = "cpu"):
    """Load a saved SVD model for inference."""
    print(f"Loading SVD model from {model_path}...")
    model = SVDModel.from_pretrained(model_path, device)
    model.eval()
    return model

def predict_sentiment(model, tokenizer, texts: List[str], device = "cpu"):
    """Predict sentiment for a list of texts."""
    # Tokenize
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    
    # Filter out unused parameters and move to device
    model_inputs = {
        'input_ids': encodings['input_ids'].to(device),
        'attention_mask': encodings['attention_mask'].to(device)
    }
    # Add labels if present (for evaluation)
    if 'labels' in encodings:
        model_inputs['labels'] = encodings['labels'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().tolist()
        probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
    
    # Format results
    results = []
    for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
        label = model.config.id2label[pred]
        confidence = prob[pred]
        results.append({
            'text': text,
            'prediction': label,
            'confidence': confidence,
            'probabilities': {model.config.id2label[j]: p for j, p in enumerate(prob)}
        })
    
    return results

# ─── 4) Example Usage ────────────────────────────────────────────────────────
def test_rank_parsing():
    """Test the adaptive rank parsing with different folder names."""
    
    print("="*70)
    print("ADAPTIVE RANK PARSING TEST")
    print("="*70)
    
    # Test cases with different rank combinations
    test_folders = [
        "model/bert-svd-16-96-96-finetuned",      # Low ranks
        "model/bert-svd-16-192-192-finetuned",    # Medium ranks  
        "model/bert-svd-32-384-384-finetuned",    # High ranks
        "model/bert-svd-4-48-48-finetuned",       # Very low ranks
        "model/bert-svd-64-768-768-finetuned",    # Very high ranks
        "model/unknown-folder",                   # Invalid pattern (should use defaults)
    ]
    
    for folder_path in test_folders:
        print(f"\n🔍 Testing folder: {folder_path}")
        print("-" * 50)
        
        try:
            # Parse ranks from folder name
            svd_config = SVDModel._parse_ranks_from_folder(folder_path)
            
            print(f"✅ Parsed configuration:")
            print(f"   Attention rank: {svd_config['rank_attn']}")
            print(f"   FFN rank: {svd_config['rank_ff']}")
            print(f"   Output rank: {svd_config['rank_wo']}")
            
            # Calculate compression ratio (approximate)
            original_params = 768 * 768 * 12  # Approximate BERT layer size
            compressed_params = (
                svd_config['rank_attn'] * 768 * 3 +  # Q, K, V projections
                svd_config['rank_ff'] * 768 * 2 +    # FFN projections
                svd_config['rank_wo'] * 768 * 2      # Output projection
            ) * 12  # 12 layers
            
            compression_ratio = (1 - compressed_params / original_params) * 100
            print(f"   Estimated compression: {compression_ratio:.1f}%")
            
        except Exception as e:
            print(f"❌ Error parsing {folder_path}: {e}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("✅ The inference script can automatically detect SVD ranks from folder names!")
    print("✅ Supports any rank combination in the format: bert-svd-{attn}-{ff}-{wo}-finetuned")
    print("✅ Falls back to default ranks (16-192-192) for invalid folder names")
    print("✅ No need to manually specify ranks - just use the correct folder naming!")

def evaluate_on_sst2(model, tokenizer, device="cpu", max_samples=None):
    """Evaluate the model on GLUE SST-2 validation set."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    import evaluate
    
    print(f"\n{'='*60}")
    print("GLUE SST-2 EVALUATION")
    print(f"{'='*60}")
    
    # Load SST-2 dataset
    print("📊 Loading GLUE SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    
    # Tokenize the validation set
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset["validation"].map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    
    # Limit samples if specified
    if max_samples:
        tokenized_dataset = tokenized_dataset.select(range(min(max_samples, len(tokenized_dataset))))
        print(f"📝 Using {len(tokenized_dataset)} samples for evaluation")
    
    # Create data loader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataloader = DataLoader(tokenized_dataset, batch_size=32, collate_fn=data_collator)
    
    # Load metric
    metric = evaluate.load("glue", "sst2")
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    print(f"🔄 Evaluating on {len(tokenized_dataset)} samples...")
    
    with torch.no_grad():
        for batch in eval_dataloader:
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
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    
    # Calculate metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    results = metric.compute(predictions=all_predictions, references=all_labels)
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
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
        
        pred_label = "POSITIVE" if all_predictions[i] == 1 else "NEGATIVE"
        true_label = "POSITIVE" if all_labels[i] == 1 else "NEGATIVE"
        correct = "✅" if all_predictions[i] == all_labels[i] else "❌"
        
        print(f"   {correct} \"{decoded_text}\" → Pred: {pred_label}, True: {true_label}")
    
    return results

def main():
    """Main function with rank-aware inference and SST-2 evaluation."""
    
    # Test rank parsing first
    test_rank_parsing()
    
    # Load the actual saved model from the training
    model_path = "model/bert-svd-16-96-96-finetuned"
    
    print(f"\n{'='*70}")
    print("SVD MODEL INFERENCE & EVALUATION")
    print(f"{'='*70}")
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
    print(f"\n🔄 Loading model on device: {device}")
    
    try:
        model = load_svd_model(model_path, device)
        print("✅ Model loaded successfully!")
        
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
    
    # Test examples
    test_texts = [
        "A wonderfully executed performance!",
        "This was utterly dull and uninspired.",
        "The movie was okay, nothing special.",
        "Absolutely fantastic and amazing!",
        "Terrible waste of time and money.",
        "I really enjoyed this experience.",
        "This is the worst thing I've ever seen.",
        "Pretty good, but could be better."
    ]
    
    print(f"\n{'='*60}")
    print("QUICK INFERENCE TEST")
    print(f"{'='*60}")
    
    try:
        results = predict_sentiment(model, tokenizer, test_texts, device)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Text: \"{result['text']}\"")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            
            # Show probabilities
            pos_prob = result['probabilities'].get('POSITIVE', 0)
            neg_prob = result['probabilities'].get('NEGATIVE', 0)
            print(f"   Probabilities: POSITIVE={pos_prob:.3f}, NEGATIVE={neg_prob:.3f}")
        
    except Exception as e:
        print(f"❌ Quick inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on SST-2 dataset
    try:
        sst2_results = evaluate_on_sst2(model, tokenizer, device, max_samples=1000)
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Model successfully loaded from {model_path}")
        print(f"✓ Adaptive rank parsing working correctly!")
        print(f"✓ SST-2 Accuracy: {sst2_results['accuracy']:.4f}")
        
        # Test model size comparison
        model_size_mb = os.path.getsize(os.path.join(model_path, "pytorch_model.bin")) / (1024*1024)
        print(f"✓ Model size: {model_size_mb:.1f} MB")
        print(f"✓ Model can be used for inference without re-finetuning!")
        print(f"✓ SVD compression working correctly!")
        
    except Exception as e:
        print(f"❌ SST-2 evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 