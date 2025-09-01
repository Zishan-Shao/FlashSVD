import os 
import sys 
import torch
import time
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from evaluate import load as load_metric

# ─── locate repo & model ─────────────────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# use your finetuned ModernBERT checkpoint
MODEL_DIR = os.path.join(REPO_ROOT, "model", "modernbert-base-sst2")
BATCH_SIZE = 32
SEQ_LEN = 128 * 4 * 4

def prepare_dataloader(tokenizer):
    raw = load_dataset("glue", "sst2", split="validation")
    def tokenize(batch):
        return tokenizer(batch["sentence"],
                         padding="max_length",
                         truncation=True,
                         max_length=SEQ_LEN)
    ds = raw.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    ds.set_format("torch")
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: {
            "input_ids":      torch.stack([x["input_ids"]      for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels":         torch.tensor([x["label"]        for x in batch]),
        },
    )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    # 1. Tokenizer + DataLoader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    loader = prepare_dataloader(tokenizer)

    # 2. Load model
    cfg = AutoConfig.from_pretrained(MODEL_DIR, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, config=cfg
    ).to(device).eval()
    
    
    
    embedding_layer = model.model.embeddings.tok_embeddings
    vocab_size     = embedding_layer.num_embeddings    # 50368
    embedding_dim  = embedding_layer.embedding_dim     # 768

    # total parameters in the embedding matrix:
    num_params = vocab_size * embedding_dim            # 50368 × 768 = 38,700,  864

    # size in bytes (float32 = 4 bytes per parameter):
    size_bytes = num_params * 4                        # ≈154,730,496 bytes

    # convert to MiB
    size_mib = size_bytes / (1024**2)                  # ≈147.6 MiB

    print(f"Embedding matrix: {vocab_size}×{embedding_dim} = {num_params:,} params")
    print(f"Size: {size_mib:.1f} MiB")
    
    
    
    # 3. Run inference & measure accuracy
    metric = load_metric("accuracy")
    # ── RESET STATS ────────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            preds = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=preds.cpu(),
                             references=batch["labels"].cpu())

    # elapsed = (time.perf_counter() - start_time)
    # acc = metric.compute()["accuracy"]
    # ── SYNCHRONIZE & READ PEAK ───────────────────────────────────────────
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    acc     = metric.compute()["accuracy"]
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"SST-2 validation accuracy: {acc*100:.2f}%")
    print(f"Total eval time:    {elapsed:.1f}s  ({elapsed/len(loader):.5f}s per batch)")
    print(f"Peak GPU memory:    {peak_mb:6.1f} MiB")

    # print(f"SST-2 validation accuracy: {acc*100:.2f}%")
    # print(f"Total eval time: {elapsed:.1f}s  ({elapsed/len(loader):.1f}s per batch)\n")
    
if __name__ == "__main__":
    main()














































# import os
# import sys
# import math
# import time

# import torch
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from transformers import (
#     AutoTokenizer,
#     AutoConfig,
#     AutoModelForMaskedLM,
#     DataCollatorForLanguageModeling,
# )

# # ─── locate repo & model ─────────────────────────────────────────────────────
# THIS_FILE = os.path.abspath(__file__)
# REPO_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)

# # use your fine-tuned ModernBERT MLM checkpoint
# MODEL_DIR = os.path.join(REPO_ROOT, "model", "modernbert-base-mlm")


# BATCH_SIZE = 4
# SEQ_LEN = 128 * 4 * 2 



# def prepare_dataloader(tokenizer):
#     # 1) load wikitext validation split
#     raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

#     # 2) tokenize
#     def tokenize_fn(examples):
#         return tokenizer(
#             examples["text"],
#             truncation=True,
#             max_length=SEQ_LEN,
#             padding="max_length",
#         )

#     tok = raw.map(
#         tokenize_fn,
#         batched=True,
#         remove_columns=["text"],
#     )

#     # 3) convert to torch tensors
#     tok.set_format(type="torch", columns=["input_ids", "attention_mask"])

#     # 4) dynamic‐mask collator
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer,
#         mlm=True,
#         mlm_probability=0.15,
#     )

#     # 5) DataLoader
#     return DataLoader(
#         tok,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         collate_fn=data_collator,
#     )

# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"\nUsing device: {device}\n")

#     # 1) tokenizer & dataloader
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#     loader = prepare_dataloader(tokenizer)

#     # 2) load MLM model
#     cfg = AutoConfig.from_pretrained(MODEL_DIR)
#     model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, config=cfg)
#     model.to(device)
#     model.eval()

#     base_mem = torch.cuda.max_memory_allocated() / 1024**2
#     print(f"Persistent dense model storage: {base_mem:6.1f} MiB")

#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
#     torch.cuda.synchronize()

#     # 3) run inference, accumulate loss
#     total_loss = 0.0
#     total_batches = 0
#     start_time = time.perf_counter()
#     with torch.no_grad():
#         for batch in loader:
#             # move inputs & labels to device
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 labels=batch["labels"],        # collator puts masked token labels here
#             )
#             loss = outputs.loss
#             total_loss += loss.item()
#             total_batches += 1

#     elapsed = time.perf_counter() - start_time
#     avg_loss = total_loss / total_batches
#     ppl = math.exp(avg_loss)
    
#     peak = torch.cuda.max_memory_allocated() / 1024**2
#     print(f"Peak Mem: {peak:6.1f} MiB | Transient: {peak-base_mem:6.1f} MiB")

#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
#     torch.cuda.synchronize()


#     print(f"Validation batches: {total_batches}")
#     print(f"Average MLM loss:   {avg_loss:.4f}")
#     print(f"Perplexity:         {ppl:.2f}")
#     print(f"Total eval time:    {elapsed:.1f}s  ({elapsed/total_batches:.2f}s per batch)\n")

# if __name__ == "__main__":
#     main()

























