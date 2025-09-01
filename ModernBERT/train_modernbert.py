import os
import torch
import numpy as np
from datasets import load_dataset#, load_metric
from evaluate import load as load_metric

# training is for finetuning the ModernBERT first, if you want to try other datasets
# you should finetuning them here with this script, using Cursor if it does not work

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)

# 1) setup
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
hf_logging.set_verbosity_error()
model_id = "answerdotai/ModernBERT-base"

# 2) load & patch config for full-attention + classification head
cfg = AutoConfig.from_pretrained(
    model_id,
    trust_remote_code=True,              # trust ModernBertConfig
)
cfg.global_every_n_layers = 1
cfg.local_window_size = cfg.max_position_embeddings
cfg.num_labels = 2                     # SST-2: positive / negative

# 3) load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    config=cfg,
    trust_remote_code=True,              # trust ModernBertForSequenceClassification
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 4) prepare data (GLUE SST-2)
raw = load_dataset("glue", "sst2")
metric = load_metric("glue", "sst2")

def preprocess_fn(examples):
    # SST-2 has one field "sentence" and label
    encodings = tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128*4,
    )
    encodings["labels"] = examples["label"]
    return encodings

# tokenize train & validation
tok_train = raw["train"].map(preprocess_fn, batched=True, remove_columns=raw["train"].column_names)
tok_val   = raw["validation"].map(preprocess_fn, batched=True, remove_columns=raw["validation"].column_names)

# 5) metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 6) training arguments
training_args = TrainingArguments(
    output_dir="out_sst2",
    overwrite_output_dir=True,
    #evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=100,
    fp16=True,
    seed=0,
)

# 7) trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# 8) fine-tune
trainer.train()

# 9) save
out_dir = "model/modernbert-base-sst2"
os.makedirs(out_dir, exist_ok=True)
trainer.save_model(out_dir)
tokenizer.save_pretrained(out_dir)
print(f"✓ Saved fine-tuned ModernBERT SST-2 model to ./{out_dir}")





