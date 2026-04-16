import os
import torch
import numpy as np
from datasets import load_dataset
from evaluate import load as load_metric

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)

# 1) setup
task_name        = "rte"                        # change as needed

# pick your device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Chosen device at script start: {device}")

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
hf_logging.set_verbosity_error()
model_id = "roberta-large"  # <-- switch to RoBERTa-large

single_sent_tasks = {"cola":128, "sst2":128, "qnli":512}
pair_sent_tasks   = {"qqp":256, "mnli":512, "stsb":256, "mrpc":128, "rte":128}  # etc.

if task_name in single_sent_tasks:
    text_inputs = ( "sentence", None )  # Cola, SST-2 
    max_length  = single_sent_tasks[task_name]

elif task_name in pair_sent_tasks:
    # explicit mapping for each task
    field_map = {
      "qqp":  ("question1",   "question2"),
      "mnli": ("premise",     "hypothesis"),
      "stsb": ("sentence1",   "sentence2"),
      "mrpc": ("sentence1",   "sentence2"),
      "rte":  ("sentence1",   "sentence2"),
      "qnli": ("question",    "sentence"),
    }
    first, second = field_map[task_name]
    text_inputs   = (first, second)
    max_length    = pair_sent_tasks[task_name]

else:
    raise ValueError(f"Unsupported task: {task_name}")


# 2) load config for sequence classification
# MNLI has 3 labels: entailment (0), neutral (1), contradiction (2)
#num_labels = 3 if task_name == "mnli" else 2
num_labels = 1 if task_name == "stsb" else (3 if task_name == "mnli" else 2)

cfg = AutoConfig.from_pretrained(
    model_id,
    num_labels=num_labels,
    problem_type="regression" if task_name == "stsb" else None,
)

# 3) load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, config=cfg)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 4) prepare data (GLUE SST-2)
raw = load_dataset("glue", task_name)
metric = load_metric("glue", task_name) # should be 
if task_name == "stsb":
    metric = load_metric("glue", "stsb")
    def compute_metrics(pred):
        preds, labels = pred
        preds = preds.squeeze(-1)
        return {
            "pearson": metric.compute(predictions=preds, references=labels)["pearson"],
            "spearman": metric.compute(predictions=preds, references=labels)["spearmanr"],
        }


def preprocess_fn(examples):
    if task_name in single_sent_tasks:
        # Single sentence tasks (e.g., SST-2, CoLA)
        encodings = tokenizer(
            examples[text_inputs[0]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    else:
        # Pair sentence tasks (e.g., MNLI, QQP)
        encodings = tokenizer(
            examples[text_inputs[0]],
            examples[text_inputs[1]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    encodings["labels"] = examples["label"]
    return encodings

tok_train = raw["train"].map(
    preprocess_fn,
    batched=True,
    remove_columns=raw["train"].column_names
)

# For MNLI, use validation_matched (you can also use validation_mismatched if needed)
val_key = "validation_matched" if task_name == "mnli" else "validation"
tok_val = raw[val_key].map(
    preprocess_fn,
    batched=True,
    remove_columns=raw[val_key].column_names
)

# 5) metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 6) training arguments
training_args = TrainingArguments(
    output_dir=f"out_{task_name}_roberta-large",
    overwrite_output_dir=True,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=8,   # consider halving batch-size for large model
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,   # to keep effective batch-size
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
out_dir = f"model/roberta-large-{task_name}"
os.makedirs(out_dir, exist_ok=True)
trainer.save_model(out_dir)
tokenizer.save_pretrained(out_dir)
print(f"✓ Saved fine-tuned RoBERTa-large-{task_name} model to ./{out_dir}")
