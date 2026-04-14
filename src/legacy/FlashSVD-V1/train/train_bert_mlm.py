import os
import math
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    BertForSequenceClassification,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)
import torch 

# ───────────────────────────────────────────────────────────────────────────────
# 0) User config ───────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
task_name       = "rte"                  # e.g. "cola","sst2","qqp","stsb","mnli","qnli"
training_mode   = "mlm"                   # "cls" = classification; "mlm" = masked LM
model_checkpoint = "bert-base-uncased"

# pick your device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Chosen device at script start: {device}")

# force Trainer to see only GPU #6 if you have multiple
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Sequence lengths per GLUE split
single_sent_tasks = {"cola":128, "sst2":128, "qnli":512}
pair_sent_tasks   = {"qqp":256, "mnli":512, "stsb":256, "mrpc":128, "rte":128} # mnli, stsb, qqp, qnli, sst2, cola [finished[]

# ───────────────────────────────────────────────────────────────────────────────
# 1) Determine text fields & max_length ────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
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

# if task_name in single_sent_tasks:
#     text_inputs = ("sentence", None)
#     max_length  = single_sent_tasks[task_name]   
# elif task_name in pair_sent_tasks:
#     if task_name == "qqp": #or task_name == "mnli":
#         first, second = "question1", "question2"
#     else:
#         # mnli, stsb, etc. all use sentence1/sentence2
#         first, second = "sentence1", "sentence2"
#     text_inputs = (first, second)
#     max_length  = pair_sent_tasks[task_name]
# else:
#     raise ValueError(f"Unsupported task: {task_name}")

# ───────────────────────────────────────────────────────────────────────────────
# 2) Load dataset & tokenizer & logging ────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
hf_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

raw = load_dataset("glue", task_name, download_mode="force_redownload")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# ───────────────────────────────────────────────────────────────────────────────
# 3) Preprocessing functions ───────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
def tokenize_cls(examples):
    t1 = examples[text_inputs[0]]
    t2 = examples[text_inputs[1]] if text_inputs[1] else None
    return tokenizer(t1, t2, padding="max_length", truncation=True, max_length=max_length)

def tokenize_mlm(examples):
    if text_inputs[1]:
        joined = [f"{a} {tokenizer.sep_token} {b}" 
                  for a,b in zip(examples[text_inputs[0]], examples[text_inputs[1]])]
        return tokenizer(joined, padding="max_length", truncation=True, max_length=max_length)
    else:
        return tokenizer(examples[text_inputs[0]], padding="max_length", truncation=True, max_length=max_length)

if training_mode == "cls":
    tok = raw.map(
        tokenize_cls, batched=True,
        remove_columns=raw["train"].column_names
    )
    labels_name = "label"
    tok.set_format("torch", columns=["input_ids","attention_mask",labels_name])

    metric = evaluate.load("accuracy" if task_name!="stsb" else "pearsonr")
    def compute_metrics(p):
        if task_name=="stsb":
            preds = p.predictions[:,0]
        else:
            preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    model = BertForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=raw["train"].features[labels_name].num_classes
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    eval_split = "validation_matched" if task_name=="mnli" else "validation"

# elif training_mode == "mlm":
#     tok = raw.map(
#         tokenize_mlm, batched=True,
#         remove_columns=raw["train"].column_names
#     )
#     tok.set_format("torch", columns=["input_ids","attention_mask"])

#     model = BertForMaskedLM.from_pretrained(model_checkpoint)
#     data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

#     compute_metrics = None
#     eval_split = "validation"
elif training_mode == "mlm":
    tok = raw.map(
        tokenize_mlm, batched=True,
        remove_columns=raw["train"].column_names
    )
    tok.set_format("torch", columns=["input_ids","attention_mask"])

    model = BertForMaskedLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)

    # ← ADD THIS:
    if task_name == "mnli":
        eval_split = "validation_matched"
    else:
        eval_split = "validation"

    compute_metrics = None

else:
    raise ValueError(f"Unknown training_mode: {training_mode}")

# ───────────────────────────────────────────────────────────────────────────────
# 3.5) Send model to chosen device ──────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
model.to(device)
print(f"Model now on device: {next(model.parameters()).device}")

# ───────────────────────────────────────────────────────────────────────────────
# 4) TrainingArguments & Trainer ────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir             = f"out/{task_name}-{training_mode}",
    num_train_epochs       = 3,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    save_strategy          = "no",
    learning_rate          = 2e-5,
    logging_steps          = 100,
    seed                   = 0,
    logging_dir            = "./logs",
    report_to              = [],
    disable_tqdm           = False,
    eval_strategy    = "steps",
    eval_steps             = 3000,
    no_cuda                = False,           # <— ensure CUDA is allowed
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tok["train"],
    eval_dataset    = tok[eval_split],
    data_collator   = data_collator,
    tokenizer       = tokenizer,
    compute_metrics = compute_metrics,
)

# Print the device the Trainer will use:
print(f"Trainer device: {trainer.args.device}")

# ───────────────────────────────────────────────────────────────────────────────
# 5) Train & Evaluate ───────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
trainer.train()
results = trainer.evaluate()

if training_mode=="mlm":
    loss = results["eval_loss"]
    print(f"▶ Eval loss: {loss:.4f} → Perplexity: {math.exp(loss):.2f}")
else:
    print(f"▶ {task_name.upper()} Eval Metrics:", results)

# ───────────────────────────────────────────────────────────────────────────────
# 6) Save model & tokenizer ────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
save_dir = f"model/{model_checkpoint.split('/')[-1]}-{task_name}-{training_mode}"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✓ Saved to ./{save_dir}")
