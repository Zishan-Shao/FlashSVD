import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)
# 1) setup
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
hf_logging.set_verbosity_error()
device = "cuda"
import math
from transformers import Trainer

# Path where you saved your fine-tuned model/tokenizer
out_dir = "model/modernbert-base-mlm"

# 9.1 Reload
model = AutoModelForMaskedLM.from_pretrained(out_dir, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)

# 9.2 Prepare test split
raw_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
tok_test = raw_test.map(tokenize_fn, batched=True, remove_columns=["text"])
# tok_test.set_format("torch")  # not strictly necessary for Trainer.evaluate()

# 9.3 Evaluate with a new Trainer to get eval_loss → perplexity
eval_trainer = Trainer(
    model=model,
    args=training_args,         # reuse your TrainingArguments
    eval_dataset=tok_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

metrics = eval_trainer.evaluate()
print(f"▶ Test MLM loss:     {metrics['eval_loss']:.4f}")
print(f"▶ Test Perplexity:    {math.exp(metrics['eval_loss']):.2f}")

