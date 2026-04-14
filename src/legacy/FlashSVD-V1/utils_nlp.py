
"""
utils_nlp.py
Reusable utilities for GLUE/MLM training with Hugging Face Transformers.
"""

import os
import math
from typing import Optional, Tuple, Dict, Any, Callable

import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    BertForSequenceClassification,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
)

# -----------------------------
# Task configuration
# -----------------------------

# GLUE task => (sentence1_key, optional sentence2_key)
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
}

def maybe_set_cuda_env(cuda_visible_devices: Optional[str]) -> None:
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def default_output_dir(mode: str, task: str, output_dir: Optional[str]) -> str:
    if output_dir and output_dir.strip():
        return output_dir
    return os.path.join("out", f"{mode}-{task}")

def load_glue(task: str) -> Tuple[DatasetDict, str]:
    """Load GLUE and return (dataset, validation_split_name)."""
    raw = load_dataset("glue", task)
    # rename label -> labels if present
    for split in raw.keys():
        if "label" in raw[split].column_names:
            raw[split] = raw[split].rename_column("label", "labels")
    val_name = "validation_matched" if task == "mnli" else "validation"
    return raw, val_name

def build_tokenize_fn(tokenizer, task: str):
    s1, s2 = TASK_TO_KEYS[task]
    if s2 is None:
        def tok(examples):
            return tokenizer(examples[s1], truncation=True)
        return tok
    else:
        def tok(examples):
            return tokenizer(examples[s1], examples[s2], truncation=True)
        return tok

def is_regression(task: str) -> bool:
    return task == "stsb"

def num_labels(task: str) -> int:
    return 1 if is_regression(task) else (3 if task == "mnli" else 2)

def build_metrics(task: str) -> Callable:
    if task == "stsb":
        metric = evaluate.load("glue", "stsb")
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.squeeze(preds)
            return metric.compute(predictions=preds, references=labels)
        return compute_metrics
    else:
        metric = evaluate.load("glue", task)
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=labels)
        return compute_metrics

def build_collator_and_model(mode: str, tokenizer, task: str):
    """Return (data_collator, model, compute_metrics or None)."""
    if mode == "mlm":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        compute_metrics = None
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels(task),
            problem_type="regression" if is_regression(task) else None,
        )
        compute_metrics = build_metrics(task)
    return data_collator, model, compute_metrics

def build_training_args(
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    logging_steps: int = 100,
    eval_steps: int = 0,
    no_cuda: bool = False,
) -> TrainingArguments:
    evaluation_strategy = "steps" if eval_steps and eval_steps > 0 else "epoch"
    save_strategy = evaluation_strategy
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps if evaluation_strategy == "steps" else None,
        load_best_model_at_end=False,
        report_to=["none"],
        no_cuda=no_cuda,
        fp16=not no_cuda,
    )

def prepare_trainer(
    model,
    training_args: TrainingArguments,
    tokenized,
    val_split: str,
    tokenizer,
    data_collator,
    compute_metrics=None,
) -> Trainer:
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[val_split],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

def maybe_add_perplexity(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if "eval_loss" in metrics:
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except Exception:
            pass
    return metrics
