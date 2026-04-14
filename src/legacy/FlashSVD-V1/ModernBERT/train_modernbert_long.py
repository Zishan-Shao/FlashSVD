import os
import torch
import numpy as np
from typing import List, Optional
from datasets import load_dataset
from evaluate import load as load_metric

# training is for finetuning the ModernBERT first, if you want to try other datasets
# this script now supports GLUE/SST-2 and MMLU (lukaemon/mmlu or hendrycks_test)

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    logging as hf_logging,
)


def _mmlu_example_to_text_and_label(ex, source: str = "lukaemon"):
    """
    Normalize an example from MMLU variants to (text, label_id in {0..3}).
    - lukaemon/mmlu: typically has fields: input, A, B, C, D, and target/answer (letter)
    - hendrycks_test: fields: question, choices (list[str]), answer (letter)
    """
    if source == "lukaemon":
        q = ex.get("input") or ex.get("question") or ""
        A = ex.get("A") or ex.get("a") or (ex.get("choices") or ["", "", "", ""])[0]
        B = ex.get("B") or ex.get("b") or (ex.get("choices") or ["", "", "", ""])[1]
        C = ex.get("C") or ex.get("c") or (ex.get("choices") or ["", "", "", ""])[2]
        D = ex.get("D") or ex.get("d") or (ex.get("choices") or ["", "", "", ""])[3]
        ans = ex.get("target") or ex.get("answer") or ex.get("label")
    else:  # hendrycks_test
        q = ex.get("question") or ex.get("input") or ""
        ch = ex.get("choices") or []
        if isinstance(ch, list) and len(ch) >= 4:
            A, B, C, D = ch[:4]
        else:
            A = ex.get("A") or ex.get("a") or ""
            B = ex.get("B") or ex.get("b") or ""
            C = ex.get("C") or ex.get("c") or ""
            D = ex.get("D") or ex.get("d") or ""
        ans = ex.get("answer") or ex.get("target") or ex.get("label")

    text = f"Question: {q}\nA) {A}\nB) {B}\nC) {C}\nD) {D}"

    # Normalize answer to id 0..3
    if isinstance(ans, (bytes, bytearray)):
        ans = ans.decode("utf-8", errors="ignore")
    if isinstance(ans, str):
        ans = ans.strip().upper()
        map_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
        label = map_idx.get(ans, 0)
    elif isinstance(ans, int):
        label = int(ans) if 0 <= int(ans) <= 3 else 0
    else:
        label = 0

    return text, label


def build_mmlu_dataset(source: str, split: str, subjects: Optional[List[str]]):
    """
    Load MMLU dataset and return HF datasets for train/eval based on split.
    If split is 'dev' or 'validation', train on that and hold out 5% for eval.
    If split is 'test', use 'dev' for train, 'test' for eval.
    """
    src = source
    subs = subjects or []

    def _load_one(cfg: str, sp: str):
        if src == "lukaemon":
            # lukaemon/mmlu supports 'dev' and 'test', with config per subject or 'all'
            ds = load_dataset("lukaemon/mmlu", cfg, split=sp)
        else:
            # hendrycks_test uses many subject configs; split names are validation/test
            sp2 = "validation" if sp in ("dev", "validation", "val") else "test"
            ds = load_dataset("hendrycks_test", cfg, split=sp2)
        return ds

    if src == "lukaemon":
        cfgs = subs if subs else ["all"]
    else:
        # pick a reasonable default subset if user didn't specify
        cfgs = subs if subs else [
            "abstract_algebra", "anatomy", "astronomy", "high_school_physics",
            "college_chemistry", "college_mathematics",
        ]

    # Train/eval split policy
    if split in ("test",):
        train_split = "dev"
        eval_split = "test"
    else:
        train_split = split if split not in ("validation", "val") else "dev"
        eval_split = train_split

    # Load datasets
    from datasets import concatenate_datasets
    train_parts, eval_parts = [], []
    for cfg in cfgs:
        tr = _load_one(cfg, train_split)
        ev = _load_one(cfg, eval_split)
        train_parts.append(tr)
        eval_parts.append(ev)

    train_ds = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
    eval_ds  = concatenate_datasets(eval_parts)  if len(eval_parts)  > 1 else eval_parts[0]

    # If train and eval are from the same split, create a small holdout for evaluation
    if eval_split == train_split:
        split_ds = train_ds.train_test_split(test_size=0.05, seed=0)
        return split_ds["train"], split_ds["test"]
    else:
        return train_ds, eval_ds


def main():
    import argparse
    hf_logging.set_verbosity_error()

    p = argparse.ArgumentParser("ModernBERT long finetuning (SST-2 or MMLU)")
    p.add_argument("--dataset", choices=["sst2", "mmlu"], default="sst2")
    p.add_argument("--model-id", default="answerdotai/ModernBERT-base")
    p.add_argument("--output-dir", default="ModernBERT/out")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    # MMLU-specific
    p.add_argument("--mmlu-source", choices=["lukaemon", "hendrycks"], default="lukaemon")
    p.add_argument("--mmlu-split", choices=["dev", "validation", "val", "test"], default="dev")
    p.add_argument("--mmlu-subjects", default="", help="Comma-separated list; empty means all/default subset")

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load & patch config for long attention and classification head
    cfg = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    cfg.global_every_n_layers = 1
    cfg.local_window_size = cfg.max_position_embeddings
    cfg._attn_implementation = "sdpa"

    if args.dataset == "sst2":
        cfg.num_labels = 2
    else:
        cfg.num_labels = 4  # A/B/C/D

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        config=cfg,
        trust_remote_code=True,
    ).to(device)

    # Data + metrics
    if args.dataset == "sst2":
        raw = load_dataset("glue", "sst2")
        metric = load_metric("glue", "sst2")

        def preprocess_fn(examples):
            enc = tokenizer(
                examples["sentence"],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
            enc["labels"] = examples["label"]
            return enc

        tok_train = raw["train"].map(preprocess_fn, batched=True, remove_columns=raw["train"].column_names)
        tok_val   = raw["validation"].map(preprocess_fn, batched=True, remove_columns=raw["validation"].column_names)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return metric.compute(predictions=preds, references=labels)

        out_dir = os.path.join("ModernBERT", "model", "modernbert-base-sst2")

    else:  # MMLU
        subjects = [s.strip() for s in args.mmlu_subjects.split(",") if s.strip()]
        train_ds, eval_ds = build_mmlu_dataset(args.mmlu_source, args.mmlu_split, subjects)

        metric = load_metric("accuracy")

        # Preprocess for classification over 4 choices
        def preprocess_fn(examples):
            texts, labels = [], []
            for i in range(len(examples[list(examples.keys())[0]])):
                ex = {k: examples[k][i] for k in examples}
                text, lab = _mmlu_example_to_text_and_label(ex, source=args.mmlu_source)
                texts.append(text)
                labels.append(lab)
            enc = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )
            enc["labels"] = labels
            return enc

        remove_cols_train = train_ds.column_names
        remove_cols_eval  = eval_ds.column_names
        tok_train = train_ds.map(preprocess_fn, batched=True, remove_columns=remove_cols_train)
        tok_val   = eval_ds.map(preprocess_fn, batched=True, remove_columns=remove_cols_eval)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return metric.compute(predictions=preds, references=labels)

        out_dir = os.path.join("ModernBERT", "model", "modernbert-base-mmlu")

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        seed=args.seed,
        evaluation_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    os.makedirs(out_dir, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"✓ Saved fine-tuned ModernBERT model to ./{out_dir}")


if __name__ == "__main__":
    main()



