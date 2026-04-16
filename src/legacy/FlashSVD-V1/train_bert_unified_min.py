
"""
train_bert_unified_min.py
A slim entry script that uses utils_nlp.py to train CLS or MLM.
"""
import os
import argparse
from transformers import AutoTokenizer, set_seed
from datasets import DatasetDict

from utils_nlp import (
    TASK_TO_KEYS,
    maybe_set_cuda_env,
    default_output_dir,
    load_glue,
    build_tokenize_fn,
    build_collator_and_model,
    build_training_args,
    prepare_trainer,
    maybe_add_perplexity,
)

def build_argparser():
    p = argparse.ArgumentParser(description="Slim unified BERT trainer using utils_nlp.")
    p.add_argument("--mode", type=str, default="cls", choices=["cls", "mlm"])
    p.add_argument("--task", type=str, default="rte", choices=list(TASK_TO_KEYS.keys()))
    p.add_argument("--model", type=str, default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--cuda_visible_devices", type=str, default=None)
    return p

def main():
    args = build_argparser().parse_args()
    maybe_set_cuda_env(args.cuda_visible_devices)
    set_seed(args.seed)

    out_dir = default_output_dir(args.mode, args.task, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    raw, val_split = load_glue(args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok_fn = build_tokenize_fn(tokenizer, args.task)
    columns = raw['train'].column_names

    if args.mode == 'cls':
        remove_columns = [c for c in columns if c not in ('label', 'labels')]
    else:
        remove_columns = columns

    tokenized = raw.map(tok_fn, batched=True, remove_columns=remove_columns)

    data_collator, model, compute_metrics = build_collator_and_model(args.mode, tokenizer, args.task)
    # If you want to override default model checkpoint from utils, do it here:
    if args.model and args.model != "bert-base-uncased":
        # reload model with user-specified checkpoint
        if args.mode == "mlm":
            from transformers import BertForMaskedLM
            model = BertForMaskedLM.from_pretrained(args.model)
        else:
            from transformers import BertForSequenceClassification
            from utils_nlp import is_regression, num_labels
            model = BertForSequenceClassification.from_pretrained(
                args.model,
                num_labels=num_labels(args.task),
                problem_type="regression" if is_regression(args.task) else None,
            )

    training_args = build_training_args(
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        no_cuda=args.no_cuda,
    )

    trainer = prepare_trainer(
        model=model,
        training_args=training_args,
        tokenized=tokenized,
        val_split=val_split,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model(out_dir)

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate(eval_dataset=tokenized[val_split])
    if args.mode == "mlm":
        eval_metrics = maybe_add_perplexity(eval_metrics)

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("\\n=== Finished ===")
    print(f"Output directory: {out_dir}")
    if args.mode == "mlm" and "perplexity" in eval_metrics:
        print(f"Perplexity: {eval_metrics['perplexity']:.4f}")
    elif args.mode == "cls":
        print("Eval metrics:", eval_metrics)

if __name__ == "__main__":
    main()
