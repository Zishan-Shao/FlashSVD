import os
import math
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT on GLUE or MLM via HuggingFace Trainer")
    parser.add_argument("--task", type=str, required=True,
                        help="GLUE task name (e.g. qqp, sst2, cola) when mode=cls")
    parser.add_argument("--mode", type=str, choices=["cls","mlm"], default="cls",
                        help="Training mode: cls for classification, mlm for masked LM")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Model checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--bsz", type=int, default=32,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X steps")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    task_name = args.task
    training_mode = args.mode
    model_checkpoint = args.model

    # GPU setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # GLUE settings
    single_sentence = {"cola":128, "sst2":128, "qnli":512}
    pair_sentence   = {"qqp":256, "mnli":512, "stsb":256}

    if training_mode == "cls":
        if task_name in single_sentence:
            text_inputs = ("sentence", None)
            max_length = single_sentence[task_name]
        elif task_name in pair_sentence:
            if task_name == "mnli":
                text_inputs = ("sentence1","sentence2")
            else:
                text_inputs = ("question1","question2")
            max_length = pair_sentence[task_name]
        else:
            raise ValueError(f"Unknown GLUE task: {task_name}")
    else:
        # mlm uses single sentence or join pair
        if task_name in single_sentence:
            text_inputs = ("sentence", None)
            max_length = single_sentence[task_name]
        elif task_name in pair_sentence:
            if task_name == "mnli":
                text_inputs = ("sentence1","sentence2")
            else:
                text_inputs = ("question1","question2")
            max_length = pair_sentence[task_name]
        else:
            raise ValueError(f"Unknown dataset key for MLM: {task_name}")

    hf_logging.set_verbosity_error()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    raw = load_dataset("glue", task_name, cache_dir="./hf_cache", download_mode="force_redownload")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    if training_mode == "cls":
        collator = DataCollatorWithPadding(tokenizer)
        def preprocess(examples):
            return tokenizer(
                examples[text_inputs[0]], 
                examples[text_inputs[1]] if text_inputs[1] else None,
                padding="max_length", truncation=True, max_length=max_length
            )
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
        def preprocess(examples):
            if text_inputs[1]:
                joined = [f"{a} {tokenizer.sep_token} {b}" 
                          for a,b in zip(examples[text_inputs[0]], examples[text_inputs[1]])]
                return tokenizer(joined, padding="max_length", truncation=True, max_length=max_length)
            else:
                return tokenizer(examples[text_inputs[0]], padding="max_length", truncation=True, max_length=max_length)

    # map + format
    remove = [text_inputs[0]] + ([text_inputs[1]] if text_inputs[1] else [])
    tok = raw.map(preprocess, batched=True,
                  remove_columns=remove,
                  load_from_cache_file=False,
                  keep_in_memory=True,
                  num_proc=1)
    columns = ["input_ids","attention_mask"]
    if training_mode == "cls":
        columns.append("label")
    tok.set_format("torch", columns=columns)

    # model
    if training_mode == "cls":
        num_labels = raw["train"].features["label"].num_classes
        model = BertForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
        metric = evaluate.load("accuracy" if task_name!="stsb" else "pearsonr")
        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1) if task_name!="stsb" else p.predictions[:,0]
            return metric.compute(predictions=preds, references=p.label_ids)
        eval_split = "validation_matched" if task_name=="mnli" else "validation"
    else:
        model = BertForMaskedLM.from_pretrained(model_checkpoint)
        compute_metrics = None
        eval_split = "validation"

    model.to(device)

    # trainer args
    training_args = TrainingArguments(
        output_dir=f"out/{task_name}-{training_mode}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        learning_rate=args.lr,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        disable_tqdm=False,
        eval_strategy="steps",
        eval_steps=args.logging_steps*10,
        no_cuda=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok["train"],
        eval_dataset=tok[eval_split],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if training_mode=="cls" else None,
    )

    trainer.train()
    results = trainer.evaluate()
    if training_mode=="mlm":
        loss = results["eval_loss"]
        print(f"▶ Eval loss: {loss:.4f} → Perplexity: {math.exp(loss):.2f}")
    else:
        print(f"▶ {task_name.upper()} Eval Results:", results)

    # save
    save_dir = f"model/{model_checkpoint.split('/')[-1]}-{task_name}-{training_mode}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved to {save_dir}")

if __name__ == "__main__":
    main()