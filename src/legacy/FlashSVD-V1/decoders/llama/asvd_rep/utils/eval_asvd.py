import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_utils import evaluate_model



# python eval_asvd.py --model_dir svd_model/model-facebook-opt-125m_actaware-1_alpha-1.0_ncalib-16_calib-wikitext2_scale-abs_mean_sens-ppl_wq-none_sigma-UV_kv-0_kvrt--1_rankalign-1_seed-233 --eval_ppl wikitext2,ptb 



def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Load model and tokenizer from the specified directory
    print(f"Loading model from {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Evaluate
    result = evaluate_model(
        model,
        tokenizer,
        args.model_dir,
        "mmlu" if args.eval_mmlu else args.eval_tasks,
        eval_ppl=args.eval_ppl,
        limit=-1,
        use_bos=args.use_bos,
    )
    print(result)
    if not os.path.exists("output"):
        os.makedirs("output")
    with open("output/eval_result.txt", "a+") as f:
        f.write(f"{args}\n")
        f.write(f"{result}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the saved SVD model directory (e.g., svd_model/svd-opt-125m-0.60-actaware)",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="Evaluate MMLU tasks",
    )
    parser.add_argument(
        "--eval_ppl",
        default="wikitext2,ptb",
        type=str,
        help="Comma-separated list of datasets for perplexity evaluation",
    )
    parser.add_argument("--eval_tasks", type=str, default="", help="Other evaluation tasks")
    parser.add_argument(
        "--use_bos",
        action="store_true",
        help="Use BOS token in evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="Random seed",
    )
    args = parser.parse_args()
    main(args) 