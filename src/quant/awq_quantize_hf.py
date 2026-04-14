#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from typing import List

# Allow running as `python quant/...py` from repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from quant.awq_compat import ensure_autoawq_compatibility


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Quantize a HuggingFace model directory with AutoAWQ (AWQ) and save a quantized HF folder."
    )
    ap.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Input HF model directory OR HF repo id (export first if needed).",
    )
    ap.add_argument("--out_dir", type=str, required=True, help="Output quantized HF directory.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite --out_dir if it already exists.")

    ap.add_argument("--wbits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--zero_point", action="store_true", help="Use asymmetric zero-point (AutoAWQ default is usually True).")
    ap.add_argument("--no_zero_point", action="store_true", help="Disable zero-point.")
    ap.add_argument("--version", type=str, default="GEMM", help="AutoAWQ kernel version string, e.g. GEMM.")

    ap.add_argument("--calib_dataset", type=str, default="", help="Optional: wikitext2|ptb|c4 (uses repo loaders).")
    ap.add_argument("--calib_nsamples", type=int, default=128)
    ap.add_argument("--calib_seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", help="float16|bfloat16|float32 (passed to from_pretrained)")
    ap.add_argument("--hf_token", type=str, default=None, help="Optional Hugging Face token.")
    return ap


def _pick_dtype(s: str):
    s = (s or "").strip().lower()
    import torch

    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    return None


def _safe_rmtree_out_dir(out_dir: str) -> None:
    if not os.path.exists(out_dir):
        return
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) == 0:
        shutil.rmtree(out_dir)
        return
    # Be conservative: only remove if it looks like a (possibly partial) HF model folder.
    try:
        if os.path.isdir(out_dir) and any(str(fn).endswith(".safetensors") for fn in os.listdir(out_dir)):
            shutil.rmtree(out_dir)
            return
    except Exception:
        pass
    hf_markers = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors",
        "pytorch_model.bin",
        "quant_config.json",
    )
    if any(os.path.exists(os.path.join(out_dir, fn)) for fn in hf_markers):
        shutil.rmtree(out_dir)
        return
    raise ValueError(f"--out_dir exists but does not look like an HF model directory: {out_dir}")


def _load_calib_texts(dataset: str, nsamples: int, seed: int, seqlen: int, tokenizer) -> List[str]:
    """
    Build a small calibration set as a list of strings to avoid relying on AutoAWQ's internal dataset downloads.
    """
    from utils.data_utils import get_loaders

    dataloader, _ = get_loaders(dataset, nsamples=int(nsamples), seed=int(seed), seqlen=int(seqlen), tokenizer=tokenizer)
    texts: List[str] = []
    for batch in dataloader:
        # batch: (1, seqlen) token ids
        ids = batch[0].tolist() if hasattr(batch, "shape") else list(batch)
        txt = tokenizer.decode(ids, skip_special_tokens=True)
        if txt.strip():
            texts.append(txt)
    return texts


def main() -> None:
    args = build_parser().parse_args()

    if os.path.isdir(args.model_dir) and not os.path.exists(os.path.join(args.model_dir, "config.json")):
        raise ValueError(
            f"--model_dir looks like a local folder but is missing config.json: {args.model_dir}\n"
            "Did the dense export/densify step finish successfully?"
        )

    if os.path.exists(args.out_dir):
        if not args.overwrite:
            raise ValueError(f"--out_dir exists: {args.out_dir} (use --overwrite)")
        _safe_rmtree_out_dir(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        ensure_autoawq_compatibility()
        from awq import AutoAWQForCausalLM  # pip install autoawq
    except Exception as e:
        raise RuntimeError(
            "AutoAWQ is not installed (pip package: autoawq). "
            "Create a separate env (python 3.10/3.11 recommended) and install autoawq.\n"
            f"Import error: {e}"
        )

    from transformers import AutoTokenizer

    dtype = _pick_dtype(args.dtype)
    print(f"[Load] {args.model_dir}")
    load_kwargs = dict(
        low_cpu_mem_usage=True,
        use_cache=False,
        torch_dtype=dtype,
        device_map=None,
    )
    try:
        if args.hf_token:
            model = AutoAWQForCausalLM.from_pretrained(args.model_dir, **load_kwargs, token=args.hf_token)
        else:
            model = AutoAWQForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    except TypeError:
        # Older AutoAWQ versions may not accept `token=`.
        model = AutoAWQForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    try:
        model = model.to(str(args.device))
    except Exception:
        pass
    try:
        if args.hf_token:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_dir,
                use_fast=False,
                trust_remote_code=True,
                token=args.hf_token,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_dir,
                use_fast=False,
                trust_remote_code=True,
            )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)

    zp = True
    if args.no_zero_point:
        zp = False
    if args.zero_point:
        zp = True

    quant_config = {
        "zero_point": bool(zp),
        "q_group_size": int(args.groupsize),
        "w_bit": int(args.wbits),
        "version": str(args.version),
    }
    print(f"[AWQ] quant_config={quant_config}")

    calib_texts = None
    if str(args.calib_dataset).strip():
        print(f"[Calib] dataset={args.calib_dataset} nsamples={args.calib_nsamples} seqlen={args.calib_seqlen}")
        calib_texts = _load_calib_texts(
            str(args.calib_dataset).strip(),
            nsamples=int(args.calib_nsamples),
            seed=int(args.seed),
            seqlen=int(args.calib_seqlen),
            tokenizer=tokenizer,
        )
        print(f"[Calib] texts={len(calib_texts)}")

    print("[AWQ] quantize...")
    # AutoAWQ's signature varies across versions; try the most common forms.
    try:
        if calib_texts is not None:
            model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
        else:
            model.quantize(tokenizer, quant_config=quant_config)
    except TypeError:
        # Older versions: quantize(quant_config, calib_data=..., tokenizer=...)
        if calib_texts is not None:
            model.quantize(quant_config=quant_config, tokenizer=tokenizer, calib_data=calib_texts)
        else:
            model.quantize(quant_config=quant_config, tokenizer=tokenizer)

    print(f"[Save] {args.out_dir}")
    model.save_quantized(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Some AutoAWQ versions don't write a usable HF config; ensure it exists and includes `model_type`.
    out_cfg = os.path.join(args.out_dir, "config.json")
    in_cfg = os.path.join(args.model_dir, "config.json") if os.path.isdir(args.model_dir) else None
    if (not os.path.exists(out_cfg)) and in_cfg and os.path.exists(in_cfg):
        shutil.copy2(in_cfg, out_cfg)
    if os.path.exists(out_cfg):
        try:
            with open(out_cfg, "r", encoding="utf-8") as f:
                cfg_json = json.load(f)
            if "model_type" not in cfg_json and in_cfg and os.path.exists(in_cfg):
                shutil.copy2(in_cfg, out_cfg)
        except Exception:
            pass
    if not os.path.exists(out_cfg):
        # As a last resort, try to save the config from the loaded model wrapper.
        try:
            base_cfg = getattr(getattr(model, "model", None), "config", None) or getattr(model, "config", None)
            if base_cfg is not None:
                base_cfg.save_pretrained(args.out_dir)
        except Exception:
            pass

    print("[Save] done.")


if __name__ == "__main__":
    main()
