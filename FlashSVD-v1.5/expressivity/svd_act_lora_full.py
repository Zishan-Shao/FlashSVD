import argparse
import json
import os
import random
import re
import sys
import time
from typing import Optional, List, Dict, Any

import torch
from accelerate import dispatch_model, infer_auto_device_map
import torch.nn as nn

# Ensure repo root is on PYTHONPATH when running from this subdirectory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.model_utils import get_model_from_huggingface
from utils.data_utils import get_calib_train_data, get_mixed_calib_train_data, get_loaders
from utils.Prompter import Prompter
from evaluater import ppl_eval
from SVDLLM import profle_svdllm, whitening
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP

# Optional local MathQA loader (avoid HF download if local dataset is present)
load_mathqa_local = None
try:
    from datasets.load_data import load_mathqa_local as _lm_local  # type: ignore
    load_mathqa_local = _lm_local
except Exception:
    try:
        import importlib.util as _ilu
        _base = os.path.join(_REPO_ROOT, 'datasets', 'load_data.py')
        if os.path.isfile(_base):
            _spec = _ilu.spec_from_file_location('local_datasets_load_data', _base)
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)  # type: ignore
                load_mathqa_local = getattr(_mod, 'load_mathqa_local', None)
    except Exception:
        load_mathqa_local = None

'''
# 这个script跑的很好，不要丢了

PPL after activation-LoRA: {'wikitext2': np.float64(9.395533107478983)}
Weight Memory: 16492.34 MiB
Peak Memory (allocated): 17248.43 MiB
Peak Memory (reserved):  21152.00 MiB

# this looks like when lora rank 16 --> 32, our performance rule!

# 这个参数量没对齐
CUDA_VISIBLE_DEVICES=3 python expressivity/svd_act_lora_full.py   --model meta-llama/Llama-2-7b-hf   --keep_ratio 0.6   --whitening_factorization svd   --whitened_cache ./checkpoints/opt/llama-2-7b-hf_whitening_only_0.6_svd.pt   --whitening_lm_datasets wikitext2,ptb,c4   --lora_nsamples 4096 --seqlen 2048 --train_batch_size 8   --epochs 2 --lr 2e-4 --lora_rank 32 --lora_alpha 32   --mix_buckets   --bucket_props LM:0.2,INST:0.6,MATH:0.2   --bucket_lm_datasets wikitext2,ptb,c4   --bucket_inst_datasets yahma/alpaca-cleaned,hellaswag,piqa,winogrande_xl,ai2_arc_easy,ai2_arc_challenge,openbookqa,sst2,cola   --bucket_math_datasets mathqa,gsm8k   --save_path ./checkpoints/opt/llama-2-7b-hf_act_lora_fullseqLM_mixed_0.6.pt

# 对齐参数量：
CUDA_VISIBLE_DEVICES=3 python expressivity/svd_act_lora_full.py \
  --model meta-llama/Llama-2-7b-hf \
  --keep_ratio 0.6 \
  --whitening_factorization svd \
  --attn_keep_ratio 0.6 \
  --mlp_keep_ratio 0.6 \
  --whitened_cache ./checkpoints/opt/llama-2-7b-hf_whitening_only_0.6_svdllm2ratio.pt \
  --whitening_lm_datasets wikitext2,ptb,c4 \
  --lora_nsamples 4096 --seqlen 2048 --train_batch_size 8 \
  --epochs 3 --lr 2e-4 --lora_rank 32 --lora_alpha 32 \
  --mix_buckets \
  --bucket_props LM:0.2,INST:0.6,MATH:0.2 \
  --bucket_lm_datasets wikitext2,ptb,c4 \
  --bucket_inst_datasets yahma/alpaca-cleaned,hellaswag,piqa,winogrande_xl,ai2_arc_easy,ai2_arc_challenge,openbookqa,sst2,cola \
  --bucket_math_datasets mathqa,gsm8k \
  --save_path ./checkpoints/opt/llama-2-7b-hf_act_lora_fullseqLM_mixed_svdllm2ratio_0.6.pt

# test speed performance:
CUDA_VISIBLE_DEVICES=3 python -u expressivity/svd_act_lora_full.py \
  --model meta-llama/Llama-2-7b-hf \
  --keep_ratio 0.4 \
  --seqlen 2048 --whitening_nsamples 256 \
  --epochs 2 --lr 2e-4 --lora_rank 32 --lora_alpha 32 \
  --force_recompress --skip_eval \
  --timing_out ./timing_svd_act_lora_full.no_eval.json


'''

class ActivationSpaceLoRAWrapper(nn.Module):
    """LoRA adapter applied in the low-rank activation space (output of V-proj)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, freeze_base: bool = True):
        super().__init__()
        self.base = base
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        self.rank = max(rank, 1)
        self.scaling = alpha / float(self.rank)
        self.lora_down = nn.Linear(base.out_features, self.rank, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False, device=base.weight.device, dtype=base.weight.dtype)
        nn.init.normal_(self.lora_down.weight, mean=0.0, std=0.02)
        # Initialize lora_up small Gaussian to avoid zero-grad cold start
        nn.init.normal_(self.lora_up.weight, mean=0.0, std=0.02)

    def forward(self, x):
        z = self.base(x)
        if self.lora_down.weight.dtype != z.dtype or self.lora_down.weight.device != z.device:
            self.lora_down.to(device=z.device, dtype=z.dtype)
            self.lora_up.to(device=z.device, dtype=z.dtype)
        delta = self.lora_up(self.lora_down(z)) * self.scaling
        return z + delta


def _ensure_tokenizer(tokenizer_obj, model_id: str, hf_token: Optional[str] = None):
    """Return a callable HF tokenizer. Reload if a placeholder/bool slipped through."""
    try:
        if tokenizer_obj is not None and not isinstance(tokenizer_obj, bool) and callable(tokenizer_obj):
            return tokenizer_obj
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        model_hint = os.getenv("SVDLLM_TOKENIZER_MODEL") or model_id
        tok = AutoTokenizer.from_pretrained(
            model_hint, trust_remote_code=True, use_fast=True, token=hf_token
        )
        if tok is not None and not isinstance(tok, bool) and callable(tok):
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token
            return tok
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        model_hint = os.getenv("SVDLLM_TOKENIZER_MODEL") or model_id
        tok = AutoTokenizer.from_pretrained(
            model_hint, trust_remote_code=True, use_fast=False, token=hf_token
        )
        if tok is not None and not isinstance(tok, bool) and callable(tok):
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token
            return tok
    except Exception:
        pass
    try:
        from transformers import LlamaTokenizerFast, LlamaTokenizer
        model_hint = os.getenv("SVDLLM_TOKENIZER_MODEL") or model_id
        for cls in (LlamaTokenizerFast, LlamaTokenizer):
            try:
                tok = cls.from_pretrained(model_hint, token=hf_token)
                if tok is not None and not isinstance(tok, bool) and callable(tok):
                    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                        tok.pad_token = tok.eos_token
                    return tok
            except Exception:
                continue
    except Exception:
        pass
    raise TypeError(
        "Tokenizer object is not callable and could not be reconstructed; "
        "check your HF cache or set SVDLLM_TOKENIZER_MODEL to a valid local tokenizer."
    )


def _freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def attach_activation_lora_llama(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    freeze_base: bool = True,
) -> List[nn.Parameter]:
    """
    Wrap V-proj modules with activation-space LoRA adapters.
    Returns the trainable LoRA parameters.
    """
    trainable: List[nn.Parameter] = []
    for mod in model.modules():
        if isinstance(mod, (SVD_LlamaAttention, SVD_LlamaMLP)):
            # Skip if no compression was applied
            if getattr(mod, "ratio", 1.0) == 1.0:
                continue
            if isinstance(mod, SVD_LlamaAttention):
                targets = {
                    "q_v_proj": getattr(mod, "q_v_proj", None),
                    "k_v_proj": getattr(mod, "k_v_proj", None),
                    "v_v_proj": getattr(mod, "v_v_proj", None),
                    "o_v_proj": getattr(mod, "o_v_proj", None),
                }
            else:
                targets = {
                    "gate_v_proj": getattr(mod, "gate_v_proj", None),
                    "down_v_proj": getattr(mod, "down_v_proj", None),
                    "up_v_proj": getattr(mod, "up_v_proj", None),
                }
            for name, base in targets.items():
                if base is None or not isinstance(base, nn.Linear):
                    continue
                # Sync metadata with actual weight shapes to avoid mismatched LoRA dims
                base.out_features = base.weight.shape[0]
                base.in_features = base.weight.shape[1]
                wrapper = ActivationSpaceLoRAWrapper(base, rank=rank, alpha=alpha, freeze_base=freeze_base)
                setattr(mod, name, wrapper)
                trainable.extend(wrapper.lora_down.parameters())
                trainable.extend(wrapper.lora_up.parameters())
    return trainable


def train_act_lora(
    model: nn.Module,
    dataloader,
    params: List[nn.Parameter],
    device: str,
    epochs: int = 1,
    train_steps: Optional[int] = None,
    lr: float = 5e-4,
    log_every: int = 10,
):
    if not params:
        print("[Train] No LoRA parameters to optimize; skipping.")
        return
    # Clear any NaNs in params before training
    with torch.no_grad():
        for p in params:
            if torch.isnan(p).any():
                mask = torch.isnan(p)
                p[mask] = 0
                print(f"[Debug] Cleared NaNs in param with shape {p.shape}")
    prev_cache = getattr(model.config, "use_cache", False)
    try:
        model.config.use_cache = False
    except Exception:
        pass
    model.train()
    opt = torch.optim.Adam(params, lr=lr)
    amp_dtype = torch.bfloat16 if (str(device).startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
    step = 0
    def _report_stats(tag: str, loss_val):
        with torch.no_grad():
            grad_norm = 0.0
            param_norm = 0.0
            for p in params:
                param_norm += p.norm().item() ** 2
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            param_norm = param_norm ** 0.5
            print(f"[Debug] {tag} loss={loss_val} param_norm={param_norm:.4f} grad_norm={grad_norm:.4f}")

    for ep in range(epochs):
        running = 0.0
        for batch in dataloader:
            if train_steps is not None and step >= int(train_steps):
                break
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                inp, tar, loss_w = batch
                loss_w = float(loss_w)
            else:
                inp, tar = batch
                loss_w = 1.0
            inp = inp.to(device)
            tar = tar.to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=str(device).startswith("cuda")):
                out = model(input_ids=inp, labels=tar)
                loss = out.loss * loss_w
            # Skip nan/inf batches to avoid poisoning training
            if not torch.isfinite(loss):
                print(f"[Train] Skip batch with non-finite loss: {loss.item()}")
                _report_stats("nonfinite", loss.item())
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            running += float(loss.item())
            step += 1
            if (step % log_every) == 0:
                _report_stats(f"epoch {ep+1} step {step}", running / log_every)
                print(f"[Train] epoch {ep+1}/{epochs} step {step}: loss={running/log_every:.6f}")
                running = 0.0
        if train_steps is not None and step >= int(train_steps):
            break
    try:
        model.config.use_cache = prev_cache
    except Exception:
        pass
    model.eval()


def train_act_lora_full_seq(
    model: nn.Module,
    dataloader,
    params: List[nn.Parameter],
    device: str,
    epochs: int = 1,
    train_steps: Optional[int] = None,
    lr: float = 5e-4,
    log_every: int = 10,
):
    """
    Train LoRA on full causal LM loss (all positions) instead of only the last token.
    Labels are set to input_ids (HF shifts internally).
    """
    if not params:
        print("[Train] No LoRA parameters to optimize; skipping.")
        return
    prev_cache = getattr(model.config, "use_cache", False)
    try:
        model.config.use_cache = False
    except Exception:
        pass
    model.train()
    opt = torch.optim.Adam(params, lr=lr)
    amp_dtype = torch.bfloat16 if (str(device).startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
    step = 0
    def _report_stats(tag: str, loss_val):
        with torch.no_grad():
            grad_norm = 0.0
            param_norm = 0.0
            for p in params:
                param_norm += p.norm().item() ** 2
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            param_norm = param_norm ** 0.5
            print(f"[Debug] {tag} loss={loss_val} param_norm={param_norm:.4f} grad_norm={grad_norm:.4f}")

    for ep in range(epochs):
        running = 0.0
        for batch in dataloader:
            if train_steps is not None and step >= int(train_steps):
                break
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                inp = batch[0]
                loss_w = float(batch[2]) if (len(batch) == 3) else 1.0
            else:
                inp = batch
                loss_w = 1.0
            inp = inp.to(device)
            labels = inp
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=str(device).startswith("cuda")):
                out = model(input_ids=inp, labels=labels)
                loss = out.loss * loss_w
            if not torch.isfinite(loss):
                print(f"[Train] Skip batch with non-finite loss: {loss.item()}")
                _report_stats("nonfinite", loss.item())
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            running += float(loss.item())
            step += 1
            if (step % log_every) == 0:
                _report_stats(f"epoch {ep+1} step {step}", running / log_every)
                print(f"[Train] (full) epoch {ep+1}/{epochs} step {step}: loss={running/log_every:.6f}")
                running = 0.0
        if train_steps is not None and step >= int(train_steps):
            break
    try:
        model.config.use_cache = prev_cache
    except Exception:
        pass
    model.eval()


def run_activation_lora(
    model_id: str = "meta-llama/Llama-2-7b-hf",
    dataset: str = "wikitext2",
    keep_ratio: float = 0.8,
    whitening_nsamples: int = 256,
    whitening_lm_datasets: Optional[str] = "wikitext2,ptb,c4",
    whitening_factorization: str = "cholesky",
    attn_keep_ratio: Optional[float] = None,
    mlp_keep_ratio: Optional[float] = None,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_nsamples: Optional[int] = None,
    seqlen: int = 1024,
    device: str = "cuda",
    seed: int = 42,
    epochs: int = 1,
    train_steps: Optional[int] = None,
    lr: float = 5e-4,
    log_every: int = 10,
    train_batch_size: int = 1,
    eval_datasets: Optional[str] = None,
    eval_max_batches: Optional[int] = None,
    full_seq_loss: bool = False,
    save_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    whitening_device: Optional[str] = None,
    whitened_cache: Optional[str] = None,
    model_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    offload_folder: Optional[str] = None,
    trust_whitened_cache: bool = False,
    max_gpu_mem: Optional[str] = None,
    max_cpu_mem: Optional[str] = None,
    # SFT-style data options (official Alpaca format)
    sft_data_path: Optional[str] = None,
    sft_cutoff_len: int = 256,
    sft_add_eos_token: bool = False,
    sft_train_on_inputs: bool = False,
    sft_seed: Optional[int] = None,
    mix_calib_buckets: bool = False,
    # Mixing options: interleave LM with SFT
    mix_lm_with_sft: bool = False,
    mix_ratio: float = 0.5,
    lm_dataset: Optional[str] = None,
    lm_nsamples: Optional[int] = None,
    lm_loss_weight: float = 1.0,
    sft_loss_weight: float = 1.0,
    # Multi-bucket mixture (LM / Instruction / Math)
    mix_buckets: bool = False,
    bucket_props: str = "LM:0.4,INST:0.4,MATH:0.2",
    bucket_lm_datasets: Optional[str] = None,
    bucket_inst_datasets: Optional[str] = None,
    bucket_math_datasets: Optional[str] = None,
    bucket_total_batches: Optional[int] = None,
    bucket_loss_weights: str = "LM:1.0,INST:1.0,MATH:1.0",
    # C4 streaming controls
    c4_stream_train_docs: int = 4000,
    c4_stream_val_docs: int = 2000,
    eval_c4_stream: bool = False,
    dump_bucket_debug: bool = False,
    skip_eval: bool = False,
    timing_out: Optional[str] = None,
    stop_after_compress: bool = False,
    force_recompress: bool = False,
):
    """
    Pipeline:
      1) Whitening (Cholesky or SVD sqrt) -> SVD compression (keeps `keep_ratio` params).
      2) Attach LoRA adapters in the low-rank activation space (V branch).
      3) Train only LoRA parameters.
    """
    t_run0 = time.perf_counter()
    timing: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_id": model_id,
        "dataset": dataset,
        "keep_ratio": float(keep_ratio),
        "seqlen": int(seqlen),
        "whitening_nsamples": int(whitening_nsamples),
        "lora_nsamples": None if lora_nsamples is None else int(lora_nsamples),
        "epochs": int(epochs),
        "train_steps": None if train_steps is None else int(train_steps),
        "device": str(device),
        "whitening_device": str(whitening_device) if whitening_device is not None else None,
        "whitened_cache": whitened_cache,
        "cache_loaded": False,
        "skip_eval": bool(skip_eval),
        "force_recompress": bool(force_recompress),
        "stages": [],
        "eval_stages": [],
    }

    def _stage(name: str, sec: float, **extra: Any) -> None:
        rec = {"name": name, "sec": float(sec)}
        rec.update(extra)
        timing["stages"].append(rec)

    def _eval_stage(name: str, sec: float, **extra: Any) -> None:
        rec = {"name": name, "sec": float(sec)}
        rec.update(extra)
        timing["eval_stages"].append(rec)

    def _write_timing(path: str) -> None:
        try:
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(timing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Time] Failed to write timing json to {path}: {e}")

    dev = device
    model = None
    tokenizer = None
    # Seed everything for reproducibility unless caller overrides
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    if sft_seed is None:
        sft_seed = seed
    profile_dev = whitening_device if whitening_device is not None else dev
    # Default cache path if none provided
    if whitened_cache is None:
        model_stub = model_id.rsplit("/", 1)[-1]
        model_stub = model_stub.replace("-", "_").replace("llama_2", "llama2")
        wf = str(whitening_factorization or "cholesky").strip().lower()
        suffix_parts = []
        if wf and wf != "cholesky":
            suffix_parts.append(wf)
        if attn_keep_ratio is not None or mlp_keep_ratio is not None:
            a = keep_ratio if attn_keep_ratio is None else float(attn_keep_ratio)
            m = keep_ratio if mlp_keep_ratio is None else float(mlp_keep_ratio)
            suffix_parts.append(f"attn{a}")
            suffix_parts.append(f"mlp{m}")
        suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)
        whitened_cache = os.path.join(
            "checkpoints", f"{model_stub}_whitening_only_{keep_ratio}{suffix}.pt"
        )

    cache_loaded = False
    # Load cached whitened checkpoint if provided
    if (not force_recompress) and whitened_cache is not None and os.path.exists(whitened_cache):
        print(f"[Cache] Loading whitened checkpoint from {whitened_cache}")
        # Explicitly allow full object load (we control the saved class) to avoid PyTorch 2.6 weights_only default.
        ckpt = torch.load(whitened_cache, map_location="cpu", weights_only=False)
        model = ckpt["model"]
        tokenizer = ckpt["tokenizer"]
        cache_meta = ckpt.get("cache_meta", {}) if isinstance(ckpt, dict) else {}
        if isinstance(tokenizer, bool):
            print("[Cache] Cached tokenizer is a boolean placeholder; will reload from HF.")
            tokenizer = None
        model = model.eval()
        cache_loaded = True
        timing["cache_loaded"] = True
        if not trust_whitened_cache:
            # Validate cache metadata when present; fall back to "safe ignore" when requesting non-default modes.
            req_wf = str(whitening_factorization or "cholesky").strip().lower()
            # We treat keep_ratio as the SVD-LLM "compression ratio" rho = k(m+n)/(mn).
            # When attn/mlp overrides are omitted, they default to keep_ratio.
            req_attn = float(keep_ratio) if attn_keep_ratio is None else float(attn_keep_ratio)
            req_mlp = float(keep_ratio) if mlp_keep_ratio is None else float(mlp_keep_ratio)
            meta_ok = True
            if isinstance(cache_meta, dict) and cache_meta:
                try:
                    saved_ratio = float(cache_meta.get("keep_ratio", keep_ratio))
                    saved_wf = str(cache_meta.get("whitening_factorization", "cholesky")).strip().lower()
                    saved_attn = cache_meta.get("attn_keep_ratio", None)
                    saved_mlp = cache_meta.get("mlp_keep_ratio", None)
                    saved_attn = None if saved_attn is None else float(saved_attn)
                    saved_mlp = None if saved_mlp is None else float(saved_mlp)
                    if abs(saved_ratio - float(keep_ratio)) > 1e-8:
                        meta_ok = False
                    if saved_wf != req_wf:
                        meta_ok = False
                    if (saved_attn is None) or (abs(saved_attn - req_attn) > 1e-8):
                        meta_ok = False
                    if (saved_mlp is None) or (abs(saved_mlp - req_mlp) > 1e-8):
                        meta_ok = False
                except Exception:
                    meta_ok = False
            else:
                # When cache_meta is missing, conservatively ignore cache for non-default whitening factorization
                # or when explicit attn/mlp overrides are requested.
                if (req_wf != "cholesky") or (attn_keep_ratio is not None) or (mlp_keep_ratio is not None):
                    meta_ok = False

            def _cache_shapes_match(m):
                try:
                    attn = m.model.layers[0].self_attn
                    mlp = m.model.layers[0].mlp
                    attn_ratio = getattr(attn, "ratio", keep_ratio)
                    mlp_ratio = getattr(mlp, "ratio", keep_ratio)
                    compat_ranks = (os.getenv("SVDLLM_COMPAT_ALL", "0") != "0") or (os.getenv("SVDLLM_COMPAT_RANKS", "0") != "0")
                    if compat_ranks:
                        exp_attn = max(1, int(attn.hidden_size * attn_ratio / 2.0))
                        exp_mlp = max(1, int(mlp.intermediate_size * mlp.hidden_size * mlp_ratio / (mlp.intermediate_size + mlp.hidden_size)))
                    else:
                        exp_attn = max(1, int(attn.hidden_size * attn_ratio))
                        exp_mlp = max(1, int(min(mlp.intermediate_size, mlp.hidden_size) * mlp_ratio))
                    if attn.q_v_proj.out_features != exp_attn:
                        return False
                    if mlp.up_v_proj.out_features != exp_mlp:
                        return False
                    return True
                except Exception:
                    return True
            if (not meta_ok) or (not _cache_shapes_match(model)):
                print(f"[Cache] Cached model settings mismatch current run; recomputing whitening instead of using {whitened_cache}.")
                model = None
                tokenizer = None
                cache_loaded = False
                timing["cache_loaded"] = False
    if model is None:
        t0 = time.perf_counter()
        model, tokenizer = get_model_from_huggingface(model_id, hf_token=hf_token)
        _stage("load_model_hf", time.perf_counter() - t0)
    tokenizer = _ensure_tokenizer(tokenizer, model_id, hf_token=hf_token)
    if model_dtype is not None:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        tgt_dtype = dtype_map.get(model_dtype.lower(), None)
        if tgt_dtype is not None:
            model = model.to(dtype=tgt_dtype)
    # Keep model on profiling device for whitening stats to save GPU memory if needed
    model = model.eval().to(profile_dev)
    try:
        model.seqlen = seqlen
    except Exception:
        pass
    eval_list = [d.strip() for d in (eval_datasets.split(",") if eval_datasets else [dataset]) if d.strip()]

    if cache_loaded:
        profiling_mat = None
    else:
        # Whitening in this script is LM-only (wikitext2/ptb/c4 by default).
        lm_whiten = whitening_lm_datasets
        if lm_whiten is None or not str(lm_whiten).strip():
            lm_whiten = bucket_lm_datasets or "wikitext2,ptb,c4"
        t0 = time.perf_counter()
        cali_white_data = get_mixed_calib_train_data(
            tokenizer=tokenizer,
            nsamples=whitening_nsamples,
            seqlen=seqlen,
            seed=seed,
            bucket_props="LM:1.0,INST:0.0,MATH:0.0",
            bucket_lm_datasets=lm_whiten,
            bucket_inst_datasets=None,
            bucket_math_datasets=None,
            dump_bucket_debug=dump_bucket_debug,
        )
        _stage("build_whitening_data", time.perf_counter() - t0)
        if profile_dev != dev:
            print(f"[Compress] Profiling on {profile_dev} to save memory; will move model to {dev} for training.")
        wf = str(whitening_factorization or "cholesky").strip().lower()
        use_svd_sqrt = (wf == "svd")
        # SVD-LLM compression ratio (paper): rho = k(m+n)/(mn).
        # Use per-family overrides when provided; otherwise default both to keep_ratio.
        attn_ratio = float(keep_ratio) if attn_keep_ratio is None else float(attn_keep_ratio)
        mlp_ratio = float(keep_ratio) if mlp_keep_ratio is None else float(mlp_keep_ratio)
        if (attn_keep_ratio is not None) or (mlp_keep_ratio is not None):
            print(f"[Compress] Heterogeneous rank enabled: attn_keep_ratio={attn_ratio}, mlp_keep_ratio={mlp_ratio}")
        else:
            print(f"[Compress] Using SVD-LLM ratio->params mapping: keep_ratio(rho)={float(keep_ratio)}")

        t0 = time.perf_counter()
        if use_svd_sqrt:
            from SVDLLM_v2 import profle_svdllm as profle_svdllm_v2
            profiling_mat = profle_svdllm_v2(model_id, model, cali_white_data, profile_dev)
        else:
            profiling_mat = profle_svdllm(model_id, model, cali_white_data, profile_dev)
        _stage("profile_svdllm", time.perf_counter() - t0, whitening_factorization=wf)

        # Move model to training device before applying SVD/finetune
        if profile_dev != dev:
            torch.cuda.empty_cache()
            model = model.to(dev)
        # Always compress with the SVD-LLM ratio definition so --keep_ratio maps to parameter/memory ratio.
        from SVDLLM_v2 import whitening_hetero
        t0 = time.perf_counter()
        whitening_hetero(
            model_id,
            model,
            profiling_mat,
            keep_ratio,
            dev,
            attn_ratio=attn_ratio,
            mlp_ratio=mlp_ratio,
        )
        _stage("apply_whitening_hetero", time.perf_counter() - t0, attn_keep_ratio=float(attn_ratio), mlp_keep_ratio=float(mlp_ratio))
        model = model.to(dev).eval()
        if whitened_cache is not None:
            cache_dir = os.path.dirname(whitened_cache)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            t0 = time.perf_counter()
            torch.save(
                {
                    "model": model.cpu(),
                    "tokenizer": tokenizer,
                    "cache_meta": {
                        "keep_ratio": float(keep_ratio),
                        "whitening_factorization": str(whitening_factorization or "cholesky").strip().lower(),
                        "attn_keep_ratio": float(attn_ratio),
                        "mlp_keep_ratio": float(mlp_ratio),
                    },
                },
                whitened_cache,
            )
            _stage("save_whitened_cache", time.perf_counter() - t0, path=whitened_cache)
            model = model.to(dev)

    # Compression-only timing summary (whitening+SVD stages only)
    compress_only_sec = 0.0
    for st in timing["stages"]:
        if st["name"] in ("build_whitening_data", "profile_svdllm", "apply_whitening_hetero", "save_whitened_cache"):
            compress_only_sec += float(st.get("sec", 0.0))
    timing["compress_only_sec"] = float(compress_only_sec)
    print(f"[Time] compress_only_sec={compress_only_sec:.2f}s (excludes eval + LoRA train; cache_loaded={bool(timing['cache_loaded'])})")
    if timing.get("cache_loaded") and (not force_recompress):
        print("[Time] Note: cached whitened checkpoint was loaded; compress_only_sec does NOT reflect a fresh compression run. Use --force_recompress to measure true compression time.")

    if stop_after_compress:
        timing["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        timing["total_sec"] = float(time.perf_counter() - t_run0)
        if timing_out:
            _write_timing(timing_out)
            print(f"[Time] Wrote timing json to: {timing_out}")
        print(f"[Time] total_sec={timing['total_sec']:.2f}s (stop_after_compress)")
        return
    # Optional hybrid device placement before eval/train
    if device_map is not None and device_map.lower() != "none":
        if offload_folder:
            os.makedirs(offload_folder, exist_ok=True)
        if isinstance(device_map, str):
            # Infer a concrete map for strings like "auto"/"balanced"
            max_mem = {}
            if str(dev).startswith("cuda"):
                try:
                    gpu_idx = torch.cuda.current_device()
                except Exception:
                    gpu_idx = 0
                max_mem[gpu_idx] = max_gpu_mem if max_gpu_mem is not None else "30GiB"  # tighten to force more offload
            max_mem["cpu"] = max_cpu_mem if max_cpu_mem is not None else "256GiB"
            map_arg = infer_auto_device_map(
                model,
                max_memory=max_mem,
                no_split_module_classes=["LlamaDecoderLayer", "SVD_LlamaAttention", "SVD_LlamaMLP"],
            )
        else:
            map_arg = device_map
        model = dispatch_model(model, device_map=map_arg, offload_dir=offload_folder, offload_buffers=True)
    else:
        model = model.to(dev)

    if not skip_eval:
        # Ensure tokenizer is still valid before eval (some cached checkpoints store placeholders)
        tokenizer = _ensure_tokenizer(tokenizer, model_id, hf_token=hf_token)
        t0 = time.perf_counter()
        try:
            ppl_eval(
                model,
                tokenizer,
                datasets=eval_list,
                model_seq_len=seqlen,
                batch_size=4,
                device=dev,
                label="PPL after whitening-only",
                max_batches=eval_max_batches,
            )
        except Exception as e:
            print(f"[Eval] Skipped PPL (post-whitening) due to: {e}")
        _eval_stage("ppl_post_whitening", time.perf_counter() - t0)

    print(f"[LoRA] Attaching activation-space adapters (rank={lora_rank}, alpha={lora_alpha})...")
    t0 = time.perf_counter()
    _freeze_all_params(model)
    lora_params = attach_activation_lora_llama(model, rank=lora_rank, alpha=lora_alpha, freeze_base=True)
    _stage("attach_activation_lora", time.perf_counter() - t0, lora_rank=int(lora_rank), lora_alpha=float(lora_alpha))
    model = model.to(dev)
    if not lora_params:
        print("[LoRA] No adapters were attached (did the model get compressed?).")
    lora_num = lora_nsamples if lora_nsamples is not None else whitening_nsamples
    # Ensure tokenizer before building loaders
    tokenizer = _ensure_tokenizer(tokenizer, model_id, hf_token=hf_token)
    t0 = time.perf_counter()
    update_loader, _ = get_loaders(
        dataset, nsamples=lora_num, seed=seed, seqlen=seqlen, tokenizer=tokenizer
    )
    _stage("build_lora_loader_base", time.perf_counter() - t0, nsamples=int(lora_num), dataset=str(dataset))

    # Re-ensure tokenizer before any data mixing paths (some checkpoints store placeholders)
    tokenizer = _ensure_tokenizer(tokenizer, model_id, hf_token=hf_token)
    # Optionally replace update_loader with SFT-style instruction data like the official repo
    if sft_data_path:
        try:
            from datasets import load_dataset
        except Exception as e:
            raise RuntimeError(f"datasets library is required for --sft_data_path but could not be imported: {e}")

        def _tokenize_prompt(prompter, dp):
            full_prompt = prompter.generate_prompt(dp.get("instruction", ""), dp.get("input", None), dp.get("output", None))
            user_prompt = None if sft_train_on_inputs else prompter.generate_prompt(dp.get("instruction", ""), dp.get("input", None))
            toks = tokenizer(full_prompt, truncation=True, max_length=sft_cutoff_len, padding=False, return_tensors=None)
            if sft_add_eos_token and toks["input_ids"] and (toks["input_ids"][-1] != tokenizer.eos_token_id) and (len(toks["input_ids"]) < sft_cutoff_len):
                toks["input_ids"].append(tokenizer.eos_token_id)
                toks["attention_mask"].append(1)
            labels = toks["input_ids"].copy()
            if user_prompt is not None:
                up = tokenizer(user_prompt, truncation=True, max_length=sft_cutoff_len, padding=False, return_tensors=None)
                user_len = len(up["input_ids"]) - (1 if sft_add_eos_token else 0)
                labels = ([-100] * user_len) + labels[user_len:]
            ids = toks["input_ids"][:sft_cutoff_len]
            labs = labels[:sft_cutoff_len]
            if len(ids) < sft_cutoff_len:
                pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                pad_n = sft_cutoff_len - len(ids)
                ids = ids + [pad_id] * pad_n
                labs = labs + ([-100] * pad_n)
            return torch.tensor(ids, dtype=torch.long).unsqueeze(0), torch.tensor(labs, dtype=torch.long).unsqueeze(0)

        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        prompter = Prompter("alpaca")
        ds = load_dataset(sft_data_path)
        train_split = ds["train"].shuffle(seed=sft_seed)
        num_take = max(1, lora_num)
        pairs = []
        for i, dp in enumerate(train_split):
            if i >= num_take:
                break
            inp, lab = _tokenize_prompt(prompter, dp)
            pairs.append((inp, lab))
        class _ListLoader:
            def __iter__(self_inner):
                return iter(pairs)
        update_loader = _ListLoader()
        # Use cutoff length during LoRA updates to match constructed labels
        seqlen = sft_cutoff_len

        if mix_lm_with_sft and not mix_buckets:
            # Build LM loader to mix with SFT
            lm_name = lm_dataset if lm_dataset is not None else dataset
            lm_num = lm_nsamples if lm_nsamples is not None else lora_num
            lm_loader, _ = get_loaders(lm_name, nsamples=lm_num, seed=seed, seqlen=seqlen, tokenizer=tokenizer)

            def _batchify(loader_like, bs: int):
                buf_inp, buf_tar = [], []
                for inp, tar in loader_like:
                    buf_inp.append(inp)
                    buf_tar.append(tar)
                    if len(buf_inp) == bs:
                        yield torch.cat(buf_inp, dim=0), torch.cat(buf_tar, dim=0)
                        buf_inp, buf_tar = [], []
                if buf_inp:
                    yield torch.cat(buf_inp, dim=0), torch.cat(buf_tar, dim=0)

            sft_batches = list(_batchify(update_loader, max(1, train_batch_size)))
            lm_batches = list(_batchify(lm_loader, max(1, train_batch_size)))
            # Align LM loss with full-seq PPL: labels should be input_ids (HF shifts internally).
            lm_batches = [(inp, inp.clone()) for (inp, _tar) in lm_batches]

            # Interleave according to mix_ratio probability for SFT
            import random
            random.seed(sft_seed)
            i = j = 0
            mixed = []
            while i < len(sft_batches) or j < len(lm_batches):
                take_sft = (random.random() < max(0.0, min(1.0, mix_ratio)))
                if take_sft and i < len(sft_batches):
                    mixed.append((sft_batches[i][0], sft_batches[i][1], float(sft_loss_weight)))
                    i += 1
                elif j < len(lm_batches):
                    mixed.append((lm_batches[j][0], lm_batches[j][1], float(lm_loss_weight)))
                    j += 1
                elif i < len(sft_batches):
                    mixed.append((sft_batches[i][0], sft_batches[i][1], float(sft_loss_weight)))
                    i += 1
            update_loader = mixed
    # Helper: batchify (inp, tar) pairs
    def _batch_loader(loader, bs: int):
        buf_inp, buf_tar = [], []
        for inp, tar in loader:
            buf_inp.append(inp)
            buf_tar.append(tar)
            if len(buf_inp) == bs:
                yield torch.cat(buf_inp, dim=0), torch.cat(buf_tar, dim=0)
                buf_inp, buf_tar = [], []
        if buf_inp:
            yield torch.cat(buf_inp, dim=0), torch.cat(buf_tar, dim=0)

    # Advanced multi-bucket mixing (LM / Instruction / Math)
    if mix_buckets:
        # Ensure tokenizer is valid before any direct tokenization in bucket mixing
        tokenizer = _ensure_tokenizer(tokenizer, model_id, hf_token=hf_token)
        # Interpret lora_nsamples as a GLOBAL budget split across buckets by bucket_props
        total_budget = int(lora_num)
        # Parse proportions string like 'LM:0.4,INST:0.4,MATH:0.2'
        def _parse_props(s: str) -> dict:
            out = {"LM": 0.4, "INST": 0.4, "MATH": 0.2}
            try:
                for seg in (s or "").split(','):
                    if not seg.strip():
                        continue
                    k, v = seg.split(':')
                    out[k.strip().upper()] = float(v)
            except Exception:
                pass
            # Normalize if necessary
            sm = sum(out.values())
            if sm > 0:
                for k in out:
                    out[k] = out[k] / sm
            return out
        props_for_budget = _parse_props(bucket_props)
        bucket_budget = {k: int(round(total_budget * props_for_budget.get(k, 0.0))) for k in ("LM","INST","MATH")}
        # Fix rounding to sum exactly to total_budget
        diff = total_budget - sum(bucket_budget.values())
        if diff != 0:
            bucket_budget["LM"] = max(0, bucket_budget.get("LM", 0) + diff)
        # 1) Build LM batches from specified datasets
        lm_names = [n.strip().lower() for n in (bucket_lm_datasets.split(',') if bucket_lm_datasets else [dataset]) if n.strip()]
        lm_batches_all: List = []
        lm_counts = {}
        # Distribute LM bucket budget evenly across LM datasets
        lm_ds_budget = {}
        if len(lm_names) > 0:
            per = bucket_budget["LM"] // len(lm_names)
            rem = bucket_budget["LM"] % len(lm_names)
            for idx, name in enumerate(lm_names):
                lm_ds_budget[name] = per + (1 if idx < rem else 0)
        for lm_name in lm_names:
            if lm_name in ("c4", "c4_stream", "allenai/c4"):
                # Stream small slice to avoid huge downloads; pre-batch to (inp, tar) pairs
                try:
                    from datasets import load_dataset
                except Exception as e:
                    raise RuntimeError(f"datasets library required to stream C4: {e}")
                try:
                    _ok = callable(tokenizer)
                except Exception:
                    _ok = False
                if not _ok or isinstance(tokenizer, bool):
                    tokenizer = _ensure_tokenizer(None, model_id, hf_token=hf_token)
                import itertools, random
                random.seed(sft_seed)
                stream = load_dataset("allenai/c4", "en", split="train", streaming=True)
                seqs: List[torch.Tensor] = []
                budget = int(lm_ds_budget.get(lm_name, 0))
                if budget <= 0:
                    continue
                # Stream documents and cut windows until reaching dataset budget
                for ex in itertools.islice(iter(stream), int(c4_stream_train_docs)):
                    t = ex.get('text') or ex.get('content') or ''
                    if not t:
                        continue
                    enc = tokenizer(t, return_tensors='pt')
                    L = enc.input_ids.shape[1]
                    if L < seqlen:
                        continue
                    i = random.randint(0, L - seqlen - 1)
                    j = i + seqlen
                    seqs.append(enc.input_ids[:, i:j])
                    if len(seqs) >= budget:
                        break
                # batchify sequences
                added = 0
                for k in range(0, len(seqs), max(1, train_batch_size)):
                    chunk = seqs[k:k+max(1, train_batch_size)]
                    if not chunk:
                        continue
                    inp = torch.cat(chunk, dim=0)
                    tar = inp.clone()
                    lm_batches_all.append((inp, tar))
                    added += 1
                key = 'c4_stream'
                lm_counts[key] = lm_counts.get(key, 0) + added
            else:
                budget = int(lm_ds_budget.get(lm_name, 0))
                if budget <= 0:
                    continue
                lm_loader, _ = get_loaders(lm_name, nsamples=budget, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
                batches = list(_batch_loader(lm_loader, max(1, train_batch_size)))
                # For LM, train full-seq causal LM (labels=input_ids) instead of last-token-only.
                batches = [(inp, inp.clone()) for (inp, _tar) in batches]
                lm_batches_all.extend(batches)
                lm_counts[lm_name] = lm_counts.get(lm_name, 0) + len(batches)

        # 2) Build Instruction batches from one or more Alpaca-style datasets
        inst_names: List[str] = []
        if bucket_inst_datasets:
            inst_names.extend([n.strip() for n in bucket_inst_datasets.split(',') if n.strip()])
        if sft_data_path:
            # Ensure explicit SFT dataset participates in INST bucket even when bucket_inst_datasets is set
            if all(n.strip().lower() != sft_data_path.strip().lower() for n in inst_names):
                inst_names.append(sft_data_path.strip())
        inst_batches_all: List = []
        inst_counts = {}
        # Distribute INST bucket budget evenly across INST datasets
        inst_ds_budget = {}
        if len(inst_names) > 0:
            per = bucket_budget["INST"] // len(inst_names)
            rem = bucket_budget["INST"] % len(inst_names)
            for idx, name in enumerate(inst_names):
                inst_ds_budget[name] = per + (1 if idx < rem else 0)
        if inst_names:
            try:
                from datasets import load_dataset
            except Exception as e:
                raise RuntimeError(f"datasets library required for instruction buckets: {e}")

            def _tokenize_prompt(prompter, dp):
                full_prompt = prompter.generate_prompt(dp.get("instruction", ""), dp.get("input", None), dp.get("output", None))
                user_prompt = None if sft_train_on_inputs else prompter.generate_prompt(dp.get("instruction", ""), dp.get("input", None))
                toks = tokenizer(full_prompt, truncation=True, max_length=sft_cutoff_len, padding=False, return_tensors=None)
                if sft_add_eos_token and toks["input_ids"] and (toks["input_ids"][-1] != tokenizer.eos_token_id) and (len(toks["input_ids"]) < sft_cutoff_len):
                    toks["input_ids"].append(tokenizer.eos_token_id)
                    toks["attention_mask"].append(1)
                labels = toks["input_ids"].copy()
                if user_prompt is not None:
                    up = tokenizer(user_prompt, truncation=True, max_length=sft_cutoff_len, padding=False, return_tensors=None)
                    user_len = len(up["input_ids"]) - (1 if sft_add_eos_token else 0)
                    labels = ([-100] * user_len) + labels[user_len:]
                ids = toks["input_ids"][:sft_cutoff_len]
                labs = labels[:sft_cutoff_len]
                if len(ids) < sft_cutoff_len:
                    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    pad_n = sft_cutoff_len - len(ids)
                    ids = ids + [pad_id] * pad_n
                    labs = labs + ([-100] * pad_n)
                return torch.tensor(ids, dtype=torch.long).unsqueeze(0), torch.tensor(labs, dtype=torch.long).unsqueeze(0)

            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

            # Adapters to unify several QA datasets to Alpaca-style prompts
            def _inst_from_hellaswag(dp):
                ctx = dp.get("ctx") or dp.get("context", "")
                endings = dp.get("endings") or []
                label = dp.get("label")
                letters = ["A", "B", "C", "D"]
                opts = [f"{letters[i]}) {endings[i]}" for i in range(min(4, len(endings)))]
                return {
                    "instruction": "Choose the most plausible ending.",
                    "input": ctx + ("\nOptions: " + " \n".join(opts) if opts else ""),
                    "output": letters[int(label)] if isinstance(label, int) else str(label),
                }

            def _inst_from_piqa(dp):
                goal = dp.get("goal", "")
                sol1 = dp.get("sol1", "")
                sol2 = dp.get("sol2", "")
                label = dp.get("label")
                return {
                    "instruction": "Pick the more sensible solution.",
                    "input": f"Goal: {goal}\nOptions: A) {sol1} B) {sol2}",
                    "output": "A" if str(label) == "0" else "B",
                }

            def _inst_from_winogrande(dp):
                sent = dp.get("sentence", "")
                o1 = dp.get("option1", "")
                o2 = dp.get("option2", "")
                ans = dp.get("answer", "")
                return {
                    "instruction": "Fill in the blank with the correct option.",
                    "input": f"{sent}\nOptions: 1) {o1} 2) {o2}",
                    "output": "1" if str(ans) in ("1", "A") else "2",
                }

            def _inst_from_ai2_arc(dp):
                stem = dp.get("question", {}).get("stem") if isinstance(dp.get("question"), dict) else dp.get("question", "")
                choices = dp.get("choices", {})
                texts = choices.get("text") if isinstance(choices, dict) else None
                labels = choices.get("label") if isinstance(choices, dict) else None
                if not texts or not labels:
                    return {"instruction": "Answer the question.", "input": stem or "", "output": dp.get("answerKey", "")}
                opts = [f"{labels[i]}) {texts[i]}" for i in range(min(4, len(texts)))]
                return {
                    "instruction": "Select the correct option.",
                    "input": (stem or "") + ("\nOptions: " + " \n".join(opts) if opts else ""),
                    "output": dp.get("answerKey", ""),
                }

            def _inst_from_openbookqa(dp):
                stem = dp.get("question_stem") or dp.get("question", "")
                choices = dp.get("choices") or {}
                texts = (choices.get("text") if isinstance(choices, dict) else None) or []
                labels = (choices.get("label") if isinstance(choices, dict) else None) or []
                opts = [f"{labels[i]}) {texts[i]}" for i in range(min(4, len(texts)))]
                return {
                    "instruction": "Answer the science question.",
                    "input": (stem or "") + ("\nOptions: " + " \n".join(opts) if opts else ""),
                    "output": dp.get("answerKey", ""),
                }

            def _inst_from_cola(dp):
                sent = dp.get("sentence", "") or dp.get("text", "")
                lab = dp.get("label", dp.get("labels", 0))
                try:
                    lab_i = int(lab)
                except Exception:
                    lab_i = 1 if str(lab).strip().lower() in ("1", "true", "yes") else 0
                out = "acceptable" if lab_i == 1 else "unacceptable"
                return {
                    "instruction": "Determine whether the following English sentence is grammatically acceptable.",
                    "input": str(sent),
                    "output": out,
                }

            def _inst_from_sst2(dp):
                sent = dp.get("sentence", "") or dp.get("text", "")
                lab = dp.get("label", dp.get("labels", 0))
                try:
                    lab_i = int(lab)
                except Exception:
                    lab_i = 1 if str(lab).strip().lower() in ("1", "true", "yes", "pos", "positive") else 0
                out = "positive" if lab_i == 1 else "negative"
                return {
                    "instruction": "Classify the sentiment of the sentence as positive or negative.",
                    "input": str(sent),
                    "output": out,
                }

            for name in inst_names:
                lname = name.lower()
                pairs = []
                try:
                    if lname in ("yahma/alpaca-cleaned", "tatsu-lab/alpaca", "alpaca", "alpaca-cleaned"):
                        prompter = Prompter("alpaca")
                        ds = load_dataset(name)
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            try:
                                inp, lab = _tokenize_prompt(prompter, dp)
                                pairs.append((inp, lab))
                            except Exception:
                                continue
                    elif lname == "hellaswag":
                        ds = load_dataset("hellaswag")
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            rec = _inst_from_hellaswag(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    elif lname == "piqa":
                        ds = load_dataset("piqa")
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            rec = _inst_from_piqa(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    elif lname.startswith("winogrande"):
                        # HF configs are named winogrande_{xs,s,m,l,xl}
                        if lname == "winogrande":
                            cfg = "winogrande_xl"
                        elif lname.startswith("winogrande/"):
                            cfg = f"winogrande_{lname.split('/', 1)[1]}"
                        elif lname.startswith("winogrande_"):
                            cfg = lname
                        else:
                            cfg = "winogrande_xl"
                        ds = load_dataset("winogrande", cfg)
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            rec = _inst_from_winogrande(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    elif lname in ("ai2_arc_easy", "arc_easy", "ai2_arc/arc-easy"):
                        ds = load_dataset("ai2_arc", "ARC-Easy")
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            rec = _inst_from_ai2_arc(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    elif lname in ("ai2_arc_challenge", "arc_challenge", "ai2_arc/arc-challenge"):
                        ds = load_dataset("ai2_arc", "ARC-Challenge")
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            rec = _inst_from_ai2_arc(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    elif lname in ("openbookqa", "openbookqa/main"):
                        try:
                            ds = load_dataset("openbookqa", "main")
                        except Exception:
                            ds = load_dataset("openbookqa")
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            rec = _inst_from_openbookqa(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    elif lname in ("cola", "glue/cola", "glue_cola", "glue-cola"):
                        ds = load_dataset("glue", "cola")
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            rec = _inst_from_cola(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    elif lname in ("sst2", "glue/sst2", "glue_sst2", "glue-sst2"):
                        ds = load_dataset("glue", "sst2")
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            rec = _inst_from_sst2(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    else:
                        # Fallback: try to treat as Alpaca-style schema
                        prompter = Prompter("alpaca")
                        ds = load_dataset(name)
                        train_split = ds["train"].shuffle(seed=sft_seed)
                        take = max(1, int(inst_ds_budget.get(name, 0)))
                        for i, dp in enumerate(train_split):
                            if i >= take:
                                break
                            try:
                                inp, lab = _tokenize_prompt(prompter, dp)
                                pairs.append((inp, lab))
                            except Exception:
                                continue
                except Exception as e:
                    print(f"[Mix] Skip instruction dataset {name}: {e}")
                    pairs = []
                if pairs:
                    batches = list(_batch_loader(pairs, max(1, train_batch_size)))
                    inst_batches_all.extend(batches)
                    inst_counts[name] = inst_counts.get(name, 0) + len(batches)

        # 3) Build Math batches (GSM8K, AQuA-RAT) with simple SFT mapping
        math_names = [n.strip().lower() for n in (bucket_math_datasets.split(',') if bucket_math_datasets else []) if n.strip()]
        math_batches_all: List = []
        if math_names:
            try:
                from datasets import load_dataset
            except Exception as e:
                raise RuntimeError(f"datasets library required for math buckets: {e}")

            def _parse_mathqa_options_text(opt_str: str) -> (List[str], Dict[str, int]):
                # Parse strings like "A) ... , B) ..."
                choices, mapping = [], {}
                parts = re.split(r"\s*([A-Ea-e])\s*\)\s*", opt_str)
                for i in range(1, len(parts), 2):
                    lab = parts[i].upper()
                    text = parts[i + 1].strip()
                    mapping[lab] = len(choices)
                    choices.append(text)
                return choices, mapping

            def _mathqa_choices_from_field(opt_field: Any) -> (List[str], Dict[str, int]):
                if isinstance(opt_field, dict):
                    out, mapping = [], {}
                    for lab in ['A', 'B', 'C', 'D', 'E']:
                        if lab in opt_field:
                            mapping[lab] = len(out)
                            out.append(opt_field[lab])
                    return out, mapping
                if isinstance(opt_field, list):
                    out = [str(x) for x in opt_field]
                    mapping = {chr(ord('A') + i): i for i in range(len(out))}
                    return out, mapping
                if isinstance(opt_field, str):
                    return _parse_mathqa_options_text(opt_field)
                return [], {}

            def _fmt_gsm8k(dp):
                q = dp.get("question", "")
                a = dp.get("answer", "")
                return {"instruction": "Solve this math problem.", "input": q, "output": a}

            def _fmt_aqua(dp):
                q = dp.get("question", "")
                opts = dp.get("options", [])
                corr = dp.get("correct", "")
                rationale = dp.get("rationale", "")
                prompt = q
                if isinstance(opts, list) and opts:
                    prompt = q + "\nOptions: " + "; ".join(opts)
                out = rationale if isinstance(rationale, str) and rationale else corr
                return {"instruction": "Choose the correct option.", "input": prompt, "output": out}

            # Distribute MATH bucket budget evenly
            math_ds_budget = {}
            if len(math_names) > 0:
                per = bucket_budget["MATH"] // len(math_names)
                rem = bucket_budget["MATH"] % len(math_names)
                for idx, name in enumerate(math_names):
                    math_ds_budget[name] = per + (1 if idx < rem else 0)

            def _pairs_from_math(name):
                pairs = []
                if name == 'gsm8k':
                    try:
                        ds = load_dataset('gsm8k', 'main')
                        split = ds['train'].shuffle(seed=sft_seed)
                        take = max(1, int(math_ds_budget.get(name, 0)))
                        for i, dp in enumerate(split):
                            if i >= take:
                                break
                            rec = _fmt_gsm8k(dp)
                            # Reuse Alpaca tokenizer path
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    except Exception as e:
                        print(f"[Mix] Skip gsm8k: {e}")
                elif name in ('mathqa', 'math_qa'):
                    try:
                        take = max(1, int(math_ds_budget.get(name, 0)))
                        rows = []
                        if load_mathqa_local is not None:
                            rows = load_mathqa_local(split='train') or []
                        if rows:
                            letters = ["A", "B", "C", "D", "E"]
                            for i, dp in enumerate(rows):
                                if i >= take:
                                    break
                                q = str(dp.get("prompt", "")).replace("\nAnswer:", "").strip()
                                choices = dp.get("choices") or []
                                ans_idx = int(dp.get("answer_idx", 0)) if choices else 0
                                opts = [f"{letters[j]}) {choices[j]}" for j in range(min(len(choices), len(letters)))]
                                rec = {
                                    "instruction": "Choose the correct option.",
                                    "input": (q + ("\nOptions: " + " \n".join(opts) if opts else "")),
                                    "output": letters[ans_idx] if ans_idx < len(letters) else "A",
                                }
                                px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                                pairs.append((px, py))
                        else:
                            ds = load_dataset('math_qa')
                            split = ds['train'].shuffle(seed=sft_seed)
                            for i, dp in enumerate(split):
                                if i >= take:
                                    break
                                q = dp.get("Problem") or dp.get("problem") or dp.get("question", "")
                                opt_field = dp.get("options") or dp.get("Options") or dp.get("choices")
                                choices, mapping = _mathqa_choices_from_field(opt_field if opt_field is not None else "")
                                if not choices:
                                    continue
                                corr = dp.get("correct") or dp.get("label") or dp.get("answer") or "A"
                                corr = str(corr).strip().upper()
                                if corr.isdigit():
                                    corr = chr(ord('A') + int(corr))
                                opts = [f"{lab}) {choices[idx]}" for lab, idx in mapping.items()]
                                rec = {
                                    "instruction": "Choose the correct option.",
                                    "input": (str(q).strip() + ("\nOptions: " + " \n".join(opts) if opts else "")),
                                    "output": corr,
                                }
                                px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                                pairs.append((px, py))
                    except Exception as e:
                        print(f"[Mix] Skip mathqa: {e}")
                elif name in ('aqua', 'aqua_rat'):
                    try:
                        ds = load_dataset('aqua_rat')
                        split = ds['train'].shuffle(seed=sft_seed)
                        take = max(1, int(math_ds_budget.get(name, 0)))
                        for i, dp in enumerate(split):
                            if i >= take:
                                break
                            rec = _fmt_aqua(dp)
                            px, py = _tokenize_prompt(Prompter('alpaca'), rec)
                            pairs.append((px, py))
                    except Exception as e:
                        print(f"[Mix] Skip aqua_rat: {e}")
                return pairs

            # Build, batchify, and extend
            math_counts = {}
            for name in math_names:
                pairs = _pairs_from_math(name)
                batches = list(_batch_loader(pairs, max(1, train_batch_size)))
                math_batches_all.extend(batches)
                math_counts[name] = math_counts.get(name, 0) + len(batches)

        # Optional debug dump of bucket loads
        if dump_bucket_debug:
            try:
                print('[BucketDebug] LM counts:', lm_counts)
                print('[BucketDebug] INST counts:', inst_counts)
                if 'math_counts' in locals():
                    print('[BucketDebug] MATH counts:', math_counts)
                # Print one sample shape if available
                if lm_batches_all:
                    x = lm_batches_all[0][0]
                    print('[BucketDebug] LM sample shape:', tuple(x.shape))
                if inst_batches_all:
                    x = inst_batches_all[0][0]
                    print('[BucketDebug] INST sample shape:', tuple(x.shape))
                if 'math_batches_all' in locals() and math_batches_all:
                    x = math_batches_all[0][0]
                    print('[BucketDebug] MATH sample shape:', tuple(x.shape))
            except Exception:
                pass

        # 4) Proportional interleave by bucket_props, assign per-bucket loss weights
        def _parse_kv(s: str, default: float) -> dict:
            out = {"LM": default, "INST": default, "MATH": default}
            try:
                for seg in s.split(','):
                    if not seg.strip():
                        continue
                    k, v = seg.split(':')
                    out[k.strip().upper()] = float(v)
            except Exception:
                pass
            return out

        props = _parse_kv(bucket_props, 1.0)
        weights = _parse_kv(bucket_loss_weights, 1.0)

        # Normalize props
        total_p = sum(max(0.0, props[k]) for k in ("LM","INST","MATH"))
        if total_p <= 0:
            props = {"LM": 1.0, "INST": 0.0, "MATH": 0.0}
            total_p = 1.0
        for k in props:
            props[k] = max(0.0, props[k]) / total_p

        # Shuffle each pool for variety
        import random
        random.seed(sft_seed)
        random.shuffle(lm_batches_all)
        random.shuffle(inst_batches_all)
        random.shuffle(math_batches_all)
        i = j = k = 0
        nb = bucket_total_batches if bucket_total_batches is not None else None
        mixed = []
        def _pick_bucket():
            r = random.random()
            cut_lm = props["LM"]
            cut_inst = cut_lm + props["INST"]
            if r < cut_lm:
                return 'LM'
            elif r < cut_inst:
                return 'INST'
            return 'MATH'
        while True:
            if nb is not None and len(mixed) >= nb:
                break
            if not (i < len(lm_batches_all) or j < len(inst_batches_all) or k < len(math_batches_all)):
                break
            bucket = _pick_bucket()
            if bucket == 'LM' and i < len(lm_batches_all):
                mixed.append((lm_batches_all[i][0], lm_batches_all[i][1], float(weights['LM'])))
                i += 1
            elif bucket == 'INST' and j < len(inst_batches_all):
                mixed.append((inst_batches_all[j][0], inst_batches_all[j][1], float(weights['INST'])))
                j += 1
            elif bucket == 'MATH' and k < len(math_batches_all):
                mixed.append((math_batches_all[k][0], math_batches_all[k][1], float(weights['MATH'])))
                k += 1
            else:
                # fall back to any available
                if i < len(lm_batches_all):
                    mixed.append((lm_batches_all[i][0], lm_batches_all[i][1], float(weights['LM'])))
                    i += 1
                elif j < len(inst_batches_all):
                    mixed.append((inst_batches_all[j][0], inst_batches_all[j][1], float(weights['INST'])))
                    j += 1
                elif k < len(math_batches_all):
                    mixed.append((math_batches_all[k][0], math_batches_all[k][1], float(weights['MATH'])))
                    k += 1
        update_loader = mixed
    else:
        # Default batching if not mixing three buckets
        if not (sft_data_path and mix_lm_with_sft):
            update_loader = list(_batch_loader(update_loader, max(1, train_batch_size)))
            # get_loaders() builds last-token-only labels by default; for LM we want full-seq PPL-aligned loss.
            if not sft_data_path:
                update_loader = [(inp, inp.clone()) for (inp, _tar) in update_loader]

    # When using SFT-style or bucket-mixed masked labels, prefer label-aware trainer (ignore full_seq_loss)
    effective_full_seq = (full_seq_loss and not sft_data_path and not mix_buckets)
    if epochs > 0:
        t0 = time.perf_counter()
        if effective_full_seq:
            train_act_lora_full_seq(
                model,
                update_loader,
                lora_params,
                device=dev,
                epochs=epochs,
                train_steps=train_steps,
                lr=lr,
                log_every=log_every,
            )
        else:
            train_act_lora(
                model,
                update_loader,
                lora_params,
                device=dev,
                epochs=epochs,
                train_steps=train_steps,
                lr=lr,
                log_every=log_every,
            )
        _stage(
            "train_activation_lora",
            time.perf_counter() - t0,
            full_seq_loss=bool(effective_full_seq),
            epochs=int(epochs),
            train_steps=None if train_steps is None else int(train_steps),
        )
        if not skip_eval:
            t1 = time.perf_counter()
            try:
                label = "PPL after activation-LoRA (full seq)" if full_seq_loss else "PPL after activation-LoRA"
                ppl_eval(
                    model,
                    tokenizer,
                    datasets=eval_list,
                    model_seq_len=seqlen,
                    batch_size=4,
                    device=dev,
                    label=label,
                    max_batches=eval_max_batches,
                )
            except Exception as e:
                print(f"[Eval] Skipped PPL (LoRA) due to: {e}")
            _eval_stage("ppl_post_lora", time.perf_counter() - t1, label=str(label))

    if save_path:
        t0 = time.perf_counter()
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({"model": model, "tokenizer": tokenizer}, save_path)
        _stage("save_checkpoint", time.perf_counter() - t0, path=save_path)
        print(f"Saved model with activation-space LoRA to: {save_path}")
    elif epochs > 0:
        print("[Warn] --save_path not provided; trained model will not be saved.")

    # Optional C4 streaming evaluation (small slice) to avoid heavy downloads
    if eval_c4_stream and (not skip_eval):
        try:
            from datasets import load_dataset
        except Exception as e:
            print(f"[Eval] Skipping C4 streaming PPL (datasets import failed): {e}")
        else:
            import itertools
            dev = device
            model.to(dev).eval()
            # Build token windows from validation stream
            seqs = []
            stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            for ex in itertools.islice(iter(stream), int(c4_stream_val_docs)):
                t = ex.get('text') or ex.get('content') or ''
                if not t:
                    continue
                enc = tokenizer(t, return_tensors='pt')
                L = enc.input_ids.shape[1]
                if L < seqlen:
                    continue
                # take first window to be deterministic-ish
                ids = enc.input_ids[:, :seqlen]
                seqs.append(ids)
            if not seqs:
                print("[Eval] No C4 validation windows were collected; skipping.")
            else:
                # Stack into batches and compute PPL
                bs = 4
                import math as _math
                losses = []
                for k in range(0, len(seqs), bs):
                    batch = torch.cat(seqs[k:k+bs], dim=0).to(dev)
                    with torch.no_grad():
                        out = model(input_ids=batch, labels=batch)
                        losses.append(out.loss.detach().float().cpu())
                if losses:
                    import torch as _torch
                    mean_loss = _torch.stack(losses).mean().item()
                    ppl = _math.exp(mean_loss)
                    print({"C4_stream_ppl": ppl})
    elif eval_c4_stream and skip_eval:
        print("[Eval] --eval_c4_stream was set but --skip_eval enabled; skipping C4 streaming PPL.")

    timing["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    timing["total_sec"] = float(time.perf_counter() - t_run0)
    eval_total_sec = float(sum(float(x.get("sec", 0.0)) for x in timing.get("eval_stages", [])))
    timing["eval_total_sec"] = eval_total_sec
    timing["total_wo_eval_sec"] = float(timing["total_sec"] - eval_total_sec)
    # Convenience aggregates for "compression + act-lora" reporting (excludes eval)
    compression_stage_names = {"build_whitening_data", "profile_svdllm", "apply_whitening_hetero", "save_whitened_cache"}
    lora_stage_names = {"attach_activation_lora", "build_lora_loader_base", "train_activation_lora"}
    compression_sec = float(sum(float(x.get("sec", 0.0)) for x in timing.get("stages", []) if x.get("name") in compression_stage_names))
    lora_sec = float(sum(float(x.get("sec", 0.0)) for x in timing.get("stages", []) if x.get("name") in lora_stage_names))
    timing["compression_sec"] = compression_sec
    timing["act_lora_sec"] = lora_sec
    timing["compression_plus_act_lora_sec"] = float(compression_sec + lora_sec)
    print(f"[Time] total_wo_eval_sec={timing['total_wo_eval_sec']:.2f}s total_sec={timing['total_sec']:.2f}s eval_total_sec={eval_total_sec:.2f}s")
    print(f"[Time] compression_plus_act_lora_sec={timing['compression_plus_act_lora_sec']:.2f}s (compression={compression_sec:.2f}s act_lora={lora_sec:.2f}s)")
    if timing_out:
        _write_timing(timing_out)
        print(f"[Time] Wrote timing json to: {timing_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dataset", type=str, default="wikitext2")
    p.add_argument(
        "--keep_ratio",
        type=float,
        default=0.8,
        help="SVD-LLM compression ratio rho in (0,1]. For W in R^{m x n}, rho = k(m+n)/(mn), i.e., compressed params = rho * (m*n).",
    )
    p.add_argument(
        "--whitening_factorization",
        type=str,
        default="cholesky",
        choices=["cholesky", "svd"],
        help="How to factorize the whitening matrix. 'cholesky' matches SVD-LLM; 'svd' uses symmetric sqrt via eig/SVD (SVD-LLM2-style).",
    )
    p.add_argument(
        "--attn_keep_ratio",
        type=float,
        default=None,
        help="Optional attention rho for heterogeneous rank (defaults to --keep_ratio when omitted).",
    )
    p.add_argument(
        "--mlp_keep_ratio",
        type=float,
        default=None,
        help="Optional MLP rho for heterogeneous rank (defaults to --keep_ratio when omitted).",
    )
    p.add_argument("--whitening_nsamples", type=int, default=256)
    p.add_argument(
        "--whitening_lm_datasets",
        type=str,
        default="wikitext2,ptb,c4",
        help="Comma-separated LM datasets used ONLY for whitening (default: wikitext2,ptb,c4). Overrides --bucket_lm_datasets for whitening stats.",
    )
    p.add_argument("--mix_calib_buckets", action="store_true", help="(Deprecated here) Whitening uses LM-only mix (wikitext2/ptb/c4). Use svd_act_lora_mixed_calibrate.py for mixed whitening.")
    p.add_argument("--eval_datasets", type=str, default=None, help="Comma-separated datasets for PPL eval (e.g., 'wikitext2_val,ptb,c4').")
    p.add_argument("--eval_max_batches", type=int, default=None, help="Limit number of batches per eval dataset (for quick smoke tests).")
    p.add_argument(
        "--lora_nsamples",
        type=int,
        default=None,
        help="Global sample budget for LoRA finetune (defaults to whitening_nsamples). When --mix_buckets is enabled, this budget is split across LM/INST/MATH per --bucket_props and then evenly across datasets in each bucket.")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42, help="Random seed for whitening/LoRA data sampling and training.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hf_token", type=str, default=None)
    # Official-compat toggles (align with original SVD-LLM whitening)
    p.add_argument("--svdllm_compat_all", action="store_true", help="Enable all official-compat behaviors (whitening XTX, official ranks, explicit attention math).")
    p.add_argument("--svdllm_compat_whitening", action="store_true", help="Use original whitening accumulation (raw X^T X without centering).")
    p.add_argument("--svdllm_compat_ranks", action="store_true", help="Use original SVD rank formulas for attention/MLP modules.")
    p.add_argument("--svdllm_compat_attention", action="store_true", help="Force explicit attention (matmul+softmax) and 3-value return like HF.")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--full_seq_loss", action="store_true", help="Train LoRA on full causal LM loss (all tokens) instead of last-token only")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_steps", type=int, default=None, help="Optional max LoRA optimization steps (stops early when reached).")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--train_batch_size", type=int, default=1, help="Batch size for LoRA finetune loader (groups samples from get_loaders).")
    p.add_argument("--whitening_device", type=str, default=None, help="Device for whitening stats (e.g., cpu to offload and save GPU memory). Defaults to training device.")
    p.add_argument("--whitened_cache", type=str, default=None, help="Path to load/save a cached whitened checkpoint to skip recomputing compression.")
    p.add_argument("--model_dtype", type=str, default=None, help="Force model dtype (e.g., float16/bfloat16/float32). Defaults to HF dtype.")
    p.add_argument("--device_map", type=str, default=None, help="Accelerate device map for hybrid placement (e.g., auto, balanced).")
    p.add_argument("--offload_folder", type=str, default=None, help="Directory for CPU offload when using device_map.")
    p.add_argument("--trust_whitened_cache", action="store_true", help="Skip shape validation when loading whitened cache (use only if you trust the cache matches current settings).")
    p.add_argument("--max_gpu_mem", type=str, default=None, help="Max GPU memory string for device_map inference (e.g., '30GiB').")
    p.add_argument("--max_cpu_mem", type=str, default=None, help="Max CPU memory string for device_map inference (e.g., '256GiB').")
    # SFT-style data (official Alpaca-LoRA format)
    p.add_argument("--sft_data_path", type=str, default=None, help="HF datasets path for instruction SFT data (e.g., yahma/alpaca-cleaned). When set, replaces LoRA update data with formatted instruction prompts.")
    p.add_argument("--sft_cutoff_len", type=int, default=256, help="Max tokenized length for instruction samples (fixed-length).")
    p.add_argument("--sft_add_eos_token", action="store_true", help="Append EOS when not present and under cutoff.")
    p.add_argument("--sft_train_on_inputs", action="store_true", help="Do not mask instruction/input tokens in labels (defaults to False like Alpaca-LoRA when omitted).")
    p.add_argument("--sft_seed", type=int, default=42, help="Shuffle seed for instruction dataset sampling.")
    # Mixed SFT+LM options
    p.add_argument("--mix_lm_with_sft", action="store_true", help="Interleave LM updates with SFT during LoRA training.")
    p.add_argument("--mix_ratio", type=float, default=0.5, help="Probability of taking an SFT batch when mixing (0..1).")
    p.add_argument("--lm_dataset", type=str, default=None, help="LM dataset to mix (defaults to --dataset).")
    p.add_argument("--lm_nsamples", type=int, default=None, help="Number of LM samples to mix (defaults to lora_nsamples).")
    p.add_argument("--lm_loss_weight", type=float, default=1.0, help="Loss weight for LM batches during mixing.")
    p.add_argument("--sft_loss_weight", type=float, default=1.0, help="Loss weight for SFT batches during mixing.")
    # Multi-bucket mixture options
    p.add_argument("--mix_buckets", action="store_true", help="Enable three-bucket mixing: LM/Instruction/Math.")
    p.add_argument("--bucket_props", type=str, default="LM:0.4,INST:0.4,MATH:0.2", help="Bucket sampling proportions, e.g., 'LM:0.4,INST:0.4,MATH:0.2'.")
    p.add_argument("--bucket_lm_datasets", type=str, default="wikitext2,ptb", help="Comma-separated LM datasets (get_loaders-compatible).")
    p.add_argument(
        "--bucket_inst_datasets",
        type=str,
        default="yahma/alpaca-cleaned",
        help="Comma-separated instruction datasets. Supports Alpaca-style (yahma/alpaca-cleaned) and several MCQ/linguistic tasks (e.g., hellaswag, piqa, winogrande_xl, ai2_arc_easy, ai2_arc_challenge, openbookqa, cola, sst2).",
    )
    p.add_argument("--bucket_math_datasets", type=str, default="gsm8k", help="Comma-separated math datasets (supports 'gsm8k', 'aqua_rat').")
    p.add_argument("--bucket_total_batches", type=int, default=None, help="Cap the number of mixed batches (defaults to available).")
    p.add_argument("--bucket_loss_weights", type=str, default="LM:1.0,INST:1.0,MATH:1.0", help="Per-bucket loss weights.")
    p.add_argument("--dump_bucket_debug", action="store_true", help="Print per-bucket dataset batch counts and sample shapes.")
    # Optional C4 streaming eval to avoid heavy downloads
    p.add_argument("--eval_c4_stream", action="store_true", help="Evaluate PPL on a small streaming slice of C4 'en' (validation).")
    p.add_argument("--c4_stream_val_docs", type=int, default=2000, help="Number of C4 validation docs to stream for eval windows.")
    p.add_argument("--c4_stream_train_docs", type=int, default=4000, help="Number of C4 train docs to stream when building LM bucket batches.")
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save checkpoint with activation-space LoRA.",
    )
    # Quiet dataset/progress logging to reduce noisy 'Resolving data files' bars
    p.add_argument("--quiet_data_logs", action="store_true", help="Suppress HuggingFace datasets/hub progress bars and reduce verbosity.")
    # Timing / eval controls
    p.add_argument("--skip_eval", action="store_true", help="Skip all evaluation (PPL + streaming eval). Useful for measuring pure compression/training time.")
    p.add_argument("--timing_out", type=str, default=None, help="Write timing breakdown JSON to this path.")
    p.add_argument("--stop_after_compress", action="store_true", help="Exit right after whitening+SVD compression (no LoRA attach/train/eval).")
    p.add_argument("--force_recompress", action="store_true", help="Ignore existing --whitened_cache and recompute whitening+SVD (for true timing).")
    args = p.parse_args()

    # Apply compat flags via environment for downstream modules
    if args.svdllm_compat_all:
        os.environ["SVDLLM_COMPAT_ALL"] = "1"
    if args.svdllm_compat_whitening:
        os.environ["SVDLLM_COMPAT_WHITENING"] = "1"
    if args.svdllm_compat_ranks:
        os.environ["SVDLLM_COMPAT_RANKS"] = "1"
    if args.svdllm_compat_attention:
        os.environ["SVDLLM_COMPAT_ATTENTION"] = "1"
    # Default to official SVD-LLM whitening/rank behaviors unless explicitly overridden.
    if os.getenv("SVDLLM_COMPAT_WHITENING") is None and not args.svdllm_compat_all and not args.svdllm_compat_whitening:
        os.environ["SVDLLM_COMPAT_WHITENING"] = "1"
    if os.getenv("SVDLLM_COMPAT_RANKS") is None and not args.svdllm_compat_all and not args.svdllm_compat_ranks:
        os.environ["SVDLLM_COMPAT_RANKS"] = "1"

    # Optional: silence datasets/hub progress output
    if args.quiet_data_logs:
        try:
            from datasets.utils.logging import set_verbosity_error, disable_progress_bar
            set_verbosity_error()
            disable_progress_bar()
        except Exception:
            pass
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

    run_activation_lora(
        model_id=args.model,
        dataset=args.dataset,
        keep_ratio=args.keep_ratio,
        whitening_nsamples=args.whitening_nsamples,
        whitening_lm_datasets=args.whitening_lm_datasets,
        whitening_factorization=args.whitening_factorization,
        attn_keep_ratio=args.attn_keep_ratio,
        mlp_keep_ratio=args.mlp_keep_ratio,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_nsamples=args.lora_nsamples,
        eval_datasets=args.eval_datasets,
        full_seq_loss=args.full_seq_loss,
        seqlen=args.seqlen,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        train_steps=args.train_steps,
        lr=args.lr,
        log_every=args.log_every,
        train_batch_size=args.train_batch_size,
        eval_max_batches=args.eval_max_batches,
        save_path=args.save_path,
        hf_token=args.hf_token,
        whitening_device=args.whitening_device,
        whitened_cache=args.whitened_cache,
        model_dtype=args.model_dtype,
        device_map=args.device_map,
        offload_folder=args.offload_folder,
        trust_whitened_cache=args.trust_whitened_cache,
        max_gpu_mem=args.max_gpu_mem,
        max_cpu_mem=args.max_cpu_mem,
        sft_data_path=args.sft_data_path,
        sft_cutoff_len=args.sft_cutoff_len,
        sft_add_eos_token=args.sft_add_eos_token,
        sft_train_on_inputs=args.sft_train_on_inputs,
        sft_seed=args.sft_seed,
        mix_calib_buckets=args.mix_calib_buckets,
        mix_lm_with_sft=args.mix_lm_with_sft,
        mix_ratio=args.mix_ratio,
        lm_dataset=args.lm_dataset,
        lm_nsamples=args.lm_nsamples,
        lm_loss_weight=args.lm_loss_weight,
        sft_loss_weight=args.sft_loss_weight,
        mix_buckets=args.mix_buckets,
        bucket_props=args.bucket_props,
        bucket_lm_datasets=args.bucket_lm_datasets,
        bucket_inst_datasets=args.bucket_inst_datasets,
        bucket_math_datasets=args.bucket_math_datasets,
        bucket_total_batches=args.bucket_total_batches,
        bucket_loss_weights=args.bucket_loss_weights,
        # C4 streaming knobs
        eval_c4_stream=args.eval_c4_stream,
        c4_stream_val_docs=args.c4_stream_val_docs,
        c4_stream_train_docs=args.c4_stream_train_docs,
        dump_bucket_debug=args.dump_bucket_debug,
        skip_eval=args.skip_eval,
        timing_out=args.timing_out,
        stop_after_compress=args.stop_after_compress,
        force_recompress=args.force_recompress,
    )


if __name__ == "__main__":
    main()
