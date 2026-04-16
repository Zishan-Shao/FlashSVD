#!/usr/bin/env python3
"""
GPTQ quantization for a repo-native `.pt` checkpoint.

Two modes are supported:
1. default: GPTQ fake quantization (quantize + dequantize back to fp16/bf16)
2. --real_pack: GPTQ fake quantization followed by replacement of quantized
   Linear modules with real packed W4/W8 modules for storage/runtime validation

The output checkpoint is still repo-native:
    torch.save({"model": model, "tokenizer": tok}, out_ckpt)

Example:
  CKPT=checkpoints/non_leak/llama2_7b_kr0.4_actlora_lmonly_diverse_big.pt
  python quant/gptq_quantize_ckpt.py --in_ckpt "$CKPT" --wbits 4 --groupsize 128 \
    --calib_dataset wikitext2 --calib_nsamples 32 --calib_seqlen 2048 \
    --eval_datasets wikitext2,ptb,c4 --eval_batch_size 4 --device cuda \
    --out_ckpt /home/zs89/FlashSVD/checkpoints/quantized/llama2_7b_kr0.4_actlora_lmonly_diverse_big_gptq_w4_g128.pt
"""

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Allow running as `python quant/...py` from repo root without relying on PYTHONPATH.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUANT_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_QUANT_ROOT)
for _path in (_REPO_ROOT, _QUANT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Ensure wrapper class is picklable (avoid get_model_from_local creating a local shim class).
try:  # pragma: no cover
    import __main__ as _main

    if not hasattr(_main, "ActivationSpaceLoRAWrapper"):
        from expressivity.non_leak.svd_act_lora_aligned_adarank import ActivationSpaceLoRAWrapper as _ASLoRA

        setattr(_main, "ActivationSpaceLoRAWrapper", _ASLoRA)
except Exception:
    pass

from utils.evaluator import ppl_eval
from gptq.gptq import GPTQ
from gptq.quant import Quantizer, quantize
from quant.common import fuse_lora_wrappers_inplace, replace_linears_with_packed_quant_inplace
from utils.data_utils import get_loaders
from utils.model_utils import find_layers, get_model_from_local


@dataclass(frozen=True)
class GPTQArgs:
    wbits: int
    groupsize: int
    percdamp: float
    sym: bool
    act_order: bool
    static_groups: bool
    true_sequential: bool
    skip_lora: bool
    cpu_acts: bool
    cpu_hessian: bool


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _resolve_default_out_ckpt(in_ckpt: str, wbits: int, groupsize: int) -> str:
    stem = os.path.basename(in_ckpt)
    if stem.endswith(".pt"):
        stem = stem[: -len(".pt")]
    return os.path.join(_REPO_ROOT, "checkpoints", "quantized", f"{stem}_gptq_w{wbits}_g{groupsize}.pt")


def _pick_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    name = str(name).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name, None)


def _estimate_model_bytes(model: nn.Module) -> int:
    # Best-effort estimate of on-disk tensor payload size.
    # Note: may over-count if weights are tied; good enough to sanity-check free space.
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        try:
            total += int(t.numel()) * int(t.element_size())
        except Exception:
            pass
    return int(total)


def _true_sequential_groups(full: Dict[str, nn.Module]) -> List[List[str]]:
    """
    Best-effort sequential grouping for LLaMA(-like) blocks.
    Works for both HF names (q_proj/k_proj/v_proj/o_proj) and this repo's SVD factor names
    (q_u_proj/q_v_proj/...).
    """
    names = set(full.keys())
    # Also allow matching wrapped linears where the actual nn.Linear lives under ".base".
    names_with_base = set(names)
    for n in list(names):
        if n.endswith(".base"):
            names_with_base.add(n[: -len(".base")])
    groups = [
        [
            # factorized
            "self_attn.k_u_proj",
            "self_attn.k_v_proj",
            "self_attn.v_u_proj",
            "self_attn.v_v_proj",
            "self_attn.q_u_proj",
            "self_attn.q_v_proj",
            # unfactorized
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        [
            # factorized
            "self_attn.o_u_proj",
            "self_attn.o_v_proj",
            # unfactorized
            "self_attn.o_proj",
        ],
        [
            # factorized
            "mlp.up_u_proj",
            "mlp.up_v_proj",
            "mlp.gate_u_proj",
            "mlp.gate_v_proj",
            # unfactorized
            "mlp.up_proj",
            "mlp.gate_proj",
        ],
        [
            # factorized
            "mlp.down_u_proj",
            "mlp.down_v_proj",
            # unfactorized
            "mlp.down_proj",
        ],
    ]
    picked: List[List[str]] = []
    for g in groups:
        gg = []
        for n in g:
            if n in names:
                gg.append(n)
            elif n in names_with_base and (n + ".base") in names:
                gg.append(n + ".base")
        if gg:
            picked.append(gg)
    if picked:
        return picked
    return [list(full.keys())]


@torch.no_grad()
def _collect_layer0_inputs(
    model: nn.Module,
    dataloader,
    dev: str,
    nsamples: int,
    seqlen: int,
    store_device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    use_cache = bool(getattr(model.config, "use_cache", False))
    model.config.use_cache = False

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("Expected a LLaMA-like model with `model.model.layers`.")

    layers = model.model.layers
    if len(layers) == 0:
        raise ValueError("model.model.layers is empty.")

    # Ensure consistent seqlen hint
    model.seqlen = int(seqlen)

    # Move embedding + norm + first layer to device to capture hidden states.
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden = int(model.config.hidden_size)
    inps = torch.zeros((nsamples, model.seqlen, hidden), dtype=dtype, device=store_device)
    cache: Dict[str, object] = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = int(cache["i"])
            if idx < nsamples:
                captured = inp.detach()
                if captured.ndim == 3 and captured.shape[0] == 1:
                    captured = captured[0]
                inps[idx].copy_(captured.to(device=store_device, dtype=dtype))
                cache["i"] = idx + 1
                cache["attention_mask"] = kwargs.get("attention_mask", None)
                cache["position_ids"] = kwargs.get("position_ids", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if int(cache["i"]) >= nsamples:
            break
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass

    layers[0] = layers[0].module

    # Move back to CPU to free device memory for GPTQ.
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    if position_ids is None:
        # Newer transformers paths may synthesize position handling internally and never
        # pass explicit `position_ids` into the decoder layer kwargs. Fall back to the
        # canonical contiguous ids used for plain left-to-right calibration windows.
        position_ids = torch.arange(model.seqlen, device=dev, dtype=torch.long).unsqueeze(0)
    return inps, attention_mask, position_ids, use_cache


@torch.no_grad()
def gptq_quantize_llama(
    model: nn.Module,
    dataloader,
    dev: str,
    cfg: GPTQArgs,
) -> List[str]:
    print("[GPTQ] starting ...")
    if not torch.cuda.is_available() and str(dev).startswith("cuda"):
        raise RuntimeError(f"CUDA not available but --device={dev} requested.")

    act_store_device = "cpu" if cfg.cpu_acts else dev
    hessian_device = "cpu" if cfg.cpu_hessian else dev
    inps, attention_mask, position_ids, use_cache = _collect_layer0_inputs(
        model,
        dataloader=dataloader,
        dev=dev,
        nsamples=len(dataloader),
        seqlen=getattr(model, "seqlen", 2048),
        store_device=act_store_device,
    )
    outs = torch.zeros_like(inps)

    layers = model.model.layers
    quantized_module_names: List[str] = []

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)
        full = find_layers(layer)
        if cfg.skip_lora:
            full = {n: m for n, m in full.items() if ("lora_down" not in n and "lora_up" not in n)}

        if not full:
            print(f"[GPTQ] layer {layer_idx}: no quantizable Linear modules found; skipping.")
            layers[layer_idx] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps
            continue

        sequential = _true_sequential_groups(full) if cfg.true_sequential else [list(full.keys())]

        for group_names in sequential:
            subset = {n: full[n] for n in group_names if n in full}
            if not subset:
                continue

            gptqs: Dict[str, GPTQ] = {}
            for name, mod in subset.items():
                g = GPTQ(mod, hessian_device=hessian_device)
                q = Quantizer()
                q.configure(cfg.wbits, perchannel=True, sym=cfg.sym, mse=False)
                g.quantizer = q
                gptqs[name] = g

            def add_batch(name: str):
                def _hook(_module, inp, out):
                    gptqs[name].add_batch(inp[0].data, out.data)

                return _hook

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
            for j in range(inps.shape[0]):
                out_j = layer(
                    inps[j].unsqueeze(0).to(dev),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
                outs[j].copy_(out_j.detach().to(device=outs.device, dtype=outs.dtype))
            for h in handles:
                h.remove()

            for name in subset:
                print(f"[GPTQ] layer={layer_idx} module={name} quantizing ...")
                gptqs[name].fasterquant(
                    percdamp=cfg.percdamp,
                    groupsize=cfg.groupsize,
                    actorder=cfg.act_order,
                    static_groups=cfg.static_groups,
                )
                quantized_module_names.append(f"model.layers.{layer_idx}.{name}")
                gptqs[name].free()

            del gptqs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Recompute layer output after all groups are quantized.
        for j in range(inps.shape[0]):
            out_j = layer(
                inps[j].unsqueeze(0).to(dev),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
            outs[j].copy_(out_j.detach().to(device=outs.device, dtype=outs.dtype))

        layers[layer_idx] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print("[GPTQ] done.")
    return quantized_module_names


@torch.no_grad()
def rtn_quantize_llama(model: nn.Module, dev: str, wbits: int, sym: bool, skip_lora: bool) -> List[str]:
    print("[RTN] fake-quantizing Linear weights (no GPTQ Hessian) ...")
    use_cache = bool(getattr(model.config, "use_cache", False))
    model.config.use_cache = False
    layers = model.model.layers
    quantized_module_names: List[str] = []
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)
        subset = find_layers(layer)
        if skip_lora:
            subset = {n: m for n, m in subset.items() if ("lora_down" not in n and "lora_up" not in n)}
        for name in subset:
            q = Quantizer()
            q.configure(wbits, perchannel=True, sym=sym, mse=False)
            W = subset[name].weight.data
            q.find_params(W, weight=True)
            subset[name].weight.data = quantize(W, q.scale, q.zero, q.maxq).to(W.dtype)
            quantized_module_names.append(f"model.layers.{layer_idx}.{name}")
        layers[layer_idx] = layer.cpu()
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    print("[RTN] done.")
    return quantized_module_names


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GPTQ quantize a repo-native .pt checkpoint and optionally repack it to real W4/W8 modules.")

    ap.add_argument(
        "--in_ckpt",
        type=str,
        default="checkpoints/non_leak/llama2_7b_kr0.4_actlora_lmonly_diverse_big.pt",
        help="Input repo-native .pt checkpoint (torch.save({'model','tokenizer'})).",
    )
    ap.add_argument("--out_ckpt", type=str, default="", help="Output .pt path. Default writes to quant/<stem>_gptq_*.pt")

    ap.add_argument("--wbits", type=int, default=4, choices=[2, 3, 4, 8, 16])
    ap.add_argument("--groupsize", type=int, default=128, help="GPTQ groupsize; -1 for full row.")
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--sym", action="store_true", help="Use symmetric quantization (recommended).")
    ap.add_argument("--act_order", action="store_true", help="Enable GPTQ activation-order heuristic.")
    ap.add_argument("--static_groups", action="store_true", help="Use static groups (useful with --act_order).")
    ap.add_argument("--true_sequential", action="store_true", help="Quantize within layer in attention/MLP groups.")
    ap.add_argument("--skip_lora", action="store_true", help="Skip modules named *lora_down/*lora_up.")

    ap.add_argument("--nearest", action="store_true", help="Use RTN (round-to-nearest) instead of GPTQ.")

    ap.add_argument("--calib_dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    ap.add_argument("--calib_nsamples", type=int, default=32)
    ap.add_argument("--calib_seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eval_datasets", type=str, default="wikitext2,ptb,c4")
    ap.add_argument("--eval_seqlen", type=int, default=2048)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--max_eval_batches", type=int, default=0, help="If >0, cap eval batches for a quick smoke test.")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", help="Model dtype: float16|bfloat16|float32")
    ap.add_argument("--eval_before", action="store_true", help="Also run PPL eval before quantization.")
    ap.add_argument("--skip_save", action="store_true", help="Run quantization + eval, but do not write output checkpoint.")
    ap.add_argument("--real_pack", action="store_true", help="After GPTQ/RTN, replace quantized Linear modules with real packed W4/W8 modules.")
    ap.add_argument("--merge_lora", action="store_true", help="Fuse LoRA / ActLoRA wrappers into base Linear weights before quantization.")
    ap.add_argument("--pack_compute_dtype", type=str, default="auto", help="Compute dtype for packed modules: auto|float16|bfloat16|float32")
    ap.add_argument(
        "--save_legacy",
        action="store_true",
        help="Use legacy (non-zip) torch.save format; sometimes more robust on networked filesystems.",
    )
    ap.add_argument(
        "--cpu_acts",
        action="store_true",
        help="Store captured calibration activations on CPU and stream them to GPU per sample.",
    )
    ap.add_argument(
        "--cpu_hessian",
        action="store_true",
        help="Accumulate GPTQ Hessians on CPU and move them to GPU only during fasterquant().",
    )

    return ap


def quantize_checkpoint(args: argparse.Namespace) -> Dict[str, object]:

    if not (os.path.exists(args.in_ckpt) and args.in_ckpt.endswith(".pt")):
        raise ValueError(f"--in_ckpt must be an existing .pt file, got: {args.in_ckpt}")

    out_ckpt = args.out_ckpt.strip() or _resolve_default_out_ckpt(args.in_ckpt, wbits=args.wbits, groupsize=args.groupsize)
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)

    eval_datasets = _parse_csv(args.eval_datasets)
    if not eval_datasets:
        raise ValueError("--eval_datasets is empty")
    if bool(args.real_pack) and int(args.wbits) not in (4, 8):
        raise ValueError("--real_pack currently only supports --wbits 4 or 8")

    target_dtype = _pick_dtype(args.dtype)
    pack_compute_dtype = None if str(args.pack_compute_dtype).strip().lower() == "auto" else _pick_dtype(args.pack_compute_dtype)

    print(f"[Load] {args.in_ckpt}")
    model, tokenizer = get_model_from_local(args.in_ckpt)
    model.eval()
    if target_dtype is not None:
        model = model.to(dtype=target_dtype)
    if args.merge_lora:
        merged = fuse_lora_wrappers_inplace(model)
        print(f"[Merge] merged_wrappers={merged}")

    if args.eval_before:
        print("[Eval] before quantization")
        ppl_eval(
            model,
            tokenizer,
            datasets=eval_datasets,
            model_seq_len=int(args.eval_seqlen),
            batch_size=int(args.eval_batch_size),
            device=str(args.device),
            label="PPL (before)",
            max_batches=(int(args.max_eval_batches) if int(args.max_eval_batches) > 0 else None),
        )

    if int(args.wbits) >= 16:
        print("[Quant] wbits=16: skipping quantization; saving original checkpoint copy.")
        quantized_names: List[str] = []
        quant_wall = 0.0
    else:
        # Calibration loader (token batches).
        dataloader, _testenc = get_loaders(
            args.calib_dataset, nsamples=int(args.calib_nsamples), seed=int(args.seed), seqlen=int(args.calib_seqlen), tokenizer=tokenizer
        )
        # Align model.seqlen with calib windows.
        model.seqlen = int(args.calib_seqlen)

        if args.nearest:
            tick = time.time()
            quantized_names = rtn_quantize_llama(
                model,
                dev=str(args.device),
                wbits=int(args.wbits),
                sym=bool(args.sym),
                skip_lora=bool(args.skip_lora),
            )
            quant_wall = time.time() - tick
            print(f"[RTN] wall time: {quant_wall:.2f}s")
        else:
            cfg = GPTQArgs(
                wbits=int(args.wbits),
                groupsize=int(args.groupsize),
                percdamp=float(args.percdamp),
                sym=bool(args.sym),
                act_order=bool(args.act_order),
                static_groups=bool(args.static_groups),
                true_sequential=bool(args.true_sequential),
                skip_lora=bool(args.skip_lora),
                cpu_acts=bool(args.cpu_acts),
                cpu_hessian=bool(args.cpu_hessian),
            )
            tick = time.time()
            quantized_names = gptq_quantize_llama(model, dataloader=dataloader, dev=str(args.device), cfg=cfg)
            quant_wall = time.time() - tick
            print(f"[GPTQ] wall time: {quant_wall:.2f}s")

    if args.real_pack:
        pack_dtype = pack_compute_dtype or target_dtype or next(iter(model.parameters())).dtype
        tick = time.time()
        replaced = replace_linears_with_packed_quant_inplace(
            model,
            module_names=quantized_names,
            bits=int(args.wbits),
            group_size=int(args.groupsize),
            sym=bool(args.sym),
            compute_dtype=pack_dtype,
        )
        print(
            f"[Pack] replaced_modules={replaced} bits={int(args.wbits)} "
            f"group_size={int(args.groupsize)} compute_dtype={pack_dtype} wall={time.time() - tick:.2f}s"
        )

    print("[Eval] after quantization")
    ppl_eval(
        model,
        tokenizer,
        datasets=eval_datasets,
        model_seq_len=int(args.eval_seqlen),
        batch_size=int(args.eval_batch_size),
        device=str(args.device),
        label="PPL (after)",
        max_batches=(int(args.max_eval_batches) if int(args.max_eval_batches) > 0 else None),
    )

    if args.skip_save:
        print("[Save] skip_save=True: not writing checkpoint.")
        return {
            "method": ("rtn" if bool(args.nearest) else "gptq"),
            "out_ckpt": out_ckpt,
            "saved": False,
            "quantized_modules": int(len(quantized_names)),
            "wall_time_sec": float(quant_wall),
            "wbits": int(args.wbits),
            "groupsize": int(args.groupsize),
            "real_pack": bool(args.real_pack),
        }

    out_dir = os.path.dirname(out_ckpt) or "."
    try:
        free_bytes = int(shutil.disk_usage(out_dir).free)
        need_bytes = _estimate_model_bytes(model)
        print(f"[Disk] out_dir={out_dir} free={free_bytes/2**30:.2f}GiB est_model={need_bytes/2**30:.2f}GiB")
    except Exception:
        pass

    # Atomic-ish save: write a tmp file then replace. Avoids leaving a corrupted partial out_ckpt on failure.
    tmp_ckpt = out_ckpt + ".tmp"
    print(f"[Save] {out_ckpt}")
    try:
        if args.save_legacy:
            torch.save({"model": model, "tokenizer": tokenizer}, tmp_ckpt, _use_new_zipfile_serialization=False)
        else:
            torch.save({"model": model, "tokenizer": tokenizer}, tmp_ckpt)
        os.replace(tmp_ckpt, out_ckpt)
    finally:
        if os.path.exists(tmp_ckpt):
            try:
                os.remove(tmp_ckpt)
            except Exception:
                pass
    print("[Save] done.")
    return {
        "method": ("rtn" if bool(args.nearest) else "gptq"),
        "out_ckpt": out_ckpt,
        "saved": True,
        "quantized_modules": int(len(quantized_names)),
        "wall_time_sec": float(quant_wall),
        "wbits": int(args.wbits),
        "groupsize": int(args.groupsize),
        "real_pack": bool(args.real_pack),
    }


def main() -> None:
    args = build_parser().parse_args()
    quantize_checkpoint(args)


if __name__ == "__main__":
    main()
