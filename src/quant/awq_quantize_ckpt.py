#!/usr/bin/env python3
"""
Real AWQ quantization for a repo-native low-rank `.pt` checkpoint.

This script loads the repo-native checkpoint, optionally merges LoRA / ActLoRA
wrappers into plain Linear layers, applies AutoAWQ in memory with a FlashSVD-
specific LLaMA wrapper, runs a smoke forward, and saves a repo-native `.pt`
checkpoint containing the real AWQ quantized modules.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_QUANT_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_QUANT_ROOT)
for _path in (_REPO_ROOT, _QUANT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # pragma: no cover
    import __main__ as _main

    if not hasattr(_main, "ActivationSpaceLoRAWrapper"):
        from expressivity.non_leak.svd_act_lora_aligned_adarank import ActivationSpaceLoRAWrapper as _ASLoRA

        setattr(_main, "ActivationSpaceLoRAWrapper", _ASLoRA)
except Exception:
    pass

from quant.awq_compat import ensure_autoawq_compatibility
from quant.common import extract_flashsvd_rank_map, fuse_lora_wrappers_inplace
from quant.packed_quant_linear import PackedQuantLinear
from utils.data_utils import get_loaders
from utils.model_utils import get_model_from_local


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


def _parse_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _resolve_default_out_ckpt(in_ckpt: str, wbits: int, groupsize: int) -> str:
    stem = os.path.basename(in_ckpt)
    if stem.endswith(".pt"):
        stem = stem[: -len(".pt")]
    return os.path.join(_REPO_ROOT, "checkpoints", "quantized", f"{stem}_awq_w{wbits}_g{groupsize}.pt")


def _load_calib_token_lists(dataset: str, nsamples: int, seed: int, seqlen: int, tokenizer) -> List[List[int]]:
    dataloader, _ = get_loaders(dataset, nsamples=int(nsamples), seed=int(seed), seqlen=int(seqlen), tokenizer=tokenizer)
    samples: List[List[int]] = []
    for batch in dataloader:
        ids = batch[0] if isinstance(batch, (tuple, list)) else batch
        if torch.is_tensor(ids):
            if ids.ndim == 2 and ids.shape[0] == 1:
                ids = ids[0]
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if ids:
            samples.append([int(x) for x in ids])
    return samples


def _maybe_linear(module):
    return module if isinstance(module, nn.Linear) else None


def _maybe_append(
    layers: List[Dict],
    *,
    prev_op,
    linear_layers: List[Optional[nn.Linear]],
    input_feat: Dict[str, torch.Tensor],
    inp_name: str,
    module2inspect=None,
    kwargs=None,
) -> None:
    usable = [layer for layer in linear_layers if isinstance(layer, nn.Linear)]
    inp = input_feat.get(inp_name)
    if prev_op is None or not usable or inp is None:
        return
    item = {"prev_op": prev_op, "layers": usable, "inp": inp}
    if module2inspect is not None:
        item["module2inspect"] = module2inspect
    if kwargs is not None:
        item["kwargs"] = kwargs
    layers.append(item)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Real AWQ quantization for a repo-native low-rank .pt checkpoint.")
    ap.add_argument("--in_ckpt", type=str, required=True, help="Input repo-native .pt checkpoint.")
    ap.add_argument("--out_ckpt", type=str, default="", help="Output repo-native AWQ .pt checkpoint.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite --out_ckpt if it exists.")
    ap.add_argument("--wbits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--zero_point", action="store_true", help="Use asymmetric zero-point.")
    ap.add_argument("--no_zero_point", action="store_true", help="Disable zero-point.")
    ap.add_argument("--version", type=str, default="GEMM", help="AutoAWQ kernel version string.")
    ap.add_argument("--calib_dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    ap.add_argument("--calib_nsamples", type=int, default=32)
    ap.add_argument("--calib_seqlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="float16", help="float16|bfloat16|float32")
    ap.add_argument("--max_chunk_memory", type=int, default=1024 * 1024 * 1024)
    ap.add_argument("--n_parallel_calib_samples", type=int, default=1)
    ap.add_argument("--max_calib_samples", type=int, default=32)
    ap.add_argument("--max_calib_seq_len", type=int, default=512)
    ap.add_argument("--smoke_seq_len", type=int, default=16)
    ap.add_argument("--skip_save", action="store_true")
    ap.add_argument("--skip_smoke", action="store_true")
    ap.add_argument("--apply_clip", action="store_true", help="Enable AWQ clipping search. Disabled by default for ragged low-rank widths.")
    ap.add_argument("--merge_lora", dest="merge_lora", action="store_true", help="Merge LoRA / ActLoRA wrappers before AWQ.")
    ap.add_argument("--no_merge_lora", dest="merge_lora", action="store_false")
    ap.set_defaults(merge_lora=True)
    return ap


def quantize_checkpoint(args: argparse.Namespace) -> Dict[str, object]:
    if not (os.path.exists(args.in_ckpt) and args.in_ckpt.endswith(".pt")):
        raise ValueError(f"--in_ckpt must be an existing .pt file, got: {args.in_ckpt}")

    out_ckpt = args.out_ckpt.strip() or _resolve_default_out_ckpt(args.in_ckpt, int(args.wbits), int(args.groupsize))
    if os.path.exists(out_ckpt) and not args.overwrite and not args.skip_save:
        raise ValueError(f"--out_ckpt exists: {out_ckpt} (use --overwrite)")
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)

    ensure_autoawq_compatibility()
    from awq.models import LlamaAWQForCausalLM
    from awq.quantize.quantizer import AwqQuantizer
    from awq.utils.module import set_op_by_name
    from awq.utils.utils import clear_memory, get_best_device

    class FlashSVDLlamaAWQForCausalLM(LlamaAWQForCausalLM):
        @staticmethod
        def get_layers_for_scaling(module, input_feat, module_kwargs):
            layers: List[Dict] = []
            attn = module.self_attn
            mlp = module.mlp

            _maybe_append(
                layers,
                prev_op=module.input_layernorm,
                linear_layers=[
                    _maybe_linear(getattr(attn, "q_v_proj", None)),
                    _maybe_linear(getattr(attn, "k_v_proj", None)),
                    _maybe_linear(getattr(attn, "v_v_proj", None)),
                ],
                input_feat=input_feat,
                inp_name="self_attn.q_v_proj",
                module2inspect=attn,
                kwargs=module_kwargs,
            )

            _maybe_append(
                layers,
                prev_op=module.post_attention_layernorm,
                linear_layers=[
                    _maybe_linear(getattr(mlp, "gate_v_proj", None)),
                    _maybe_linear(getattr(mlp, "up_v_proj", None)),
                ],
                input_feat=input_feat,
                inp_name="mlp.gate_v_proj",
                module2inspect=mlp,
            )

            return layers

    class FlashSVDAwqQuantizer(AwqQuantizer):
        def pseudo_quantize_tensor(self, w: torch.Tensor):
            org_w_shape = w.shape
            assert len(org_w_shape) == 2
            assert torch.isnan(w).sum() == 0

            def _quantize_chunk(chunk: torch.Tensor):
                if self.zero_point:
                    max_val = chunk.amax(dim=1, keepdim=True)
                    min_val = chunk.amin(dim=1, keepdim=True)
                    max_int = 2**self.w_bit - 1
                    min_int = 0
                    scales = (max_val - min_val).clamp(min=1e-5) / max_int
                    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
                    quant = (torch.clamp(torch.round(chunk / scales) + zeros, min_int, max_int) - zeros) * scales
                    return quant, scales.squeeze(1), zeros.squeeze(1)
                max_val = chunk.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                max_int = 2 ** (self.w_bit - 1) - 1
                min_int = -(2 ** (self.w_bit - 1))
                scales = max_val / max_int
                quant = torch.clamp(torch.round(chunk / scales), min_int, max_int) * scales
                return quant, scales.squeeze(1), None

            if self.group_size > 0:
                q_chunks = []
                scale_chunks = []
                zero_chunks = []
                for start in range(0, org_w_shape[-1], self.group_size):
                    end = min(org_w_shape[-1], start + self.group_size)
                    q_chunk, scales, zeros = _quantize_chunk(w[:, start:end])
                    q_chunks.append(q_chunk)
                    scale_chunks.append(scales)
                    if zeros is not None:
                        zero_chunks.append(zeros)
                wq = torch.cat(q_chunks, dim=1)
                scales = torch.stack(scale_chunks, dim=1)
                zeros = torch.stack(zero_chunks, dim=1) if zero_chunks else None
            else:
                wq, scales_1d, zeros_1d = _quantize_chunk(w)
                scales = scales_1d.unsqueeze(1)
                zeros = zeros_1d.unsqueeze(1) if zeros_1d is not None else None

            assert torch.isnan(scales).sum() == 0
            assert torch.isnan(wq).sum() == 0
            return wq.reshape(org_w_shape), scales, zeros

        def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
            for name, linear_layer in named_linears.items():
                linear_layer = linear_layer.to(get_best_device()).half()
                quantized_weight, _scales, _zeros = self.pseudo_quantize_tensor(linear_layer.weight.data)
                linear_layer.weight.data = quantized_weight.to(dtype=linear_layer.weight.dtype)
                packed = PackedQuantLinear.from_linear(
                    linear_layer,
                    bits=int(self.w_bit),
                    group_size=int(self.group_size),
                    sym=(not bool(self.zero_point)),
                    compute_dtype=linear_layer.weight.dtype,
                )
                linear_layer.cpu()
                packed = packed.to(next(module.parameters()).device)
                set_op_by_name(module, name, packed)
                clear_memory()

    dtype = _pick_dtype(args.dtype)
    print(f"[Load] {args.in_ckpt}")
    model, tokenizer = get_model_from_local(args.in_ckpt)
    model.eval()
    if dtype is not None:
        model = model.to(dtype=dtype)

    if args.merge_lora:
        merged = fuse_lora_wrappers_inplace(model)
        print(f"[Merge] merged_wrappers={merged}")

    calib_tokens = _load_calib_token_lists(
        str(args.calib_dataset).strip(),
        nsamples=int(args.calib_nsamples),
        seed=int(args.seed),
        seqlen=int(args.calib_seqlen),
        tokenizer=tokenizer,
    )
    print(f"[Calib] dataset={args.calib_dataset} samples={len(calib_tokens)} seqlen={args.calib_seqlen}")

    zero_point = True
    if args.no_zero_point:
        zero_point = False
    if args.zero_point:
        zero_point = True
    quant_config = {
        "zero_point": bool(zero_point),
        "q_group_size": int(args.groupsize),
        "w_bit": int(args.wbits),
        "version": str(args.version),
    }
    print(f"[AWQ] quant_config={quant_config}")

    awq_model = FlashSVDLlamaAWQForCausalLM(model, "llama", False, model.config, quant_config, None)
    tick = time.time()
    awq_model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_tokens,
        n_parallel_calib_samples=(int(args.n_parallel_calib_samples) if int(args.n_parallel_calib_samples) > 0 else None),
        max_calib_samples=int(args.max_calib_samples),
        max_calib_seq_len=int(args.max_calib_seq_len),
        max_chunk_memory=int(args.max_chunk_memory),
        apply_clip=bool(args.apply_clip),
        quantizer_cls=FlashSVDAwqQuantizer,
    )
    wall = time.time() - tick
    rank_map = extract_flashsvd_rank_map(awq_model.model)
    print(f"[AWQ] wall time: {wall:.2f}s")
    print(f"[RankMap] layers={len(rank_map)}")

    if not args.skip_smoke:
        dev = str(args.device)
        awq_model.model = awq_model.model.to(dev)
        input_ids = torch.randint(0, awq_model.model.config.vocab_size, (1, int(args.smoke_seq_len)), device=dev)
        with torch.no_grad():
            out = awq_model.model(input_ids)
        print(f"[Smoke] logits_shape={tuple(out.logits.shape)} dtype={out.logits.dtype}")

    if args.skip_save:
        print("[Save] skip_save=True: not writing checkpoint.")
        return {
            "method": "awq",
            "out_ckpt": out_ckpt,
            "saved": False,
            "wall_time_sec": float(wall),
            "rank_map_layers": int(len(rank_map)),
            "wbits": int(args.wbits),
            "groupsize": int(args.groupsize),
        }

    print(f"[Save] {out_ckpt}")
    save_model = awq_model.model.cpu()
    torch.save({"model": save_model, "tokenizer": tokenizer}, out_ckpt)
    print("[Save] done.")
    return {
        "method": "awq",
        "out_ckpt": out_ckpt,
        "saved": True,
        "wall_time_sec": float(wall),
        "rank_map_layers": int(len(rank_map)),
        "wbits": int(args.wbits),
        "groupsize": int(args.groupsize),
    }


def main() -> None:
    args = build_parser().parse_args()
    quantize_checkpoint(args)


if __name__ == "__main__":
    main()
