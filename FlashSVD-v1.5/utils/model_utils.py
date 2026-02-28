from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Type

import torch
import torch.nn as nn


def find_layers(
    module: nn.Module,
    layers: Tuple[Type[nn.Module], ...] = (nn.Linear,),
    *,
    prefix: str = "",
) -> Dict[str, nn.Module]:
    """Recursively collect layers of given types.

    The returned keys match `module.named_modules()` names, which is what the
    SVD-LLM scripts expect (e.g. "self_attn.q_proj", "mlp.gate_proj").
    """
    found: Dict[str, nn.Module] = {}
    for name, child in module.named_modules():
        if name == "":
            continue
        if isinstance(child, layers):
            found[prefix + name] = child
    return found


def _token_arg(hf_token: Optional[str]):
    # transformers>=4.35 uses `token=`; older versions use `use_auth_token=`.
    return {"token": hf_token} if hf_token else {}


def _is_tokenizer_like(obj) -> bool:
    if obj is None or isinstance(obj, bool):
        return False
    # HF tokenizers are callable and implement encode/decode.
    if callable(obj):
        return True
    return hasattr(obj, "encode") and hasattr(obj, "decode")


def _load_tokenizer(model_id: str, *, hf_token: Optional[str] = None):
    from transformers import AutoTokenizer

    last_err: Optional[Exception] = None
    for use_fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=use_fast,
                **_token_arg(hf_token),
            )
        except Exception as e:
            last_err = e
            continue

        if not _is_tokenizer_like(tok):
            # Extremely defensive: we've observed environments where a non-tokenizer
            # placeholder (e.g. bool) can surface here.
            continue

        try:
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
        except Exception:
            pass
        return tok

    if last_err is not None:
        raise last_err
    raise RuntimeError(f"Failed to load a usable tokenizer for: {model_id}")


def ensure_transformers_layer_idx(model: nn.Module) -> None:
    """Ensure decoder self-attention modules have `layer_idx`.

    Transformers>=4.4x Cache implementations (DynamicCache/StaticCache/...) use
    `layer_idx` to route KV updates. Some pickled checkpoints (e.g. after module
    replacement) may miss this attribute.
    """
    try:
        # LLaMA / Mistral style
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is not None:
            for i, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    # Force-correct even if present: some replacement pipelines set all to 0.
                    setattr(attn, "layer_idx", int(i))
            return
        # OPT style
        dec_layers = getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None)
        if dec_layers is not None:
            for i, layer in enumerate(dec_layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    setattr(attn, "layer_idx", int(i))
    except Exception:
        # Best-effort only.
        return


def get_model_from_huggingface(
    model_id: str,
    *,
    hf_token: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load (model, tokenizer) from HuggingFace hub.

    Note: we keep the model on CPU; callers can move/cast as needed.
    """
    from transformers import AutoModelForCausalLM

    tok = _load_tokenizer(model_id, hf_token=hf_token)

    # Avoid HF warnings about dtype; let caller cast later if desired.
    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True, **_token_arg(hf_token))
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    ensure_transformers_layer_idx(model)

    # Common in quant/ptq pipelines.
    model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
    return model, tok


def get_model_from_local(path: str):
    """Load (model, tokenizer) from a local torch.save checkpoint.

    Expected format: torch.save({'model': model, 'tokenizer': tokenizer}, path)
    """
    p = Path(path)
    if p.is_dir():
        from transformers import AutoModelForCausalLM

        tok = _load_tokenizer(str(p), hf_token=None)
        model = AutoModelForCausalLM.from_pretrained(str(p), trust_remote_code=True, low_cpu_mem_usage=True)
        ensure_transformers_layer_idx(model)
        model.seqlen = int(getattr(model.config, "max_position_embeddings", 2048))
        return model, tok

    obj = _torch_load_local_checkpoint(p)
    if isinstance(obj, dict) and "model" in obj and "tokenizer" in obj:
        model = obj["model"]
        ensure_transformers_layer_idx(model)
        return model, obj["tokenizer"]
    if hasattr(obj, "forward"):
        # A pickled HF model (rare but possible): try to recover tokenizer.
        model = obj
        ensure_transformers_layer_idx(model)
        model_id = getattr(model, "name_or_path", None)
        if model_id:
            try:
                tok = _load_tokenizer(str(model_id), hf_token=None)
                return model, tok
            except Exception:
                pass
        raise ValueError(
            f"Loaded a model object from {path} but could not recover a tokenizer. "
            "Please re-save as {'model': model, 'tokenizer': tok}."
        )
    raise ValueError(f"Unrecognized checkpoint format at: {path}")


@torch.no_grad()
def measure_param_bytes(model: nn.Module) -> int:
    """Sum unique parameter storage bytes (avoids double-count on tied weights)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p is None:
            continue
        try:
            st = p.untyped_storage()
            key = (int(st.data_ptr()), int(st.nbytes()))
            nbytes = int(st.nbytes())
        except Exception:
            st = p.storage()
            nbytes = int(st.size()) * int(p.element_size())
            key = (int(st.data_ptr()), int(nbytes))
        if key in seen:
            continue
        seen.add(key)
        total += nbytes
    return total


def mib(nbytes: int) -> float:
    return float(nbytes) / (1024.0**2)


def set_env_flag(name: str, value: bool):
    if value:
        os.environ[name] = "1"
    else:
        # Don't leave stale "0" around; scripts typically check != "0".
        os.environ.pop(name, None)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _looks_like_flashsvd_checkpoint(path: Path) -> bool:
    try:
        root = Path(__file__).resolve().parents[1]  # FlashSVD-v1.5/
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            return False
    except Exception:
        # Best-effort: if we can't resolve, fall back to name heuristics only.
        pass
    name = path.name.lower()
    # Common outputs produced by SVDLLM(.py) / SVDLLM_flashsvd(.py)
    return any(
        key in name
        for key in (
            "_whitening_only_",
            "_whitening_then_update_",
            "_update_only_",
            "_profiling_",
            "svdllm",
            "flashsvd",
            "merge.pt",
        )
    )


def _torch_load_local_checkpoint(path: Path):
    """Load a local torch checkpoint, handling PyTorch>=2.6 weights_only default.

    PyTorch 2.6 changed `torch.load` default `weights_only=True`, which rejects
    pickled model/tokenizer objects (our SVDLLM scripts save them as a dict).
    We try a safe load first and fall back to `weights_only=False` only when we
    can reasonably assume the checkpoint is trusted.
    """

    def _load(*, weights_only: Optional[bool]):
        if weights_only is None:
            return torch.load(str(path), map_location="cpu")
        try:
            return torch.load(str(path), map_location="cpu", weights_only=weights_only)
        except TypeError:
            # Older PyTorch without weights_only.
            return torch.load(str(path), map_location="cpu")

    try:
        return _load(weights_only=True)
    except Exception as e:
        msg = str(e)
        is_weights_only = "Weights only load failed" in msg or "weights_only" in msg
        if not is_weights_only:
            raise

        trusted = _env_truthy("FLASH_SVD_TRUST_PICKLE") or _looks_like_flashsvd_checkpoint(path)
        if not trusted:
            raise RuntimeError(
                f"Refusing to load pickled checkpoint with weights_only=False: {path}\n"
                "PyTorch>=2.6 defaults torch.load(weights_only=True), which can't load "
                "our pickled {'model': model, 'tokenizer': tok} format.\n"
                "If you trust this checkpoint, re-run with: FLASH_SVD_TRUST_PICKLE=1"
            ) from e

        # Trusted fallback: allow pickled model/tokenizer objects.
        return _load(weights_only=False)
