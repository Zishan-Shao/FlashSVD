from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from utils.evaluator import (
    _can_use_manual_llama_decode_step,
    _manual_llama_decode_step,
    decode_kvcache_eval,
)
from utils.model_utils import (
    get_model_from_huggingface,
    get_model_from_local,
    get_model_from_source,
)


_RUNTIME_ENV_VARS = (
    "SVDLLM_FLASH_FALLBACK",
    "FLASH_SVD_DISABLE_FFN",
    "FLASH_SVD_BASELINE_LR_KVCACHE",
    "FLASH_SVD_ENABLE_DENSE_ATTN_DECODE",
    "FLASH_SVD_DENSE_DECODE_BACKEND",
    "FLASH_SVD_DENSE_DECODE_GRAPH",
    "FLASH_SVD_DENSE_DECODE_AUTOTUNE_SCOPE",
    "FLASH_SVD_USE_CUTLASS_ROPEATTN",
    "FLASH_SVD_BASELINE_DENSE_KVCACHE",
    "FLASH_SVD_REFERENCE_DENSE_ATTN",
    "FLASH_SVD_ENABLE_EXPERIMENTAL_FFN",
    "FLASH_SVD_MLP_CUDA_GRAPH",
    "FLASH_SVD_MLP_CUDA_GRAPH_ALIAS_OUTPUT",
    "FLASH_SVD_MLP_CUDA_GRAPH_SCOPE",
    "FLASH_SVD_FFN_BACKEND",
)


def _set_env_flag(name: str, enabled: bool) -> None:
    if enabled:
        os.environ[name] = "1"
    else:
        os.environ.pop(name, None)


def _set_env_value(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)


def _backend_needs_experimental_ffn(backend: str) -> bool:
    raw = str(backend).strip().lower().replace("-", "_")
    return raw in {
        "flashsvd_mlp_dual_split_prod",
        "flashsvd_mlp_dual_split_cutlass",
        "flashsvd_mlp_dual_split_triton_v1",
        "flashsvd_mlp_dual_split_triton_v2",
        "flashsvd_mlp_dual_split_triton_v2_sm80",
        "flashsvd_mlp_dual_split_triton_v3",
        "dual_split_cublas",
        "dual_split_cutlass",
        "dual_split_kernel",
        "dual_split_kernel_v2",
        "dual_split_kernel_v2_sm80",
        "dual_split_kernel_v3",
    }


def runtime_env_snapshot() -> dict[str, str | None]:
    return {name: os.getenv(name) for name in _RUNTIME_ENV_VARS}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_runtime(
    mode: str = "flashsvd",
    *,
    ffn_backend: str = "auto",
    enable_mlp_graph: bool = True,
    mlp_graph_scope: str = "layer_tail",
    graph_alias_output: bool = False,
    enable_flash_dense_attn: bool = True,
    dense_decode_backend: str = "auto",
    enable_dense_decode_graph: bool = True,
    dense_decode_autotune_scope: str = "attention",
    enable_cutlass_rope_attn: bool = False,
) -> dict[str, str | None]:
    """Configure environment flags for the current FlashSVD v1.5 serving path.

    `mode="flashsvd"` enables the dense-KV decode route and the production MLP
    backend. `mode="hf"` disables FlashSVD runtime overrides and falls back to
    the default Hugging Face generation path.
    """

    raw_mode = str(mode).strip().lower().replace("-", "_")
    if raw_mode not in {"flashsvd", "hf", "baseline"}:
        raise ValueError(f"Unsupported runtime mode: {mode}")

    if raw_mode == "flashsvd":
        _set_env_flag("SVDLLM_FLASH_FALLBACK", False)
        _set_env_flag("FLASH_SVD_DISABLE_FFN", False)
        _set_env_flag("FLASH_SVD_BASELINE_LR_KVCACHE", False)
        _set_env_flag("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", enable_flash_dense_attn)
        _set_env_value(
            "FLASH_SVD_DENSE_DECODE_BACKEND",
            str(dense_decode_backend) if enable_flash_dense_attn else None,
        )
        _set_env_flag("FLASH_SVD_DENSE_DECODE_GRAPH", enable_dense_decode_graph and enable_flash_dense_attn)
        _set_env_value(
            "FLASH_SVD_DENSE_DECODE_AUTOTUNE_SCOPE",
            str(dense_decode_autotune_scope) if enable_flash_dense_attn else None,
        )
        _set_env_flag("FLASH_SVD_USE_CUTLASS_ROPEATTN", enable_cutlass_rope_attn)
        _set_env_flag("FLASH_SVD_BASELINE_DENSE_KVCACHE", False)
        _set_env_flag("FLASH_SVD_REFERENCE_DENSE_ATTN", False)
        _set_env_flag("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", _backend_needs_experimental_ffn(ffn_backend))
        _set_env_flag("FLASH_SVD_MLP_CUDA_GRAPH", enable_mlp_graph)
        _set_env_flag("FLASH_SVD_MLP_CUDA_GRAPH_ALIAS_OUTPUT", graph_alias_output)
        _set_env_value("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE", str(mlp_graph_scope))
        _set_env_value("FLASH_SVD_FFN_BACKEND", str(ffn_backend))
        return runtime_env_snapshot()

    _set_env_flag("SVDLLM_FLASH_FALLBACK", True)
    _set_env_flag("FLASH_SVD_DISABLE_FFN", True)
    _set_env_flag("FLASH_SVD_BASELINE_LR_KVCACHE", False)
    _set_env_flag("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", False)
    _set_env_value("FLASH_SVD_DENSE_DECODE_BACKEND", None)
    _set_env_flag("FLASH_SVD_DENSE_DECODE_GRAPH", False)
    _set_env_value("FLASH_SVD_DENSE_DECODE_AUTOTUNE_SCOPE", None)
    _set_env_flag("FLASH_SVD_USE_CUTLASS_ROPEATTN", False)
    _set_env_flag("FLASH_SVD_BASELINE_DENSE_KVCACHE", False)
    _set_env_flag("FLASH_SVD_REFERENCE_DENSE_ATTN", False)
    _set_env_flag("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", False)
    _set_env_flag("FLASH_SVD_MLP_CUDA_GRAPH", False)
    _set_env_flag("FLASH_SVD_MLP_CUDA_GRAPH_ALIAS_OUTPUT", False)
    _set_env_value("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE", None)
    _set_env_value("FLASH_SVD_FFN_BACKEND", None)
    return runtime_env_snapshot()


def cast_model_for_inference(
    model: Any,
    *,
    dtype: str = "auto",
    device: str | torch.device | None = None,
):
    raw_dtype = str(dtype).strip().lower()
    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    if raw_dtype == "auto":
        if target_device.type == "cuda":
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            target_dtype = torch.float32
    elif raw_dtype in {"fp16", "float16", "half"}:
        target_dtype = torch.float16
    elif raw_dtype in {"bf16", "bfloat16"}:
        target_dtype = torch.bfloat16
    elif raw_dtype in {"fp32", "float32"}:
        target_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    model = model.to(device=target_device, dtype=target_dtype)
    model.eval()
    return model


def load_for_inference(
    source: str,
    *,
    hf_token: str | None = None,
    dtype: str = "auto",
    device: str | torch.device = "cpu",
):
    model, tokenizer = get_model_from_source(source, hf_token=hf_token)
    model = cast_model_for_inference(model, dtype=dtype, device=device)
    return model, tokenizer


def _sample_next_token(logits: torch.Tensor, *, do_sample: bool, temperature: float, top_p: float) -> torch.Tensor:
    if (not do_sample) or temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / float(max(temperature, 1e-5))
    probs = F.softmax(scaled, dim=-1)
    if float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > float(top_p)
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return torch.gather(sorted_idx, dim=-1, index=sampled)
    return torch.multinomial(probs, num_samples=1)


def _build_generation_cache(model: Any, *, prompt_len: int, max_new_tokens: int, batch_size: int, device: torch.device):
    dtype = next(iter(model.parameters())).dtype
    max_cache_len = int(prompt_len) + int(max_new_tokens) + 4
    use_flash_dense = _env_truthy("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE") and not _env_truthy("SVDLLM_FLASH_FALLBACK")
    use_baseline_dense = _env_truthy("FLASH_SVD_BASELINE_DENSE_KVCACHE")
    if use_flash_dense or use_baseline_dense:
        from runtime.cache.attn_dense_kv import FlashSVDV15DenseKVCache

        return FlashSVDV15DenseKVCache(
            model.config,
            max_batch_size=int(batch_size),
            max_cache_len=int(max_cache_len),
            device=device,
            dtype=dtype,
        ), bool(use_flash_dense)

    from transformers.cache_utils import StaticCache

    return StaticCache(
        model.config,
        max_batch_size=int(batch_size),
        max_cache_len=int(max_cache_len),
        device=device,
        dtype=dtype,
    ), False


def _generate_text_llama_incremental(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    if int(input_ids.shape[0]) != 1:
        raise ValueError("generate_text currently supports batch_size=1 only.")
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = int(input_ids.shape[1])
    cache, use_manual_decode = _build_generation_cache(
        model,
        prompt_len=prompt_len,
        max_new_tokens=int(max_new_tokens),
        batch_size=int(input_ids.shape[0]),
        device=device,
    )

    prefill_kwargs = {
        "input_ids": input_ids,
        "use_cache": True,
        "past_key_values": cache,
        "cache_position": torch.arange(prompt_len, device=device, dtype=torch.long),
    }
    if attention_mask is not None:
        prefill_kwargs["attention_mask"] = attention_mask
    out = model(**prefill_kwargs)

    output_ids = input_ids[0].detach().cpu().tolist()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    next_token = _sample_next_token(
        out.logits[:, -1, :],
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
    )

    for step in range(int(max_new_tokens)):
        next_token_id = int(next_token.item())
        output_ids.append(next_token_id)
        if eos_token_id is not None and next_token_id == int(eos_token_id):
            break
        if step + 1 >= int(max_new_tokens):
            break
        cache_pos = torch.tensor([prompt_len + step], device=device, dtype=torch.long)
        if use_manual_decode:
            out = _manual_llama_decode_step(
                model,
                next_token,
                past_key_values=cache,
                cache_position=cache_pos,
            )
        else:
            out = model(
                input_ids=next_token,
                use_cache=True,
                past_key_values=cache,
                cache_position=cache_pos,
            )
        next_token = _sample_next_token(
            out.logits[:, -1, :],
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
        )

    completion_ids = output_ids[prompt_len:]
    return {
        "prompt_token_count": prompt_len,
        "generated_token_count": int(len(completion_ids)),
        "output_ids": output_ids,
        "completion_ids": completion_ids,
        "text": tokenizer.decode(output_ids, skip_special_tokens=False),
        "completion_text": tokenizer.decode(completion_ids, skip_special_tokens=False),
    }


@torch.inference_mode()
def generate_text(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    max_new_tokens: int = 64,
    device: str | torch.device | None = None,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Run a single-prompt generation and return both text and token ids."""

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    if _can_use_manual_llama_decode_step(model):
        return _generate_text_llama_incremental(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=int(max_new_tokens),
            device=target_device,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
        )

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(target_device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=target_device)
    else:
        attention_mask = attention_mask.to(target_device)

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(max_new_tokens),
        "use_cache": True,
        "do_sample": bool(do_sample),
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = int(pad_token_id)
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = int(eos_token_id)
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)

    output_ids = model.generate(**generate_kwargs)
    if int(output_ids.shape[0]) != 1:
        raise ValueError("generate_text currently supports batch_size=1 only.")

    prompt_token_count = int(input_ids.shape[1])
    output_list = output_ids[0].detach().cpu().tolist()
    completion_ids = output_list[prompt_token_count:]
    return {
        "prompt_token_count": prompt_token_count,
        "generated_token_count": int(len(completion_ids)),
        "output_ids": output_list,
        "completion_ids": completion_ids,
        "text": tokenizer.decode(output_list, skip_special_tokens=False),
        "completion_text": tokenizer.decode(completion_ids, skip_special_tokens=False),
    }


__all__ = [
    "cast_model_for_inference",
    "configure_runtime",
    "decode_kvcache_eval",
    "generate_text",
    "get_model_from_huggingface",
    "get_model_from_local",
    "get_model_from_source",
    "load_for_inference",
    "runtime_env_snapshot",
]
