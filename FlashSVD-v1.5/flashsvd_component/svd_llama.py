import math
import os
import inspect
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig

from kernels.flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors
from kernels.flashsvdsilu import (
    flashsvd_ffn_dual_split_token,
    flashsvd_ffn_dual_split_token_v2,
    flashsvd_ffn_dual_split_token_v2_sm80,
    flashsvd_ffn_dual_split_token_v3,
)

import importlib.util
from pathlib import Path


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


_DECODE_ATTN_MODS = None
_LRKV_MHA_SDPA_WARNED = False
_LRKV_RANK_MISMATCH_WARNED = False
_LRKV_MHA_STREAM_AUTO_WARNED = False
_LOWRANK_CACHE_CLS = None
_LOWRANK_CACHE_CLS_RESOLVED = False
_DENSE_CACHE_CLS = None
_DENSE_CACHE_CLS_RESOLVED = False
_EXPERIMENTAL_FFN_WARNED = False
_DECODE_AUTOTUNE_CACHE = {}
_DECODE_AUTOTUNE_LOGGED = set()
_DECODE_VARIANT_CACHE = {}
_DECODE_VARIANT_LOGGED = set()
_HF_LLAMA_LAYER_GRAPH_PATCHED = False
_HF_LLAMA_DECODER_LAYER_FORWARD = None
_FLASH_ATTN_WITH_KVCACHE = None
_FLASH_ATTN_WITH_KVCACHE_RESOLVED = False
_DENSE_TOKEN_DECODE_MOD = None


def _load_decode_mod(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load decode attention module spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_flashsvd_decode_attn_mods():
    global _DECODE_ATTN_MODS
    if _DECODE_ATTN_MODS is not None:
        return _DECODE_ATTN_MODS

    base = (
        Path(__file__).resolve().parents[1]
        / "kernels"
        / "flashsvd-v1.5"
        / "flashsvdropeattn"
    )
    mods = {}

    path_v15 = base / "flashsvdropeattn_v1.5_decode.py"
    if path_v15.exists():
        try:
            mods["v15"] = _load_decode_mod(path_v15, "flashsvdropeattn_v15_decode")
        except Exception:
            pass

    path_v16 = base / "flashsvdropeattn_v1.6_decode_opt.py"
    if path_v16.exists():
        try:
            mods["v16"] = _load_decode_mod(path_v16, "flashsvdropeattn_v16_decode_opt")
        except Exception:
            pass

    if not mods:
        raise ImportError(f"Failed to load any decode attention module from: {base}")
    _DECODE_ATTN_MODS = mods
    return mods


def _get_flashsvd_decode_attn_mod():
    mods = _get_flashsvd_decode_attn_mods()
    if "v16" in mods:
        return mods["v16"]
    return next(iter(mods.values()))


def _maybe_kwargs(fn, kwargs):
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _canonical_decode_variant(name: str) -> str:
    raw = str(name or "").strip().lower().replace("-", "_")
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"v15", "v1.5", "1.5"}:
        return "v15"
    if raw in {"v16_v1", "v1", "legacy"}:
        return "v16_v1"
    if raw in {"v16_v2", "v16", "v2"}:
        return "v16_v2"
    return "auto"


def _available_decode_variants(mods) -> list[str]:
    out: list[str] = []
    m15 = mods.get("v15", None)
    if m15 is not None and hasattr(m15, "DecodePackedFactors") and hasattr(m15, "flashsvd_attn_decode_packed"):
        out.append("v15")
    m16 = mods.get("v16", None)
    if m16 is not None and hasattr(m16, "DecodePackedFactors"):
        if hasattr(m16, "flashsvd_attn_decode_packed_v1"):
            out.append("v16_v1")
        if hasattr(m16, "flashsvd_attn_decode_packed"):
            out.append("v16_v2")
    return out


def _resolve_decode_variant(mods, variant: str):
    v = _canonical_decode_variant(variant)
    if v == "v15":
        mod = mods.get("v15", None)
        if mod is None:
            return None, None
        return mod, getattr(mod, "flashsvd_attn_decode_packed", None)
    if v == "v16_v1":
        mod = mods.get("v16", None)
        if mod is None:
            return None, None
        fn = getattr(mod, "flashsvd_attn_decode_packed_v1", None)
        if fn is None:
            fn = getattr(mod, "flashsvd_attn_decode_packed", None)
        return mod, fn
    if v == "v16_v2":
        mod = mods.get("v16", None)
        if mod is None:
            return None, None
        return mod, getattr(mod, "flashsvd_attn_decode_packed", None)
    return None, None


def _get_lowrank_cache_cls():
    global _LOWRANK_CACHE_CLS
    global _LOWRANK_CACHE_CLS_RESOLVED
    if _LOWRANK_CACHE_CLS_RESOLVED:
        return _LOWRANK_CACHE_CLS
    try:
        from flashsvd_component.lowrank_cache import LowRankKVCache
    except Exception:
        LowRankKVCache = None  # type: ignore[assignment]
    _LOWRANK_CACHE_CLS = LowRankKVCache
    _LOWRANK_CACHE_CLS_RESOLVED = True
    return _LOWRANK_CACHE_CLS


def _get_dense_cache_cls():
    global _DENSE_CACHE_CLS
    global _DENSE_CACHE_CLS_RESOLVED
    if _DENSE_CACHE_CLS_RESOLVED:
        return _DENSE_CACHE_CLS
    try:
        from flashsvd_component.dense_cache import FlashSVDDenseKVCache
    except Exception:
        FlashSVDDenseKVCache = None  # type: ignore[assignment]
    _DENSE_CACHE_CLS = FlashSVDDenseKVCache
    _DENSE_CACHE_CLS_RESOLVED = True
    return _DENSE_CACHE_CLS


def _get_flash_attn_with_kvcache():
    global _FLASH_ATTN_WITH_KVCACHE
    global _FLASH_ATTN_WITH_KVCACHE_RESOLVED
    if _FLASH_ATTN_WITH_KVCACHE_RESOLVED:
        return _FLASH_ATTN_WITH_KVCACHE
    fn = None
    try:
        from flash_attn import flash_attn_with_kvcache  # type: ignore

        fn = flash_attn_with_kvcache
    except Exception:
        try:
            from flash_attn.flash_attn_interface import flash_attn_with_kvcache  # type: ignore

            fn = flash_attn_with_kvcache
        except Exception:
            fn = None
    _FLASH_ATTN_WITH_KVCACHE = fn
    _FLASH_ATTN_WITH_KVCACHE_RESOLVED = True
    return _FLASH_ATTN_WITH_KVCACHE


def _call_flash_attn_with_kvcache(
    flash_attn_with_kvcache,
    q_bmhd: torch.Tensor,
    k_cache_bmhd: torch.Tensor,
    v_cache_bmhd: torch.Tensor,
    *,
    k_bmhd: Optional[torch.Tensor] = None,
    v_bmhd: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    causal: bool,
):
    sig = inspect.signature(flash_attn_with_kvcache)
    params = sig.parameters
    kwargs: dict[str, object] = {}
    if "k" in params and k_bmhd is not None:
        kwargs["k"] = k_bmhd
    if "v" in params and v_bmhd is not None:
        kwargs["v"] = v_bmhd
    if "cache_seqlens" in params and cache_seqlens is not None:
        kwargs["cache_seqlens"] = cache_seqlens
    if "rotary_cos" in params and rotary_cos is not None:
        kwargs["rotary_cos"] = rotary_cos
    if "rotary_sin" in params and rotary_sin is not None:
        kwargs["rotary_sin"] = rotary_sin
    if "rotary_interleaved" in params:
        kwargs["rotary_interleaved"] = False
    if "causal" in params:
        kwargs["causal"] = causal
    if "dropout_p" in params:
        kwargs["dropout_p"] = 0.0
    if "window_size" in params:
        kwargs["window_size"] = (-1, -1)
    return flash_attn_with_kvcache(q_bmhd, k_cache_bmhd, v_cache_bmhd, **kwargs)


def _get_dense_token_decode_mod():
    global _DENSE_TOKEN_DECODE_MOD
    if _DENSE_TOKEN_DECODE_MOD is not None:
        return _DENSE_TOKEN_DECODE_MOD
    path = (
        Path(__file__).resolve().parents[1]
        / "kernels"
        / "flashsvd-v1.5"
        / "flashsvdropeattn_short"
        / "flashsvdropeattn_dense_decode.py"
    )
    _DENSE_TOKEN_DECODE_MOD = _load_decode_mod(path, "flashsvdropeattn_dense_decode_component")
    return _DENSE_TOKEN_DECODE_MOD


def _flashsvd_ffn_backend() -> str:
    raw = str(os.getenv("FLASH_SVD_FFN_BACKEND", "auto")).strip().lower().replace("-", "_")
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"dual_split_cublas", "dual_cublas", "exact_cublas", "dual_split_exact"}:
        return "dual_split_cublas"
    if raw in {"dual_split_cublas_legacy", "dual_split_linear", "exact_linear"}:
        return "dual_split_cublas_legacy"
    if raw in {"dual_split_kernel", "dual_kernel", "exact_kernel", "dual_split"}:
        return "dual_split_kernel"
    if raw in {"dual_split_kernel_v2", "dual_kernel_v2", "exact_kernel_v2", "dual_split_v2"}:
        return "dual_split_kernel_v2"
    if raw in {"dual_split_kernel_v2_sm80", "dual_kernel_v2_sm80", "exact_kernel_v2_sm80", "dual_split_v2_sm80", "dual_split_kernel_v2_5", "dual_kernel_v2_5", "exact_kernel_v2_5", "dual_split_v2_5"}:
        return "dual_split_kernel_v2_sm80"
    if raw in {"dual_split_kernel_v3", "dual_kernel_v3", "exact_kernel_v3", "dual_split_v3"}:
        return "dual_split_kernel_v3"
    if raw in {"generic", "triton", "flash"}:
        return "dual_split_cublas_legacy"
    return "auto"


def _flashsvd_experimental_ffn_enabled() -> bool:
    return os.getenv("FLASH_SVD_ENABLE_EXPERIMENTAL_FFN", "0") != "0"


def _flashsvd_dense_attn_enabled() -> bool:
    return os.getenv("FLASH_SVD_ENABLE_DENSE_ATTN_DECODE", "0") != "0"


def _flashsvd_baseline_dense_kvcache_enabled() -> bool:
    return os.getenv("FLASH_SVD_BASELINE_DENSE_KVCACHE", "0") != "0"


def _flashsvd_reference_dense_attn_enabled() -> bool:
    return os.getenv("FLASH_SVD_REFERENCE_DENSE_ATTN", "0") != "0"


def _flashsvd_cuda_graph_enabled() -> bool:
    return os.getenv("FLASH_SVD_MLP_CUDA_GRAPH", "0") == "1"


def _flashsvd_cuda_graph_scope() -> str:
    raw = str(os.getenv("FLASH_SVD_MLP_CUDA_GRAPH_SCOPE", "auto")).strip().lower().replace("-", "_")
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"mlp", "module"}:
        return "mlp"
    if raw in {"layer", "layer_tail", "tail"}:
        return "layer_tail"
    return "auto"


def _flashsvd_cuda_graph_alias_output() -> bool:
    return os.getenv("FLASH_SVD_MLP_CUDA_GRAPH_ALIAS_OUTPUT", "0") == "1"


def _llama_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + float(eps))
    return weight * hidden_states.to(input_dtype)


def _unwrap_hidden_states_arg(hidden_states):
    if torch.is_tensor(hidden_states):
        return hidden_states
    if isinstance(hidden_states, (tuple, list)) and hidden_states:
        candidate = hidden_states[0]
        if torch.is_tensor(candidate):
            return candidate
    return hidden_states


def _should_use_flashsvd_layer_tail_cuda_graph(layer, hidden_states: torch.Tensor) -> bool:
    hidden_states = _unwrap_hidden_states_arg(hidden_states)
    return (
        isinstance(getattr(layer, "mlp", None), SVD_LlamaMLP)
        and torch.is_tensor(hidden_states)
        and hidden_states.is_cuda
        and (not layer.training)
        and (not torch.is_grad_enabled())
        and _flashsvd_cuda_graph_enabled()
        and _flashsvd_cuda_graph_scope() in {"auto", "layer_tail"}
        and os.getenv("SVDLLM_FLASH_FALLBACK", "0") == "0"
        and os.getenv("FLASH_SVD_DISABLE_FFN", "0") == "0"
        and hidden_states.shape[0] <= 4
        and hidden_states.shape[1] <= 4
    )


def _flashsvd_llama_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    global _HF_LLAMA_DECODER_LAYER_FORWARD
    hidden_states = _unwrap_hidden_states_arg(hidden_states)

    if _HF_LLAMA_DECODER_LAYER_FORWARD is None or not _should_use_flashsvd_layer_tail_cuda_graph(self, hidden_states):
        call_kwargs = _maybe_kwargs(
            _HF_LLAMA_DECODER_LAYER_FORWARD,
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_value": past_key_value,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings,
                **kwargs,
            },
        )
        return _HF_LLAMA_DECODER_LAYER_FORWARD(self, hidden_states, **call_kwargs)

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    norm_eps = getattr(
        self.post_attention_layernorm,
        "variance_epsilon",
        getattr(self.post_attention_layernorm, "eps", 1e-6),
    )
    hidden_states = self.mlp.flashsvd_post_attention_tail(
        hidden_states,
        self.post_attention_layernorm.weight,
        float(norm_eps),
        use_cuda_graph=True,
    )

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    return outputs


def enable_flashsvd_llama_layer_tail_cuda_graph() -> bool:
    global _HF_LLAMA_LAYER_GRAPH_PATCHED
    global _HF_LLAMA_DECODER_LAYER_FORWARD

    if _HF_LLAMA_LAYER_GRAPH_PATCHED:
        return True

    try:
        from transformers.models.llama import modeling_llama
    except Exception:
        return False

    decoder_cls = getattr(modeling_llama, "LlamaDecoderLayer", None)
    if decoder_cls is None:
        return False
    if _HF_LLAMA_DECODER_LAYER_FORWARD is None:
        _HF_LLAMA_DECODER_LAYER_FORWARD = decoder_cls.forward
    decoder_cls.forward = _flashsvd_llama_decoder_layer_forward
    _HF_LLAMA_LAYER_GRAPH_PATCHED = True
    return True

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SVD_LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        ratio=1
    ):
        super().__init__()
        self.ratio = ratio
        low_rank = int(intermediate_size * hidden_size * self.ratio / (intermediate_size + hidden_size))
        self.gate_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        
        self.down_u_proj = nn.Linear(low_rank, hidden_size, bias=False)
        self.down_v_proj = nn.Linear(intermediate_size, low_rank, bias=False)
        
        self.up_u_proj = nn.Linear(low_rank, intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(hidden_size, low_rank, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self._flashsvd_graph_cache = {}
        self._flashsvd_layer_graph_cache = {}
        self._dual_split_cublas_cache = {}
        self._dual_split_kernel_cache = {}
        self._dual_split_workspace_cache = {}

    def _ensure_runtime_state(self) -> None:
        # Older pickled checkpoints may bypass __init__ and miss these caches.
        if not hasattr(self, "_flashsvd_graph_cache"):
            self._flashsvd_graph_cache = {}
        if not hasattr(self, "_flashsvd_layer_graph_cache"):
            self._flashsvd_layer_graph_cache = {}
        if not hasattr(self, "_dual_split_cublas_cache"):
            self._dual_split_cublas_cache = {}
        if not hasattr(self, "_dual_split_kernel_cache"):
            self._dual_split_kernel_cache = {}
        if not hasattr(self, "_dual_split_workspace_cache"):
            self._dual_split_workspace_cache = {}

    def _use_flashsvd_cuda_graph(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and (not self.training)
            and (not torch.is_grad_enabled())
            and _flashsvd_cuda_graph_enabled()
            and _flashsvd_cuda_graph_scope() in {"auto", "mlp"}
            and x.shape[0] <= 4
            and x.shape[1] <= 4
        )

    def _get_dual_split_kernel_factors(self, device: torch.device, dtype: torch.dtype):
        self._ensure_runtime_state()
        key = (device.type, device.index, dtype)
        versions = (
            self.gate_u_proj.weight._version,
            self.up_u_proj.weight._version,
            self.down_v_proj.weight._version,
            self.down_u_proj.weight._version,
        )
        cached = self._dual_split_kernel_cache.get(key)
        if cached is not None and cached["versions"] == versions:
            return cached["gate_u"], cached["up_u"], cached["down_v"], cached["down_u"], cached["b2"]

        gate_u = self.gate_u_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        up_u = self.up_u_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        down_v = self.down_v_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        down_u = self.down_u_proj.weight.t().to(device=device, dtype=dtype).contiguous()
        b2 = torch.zeros((self.down_u_proj.out_features,), device=device, dtype=dtype)
        self._dual_split_kernel_cache[key] = {
            "versions": versions,
            "gate_u": gate_u,
            "up_u": up_u,
            "down_v": down_v,
            "down_u": down_u,
            "b2": b2,
        }
        return gate_u, up_u, down_v, down_u, b2

    def _get_dual_split_cublas_factors(self, device: torch.device, dtype: torch.dtype):
        self._ensure_runtime_state()
        key = (device.type, device.index, dtype)
        versions = (
            self.up_v_proj.weight._version,
            self.gate_v_proj.weight._version,
        )
        cached = self._dual_split_cublas_cache.get(key)
        if cached is not None and cached["versions"] == versions:
            return cached["v_cat"]

        v_cat = torch.cat(
            [
                self.up_v_proj.weight.to(device=device, dtype=dtype),
                self.gate_v_proj.weight.to(device=device, dtype=dtype),
            ],
            dim=0,
        ).contiguous()
        self._dual_split_cublas_cache[key] = {
            "versions": versions,
            "v_cat": v_cat,
        }
        return v_cat

    def _get_dual_split_workspace(self, device: torch.device, T: int, R: int) -> torch.Tensor:
        self._ensure_runtime_state()
        key = (device.type, device.index, int(T), int(R))
        cached = self._dual_split_workspace_cache.get(key)
        if cached is not None:
            return cached
        buf = torch.empty((T, R), device=device, dtype=torch.float32)
        self._dual_split_workspace_cache[key] = buf
        return buf

    def _prefer_token_decode(self, x: torch.Tensor) -> bool:
        self._ensure_runtime_state()
        return bool(
            x.is_cuda
            and x.shape[0] <= 4
            and x.shape[1] <= 4
            and os.getenv("SVDLLM_FLASH_FALLBACK", "0") == "0"
            and os.getenv("FLASH_SVD_DISABLE_FFN", "0") == "0"
        )

    def _forward_dual_split_cublas_packed(self, x: torch.Tensor) -> torch.Tensor:
        v_cat = self._get_dual_split_cublas_factors(x.device, x.dtype)
        p_cat = F.linear(x, v_cat)
        r = self.up_v_proj.out_features
        p_up, p_gate = p_cat.split((r, r), dim=-1)
        gate = self.gate_u_proj(p_gate)
        up = self.up_u_proj(p_up)
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))

    def _flashsvd_versions(self):
        self._ensure_runtime_state()
        return (
            self.up_v_proj.weight._version,
            self.gate_v_proj.weight._version,
            self.up_u_proj.weight._version,
            self.gate_u_proj.weight._version,
            self.down_v_proj.weight._version,
            self.down_u_proj.weight._version,
        )

    def _forward_flashsvd_core(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        backend = _flashsvd_ffn_backend()
        experimental_ok = _flashsvd_experimental_ffn_enabled()
        use_token_decode = self._prefer_token_decode(x)
        if backend == "dual_split_cublas":
            if experimental_ok:
                return self._forward_dual_split_cublas_packed(x)
            backend = "dual_split_cublas_legacy"
        if backend in {"dual_split_kernel", "dual_split_kernel_v2", "dual_split_kernel_v2_sm80", "dual_split_kernel_v3"}:
            if not experimental_ok:
                global _EXPERIMENTAL_FFN_WARNED
                if not _EXPERIMENTAL_FFN_WARNED:
                    print(
                        "[FlashSVD] Experimental FFN kernel backend requested but "
                        "FLASH_SVD_ENABLE_EXPERIMENTAL_FFN=0; falling back to dual_split_cublas_legacy "
                        "to preserve token-identical decode."
                    )
                    _EXPERIMENTAL_FFN_WARNED = True
                backend = "dual_split_cublas_legacy"
        if backend == "auto":
            backend = "dual_split_cublas_legacy" if use_token_decode else "generic"
        if backend == "dual_split_cublas":
            return self._forward_dual_split_cublas_packed(x)
        if backend == "dual_split_cublas_legacy":
            p_up = self.up_v_proj(x)
            p_gate = self.gate_v_proj(x)
            gate = self.gate_u_proj(p_gate)
            up = self.up_u_proj(p_up)
            return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
        if backend == "dual_split_kernel":
            if use_token_decode:
                v_cat = self._get_dual_split_cublas_factors(x.device, x.dtype)
                p_cat = F.linear(x, v_cat)
                r = self.up_v_proj.out_features
                p_up, p_gate = p_cat.split((r, r), dim=-1)
                gate_u, up_u, down_v, down_u, kernel_b2 = self._get_dual_split_kernel_factors(x.device, x.dtype)
                return flashsvd_ffn_dual_split_token(p_up, p_gate, gate_u, up_u, down_v, down_u, kernel_b2)
            p_up = self.up_v_proj(x)
            p_gate = self.gate_v_proj(x)
            gate = self.gate_u_proj(p_gate)
            up = self.up_u_proj(p_up)
            return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
        if backend == "dual_split_kernel_v2":
            if use_token_decode:
                v_cat = self._get_dual_split_cublas_factors(x.device, x.dtype)
                p_cat = F.linear(x, v_cat)
                r = self.up_v_proj.out_features
                p_up, p_gate = p_cat.split((r, r), dim=-1)
                gate_u, up_u, down_v, down_u, kernel_b2 = self._get_dual_split_kernel_factors(x.device, x.dtype)
                workspace_s2d = self._get_dual_split_workspace(x.device, int(p_up.shape[0] * p_up.shape[1]), int(r))
                return flashsvd_ffn_dual_split_token_v2(
                    p_up,
                    p_gate,
                    gate_u,
                    up_u,
                    down_v,
                    down_u,
                    kernel_b2,
                    workspace_s2d=workspace_s2d,
                )
            p_up = self.up_v_proj(x)
            p_gate = self.gate_v_proj(x)
            gate = self.gate_u_proj(p_gate)
            up = self.up_u_proj(p_up)
            return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
        if backend == "dual_split_kernel_v2_sm80":
            if use_token_decode:
                v_cat = self._get_dual_split_cublas_factors(x.device, x.dtype)
                p_cat = F.linear(x, v_cat)
                r = self.up_v_proj.out_features
                p_up, p_gate = p_cat.split((r, r), dim=-1)
                gate_u, up_u, down_v, down_u, kernel_b2 = self._get_dual_split_kernel_factors(x.device, x.dtype)
                workspace_s2d = self._get_dual_split_workspace(x.device, int(p_up.shape[0] * p_up.shape[1]), int(r))
                return flashsvd_ffn_dual_split_token_v2_sm80(
                    p_up,
                    p_gate,
                    gate_u,
                    up_u,
                    down_v,
                    down_u,
                    kernel_b2,
                    workspace_s2d=workspace_s2d,
                )
            p_up = self.up_v_proj(x)
            p_gate = self.gate_v_proj(x)
            gate = self.gate_u_proj(p_gate)
            up = self.up_u_proj(p_up)
            return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
        if backend == "dual_split_kernel_v3":
            if use_token_decode:
                v_cat = self._get_dual_split_cublas_factors(x.device, x.dtype)
                p_cat = F.linear(x, v_cat)
                r = self.up_v_proj.out_features
                p_up, p_gate = p_cat.split((r, r), dim=-1)
                gate_u, up_u, down_v, down_u, kernel_b2 = self._get_dual_split_kernel_factors(x.device, x.dtype)
                workspace_y = self._get_dual_split_workspace(x.device, int(p_up.shape[0] * p_up.shape[1]), int(self.down_u_proj.out_features))
                return flashsvd_ffn_dual_split_token_v3(
                    p_up,
                    p_gate,
                    gate_u,
                    up_u,
                    down_v,
                    down_u,
                    kernel_b2,
                    workspace_y=workspace_y,
                )
            p_up = self.up_v_proj(x)
            p_gate = self.gate_v_proj(x)
            gate = self.gate_u_proj(p_gate)
            up = self.up_u_proj(p_up)
            return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
        if backend in {"auto", "generic"}:
            p_up = self.up_v_proj(x)
            p_gate = self.gate_v_proj(x)
            gate = self.gate_u_proj(p_gate)
            up = self.up_u_proj(p_up)
            return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
        p_up = self.up_v_proj(x)
        p_gate = self.gate_v_proj(x)
        gate = self.gate_u_proj(p_gate)
        up = self.up_u_proj(p_up)
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))

    def _get_flashsvd_graph(
        self,
        x: torch.Tensor,
    ):
        versions = self._flashsvd_versions()
        key = (x.device.type, x.device.index, x.dtype, tuple(x.shape), versions, _flashsvd_ffn_backend())
        cached = self._flashsvd_graph_cache.get(key)
        if cached is not None:
            return cached

        static_x = torch.empty_like(x)
        static_x.zero_()

        warm_stream = torch.cuda.Stream(device=x.device)
        current_stream = torch.cuda.current_stream(device=x.device)
        warm_stream.wait_stream(current_stream)
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                Y = self._forward_flashsvd_core(static_x)
                _ = Y.reshape(-1)[0]
        current_stream.wait_stream(warm_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            Y_static = self._forward_flashsvd_core(static_x)

        cached = {
            "x": static_x,
            "y": Y_static,
            "graph": graph,
        }
        self._flashsvd_graph_cache[key] = cached
        return cached

    def _use_flashsvd_layer_tail_cuda_graph(self, x: torch.Tensor) -> bool:
        return (
            x.is_cuda
            and (not self.training)
            and (not torch.is_grad_enabled())
            and _flashsvd_cuda_graph_enabled()
            and _flashsvd_cuda_graph_scope() in {"auto", "layer_tail"}
            and x.shape[0] <= 4
            and x.shape[1] <= 4
        )

    def _get_flashsvd_layer_tail_graph(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ):
        versions = self._flashsvd_versions() + (norm_weight._version,)
        key = (x.device.type, x.device.index, x.dtype, tuple(x.shape), float(norm_eps), versions, _flashsvd_ffn_backend())
        cached = self._flashsvd_layer_graph_cache.get(key)
        if cached is not None:
            return cached

        static_x = torch.empty_like(x)
        static_x.zero_()

        warm_stream = torch.cuda.Stream(device=x.device)
        current_stream = torch.cuda.current_stream(device=x.device)
        warm_stream.wait_stream(current_stream)
        with torch.cuda.stream(warm_stream):
            for _ in range(3):
                normed = _llama_rms_norm(static_x, norm_weight, norm_eps)
                Y = static_x + self._forward_flashsvd_core(normed)
                _ = Y.reshape(-1)[0]
        current_stream.wait_stream(warm_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            normed_static = _llama_rms_norm(static_x, norm_weight, norm_eps)
            Y_static = static_x + self._forward_flashsvd_core(normed_static)

        cached = {
            "x": static_x,
            "y": Y_static,
            "graph": graph,
        }
        self._flashsvd_layer_graph_cache[key] = cached
        return cached

    def flashsvd_post_attention_tail(
        self,
        hidden_states: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        *,
        use_cuda_graph: bool | None = None,
    ) -> torch.Tensor:
        if (
            (not hidden_states.is_cuda)
            or os.getenv("SVDLLM_FLASH_FALLBACK", "0") != "0"
            or os.getenv("FLASH_SVD_DISABLE_FFN", "0") != "0"
        ):
            residual = hidden_states
            normed = _llama_rms_norm(hidden_states, norm_weight, norm_eps)
            return residual + self.forward(normed)

        if use_cuda_graph is None:
            use_cuda_graph = self._use_flashsvd_layer_tail_cuda_graph(hidden_states)
        if use_cuda_graph:
            bundle = self._get_flashsvd_layer_tail_graph(
                hidden_states,
                norm_weight,
                norm_eps,
            )
            bundle["x"].copy_(hidden_states)
            bundle["graph"].replay()
            if _flashsvd_cuda_graph_alias_output():
                return bundle["y"]
            return bundle["y"].clone()

        normed = _llama_rms_norm(hidden_states, norm_weight, norm_eps)
        return hidden_states + self._forward_flashsvd_core(normed)

    def forward(self, x):
        self._ensure_runtime_state()
        # Fast path: Triton FlashSVD SwiGLU on CUDA using shared rank-space P
        if (
            x.is_cuda
            and os.getenv("SVDLLM_FLASH_FALLBACK", "0") == "0"
            and os.getenv("FLASH_SVD_DISABLE_FFN", "0") == "0"
        ):
            if self._use_flashsvd_cuda_graph(x):
                bundle = self._get_flashsvd_graph(x)
                bundle["x"].copy_(x)
                bundle["graph"].replay()
                if _flashsvd_cuda_graph_alias_output():
                    return bundle["y"]
                return bundle["y"].clone()

            return self._forward_flashsvd_core(x)

        # Fallback (CPU or non-CUDA): baseline low-rank SwiGLU
        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))


class SVD_LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, ratio=1):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # HF-compatible attributes used by attention helper fns (repeat_kv for GQA).
        self.num_key_value_heads = int(getattr(config, "num_key_value_heads", self.num_heads) or self.num_heads)
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads {self.num_heads} must be divisible by num_key_value_heads {self.num_key_value_heads}")
        self.num_key_value_groups = int(self.num_heads // self.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.ratio = float(ratio)
        self.use_lowrank_cache = bool(self.ratio < 1.0)
        setattr(
            self.config,
            "flash_svd_use_lowrank_cache",
            bool(getattr(self.config, "flash_svd_use_lowrank_cache", False) or self.use_lowrank_cache),
        )

        # HF LlamaAttention uses `layer_idx` to index into Cache objects. Preserve it for KV cache support.
        self.layer_idx = int(getattr(config, "layer_idx", 0) or 0)
        self.is_causal = True
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.scaling = self.head_dim**-0.5

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        low_rank = max(1, int(self.hidden_size * self.ratio / 2))
        self.low_rank = low_rank
        if self.use_lowrank_cache:
            setattr(self.config, "flash_svd_lowrank_rank", int(low_rank))
        self.q_u_proj = nn.Linear(low_rank, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.k_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.v_u_proj = nn.Linear(low_rank, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(self.hidden_size, low_rank, bias=False)

        self.o_u_proj = nn.Linear(low_rank, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(self.num_heads * self.head_dim, low_rank, bias=False)

        rope_theta = float(getattr(config, "rope_theta", 10000.0))
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=rope_theta,
        )

        # Flash SVD + RoPE attention kernel wrapper
        self.flash_attn = FlashSVDRoPEAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rotary_emb=self.rotary_emb,
        )

        # Decode-kernel cached factors (computed lazily after weights are loaded).
        self._decode_Vq = None
        self._decode_Vk = None
        self._decode_Vv = None
        self._decode_ptrs = None
        self._dense_decode_prepacked_cache = {}

    def _ensure_runtime_state(self) -> None:
        # Older pickled checkpoints may lack attrs introduced after serialization.
        if not hasattr(self, "config"):
            raise AttributeError("SVD_LlamaAttention checkpoint is missing `config`.")
        if not hasattr(self, "hidden_size"):
            self.hidden_size = int(getattr(self.config, "hidden_size"))
        if not hasattr(self, "num_heads"):
            self.num_heads = int(getattr(self.config, "num_attention_heads"))
        if not hasattr(self, "head_dim"):
            self.head_dim = int(self.hidden_size // self.num_heads)
        if not hasattr(self, "num_key_value_heads"):
            kv_heads = int(getattr(self.config, "num_key_value_heads", self.num_heads) or self.num_heads)
            try:
                out0 = int(self.k_u_proj.weight.shape[0])
                if self.head_dim > 0 and out0 % self.head_dim == 0:
                    kv_heads = int(out0 // self.head_dim)
            except Exception:
                pass
            self.num_key_value_heads = kv_heads
        if not hasattr(self, "num_key_value_groups"):
            self.num_key_value_groups = int(self.num_heads // max(1, self.num_key_value_heads))
        if not hasattr(self, "max_position_embeddings"):
            self.max_position_embeddings = int(getattr(self.config, "max_position_embeddings", 2048))
        if not hasattr(self, "ratio"):
            self.ratio = float(getattr(self, "ratio", 1.0))
        if not hasattr(self, "use_lowrank_cache"):
            self.use_lowrank_cache = bool(float(self.ratio) < 1.0)
        if not hasattr(self, "layer_idx"):
            self.layer_idx = int(getattr(self.config, "layer_idx", 0) or 0)
        if not hasattr(self, "is_causal"):
            self.is_causal = True
        if not hasattr(self, "attention_dropout"):
            self.attention_dropout = float(getattr(self.config, "attention_dropout", 0.0))
        if not hasattr(self, "scaling"):
            self.scaling = self.head_dim**-0.5
        if not hasattr(self, "low_rank"):
            self.low_rank = int(getattr(self.q_v_proj, "out_features", 0) or 0)
        if not hasattr(self, "rotary_emb"):
            rope_theta = float(getattr(self.config, "rope_theta", 10000.0))
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=rope_theta,
            )
        if not hasattr(self, "flash_attn"):
            self.flash_attn = FlashSVDRoPEAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                rotary_emb=self.rotary_emb,
            )
        if not hasattr(self, "_decode_Vq"):
            self._decode_Vq = None
        if not hasattr(self, "_decode_Vk"):
            self._decode_Vk = None
        if not hasattr(self, "_decode_Vv"):
            self._decode_Vv = None
        if not hasattr(self, "_decode_ptrs"):
            self._decode_ptrs = None
        if not hasattr(self, "_dense_decode_prepacked_cache"):
            self._dense_decode_prepacked_cache = {}

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        self._ensure_runtime_state()
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _apply_rope_hf(
        self,
        q_bhsd: torch.Tensor,
        k_bhsd: torch.Tensor,
        cos_bsd: torch.Tensor,
        sin_bsd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos_bsd.unsqueeze(1)
        sin = sin_bsd.unsqueeze(1)
        q_embed = (q_bhsd * cos) + (rotate_half(q_bhsd) * sin)
        k_embed = (k_bhsd * cos) + (rotate_half(k_bhsd) * sin)
        return q_embed, k_embed

    def _get_decode_factors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return contiguous (Vq, Vk, Vv) for decode kernels.

        Shapes:
          Vq: [H,  R, Dh]
          Vk: [Hk, R, Dh]
          Vv: [Hk, R, Dh]
        """
        self._ensure_runtime_state()
        # Backward-compat for older pickled checkpoints: these attrs may be absent.
        if not hasattr(self, "_decode_ptrs"):
            self._decode_ptrs = None  # type: ignore[assignment]
        if not hasattr(self, "_decode_Vq"):
            self._decode_Vq = None  # type: ignore[assignment]
        if not hasattr(self, "_decode_Vk"):
            self._decode_Vk = None  # type: ignore[assignment]
        if not hasattr(self, "_decode_Vv"):
            self._decode_Vv = None  # type: ignore[assignment]

        H, dh = int(self.num_heads), int(self.head_dim)
        Hk_attr = int(getattr(self, "num_key_value_heads", H) or H)
        R = int(self.q_v_proj.out_features)

        # Infer Hk from weight shapes (preferred) to support checkpoints that
        # preserve true GQA (k/v have Hk heads) as well as older MHA-style ones.
        try:
            k0 = int(self.k_u_proj.weight.shape[0])
            v0 = int(self.v_u_proj.weight.shape[0])
            if k0 % dh != 0 or v0 % dh != 0:
                raise ValueError("k_u_proj/v_u_proj out_features not divisible by head_dim")
            Hk_w = k0 // dh
            Hk_vw = v0 // dh
            if Hk_w == Hk_vw:
                Hk = int(Hk_w)
            elif Hk_attr in (Hk_w, Hk_vw):
                Hk = int(Hk_attr)
            else:
                # Best-effort fallback for odd checkpoints; prefer the smaller head count.
                Hk = int(min(Hk_w, Hk_vw))
        except Exception:
            Hk = Hk_attr
        Hk = int(Hk or Hk_attr or H)

        ptrs = (
            int(self.q_u_proj.weight.data_ptr()),
            int(self.k_u_proj.weight.data_ptr()),
            int(self.v_u_proj.weight.data_ptr()),
            int(self.q_u_proj.weight.shape[0]),
            int(self.q_u_proj.weight.shape[1]),
            int(self.k_u_proj.weight.shape[0]),
            int(self.k_u_proj.weight.shape[1]),
            int(self.v_u_proj.weight.shape[0]),
            int(self.v_u_proj.weight.shape[1]),
            str(self.q_u_proj.weight.dtype),
            str(self.q_u_proj.weight.device),
        )
        if self._decode_ptrs != ptrs or self._decode_Vq is None or self._decode_Vk is None or self._decode_Vv is None:
            self._decode_Vq = self.q_u_proj.weight.view(H, dh, R).permute(0, 2, 1).contiguous()
            self._decode_Vk = self.k_u_proj.weight.view(Hk, dh, R).permute(0, 2, 1).contiguous()
            self._decode_Vv = self.v_u_proj.weight.view(Hk, dh, R).permute(0, 2, 1).contiguous()
            self._decode_ptrs = ptrs
        return self._decode_Vq, self._decode_Vk, self._decode_Vv

    def _project_rank_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to rank-space factors.

        For Wk=UkVk and Wv=UvVv, the cache stores XUk and XUv (pre-RoPE).
        """
        self._ensure_runtime_state()
        Pq_rank = self.q_v_proj(hidden_states)
        Pk_rank = self.k_v_proj(hidden_states)
        Pv_rank = self.v_v_proj(hidden_states)
        return Pq_rank, Pk_rank, Pv_rank

    def _get_dense_decode_tensors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self._ensure_runtime_state()
        dense_decode_mod = _get_dense_token_decode_mod()
        key = (device.type, device.index, dtype)
        ptrs = (
            int(self.q_v_proj.weight.data_ptr()),
            int(self.k_v_proj.weight.data_ptr()),
            int(self.v_v_proj.weight.data_ptr()),
            int(self.q_u_proj.weight.data_ptr()),
            int(self.k_u_proj.weight.data_ptr()),
            int(self.v_u_proj.weight.data_ptr()),
            int(self.q_v_proj.weight._version),
            int(self.k_v_proj.weight._version),
            int(self.v_v_proj.weight._version),
            int(self.q_u_proj.weight._version),
            int(self.k_u_proj.weight._version),
            int(self.v_u_proj.weight._version),
        )
        cached = self._dense_decode_prepacked_cache.get(key)
        if cached is not None and cached["ptrs"] == ptrs:
            return cached["packed_qkv_rank"], cached["vq_flat"], cached["vk_flat"], cached["vv_flat"]

        packed_qkv_rank = torch.cat(
            [
                self.q_v_proj.weight.t(),
                self.k_v_proj.weight.t(),
                self.v_v_proj.weight.t(),
            ],
            dim=1,
        ).to(device=device, dtype=dtype).contiguous()
        vq, vk, vv = self._get_decode_factors()
        vq = vq.to(device=device, dtype=dtype).contiguous()
        vk = vk.to(device=device, dtype=dtype).contiguous()
        vv = vv.to(device=device, dtype=dtype).contiguous()
        vq_flat, vk_flat, vv_flat = dense_decode_mod.pack_qkv_shared_bases(vq, vk, vv)

        self._dense_decode_prepacked_cache[key] = {
            "ptrs": ptrs,
            "packed_qkv_rank": packed_qkv_rank,
            "vq_flat": vq_flat,
            "vk_flat": vk_flat,
            "vv_flat": vv_flat,
        }
        return packed_qkv_rank, vq_flat, vk_flat, vv_flat

    def _can_use_flashsvd_dense_decode_graph(self, hidden_states: torch.Tensor, past_key_value) -> bool:
        DenseKVCache = _get_dense_cache_cls()
        flash_attn_with_kvcache = _get_flash_attn_with_kvcache()
        return bool(
            past_key_value is not None
            and DenseKVCache is not None
            and isinstance(past_key_value, DenseKVCache)
            and hidden_states.is_cuda
            and torch.cuda.is_available()
            and (not self.training)
            and _flashsvd_dense_attn_enabled()
            and (not _flashsvd_baseline_dense_kvcache_enabled())
            and flash_attn_with_kvcache is not None
            and int(hidden_states.shape[1]) == 1
            and int(self.q_v_proj.out_features) == int(self.k_v_proj.out_features) == int(self.v_v_proj.out_features)
        )

    @staticmethod
    def _decode_positions_from_cache(
        *,
        q_len: int,
        device: torch.device,
        past_key_value,
        cache_position: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        q_len = int(q_len)
        if cache_position is not None:
            pos = cache_position.to(device=device, dtype=torch.long).reshape(-1)
            if int(pos.numel()) == q_len:
                return pos
            if int(pos.numel()) == 1:
                if q_len == 1:
                    return pos
                return pos[0] + torch.arange(q_len, device=device, dtype=torch.long)
            return pos[:q_len]

        start = int(past_key_value.get_seq_length())
        return torch.arange(start, start + q_len, device=device, dtype=torch.long)

    def _flashsvd_dense_cache_attend_fa2_internal_rope(
        self,
        q_bmhd: torch.Tensor,
        k_bmhd: torch.Tensor,
        v_bmhd: torch.Tensor,
        past_key_value,
        *,
        cache_position: Optional[torch.LongTensor],
        cache_bindings: Optional[dict[str, torch.Tensor | object]] = None,
        advance_cache: bool,
    ) -> torch.Tensor:
        q_len = int(q_bmhd.shape[1])
        if cache_bindings is None:
            flash_attn_with_kvcache = _get_flash_attn_with_kvcache()
            seqlen_k = int(past_key_value.get_seq_length())
            smax = int(past_key_value.get_max_cache_shape() or max(seqlen_k + q_len, 1))
            k_cache_bmhd, v_cache_bmhd, cache_seqlens = past_key_value.prepare_fa2_step(
                int(self.layer_idx),
                batch_size=int(q_bmhd.shape[0]),
                cache_position=cache_position,
            )
            rotary_cos, rotary_sin = past_key_value.get_rope_tables(
                seqlen=smax,
                head_dim=self.head_dim,
                device=q_bmhd.device,
                dtype=q_bmhd.dtype,
            )
        else:
            flash_attn_with_kvcache = cache_bindings["flash_attn_with_kvcache"]
            k_cache_bmhd = cache_bindings["k_cache_bmhd"]
            v_cache_bmhd = cache_bindings["v_cache_bmhd"]
            cache_seqlens = cache_bindings["cache_seqlens"]
            rotary_cos = cache_bindings["rotary_cos"]
            rotary_sin = cache_bindings["rotary_sin"]

        out = _call_flash_attn_with_kvcache(
            flash_attn_with_kvcache,
            q_bmhd,
            k_cache_bmhd,
            v_cache_bmhd,
            k_bmhd=k_bmhd,
            v_bmhd=v_bmhd,
            cache_seqlens=cache_seqlens,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            causal=True,
        )
        if advance_cache:
            past_key_value.advance_after_fa2(
                int(self.layer_idx),
                q_len=q_len,
                cache_position=cache_position,
            )
        if out.shape == (int(q_bmhd.shape[0]), q_len, self.num_heads, self.head_dim):
            out_dense = out.reshape(int(q_bmhd.shape[0]), q_len, self.num_heads * self.head_dim).contiguous()
        elif out.shape == (int(q_bmhd.shape[0]), self.num_heads, q_len, self.head_dim):
            out_dense = out.transpose(1, 2).reshape(int(q_bmhd.shape[0]), q_len, self.num_heads * self.head_dim).contiguous()
        else:
            raise ValueError(f"Unexpected flash_attn_with_kvcache output shape: {tuple(out.shape)}")
        return self.o_u_proj(self.o_v_proj(out_dense))

    def _dense_cache_baseline_attend_post_rope(
        self,
        q_bmhd: torch.Tensor,
        k_bmhd: torch.Tensor,
        v_bmhd: torch.Tensor,
        past_key_value,
        *,
        cache_position: Optional[torch.LongTensor],
        cache_bindings: Optional[dict[str, torch.Tensor | object]] = None,
    ) -> torch.Tensor:
        q_len = int(q_bmhd.shape[1])
        smax = int(past_key_value.get_max_cache_shape() or max(int(past_key_value.get_seq_length()) + q_len, 1))
        rotary_cos, rotary_sin = past_key_value.get_rope_tables(
            seqlen=smax,
            head_dim=self.head_dim,
            device=q_bmhd.device,
            dtype=q_bmhd.dtype,
        )
        pos = self._decode_positions_from_cache(
            q_len=q_len,
            device=q_bmhd.device,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        cos = rotary_cos.index_select(0, pos).view(1, q_len, 1, self.head_dim // 2)
        sin = rotary_sin.index_select(0, pos).view(1, q_len, 1, self.head_dim // 2)
        q_bmhd = self._apply_rope_tables(q_bmhd, cos, sin)
        k_bmhd = self._apply_rope_tables(k_bmhd, cos, sin)

        past_key_value.update(
            k_bmhd.transpose(1, 2).contiguous(),
            v_bmhd.transpose(1, 2).contiguous(),
            int(self.layer_idx),
            {"cache_position": cache_position},
        )

        flash_attn_with_kvcache = _get_flash_attn_with_kvcache()
        if cache_bindings is not None:
            flash_attn_with_kvcache = cache_bindings.get("flash_attn_with_kvcache", flash_attn_with_kvcache)
        if flash_attn_with_kvcache is None:
            raise RuntimeError("flash_attn_with_kvcache is required for DenseKVCache decode.")

        k_cache_bmhd, v_cache_bmhd, cache_seqlens = past_key_value.prepare_fa2_step(
            int(self.layer_idx),
            batch_size=int(q_bmhd.shape[0]),
            cache_position=None,
        )
        out = _call_flash_attn_with_kvcache(
            flash_attn_with_kvcache,
            q_bmhd,
            k_cache_bmhd,
            v_cache_bmhd,
            cache_seqlens=cache_seqlens,
            causal=True,
        )
        if out.shape == (int(q_bmhd.shape[0]), q_len, self.num_heads, self.head_dim):
            out_dense = out.reshape(int(q_bmhd.shape[0]), q_len, self.num_heads * self.head_dim).contiguous()
        elif out.shape == (int(q_bmhd.shape[0]), self.num_heads, q_len, self.head_dim):
            out_dense = out.transpose(1, 2).reshape(int(q_bmhd.shape[0]), q_len, self.num_heads * self.head_dim).contiguous()
        else:
            raise ValueError(f"Unexpected flash_attn_with_kvcache output shape: {tuple(out.shape)}")
        return self.o_u_proj(self.o_v_proj(out_dense))

    def _flashsvd_dense_decode_token_from_hidden(
        self,
        hidden_states: torch.Tensor,
        past_key_value,
        *,
        cache_position: Optional[torch.LongTensor],
        cache_bindings: Optional[dict[str, torch.Tensor | object]] = None,
        advance_cache: bool,
    ) -> torch.Tensor:
        dense_decode_mod = _get_dense_token_decode_mod()
        packed_qkv_rank, vq_flat, vk_flat, vv_flat = self._get_dense_decode_tensors(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        hidden_flat = hidden_states[:, 0, :].contiguous()
        packed_rank = torch.matmul(hidden_flat, packed_qkv_rank)
        rank = int(self.q_v_proj.out_features)
        q_rank, k_rank, v_rank = torch.split(packed_rank, rank, dim=1)
        q_bhd, k_bkd, v_bkd = dense_decode_mod.reconstruct_qkv_token_shared_prepacked(
            q_rank,
            k_rank,
            v_rank,
            vq_flat,
            vk_flat,
            vv_flat,
            H=self.num_heads,
            Hk=self.num_key_value_heads,
            Dh=self.head_dim,
        )
        q_bmhd = q_bhd[:, None, :, :].contiguous()
        k_bmhd = k_bkd[:, None, :, :].contiguous()
        v_bmhd = v_bkd[:, None, :, :].contiguous()
        return self._flashsvd_dense_cache_attend_fa2_internal_rope(
            q_bmhd,
            k_bmhd,
            v_bmhd,
            past_key_value,
            cache_position=cache_position,
            cache_bindings=cache_bindings,
            advance_cache=advance_cache,
        )

    def _baseline_dense_decode_token_from_hidden(
        self,
        hidden_states: torch.Tensor,
        past_key_value,
        *,
        cache_position: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        Pq_rank, Pk_rank, Pv_rank = self._project_rank_qkv(hidden_states)
        q_bmhd = self.q_u_proj(Pq_rank).view(hidden_shape)
        k_bmhd = self.k_u_proj(Pk_rank).view(hidden_shape)
        v_bmhd = self.v_u_proj(Pv_rank).view(hidden_shape)
        return self._dense_cache_baseline_attend_post_rope(
            q_bmhd,
            k_bmhd,
            v_bmhd,
            past_key_value,
            cache_position=cache_position,
            cache_bindings=None,
        )

    def _update_lowrank_kv_cache(
        self,
        past_key_value,
        Pk_rank: torch.Tensor,
        Pv_rank: torch.Tensor,
        cache_position: Optional[torch.LongTensor],
    ) -> None:
        past_key_value.update(Pk_rank, Pv_rank, int(self.layer_idx), {"cache_position": cache_position})

    @staticmethod
    def _apply_rope_tables(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x0 = x[..., :half]
        x1 = x[..., half:]
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1)

    @staticmethod
    def _normalize_bn(split_k: int, bn: int, bn_max: int) -> int:
        split_k = max(1, int(split_k))
        bn = max(16, min(int(bn), int(bn_max), split_k))
        bn = 1 << (int(bn).bit_length() - 1)
        while bn > 16 and (split_k % bn) != 0:
            bn //= 2
        return max(16, bn)

    @staticmethod
    def _parse_decode_variant_map(raw: str) -> dict[int, str]:
        table: dict[int, str] = {}
        text = str(raw or "").strip()
        if not text:
            return table
        for item in text.split(","):
            tok = item.strip()
            if not tok or ":" not in tok:
                continue
            b_s, v_s = tok.split(":", 1)
            try:
                b = int(b_s.strip())
            except Exception:
                continue
            if b <= 0:
                continue
            table[b] = _canonical_decode_variant(v_s.strip())
        return table

    def _select_decode_variant(
        self,
        *,
        mods,
        hidden_states: torch.Tensor,
        bsz: int,
        H: int,
        Hk: int,
        R: int,
        Dh: int,
        seqlen_k: int,
        Pq_q: torch.Tensor,
        Pk: torch.Tensor,
        Pv: torch.Tensor,
        Vq: torch.Tensor,
        Vk: torch.Tensor,
        Vv: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        call_kwargs: dict[str, object],
    ) -> str:
        global _DECODE_VARIANT_CACHE
        global _DECODE_VARIANT_LOGGED

        available = _available_decode_variants(mods)
        if not available:
            return "v16_v2"

        requested = _canonical_decode_variant(os.getenv("FLASH_SVD_DECODE_KERNEL_VARIANT", "auto"))
        if requested != "auto":
            if requested in available:
                return requested
            warn_key = ("requested_missing", requested, tuple(available))
            if warn_key not in _DECODE_VARIANT_LOGGED:
                print(
                    "[FlashSVD][decode_variant] requested "
                    f"{requested} not available; fallback to auto (available={','.join(available)})"
                )
                _DECODE_VARIANT_LOGGED.add(warn_key)
            requested = "auto"

        # Optional fixed batch-size policy, e.g.:
        # FLASH_SVD_DECODE_KERNEL_MAP="1:v16_v1,2:v16_v1,8:v15,16:v16_v1"
        variant_map = self._parse_decode_variant_map(os.getenv("FLASH_SVD_DECODE_KERNEL_MAP", ""))
        if requested == "auto" and bsz in variant_map and variant_map[bsz] in available:
            return variant_map[bsz]

        include_v2 = os.getenv("FLASH_SVD_DECODE_AUTO_INCLUDE_V2", "0") != "0"
        candidates = [v for v in ("v15", "v16_v1") if v in available]
        if include_v2 and "v16_v2" in available:
            candidates.append("v16_v2")
        if not candidates:
            candidates = list(available)
        if len(candidates) == 1:
            return candidates[0]

        # Heuristic default before any online tuning.
        # Large-rank REP==1 decode often favors the v1.5 path in end-to-end serving.
        if int(R) >= 768 and "v15" in candidates:
            default_variant = "v15"
        elif "v16_v1" in candidates:
            default_variant = "v16_v1"
        else:
            default_variant = candidates[0]

        # Keep online variant autotune opt-in for serving stability.
        if os.getenv("FLASH_SVD_DECODE_KERNEL_AUTOTUNE", "0") == "0":
            return default_variant

        try:
            tune_warmup = max(1, int(os.getenv("FLASH_SVD_DECODE_KERNEL_AUTOTUNE_WARMUP", "1")))
        except Exception:
            tune_warmup = 1
        try:
            tune_iters = max(1, int(os.getenv("FLASH_SVD_DECODE_KERNEL_AUTOTUNE_ITERS", "2")))
        except Exception:
            tune_iters = 2

        split_key = int(call_kwargs.get("split_k", 0) or 0)
        bn_key = int(call_kwargs.get("bn", 0) or 0)
        br_key = int(call_kwargs.get("br", 0) or 0)
        pad_key = int(bool(call_kwargs.get("pad_to_16", False)))
        vk_key = int(bool(call_kwargs.get("vk_resident", False)))
        # Variant preference should be stable across a request; avoid retuning on
        # every seqlen bucket transition.
        cache_key = (
            int(getattr(hidden_states.device, "index", -1) or -1),
            str(hidden_states.dtype),
            int(bsz),
            int(H),
            int(Hk),
            int(R),
            int(Dh),
            int(split_key),
            int(bn_key),
            int(br_key),
            int(pad_key),
            int(vk_key),
        )
        cached = _DECODE_VARIANT_CACHE.get(cache_key, None)
        if cached in candidates:
            return str(cached)

        best_variant = default_variant
        best_ms = float("inf")

        for variant in candidates:
            mod_v, fn_v = _resolve_decode_variant(mods, variant)
            if mod_v is None or fn_v is None:
                continue
            try:
                f_v = mod_v.DecodePackedFactors(Pq=Pq_q, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)
                kw_v = _maybe_kwargs(fn_v, dict(call_kwargs))
                for _ in range(tune_warmup):
                    fn_v(f_v, rotary_cos, rotary_sin, **kw_v)
                torch.cuda.synchronize(hidden_states.device)
                ev_s = torch.cuda.Event(enable_timing=True)
                ev_e = torch.cuda.Event(enable_timing=True)
                ev_s.record()
                for _ in range(tune_iters):
                    fn_v(f_v, rotary_cos, rotary_sin, **kw_v)
                ev_e.record()
                torch.cuda.synchronize(hidden_states.device)
                ms = float(ev_s.elapsed_time(ev_e)) / float(tune_iters)
            except Exception:
                continue
            if ms < best_ms:
                best_ms = ms
                best_variant = variant

        _DECODE_VARIANT_CACHE[cache_key] = best_variant
        log_key = ("autotuned", cache_key)
        if log_key not in _DECODE_VARIANT_LOGGED:
            extra = f"{best_ms:.3f} ms" if math.isfinite(best_ms) else "n/a"
            print(
                "[FlashSVD][decode_variant] "
                f"B={bsz} H={H} Hk={Hk} R={R} Dh={Dh} "
                f"selected={best_variant} "
                f"candidates={','.join(candidates)} ({extra})"
            )
            _DECODE_VARIANT_LOGGED.add(log_key)
        return best_variant

    def _autotune_fused_decode_cfg(
        self,
        *,
        mod,
        f,
        past_key_value,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        hidden_states: torch.Tensor,
        bsz: int,
        H: int,
        Hk: int,
        R: int,
        Dh: int,
        Smax: int,
        seqlen_k: int,
        split_k: int,
        bn: int,
        br: int,
        bn_max: int,
        num_warps_stage1: int,
        num_stages_stage1: int,
        num_warps_stage2: int,
        num_stages_stage2: int,
        pad_to_16: bool,
        vk_resident: bool,
    ) -> Tuple[int, int, int]:
        if os.getenv("FLASH_SVD_DECODE_AUTOTUNE", "1") == "0":
            return split_k, bn, br
        if any(
            os.getenv(name, "").strip()
            for name in ("FLASH_SVD_DECODE_SPLIT_K", "FLASH_SVD_DECODE_BN", "FLASH_SVD_DECODE_BR")
        ):
            return split_k, bn, br
        if not (hidden_states.is_cuda and torch.cuda.is_available()):
            return split_k, bn, br
        if int(os.getenv("FLASH_SVD_DECODE_MHA_STREAM", "0") != "0"):
            return split_k, bn, br

        global _DECODE_AUTOTUNE_CACHE
        global _DECODE_AUTOTUNE_LOGGED
        device = hidden_states.device
        key = (
            int(getattr(device, "index", -1) or -1),
            str(hidden_states.dtype),
            int(bsz),
            int(H),
            int(Hk),
            int(R),
            int(Dh),
            int(bool(pad_to_16)),
            int(bool(vk_resident)),
            int(num_warps_stage1),
            int(num_stages_stage1),
            int(num_warps_stage2),
            int(num_stages_stage2),
        )
        cached = _DECODE_AUTOTUNE_CACHE.get(key, None)
        if cached is not None:
            return int(cached[0]), int(cached[1]), int(cached[2])

        try:
            iters = max(1, int(os.getenv("FLASH_SVD_DECODE_AUTOTUNE_ITERS", "2")))
        except Exception:
            iters = 2
        try:
            warmup = max(1, int(os.getenv("FLASH_SVD_DECODE_AUTOTUNE_WARMUP", "1")))
        except Exception:
            warmup = 1

        split_candidates = [split_k]
        for cand in (512, 1024, 256):
            if cand not in split_candidates:
                split_candidates.append(cand)

        bn_candidates = [bn]
        bn_half = max(16, bn // 2)
        if bn_half not in bn_candidates:
            bn_candidates.append(bn_half)
        bn_up = min(split_k, max(16, bn * 2))
        if bn_up not in bn_candidates:
            bn_candidates.append(bn_up)

        br_candidates = [min(br, R)]
        if R >= 128 and 128 not in br_candidates:
            br_candidates.append(128)
        if R >= 64 and 64 not in br_candidates:
            br_candidates.append(64)

        candidates = []
        seen_cfg = set()
        for split_try in split_candidates:
            split_eff = max(1, int(split_try))
            bn_eff = self._normalize_bn(split_eff, bn, bn_max)
            br_eff = max(1, min(int(br), int(R)))
            cfg = (split_eff, bn_eff, br_eff)
            if cfg not in seen_cfg:
                candidates.append(cfg)
                seen_cfg.add(cfg)
        for bn_try in bn_candidates:
            split_eff = max(1, int(split_k))
            bn_eff = self._normalize_bn(split_eff, bn_try, bn_max)
            br_eff = max(1, min(int(br), int(R)))
            cfg = (split_eff, bn_eff, br_eff)
            if cfg not in seen_cfg:
                candidates.append(cfg)
                seen_cfg.add(cfg)
        for br_try in br_candidates:
            split_eff = max(1, int(split_k))
            bn_eff = self._normalize_bn(split_eff, bn, bn_max)
            br_eff = max(1, min(int(br_try), int(R)))
            cfg = (split_eff, bn_eff, br_eff)
            if cfg not in seen_cfg:
                candidates.append(cfg)
                seen_cfg.add(cfg)

        if len(candidates) > 6:
            candidates = candidates[:6]
        if len(candidates) <= 1:
            _DECODE_AUTOTUNE_CACHE[key] = candidates[0] if candidates else (split_k, bn, br)
            return _DECODE_AUTOTUNE_CACHE[key]

        out_buf = torch.empty((bsz, H, Dh), device=device, dtype=hidden_states.dtype)

        def _run_cfg(cfg: Tuple[int, int, int]) -> float:
            split_eff, bn_eff, br_eff = cfg
            max_splits_eff = max(1, (Smax + split_eff - 1) // split_eff)
            ws_eff = past_key_value.get_decode_workspace(
                batch_size=bsz,
                num_heads=H,
                rank=R,
                head_dim=Dh,
                max_splits=max_splits_eff,
                device=device,
                dtype=hidden_states.dtype,
            )
            num_splits_eff = max(1, (seqlen_k + split_eff - 1) // split_eff)
            workspace_eff = (
                ws_eff.M[:, :, :num_splits_eff],
                ws_eff.L[:, :, :num_splits_eff],
                ws_eff.Acc[:, :, :num_splits_eff, :],
            )
            q_buffers_eff = (ws_eff.Q0, ws_eff.Q1)

            for _ in range(warmup):
                mod.flashsvd_attn_decode_packed(
                    f,
                    rotary_cos,
                    rotary_sin,
                    seqlen_k=seqlen_k,
                    causal=True,
                    split_k=split_eff,
                    bn=bn_eff,
                    br=br_eff,
                    num_warps_stage1=num_warps_stage1,
                    num_stages_stage1=num_stages_stage1,
                    num_warps_stage2=num_warps_stage2,
                    num_stages_stage2=num_stages_stage2,
                    q_buffers=q_buffers_eff,
                    workspace=workspace_eff,
                    out=out_buf,
                    precompute_q=True,
                    writethrough=True,
                    pad_to_16=bool(pad_to_16),
                    vk_resident=bool(vk_resident),
                )
            torch.cuda.synchronize(device)
            ev_s = torch.cuda.Event(enable_timing=True)
            ev_e = torch.cuda.Event(enable_timing=True)
            ev_s.record()
            for _ in range(iters):
                mod.flashsvd_attn_decode_packed(
                    f,
                    rotary_cos,
                    rotary_sin,
                    seqlen_k=seqlen_k,
                    causal=True,
                    split_k=split_eff,
                    bn=bn_eff,
                    br=br_eff,
                    num_warps_stage1=num_warps_stage1,
                    num_stages_stage1=num_stages_stage1,
                    num_warps_stage2=num_warps_stage2,
                    num_stages_stage2=num_stages_stage2,
                    q_buffers=q_buffers_eff,
                    workspace=workspace_eff,
                    out=out_buf,
                    precompute_q=True,
                    writethrough=True,
                    pad_to_16=bool(pad_to_16),
                    vk_resident=bool(vk_resident),
                )
            ev_e.record()
            torch.cuda.synchronize(device)
            return float(ev_s.elapsed_time(ev_e)) / float(iters)

        best_cfg = candidates[0]
        best_ms = float("inf")
        for cfg in candidates:
            try:
                ms = _run_cfg(cfg)
            except Exception:
                continue
            if ms < best_ms:
                best_ms = ms
                best_cfg = cfg

        _DECODE_AUTOTUNE_CACHE[key] = best_cfg
        if key not in _DECODE_AUTOTUNE_LOGGED:
            print(
                "[FlashSVD][decode_autotune] "
                f"H={H} Hk={Hk} R={R} Dh={Dh} "
                f"split_k={best_cfg[0]} bn={best_cfg[1]} br={best_cfg[2]} "
                f"({best_ms:.3f} ms, {len(candidates)} cfgs)"
            )
            _DECODE_AUTOTUNE_LOGGED.add(key)
        return int(best_cfg[0]), int(best_cfg[1]), int(best_cfg[2])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # HF >=4.40 passes `past_key_values`; accept it for compatibility
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._ensure_runtime_state()
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values
        bsz, q_len, _ = hidden_states.size()
        scaling = float(getattr(self, "scaling", self.head_dim**-0.5))
        attention_dropout = float(getattr(self, "attention_dropout", 0.0))
        fix_pad_query_mask = bool(getattr(self, "fix_pad_query_mask", False))
        debug_pad_query_mask = bool(getattr(self, "debug_pad_query_mask", False))

        # KV-cache path: follow HF semantics (Cache.update happens inside attention) and return 2-tuple.
        # This enables `model.generate()` decoding in transformers>=4.4x.
        if past_key_value is not None or use_cache:
            DenseKVCache = _get_dense_cache_cls()
            if DenseKVCache is not None and isinstance(past_key_value, DenseKVCache):
                if (
                    _flashsvd_baseline_dense_kvcache_enabled()
                    and int(q_len) == 1
                    and _get_flash_attn_with_kvcache() is not None
                ):
                    attn_output = self._baseline_dense_decode_token_from_hidden(
                        hidden_states,
                        past_key_value,
                        cache_position=cache_position,
                    )
                    return attn_output, None
                if self._can_use_flashsvd_dense_decode_graph(hidden_states, past_key_value):
                    attn_output = self._flashsvd_dense_decode_token_from_hidden(
                        hidden_states,
                        past_key_value,
                        cache_position=cache_position,
                        cache_bindings=None,
                        advance_cache=True,
                    )
                    return attn_output, None

            # Low-rank KV-cache path: cache rank-space Pk/Pv (pre-RoPE) and use Triton decode kernel when q_len==1.
            LowRankKVCache = _get_lowrank_cache_cls()

            if LowRankKVCache is not None and isinstance(past_key_value, LowRankKVCache):
                baseline_lr_kvcache = os.getenv("FLASH_SVD_BASELINE_LR_KVCACHE", "0") != "0"
                force_flashsvd_kernel = os.getenv("FLASH_SVD_FORCE_ATTENTION_KERNEL", "0") != "0"
                auto_mha_sdpa = os.getenv("FLASH_SVD_AUTO_MHA_SDPA", "0") != "0"
                try:
                    mha_sdpa_r_thr = int(os.getenv("FLASH_SVD_MHA_SDPA_R_THRESHOLD", "512"))
                except Exception:
                    mha_sdpa_r_thr = 512
                # Default to fused FlashSVD decode for low-rank KV cache.
                # Enable this only when explicitly requested.
                auto_mha_stream = os.getenv("FLASH_SVD_AUTO_MHA_STREAM", "0") != "0"
                manual_mha_stream = os.getenv("FLASH_SVD_DECODE_MHA_STREAM", "0") != "0"
                try:
                    mha_stream_r_thr = int(os.getenv("FLASH_SVD_MHA_STREAM_R_THRESHOLD", "768"))
                except Exception:
                    mha_stream_r_thr = 768

                H = int(getattr(self, "num_heads", 0) or 0)
                Hk = int(getattr(self, "num_key_value_heads", H) or H)
                rep = max(1, H // max(1, Hk))
                R_attn = int(getattr(getattr(self, "q_v_proj", None), "out_features", 0) or 0)
                mha_stream_eligible = bool(
                    hidden_states.is_cuda
                    and torch.cuda.is_available()
                    and rep == 1
                    and H == Hk
                )
                auto_stream = bool(
                    mha_stream_eligible
                    and (not manual_mha_stream)
                    and auto_mha_stream
                    and R_attn >= int(mha_stream_r_thr)
                )
                use_mha_stream = bool(mha_stream_eligible and (manual_mha_stream or auto_stream))

                # Auto policy: for REP==1 (non-GQA) and large ranks, FlashSVD decode kernels
                # tend to be bandwidth-dominated and can underperform an SDPA baseline that
                # reconstructs K/V for all heads via GEMMs.
                auto_sdpa = bool(auto_mha_sdpa and rep == 1 and R_attn >= int(mha_sdpa_r_thr))
                use_flashsvd_kernel = bool(
                    (not baseline_lr_kvcache)
                    and (force_flashsvd_kernel or use_mha_stream or (not auto_sdpa))
                )
                if auto_sdpa and (not baseline_lr_kvcache) and (not force_flashsvd_kernel) and (not use_mha_stream):
                    global _LRKV_MHA_SDPA_WARNED
                    if not _LRKV_MHA_SDPA_WARNED:
                        print(
                            "[FlashSVD] LowRankKVCache: detected REP=1 (non-GQA) with "
                            f"R={R_attn} >= {mha_sdpa_r_thr}; using SDPA baseline instead of FlashSVD kernels. "
                            "Set FLASH_SVD_FORCE_ATTENTION_KERNEL=1 to override."
                        )
                        _LRKV_MHA_SDPA_WARNED = True
                if auto_stream and (not baseline_lr_kvcache) and (not force_flashsvd_kernel):
                    global _LRKV_MHA_STREAM_AUTO_WARNED
                    if not _LRKV_MHA_STREAM_AUTO_WARNED:
                        print(
                            "[FlashSVD] LowRankKVCache: REP=1 with "
                            f"R={R_attn} >= {mha_stream_r_thr}; using MHA streamed decode path."
                        )
                        _LRKV_MHA_STREAM_AUTO_WARNED = True

                # Optional fine-grained decode profiling (one layer, limited steps).
                # Enable with: FLASH_SVD_PROFILE_ATTN_DECODE=1
                # Optional: FLASH_SVD_PROFILE_ATTN_LAYER=0, FLASH_SVD_PROFILE_ATTN_STEPS=20
                try:
                    prof_enabled = os.getenv("FLASH_SVD_PROFILE_ATTN_DECODE", "0") != "0"
                    prof_layer = int(os.getenv("FLASH_SVD_PROFILE_ATTN_LAYER", "0"))
                    prof_steps = int(os.getenv("FLASH_SVD_PROFILE_ATTN_STEPS", "20"))
                except Exception:
                    prof_enabled, prof_layer, prof_steps = False, 0, 0

                layer_idx = int(getattr(self, "layer_idx", 0))
                do_prof = bool(
                    prof_enabled
                    and prof_steps > 0
                    and hidden_states.is_cuda
                    and torch.cuda.is_available()
                    and q_len == 1
                    and layer_idx == prof_layer
                    and use_flashsvd_kernel
                )
                if do_prof and not hasattr(self, "_attn_decode_prof_done"):
                    self._attn_decode_prof_done = False  # type: ignore[attr-defined]
                    self._attn_decode_prof_count = 0  # type: ignore[attr-defined]
                    self._attn_decode_prof_events = {  # type: ignore[attr-defined]
                        "proj": [],
                        "cache": [],
                        "rope": [],
                        "kernel": [],
                        "out": [],
                        "total": [],
                    }

                # Rank-space projections
                if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                    evs = self._attn_decode_prof_events  # type: ignore[attr-defined]
                    ev_total_s = torch.cuda.Event(enable_timing=True)
                    ev_total_e = torch.cuda.Event(enable_timing=True)
                    ev_total_s.record()

                    ev_proj_s = torch.cuda.Event(enable_timing=True)
                    ev_proj_e = torch.cuda.Event(enable_timing=True)
                    ev_proj_s.record()
                    Pq_rank, Pk_rank_step, Pv_rank_step = self._project_rank_qkv(hidden_states)  # [B, q, R]
                    ev_proj_e.record()
                    evs["proj"].append((ev_proj_s, ev_proj_e))
                else:
                    Pq_rank, Pk_rank_step, Pv_rank_step = self._project_rank_qkv(hidden_states)  # [B, q, R]

                # Some checkpoints use different ranks for Q vs K/V (e.g., adaptive-rank variants).
                # FlashSVD attention kernels currently assume a shared rank across Q/K/V.
                Rq = int(Pq_rank.shape[-1])
                Rk = int(Pk_rank_step.shape[-1])
                Rv = int(Pv_rank_step.shape[-1])
                if not (Rq == Rk == Rv):
                    # Force baseline (SDPA) path for correctness.
                    use_flashsvd_kernel = False
                    global _LRKV_RANK_MISMATCH_WARNED
                    if not _LRKV_RANK_MISMATCH_WARNED and not baseline_lr_kvcache:
                        print(
                            "[FlashSVD] LowRankKVCache: detected mismatched ranks "
                            f"(Rq={Rq}, Rk={Rk}, Rv={Rv}); FlashSVD attention kernels assume a shared rank. "
                            "Falling back to SDPA baseline for attention compute (still caches K/V in rank-space)."
                        )
                        _LRKV_RANK_MISMATCH_WARNED = True

                if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                    ev_cache_s = torch.cuda.Event(enable_timing=True)
                    ev_cache_e = torch.cuda.Event(enable_timing=True)
                    ev_cache_s.record()
                    self._update_lowrank_kv_cache(past_key_value, Pk_rank_step, Pv_rank_step, cache_position)
                    ev_cache_e.record()
                    evs["cache"].append((ev_cache_s, ev_cache_e))
                else:
                    self._update_lowrank_kv_cache(past_key_value, Pk_rank_step, Pv_rank_step, cache_position)

                if q_len == 1:
                    # Decode with split-K low-rank kernel; attend to keys up to current cache position.
                    # IMPORTANT: avoid `.item()` on CUDA tensors here. It introduces a device sync
                    # per layer per token and can destroy end-to-end decode throughput.
                    # LowRankKVCache maintains a host-side `_seen_tokens` counter; rely on it.
                    seqlen_k = int(past_key_value.get_seq_length())
                    Smax = int(past_key_value.get_max_cache_shape() or seqlen_k)

                    # Build factor views for the kernel
                    H = self.num_heads
                    Hk = int(getattr(self, "num_key_value_heads", H) or H)
                    R = int(Pq_rank.shape[-1])
                    Dh = self.head_dim
                    split_k_env = os.getenv("FLASH_SVD_DECODE_SPLIT_K", "").strip()
                    if split_k_env:
                        try:
                            split_k_cfg = max(1, int(split_k_env))
                        except Exception:
                            split_k_cfg = 512
                    else:
                        # Safer default for serving decode: REP==1 usually prefers smaller split_k.
                        split_k_cfg = 512 if rep == 1 else (1024 if R >= 768 else 512)

                    # MHA-specialized streamed decode path (REP==1): avoid SDPA peak memory and
                    # avoid per-head small GEMMs in Triton by using head-fused GEMMs (k_u_proj)
                    # on chunks of the KV cache, then online-softmax + rank-space value accumulation.
                    # Controlled by FLASH_SVD_DECODE_MHA_STREAM=1 or auto policy
                    # (FLASH_SVD_AUTO_MHA_STREAM=1) for REP==1 and large R.

                    if not use_flashsvd_kernel:
                        # Baseline path: LowRankKVCache storage, but compute attention with SDPA
                        # (i.e., no FlashSVD attention kernels). This is primarily for A/B timing.
                        # NOTE: This reconstructs dense K/V from rank cache each step, so it can be slow.
                        # Query in head-space: [B, H, 1, Dh]
                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, self.head_dim)
                        query_states = self.q_u_proj(Pq_rank).view(hidden_shape).transpose(1, 2)

                        # Keys/values from rank cache (valid range only): [B, Hk, S, Dh]
                        Pk_valid = past_key_value.key_cache[layer_idx][:bsz, :seqlen_k]
                        Pv_valid = past_key_value.value_cache[layer_idx][:bsz, :seqlen_k]
                        key_states = self.k_u_proj(Pk_valid).view(bsz, seqlen_k, Hk, Dh).transpose(1, 2)
                        value_states = self.v_u_proj(Pv_valid).view(bsz, seqlen_k, Hk, Dh).transpose(1, 2)

                        # RoPE (tables cached in LowRankKVCache): cos/sin are [Smax, Dh/2]
                        rotary_cos, rotary_sin = past_key_value.get_rope_tables(
                            seqlen=Smax, head_dim=Dh, device=hidden_states.device, dtype=hidden_states.dtype
                        )
                        pos_q = max(0, seqlen_k - 1)
                        cos_q = rotary_cos[pos_q].view(1, 1, 1, Dh // 2)
                        sin_q = rotary_sin[pos_q].view(1, 1, 1, Dh // 2)
                        query_states = self._apply_rope_tables(query_states, cos_q, sin_q)

                        cos_k = rotary_cos[:seqlen_k].view(1, 1, seqlen_k, Dh // 2)
                        sin_k = rotary_sin[:seqlen_k].view(1, 1, seqlen_k, Dh // 2)
                        key_states = self._apply_rope_tables(key_states, cos_k, sin_k)

                        # GQA: repeat KV heads to match query heads for SDPA.
                        if Hk != H:
                            rep = int(H // max(1, Hk))
                            if Hk * rep != H:
                                raise ValueError(f"Invalid GQA config: H={H}, Hk={Hk}")
                            key_states = key_states.repeat_interleave(rep, dim=1)
                            value_states = value_states.repeat_interleave(rep, dim=1)

                        # Decode uses query length 1 and we slice keys up to current position, so no causal mask needed.
                        attn_out = F.scaled_dot_product_attention(
                            query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
                        )  # [B, H, 1, Dh]
                        attn_output = attn_out.transpose(1, 2).reshape(bsz, 1, H * Dh).contiguous()
                        attn_output = self.o_u_proj(self.o_v_proj(attn_output))
                        return attn_output, None

                    if use_mha_stream:
                        # Query in head-space: [B, H, 1, Dh]
                        # RoPE tables: [Smax, Dh/2]
                        if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                            ev_rope_s = torch.cuda.Event(enable_timing=True)
                            ev_rope_e = torch.cuda.Event(enable_timing=True)
                            ev_rope_s.record()
                            rotary_cos, rotary_sin = past_key_value.get_rope_tables(
                                seqlen=Smax, head_dim=Dh, device=hidden_states.device, dtype=hidden_states.dtype
                            )
                            ev_rope_e.record()
                            evs["rope"].append((ev_rope_s, ev_rope_e))
                        else:
                            rotary_cos, rotary_sin = past_key_value.get_rope_tables(
                                seqlen=Smax, head_dim=Dh, device=hidden_states.device, dtype=hidden_states.dtype
                            )
                        pos_q = max(0, seqlen_k - 1)

                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, self.head_dim)
                        split_k = max(1, int(split_k_cfg))
                        scale = scaling

                        if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                            ev_kernel_s = torch.cuda.Event(enable_timing=True)
                            ev_kernel_e = torch.cuda.Event(enable_timing=True)
                            ev_kernel_s.record()

                        query_states = self.q_u_proj(Pq_rank).view(hidden_shape).transpose(1, 2)  # [B,H,1,Dh]
                        cos_q = rotary_cos[pos_q].view(1, 1, 1, Dh // 2)
                        sin_q = rotary_sin[pos_q].view(1, 1, 1, Dh // 2)
                        query_states = self._apply_rope_tables(query_states, cos_q, sin_q)

                        m_i = torch.full((bsz, H), float("-inf"), device=hidden_states.device, dtype=torch.float32)
                        l_i = torch.zeros((bsz, H), device=hidden_states.device, dtype=torch.float32)
                        acc_r = torch.zeros((bsz, H, R), device=hidden_states.device, dtype=torch.float32)

                        # Iterate over KV in chunks (streaming)
                        for start in range(0, seqlen_k, split_k):
                            end = min(start + split_k, seqlen_k)
                            slen = end - start

                            Pk_split = past_key_value.key_cache[layer_idx][:bsz, start:end]  # [B,S,R]
                            Pv_split = past_key_value.value_cache[layer_idx][:bsz, start:end]  # [B,S,R]

                            # Dense K for all heads in this chunk: [B,S,H*Dh] -> [B,H,S,Dh]
                            k_flat = self.k_u_proj(Pk_split)
                            k_bhsd = k_flat.view(bsz, slen, H, Dh).permute(0, 2, 1, 3).contiguous()

                            cos_k = rotary_cos[start:end].view(1, 1, slen, Dh // 2)
                            sin_k = rotary_sin[start:end].view(1, 1, slen, Dh // 2)
                            k_bhsd = self._apply_rope_tables(k_bhsd, cos_k, sin_k)

                            # Scores: [B,H,1,Dh] x [B,H,Dh,S] -> [B,H,S]
                            scores = torch.matmul(query_states, k_bhsd.transpose(-1, -2)).squeeze(2)
                            scores = scores.to(torch.float32) * scale

                            m_curr = scores.max(dim=-1).values
                            m_new = torch.maximum(m_i, m_curr)
                            alpha = torch.exp(m_i - m_new)

                            p = torch.exp(scores - m_new.unsqueeze(-1))
                            l_i = l_i * alpha + p.sum(dim=-1)

                            # Rank-space value accumulation: [B,H,S] @ [B,S,R] -> [B,H,R]
                            acc_add = torch.matmul(p.to(hidden_states.dtype), Pv_split).to(torch.float32)
                            acc_r = acc_r * alpha.unsqueeze(-1) + acc_add
                            m_i = m_new

                        den = torch.where(l_i > 0, l_i, torch.ones_like(l_i))
                        w_r = acc_r / den.unsqueeze(-1)
                        w_r = torch.where(l_i.unsqueeze(-1) > 0, w_r, torch.zeros_like(w_r))

                        # Lift once: [B,H,R] x [H,R,Dh] -> [B,H,Dh]
                        _, _, Vv = self._get_decode_factors()
                        out_bhd = torch.einsum("bhr,hrd->bhd", w_r.to(hidden_states.dtype), Vv)

                        if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                            ev_kernel_e.record()
                            evs["kernel"].append((ev_kernel_s, ev_kernel_e))

                            ev_out_s = torch.cuda.Event(enable_timing=True)
                            ev_out_e = torch.cuda.Event(enable_timing=True)
                            ev_out_s.record()
                            attn_output = out_bhd.reshape(bsz, 1, H * Dh)
                            attn_output = self.o_u_proj(self.o_v_proj(attn_output))
                            ev_out_e.record()
                            evs["out"].append((ev_out_s, ev_out_e))

                            ev_total_e.record()
                            evs["total"].append((ev_total_s, ev_total_e))

                            # Flush and print after N steps.
                            self._attn_decode_prof_count += 1  # type: ignore[attr-defined]
                            if self._attn_decode_prof_count >= prof_steps and not self._attn_decode_prof_done:  # type: ignore[attr-defined]
                                torch.cuda.synchronize()

                                def _avg_ms(key: str) -> float:
                                    pairs = evs.get(key, [])
                                    if not pairs:
                                        return 0.0
                                    total = sum(float(s.elapsed_time(e)) for s, e in pairs)
                                    return total / float(len(pairs))

                                rep_dbg = max(1, int(self.num_heads // max(1, int(getattr(self, "num_key_value_heads", self.num_heads)))))
                                num_splits_dbg = max(1, (seqlen_k + split_k - 1) // split_k)
                                print(
                                    "[FlashSVD][attn_decode_prof] path=mha_stream "
                                    f"layer={layer_idx} steps={int(self._attn_decode_prof_count)} "
                                    f"H={H} Hk={int(getattr(self, 'num_key_value_heads', H))} REP={rep_dbg} "
                                    f"Dh={Dh} R={R} seqlen_k={seqlen_k} Smax={Smax} "
                                    f"split_k={split_k} num_splits={num_splits_dbg}"
                                )
                                print(
                                    "[FlashSVD][attn_decode_prof] ms: "
                                    f"proj={_avg_ms('proj'):.3f} "
                                    f"cache={_avg_ms('cache'):.3f} "
                                    f"rope={_avg_ms('rope'):.3f} "
                                    f"kernel={_avg_ms('kernel'):.3f} "
                                    f"out={_avg_ms('out'):.3f} "
                                    f"total={_avg_ms('total'):.3f}"
                                )
                                self._attn_decode_prof_done = True  # type: ignore[attr-defined]
                        else:
                            attn_output = out_bhd.reshape(bsz, 1, H * Dh)
                            attn_output = self.o_u_proj(self.o_v_proj(attn_output))
                        return attn_output, None

                    # Query: [B, H, R] (broadcast across heads if rank-space is shared)
                    Pq_q = Pq_rank[:, 0, :].unsqueeze(1).expand(bsz, H, R)

                    # KV caches: [B, Smax, Hk, R] with 0-stride head dim to avoid materialization
                    Pk_cache = past_key_value.key_cache[layer_idx][:bsz, :Smax]  # [B, Smax, R]
                    Pv_cache = past_key_value.value_cache[layer_idx][:bsz, :Smax]
                    Pk = Pk_cache.unsqueeze(2).expand(bsz, Smax, Hk, R)
                    Pv = Pv_cache.unsqueeze(2).expand(bsz, Smax, Hk, R)

                    Vq, Vk, Vv = self._get_decode_factors()

                    # RoPE tables: [Smax, Dh/2]
                    if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                        ev_rope_s = torch.cuda.Event(enable_timing=True)
                        ev_rope_e = torch.cuda.Event(enable_timing=True)
                        ev_rope_s.record()
                        rotary_cos, rotary_sin = past_key_value.get_rope_tables(
                            seqlen=Smax, head_dim=Dh, device=hidden_states.device, dtype=hidden_states.dtype
                        )
                        ev_rope_e.record()
                        evs["rope"].append((ev_rope_s, ev_rope_e))
                    else:
                        rotary_cos, rotary_sin = past_key_value.get_rope_tables(
                            seqlen=Smax, head_dim=Dh, device=hidden_states.device, dtype=hidden_states.dtype
                        )

                    mods = _get_flashsvd_decode_attn_mods()

                    # Persistent workspace/buffers (slice to current num_splits)
                    split_k = max(1, int(split_k_cfg))
                    bn_env = os.getenv("FLASH_SVD_DECODE_BN", "").strip()
                    br_env = os.getenv("FLASH_SVD_DECODE_BR", "").strip()
                    try:
                        bn = int(bn_env) if bn_env else 64
                    except Exception:
                        bn = 64
                    try:
                        br = int(br_env) if br_env else 64
                    except Exception:
                        br = 64
                    # Decode kernel (v1.6) uses a (BN x R) Pv tile and optionally keeps Vk (R x Dh)
                    # resident. For large ranks (e.g. R=1024 for ratio=0.5 on LLaMA-7B),
                    # the resident path can exceed SMEM limits (A100: ~164KB).
                    # Auto-clamp BN / stages and disable Vk residency when needed.
                    dtype = hidden_states.dtype
                    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4

                    def _env_flag(name: str, default: str) -> str:
                        return os.getenv(name, default).strip().lower()

                    vk_res_env = _env_flag("FLASH_SVD_DECODE_VK_RESIDENT", "auto")
                    if vk_res_env in {"1", "true", "yes", "y", "on"}:
                        vk_resident = True
                    elif vk_res_env in {"0", "false", "no", "n", "off"}:
                        vk_resident = False
                    else:
                        # Auto: only allow Vk residency for small ranks.
                        vk_resident = R <= 384

                    # Padding REP to 16 is beneficial for true GQA (REP>1) to unlock tensor cores,
                    # but for REP==1 it inflates work by 16x (GROUP_M=16 with 15 masked lanes).
                    rep = max(1, int(self.num_heads // max(1, int(getattr(self, "num_key_value_heads", self.num_heads)))))
                    pad_env = _env_flag("FLASH_SVD_DECODE_PAD_TO_16", "auto")
                    if pad_env in {"1", "true", "yes", "y", "on"}:
                        pad_to_16 = True
                    elif pad_env in {"0", "false", "no", "n", "off"}:
                        pad_to_16 = False
                    else:
                        pad_to_16 = rep > 1

                    # Limit BN so the Pv tile stays within a conservative SMEM budget.
                    # (We can't query exact SMEM here; keep it robust across GPUs.)
                    # If FLASH_SVD_DECODE_SMEM_BUDGET is unset, derive from device SMEM so
                    # large-rank decode (e.g., R=1024) can use a larger BN on high-SMEM GPUs.
                    budget_env = os.getenv("FLASH_SVD_DECODE_SMEM_BUDGET", "").strip()
                    if budget_env:
                        try:
                            budget_bytes = int(budget_env)
                        except Exception:
                            budget_bytes = 64 * 1024
                    else:
                        try:
                            props = torch.cuda.get_device_properties(hidden_states.device)
                            smem_per_sm = int(getattr(props, "shared_memory_per_multiprocessor", 0) or 0)
                            if smem_per_sm > 0:
                                budget_bytes = max(64 * 1024, min(160 * 1024, smem_per_sm - (8 * 1024)))
                            else:
                                budget_bytes = 64 * 1024
                        except Exception:
                            budget_bytes = 64 * 1024
                    if R > 0:
                        bn_max = max(16, budget_bytes // max(1, R * bytes_per_elem))
                    else:
                        bn_max = 16
                    bn = self._normalize_bn(split_k, bn, bn_max)

                    # For very large ranks, also reduce pipeline stages (SMEM is often double-buffered).
                    num_warps_stage1 = int(os.getenv("FLASH_SVD_DECODE_WARPS1", "4"))
                    num_stages_stage1 = int(os.getenv("FLASH_SVD_DECODE_STAGES1", "2"))
                    num_warps_stage2 = int(os.getenv("FLASH_SVD_DECODE_WARPS2", "4"))
                    num_stages_stage2 = int(os.getenv("FLASH_SVD_DECODE_STAGES2", "1"))
                    if R >= 512:
                        vk_resident = False
                        num_stages_stage1 = min(num_stages_stage1, 1)

                    br = max(1, min(int(br), int(R)))

                    max_splits = max(1, (Smax + split_k - 1) // split_k)
                    ws = past_key_value.get_decode_workspace(
                        batch_size=bsz,
                        num_heads=H,
                        rank=R,
                        head_dim=Dh,
                        max_splits=max_splits,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )

                    num_splits = max(1, (seqlen_k + split_k - 1) // split_k)
                    workspace = (
                        ws.M[:, :, :num_splits],
                        ws.L[:, :, :num_splits],
                        ws.Acc[:, :, :num_splits, :],
                    )
                    q_buffers = (ws.Q0, ws.Q1)

                    call_kwargs: dict[str, object] = dict(
                        seqlen_k=seqlen_k,
                        causal=True,
                        split_k=split_k,
                        bn=bn,
                        br=min(br, R),
                        num_warps_stage1=num_warps_stage1,
                        num_stages_stage1=num_stages_stage1,
                        num_warps_stage2=num_warps_stage2,
                        num_stages_stage2=num_stages_stage2,
                        q_buffers=q_buffers,
                        workspace=workspace,
                        precompute_q=True,
                        writethrough=True,
                        pad_to_16=bool(pad_to_16),
                        vk_resident=bool(vk_resident),
                    )

                    selected_variant = self._select_decode_variant(
                        mods=mods,
                        hidden_states=hidden_states,
                        bsz=bsz,
                        H=H,
                        Hk=Hk,
                        R=R,
                        Dh=Dh,
                        seqlen_k=seqlen_k,
                        Pq_q=Pq_q,
                        Pk=Pk,
                        Pv=Pv,
                        Vq=Vq,
                        Vk=Vk,
                        Vv=Vv,
                        rotary_cos=rotary_cos,
                        rotary_sin=rotary_sin,
                        call_kwargs=call_kwargs,
                    )
                    mod, decode_fn = _resolve_decode_variant(mods, selected_variant)
                    if mod is None or decode_fn is None:
                        mod = _get_flashsvd_decode_attn_mod()
                        decode_fn = getattr(mod, "flashsvd_attn_decode_packed")
                        selected_variant = "v16_v2"

                    if selected_variant == "v16_v2":
                        f_v2 = mod.DecodePackedFactors(Pq=Pq_q, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)
                        split_k, bn, br = self._autotune_fused_decode_cfg(
                            mod=mod,
                            f=f_v2,
                            past_key_value=past_key_value,
                            rotary_cos=rotary_cos,
                            rotary_sin=rotary_sin,
                            hidden_states=hidden_states,
                            bsz=bsz,
                            H=H,
                            Hk=Hk,
                            R=R,
                            Dh=Dh,
                            Smax=Smax,
                            seqlen_k=seqlen_k,
                            split_k=split_k,
                            bn=bn,
                            br=br,
                            bn_max=bn_max,
                            num_warps_stage1=num_warps_stage1,
                            num_stages_stage1=num_stages_stage1,
                            num_warps_stage2=num_warps_stage2,
                            num_stages_stage2=num_stages_stage2,
                            pad_to_16=bool(pad_to_16),
                            vk_resident=bool(vk_resident),
                        )
                        max_splits = max(1, (Smax + split_k - 1) // split_k)
                        ws = past_key_value.get_decode_workspace(
                            batch_size=bsz,
                            num_heads=H,
                            rank=R,
                            head_dim=Dh,
                            max_splits=max_splits,
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                        num_splits = max(1, (seqlen_k + split_k - 1) // split_k)
                        workspace = (
                            ws.M[:, :, :num_splits],
                            ws.L[:, :, :num_splits],
                            ws.Acc[:, :, :num_splits, :],
                        )
                        q_buffers = (ws.Q0, ws.Q1)
                        call_kwargs = dict(
                            call_kwargs,
                            split_k=split_k,
                            bn=bn,
                            br=min(br, R),
                            workspace=workspace,
                            q_buffers=q_buffers,
                        )

                    f = mod.DecodePackedFactors(Pq=Pq_q, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)
                    decode_kwargs = _maybe_kwargs(decode_fn, call_kwargs)

                    def _run_decode_once():
                        return decode_fn(f, rotary_cos, rotary_sin, **decode_kwargs)

                    def _run_with_fallback():
                        try:
                            return _run_decode_once()
                        except Exception:
                            for alt in ("v15", "v16_v1", "v16_v2"):
                                if alt == selected_variant:
                                    continue
                                mod_alt, fn_alt = _resolve_decode_variant(mods, alt)
                                if mod_alt is None or fn_alt is None:
                                    continue
                                try:
                                    f_alt = mod_alt.DecodePackedFactors(Pq=Pq_q, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)
                                    kw_alt = _maybe_kwargs(fn_alt, call_kwargs)
                                    return fn_alt(f_alt, rotary_cos, rotary_sin, **kw_alt)
                                except Exception:
                                    continue
                            raise

                    if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                        ev_k_s = torch.cuda.Event(enable_timing=True)
                        ev_k_e = torch.cuda.Event(enable_timing=True)
                        ev_k_s.record()
                        O_bhd = _run_with_fallback()  # [B, H, Dh]
                        ev_k_e.record()
                        evs["kernel"].append((ev_k_s, ev_k_e))
                    else:
                        O_bhd = _run_with_fallback()  # [B, H, Dh]

                    if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                        ev_out_s = torch.cuda.Event(enable_timing=True)
                        ev_out_e = torch.cuda.Event(enable_timing=True)
                        ev_out_s.record()
                        attn_output = O_bhd.reshape(bsz, 1, H * Dh)
                        attn_output = self.o_u_proj(self.o_v_proj(attn_output))
                        ev_out_e.record()
                        evs["out"].append((ev_out_s, ev_out_e))
                        ev_total_e.record()
                        evs["total"].append((ev_total_s, ev_total_e))

                        # Flush and print after N steps.
                        self._attn_decode_prof_count += 1  # type: ignore[attr-defined]
                        if self._attn_decode_prof_count >= prof_steps and not self._attn_decode_prof_done:  # type: ignore[attr-defined]
                            torch.cuda.synchronize()

                            def _avg_ms(key: str) -> float:
                                pairs = evs.get(key, [])
                                if not pairs:
                                    return 0.0
                                total = sum(float(s.elapsed_time(e)) for s, e in pairs)
                                return total / float(len(pairs))

                            rep_dbg = max(1, int(self.num_heads // max(1, int(getattr(self, "num_key_value_heads", self.num_heads)))))
                            num_splits_dbg = max(1, (seqlen_k + split_k - 1) // split_k)
                            print(
                                "[FlashSVD][attn_decode_prof] "
                                f"layer={layer_idx} steps={int(self._attn_decode_prof_count)} "
                                f"H={H} Hk={int(getattr(self, 'num_key_value_heads', H))} REP={rep_dbg} "
                                f"Dh={Dh} R={R} seqlen_k={seqlen_k} Smax={Smax} "
                                f"variant={selected_variant} "
                                f"split_k={split_k} bn={bn} br={min(br, R)} num_splits={num_splits_dbg} "
                                f"pad_to_16={bool(pad_to_16)} vk_resident={bool(vk_resident)} "
                                f"warps1={num_warps_stage1} stages1={num_stages_stage1} "
                                f"warps2={num_warps_stage2} stages2={num_stages_stage2}"
                            )
                            print(
                                "[FlashSVD][attn_decode_prof] ms: "
                                f"proj={_avg_ms('proj'):.3f} "
                                f"cache={_avg_ms('cache'):.3f} "
                                f"rope={_avg_ms('rope'):.3f} "
                                f"kernel={_avg_ms('kernel'):.3f} "
                                f"out={_avg_ms('out'):.3f} "
                                f"total={_avg_ms('total'):.3f}"
                            )
                            self._attn_decode_prof_done = True  # type: ignore[attr-defined]
                    else:
                        attn_output = O_bhd.reshape(bsz, 1, H * Dh)
                        attn_output = self.o_u_proj(self.o_v_proj(attn_output))
                    return attn_output, None

                # Prefill (q_len>1): use FlashSVD full-seq kernel and populate low-rank cache.
                B, M, R = Pq_rank.shape
                H, dh = self.num_heads, self.head_dim

                if not use_flashsvd_kernel:
                    # Baseline path (no FlashSVD kernels): build dense Q/K/V from rank factors and use SDPA.
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.head_dim)

                    query_states = self.q_u_proj(Pq_rank).view(hidden_shape).transpose(1, 2)
                    key_states = self.k_u_proj(Pk_rank_step).view(hidden_shape).transpose(1, 2)
                    value_states = self.v_u_proj(Pv_rank_step).view(hidden_shape).transpose(1, 2)

                    if position_embeddings is None:
                        # Prefer cache_position (when provided) to match decode offsets.
                        if position_ids is None:
                            if cache_position is not None:
                                if cache_position.dim() == 1:
                                    position_ids = cache_position.to(device=hidden_states.device).unsqueeze(0).expand(bsz, q_len)
                                else:
                                    position_ids = cache_position.to(device=hidden_states.device)
                            else:
                                position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, q_len)
                        cos, sin = self.rotary_emb(value_states, seq_len=int(position_ids.max().item()) + 1)
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
                    else:
                        cos, sin = position_embeddings
                        query_states, key_states = self._apply_rope_hf(query_states, key_states, cos, sin)

                    # GQA: repeat KV heads to match query heads for SDPA.
                    if key_states.shape[1] != query_states.shape[1]:
                        Hq = int(query_states.shape[1])
                        Hk = int(key_states.shape[1])
                        rep = int(Hq // max(1, Hk))
                        if Hk * rep != Hq:
                            raise ValueError(f"Invalid GQA config: H={Hq}, Hk={Hk}")
                        key_states = key_states.repeat_interleave(rep, dim=1)
                        value_states = value_states.repeat_interleave(rep, dim=1)

                    is_causal = attention_mask is None and getattr(self, "is_causal", True)
                    attn_out = F.scaled_dot_product_attention(
                        query_states,
                        key_states,
                        value_states,
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=is_causal,
                    ).transpose(1, 2)
                    attn_output = attn_out.reshape(B, M, H * dh).contiguous()
                    attn_output = self.o_u_proj(self.o_v_proj(attn_output))
                    return attn_output, None

                Hk = int(getattr(self, "num_key_value_heads", H) or H)

                # Expand along heads (rank factors are shared across heads). Zero-stride avoids materialization.
                Pq4 = Pq_rank.unsqueeze(1).expand(B, H, M, R)
                Pk4 = Pk_rank_step.unsqueeze(1).expand(B, Hk, M, R)
                Pv4 = Pv_rank_step.unsqueeze(1).expand(B, Hk, M, R)

                Vq, Vk, Vv = self._get_decode_factors()

                # Position ids default: [B, M]
                if position_ids is None:
                    position_ids = torch.arange(M, device=hidden_states.device).unsqueeze(0).expand(B, M)

                qkv = QKVFactors(Pq=Pq4, Pk=Pk4, Pv=Pv4, Vq=Vq, Vk=Vk, Vv=Vv, bq=None, bk=None, bv=None)
                attn_bmhd = self.flash_attn(
                    qkv,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )  # [B, M, H, dh]

                attn_output = attn_bmhd.reshape(B, M, H * dh)
                attn_output = self.o_u_proj(self.o_v_proj(attn_output))
                return attn_output, None

            # Build dense Q/K/V via low-rank projections (for cache updates we need dense K/V).
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_u_proj(self.q_v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
            key_states = self.k_u_proj(self.k_v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)
            value_states = self.v_u_proj(self.v_v_proj(hidden_states)).view(hidden_shape).transpose(1, 2)

            if position_embeddings is None:
                # Fallback: build RoPE cos/sin from position_ids if caller didn't pass shared position embeddings.
                if position_ids is None:
                    position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, q_len)
                cos, sin = self.rotary_emb(value_states, seq_len=int(position_ids.max().item()) + 1)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            else:
                cos, sin = position_embeddings
                query_states, key_states = self._apply_rope_hf(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin/cos are specific to RoPE models; cache_position needed for static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, int(self.layer_idx), cache_kwargs)

            # Reuse HF attention helpers to match mask semantics for sdpa/flash/flex, etc.
            try:
                from transformers.models.llama.modeling_llama import eager_attention_forward, ALL_ATTENTION_FUNCTIONS

                attention_interface = eager_attention_forward
                if getattr(self.config, "_attn_implementation", "eager") != "eager":
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else attention_dropout,
                    scaling=scaling,
                    output_attentions=output_attentions,
                    **kwargs,
                )
            except Exception:
                # Minimal fallback: SDPA (no attn weights).
                is_causal = attention_mask is None and getattr(self, "is_causal", True)
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                ).transpose(1, 2)
                attn_weights = None

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_u_proj(self.o_v_proj(attn_output))
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights

        # Build low-rank P factors: [B, M, R] -> expand to [B, H, M, R]
        Pq = self.q_v_proj(hidden_states)  # [B, M, R]
        Pk = self.k_v_proj(hidden_states)  # [B, M, R]
        Pv = self.v_v_proj(hidden_states)  # [B, M, R]

        B, M, R = Pq.shape
        H, dh = self.num_heads, self.head_dim

        # Expand along heads (rank factors are shared across heads)
        # Expand across heads as views (zero stride on H) to avoid materialization
        Pq = Pq.unsqueeze(1).expand(B, H, M, R)
        Pk = Pk.unsqueeze(1).expand(B, H, M, R)
        Pv = Pv.unsqueeze(1).expand(B, H, M, R)

        # Build V factors from effective projection weights: [H, R, dh]
        # Include LoRA delta if adapters are active by reading lora_A/lora_B
        def _eff_weight(linear: nn.Module):
            W = linear.weight
            if hasattr(linear, 'lora_A') and hasattr(linear, 'lora_B'):
                adapter = getattr(linear, 'active_adapter', None)
                try:
                    if adapter is not None and adapter in linear.lora_A and adapter in linear.lora_B:
                        W = W + (linear.lora_B[adapter].weight @ linear.lora_A[adapter].weight) * linear.scaling[adapter]
                except Exception:
                    pass
            return W

        Vq = _eff_weight(self.q_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()
        Vk = _eff_weight(self.k_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()
        Vv = _eff_weight(self.v_u_proj).view(H, dh, R).permute(0, 2, 1).contiguous()

        # No biases in low-rank projections by default
        bq = bk = bv = None

        # Position ids default: [B, M]
        if position_ids is None:
            position_ids = torch.arange(M, device=hidden_states.device).unsqueeze(0).expand(B, M)

        # Attention mask handling: support 2D pad mask [B, M] or 4D additive [B,1,M,M]
        add_mask = None
        pad_mask = None
        pad_query_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = attention_mask
                # Convert 2D pad mask to 4D additive mask for FlashSVD compatibility
                pm = pad_mask.to(torch.bool)
                valid = pm[:, None, :, None] & pm[:, None, None, :]
                add_mask = torch.zeros((B, 1, M, M), device=pad_mask.device, dtype=torch.float32)
                add_mask = add_mask.masked_fill(~valid, float("-inf"))
                if fix_pad_query_mask or debug_pad_query_mask:
                    pad_query_mask = ~pad_mask.to(torch.bool)
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] != M or attention_mask.shape[-1] != M:
                    raise NotImplementedError("Attention mask with differing q/k lengths not supported here.")
                add_mask = attention_mask
                if fix_pad_query_mask or debug_pad_query_mask:
                    if add_mask.dtype == torch.bool:
                        row_all_masked = ~add_mask.any(dim=-1)
                    elif torch.is_floating_point(add_mask):
                        row_all_masked = torch.isneginf(add_mask).all(dim=-1)
                        if not row_all_masked.any():
                            row_all_masked = (add_mask <= -1e4).all(dim=-1)
                    else:
                        row_all_masked = None
                    if row_all_masked is not None and row_all_masked.any():
                        if debug_pad_query_mask and not getattr(self, "_pad_query_warned", False):
                            num = int(row_all_masked.sum().item())
                            print(f"[FlashSVD] Detected {num} fully-masked query rows; "
                                  f"consider fixing pad-query rows or forcing right padding.")
                            self._pad_query_warned = True
                        if fix_pad_query_mask:
                            add_mask = add_mask.masked_fill(row_all_masked.unsqueeze(-1), 0.0)
                        pad_query_mask = row_all_masked.squeeze(1)
            else:
                raise ValueError(f"Unsupported attention_mask shape: {tuple(attention_mask.shape)}")

        qkv = QKVFactors(Pq=Pq, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv, bq=bq, bk=bk, bv=bv)

        if os.getenv("SVDLLM_FLASH_FALLBACK", "0") != "0" or _flashsvd_reference_dense_attn_enabled():
            # Fallback: explicit attention without FlashSVD kernel
            Q = torch.einsum("bhmr,hrd->bhmd", Pq, Vq)
            K = torch.einsum("bhmr,hrd->bhmd", Pk, Vk)
            V = torch.einsum("bhmr,hrd->bhmd", Pv, Vv)
            if bq is not None:
                Q = Q + bq.view(1, H, 1, dh)
            if bk is not None:
                K = K + bk.view(1, H, 1, dh)
            if bv is not None:
                V = V + bv.view(1, H, 1, dh)
            cos, sin = self.rotary_emb(Q, seq_len=M)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)
            attn_mask_sdpa = None
            is_causal = True
            if add_mask is not None:
                attn_mask_sdpa = add_mask
                is_causal = False
            elif pad_mask is not None:
                pm = pad_mask.to(torch.bool)
                attn_mask_sdpa = pm[:, None, :, None] & pm[:, None, None, :]
            attn_out_bhmd = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask_sdpa, dropout_p=0.0, is_causal=is_causal
            )
            if pad_mask is not None:
                attn_out_bhmd = attn_out_bhmd.masked_fill(~pad_mask[:, None, :, None].to(torch.bool), 0.0)
            attn_bmhd = attn_out_bhmd.permute(0, 2, 1, 3).contiguous()
        else:
            attn_bmhd = self.flash_attn(
                qkv,
                attention_mask=add_mask if add_mask is not None else pad_mask,
                position_ids=position_ids,
            )  # [B, M, H, dh]

        # Fold heads back to [B, M, D]
        attn_output = attn_bmhd.reshape(B, M, H * dh)
        if fix_pad_query_mask and pad_query_mask is not None:
            attn_output = attn_output.masked_fill(pad_query_mask[:, :, None], 0.0)

        # Low-rank output projection
        attn_output = self.o_u_proj(self.o_v_proj(attn_output))

        attn_weights = None
        return attn_output, attn_weights
    
