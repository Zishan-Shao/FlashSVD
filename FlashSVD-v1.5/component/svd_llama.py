from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

from flashsvd_component.svd_llama import SVD_LlamaAttention as _FlashSVD_LlamaAttention
from flashsvd_component.svd_llama import SVD_LlamaMLP as _FlashSVD_LlamaMLP
from flashsvd_component.svd_llama import enable_flashsvd_llama_layer_tail_cuda_graph

# ---------------------------------------------------------------------------
# Compatibility shims for older pickled checkpoints
# ---------------------------------------------------------------------------
# Some training pipelines historically defined / re-exported HuggingFace LLaMA
# modules from this file. When such models are pickled (torch.save(model)),
# the unpickler looks up classes by `component.svd_llama.<ClassName>`.
# Keep lightweight aliases here so we can load older checkpoints.
try:  # pragma: no cover
    from transformers.models.llama import modeling_llama as _hf_llama  # type: ignore
    import inspect as _inspect

    # Export common HF classes under `component.svd_llama.*` so unpickling works
    # across LLaMA 1/2/3 checkpoints trained with older codebases.
    for _name in dir(_hf_llama):
        if not _name.startswith("Llama"):
            continue
        _obj = getattr(_hf_llama, _name, None)
        if _inspect.isclass(_obj):
            globals()[_name] = _obj
except Exception:
    # If transformers isn't available, we still want the rest of this module
    # (FlashSVD wrappers) to import.
    pass


class SVD_LlamaAttention(_FlashSVD_LlamaAttention):
    """Wrapper that matches the `SVDLLM.py` constructor signature."""

    def __init__(self, config, ratio=1, *, compat_ranks: bool = False, compat_attention: bool = False):
        # `flashsvd_component` already uses the official param-ratio->rank mapping.
        # We accept compat flags for API-compat with `SVDLLM.py`.
        super().__init__(config=config, ratio=ratio)
        compat_attention = bool(compat_attention)
        # Keep both names for backward-compat with older pickled checkpoints.
        self._compat_attention = compat_attention
        self.compat_attention = compat_attention

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        # Backward-compat for older pickled checkpoints: this attribute may be absent.
        compat_attention = bool(getattr(self, "_compat_attention", getattr(self, "compat_attention", False)))
        if not compat_attention:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                past_key_values=past_key_values,
                **kwargs,
            )

        # Force explicit attention math (no FlashSVD kernel) for this call only.
        old = os.environ.get("SVDLLM_FLASH_FALLBACK", None)
        os.environ["SVDLLM_FLASH_FALLBACK"] = "1"
        try:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                past_key_values=past_key_values,
                **kwargs,
            )
        finally:
            if old is None:
                os.environ.pop("SVDLLM_FLASH_FALLBACK", None)
            else:
                os.environ["SVDLLM_FLASH_FALLBACK"] = old


class SVD_LlamaMLP(_FlashSVD_LlamaMLP):
    """Wrapper that matches the `SVDLLM.py` constructor signature."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, ratio=1, *, compat_ranks: bool = False):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            ratio=ratio,
        )
