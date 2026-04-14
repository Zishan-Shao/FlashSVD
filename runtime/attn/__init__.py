from __future__ import annotations

from .decode_registry import (
    call_flash_attn_with_kvcache,
    get_default_flashsvd_decode_attn_mod,
    get_default_flashsvd_attn_legacy_sparsekv_decode_mod,
    get_dense_token_decode_mod,
    get_flash_attn_with_kvcache,
    get_flashsvd_decode_attn_mods,
    get_flashsvd_attn_legacy_sparsekv_decode_mods,
    get_flashsvd_v15_attn_densekv_decode_mod,
    maybe_kwargs,
    resolve_decode_variant,
    resolve_legacy_sparsekv_decode_variant,
    select_decode_variant,
    select_legacy_sparsekv_decode_variant,
)

__all__ = [
    "call_flash_attn_with_kvcache",
    "get_default_flashsvd_decode_attn_mod",
    "get_default_flashsvd_attn_legacy_sparsekv_decode_mod",
    "get_dense_token_decode_mod",
    "get_flash_attn_with_kvcache",
    "get_flashsvd_decode_attn_mods",
    "get_flashsvd_attn_legacy_sparsekv_decode_mods",
    "get_flashsvd_v15_attn_densekv_decode_mod",
    "maybe_kwargs",
    "resolve_decode_variant",
    "resolve_legacy_sparsekv_decode_variant",
    "select_decode_variant",
    "select_legacy_sparsekv_decode_variant",
]
