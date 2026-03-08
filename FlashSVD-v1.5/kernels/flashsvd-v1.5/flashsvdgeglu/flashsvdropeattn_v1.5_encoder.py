#!/usr/bin/env python3
"""FlashSVD RoPE attention v1.5 (encoder)

Redesigned for ModernBERT-like encoder usage:
- consistent low-rank input contract: P* in [B, L, R], V* in [R, H*Dh] (or [H, R, Dh])
- adaptive sliding-window path via chunked SDPA
- supports 2D padding mask and 4D additive/sliding masks

This module focuses on real throughput on encoder workloads with local attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_PACKED_QKV_U_CACHE: dict[tuple, torch.Tensor] = {}


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


@dataclass
class QKVFactors:
    # Rank-space activations
    Pq: torch.Tensor  # [B, L, R]
    Pk: torch.Tensor  # [B, L, R]
    Pv: torch.Tensor  # [B, L, R]
    # Rank-to-head factors
    Vq: torch.Tensor  # [R, H*Dh] or [H, R, Dh]
    Vk: torch.Tensor
    Vv: torch.Tensor
    # Optional head-space biases
    bq: Optional[torch.Tensor] = None  # [H*Dh] or [H, Dh]
    bk: Optional[torch.Tensor] = None
    bv: Optional[torch.Tensor] = None


def _tensor_version(t: torch.Tensor) -> int:
    try:
        return int(getattr(t, "_version", -1))
    except Exception:
        return -1


def _packed_qkv_u_cache_key(Uq: torch.Tensor, Uk: torch.Tensor, Uv: torch.Tensor) -> tuple:
    return (
        Uq.device.type,
        Uq.device.index,
        str(Uq.dtype),
        int(Uq.shape[0]),
        int(Uq.shape[1]),
        int(Uq.data_ptr()),
        int(Uk.data_ptr()),
        int(Uv.data_ptr()),
        _tensor_version(Uq),
        _tensor_version(Uk),
        _tensor_version(Uv),
    )


def precompute_qkv_u(Uq: torch.Tensor, Uk: torch.Tensor, Uv: torch.Tensor) -> torch.Tensor:
    return torch.cat((Uq, Uk, Uv), dim=1).contiguous()


def get_precomputed_qkv_u(Uq: torch.Tensor, Uk: torch.Tensor, Uv: torch.Tensor) -> torch.Tensor:
    key = _packed_qkv_u_cache_key(Uq, Uk, Uv)
    cached = _PACKED_QKV_U_CACHE.get(key)
    if cached is not None:
        return cached
    packed = precompute_qkv_u(Uq, Uk, Uv)
    _PACKED_QKV_U_CACHE[key] = packed
    return packed


def project_qkv_rank_packed(
    x: torch.Tensor,
    Uq: torch.Tensor,
    Uk: torch.Tensor,
    Uv: torch.Tensor,
    *,
    packed_u: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R = int(Uq.shape[1])
    if Uk.shape != Uq.shape or Uv.shape != Uq.shape:
        raise ValueError(f"Packed QKV rank projection expects matching U shapes, got {tuple(Uq.shape)}, {tuple(Uk.shape)}, {tuple(Uv.shape)}")
    u_cat = packed_u if packed_u is not None else get_precomputed_qkv_u(Uq, Uk, Uv)
    p_cat = x.matmul(u_cat)
    return p_cat.split((R, R, R), dim=-1)


class FlashSVDRoPEAttention(nn.Module):
    """Low-rank QKV + RoPE + SDPA, with adaptive sliding-window execution."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        rotary_emb,
        *,
        chunk_q: int = 128,
        default_window_radius: Optional[int] = None,
        enable_sliding_chunk: bool = True,
        auto_infer_window: bool = True,
        # Kept for backward compatibility with old constructor.
        bm: int = 64,
        bn: int = 64,
        bdh: Optional[int] = None,
        br: int = 32,
        use_autotune: bool = True,
        pinned_cfg: Optional[dict] = None,
    ):
        super().__init__()
        del bm, bn, br, use_autotune, pinned_cfg
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rotary_emb = rotary_emb
        self.chunk_q = int(chunk_q)
        self.default_window_radius = default_window_radius
        self.enable_sliding_chunk = bool(enable_sliding_chunk)
        self.auto_infer_window = bool(auto_infer_window)

        if bdh is not None:
            assert int(bdh) == self.head_dim, "bdh must match head_dim"

    def _to_v_flat(self, V: torch.Tensor, R: int) -> torch.Tensor:
        H, Dh = self.num_heads, self.head_dim
        if V.dim() == 2:
            if V.shape != (R, H * Dh):
                raise ValueError(f"V shape mismatch. expected {(R, H * Dh)}, got {tuple(V.shape)}")
            return V.contiguous()
        if V.dim() == 3:
            if V.shape == (H, R, Dh):
                return V.permute(1, 0, 2).reshape(R, H * Dh).contiguous()
            if V.shape == (R, H, Dh):
                return V.reshape(R, H * Dh).contiguous()
        raise ValueError(f"Unsupported V shape: {tuple(V.shape)}")

    def _to_bias_flat(self, b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if b is None:
            return None
        H, Dh = self.num_heads, self.head_dim
        if b.dim() == 1:
            if b.numel() != H * Dh:
                raise ValueError(f"bias mismatch. expected {H * Dh}, got {b.numel()}")
            return b.contiguous()
        if b.dim() == 2 and b.shape == (H, Dh):
            return b.reshape(H * Dh).contiguous()
        raise ValueError(f"Unsupported bias shape: {tuple(b.shape)}")

    def _build_rope_tables(
        self,
        position_ids: torch.Tensor,
        B: int,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H, Dh = self.num_heads, self.head_dim
        dummy = torch.empty((B * H, L, Dh), device=device, dtype=dtype)
        posf = position_ids.unsqueeze(1).expand(B, H, L).reshape(B * H, L)
        cos, sin = self.rotary_emb(dummy, position_ids=posf)
        cos = cos.view(B, H, L, Dh).contiguous()
        sin = sin.view(B, H, L, Dh).contiguous()
        return cos, sin

    def _infer_window_radius_from_mask(self, sliding_window_mask: torch.Tensor) -> Optional[int]:
        if sliding_window_mask is None or sliding_window_mask.dim() != 4:
            return None
        _, _, Lq, Lk = sliding_window_mask.shape
        if Lq != Lk:
            return None

        row_idx = Lq // 2
        row = sliding_window_mask[0, 0, row_idx]
        if row.dtype == torch.bool:
            allow = row
        else:
            # HF ModernBERT uses additive mask with 0 allowed / very negative disallowed.
            thresh = -1e4
            allow = row > thresh
        idx = torch.nonzero(allow, as_tuple=False).flatten()
        if idx.numel() == 0:
            return None
        radius = int(torch.max(torch.abs(idx - row_idx)).item())
        return max(radius, 0)

    def _project_packed(self, P: torch.Tensor, V: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
        # P: [B, T, R], V: [R, H*Dh] -> [B, H, T, Dh]
        B, T, _ = P.shape
        H, Dh = self.num_heads, self.head_dim
        out = P.matmul(V)
        if b is not None:
            out = out + b.view(1, 1, H * Dh)
        return out.view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()

    def _to_sdpa_mask(
        self,
        mask: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
        q_start: Optional[int] = None,
        q_end: Optional[int] = None,
        k_start: Optional[int] = None,
        k_end: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        if mask.dim() == 2:
            # 2D padding mask [B, L] with 1 valid / 0 pad
            assert q_start is not None and q_end is not None and k_start is not None and k_end is not None
            q_valid = mask[:, q_start:q_end].to(torch.bool)
            k_valid = mask[:, k_start:k_end].to(torch.bool)
            allow = q_valid[:, :, None] & k_valid[:, None, :]
            add = torch.zeros(
                (allow.shape[0], 1, allow.shape[1], allow.shape[2]),
                device=mask.device,
                dtype=dtype,
            )
            add.masked_fill_(~allow[:, None, :, :], torch.finfo(dtype).min)
            return add

        if mask.dim() == 4:
            if q_start is None:
                add = mask
            else:
                add = mask[:, :, q_start:q_end, k_start:k_end]
            if add.dtype == torch.bool:
                out = torch.zeros_like(add, dtype=dtype)
                out.masked_fill_(~add, torch.finfo(dtype).min)
                return out
            return add.to(dtype)

        raise ValueError(f"Unsupported attention mask shape: {tuple(mask.shape)}")

    @torch.no_grad()
    def forward(
        self,
        qkv_factors: QKVFactors,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        sliding_window_mask: Optional[torch.Tensor] = None,
        window_radius: Optional[int] = None,
    ) -> torch.Tensor:
        Pq, Pk, Pv = qkv_factors.Pq, qkv_factors.Pk, qkv_factors.Pv
        B, L, R = Pq.shape
        H, Dh = self.num_heads, self.head_dim

        if position_ids is None:
            position_ids = torch.arange(L, device=Pq.device).unsqueeze(0).expand(B, L)

        Vq = self._to_v_flat(qkv_factors.Vq, R)
        Vk = self._to_v_flat(qkv_factors.Vk, R)
        Vv = self._to_v_flat(qkv_factors.Vv, R)
        bq = self._to_bias_flat(qkv_factors.bq)
        bk = self._to_bias_flat(qkv_factors.bk)
        bv = self._to_bias_flat(qkv_factors.bv)

        cos, sin = self._build_rope_tables(position_ids, B, L, Pq.device, Pq.dtype)
        q = self._project_packed(Pq, Vq, bq)
        k = self._project_packed(Pk, Vk, bk)
        v = self._project_packed(Pv, Vv, bv)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Prefer explicit sliding mask when provided (ModernBERT path).
        effective_mask = sliding_window_mask if sliding_window_mask is not None else attention_mask

        if window_radius is None:
            window_radius = self.default_window_radius
        if window_radius is None and sliding_window_mask is not None and self.auto_infer_window:
            window_radius = self._infer_window_radius_from_mask(sliding_window_mask)

        if self.enable_sliding_chunk and window_radius is not None and window_radius < (L - 1):
            return self._forward_sliding_chunked_qkv(
                q,
                k,
                v,
                effective_mask,
                attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None,
                window_radius=int(window_radius),
            )

        add_mask = self._to_sdpa_mask(effective_mask, dtype=q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask, dropout_p=0.0, is_causal=False)

        # If only 2D attention mask is provided, explicitly zero padded query rows.
        if attention_mask is not None and attention_mask.dim() == 2:
            q_valid = attention_mask.to(torch.bool)
            out = out * q_valid[:, None, :, None]
        return out

    def _forward_sliding_chunked_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        effective_mask: Optional[torch.Tensor],
        padding_mask_2d: Optional[torch.Tensor],
        *,
        window_radius: int,
    ) -> torch.Tensor:
        B, H, L, Dh = q.shape
        out = torch.empty((B, H, L, Dh), device=q.device, dtype=q.dtype)

        for q0 in range(0, L, self.chunk_q):
            q1 = min(L, q0 + self.chunk_q)
            k0 = max(0, q0 - window_radius)
            k1 = min(L, q1 + window_radius)

            if effective_mask is not None and effective_mask.dim() == 4:
                add_mask = self._to_sdpa_mask(
                    effective_mask,
                    dtype=q.dtype,
                    q_start=q0,
                    q_end=q1,
                    k_start=k0,
                    k_end=k1,
                )
            elif padding_mask_2d is not None:
                add_mask = self._to_sdpa_mask(
                    padding_mask_2d,
                    dtype=q.dtype,
                    q_start=q0,
                    q_end=q1,
                    k_start=k0,
                    k_end=k1,
                )
            else:
                add_mask = None

            o = F.scaled_dot_product_attention(
                q[:, :, q0:q1, :],
                k[:, :, k0:k1, :],
                v[:, :, k0:k1, :],
                attn_mask=add_mask,
                dropout_p=0.0,
                is_causal=False,
            )

            if padding_mask_2d is not None:
                q_valid = padding_mask_2d[:, q0:q1].to(torch.bool)
                o = o * q_valid[:, None, :, None]

            out[:, :, q0:q1, :] = o

        return out


# Alias for readability in some scripts.
FlashSVDRoPEAttentionEncoder = FlashSVDRoPEAttention


def _make_modernbert_rotary_compat(cfg, dh: int, device: Optional[torch.device] = None):
    from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding

    base = float(getattr(cfg, "local_rope_theta", getattr(cfg, "global_rope_theta", 10000.0)))
    builders = [
        lambda: ModernBertRotaryEmbedding(cfg, dim=dh, base=base),
        lambda: ModernBertRotaryEmbedding(cfg, dh, base),
        lambda: ModernBertRotaryEmbedding(cfg),
        lambda: ModernBertRotaryEmbedding(dh, base),
    ]
    last_err = None
    for b in builders:
        try:
            rot = b()
            if device is not None and hasattr(rot, "to"):
                rot = rot.to(device)
            return rot
        except TypeError as e:
            last_err = e
            continue
    raise TypeError(f"Unable to construct ModernBertRotaryEmbedding for this transformers version: {last_err}")


if __name__ == "__main__":
    # quick smoke test
    assert torch.cuda.is_available(), "CUDA is required"
    from transformers import ModernBertConfig
    torch.manual_seed(0)
    cfg = ModernBertConfig()

    B, L = 2, 256
    H = cfg.num_attention_heads
    Dh = cfg.hidden_size // cfg.num_attention_heads
    R = 192
    dtype = torch.bfloat16
    device = "cuda"
    rotary = _make_modernbert_rotary_compat(cfg, Dh, device=None)

    qkv = QKVFactors(
        Pq=torch.randn(B, L, R, device=device, dtype=dtype),
        Pk=torch.randn(B, L, R, device=device, dtype=dtype),
        Pv=torch.randn(B, L, R, device=device, dtype=dtype),
        Vq=torch.randn(R, H * Dh, device=device, dtype=dtype),
        Vk=torch.randn(R, H * Dh, device=device, dtype=dtype),
        Vv=torch.randn(R, H * Dh, device=device, dtype=dtype),
        bq=torch.zeros(H * Dh, device=device, dtype=dtype),
        bk=torch.zeros(H * Dh, device=device, dtype=dtype),
        bv=torch.zeros(H * Dh, device=device, dtype=dtype),
    )

    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
    am = torch.ones(B, L, device=device, dtype=torch.int32)

    mod = FlashSVDRoPEAttention(H, Dh, rotary, default_window_radius=cfg.local_attention // 2).to(device)
    out = mod(qkv, attention_mask=am, position_ids=pos, sliding_window_mask=None)
    print("[flashsvd_ropeattn_v1.5_encoder] out:", tuple(out.shape), out.dtype)
