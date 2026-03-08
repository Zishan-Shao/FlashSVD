from __future__ import annotations

from typing import Any, Optional

import torch

from transformers.cache_utils import Cache


class FlashSVDDenseKVCache(Cache):
    """Dense KV cache with FA2-friendly layout [B, S, Hk, Dh].

    This cache is intended for the short/mid-context FlashSVD decode path:
    - low-rank weights remain in the model
    - only the current token is reconstructed to dense q/k/v
    - K is stored after RoPE so reference PyTorch decode and FlashSVD decode can share
      the same FA2 cache contract
    """

    is_compileable = False

    def __init__(
        self,
        config,
        *,
        max_batch_size: int,
        max_cache_len: int,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float16,
        layer_device_map: Optional[dict[int, torch.device | str | int]] = None,
    ) -> None:
        try:
            super().__init__(layers=[])
        except TypeError:
            super().__init__()

        self._max_batch_size = int(max_batch_size)
        self._max_cache_len = int(max_cache_len)
        self._dtype = dtype
        self._rope_theta = float(getattr(config, "rope_theta", 10000.0))

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._layer_heads: list[int] = []
        self._layer_head_dim: list[int] = []
        self._layer_initialized: list[bool] = []
        self._layer_device: list[torch.device] = []
        self._fa2_cache_seqlens: dict[tuple[int, int], torch.Tensor] = {}

        dev = torch.device(device) if device is not None else None
        n_layers = int(getattr(config, "num_hidden_layers", 0))
        for idx in range(n_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
                layer_device = torch.device("cuda", layer_device) if isinstance(layer_device, int) else torch.device(layer_device)
            else:
                layer_device = dev if dev is not None else torch.device("cpu")
            self._layer_device.append(layer_device)
            self.key_cache.append(torch.zeros((self._max_batch_size, self._max_cache_len, 0, 0), dtype=self._dtype, device=layer_device))
            self.value_cache.append(torch.zeros((self._max_batch_size, self._max_cache_len, 0, 0), dtype=self._dtype, device=layer_device))
            self._layer_heads.append(0)
            self._layer_head_dim.append(0)
            self._layer_initialized.append(False)

        self._seen_tokens = 0
        self._rope_tables: dict[tuple[int, int, float, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_cache_seqlens_buffer(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        layer_idx = int(layer_idx)
        batch_size = int(batch_size)
        key = (layer_idx, batch_size)
        device = self.key_cache[layer_idx].device
        cached = self._fa2_cache_seqlens.get(key)
        if cached is not None and cached.device == device and int(cached.numel()) == batch_size:
            return cached
        buf = torch.empty((batch_size,), device=device, dtype=torch.int32)
        self._fa2_cache_seqlens[key] = buf
        return buf

    @staticmethod
    def _fill_cache_seqlens_buffer(
        cache_seqlens: torch.Tensor,
        *,
        seen_tokens: int,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cache_position is None:
            cache_seqlens.fill_(int(seen_tokens))
            return cache_seqlens

        if torch.is_tensor(cache_position):
            pos = cache_position.to(device=cache_seqlens.device, dtype=torch.int32).reshape(-1)
            if int(pos.numel()) == 0:
                cache_seqlens.zero_()
            elif int(pos.numel()) == 1:
                cache_seqlens.copy_(pos.expand_as(cache_seqlens))
            else:
                cache_seqlens.copy_(pos[: int(cache_seqlens.shape[0])])
            return cache_seqlens

        cache_seqlens.fill_(int(cache_position))
        return cache_seqlens

    @property
    def max_batch_size(self) -> int:
        return int(self._max_batch_size)

    @property
    def max_cache_len(self) -> int:
        return int(self._max_cache_len)

    def _ensure_layer_shape(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        layer_idx = int(layer_idx)
        h = int(key_states.shape[1])
        dh = int(key_states.shape[-1])
        cache = self.key_cache[layer_idx]
        if int(cache.shape[2]) == h and int(cache.shape[3]) == dh:
            return
        if self._layer_initialized[layer_idx]:
            raise RuntimeError(
                f"FlashSVDDenseKVCache shape changed after init for layer {layer_idx}: "
                f"old={(int(cache.shape[2]), int(cache.shape[3]))}, new={(h, dh)}"
            )
        layer_device = self._layer_device[layer_idx]
        self.key_cache[layer_idx] = torch.zeros(
            (self._max_batch_size, self._max_cache_len, h, dh),
            dtype=self._dtype,
            device=layer_device,
        )
        self.value_cache[layer_idx] = torch.zeros_like(self.key_cache[layer_idx])
        self._layer_heads[layer_idx] = h
        self._layer_head_dim[layer_idx] = dh

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_kwargs is None:
            cache_kwargs = {}

        layer_idx = int(layer_idx)
        self._ensure_layer_shape(layer_idx, key_states, value_states)
        kc = self.key_cache[layer_idx]
        vc = self.value_cache[layer_idx]

        key_states = key_states.to(device=kc.device, dtype=kc.dtype)
        value_states = value_states.to(device=vc.device, dtype=vc.dtype)
        key_bshd = key_states.transpose(1, 2).contiguous()
        value_bshd = value_states.transpose(1, 2).contiguous()

        cache_position = cache_kwargs.get("cache_position", None)
        if cache_position is None:
            q_len = int(key_bshd.shape[1])
            kc[: key_bshd.shape[0], :q_len].copy_(key_bshd)
            vc[: value_bshd.shape[0], :q_len].copy_(value_bshd)
            if layer_idx == 0:
                self._seen_tokens = max(int(self._seen_tokens), q_len)
            self._layer_initialized[layer_idx] = True
            return (
                kc[: key_bshd.shape[0], :q_len].transpose(1, 2).contiguous(),
                vc[: value_bshd.shape[0], :q_len].transpose(1, 2).contiguous(),
            )

        cache_position_cpu = cache_position
        cache_position = cache_position.to(device=kc.device)
        kc[: key_bshd.shape[0]].index_copy_(1, cache_position, key_bshd)
        vc[: value_bshd.shape[0]].index_copy_(1, cache_position, value_bshd)
        self._layer_initialized[layer_idx] = True

        if layer_idx == 0:
            q_len = int(key_bshd.shape[1])
            if cache_position_cpu is not None and not getattr(cache_position_cpu, "is_cuda", False):
                try:
                    self._seen_tokens = max(int(self._seen_tokens), int(cache_position_cpu.max().item()) + 1)
                except Exception:
                    self._seen_tokens = max(int(self._seen_tokens), int(self._seen_tokens) + q_len)
            else:
                self._seen_tokens = max(int(self._seen_tokens), int(self._seen_tokens) + q_len)

        seqlen = int(self.get_seq_length())
        return (
            kc[: key_bshd.shape[0], :seqlen].transpose(1, 2).contiguous(),
            vc[: value_bshd.shape[0], :seqlen].transpose(1, 2).contiguous(),
        )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:  # noqa: ARG002
        return int(self._seen_tokens)

    def get_max_cache_shape(self) -> Optional[int]:
        return int(self.max_cache_len)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:  # noqa: ARG002
        query_length = int(cache_position.shape[0])
        past_seen_tokens = int(self.get_seq_length())
        return query_length + past_seen_tokens, 0

    def get_fa2_kvcache(
        self,
        layer_idx: int,
        *,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.prepare_fa2_step(layer_idx, batch_size=batch_size, cache_position=None)

    def prepare_fa2_step(
        self,
        layer_idx: int,
        *,
        batch_size: int,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_idx = int(layer_idx)
        kc = self.key_cache[layer_idx][: int(batch_size)]
        vc = self.value_cache[layer_idx][: int(batch_size)]
        cache_seqlens = self._get_cache_seqlens_buffer(layer_idx, int(batch_size))
        self._fill_cache_seqlens_buffer(
            cache_seqlens,
            seen_tokens=int(self.get_seq_length()),
            cache_position=cache_position,
        )
        return kc, vc, cache_seqlens

    def advance_after_fa2(
        self,
        layer_idx: int,
        *,
        q_len: int,
        cache_position: Optional[torch.Tensor],
    ) -> None:
        if int(layer_idx) != 0:
            return
        q_len = int(q_len)
        if cache_position is not None and not getattr(cache_position, "is_cuda", False):
            try:
                self._seen_tokens = max(int(self._seen_tokens), int(cache_position.max().item()) + 1)
                return
            except Exception:
                pass
        self._seen_tokens = max(int(self._seen_tokens), int(self._seen_tokens) + q_len)

    def get_rope_tables(
        self,
        *,
        seqlen: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seqlen = int(seqlen)
        head_dim = int(head_dim)
        key = (seqlen, head_dim, float(self._rope_theta), dtype, device)
        cached = self._rope_tables.get(key, None)
        if cached is not None:
            return cached

        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        half = head_dim // 2
        pos = torch.arange(seqlen, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            float(self._rope_theta) ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
        )
        ang = torch.einsum("m,d->md", pos, inv_freq)
        cos = torch.cos(ang).to(dtype).contiguous()
        sin = torch.sin(ang).to(dtype).contiguous()
        self._rope_tables[key] = (cos, sin)
        return cos, sin
