from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from transformers.cache_utils import Cache


@dataclass
class _DecodeWorkspace:
    # Stage1 workspace: [B,H,S] float32 (S = max_splits)
    M: torch.Tensor
    L: torch.Tensor
    # Rank accumulator: [B,H,S,R] float32
    Acc: torch.Tensor
    # Precomputed Q halves: [B,H,HALF] dtype
    Q0: torch.Tensor
    Q1: torch.Tensor


class LowRankKVCache(Cache):
    """Low-rank KV cache that stores *rank-space* factors (pre-RoPE).

    We cache:
      - Pk = X @ Wk_v^T  (shape [B, T, Rk])
      - Pv = X @ Wv_v^T  (shape [B, T, Rv])

    During decode we can reconstruct head-space K/V on the fly using the per-head
    basis weights (Uk) and apply RoPE inside the decode kernel.

    This is intended to be used with FlashSVD low-rank attention modules, not
    the original dense HF LlamaAttention.
    """

    is_compileable = False

    def __init__(
        self,
        config,
        *,
        max_batch_size: int,
        max_cache_len: int,
        rank: Optional[int] = None,
        rank_k: Optional[int] = None,
        rank_v: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float16,
        layer_device_map: Optional[dict[int, torch.device | str | int]] = None,
    ) -> None:
        # transformers>=4.57 Cache.__init__ requires either `layers` or
        # `layer_class_to_replicate`; older versions accepted no args.
        try:
            super().__init__(layers=[])
        except TypeError:
            super().__init__()
        self._max_batch_size = int(max_batch_size)
        self._max_cache_len = int(max_cache_len)
        self._dtype = dtype

        # NOTE: some checkpoints use different ranks for K/V (and even vary per-layer).
        # We therefore support per-layer ranks and lazy allocation based on the first
        # observed key_states/value_states shapes in `update()`.
        if rank is not None:
            rank_k = rank_v = int(rank)
        self._init_rank_k = int(rank_k or 0)
        self._init_rank_v = int(rank_v or 0)
        # Keep a legacy `rank` attribute for backward compat; when ranks differ
        # this is only a hint (kernels that assume shared rank will be disabled).
        self.rank = int(self._init_rank_k or self._init_rank_v or 0)

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self._layer_key_rank: list[int] = []
        self._layer_value_rank: list[int] = []
        self._layer_initialized: list[bool] = []
        self._layer_device: list[torch.device] = []

        dev = torch.device(device) if device is not None else None

        for idx in range(int(getattr(config, "num_hidden_layers", 0))):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
                layer_device = torch.device("cuda", layer_device) if isinstance(layer_device, int) else torch.device(layer_device)
            else:
                layer_device = dev if dev is not None else torch.device("cpu")

            self._layer_device.append(layer_device)
            # Initialize with 0-rank tensors unless explicit ranks were provided.
            k_rank0 = int(self._init_rank_k)
            v_rank0 = int(self._init_rank_v)
            self.key_cache.append(torch.zeros((self._max_batch_size, self._max_cache_len, k_rank0), dtype=self._dtype, device=layer_device))
            self.value_cache.append(torch.zeros((self._max_batch_size, self._max_cache_len, v_rank0), dtype=self._dtype, device=layer_device))
            self._layer_key_rank.append(k_rank0)
            self._layer_value_rank.append(v_rank0)
            self._layer_initialized.append(False)

        self._seen_tokens = 0
        self._decode_workspaces: dict[tuple[int, int, int, int, torch.dtype, torch.device], _DecodeWorkspace] = {}
        self._rope_tables: dict[tuple[int, int, float, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

        self._rope_theta = float(getattr(config, "rope_theta", 10000.0))

    @property
    def max_batch_size(self) -> int:
        return int(self._max_batch_size)

    @property
    def max_cache_len(self) -> int:
        return int(self._max_cache_len)

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
        k_rank = int(key_states.shape[-1])
        v_rank = int(value_states.shape[-1])

        # Lazy per-layer allocation (and tolerate initial rank guess mismatches before first write).
        kc = self.key_cache[layer_idx]
        vc = self.value_cache[layer_idx]
        if int(kc.shape[-1]) != k_rank or int(vc.shape[-1]) != v_rank:
            if self._layer_initialized[layer_idx]:
                raise RuntimeError(
                    "LowRankKVCache rank changed after initialization for layer "
                    f"{layer_idx}: cache.key_rank={int(kc.shape[-1])} cache.value_rank={int(vc.shape[-1])} "
                    f"but got key_states.shape={tuple(key_states.shape)} value_states.shape={tuple(value_states.shape)}."
                )
            layer_device = self._layer_device[layer_idx]
            self.key_cache[layer_idx] = torch.zeros((self._max_batch_size, self._max_cache_len, k_rank), dtype=self._dtype, device=layer_device)
            self.value_cache[layer_idx] = torch.zeros((self._max_batch_size, self._max_cache_len, v_rank), dtype=self._dtype, device=layer_device)
            self._layer_key_rank[layer_idx] = int(k_rank)
            self._layer_value_rank[layer_idx] = int(v_rank)
            # Update the legacy hint rank.
            self.rank = int(self.rank or k_rank or v_rank)
            kc = self.key_cache[layer_idx]
            vc = self.value_cache[layer_idx]

        key_states = key_states.to(dtype=kc.dtype, device=kc.device)
        value_states = value_states.to(dtype=vc.dtype, device=vc.device)
        # Consider this layer initialized once we have the first actual write.
        self._layer_initialized[layer_idx] = True

        cache_position = cache_kwargs.get("cache_position", None)
        if cache_position is None:
            # Prefill: overwrite from start.
            self.key_cache[layer_idx][: key_states.shape[0], : key_states.shape[1]].copy_(key_states)
            self.value_cache[layer_idx][: value_states.shape[0], : value_states.shape[1]].copy_(value_states)
            if layer_idx == 0:
                self._seen_tokens = max(int(self._seen_tokens), int(key_states.shape[1]))
            return key_states, value_states

        # Generation: update at specific positions.
        # NOTE: avoid `.item()` on CUDA tensors in the hot decode path; it introduces
        # host-device sync barriers and kills end-to-end decode throughput.
        # We'll update `_seen_tokens` without reading `cache_position` when it's on CUDA
        # (assume sequential generation), and fall back to exact logic for CPU tensors.
        cache_position_cpu = cache_position
        cache_position = cache_position.to(device=self.key_cache[layer_idx].device)
        self.key_cache[layer_idx][: key_states.shape[0]].index_copy_(1, cache_position, key_states)
        self.value_cache[layer_idx][: value_states.shape[0]].index_copy_(1, cache_position, value_states)

        if layer_idx == 0:
            q_len = int(key_states.shape[1])
            # Exact (no GPU sync) when cache_position came from CPU.
            if cache_position_cpu is not None and not getattr(cache_position_cpu, "is_cuda", False):
                try:
                    self._seen_tokens = max(int(self._seen_tokens), int(cache_position_cpu.max().item()) + 1)
                except Exception:
                    self._seen_tokens = max(int(self._seen_tokens), int(self._seen_tokens) + q_len)
            else:
                # Fast path for CUDA cache_position: assume positions are appended sequentially.
                self._seen_tokens = max(int(self._seen_tokens), int(self._seen_tokens) + q_len)
        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:  # noqa: ARG002
        return int(self._seen_tokens)

    def get_max_cache_shape(self) -> Optional[int]:
        return int(self.max_cache_len)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:  # noqa: ARG002
        # Match DynamicCache semantics: kv_length = past + query_length.
        query_length = int(cache_position.shape[0])
        past_seen_tokens = int(self.get_seq_length())
        return query_length + past_seen_tokens, 0

    def get_rope_tables(
        self,
        *,
        seqlen: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) RoPE tables of shape [seqlen, head_dim/2]."""
        seqlen = int(seqlen)
        head_dim = int(head_dim)
        key = (seqlen, head_dim, float(self._rope_theta), dtype, device)
        cached = self._rope_tables.get(key, None)
        if cached is not None:
            return cached

        assert head_dim % 2 == 0
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

    def get_decode_workspace(
        self,
        *,
        batch_size: int,
        num_heads: int,
        rank: int,
        head_dim: int,
        max_splits: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _DecodeWorkspace:
        """Get or allocate persistent decode workspace buffers."""
        batch_size = int(batch_size)
        num_heads = int(num_heads)
        rank = int(rank)
        head_dim = int(head_dim)
        max_splits = int(max_splits)
        assert head_dim % 2 == 0
        half = head_dim // 2

        key = (batch_size, num_heads, rank, max_splits, dtype, device)
        ws = self._decode_workspaces.get(key, None)
        if ws is not None:
            return ws

        M = torch.empty((batch_size, num_heads, max_splits), device=device, dtype=torch.float32)
        L = torch.empty((batch_size, num_heads, max_splits), device=device, dtype=torch.float32)
        Acc = torch.empty((batch_size, num_heads, max_splits, rank), device=device, dtype=torch.float32)
        Q0 = torch.empty((batch_size, num_heads, half), device=device, dtype=dtype)
        Q1 = torch.empty((batch_size, num_heads, half), device=device, dtype=dtype)

        ws = _DecodeWorkspace(M=M, L=L, Acc=Acc, Q0=Q0, Q1=Q1)
        self._decode_workspaces[key] = ws
        return ws
