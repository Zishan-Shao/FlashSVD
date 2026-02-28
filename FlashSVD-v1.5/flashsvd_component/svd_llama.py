import math
import os
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig

from kernels.flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors
from kernels.flashsvdswiglu import flashsvd_ffn_swiglu

import importlib.util
from pathlib import Path


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


_DECODE_ATTN_MOD = None
_LRKV_MHA_SDPA_WARNED = False
_LRKV_RANK_MISMATCH_WARNED = False


def _get_flashsvd_decode_attn_mod():
    global _DECODE_ATTN_MOD
    if _DECODE_ATTN_MOD is not None:
        return _DECODE_ATTN_MOD

    path = (
        Path(__file__).resolve().parents[1]
        / "kernels"
        / "flashsvd-v1.5"
        / "flashsvdropeattn"
        / "flashsvdropeattn_v1.6_decode_opt.py"
    )
    spec = importlib.util.spec_from_file_location("flashsvdropeattn_v16_decode_opt", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load decode attention module spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _DECODE_ATTN_MOD = mod
    return mod

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

    def forward(self, x):
        # Fast path: Triton FlashSVD SwiGLU on CUDA using shared rank-space P
        if (
            x.is_cuda
            and os.getenv("SVDLLM_FLASH_FALLBACK", "0") == "0"
            and os.getenv("FLASH_SVD_DISABLE_FFN", "0") == "0"
        ):
            B, L, _ = x.shape
            R1 = self.up_v_proj.out_features
            D = self.up_u_proj.out_features

            # Rank-space input P via one low-rank projection (shared)
            P = self.up_v_proj(x)  # [B, L, R1]

            # Combine rank->intermediate factors for up and gate into V1 = [R1, 2D]
            V1u = self.up_u_proj.weight.t()    # [R1, D]
            V1v = self.gate_u_proj.weight.t()  # [R1, D]
            V1 = torch.cat([V1u, V1v], dim=1)

            # Down path factors
            U2 = self.down_v_proj.weight.t()   # [D,  R2]
            V2 = self.down_u_proj.weight.t()   # [R2, H]

            # Biases are absent in this module; pass zeros
            b1 = torch.zeros(2 * D, device=x.device, dtype=x.dtype)
            b2 = torch.zeros(V2.shape[1], device=x.device, dtype=x.dtype)

            y = flashsvd_ffn_swiglu(P, V1, U2, V2, b1, b2, use_autotune=True)
            return y

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
        self.ratio = ratio # 1 means no truncate, just keep normal attn
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
        low_rank = int(self.hidden_size * self.ratio/2)
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
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
        if past_key_value is None and past_key_values is not None:
            past_key_value = past_key_values
        bsz, q_len, _ = hidden_states.size()
        fix_pad_query_mask = bool(getattr(self, "fix_pad_query_mask", False))
        debug_pad_query_mask = bool(getattr(self, "debug_pad_query_mask", False))

        # KV-cache path: follow HF semantics (Cache.update happens inside attention) and return 2-tuple.
        # This enables `model.generate()` decoding in transformers>=4.4x.
        if past_key_value is not None or use_cache:
            # Low-rank KV-cache path: cache rank-space Pk/Pv (pre-RoPE) and use Triton decode kernel when q_len==1.
            try:
                from flashsvd_component.lowrank_cache import LowRankKVCache
            except Exception:
                LowRankKVCache = None  # type: ignore[assignment]

            if LowRankKVCache is not None and isinstance(past_key_value, LowRankKVCache):
                baseline_lr_kvcache = os.getenv("FLASH_SVD_BASELINE_LR_KVCACHE", "0") != "0"
                force_flashsvd_kernel = os.getenv("FLASH_SVD_FORCE_ATTENTION_KERNEL", "0") != "0"
                auto_mha_sdpa = os.getenv("FLASH_SVD_AUTO_MHA_SDPA", "0") != "0"
                try:
                    mha_sdpa_r_thr = int(os.getenv("FLASH_SVD_MHA_SDPA_R_THRESHOLD", "512"))
                except Exception:
                    mha_sdpa_r_thr = 512

                H = int(getattr(self, "num_heads", 0) or 0)
                Hk = int(getattr(self, "num_key_value_heads", H) or H)
                rep = max(1, H // max(1, Hk))
                R_attn = int(getattr(getattr(self, "q_v_proj", None), "out_features", 0) or 0)

                # Auto policy: for REP==1 (non-GQA) and large ranks, FlashSVD decode kernels
                # tend to be bandwidth-dominated and can underperform an SDPA baseline that
                # reconstructs K/V for all heads via GEMMs.
                auto_sdpa = bool(auto_mha_sdpa and rep == 1 and R_attn >= int(mha_sdpa_r_thr))
                use_flashsvd_kernel = bool(
                    (not baseline_lr_kvcache)
                    and (force_flashsvd_kernel or (not auto_sdpa))
                )
                if auto_sdpa and (not baseline_lr_kvcache) and (not force_flashsvd_kernel):
                    global _LRKV_MHA_SDPA_WARNED
                    if not _LRKV_MHA_SDPA_WARNED:
                        print(
                            "[FlashSVD] LowRankKVCache: detected REP=1 (non-GQA) with "
                            f"R={R_attn} >= {mha_sdpa_r_thr}; using SDPA baseline instead of FlashSVD kernels. "
                            "Set FLASH_SVD_FORCE_ATTENTION_KERNEL=1 to override."
                        )
                        _LRKV_MHA_SDPA_WARNED = True

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
                    Pq = self.q_v_proj(hidden_states)  # [B, q, R]
                    Pk_step = self.k_v_proj(hidden_states)  # [B, q, R]
                    Pv_step = self.v_v_proj(hidden_states)  # [B, q, R]
                    ev_proj_e.record()
                    evs["proj"].append((ev_proj_s, ev_proj_e))
                else:
                    Pq = self.q_v_proj(hidden_states)  # [B, q, R]
                    Pk_step = self.k_v_proj(hidden_states)  # [B, q, R]
                    Pv_step = self.v_v_proj(hidden_states)  # [B, q, R]

                # Some checkpoints use different ranks for Q vs K/V (e.g., adaptive-rank variants).
                # FlashSVD attention kernels currently assume a shared rank across Q/K/V.
                Rq = int(Pq.shape[-1])
                Rk = int(Pk_step.shape[-1])
                Rv = int(Pv_step.shape[-1])
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
                    past_key_value.update(Pk_step, Pv_step, layer_idx, {"cache_position": cache_position})
                    ev_cache_e.record()
                    evs["cache"].append((ev_cache_s, ev_cache_e))
                else:
                    past_key_value.update(Pk_step, Pv_step, layer_idx, {"cache_position": cache_position})

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
                    R = int(Pq.shape[-1])
                    Dh = self.head_dim

                    # MHA-specialized streamed decode path (REP==1): avoid SDPA peak memory and
                    # avoid per-head small GEMMs in Triton by using head-fused GEMMs (k_u_proj)
                    # on chunks of the KV cache, then online-softmax + rank-space value accumulation.
                    # Enable with: FLASH_SVD_DECODE_MHA_STREAM=1
                    try:
                        use_mha_stream = (
                            os.getenv("FLASH_SVD_DECODE_MHA_STREAM", "0") != "0"
                            and hidden_states.is_cuda
                            and torch.cuda.is_available()
                            and int(getattr(self, "num_heads", 0) or 0)
                            == int(getattr(self, "num_key_value_heads", getattr(self, "num_heads", 0)) or 0)
                        )
                    except Exception:
                        use_mha_stream = False

                    if not use_flashsvd_kernel:
                        # Baseline path: LowRankKVCache storage, but compute attention with SDPA
                        # (i.e., no FlashSVD attention kernels). This is primarily for A/B timing.
                        # NOTE: This reconstructs dense K/V from rank cache each step, so it can be slow.
                        def _apply_rope_tables(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
                            half = x.shape[-1] // 2
                            x0 = x[..., :half]
                            x1 = x[..., half:]
                            y0 = x0 * cos - x1 * sin
                            y1 = x1 * cos + x0 * sin
                            return torch.cat((y0, y1), dim=-1)

                        # Query in head-space: [B, H, 1, Dh]
                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, self.head_dim)
                        query_states = self.q_u_proj(Pq).view(hidden_shape).transpose(1, 2)

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
                        query_states = _apply_rope_tables(query_states, cos_q, sin_q)

                        cos_k = rotary_cos[:seqlen_k].view(1, 1, seqlen_k, Dh // 2)
                        sin_k = rotary_sin[:seqlen_k].view(1, 1, seqlen_k, Dh // 2)
                        key_states = _apply_rope_tables(key_states, cos_k, sin_k)

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
                        def _apply_rope_tables(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
                            half = x.shape[-1] // 2
                            x0 = x[..., :half]
                            x1 = x[..., half:]
                            y0 = x0 * cos - x1 * sin
                            y1 = x1 * cos + x0 * sin
                            return torch.cat((y0, y1), dim=-1)

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
                        split_k = max(1, int(os.getenv("FLASH_SVD_DECODE_SPLIT_K", "512")))
                        scale = float(self.scaling)

                        if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                            ev_kernel_s = torch.cuda.Event(enable_timing=True)
                            ev_kernel_e = torch.cuda.Event(enable_timing=True)
                            ev_kernel_s.record()

                        query_states = self.q_u_proj(Pq).view(hidden_shape).transpose(1, 2)  # [B,H,1,Dh]
                        cos_q = rotary_cos[pos_q].view(1, 1, 1, Dh // 2)
                        sin_q = rotary_sin[pos_q].view(1, 1, 1, Dh // 2)
                        query_states = _apply_rope_tables(query_states, cos_q, sin_q)

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
                            k_bhsd = _apply_rope_tables(k_bhsd, cos_k, sin_k)

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
                    Pq_q = Pq[:, 0, :].unsqueeze(1).expand(bsz, H, R)

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

                    mod = _get_flashsvd_decode_attn_mod()
                    f = mod.DecodePackedFactors(Pq=Pq_q, Pk=Pk, Pv=Pv, Vq=Vq, Vk=Vk, Vv=Vv)

                    # Persistent workspace/buffers (slice to current num_splits)
                    split_k = int(os.getenv("FLASH_SVD_DECODE_SPLIT_K", "512"))
                    bn = int(os.getenv("FLASH_SVD_DECODE_BN", "64"))
                    br = int(os.getenv("FLASH_SVD_DECODE_BR", "64"))
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
                    # budget_bytes defaults to 64KiB (fits comfortably on A100 even with pipelining).
                    try:
                        budget_bytes = int(os.getenv("FLASH_SVD_DECODE_SMEM_BUDGET", str(64 * 1024)))
                    except Exception:
                        budget_bytes = 64 * 1024
                    if R > 0:
                        bn_max = max(16, budget_bytes // max(1, R * bytes_per_elem))
                        bn = min(bn, bn_max)
                    # Round BN down to a power-of-two divisor of split_k (required by wrapper asserts / tuning).
                    if bn < 16:
                        bn = 16
                    # power-of-two floor
                    bn = 1 << (int(bn).bit_length() - 1)
                    # ensure divides split_k
                    while bn > 16 and (split_k % bn) != 0:
                        bn //= 2

                    # For very large ranks, also reduce pipeline stages (SMEM is often double-buffered).
                    num_warps_stage1 = int(os.getenv("FLASH_SVD_DECODE_WARPS1", "4"))
                    num_stages_stage1 = int(os.getenv("FLASH_SVD_DECODE_STAGES1", "2"))
                    num_warps_stage2 = int(os.getenv("FLASH_SVD_DECODE_WARPS2", "4"))
                    num_stages_stage2 = int(os.getenv("FLASH_SVD_DECODE_STAGES2", "1"))
                    if R >= 512:
                        vk_resident = False
                        num_stages_stage1 = min(num_stages_stage1, 1)

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

                    if do_prof and not getattr(self, "_attn_decode_prof_done", False):  # type: ignore[attr-defined]
                        ev_k_s = torch.cuda.Event(enable_timing=True)
                        ev_k_e = torch.cuda.Event(enable_timing=True)
                        ev_k_s.record()
                        O_bhd = mod.flashsvd_attn_decode_packed(
                            f,
                            rotary_cos,
                            rotary_sin,
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
                        )  # [B, H, Dh]
                        ev_k_e.record()
                        evs["kernel"].append((ev_k_s, ev_k_e))
                    else:
                        O_bhd = mod.flashsvd_attn_decode_packed(
                            f,
                            rotary_cos,
                            rotary_sin,
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
                        )  # [B, H, Dh]

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
                B, M, R = Pq.shape
                H, dh = self.num_heads, self.head_dim

                if not use_flashsvd_kernel:
                    # Baseline path (no FlashSVD kernels): build dense Q/K/V from rank factors and use SDPA.
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.head_dim)

                    query_states = self.q_u_proj(Pq).view(hidden_shape).transpose(1, 2)
                    key_states = self.k_u_proj(Pk_step).view(hidden_shape).transpose(1, 2)
                    value_states = self.v_u_proj(Pv_step).view(hidden_shape).transpose(1, 2)

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
                Pq4 = Pq.unsqueeze(1).expand(B, H, M, R)
                Pk4 = Pk_step.unsqueeze(1).expand(B, Hk, M, R)
                Pv4 = Pv_step.unsqueeze(1).expand(B, Hk, M, R)

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
                layer_idx = int(getattr(self, "layer_idx", 0))
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

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
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
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

        if os.getenv("SVDLLM_FLASH_FALLBACK", "0") != "0":
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
    
