# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: this code is directly copied and modified from the huggingface transformer

from typing import Callable, Optional, Union
import math

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub
from ...masking_utils import create_causal_mask
# from ...modeling_layers import (
#     GenericForQuestionAnswering,
#     GenericForSequenceClassification,
#     GenericForTokenClassification,
#     GradientCheckpointingLayer,
# )
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.generic import check_model_inputs
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)


# FlashSVD integrations
try:
    # SwiGLU FFN kernel (P @ V1 -> split -> SiLU* -> @ U2 -> @ V2)
    from .flashsvdswiglu import flashsvd_ffn_swiglu  # type: ignore
    # RoPE-aware FlashSVD attention kernel and factors struct
    from .flashsvdropeattn import FlashSVDRoPEAttention, QKVFactors  # type: ignore
    _FLASH_SVD_AVAILABLE = True
except Exception:
    _FLASH_SVD_AVAILABLE = False

# --- Low-rank SVD utilities -------------------------------------------------
class LowRankLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that factors W as U @ V with a user-specified rank.
    Forward computes: y = (x @ V^T) @ U^T + bias, so inference only touches the low-rank factors.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, device=None, dtype=None, init: str = "factorized"):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self._init_mode = str(init)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.U = nn.Parameter(torch.empty(self.out_features, self.rank, **factory_kwargs))
        self.V = nn.Parameter(torch.empty(self.rank, self.in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(self.out_features, **factory_kwargs)) if bias else None
        self.reset_parameters()
    
    def reset_parameters(self):
        # Two options:
        #  - "factorized" (default): init U and V with xavier-like scaling and 1/sqrt(rank) dampening
        #  - "svd_xavier": sample dense W with xavier_uniform, then SVD to rank-r (costly, but exact W init)
        if self._init_mode == "svd_xavier":
            W = torch.empty(self.out_features, self.in_features, device=self.U.device, dtype=torch.float32)
            nn.init.xavier_uniform_(W)
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                r = min(self.rank, U.shape[1], Vh.shape[0])
                U_lr = U[:, :r] * S[:r].unsqueeze(0)
                V_r = Vh[:r, :]
                self.U.copy_(U_lr.to(self.U.dtype))
                self.V.copy_(V_r.to(self.V.dtype))
        else:
            # factorized
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
            with torch.no_grad():
                scale = 1.0 / math.sqrt(max(1, self.rank))
                self.U.mul_(scale)
                self.V.mul_(scale)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tmp = nn.functional.linear(input, self.V, None)   # (..., rank)
        out = nn.functional.linear(tmp, self.U, self.bias) # (..., out_features)
        return out

    @staticmethod
    def from_linear(linear: nn.Linear, rank: int) -> "LowRankLinear":
        # Compute rank-r SVD of W and fold Σ into U so we keep two matmuls at inference.
        W = linear.weight.detach()
        orig_dtype = W.dtype
        device = W.device
        W32 = W.to(torch.float32)
        try:
            U, S, Vh = torch.linalg.svd(W32, full_matrices=False)
        except Exception:
            U, S, Vh = torch.linalg.svd(W32.cpu(), full_matrices=False)
            U, S, Vh = U.to(device), S.to(device), Vh.to(device)
        r = max(1, min(int(rank), U.shape[1], Vh.shape[0]))
        U_r = U[:, :r].contiguous()
        S_r = S[:r]
        V_r = Vh[:r, :].contiguous()
        U_lr = U_r * S_r.unsqueeze(0)  # fold Σ into U

        lr = LowRankLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=r,
            bias=linear.bias is not None,
            device=device,
            dtype=orig_dtype,
        )
        with torch.no_grad():
            lr.U.copy_(U_lr.to(orig_dtype))
            lr.V.copy_(V_r.to(orig_dtype))
            if linear.bias is not None:
                lr.bias.copy_(linear.bias.detach().to(orig_dtype))
        return lr

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"

    
    # Allow loading checkpoints that store a dense weight at "<prefix>.weight"
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        w_key = prefix + "weight"
        if w_key in state_dict:
            # Convert dense weight to U/V on the fly via rank-r SVD.
            W = state_dict.pop(w_key).to(torch.float32)
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
                U, S, Vh = U.to(self.U.device), S.to(self.U.device), Vh.to(self.V.device)
            r = min(self.rank, U.shape[1], Vh.shape[0])
            U_lr = U[:, :r] * S[:r].unsqueeze(0)
            V_r = Vh[:r, :]
            with torch.no_grad():
                self.U.copy_(U_lr.to(self.U.dtype))
                self.V.copy_(V_r.to(self.V.dtype))
            # bias (if present) is handled by the base loader below
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )




@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class FlashSVDSwiGLUMLP(nn.Module):
    """FlashSVD SwiGLU MLP drop-in aligned with LLaMA FFN (SwiGLU).

    Constructs rank-space factors by SVD on the concatenated first layer
    [gate_proj; up_proj] and the down_proj.

    - U1: [hidden, r_ff]
    - V1: [r_ff, 2*intermediate]
    - U2: [intermediate, r_ff]
    - V2: [r_ff, hidden]
    - b1: [2*intermediate] (concat biases of gate/up, zeros if None)
    - b2: [hidden] (down bias or zeros)
    """

    def __init__(self, mlp: LlamaMLP, rank_ff: int, *, block_cfg: Optional[dict] = None):
        super().__init__()
        if not _FLASH_SVD_AVAILABLE:
            raise RuntimeError("FlashSVD kernels not available; cannot construct FlashSVDSwiGLUMLP")
        self.hidden_size = mlp.hidden_size
        self.intermediate_size = mlp.intermediate_size
        self.rank_ff = int(max(1, rank_ff))
        self.block_cfg = block_cfg or {"BL": 64, "BD": 64, "BR1": 64, "BR2": 64}

        # Build concatenated first-layer weight: [W_gate; W_up] with shape (2D, H)
        W_gate = mlp.gate_proj.weight.detach().to(torch.float32)  # [D, H]
        W_up = mlp.up_proj.weight.detach().to(torch.float32)      # [D, H]
        W_cat = torch.cat([W_gate, W_up], dim=0)                  # [2D, H]
        # SVD of W_cat^T (H x 2D)
        W_cat_T = W_cat.t()
        try:
            U, S, Vh = torch.linalg.svd(W_cat_T, full_matrices=False)
        except Exception:
            U, S, Vh = torch.linalg.svd(W_cat_T.cpu(), full_matrices=False)
            U, S, Vh = U.to(W_cat_T.device), S.to(W_cat_T.device), Vh.to(W_cat_T.device)
        r1 = min(self.rank_ff, U.shape[1], Vh.shape[0])
        U1 = (U[:, :r1] * S[:r1].unsqueeze(0)).to(mlp.gate_proj.weight.dtype)  # [H, r1]
        V1 = Vh[:r1, :].to(mlp.gate_proj.weight.dtype)                          # [r1, 2D]

        # Down projection SVD: W_down^T (D x H)
        W_down_T = mlp.down_proj.weight.detach().to(torch.float32).t()  # [D, H]
        try:
            U_d, S_d, Vh_d = torch.linalg.svd(W_down_T, full_matrices=False)
        except Exception:
            U_d, S_d, Vh_d = torch.linalg.svd(W_down_T.cpu(), full_matrices=False)
            U_d, S_d, Vh_d = U_d.to(W_down_T.device), S_d.to(W_down_T.device), Vh_d.to(W_down_T.device)
        r2 = min(self.rank_ff, U_d.shape[1], Vh_d.shape[0])
        U2 = (U_d[:, :r2] * S_d[:r2].unsqueeze(0)).to(mlp.down_proj.weight.dtype)  # [D, r2]
        V2 = Vh_d[:r2, :].to(mlp.down_proj.weight.dtype)                            # [r2, H]

        # Biases
        b_gate = mlp.gate_proj.bias.detach() if mlp.gate_proj.bias is not None else torch.zeros(self.intermediate_size, dtype=mlp.gate_proj.weight.dtype, device=mlp.gate_proj.weight.device)
        b_up = mlp.up_proj.bias.detach() if mlp.up_proj.bias is not None else torch.zeros(self.intermediate_size, dtype=mlp.up_proj.weight.dtype, device=mlp.up_proj.weight.device)
        b1 = torch.cat([b_gate.to(V1.dtype), b_up.to(V1.dtype)], dim=0)  # [2D]
        b2 = mlp.down_proj.bias.detach() if mlp.down_proj.bias is not None else torch.zeros(self.hidden_size, dtype=mlp.down_proj.weight.dtype, device=mlp.down_proj.weight.device)

        # Store as buffers (inference only)
        self.register_buffer("U1", U1.contiguous(), persistent=False)  # [H, r1]
        self.register_buffer("V1", V1.contiguous(), persistent=False)  # [r1, 2D]
        self.register_buffer("U2", U2.contiguous(), persistent=False)  # [D, r2]
        self.register_buffer("V2", V2.contiguous(), persistent=False)  # [r2, H]
        self.register_buffer("b1", b1.contiguous(), persistent=False)
        self.register_buffer("b2", b2.to(V2.dtype).contiguous(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, H]
        P = x.matmul(self.U1)  # [B, L, r1]
        cfg = self.block_cfg
        out = flashsvd_ffn_swiglu(
            P, self.V1, self.U2, self.V2, self.b1, self.b2,
            BL=cfg.get("BL", 64), BD=cfg.get("BD", 64), BR1=cfg.get("BR1", 64), BR2=cfg.get("BR2", 64),
        )
        return out

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class FlashSVDLlamaAttention(nn.Module):
    """RoPE-aware FlashSVD attention drop-in.

    Falls back to the original dense/eager path when past_key_values is provided,
    since the current kernel path does not manage KV cache.
    """

    def __init__(self, base_attn: LlamaAttention, rotary_emb: nn.Module, rank_attn: int, *, bm: int = 64, bn: int = 64):
        super().__init__()
        if not _FLASH_SVD_AVAILABLE:
            raise RuntimeError("FlashSVD kernels not available; cannot construct FlashSVDLlamaAttention")
        self.config = base_attn.config
        self.layer_idx = base_attn.layer_idx
        self.head_dim = base_attn.head_dim
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = base_attn.num_key_value_groups
        self.scaling = base_attn.scaling
        self.attention_dropout = base_attn.attention_dropout
        self.is_causal = True

        # Keep output projection dense
        self.o_proj = base_attn.o_proj

        # Factorize q/k/v weights: W^T SVD → U (H, r) & V (r, H*dh)
        d_model = self.config.hidden_size
        Dq = self.num_heads * self.head_dim
        Dkv = self.num_kv_heads * self.head_dim
        dtype = base_attn.q_proj.weight.dtype

        def svd_uv(weight: torch.Tensor, r: int):
            WT = weight.detach().to(torch.float32).t()  # [d_model, Dout]
            try:
                U, S, Vh = torch.linalg.svd(WT, full_matrices=False)
            except Exception:
                U, S, Vh = torch.linalg.svd(WT.cpu(), full_matrices=False)
                U, S, Vh = U.to(WT.device), S.to(WT.device), Vh.to(WT.device)
            rr = min(r, U.shape[1], Vh.shape[0])
            U_lr = (U[:, :rr] * S[:rr].unsqueeze(0)).to(dtype)   # [d_model, r]
            V_lr = Vh[:rr, :].to(dtype)                          # [r, Dout]
            return U_lr.contiguous(), V_lr.contiguous()

        rank = int(max(1, rank_attn))
        Uq, Vq = svd_uv(base_attn.q_proj.weight, rank)
        Uk, Vk = svd_uv(base_attn.k_proj.weight, rank)
        Uv, Vv = svd_uv(base_attn.v_proj.weight, rank)

        # Expand K/V factors from H_kv heads to H heads by repetition across groups
        n_rep = self.num_heads // self.num_kv_heads
        if Dkv != Dq and n_rep > 1:
            # Vk/Vv: [r, H_kv*dh] -> [r, H*dh]
            Vk = Vk.view(Vk.shape[0], self.num_kv_heads, self.head_dim).repeat(1, n_rep, 1).reshape(Vk.shape[0], Dq)
            Vv = Vv.view(Vv.shape[0], self.num_kv_heads, self.head_dim).repeat(1, n_rep, 1).reshape(Vv.shape[0], Dq)

        self.register_buffer("Uq", Uq, persistent=False)
        self.register_buffer("Vq", Vq, persistent=False)
        self.register_buffer("Uk", Uk, persistent=False)
        self.register_buffer("Vk", Vk.contiguous(), persistent=False)
        self.register_buffer("Uv", Uv, persistent=False)
        self.register_buffer("Vv", Vv.contiguous(), persistent=False)

        # Optional biases flattened over heads
        bq = base_attn.q_proj.bias.detach() if base_attn.q_proj.bias is not None else None
        bk = base_attn.k_proj.bias.detach() if base_attn.k_proj.bias is not None else None
        bv = base_attn.v_proj.bias.detach() if base_attn.v_proj.bias is not None else None
        if bq is not None:
            self.register_buffer("bq", bq.contiguous(), persistent=False)
        else:
            self.bq = None
        if bk is not None:
            if Dkv != Dq and n_rep > 1:
                bk = bk.view(self.num_kv_heads, self.head_dim).repeat(n_rep, 1).reshape(Dq)
            self.register_buffer("bk", bk.contiguous(), persistent=False)
        else:
            self.bk = None
        if bv is not None:
            if Dkv != Dq and n_rep > 1:
                bv = bv.view(self.num_kv_heads, self.head_dim).repeat(n_rep, 1).reshape(Dq)
            self.register_buffer("bv", bv.contiguous(), persistent=False)
        else:
            self.bv = None

        # Rotary and kernel
        self.rotary_emb = rotary_emb
        self.flash = FlashSVDRoPEAttention(self.config.num_attention_heads, self.head_dim, self.rotary_emb, bm=bm, bn=bn)

        # Fallback eager path for cache usage
        self._fallback = base_attn

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # If cache is in use, fallback to dense path to preserve correctness
        if past_key_values is not None:
            return self._fallback(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        B, M, Hdim = hidden_states.shape
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) if cache_position is not None else torch.arange(M, device=hidden_states.device)[None, :].expand(B, -1)

        # Rank-space projections
        Pq = hidden_states.matmul(self.Uq)  # [B, M, r]
        Pk = hidden_states.matmul(self.Uk)
        Pv = hidden_states.matmul(self.Uv)

        qkv = QKVFactors(
            Pq=Pq, Pk=Pk, Pv=Pv,
            Vq=self.Vq, Vk=self.Vk, Vv=self.Vv,
            bq=self.bq, bk=self.bk, bv=self.bv,
        )

        O = self.flash(qkv, attention_mask=attention_mask, position_ids=position_ids)
        attn_output = O.transpose(1, 2).contiguous().view(B, M, Hdim)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LlamaDecoderLayer,
        "attentions": LlamaAttention,
    }


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def apply_flashsvd(
        self,
        rank_attn: Optional[int] = None,
        rank_ff: Optional[int] = None,
        *,
        attn_bm: int = 64,
        attn_bn: int = 64,
        ffn_block_cfg: Optional[dict] = None,
    ) -> None:
        """Replace Attention and/or MLP with FlashSVD kernels for inference.

        - Attention uses RoPE-aware streaming kernel without KV cache support; if KV cache is
          needed at runtime, the dense fallback path will be used automatically.
        - FFN uses GEGLU FlashSVD kernel. This matches LLaMA's SwiGLU path structure.
        """
        if not _FLASH_SVD_AVAILABLE:
            logger.warning("FlashSVD not available; apply_flashsvd is a no-op.")
            return

        for layer in self.model.layers:
            if rank_attn is not None and rank_attn > 0:
                base_attn: LlamaAttention = layer.self_attn
                rotary = getattr(self.model, "rotary_emb", None)
                # The flash module expects a rotary emb callable compatible with kernel
                attn_mod = FlashSVDLlamaAttention(base_attn, rotary, rank_attn, bm=attn_bm, bn=attn_bn)
                layer.self_attn = attn_mod

            if rank_ff is not None and rank_ff > 0:
                base_mlp: LlamaMLP = layer.mlp
                layer.mlp = FlashSVDSwiGLUMLP(base_mlp, rank_ff, block_cfg=ffn_block_cfg)

    @torch.no_grad()
    def apply_svd(
        self,
        rank_q: Optional[int] = None,
        rank_kv: Optional[int] = None,
        rank_o: Optional[int] = None,
        rank_ff: Optional[int] = None,
    ) -> None:
        """
        Replace attention and MLP Linear layers with LowRankLinear using rank‑r SVD.

        - q_proj:       rank_q
        - k_proj, v_proj: rank_kv
        - o_proj:       rank_o
        - gate_proj, up_proj, down_proj (MLP): rank_ff

        If a rank is None or >= min(out_features, in_features), the original layer is kept.
        """
        for layer in self.model.layers:
            attn = layer.self_attn
            mlp  = layer.mlp

            # Attention projections
            def maybe_replace_linear(module: nn.Module, attr: str, rank: Optional[int]):
                if rank is None:
                    return
                lin: nn.Linear = getattr(module, attr)
                max_rank = min(lin.out_features, lin.in_features)
                if rank >= max_rank:
                    return
                lr = LowRankLinear.from_linear(lin, rank=rank)
                setattr(module, attr, lr)

            maybe_replace_linear(attn, "q_proj", rank_q)
            maybe_replace_linear(attn, "k_proj", rank_kv)
            maybe_replace_linear(attn, "v_proj", rank_kv)
            maybe_replace_linear(attn, "o_proj", rank_o)

            # MLP projections
            maybe_replace_linear(mlp, "gate_proj", rank_ff)
            maybe_replace_linear(mlp, "up_proj",   rank_ff)
            maybe_replace_linear(mlp, "down_proj", rank_ff)

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaForSequenceClassification(GenericForSequenceClassification, LlamaPreTrainedModel): ...


class LlamaForQuestionAnswering(GenericForQuestionAnswering, LlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class LlamaForTokenClassification(GenericForTokenClassification, LlamaPreTrainedModel): ...


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]