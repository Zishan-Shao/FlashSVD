from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return int(value)
    return int(((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple))


def _pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    if qweight.dtype != torch.uint8:
        qweight = qweight.to(torch.uint8)
    if qweight.shape[1] % 2 != 0:
        pad = torch.zeros((qweight.shape[0], 1), dtype=torch.uint8, device=qweight.device)
        qweight = torch.cat([qweight, pad], dim=1)
    lo = qweight[:, 0::2] & 0x0F
    hi = (qweight[:, 1::2] & 0x0F) << 4
    return lo | hi


def _unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out[:, :in_features]


def _pack_tinygemm_scales_and_zeros(scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    if scales.shape != zeros.shape:
        raise ValueError(f"scales/zeros shape mismatch: {tuple(scales.shape)} vs {tuple(zeros.shape)}")
    dtype = scales.dtype
    zeros = zeros.to(dtype=dtype)
    dim = scales.dim()
    return torch.cat([scales.unsqueeze(-1), zeros.unsqueeze(-1)], dim=dim).transpose(-3, -2).contiguous()


class PackedQuantLinear(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bits: int,
        group_size: int,
        bias: bool,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if bits not in (4, 8):
            raise ValueError(f"PackedQuantLinear only supports 4/8 bits, got {bits}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.compute_dtype = compute_dtype
        self.register_buffer("qweight", torch.empty(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16), persistent=True)
        self.register_buffer("zeros", torch.empty(0, dtype=torch.int16), persistent=True)
        if bias:
            self.register_buffer("bias", torch.empty(self.out_features, dtype=compute_dtype), persistent=True)
        else:
            self.bias = None
        self._reset_backend_cache()

    def _reset_backend_cache(self) -> None:
        self._backend_kind: Optional[str] = None
        self._backend_device: Optional[torch.device] = None
        self._backend_input_features: Optional[int] = None
        self._backend_out_features: Optional[int] = None
        self._backend_group_size: Optional[int] = None
        self._backend_packed_weight: Optional[torch.Tensor] = None
        self._backend_scale_and_zero: Optional[torch.Tensor] = None
        self._backend_qweight_int8: Optional[torch.Tensor] = None
        self._backend_scales_1d: Optional[torch.Tensor] = None
        self._backend_row_correction: Optional[torch.Tensor] = None
        self._backend_dense_weight: Optional[torch.Tensor] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in list(state.keys()):
            if key.startswith("_backend_"):
                state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_backend_cache()

    @property
    def weight(self) -> torch.Tensor:
        return self.dequantize()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        bits: int,
        group_size: int,
        sym: bool,
        compute_dtype: Optional[torch.dtype] = None,
    ) -> "PackedQuantLinear":
        weight = linear.weight.detach()
        bias = None if linear.bias is None else linear.bias.detach()
        module = cls(
            in_features=int(linear.in_features),
            out_features=int(linear.out_features),
            bits=int(bits),
            group_size=int(group_size),
            bias=(bias is not None),
            compute_dtype=compute_dtype or weight.dtype,
        )
        module.pack_weight(weight, bias=bias, sym=bool(sym))
        return module

    def pack_weight(self, weight: torch.Tensor, *, bias: Optional[torch.Tensor], sym: bool) -> None:
        weight = weight.detach().to(torch.float32)
        out_features, in_features = weight.shape
        if int(out_features) != self.out_features or int(in_features) != self.in_features:
            raise ValueError(
                f"weight shape mismatch: got {tuple(weight.shape)}, expected {(self.out_features, self.in_features)}"
            )

        group = self.in_features if self.group_size <= 0 else int(self.group_size)
        num_groups = int(math.ceil(self.in_features / group))
        q_chunks = []
        scale_chunks = []
        zero_chunks = []
        maxq = float((1 << self.bits) - 1)

        for group_idx in range(num_groups):
            start = int(group_idx * group)
            end = int(min(self.in_features, start + group))
            chunk = weight[:, start:end]
            xmin = chunk.min(dim=1).values
            xmax = chunk.max(dim=1).values
            if sym:
                xmax = torch.maximum(xmax.abs(), xmin.abs())
                xmin = -xmax
            zero_mask = (xmin == 0) & (xmax == 0)
            xmin = torch.where(zero_mask, torch.full_like(xmin, -1.0), xmin)
            xmax = torch.where(zero_mask, torch.full_like(xmax, 1.0), xmax)
            scale = (xmax - xmin) / maxq
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            if sym:
                zero = torch.full_like(scale, int((maxq + 1) / 2))
            else:
                zero = torch.round(-xmin / scale)
            q = torch.clamp(torch.round(chunk / scale.unsqueeze(1)) + zero.unsqueeze(1), 0, maxq).to(torch.uint8)
            q_chunks.append(q)
            scale_chunks.append(scale.to(torch.float16))
            zero_chunks.append(zero.to(torch.int16))

        qweight = torch.cat(q_chunks, dim=1).contiguous()
        self.scales = torch.stack(scale_chunks, dim=1).contiguous()
        self.zeros = torch.stack(zero_chunks, dim=1).contiguous()
        self.qweight = _pack_int4(qweight) if self.bits == 4 else qweight

        if self.bias is not None:
            self.bias = bias.to(dtype=self.compute_dtype).contiguous() if bias is not None else None
        self._reset_backend_cache()

    def _maybe_build_cuda_backend(self, device: torch.device) -> bool:
        mode = str(os.environ.get("FLASH_SVD_QUANT_BACKEND", "auto")).strip().lower()
        if mode in {"0", "off", "false", "disable", "disabled"}:
            return False
        if device.type != "cuda":
            return False
        if self.qweight.numel() == 0 or self.scales.numel() == 0 or self.zeros.numel() == 0:
            return False
        if self._backend_device == device and self._backend_kind is not None:
            return True
        self._reset_backend_cache()
        if mode in {"dense", "cached_dense", "fp16", "dequant_cache"}:
            self._build_cuda_dense_backend(device)
            return True
        if mode in {"quant", "int4", "int8", "packed"}:
            if self.bits == 4 and hasattr(torch.ops.aten, "_weight_int4pack_mm") and hasattr(
                torch.ops.aten, "_convert_weight_to_int4pack"
            ):
                self._build_cuda_int4_backend(device)
                return True
            if self.bits == 8 and hasattr(torch.ops.aten, "_weight_int8pack_mm"):
                return self._build_cuda_int8_backend(device)
            self._build_cuda_dense_backend(device)
            return True
        if self._prefer_cached_dense_cuda():
            self._build_cuda_dense_backend(device)
            return True
        if self.bits == 4 and hasattr(torch.ops.aten, "_weight_int4pack_mm") and hasattr(
            torch.ops.aten, "_convert_weight_to_int4pack"
        ):
            self._build_cuda_int4_backend(device)
            return True
        if self.bits == 8 and hasattr(torch.ops.aten, "_weight_int8pack_mm"):
            return self._build_cuda_int8_backend(device)
        self._build_cuda_dense_backend(device)
        return True

    def _prefer_cached_dense_cuda(self) -> bool:
        # Ampere int4 tinygemm tends to lose on small decode-side low-rank projections.
        # Keep quantized execution for larger MLP lifts where one matrix dimension is large.
        max_dim = max(int(self.in_features), int(self.out_features))
        if self.bits == 4:
            return max_dim < 8192
        if self.bits == 8:
            return max_dim < 8192
        return False

    def _build_cuda_dense_backend(self, device: torch.device) -> None:
        self._backend_kind = "cuda_cached_dense"
        self._backend_device = device
        self._backend_dense_weight = self.dequantize(dtype=self.compute_dtype, device=device).contiguous()
        self._backend_input_features = self.in_features
        self._backend_out_features = self.out_features
        self._backend_group_size = self.in_features

    def _build_cuda_int8_backend(self, device: torch.device) -> bool:
        if self.scales.ndim != 2 or self.zeros.ndim != 2 or int(self.scales.shape[1]) != 1 or int(self.zeros.shape[1]) != 1:
            return False
        qweight = self.qweight.to(device=device, dtype=torch.int16)
        scales = self.scales[:, 0].to(device=device, dtype=self.compute_dtype)
        zeros = self.zeros[:, 0].to(device=device, dtype=self.compute_dtype)
        qweight_int8 = torch.clamp(qweight - 128, -128, 127).to(torch.int8).contiguous()
        row_correction = ((128.0 - zeros) * scales).contiguous()
        if torch.all(row_correction == 0):
            row_correction = None
        self._backend_kind = "cuda_int8pack"
        self._backend_device = device
        self._backend_qweight_int8 = qweight_int8
        self._backend_scales_1d = scales.contiguous()
        self._backend_row_correction = row_correction
        self._backend_input_features = self.in_features
        self._backend_out_features = self.out_features
        self._backend_group_size = self.in_features
        return True

    def _build_cuda_int4_backend(self, device: torch.device) -> None:
        group = self.in_features if self.group_size <= 0 else int(self.group_size)
        qweight = _unpack_int4(self.qweight, self.in_features).to(device=device, dtype=torch.int32)
        backend_qparam_dtype = torch.bfloat16
        scales = self.scales.to(device=device, dtype=backend_qparam_dtype)
        zeros_int = self.zeros.to(device=device, dtype=backend_qparam_dtype)

        padded_in = _find_multiple(self.in_features, 1024)
        padded_out = _find_multiple(self.out_features, 8)
        padded_groups = int(math.ceil(float(padded_in) / float(group)))

        if padded_in > self.in_features:
            qweight = F.pad(qweight, (0, padded_in - self.in_features))
        if padded_out > self.out_features:
            qweight = F.pad(qweight, (0, 0, 0, padded_out - self.out_features))
        if int(scales.shape[1]) < padded_groups:
            scales = F.pad(scales, (0, padded_groups - int(scales.shape[1])))
            zeros_int = F.pad(zeros_int, (0, padded_groups - int(zeros_int.shape[1])))
        if padded_out > self.out_features:
            scales = F.pad(scales, (0, 0, 0, padded_out - self.out_features))
            zeros_int = F.pad(zeros_int, (0, 0, 0, padded_out - self.out_features))

        midpoint = float((1 << self.bits) / 2)
        zeros = (midpoint - zeros_int) * scales

        packed_pairs = ((qweight[:, 0::2] << 4) | qweight[:, 1::2]).to(torch.uint8).contiguous()
        packed_weight = torch.ops.aten._convert_weight_to_int4pack(packed_pairs, 8)
        scale_and_zero = _pack_tinygemm_scales_and_zeros(scales.contiguous(), zeros.contiguous())

        self._backend_kind = "cuda_int4pack"
        self._backend_device = device
        self._backend_packed_weight = packed_weight
        self._backend_scale_and_zero = scale_and_zero
        self._backend_input_features = padded_in
        self._backend_out_features = padded_out
        self._backend_group_size = group

    def dequantize(self, *, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
        dtype = dtype or self.compute_dtype
        device = device or self.qweight.device
        qweight = _unpack_int4(self.qweight, self.in_features) if self.bits == 4 else self.qweight
        qweight = qweight.to(device=device, dtype=torch.float32)
        scales = self.scales.to(device=device, dtype=torch.float32)
        zeros = self.zeros.to(device=device, dtype=torch.float32)
        group = self.in_features if self.group_size <= 0 else int(self.group_size)
        chunks = []
        for group_idx in range(int(scales.shape[1])):
            start = int(group_idx * group)
            end = int(min(self.in_features, start + group))
            q = qweight[:, start:end]
            w = scales[:, group_idx].unsqueeze(1) * (q - zeros[:, group_idx].unsqueeze(1))
            chunks.append(w)
        return torch.cat(chunks, dim=1).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and self._maybe_build_cuda_backend(x.device):
            if self._backend_kind == "cuda_int4pack":
                return self._forward_cuda_int4pack(x)
            if self._backend_kind == "cuda_int8pack":
                return self._forward_cuda_int8pack(x)
            if self._backend_kind == "cuda_cached_dense":
                return self._forward_cuda_cached_dense(x)
        weight = self.dequantize(dtype=(x.dtype if torch.is_floating_point(x) else self.compute_dtype), device=x.device)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=weight.dtype)
        x_in = x.to(dtype=weight.dtype) if x.dtype != weight.dtype else x
        return F.linear(x_in, weight, bias)

    def _forward_cuda_int4pack(self, x: torch.Tensor) -> torch.Tensor:
        assert self._backend_packed_weight is not None and self._backend_scale_and_zero is not None
        orig_shape = x.shape
        orig_dtype = x.dtype if torch.is_floating_point(x) else self.compute_dtype
        x2d = x.reshape(-1, orig_shape[-1]).to(torch.bfloat16)
        padded_in = int(self._backend_input_features or orig_shape[-1])
        if padded_in > orig_shape[-1]:
            x2d = F.pad(x2d, (0, padded_in - orig_shape[-1]))
        y = torch.ops.aten._weight_int4pack_mm(
            x2d.contiguous(),
            self._backend_packed_weight,
            int(self._backend_group_size or self.in_features),
            self._backend_scale_and_zero,
        )
        y = y[:, : self.out_features]
        y = y.reshape(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(device=x.device, dtype=y.dtype)
        return y.to(orig_dtype)

    def _forward_cuda_int8pack(self, x: torch.Tensor) -> torch.Tensor:
        assert self._backend_qweight_int8 is not None and self._backend_scales_1d is not None
        orig_shape = x.shape
        orig_dtype = x.dtype if torch.is_floating_point(x) else self.compute_dtype
        x2d = x.reshape(-1, orig_shape[-1])
        x_in = x2d if x2d.dtype == self._backend_scales_1d.dtype else x2d.to(self._backend_scales_1d.dtype)
        y = torch.ops.aten._weight_int8pack_mm(
            x_in.contiguous(),
            self._backend_qweight_int8,
            self._backend_scales_1d,
        )
        if self._backend_row_correction is not None:
            x_sum = x_in.sum(dim=-1, keepdim=True)
            y = y + x_sum * self._backend_row_correction.unsqueeze(0)
        if self.bias is not None:
            y = y + self.bias.to(device=x.device, dtype=y.dtype)
        y = y.reshape(*orig_shape[:-1], self.out_features)
        return y.to(orig_dtype)

    def _forward_cuda_cached_dense(self, x: torch.Tensor) -> torch.Tensor:
        assert self._backend_dense_weight is not None
        weight = self._backend_dense_weight
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=weight.dtype)
        x_in = x if x.dtype == weight.dtype else x.to(weight.dtype)
        return F.linear(x_in, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}"
        )
