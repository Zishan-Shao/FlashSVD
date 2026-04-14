# Kernel Call Taxonomy

This note explains why the pre-fusion decode path had so many calls.

## Important distinction

Profiler rows like `aten::linear` or `aten::to` are not all standalone custom kernels. They are operator classes that trigger kernels, launch kernels, or create staging traffic. The single strongest fragmentation signal is still `cudaLaunchKernel`.

## Main call families before per-layer graph

### 1. Projection / GEMM family

- `aten::linear`
- `aten::matmul`
- `aten::mm`

These come from current-token low-rank projections, QKV reconstruct, output projection, and MLP gate/up/down work.

### 2. Norm / activation / residual family

- `aten::pow`
- `aten::mean`
- `aten::rsqrt`
- `aten::mul`
- `aten::add`
- `aten::silu`

These come from RMSNorm, residual connections, and SwiGLU.

### 3. Data movement / dtype conversion family

- `aten::to`
- `aten::_to_copy`
- `aten::copy_`
- `cudaMemcpyAsync`

These are the ugly but real serving-path costs: bf16/fp32 casts, static-buffer writes, graph staging, and intermediate materialization.

### 4. Layout / view family

- `aten::transpose`
- `aten::reshape`
- `aten::slice`
- `aten::t`
- `aten::as_strided`
- `aten::empty_strided`

These are not necessarily dominant compute, but they are good evidence that the path is tensor-fragmented and layout-noisy.

### 5. Launch overhead family

- `cudaLaunchKernel`
- `cudaGraphLaunch`

This is the systems story. Before per-layer fusion, the decode path paid too many launches and too many graph boundaries per token.

## What the numbers say

On the motivation study (`prompt_len=512`):

- no graph:
  - `cudaLaunchKernel`: `1174 / token`
  - `copy_`: `199 / token`
  - `to/_to_copy`: `402 / token`
- split graph:
  - `cudaLaunchKernel`: `630 / token`
  - `cudaGraphLaunch`: `64 / token`
  - `copy_`: `327 / token`
  - `to/_to_copy`: `402 / token`
- per-layer graph:
  - `cudaLaunchKernel`: `54 / token`
  - `cudaGraphLaunch`: `32 / token`
  - `copy_`: `135 / token`
  - `to/_to_copy`: `82 / token`

So the old path was not failing because one kernel was slow. It was failing because the token-serving path was too fragmented.
