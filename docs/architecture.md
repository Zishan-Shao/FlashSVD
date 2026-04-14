# FlashSVD Architecture

`flashsvd/` is now the canonical runtime package for FlashSVD v1.5.

## Layout

- `flashsvd/models/`
  Hugging Face-facing model integration and module definitions.
- `flashsvd/runtime/`
  Decode dispatch, backend selection, and cache plumbing.
- `flashsvd/kernels/`
  Triton / kernel implementations only.
- `flashsvd/methods/`
  Compression and conversion flows such as SVD-LLM variants.
- `flashsvd/quant/`
  Quantization and LoRA fusion utilities.
- `flashsvd/utils/`
  Model loading, evaluation, and checkpoint helpers.
- `benchmarks/`
  Benchmarks and correctness scripts for the v1.5 serving path.
- `tests/`
  Import and compatibility smoke tests.

## Compatibility

The repository still ships top-level compatibility packages:

- `component/`
- `flashsvd_component/`
- `models/`
- `runtime/`
- `utils/`
- `quant/`
- `methods/`
- `kernels/`

They exist so older checkpoints and local scripts keep working while the repo
transitions to `flashsvd.*` imports.

## Current Production Path

The production decode route remains:

- attention: dense KV cache + reconstruct current token + FlashAttention KV cache
- MLP: `flashsvd_mlp_dual_split_prod`

See `FlashSVD-v1.5/notes/CURRENT_STATUS.md` for the latest verified command
lines and measured results.
