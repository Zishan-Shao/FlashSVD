# FlashSVD Architecture

FlashSVD v1.5 keeps the main runtime code in the existing repository layout.
The repo-local demo entrypoint is [demo_flashsvd_v15.py](../demo_flashsvd_v15.py),
and its helper code lives under `scripts/demo_support/`.

## Layout

- `demo_flashsvd_v15.py`
  Root-level demo entrypoint for the current v1.5 serving recipe.
- `models/`
  Hugging Face-facing model integration and module definitions.
- `runtime/`
  Decode dispatch, backend selection, and cache plumbing.
- `kernels/`
  Triton / kernel implementations only.
- `src/`
  Compression and conversion flows such as SVD-LLM variants.
- `utils/`
  Model loading, evaluation, and checkpoint helpers.
- `benchmark/`
  Benchmarks and correctness scripts for the v1.5 serving path.
- `scripts/`
  Demo support helpers, job scripts, and smoke tests.

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

They exist so older checkpoints and local scripts keep working.

## Current Production Path

The production decode route remains:

- attention: dense KV cache + reconstruct current token + FlashAttention KV cache
- MLP: `auto` routing with graph-enabled packed production execution

See `docs/notes/CURRENT_STATUS.md` for the latest verified command lines and
measured results.
