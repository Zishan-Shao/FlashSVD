# FlashSVD

FlashSVD is a streaming inference runtime for SVD-compressed language models.
This repository now treats **FlashSVD v1.5** as the primary product path.

The canonical Python package is `flashsvd/`. Older trees and scripts are kept
for compatibility and research reference, but new work should target
`flashsvd.*`, `benchmarks/`, and `tests/`.

## Main Layout

```text
flashsvd/
  models/      Hugging Face-facing model integration
  runtime/     decode dispatch, cache plumbing, backend selection
  kernels/     Triton / kernel implementations
  methods/     compression flows such as SVD-LLM
  quant/       quantization and LoRA fusion helpers
  utils/       loading, evaluation, checkpoint utilities
benchmarks/    benchmark and correctness scripts for v1.5
tests/         import and compatibility smoke tests
docs/          repository architecture notes
```

More detail: [docs/architecture.md](/home/zs89/FlashSVD/docs/architecture.md)

## Current Serving Direction

The production decode path follows the v1.5 runtime described in
[FlashSVD-v1.5/notes/CURRENT_STATUS.md](/home/zs89/FlashSVD/FlashSVD-v1.5/notes/CURRENT_STATUS.md):

- attention: dense KV cache + reconstruct current token + `flash_attn_with_kvcache`
- MLP: `flashsvd_mlp_dual_split_prod`

The verified notes in that file were last checked on March 10-11, 2026 and
report about `1.50x` to `1.53x` end-to-end decode speedup on the documented
Llama-2-7B configuration.

## Installation

FlashSVD v1.5 requires Python `3.10+`.

```bash
git clone https://github.com/Zishan-Shao/FlashSVD.git
cd FlashSVD

# install PyTorch yourself first, then:
pip install -e .[test]
```

If you prefer the local helper:

```bash
./install_local.sh
```

## Quick Start

### Python API

```python
from flashsvd import get_model_from_local, decode_kvcache_eval

model, tokenizer = get_model_from_local("/path/to/checkpoint")
metrics = decode_kvcache_eval(
    model,
    prompt_len=512,
    new_tokens=32,
    device="cuda",
    flashsvd_dense_cache=True,
)
print(metrics)
```

### Benchmarks

Headline decode benchmark:

```bash
python benchmarks/decode/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /path/to/checkpoint.pt \
  --dtype bf16 \
  --device cuda \
  --prompt_len 512 \
  --new_tokens 32 \
  --warmup 3 \
  --batch_size 1 \
  --flashsvd_ffn_backend flashsvd_mlp_dual_split_prod \
  --experimental_flash_dense_attn \
  --baseline_dense_kvcache
```

Correctness check:

```bash
python benchmarks/decode/check_flashsvd_decode_correctness.py \
  --checkpoint /path/to/checkpoint.pt \
  --dtype bf16 \
  --device cuda \
  --batch_size 1 \
  --decode_steps 16 \
  --legacy_backend flashsvd_mlp_dual_split_exact_legacy \
  --test_backend flashsvd_mlp_dual_split_prod \
  --flash_dense_attn \
  --baseline_dense_kvcache \
  --reference_dense_attn
```

## Compatibility

Legacy imports such as `component.*`, `flashsvd_component.*`, `models.*`, and
`runtime.*` are still supported through compatibility shims so local checkpoints
and older scripts continue to load while the repo migrates to `flashsvd.*`.
