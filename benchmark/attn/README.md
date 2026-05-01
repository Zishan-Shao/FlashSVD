# Attention Benchmarks

This folder holds current attention-focused benchmarks that are worth exposing
next to `benchmark/decode/` and `benchmark/mlp/`.

Current entrypoints:

- `decode_compare.py`
  Single-token decoding attention comparison:
  `FlashSVD-v1.5`, `sparse`, `sparse+FA2-only`, and `dense+FA2-only`.

Example:

```bash
python /path/to/FlashSVD/FlashSVD-v1.5/benchmark/attn/decode_compare.py \
  --llama llama2-7b \
  --target-param-ratio 0.5 \
  --rank-formula global \
  --factor-layout shared \
  --B 1 \
  --Ls 256,512,1024,2048,4096,8192 \
  --dtype bf16 \
  --dense-backend fa2
```
