# DobiSVD Experiment Writeup

Date: 2026-03-17

## Claim

FlashSVD v1.5 also supports DobiSVD checkpoints. In particular, we can directly load and run the public `Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.4` checkpoint inside the same FlashSVD runtime used for SVD-LLM v1/v2 and Basis Sharing.

However, DobiSVD uses non-uniform per-projection ranks, so current FlashSVD support is correctness-first rather than full fast-path serving. The model still benefits from some FlashSVD runtime improvements, but it cannot fully use the shared-rank attention kernels that drive the larger speedups on SVD-LLM v1/v2.

## Checkpoint

- HF model: `Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.4`
- Local file: `/home/zs89/FlashSVD/checkpoints/dobisvd/dobisvd_0.4/DobiSVD_Model.pt`

## Why Dobi is different

The released `noremapping` checkpoint is not uniform-rank. On layer 0:

- Attention ranks: `Rq=700`, `Rk=826`, `Rv=732`, `Ro=760`
- MLP ranks: `Rgate=1382`, `Rup=1292`, `Rdown=1162`

This differs from the checkpoint families for which FlashSVD v1.5 currently has the strongest fast paths:

- Shared-rank attention fast path assumes `Rq == Rk == Rv`
- Packed exact MLP path works best when at least `Rgate == Rup`
- Experimental dual-split MLP kernels assume `Rgate == Rup == Rdown`

Therefore, DobiSVD support currently falls back to exact attention / exact MLP where the shared-rank assumptions do not hold.

## What support means in practice

FlashSVD v1.5 now supports DobiSVD in the following sense:

- The checkpoint can be loaded directly through `get_model_from_local()`
- Dobi's `SVDTransformLayer(ALinear, BLinear)` modules are converted into native `SVD_LlamaAttention` and `SVD_LlamaMLP`
- Full-sequence forward and cache-based decode both run correctly
- Existing benchmark / correctness harnesses work without custom Dobi-only codepaths

This is a real end-to-end integration result rather than an offline conversion only.

## Validation

### CPU smoke

- Full forward succeeds with `logits` shape `(1, 3, 32000)`
- Cache decode succeeds with `logits` shape `(1, 1, 32000)`

### Correctness

Command:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zs89/FlashSVD/FlashSVD-v1.5 \
python /home/zs89/FlashSVD/FlashSVD-v1.5/benchmark/decode/check_flashsvd_decode_correctness.py \
  --checkpoint /home/zs89/FlashSVD/checkpoints/dobisvd/dobisvd_0.4/DobiSVD_Model.pt \
  --device cuda:0 \
  --dtype bf16 \
  --prompt_len 64 \
  --decode_steps 4
```

Observed output:

- `full_max_abs = 0`
- `decode_max_abs = 0`
- `greedy_token_match = 1.000000`

This shows that the FlashSVD-integrated Dobi model is numerically identical to the baseline on this test.

## Performance

### Short smoke benchmark

Config:

- `A100`, `bf16`, `batch=1`, `prompt_len=64`, `new_tokens=4`, `warmup=1`

Results:

| Model | Prefill | Decode |
| --- | ---: | ---: |
| Low-rank baseline | `0.385 s` | `32.348 ms/token` |
| FlashSVD | `0.039 s` | `28.348 ms/token` |
| Speedup | `9.87x` | `1.14x` |

### More realistic benchmark

Config:

- `A100`, `bf16`, `batch=1`, `prompt_len=256`, `new_tokens=16`, `warmup=1`

Results:

| Model | Prefill | Decode |
| --- | ---: | ---: |
| Low-rank baseline | `0.217 s` | `31.049 ms/token` |
| FlashSVD | `0.052 s` | `28.121 ms/token` |
| Speedup | `4.17x` | `1.10x` |

## Interpretation

The key takeaway is:

- FlashSVD can support DobiSVD checkpoints end-to-end
- The current support already provides a modest decode speedup (`~1.10x-1.14x`)
- The gain is much smaller than on SVD-LLM v1/v2 because the released Dobi `noremapping` checkpoint is non-uniform-rank and therefore cannot fully exploit FlashSVD's shared-rank kernels

So the Dobi result should be framed as:

> FlashSVD v1.5 is not limited to SVD-LLM-style checkpoints; it also supports DobiSVD checkpoints. On the public DobiSVD noremapping Llama-2-7B 0.4 checkpoint, FlashSVD preserves exact correctness and provides a modest decode speedup, while larger gains remain limited by Dobi's non-uniform per-projection ranks.

## Suggested paper wording

One concise paragraph that can be adapted into the paper:

> We additionally verified that FlashSVD v1.5 is compatible with DobiSVD checkpoints. We tested the public `DobiSVD_Noremapping-Llama-2-7b-hf-0.4` release and mapped its `SVDTransformLayer` modules into the same native FlashSVD runtime used for SVD-LLM and Basis Sharing. Unlike SVD-LLM, this Dobi checkpoint uses non-uniform per-projection ranks (`Rq \\neq Rk \\neq Rv`, and similarly in the MLP), so it cannot fully use our shared-rank attention and packed MLP fast paths. Even so, FlashSVD preserves exact decoding outputs and still provides a modest decode speedup of about `1.1x`, showing that our runtime can generalize beyond the checkpoint families it was originally designed for.
