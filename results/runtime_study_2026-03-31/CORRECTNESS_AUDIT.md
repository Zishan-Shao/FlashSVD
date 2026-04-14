# Correctness Audit Against FP32 No-Cache Gold Reference

## Setting

- Checkpoint: `/home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt`
- Gold reference: `fp32 no-cache`, greedy decode, `64` generated tokens
- Evaluated systems: `StaticCache`, `DenseKVCacheBaseline`, `FlashSVD-v1.5 prod`
- Evaluated dtype for cached systems: `bf16`
- Number of prompts: `20`
- Raw details: `raw/correctness_gold_reference_details.json`
- Aggregate table: `tables/correctness_gold_reference_summary.json`

## Aggregate Results

- `StaticCache`
  - exact match: `12/20`
  - first-token match: `20/20`
  - mean token match: `0.6828`
  - median token match: `1.0`
  - mean first divergence over divergent prompts: `9.38`
- `DenseKVCacheBaseline`
  - exact match: `12/20`
  - first-token match: `18/20`
  - mean token match: `0.6438`
  - median token match: `1.0`
  - mean first divergence over divergent prompts: `6.38`
- `FlashSVD-v1.5 prod`
  - exact match: `12/20`
  - first-token match: `20/20`
  - mean token match: `0.7086`
  - median token match: `1.0`
  - mean first divergence over divergent prompts: `8.50`

## Interpretation

- None of the three cached paths is perfectly identical to the `fp32 no-cache` gold reference on every prompt.
- `FlashSVD-v1.5 prod` is the closest system overall by mean token match.
- `StaticCache` and `FlashSVD-v1.5 prod` both preserve the first generated token on all `20/20` prompts.
- `DenseKVCacheBaseline` is the weakest on first-token stability (`18/20`) and has the lowest mean token match.

## FP32 Cached Check

- `StaticCache fp32 cached` matches the `fp32 no-cache` gold reference exactly on `20/20` prompts.
- This means the cache mechanism itself is not the root problem; the HF/StaticCache path can be exact in `fp32`.
- `DenseKVCacheBaseline fp32` is not runnable because it depends on `flash_attn_with_kvcache`, and FlashAttention only supports `fp16/bf16`.
- `FlashSVD-v1.5 prod fp32` is likewise not runnable for the same reason.
- Summary file: `tables/correctness_fp32_cached_summary.json`

## Pairwise Agreement

- `StaticCache` vs `FlashSVD-v1.5 prod`: exact agreement on `15/20` prompts
- `StaticCache` vs `DenseKVCacheBaseline`: exact agreement on `14/20` prompts
- `DenseKVCacheBaseline` vs `FlashSVD-v1.5 prod`: exact agreement on `11/20` prompts

## Notable Cases

- `FlashSVD-v1.5 prod` uniquely matches gold on prompt `13` and prompt `15`.
- `DenseKVCacheBaseline` uniquely matches gold on prompt `18`.
- `StaticCache` has no prompt where it is the only exact match.
- Prompt `3` (`"FlashSVD accelerates low-rank language models by"`) is unstable for all three cached systems.
- Prompt `1` (`"The capital of France is"`) is exact for `StaticCache` and `DenseKVCacheBaseline`, but `FlashSVD-v1.5 prod` diverges at step `3`.

## Bottom Line

- The earlier claim that `DenseKVCacheBaseline` is simply "wrong" is too strong.
- The stronger statement supported by this audit is:
  - `FlashSVD-v1.5 prod` is not obviously producing garbage; it is competitive with, and often closer to gold than, the other cached baselines.
  - `DenseKVCacheBaseline` is the least stable of the three on first-token agreement and average token agreement.
  - Correctness on this compressed checkpoint is numerically fragile for some prompts, so paper claims should use `fp32 no-cache` as the gold reference instead of assuming any low-precision cache path is exact.
  - If a reviewer asks whether cache can be exact at all, the answer is yes: `StaticCache fp32 cached` reproduces the `fp32 no-cache` gold reference exactly on this 20-prompt audit.
