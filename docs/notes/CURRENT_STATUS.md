# FlashSVD-v1.5 Current Status

Last verified: 2026-03-10 / 2026-03-11

## Serving Direction

- Attention production decode path:
  dense KV cache + reconstruct current token + `flash_attn_with_kvcache`
- MLP production decode path:
  `flashsvd_mlp_dual_split_prod`
- Exact-aligned dense-KV compare:
  keep MLP on `flashsvd_mlp_dual_split_exact_legacy`

## Which Benchmark To Trust

Use [`../benchmark/decode/bench_flashsvd_vs_svd_decode.py`](../benchmark/decode/bench_flashsvd_vs_svd_decode.py)
for headline decode speed.

Do not use [`../benchmark/mlp/compare_decode_backends.py`](../benchmark/mlp/compare_decode_backends.py)
as the headline decode benchmark:

- it is mainly for MLP backend A/B
- it runs with `enable_flash_dense_attn=False`
- the number it prints is not the current dense-KV attention winner path

## Verified Results

### Attention Route Compare

Command:

```bash
python benchmark/attn/decode_compare.py \
  --llama llama2-7b \
  --target-param-ratio 0.5 \
  --rank-formula global \
  --factor-layout shared \
  --B 1 \
  --Ls 256,1024 \
  --dtype bf16 \
  --dense-backend fa2 \
  --warmup 1 \
  --iters 2 \
  --report compact
```

Result snapshot:

- `L=256`: winner `rank_tok+fa2_kv`, `0.1071 ms`
- `L=1024`: winner `rank_tok+fa2_kv`, `0.1194 ms`
- `FlashSVD-attn-legacy-v1.6-sparsekv-v2*` still fails in this script on these settings

### Exact Dense-KV Correctness

Command:

```bash
python benchmark/decode/check_flashsvd_decode_correctness.py \
  --checkpoint /home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --batch_size 1 \
  --decode_steps 16 \
  --legacy_backend flashsvd_mlp_dual_split_exact_legacy \
  --test_backend flashsvd_mlp_dual_split_exact_legacy \
  --prompt_text "The capital of France is" \
  --flash_dense_attn \
  --baseline_dense_kvcache \
  --reference_dense_attn \
  --no-print_tokens
```

Result snapshot:

- `flash_legacy`: `greedy_token_match=1.000000`
- `flash_test`: `greedy_token_match=1.000000`
- full-seq logits match exactly in this configuration

### Packed Serving Correctness

Command:

```bash
python benchmark/decode/check_flashsvd_decode_correctness.py \
  --checkpoint /home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --batch_size 1 \
  --decode_steps 16 \
  --legacy_backend flashsvd_mlp_dual_split_exact_legacy \
  --test_backend flashsvd_mlp_dual_split_prod \
  --prompt_text "The capital of France is" \
  --flash_dense_attn \
  --baseline_dense_kvcache \
  --reference_dense_attn \
  --no-print_tokens
```

Result snapshot:

- `flash_test`: `greedy_token_match=1.000000`
- decode drift remains non-zero:
  `decode_max_abs=0.40625`
- interpret this as "good serving path, not strict bf16 exactness"

### Exact-Aligned End-to-End Decode

Command:

```bash
python benchmark/decode/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --prompt_len 512 \
  --new_tokens 32 \
  --warmup 3 \
  --batch_size 1 \
  --flashsvd_ffn_backend flashsvd_mlp_dual_split_exact_legacy \
  --experimental_flash_dense_attn \
  --baseline_dense_kvcache
```

Result snapshot:

- SVD baseline: `36.377 ms/token`
- FlashSVD: `23.848 ms/token`
- speedup: `1.53x`

### Faster Packed Serving Decode

Command:

```bash
python benchmark/decode/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /home/zs89/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
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

Result snapshot:

- SVD baseline: `37.804 ms/token`
- FlashSVD: `25.273 ms/token`
- speedup: `1.50x`

### MLP Graph Smoke

Command:

```bash
python benchmark/mlp/bench_svd_llama_decode_graph.py \
  --hidden-size 1024 \
  --intermediate-size 2816 \
  --ratio 0.5 \
  --seq-len 1 \
  --batches 1 \
  --num-heads 8 \
  --num-kv-heads 8 \
  --dtype bfloat16 \
  --warmup 1 \
  --runs 2
```

Result snapshot:

- `mlp_graph`: about `1.37x` vs eager
- `layer_tail_graph`: about `2.01x` vs eager

## Current Interpretation

- The current headline decode gain is still about `1.5x`, not `1.1x`.
- The smaller `1.08x ~ 1.10x` numbers from `compare_decode_backends.py` are
  expected because that script disables the dense attention winner and mainly
  isolates MLP backend differences.
- Attention is still the main decode win.
- Exact MLP helps, but only modestly.
