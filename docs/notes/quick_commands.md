# FlashSVD-v1.5 Quick Commands

Run from:

```bash
cd /path/to/FlashSVD/FlashSVD-v1.5
```

Use [`CURRENT_STATUS.md`](./CURRENT_STATUS.md) for the latest verified numbers.

## Main Decode Commands

Exact dense-KV correctness:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python benchmark/decode/check_flashsvd_decode_correctness.py \
  --checkpoint /path/to/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
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

Exact-aligned dense-KV decode benchmark:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python benchmark/decode/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /path/to/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
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

Faster packed serving decode benchmark:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python benchmark/decode/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /path/to/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
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

Full-sequence eval / PPL-aligned eval:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python SVDLLM_flashsvd.py \
  --model jeffwan/llama-7b-hf \
  --model_path /path/to/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --step 4 \
  --dataset wikitext2 \
  --DEV cuda \
  --model_seq_len 512 \
  --eval_batch_size 1 \
  --eval_dtype bf16 \
  --flashsvd_ffn_backend flashsvd_mlp_dual_split_exact_legacy \
  --reference_dense_attn
```

## Component Benchmarks

Attention route compare:

```bash
CUDA_VISIBLE_DEVICES=5 \
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

MLP decode-graph smoke:

```bash
CUDA_VISIBLE_DEVICES=5 \
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

Real-checkpoint MLP benchmark:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python benchmark/mlp/bench_real_checkpoint_mlp.py \
  --checkpoint /path/to/FlashSVD/checkpoints/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --layer 0 \
  --batch_size 1 \
  --seq_len 1 \
  --warmup 1 \
  --iters 2 \
  --backends baseline,flashsvd_mlp_dual_split_exact_legacy,flashsvd_mlp_dual_split_prod
```

## Legacy / Archive

Legacy experimental kernel microbench:

```bash
CUDA_VISIBLE_DEVICES=5 \
python benchmark/legacy/archive/flashsvdgeglu/kernel_microbench.py \
  --kernel both \
  --B 2 \
  --L 256 \
  --hidden-size 768 \
  --num-heads 12 \
  --intermediate-size 1152 \
  --target-param-ratio 0.5 \
  --dtype bf16 \
  --warmup 2 \
  --iters 5
```

## Notes

- Fair dense-KV compare means:
  - baseline: low-rank weights + dense KV cache + explicit RoPE + FA2
  - FlashSVD: low-rank weights + dense KV cache + FlashSVD dense decode path
- Read `flash_legacy` as the fair attention compare when both decode backends are `flashsvd_mlp_dual_split_exact_legacy`.
- For the main decode benchmark, do not use `--flashsvd_ffn_backend auto`.
- Attention winner is only enabled when you pass `--experimental_flash_dense_attn` or `--flashsvd_dense_cache`.
- `benchmark/mlp/compare_decode_backends.py` is an MLP backend compare, not the headline decode benchmark.
- Packed `flashsvd_mlp_dual_split_prod` is the faster serving path, but it is not strict bf16 exactness.
