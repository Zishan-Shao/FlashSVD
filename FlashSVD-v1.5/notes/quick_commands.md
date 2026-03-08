# FlashSVD-v1.5 Quick Commands

Run from:

```bash
cd /home/zs89/FlashSVD/FlashSVD-v1.5
```

## Fair Dense-KV Compare

Definition:

- baseline: low-rank weights + dense KV cache + explicit PyTorch `Q/K -> RoPE` + `FA2`
- FlashSVD: low-rank weights + dense KV cache + `FA2 with internal RoPE`
- for exact-ish compare, keep FFN on `dual_split_cublas_legacy`

Generation / correctness:

Read `flash_legacy` as the fair compare result. That is the dense-KV FlashSVD path with the exact FFN.

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python check_flashsvd_decode_correctness.py \
  --checkpoint /home/zs89/FlashSVD/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --batch_size 1 \
  --decode_steps 16 \
  --legacy_backend dual_split_cublas_legacy \
  --test_backend dual_split_cublas_legacy \
  --prompt_text "The capital of France is" \
  --flash_dense_attn \
  --baseline_dense_kvcache \
  --reference_dense_attn \
  --print_text \
  --no-print_tokens
```

Decode benchmark:

Exact dense-KV decode benchmark (same low-rank weights, baseline also uses dense KV):

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python bench_flashsvd_vs_svd_decode.py \
  --checkpoint /home/zs89/FlashSVD/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --prompt_len 512 \
  --new_tokens 32 \
  --warmup 3 \
  --batch_size 1 \
  --flashsvd_ffn_backend dual_split_cublas_legacy \
  --experimental_flash_dense_attn \
  --baseline_dense_kvcache
```

Baseline decode profile:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python SVDLLM_flashsvd.py \
  --model jeffwan/llama-7b-hf \
  --model_path /home/zs89/FlashSVD/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --step 6 \
  --dataset wikitext2 \
  --DEV cuda \
  --model_seq_len 512 \
  --eval_batch_size 1 \
  --eval_dtype bf16 \
  --prompt_len 512 \
  --new_tokens 8 \
  --decode_warmup 3 \
  --flashsvd_ffn_backend dual_split_cublas_legacy \
  --baseline_dense_kvcache \
  --profile_decode
```

FlashSVD decode profile:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python SVDLLM_flashsvd.py \
  --model jeffwan/llama-7b-hf \
  --model_path /home/zs89/FlashSVD/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --step 6 \
  --dataset wikitext2 \
  --DEV cuda \
  --model_seq_len 512 \
  --eval_batch_size 1 \
  --eval_dtype bf16 \
  --prompt_len 512 \
  --new_tokens 8 \
  --decode_warmup 3 \
  --flashsvd_ffn_backend dual_split_cublas_legacy \
  --flashsvd_dense_cache \
  --profile_decode
```

## Faster Serving Path

Faster dense-KV serving benchmark (same low-rank weights, but packed FFN is not exact in bf16):

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python bench_flashsvd_vs_svd_decode.py \
  --checkpoint /home/zs89/FlashSVD/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --prompt_len 512 \
  --new_tokens 32 \
  --warmup 3 \
  --batch_size 1 \
  --flashsvd_ffn_backend dual_split_cublas \
  --experimental_flash_dense_attn \
  --baseline_dense_kvcache
```

## Eval

Exact full-sequence eval / PPL-aligned eval:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python SVDLLM_flashsvd.py \
  --model jeffwan/llama-7b-hf \
  --model_path /home/zs89/FlashSVD/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --step 4 \
  --dataset wikitext2 \
  --DEV cuda \
  --model_seq_len 512 \
  --eval_batch_size 1 \
  --eval_dtype bf16 \
  --flashsvd_ffn_backend dual_split_cublas_legacy \
  --reference_dense_attn
```

If you want the old StaticCache exact reference instead:

```bash
CUDA_VISIBLE_DEVICES=5 FLASH_SVD_TRUST_PICKLE=1 \
python check_flashsvd_decode_correctness.py \
  --checkpoint /home/zs89/FlashSVD/jeffwan_llama_7b_hf_whitening_only_0.5.pt \
  --dtype bf16 \
  --device cuda \
  --batch_size 1 \
  --decode_steps 16 \
  --legacy_backend dual_split_cublas_legacy \
  --test_backend dual_split_cublas_legacy \
  --prompt_text "The capital of France is" \
  --print_text \
  --no-print_tokens
```

## FlashSVDGeGLU / ModernBERT Encoder

Kernel eval (exact operator-level compare vs PyTorch reference; not PPL):

```bash
CUDA_VISIBLE_DEVICES=5 \
python kernels/flashsvd-v1.5/flashsvdgeglu/kernel_microbench.py \
  --kernel both \
  --B 2 \
  --L 1024 \
  --hidden-size 768 \
  --num-heads 12 \
  --intermediate-size 1152 \
  --target-param-ratio 0.5 \
  --dtype bf16 \
  --warmup 10 \
  --iters 50
```

## Notes

- For decoder correctness / exactness, the real baseline is the compressed **Low-Rank** model, not the original dense model.
- Recommended fair compare now means:
  - baseline: low-rank weights + dense KV cache + explicit PyTorch `Q/K/V -> RoPE -> FA2`
  - FlashSVD: low-rank weights + dense KV cache + FlashSVD dense decode path
- In `check_flashsvd_decode_correctness.py`, read `flash_legacy` as the fair attention compare when both backends are `dual_split_cublas_legacy`.
- `bench_flashsvd_vs_svd_decode.py --flashsvd_ffn_backend auto` does **not** hit the current packed MLP path. Use:
  - `dual_split_cublas_legacy` for the aligned dense-KV exact benchmark above
  - `dual_split_cublas` for the faster packed serving path
- Decode attention winner is only enabled when you pass `--experimental_flash_dense_attn` in `bench_flashsvd_vs_svd_decode.py` or `--flashsvd_dense_cache` in `SVDLLM_flashsvd.py`.
- `step 6` decode and `step 4` eval can now be aligned more tightly by using:
  - `--flashsvd_ffn_backend dual_split_cublas_legacy`
  - `--reference_dense_attn` for full-seq eval
  - `--flashsvd_dense_cache` / `--baseline_dense_kvcache` for dense-KV decode A/B
- In bf16, dense-KV `flash_legacy` currently matches the aligned low-rank baseline on full-seq logits and on greedy decode, but decode logits still show small numeric drift.
- Packed FFN (`dual_split_cublas`) is faster, but is not yet strict-exact in bf16.
- For `FlashSVDGeGLU`, `kernel_microbench.py` is only an exact operator-level compare against the same PyTorch reference math.
- There is currently no recommended `FlashSVDGeGLU` model-level throughput / PPL command in this repo that benchmarks one low-rank model implementation against another low-rank baseline while keeping weights fixed.
- `encoder_compare.py` is a legacy dense-vs-FlashSVD synthetic compare and is not part of the recommended low-rank model inference speedup workflow.
- There is currently no dedicated ModernBERT / `FlashSVDGeGLU` model-level PPL command in this repo that toggles kernel vs no-kernel while keeping the same weights fixed.
- For the current encoder FFN winner path, use `--ffn-variant preg`.
