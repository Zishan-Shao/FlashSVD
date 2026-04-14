# DobiSVD Support Notes

Date: 2026-03-17

## Checkpoint

- HF repo: `Qinsi1/DobiSVD_Noremapping-Llama-2-7b-hf-0.4`
- Local file: `/home/zs89/FlashSVD/checkpoints/dobisvd/dobisvd_0.4/DobiSVD_Model.pt`

## Structure

- The checkpoint is a trusted pickle: `{'model': LlamaForCausalLM, 'tokenizer': ...}`.
- Each projection is stored as `modules.module.SVDTransformLayer`.
- Mapping:
  - `ALinear.weight`: input-side low-rank projection (`v_proj`)
  - `BLinear.weight`: output-side reconstruction (`u_proj`)

Observed layer-0 ranks:

- Attention: `Rq=700`, `Rk=826`, `Rv=732`, `Ro=760`
- MLP: `Rgate=1382`, `Rup=1292`, `Rdown=1162`

This means Dobi is not a uniform-rank checkpoint. Current FlashSVD fast paths that assume shared attention rank or uniform MLP rank cannot be applied blindly.

## Loader / runtime support added

- Trusted pickle load now installs compatibility shims for:
  - `transformers.models.llama.modeling_llama.LlamaSdpaAttention`
  - `modules.module.SVDTransformLayer`
- `get_model_from_local()` now converts Dobi projections into native:
  - `SVD_LlamaAttention`
  - `SVD_LlamaMLP`
- Top-level `model.model.rotary_emb` is restored for newer transformers compatibility.

## Runtime policy for Dobi

- Attention:
  - If `Rq == Rk == Rv`, FlashSVD shared-rank decode/prefill fast paths remain available.
  - If ranks differ, attention falls back to exact dense reconstruction + SDPA/standard cache path.
- MLP:
  - `flashsvd_mlp_dual_split_prod` packed path is used only when `Rgate == Rup`.
  - Experimental dual-split kernels require `Rgate == Rup == Rdown`; otherwise they fall back to exact MLP.

This keeps Dobi correct without forcing shared-rank assumptions that do not hold.

## Validation

### CPU smoke

- `get_model_from_local('/home/zs89/FlashSVD/checkpoints/dobisvd/dobisvd_0.4/DobiSVD_Model.pt')` succeeds.
- Full forward succeeds with `logits` shape `(1, 3, 32000)`.
- Cache decode smoke succeeds with `logits` shape `(1, 1, 32000)`.

### GPU benchmark smoke

Command:

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/home/zs89/FlashSVD/FlashSVD-v1.5 \
python /home/zs89/FlashSVD/FlashSVD-v1.5/benchmark/decode/bench_flashsvd_vs_svd_decode.py \
  --checkpoint /home/zs89/FlashSVD/checkpoints/dobisvd/dobisvd_0.4/DobiSVD_Model.pt \
  --device cuda:0 \
  --dtype bf16 \
  --prompt_len 64 \
  --new_tokens 4 \
  --batch_size 1 \
  --warmup 1
```

Observed output:

- Low-rank baseline decode: `32.348 ms/token`
- FlashSVD decode: `28.348 ms/token`
- Speedup: `1.14x`

### Correctness smoke

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

## Practical takeaway

- FlashSVD now supports Dobi checkpoints end-to-end.
- Because Dobi uses per-projection non-uniform ranks, current support is correctness-first.
- Some FlashSVD production tricks still help, but the main shared-rank attention kernels are not yet applicable to this checkpoint family.
