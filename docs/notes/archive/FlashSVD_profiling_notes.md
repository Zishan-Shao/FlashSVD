# FlashSVD Kernel Profiling Notes (A100 / H200) — Practical Tips, Commands, Results, Sweeps

> This document summarizes the profiling workflow we used for the **FlashSVD fused attention kernel**
> (low‑rank reconstruction + RoPE + online softmax + GQA grouping), plus **real profiling results**
> from an **H200 NVL** run, and a **recommended parameter sweep plan**.

---

## 0) TL;DR

- **Always warm up**: first run includes Triton compilation + CUDA module loads → ignore.
- Prefer **end‑to‑end** timings (factors → output) for “fair” comparisons, not “attention‑only”.
- Use **torch.profiler** to confirm kernel fusion + count kernels.
- Use **nsys** to verify kernel dominance + timeline; `nsys stats` report names may differ by version.
- **Hardware counters may be locked** (`ERR_NVGPUCTRPERM`): you can still use `nsys` timeline + stats,
  but GPU utilization metrics and `ncu` counters require admin changes.
- For tuning without counters: do **BM/BN/BR/warps/stages sweeps** and interpret trends.

---

## 1) Kernel semantics (FlashAttention‑aligned)

**packed**
- No `attention_mask` tensor.
- All tokens are valid.
- Masking supported via `causal` and `window_size=(window_left, window_right)`.

**varlen**
- Ragged / padded sequences represented via `cu_seqlens` + `max_seqlen`.
- Masking supported via `causal` and `window_size`.

---

## 2) Environment + reproducibility checklist

### 2.1 Recommended env vars
```bash
export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/tmp/triton_cache
# Optional: keep CUDA libs stable if multiple toolkits exist
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

### 2.2 Measure “steady state”
Run the same command twice:
- Run #1 includes compilation / caching → ignore
- Run #2 is steady state

---

## 3) Quick performance measurement (the “daily driver”)

### 3.1 Packed (prefill‑like, no padding mask tensor)
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode packed \
  --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 \
  --dtype bf16 \
  --bm 64 --bn 64 --br 64 --warps 8 --stages 3 \
  --causal --warmup 50 --iters 200
```

### 3.2 Varlen (padding via cu_seqlens)
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode varlen \
  --B 64 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 \
  --dtype bf16 --causal --warmup 50 --iters 200
```

**Notes**
- `tokens/s` in packed uses `B*S`.
- `tokens/s` in varlen uses packed token count `T` (more realistic).

---

## 4) Numerical accuracy & stability

### 4.1 Small‑S fp32 reference check (O(S²), keep S small)
Packed:
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode packed \
  --B 2 --S 256 --H 32 --Hk 8 --Dh 128 --R 64 \
  --dtype bf16 --causal --check
```

Varlen:
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode varlen \
  --B 4 --S 256 --H 32 --Hk 8 --Dh 128 --R 64 \
  --dtype bf16 --causal --check
```

### 4.2 Stress stability (scale Q/K factors)
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode packed \
  --B 2 --S 1024 --H 32 --Hk 8 --Dh 128 --R 64 \
  --dtype bf16 --causal \
  --stress --stress_scales 1,3,10,30
```

---

## 5) torch.profiler: kernel count + time breakdown + chrome trace

### 5.1 Run
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode packed \
  --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 \
  --dtype bf16 --bm 64 --bn 64 --br 64 --warps 8 --stages 3 \
  --causal --profile --trace trace.json
```

### 5.2 Interpretation tips
- CPU time may be dominated by `cudaDeviceSynchronize` during profiling — that’s expected.
- Key questions:
  1) Is the fused path basically **one dominant CUDA kernel**? ✅ (good)
  2) Any unexpected extra kernels (copies / casts / `contiguous`) ? ❌ (bad)
  3) Is per‑call time stable? ✅ (good)

---

## 6) Nsight Systems (nsys): timeline + kernel summary (no counters required)

### 6.1 Correct command structure (env var placement)
✅ Correct:
```bash
CUDA_VISIBLE_DEVICES=1 nsys profile -o nsys_flashsvd \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  python flashsvd.py --mode packed \
    --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 \
    --dtype bf16 --bm 64 --bn 64 --br 64 --warps 8 --stages 3 \
    --causal --warmup 10 --iters 50
```

### 6.2 Kernel/API summaries
List available reports:
```bash
nsys stats --help-reports
```

In our environment, these work:
```bash
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum nsys_flashsvd.nsys-rep
```

---

## 7) Nsight Compute (ncu): hardware counters (may require admin permission)

### 7.1 ncu path (installed but not in PATH)
We found:
- `/usr/local/cuda-12.8/bin/ncu`

Test:
```bash
/usr/local/cuda-12.8/bin/ncu --version
```

### 7.2 Typical ncu command (reduce iters)
```bash
CUDA_VISIBLE_DEVICES=1 /usr/local/cuda-12.8/bin/ncu \
  --set full --target-processes all -o ncu_flashsvd \
  python flashsvd.py --mode packed \
    --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 \
    --dtype bf16 --bm 64 --bn 64 --br 64 --warps 8 --stages 3 \
    --causal --warmup 5 --iters 20
```

### 7.3 Common failure: `ERR_NVGPUCTRPERM`
If your cluster disables performance counters for non‑admin users:
- `nsys --gpu-metrics-*` fails
- `ncu` can’t collect most metrics

**Admin request (copy/paste):**
> Please enable non‑admin access to NVIDIA GPU performance counters for Nsight Compute/Systems  
> (e.g., driver parameter `NVreg_RestrictProfilingToAdminUsers=0` or an equivalent policy),  
> or provide a profiling node/user group with counter access.

---

# 8) ✅ Our recorded profiling results (H200 NVL)

All results below are from an **H200 NVL** node on:

- **packed**
- `B=8, S=2048, H=32, Hk=8, Dh=128, R=64`
- `dtype=bf16`
- `BM=64, BN=64, BR=64, warps=8, stages=3`
- `causal=True`, `window=(-1,-1)`

## 8.1 End‑to‑end benchmark numbers

**flashsvd.py output**
- Latency: **~10.95 ms**
- Throughput: **~1.50M tokens/s** (tokens = B*S)
- `eff TFLOPs(QK+PV)`: **~25.1 TF/s** (attention math only)
- Peak memory: **~353 MB allocated**, **~354 MB reserved**

> Note: `eff TFLOPs(QK+PV)` excludes the low‑rank reconstruction work.  
> For fair comparisons against an “unfused pipeline”, report **end‑to‑end** metrics too (see §10).

## 8.2 torch.profiler summary (packed, same config)

Key takeaways from the printed table:
- The fused kernel `flashsvd_rope_fwd_packed_R` accounts for **~100% of CUDA time** inside the profiled region.
- Per‑call CUDA time around **10.95 ms**.
- CPU time dominated by `cudaDeviceSynchronize` (expected during profiling).

A representative row (simplified):
- `flashsvd_rope_fwd_packed_R`: `Self CUDA ~328ms`, `#calls=30`, `CUDA time avg ~10.95ms`

## 8.3 nsys kernel summary (cuda_gpu_kern_sum)

From:
```bash
nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum nsys_flashsvd.nsys-rep
```

**CUDA GPU Kernel Summary**
- `flashsvd_rope_fwd_packed_R`: **99.7%** of GPU kernel time  
  - Total: **241,001,226 ns**
  - Instances: **22**
  - Avg: **10,954,601 ns** (~10.95 ms)
  - Min/Max: **10.938 ms / 10.985 ms**
  - Stddev: **~9.6 µs** (very stable)

Other kernels (fill, distribution, sin/cos, arange) are **≤0.2%** total and mostly from initialization.

**Interpretation**
- ✅ “Single dominant kernel” behavior is FlashAttention‑like: no hidden extra kernels in the hot path.
- ✅ Runtime stability is excellent (very small stddev).

## 8.4 nsys CUDA API summary (cuda_api_sum)

Highlights:
- `cudaLaunchKernel`: median **~6.86 µs**, but has rare large outliers (up to **~41 ms**) → typically initial module load/JIT overhead.
- `cudaDeviceSynchronize`: dominates API time in profiled runs because we synchronize for measurement.

**Interpretation**
- Steady‑state launch overhead is fine; the large average is skewed by a few expensive events.
- Use warmup and ignore the first run for clean steady‑state numbers.

---

# 9) Accuracy & stability results we observed

## 9.1 Packed small‑S fp32 reference check
Config:
- `B=2, S=256, H=32, Hk=8, Dh=128, R=64, bf16, causal`

Observed:
- `finite=True`
- `max_abs ≈ 9.57e+00`
- `rel_fro ≈ 2.80e-02` (~2.8%)

## 9.2 Varlen small‑S check
Config:
- `B=4, T≈797, max_seqlen≈249` with same `H,Hk,Dh,R` and `causal=True`

Observed:
- `finite=True`
- `max_abs ≈ 1.01e+01`
- `rel_fro_mean ≈ 2.62e-02` (~2.6%)

## 9.3 Stress stability (scale Q/K factors)
Config:
- `B=2, S=1024, H=32, Hk=8, Dh=128, R=64, bf16, causal`

Observed:
- `scale ∈ {1,3,10,30}` → **finite=True** for all
- `max|O|` stayed bounded (~45), which is expected since output is a convex combination of V.

---

# 10) Sweeps: a practical plan to tune BM/BN/BR/warps/stages

## 10.1 Why sweeps matter here
For GQA with `REP=H/Hk > 1`, a single program carries multiple heads’ state:
- per‑head `m_i`, `l_i`, `acc`
- per‑head `q0r`, `q1r`
This can increase register pressure → lower occupancy or spill.

Because counters may be locked, **parameter sweeps + trend analysis** is the most reliable tuning method.

## 10.2 Minimal sweep set (recommended on H200)
Start with these “high‑leverage” changes:

### A) Reduce BM and BR (often helps if register pressure/spill is the bottleneck)
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode packed --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 --dtype bf16 --causal \
  --bm 32 --bn 64 --br 32 --warps 4 --stages 4 --warmup 30 --iters 120
```

### B) Increase BN (H200 can benefit from bigger key tiles)
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode packed --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 --dtype bf16 --causal \
  --bm 32 --bn 128 --br 32 --warps 4 --stages 4 --warmup 30 --iters 120
```

### C) Keep BM moderate, BN larger, BR smaller
```bash
CUDA_VISIBLE_DEVICES=1 python flashsvd.py --mode packed --B 8 --S 2048 --H 32 --Hk 8 --Dh 128 --R 64 --dtype bf16 --causal \
  --bm 64 --bn 128 --br 32 --warps 8 --stages 4 --warmup 30 --iters 120
```

## 10.3 Full sweep loop (bash)
This runs a small grid of configs and prints a line per run. It’s crude but effective.

```bash
export CUDA_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR=/tmp/triton_cache

B=8; S=2048; H=32; HK=8; DH=128; R=64; DT=bf16

for BM in 32 64 128; do
  for BN in 64 128; do
    for BR in 32 64; do
      for W in 4 8; do
        for ST in 3 4; do
          echo "BM=$BM BN=$BN BR=$BR W=$W ST=$ST"
          python flashsvd.py --mode packed --B $B --S $S --H $H --Hk $HK --Dh $DH --R $R --dtype $DT --causal \
            --bm $BM --bn $BN --br $BR --warps $W --stages $ST --warmup 30 --iters 100
        done
      done
    done
  done
done
```

## 10.4 How to interpret sweep trends (without counters)
- If **smaller BM/BR** improves latency → likely register pressure / occupancy / spill
- If **larger BN** improves latency → likely better amortization/memory behavior in K/V rebuild & dot
- If **stages↑** improves latency → load/compute overlap was insufficient

---

# 11) Bonus: “Fair baseline” benchmarking (fused vs unfused pipeline)

Because our kernel fuses:
- low‑rank reconstruction (P@V)
- RoPE
- attention (QK + softmax + PV)
…a fair baseline should include the same steps:

1) reconstruct dense Q,K,V from low‑rank factors
2) apply RoPE
3) run FlashAttention (preferred) or SDPA fallback

We provide a harness script (recommended):
- `bench_flashsvd.py`
which prints:
- end‑to‑end latency
- peak memory
- kernel count (if profiling enabled)
- FLOPs:
  - attention‑only TFLOPs (QK+PV)
  - **end‑to‑end TFLOPs (recon + attention)** ← recommended

---

## Appendix: Useful one-liners

### List available nsys stats reports
```bash
nsys stats --help-reports
```

### Export CSV from nsys stats
```bash
nsys stats --report cuda_gpu_kern_sum --format csv --output . nsys_flashsvd.nsys-rep
```

### Quick “is the GPU busy” sampling
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,clocks.mem \
  --format=csv -l 1
```
