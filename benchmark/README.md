# FlashSVD-v1.5 Benchmarks

This folder is the entrypoint for:

- correctness checks
- end-to-end decode benchmarks
- MLP microbenchmarks
- legacy experiment harnesses that should not live next to production kernels

Layout:

- `attn/`
  current attention-focused microbenchmarks
- `decode/`
  decode correctness and end-to-end serving benchmarks
- `mlp/`
  MLP microbenchmarks and backend comparisons
- `legacy/archive/`
  old archive compare and microbench scripts that were moved out of `kernels/flashsvd-archive/`
  including the unified `decoding/` legacy folder
- `legacy/experimental/`
  old experimental kernels kept only for legacy benchmark scripts
- `mlp/legacy_swiglu/`
  older SwiGLU experiment code kept only for benchmarking and regression work
