# Paper Results Bundle

This folder packages the current FlashSVD-v1.5 runtime study into paper-facing figures, compact tables, and setup notes.

## Figures

- [fig01_main_decode_latency.svg](./figures/fig01_main_decode_latency.svg)
- [fig02_stage_breakdown.svg](./figures/fig02_stage_breakdown.svg)
- [fig03_graph_ablation.svg](./figures/fig03_graph_ablation.svg)
- [fig04_kernel_fragmentation_counts.svg](./figures/fig04_kernel_fragmentation_counts.svg)
- [fig05_runtime_overhead_cpu.svg](./figures/fig05_runtime_overhead_cpu.svg)
- [fig06_module_breakdown_nograph.svg](./figures/fig06_module_breakdown_nograph.svg)
- [fig07_correctness_audit.svg](./figures/fig07_correctness_audit.svg)
- [fig08_attention_route_microbench.svg](./figures/fig08_attention_route_microbench.svg)
- [fig09_attention_reconstruct_ablation.svg](./figures/fig09_attention_reconstruct_ablation.svg)
- [fig10_mlp_backend_ablation.svg](./figures/fig10_mlp_backend_ablation.svg)
- [fig11_nograph_top_op_classes.svg](./figures/fig11_nograph_top_op_classes.svg)

## Tables

- [main_results.csv](./tables/main_results.csv)
- [graph_ablation.csv](./tables/graph_ablation.csv)
- [module_breakdown.csv](./tables/module_breakdown.csv)
- [correctness_summary.csv](./tables/correctness_summary.csv)
- [motivation_counts.csv](./tables/motivation_counts.csv)
- [attn_route_microbench.csv](./tables/attn_route_microbench.csv)
- [attn_reconstruct_summary.csv](./tables/attn_reconstruct_summary.csv)
- [mlp_backend_summary.csv](./tables/mlp_backend_summary.csv)
- [nograph_top_op_classes_short.csv](./tables/nograph_top_op_classes_short.csv)
- [correctness_prompts.csv](./tables/correctness_prompts.csv)

## Notes

- Experiment setup and dataset notes: [EXPERIMENT_SETUP_AND_DATASETS.md](./EXPERIMENT_SETUP_AND_DATASETS.md)
- Kernel-call explanation: [KERNEL_CALL_TAXONOMY.md](./KERNEL_CALL_TAXONOMY.md)

## Main takeaways

- The active per-layer graph runtime is the dominant systems win. It reduces launch and staging overhead enough to turn FlashSVD-v1.5 into a large end-to-end decode speedup over both `StaticCache` and `DenseKVCacheBaseline`.
- The fairness story is solid when phrased as a same-condition `bf16` serving comparison, with `fp32 no-cache` plus `fp32 StaticCache cached` used as the correctness anchor.
- The remaining bottleneck is not one giant math kernel. It is the thin-serving problem: launch count, graph boundaries, copies, and dtype/layout traffic.
