# Main Results Bundle

This folder contains a compact, main-text-oriented results package derived from the current FlashSVD v1.5 runtime studies.

## Recommended main-paper items

- `main/figure_main.pdf`: one compact figure for headline runtime results, family coverage, and long-prompt behavior.
- `main/figure_robustness.pdf`: one 4-panel robustness figure combining ratio sweep and long-generation decode trends.
- `main/figure_ratio_robustness.pdf`: a 2-panel ratio-only figure, useful if Table 3 is converted into a figure.
- `main/figure_long_generation_robustness.pdf`: a 2-panel long-generation-only figure, useful if Table 4 is converted into a figure.
- `ablation/figure_ablation.pdf`: one compact figure for graph granularity and supporting runtime ablations.
- `ablation_255/figure_ablation_255.pdf`: ablation figure bundle rewritten around the current exported-HF `2.55x` headline.
- `main/table_main.tex`: one unified main-text table over the public LowRankArena checkpoint sweep, with one row per family/ratio entry and grouped columns for FlashSVD absolute latency plus speedups against DenseKV and dense LLaMA-7B baselines.
- `ablation/table_ablation.tex`: graph ablation, fragmentation counts, attention-route microbench, and kernel micro-ablation tables.

## Notes

- The figures are exported as SVG, PNG, and PDF. The PDF export is rasterized at 300 DPI.
- The tables are ready to include in LaTeX and assume `booktabs` and `threeparttable`.
- `main/figure_robustness_notes.md` contains a suggested caption and the exact data sources used for the new robustness figures.
- `ablation_255/analysis.md` explains how to map the current `2.55x` headline onto graph, fragmentation, and attention-route ablations without conflating internal ablation gains with external baseline speedups.
- In this results bundle, `DenseKV` refers to the dense-KV baseline that uses dense KV cache and `flash_attn_with_kvcache` decode on the same compressed checkpoint.
- When `StaticCache` is referenced for the exported-HF LowRankArena checkpoints, it refers to the standard HF `StaticCache` path; the verified checkpoint reload path uses HF attention with `_attn_implementation = sdpa`.
