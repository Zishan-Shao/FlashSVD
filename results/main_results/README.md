# Main Results Bundle

This folder contains a compact, main-text-oriented results package derived from the current FlashSVD v1.5 runtime studies.

## Recommended main-paper items

- `main/figure_main.pdf`: one compact figure for headline runtime results, family coverage, and long-prompt behavior.
- `ablation/figure_ablation.pdf`: one compact figure for graph granularity and supporting runtime ablations.
- `main/table_main.tex`: main repeated-run table, family summary table, long-prompt table, and decode-length sweep table up to 16K generated tokens.
- `ablation/table_ablation.tex`: graph ablation, fragmentation counts, attention-route microbench, and kernel micro-ablation tables.

## Notes

- The figures are exported as SVG, PNG, and PDF. The PDF export is rasterized at 300 DPI.
- The tables are ready to include in LaTeX and assume `booktabs` and `threeparttable`.
