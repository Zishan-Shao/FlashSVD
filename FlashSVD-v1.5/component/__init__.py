"""Compatibility components for `SVDLLM.py`.

The original SVD-LLM scripts expect a `component.*` package. In this repo we
vendor FlashSVD-enabled implementations under `flashsvd_component.*`.

This package provides thin wrappers so `SVDLLM.py` can run unmodified and
produce checkpoints.
"""

