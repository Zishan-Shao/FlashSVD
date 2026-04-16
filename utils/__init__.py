"""Utility helpers for SVD-LLM style scripts (FlashSVD-v1.5).

This package is intentionally lightweight and self-contained so that
`FlashSVD-v1.5/SVDLLM*.py` scripts can run without depending on the
original upstream repo layout.
"""

from _path_setup import ensure_active_src_on_path

ensure_active_src_on_path()
