from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


def _read_readme() -> str:
    readme_path = Path("README.md")
    if not readme_path.exists():
        return ""
    return readme_path.read_text(encoding="utf-8")


ROOT_PACKAGES = find_packages(
    where=".",
    exclude=[
        "src",
        "src.*",
        "scripts",
        "scripts.*",
        "component",
        "component.*",
        "flashsvd_component",
        "flashsvd_component.*",
        "methods",
        "methods.*",
        "quant",
        "quant.*",
    ],
)

SRC_PACKAGES = find_packages(
    where="src",
    include=[
        "component",
        "component.*",
        "flashsvd_component",
        "flashsvd_component.*",
        "methods",
        "methods.*",
        "quant",
        "quant.*",
    ],
)

PACKAGES = sorted(set(ROOT_PACKAGES + SRC_PACKAGES))


setup(
    name="flashsvd",
    version="1.5.0",
    description="FlashSVD v1.5 streaming runtime",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    packages=PACKAGES,
    package_dir={
        "component": "src/component",
        "flashsvd_component": "src/flashsvd_component",
        "methods": "src/methods",
        "methods.basis_sharing": "src/methods/basis_sharing",
        "methods.svdllm": "src/methods/svdllm",
        "quant": "src/quant",
    },
    python_requires=">=3.10",
    install_requires=[
        "accelerate",
        "datasets",
        "huggingface-hub",
        "numpy",
        "safetensors",
        "sentencepiece",
        "transformers",
        "tqdm",
    ],
    extras_require={
        "test": ["pytest"],
        "research": [
            "evaluate",
            "matplotlib",
            "scikit-learn",
            "scipy",
        ],
    },
    include_package_data=True,
)
