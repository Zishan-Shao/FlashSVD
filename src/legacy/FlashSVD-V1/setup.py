from setuptools import setup, find_packages
from pathlib import Path

readme = ""
p = Path("README.md")
if p.exists():
    readme = p.read_text(encoding="utf-8")

setup(
    name="flashsvd-local",
    version="0.0.0",              
    description="FlashSVD (local install)",
    long_description=readme,
    long_description_content_type="text/markdown",

    package_dir={"": "src"},
    packages=find_packages(where="."),

    python_requires=">=3.8",
    install_requires=[
        "transformers",
        "datasets",
        "evaluate",
        "accelerate",
        "matplotlib",
        "numpy",
        "tqdm",
        "gradio",
        "scipy",
        "scikit-learn",
        "triton"
    ],
    scripts=[
    ],

    include_package_data=True,
)