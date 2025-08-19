# FlashSVD: Efficient Transformer Architecture with Low-Rank Decomposition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the official implementation of **FlashSVD**, a novel approach for efficient transformer architectures using low-rank SVD decomposition. FlashSVD significantly reduces computational complexity and memory usage while maintaining model performance across various NLP tasks.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Benchmarking](#benchmarking)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Overview

FlashSVD introduces an efficient transformer architecture that leverages Singular Value Decomposition (SVD) to reduce the computational complexity of attention mechanisms and feed-forward networks. The approach achieves significant speedup and memory reduction while maintaining competitive performance on standard NLP benchmarks.

### Pipeline

![FlashSVD Pipeline](figs/pipeline.png)

The figure above illustrates the FlashSVD computation pipeline, showing the efficient flow from input through low-rank attention and feed-forward layers.


### Key Contributions

- **Low-Rank Attention**: Efficient attention computation using SVD decomposition
- **Optimized FFN**: Streamlined feed-forward networks with reduced parameters
- **Memory-Efficient**: Significant reduction in GPU memory usage
- **Performance-Preserving**: Maintains accuracy across GLUE benchmark tasks

## Key Features

- **Multiple Model Support**: BERT, RoBERTa, and RoBERTa-Large implementations
- **Flexible Rank Configuration**: Adjustable rank parameters for attention and FFN
- **Easy Integration**: Available triton kernel for direct drop-in in forward process
- **CUDA Optimized**: Efficient GPU implementations using Triton kernels

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- PyTorch 1.12+ with CUDA support

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/FlashSVD.git
   cd FlashSVD
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets evaluate accelerate
   pip install triton 
   ```

3. **Install additional requirements:**
   ```bash
   pip install matplotlib numpy tqdm
   ```

## Quick Start

### 1. Training BERT with FlashSVD

Train BERT models with specific GLUE tasks and rank configurations:

```bash
# Train BERT on SST-2 task with custom ranks
python train_bert.py --task sst2 --mode cls --epochs 3 --bsz 32

# Train BERT on QNLI task
python train_bert.py --task qnli --mode cls --epochs 3 --bsz 32

# Train BERT on RTE task
python train_bert.py --task rte --mode cls --epochs 3 --bsz 32

# Train RoBERTa models
python train_roberta.py
python train_roberta_large.py
```

### 2. Inference and Profiling

After training, run inference with profiling in the BERT/BERTFW directories:

```bash
# Navigate to BERT directory for standard inference
cd BERT/
python profile_flashsvd.py  # or your specific profiling script

# Navigate to BERTFW directory for FlashSVD inference
cd BERTFW/
python profile_flashfwsvd.py  # or your specific profiling script
```

The profiling scripts will provide detailed performance metrics including:
- Inference latency
- Memory usage
- Comparison between standard and FlashSVD implementations



## Results

### Performance Comparison

FlashSVD achieves significant improvements in efficiency:

- **Memory Reduction**: Up to 70.2% reduction in peak activation memory
- **Intermediate Memory**: 75% reduction in transient memory usage
- **Accuracy Preservation**: No accuracy loss with upstream compression methods
- **Practical Deployment**: Enables memory-constrained deployment of low-rank LLMs

### Rank Loss Analysis

![Rank Loss Comparison](figs/rank_loss_comparison.png)

The figure above shows the trade-off between rank reduction and model performance across different tasks.

### Key Contributions

Our work addresses the critical limitation of previous SVD-based approaches by introducing:

- **End-to-end rank-aware streaming inference framework**
- **Fused low-rank projection kernels** for both attention and FFN
- **Tile-based computation** that avoids materializing full-size activation buffers
- **Seamless integration** with any SVD-compressed model

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{shao2025flashsvd,
  title={FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models},
  author={Shao, Zishan and Wang, Yixiao and Wang, Qinsi and Jiang, Ting and Du, Zhixu and Ye, Hancheng and Zhuo, Danyang and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2508.01506},
  year={2025}
}
```

**Paper**: [FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models](https://arxiv.org/abs/2508.01506)

## Project Structure

```
FlashSVD/
├── src/                    # Core implementation
│   ├── kernels/           # CUDA kernels and optimizations
│   └── utils/             # Utility functions and blocks
├── models/                # Pre-trained model checkpoints
│   ├── BERT/             # BERT model variants
│   └── RoBERTa/          # RoBERTa model variants
├── benchmark/             # Performance evaluation scripts
├── figs/                  # Paper figures and diagrams
├── train_*.py            # Training scripts for different models
└── README.md             # This file
```

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

**Note**: This implementation is based on research work. For questions or issues, please open an issue on GitHub.
