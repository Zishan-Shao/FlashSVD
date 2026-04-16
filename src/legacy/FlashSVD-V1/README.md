# FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Stars](https://img.shields.io/github/stars/Zishan-Shao/FlashSVD?style=social)](https://github.com/Zishan-Shao/FlashSVD)

This repository contains the official implementation of **FlashSVD**, a novel end-to-end rank-aware streaming inference framework specifically designed for SVD-compressed large language models. FlashSVD addresses the critical limitation of previous SVD-based compression techniques by eliminating activation memory overhead during inference.


### Support & Contact

- Issues and bugs: please open a GitHub issue.
- Feature requests (e.g., new model support or SVD methods): open an issue with details.
- Collaboration or questions: email Zishan at zs89@duke.edu.

We aim to respond promptly. This is an active, long-term project, and we welcome community contributions.

**📄Paper**: [FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models](https://arxiv.org/abs/2508.01506)

> 🙏 If you find FlashSVD useful in your research, we kindly ask that you `cite our paper` (see [Citation](#citation)). If this repository is helpful, please consider `starring 🌟` it to support the project — thank you!
>
> [![Cite FlashSVD](https://img.shields.io/badge/Cite-FlashSVD-brightgreen)](#citation) [![Star this repo](https://img.shields.io/badge/Star-This%20repo-yellow?logo=github)](https://github.com/Zishan-Shao/FlashSVD/stargazers)


## 🚀 Announcement
Our system involves have several popular SVD method (Unofficial) replication with code:

#### Encoders:

 - [Language model compression with weighted low-rank factorization](https://arxiv.org/abs/2207.00112): Fisher-Weighted SVD (FWSVD) is supported for BERT, RoBERTa, and ModernBERT
 - [DRONE: Data-aware Low-rank Compression for Large NLP Models](https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf): data whitening method enabled now on BERT
 - [Adaptive Rank Selections for Low-Rank Approximation of Language Models](https://aclanthology.org/2024.naacl-long.13/): AdaSVD code on BERT

#### Decoders:

- [ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models](https://proceedings.neurips.cc/paper/2021/file/f56de5ef149cf0aedcc8f4797031e229-Paper.pdf): enabling low-rank kv-cache inference. We support asvd with uniform rank (hetero-rank case in development) on `Llama-2-7b model` and `gpt2` model

### 🛠️ TODO
- [✅] BERT (CLS / MLM) support  
- [✅] Fisher-Weighted SVD (FWSVD) replication  
- [✅] DRONE & Adaptive Rank Selection integration  
- [✅] LLaMA support (ASVD)  
- [ ] Qwen integration  
- [ ] LLaMA, GPT-2 support (SVD-LLM, Dobi-SVD etc.)  
- [✅] GPT-2 SVD (ASVD)  
- [✅] Add benchmark results and visualization tools


## 🔍 Overview

Singular Value Decomposition (SVD) has recently seen a surge of interest as a simple yet powerful tool for large language models (LLMs) compression, with a growing number of works demonstrating 20-80% parameter reductions at minimal accuracy loss. However, previous SVD-based approaches have focused primarily on reducing the memory footprint of model weights, largely overlooking the additional activation memory overhead incurred during inference when applying truncated factors via standard dense CUDA kernels.

Our experiments demonstrate that this activation overhead, scaling with sequence length and hidden dimension, prevents current SVD compression techniques from achieving any reduction in peak inference memory, thereby limiting their viability for real-world, on-device deployments.

### Pipeline

![FlashSVD Pipeline](figs/pipeline.png)

The figure above illustrates the FlashSVD computation pipeline, showing the efficient flow from input through low-rank attention and feed-forward layers.


### 🧰 Key Contributions

We introduce **FlashSVD**, a novel, end-to-end rank-aware streaming inference framework specifically designed for SVD-compressed large language models. FlashSVD can be seamlessly integrated with any model that employs SVD-based methods for parameter reduction. By fusing low-rank projection kernels directly into both the self-attention and feed-forward network (FFN) pipelines, FlashSVD avoids materializing full-size activation buffers. Instead, small tiles of the truncated factors are loaded into on-chip SRAM, multiplied and reduced on the fly, and immediately evicted, preserving high GPU occupancy and adding no extra latency.

- **End-to-End Streaming Framework**: Rank-aware inference system for SVD-compressed models
- **Fused Low-Rank Kernels**: Direct integration into attention and FFN pipelines  
- **Tile-Based Computation**: Avoids materializing full-size activation buffers
- **Memory-Efficient Deployment**: Up to 70.2% reduction in peak activation memory

## 🧠 Key Features

- **Universal Integration**: Seamlessly works with any SVD-compressed model
- **Streaming Inference**: Tile-based computation avoids activation buffer materialization
- **GPU Optimized**: Fused kernels preserve high GPU occupancy with no extra latency on medium-low ranked cases
- **Memory Efficient**: Up to 70.2% reduction in peak activation memory
- **Accuracy Preserving**: No accuracy loss with upstream compression methods

## 🧩 Installation

### Prerequisites
- On Linux
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- PyTorch 1.12+ with CUDA support

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zishan-Shao/FlashSVD.git
   cd FlashSVD
   ```

2. **Install dependencies**
   
   ⚠️ **You must manually install PyTorch (CPU or CUDA version) before running the script.**  
   Example manual installation:

   ```bash
   # CUDA 12.1
   pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

   # or CPU only
   pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
   ```

   ### Option A — Using the provided `install_local.sh` (recommended)
   If you just want to install the project locally with an isolated environment, 
   run the included helper script:

   ```bash
   # GPU mode (requires you to manually install CUDA-enabled PyTorch beforehand)
   ./install_local.sh
   ```
   > 💡 **This script will:**
   >
   > - Create and activate a local virtual environment (`.venv`)
   > - Upgrade `pip`
   > - Check your existing PyTorch installation (no auto CUDA download)
   > - Install project dependencies in editable mode (`pip install -e .`)
   > 

   ### Option B — Using Conda environment (alternative for GPU users)
   If you prefer Conda for easier CUDA management:

   ```bash
   conda create -n flashsvd python=3.10 
   conda activate flashsvd

   # Install project dependencies
   pip install -e .
   ```
   ### Option C — Manual venv setup (lightweight alternative)
   If you want to manage everything manually without Conda or the helper script:
   ```python
   python3 -m venv .venv
   source .venv/bin/activate   

   # Manually install PyTorch first (choose one)
   pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio   # CUDA 12.1
   # Then install this project
   pip install -e .

   ```
## ⚡ Quick Start

### 1. Training BERT with FlashSVD in Gradio UI (Recommended)
This project provides an interactive **Gradio interface** to run the unified training script easily.
```bash
# Launch the Gradio app 
python app.py
```
   ### UI Overview

   #### Mode
   Choose `cls` (classification) or `mlm` (masked language modeling).  
   The available tasks will automatically change depending on the selected mode.

   #### Task
   Supported GLUE tasks (e.g., `sst2`, `qnli`, `mnli`, `stsb`, etc.).

   #### Model Checkpoint
   Example — `bert-base-uncased`.

   #### Training Parameters
   - `epochs`
   - `batch size`
   - `learning rate`
   - `logging steps`
   - `evaluation steps`
   - `random seed`

   #### Advanced Options
   - **Force CPU (`--no_cuda`)**: Force CPU training.
   - **CUDA_VISIBLE_DEVICES**: e.g., `0` or `0,1` if multiple GPUs are available.
   - **Extra CLI Args**: Additional command-line arguments appended at runtime.

   #### Logging
   - **Log Directory**: default `runs/`
   - **Run Name**: leave blank to auto-generate a timestamp name.
   - **Append if log exists**: append logs instead of overwriting.
   - Logs are stored under `runs/<run_name>.log` and can be downloaded in the UI.

   #### Internal Execution
   Internally, the Gradio app executes a command equivalent to:

   ```bash
   python -u $TRAIN_UNIFIED_SCRIPT \
   --mode <mode> --task <task> --model <checkpoint> \
   --epochs <n> --batch_size <n> --learning_rate <lr> \
   --logging_steps <n> --eval_steps <n> --seed <n> \
   [--output_dir <path>] [--no_cuda] [<extra_args>]
   ```

### Command Line Usage (Without UI)
Train BERT models with specific GLUE tasks and rank configurations:

```bash
python train_bert_unified_min.py \
  --mode cls --task sst2 --model bert-base-uncased \
  --epochs 3 --batch_size 32 --learning_rate 2e-5 \
  --logging_steps 100 --eval_steps 0 --seed 0

# Masked Language Modeling example
python train_bert_unified_min.py \
  --mode mlm --task mnli --model bert-base-uncased \
  --epochs 3 --batch_size 32 --learning_rate 2e-5 \
  --logging_steps 100 --eval_steps 0 --seed 0
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



## 📊 Results

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

<!-- **Paper**: [FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models](https://arxiv.org/abs/2508.01506) -->
<!-- 
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
``` -->

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

**Note**: For support, feature requests, or collaborations, please open a GitHub issue or email Zishan (zs89@duke.edu).
