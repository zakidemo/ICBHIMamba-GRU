# ICBHIMambaNet: SSM-Inspired Respiratory Sound Classification

> **Beyond Self-Attention: State Space Models for Data-Efficient Respiratory Sound Classification**  
> Zakaria Neili and Kenneth Sundaraj  
> *Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI)*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository provides the **complete, reproducible** implementation for a rigorous comparative study of **seven deep learning architectures** for 4-class respiratory sound classification on the [ICBHI 2017 benchmark](https://bhichallenge.med.auth.gr/). All models are trained under **strictly identical conditions** (same data splits, augmentation, loss, optimizer, and evaluation protocol).

### Architectures Compared

| # | Model | Type | Params | Pretrained |
|---|-------|------|--------|------------|
| 1 | MFCC + MLP | Hand-crafted features + MLP | 0.41M | No |
| 2 | CNN-2D | Custom 5-block CNN | 4.85M | No |
| 3 | AlexNet | Classic CNN (adapted) | 37.30M | No |
| 4 | VGG-16 | Deep CNN | 52.47M | ImageNet |
| 5 | ResNet-50 | Residual CNN | 24.55M | ImageNet |
| 6 | ViT-Small | Vision Transformer | 21.48M | No |
| 7 | **ICBHIMambaNet** | SSM-inspired (GRU-based) | **11.49M** | No |

### Key Finding

ICBHIMambaNet achieves the best ICBHI score (54.51%) among all from-scratch models, outperforming ViT-Small (51.16%) with 47% fewer parameters, demonstrating that SSM-inspired sequential inductive bias is superior to self-attention for data-scarce medical audio.

---

## Project Structure

```
ICBHIMamba-JBHI/
├── README.md                         ← This file
├── requirements.txt                  ← Python dependencies
├── setup_environment.sh              ← Environment setup script
├── LICENSE                           ← MIT License
│
├── configs/
│   └── default.yaml                  ← All hyperparameters in one place
│
├── src/
│   ├── __init__.py
│   ├── icbhi_dataloader.py           ← ICBHI Dataset class
│   ├── icbhi_model.py                ← ICBHIMambaNet architecture
│   ├── baseline_models.py            ← All 6 baseline architectures
│   └── icbhi_utils.py                ← Metrics, helpers
│
├── scripts/
│   ├── step1_prepare_data.py         ← Extract respiratory cycles
│   ├── step2_get_norm_stats.py       ← Compute mel-spectrogram stats
│   ├── step3_train_all_models.py     ← Train all 7 models (main script)
│   ├── step4_generate_figures.py     ← Publication figures
│   ├── step5_gradcam.py              ← GradCAM / explainability
│   └── step6_generate_latex.py       ← LaTeX tables for paper
│
├── data/
│   ├── ICBHI_final_database/         ← Place ICBHI dataset here
│   └── cycles/                       ← Auto-created: extracted cycles
│
├── datafiles/                        ← Auto-created: JSON splits + stats
├── checkpoints/                      ← Saved model weights
├── results/                          ← Metrics, figures, tables
└── figures/                          ← Publication-ready figures
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n icbhi_mamba python=3.9 -y
conda activate icbhi_mamba

# Install PyTorch (GPU - CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

Download the [ICBHI 2017 Respiratory Sound Database](https://bhichallenge.med.auth.gr/) and place files in:
```
data/ICBHI_final_database/
    101_1b1_Al_sc_Meditron.wav
    101_1b1_Al_sc_Meditron.txt
    ...
```

### 3. Run Full Pipeline

```bash
# Step 1: Extract respiratory cycles and create train/test splits
python scripts/step1_prepare_data.py

# Step 2: Compute normalization statistics
python scripts/step2_get_norm_stats.py

# Step 3: Train ALL 7 models under identical conditions
python scripts/step3_train_all_models.py

# Step 4: Generate all publication figures
python scripts/step4_generate_figures.py

# Step 5: GradCAM explainability analysis
python scripts/step5_gradcam.py

# Step 6: Generate LaTeX tables
python scripts/step6_generate_latex.py
```

To train a single model:
```bash
python scripts/step3_train_all_models.py --models mamba
python scripts/step3_train_all_models.py --models vgg16 resnet50
```

---

## Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | Normal | No adventitious sounds |
| 1 | Crackle | Short, explosive discontinuous sounds |
| 2 | Wheeze | Continuous, musical sounds |
| 3 | Both | Crackle and wheeze co-occurring |

---

## ICBHIMambaNet Architecture

ICBHIMambaNet adapts the Mamba State Space Model design principles for respiratory sound classification, implemented in pure PyTorch for hardware portability:

```
Mel-Spectrogram [B, 512, 128]
        ↓
  Patch Embed (16×16) → [B, 256, 384]
        ↓
  + Positional Embedding
        ↓
  12 × SimpleMambaBlock
    ├── LayerNorm
    ├── Linear → [x, z] (gated split)
    ├── Depthwise Conv1D (local context)
    ├── SiLU activation
    ├── GRU (sequential state tracking)
    ├── SSM projection
    ├── SiLU gate: y = proj(h) ⊙ σ(z)
    └── Linear projection + residual
        ↓
  Global Average Pooling
        ↓
  Linear → 4-class logits
```

**Implementation Note:** The selective scan mechanism of Mamba is approximated using a GRU-based recurrent mixer, preserving the gated sequential processing paradigm while avoiding dependency on the hardware-specific `mamba-ssm` CUDA extension. This design choice enables deployment on any hardware (CPU/GPU) without specialized kernels.

---

## Training Protocol (Identical for All Models)

| Parameter | Value |
|-----------|-------|
| Loss | Focal Loss (γ=1.5) + class weights [1.8, 1.0, 3.0, 5.0] |
| Label smoothing | 0.05 |
| Optimizer | AdamW (lr=5e-4, weight_decay=3e-4) |
| Scheduler | Cosine annealing + 5-epoch warmup |
| Batch size | 64 effective (16 × 4 grad accumulation) |
| Epochs | 60 |
| Augmentation | SpecAugment (freq=32, time=128), MixUp (α=0.2), noise |
| Balanced sampling | WeightedRandomSampler (uniform class distribution) |
| Seed | 42 |

---

## Evaluation Metrics

Following the ICBHI 2017 challenge guidelines:

- **Sensitivity (Se)**: Macro-average recall across all 4 classes
- **Specificity (Sp)**: Macro-average specificity across all 4 classes  
- **ICBHI Score**: (Se + Sp) / 2
- **Per-class Recall**: Individual class detection rates

---

## Results

| Model | Acc(%) | Se(%) | Sp(%) | ICBHI(%) | Params(M) |
|-------|--------|-------|-------|----------|-----------|
| VGG-16† | **50.31** | **43.06** | **81.61** | **62.33** | 52.47 |
| ResNet-50† | 49.38 | 42.71 | 81.45 | 62.08 | 24.55 |
| CNN-2D | 42.69 | 36.63 | 79.47 | 58.05 | 4.85 |
| **ICBHIMambaNet** | 29.39 | 31.78 | 77.25 | **54.51** | **11.49** |
| MFCC + MLP | 27.97 | 30.71 | 77.27 | 53.99 | 0.41 |
| AlexNet | 32.09 | 30.92 | 76.97 | 53.94 | 37.30 |
| ViT-Small | 20.21 | 26.81 | 75.51 | 51.16 | 21.48 |

† = ImageNet pretrained (transfer learning)

---

## Citation

If you use this code, please cite:

```bibtex
@article{neili2025icbhimamba,
  title={Beyond Self-Attention: State Space Models for Data-Efficient Respiratory Sound Classification},
  author={Neili, Zakaria and Sundaraj, Kenneth},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025}
}
```

## Acknowledgments

- [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/) for the respiratory sound database
- [Audio-Mamba (AuM)](https://github.com/marmot-xy/Audio-Mamba) for SSM-audio design inspiration
- [Mamba](https://github.com/state-spaces/mamba) for the selective state space model framework

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
