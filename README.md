# Mamba-GRU---an SSM-inspired architecture using a GRU-based recurrent state mixer
> **Mamba-GRU: Selective State Space Modeling in Addressing Minority-Class Collapse Towards Clinical Respiratory Sound Classification**  
> Zakaria Neili and Kenneth Sundaraj  
> *Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI)*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository provides the **complete, reproducible** implementation for a rigorous comparative study of **eight deep learning architectures** for 4-class respiratory sound classification on the [ICBHI 2017 benchmark](https://bhichallenge.med.auth.gr/). All models are trained under **strictly identical conditions** (same data splits, augmentation, loss, optimizer, and evaluation protocol).

### Architectures Compared

| # | Model | Type | Params | Pretrained |
|---|-------|------|--------|------------|
| 1 | MFCC + MLP | Hand-crafted features + MLP | 0.41M | No |
| 2 | CNN-2D | Custom 5-block CNN | 4.85M | No |
| 3 | AlexNet | Classic CNN (adapted) | 37.30M | No |
| 4 | VGG-16 | Deep CNN | 52.47M | ImageNet |
| 5 | ResNet-50 | Residual CNN | 24.55M | ImageNet |
| 6 | ViT-Small | Vision Transformer | 21.48M | No |
| 7 | **Mamba-GRU** | SSM-inspired (GRU-based) | **11.49M** | No |
| 8 | Mamba-SSM | native Mamba selective scan kernel | 11.78M | No |

### Key Finding

We present Mamba-GRU, an SSM-inspired model using a GRU-based recurrent state mixer, which is the only from-scratch sequence model that maintains
balanced class discrimination across all four pathological sound types, achieving 56.33% ICBHI score with only 11.49M parameters.

---

## Project Structure

```
Mamba-GRU-JBHI/
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

## Mamba-GRU Architecture

Mamba-GRU adapts the Mamba State Space Model design principles for respiratory sound classification, implemented in pure PyTorch for hardware portability:

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

### Results

| Model     | Acc (%)      | Se (%)          | Sp (%)| ICBHI(%)    | Normal (%) | Crackle (%) | Wheeze (%) | Both (%) | Params    | Inference(ms)|
|-----------|--------------|-----------------|-------|-------------|------------|-------------|------------|----------|-----------|--------------|
| VGG-16    | 60.25        | 47.65           | 84.61 | 66.13       | 72.00      | 53.71       | 41.26      | 23.61    | 52.47M    | 3.675        |
| ResNet-50 | 59.07        | 44.36           | 83.87 | 64.11       | 75.48      | 39.39       | 51.46      | 11.11    | 24.55M    | 3.578        |
| CNN-2D    | 39.82        | 34.64           | 79.64 | 57.14       | 48.39      | 24.81       | 41.75      | 23.61    | 4.85M     | 1.942        |
| ViT-Small | 26.45        | 34.03           | 78.73 | 56.38       | 37.81      | 0.00        | 13.59      | 84.72    | 21.48M    | 3.541        |
| Mamba-GRU |35.18         | 35.87           | 76.80 | 56.33       | 40.77      | 29.67       | 17.48      | 55.56    | **11.49M**| 5.227        |
| Mamba-SSM | 24.79        | 33.23           | 78.83 | 56.03       | 37.03      | 0.00        | 1.46       | 94.44    | 11.78M    | 4.112        |
| MFCC+MLP  | 20.15        | 31.55           | 77.26 | 54.41       | 26.71      | 0.00        | 9.22       | 90.28    | 0.41M     | 1.032        |
| AlexNet   | 17.31        | 30.55           | 76.80 | 53.68       | 21.94      | 0.00        | 5.83       | 94.44    | 37.30M    | 1.529        |

* = ImageNet pretrained (transfer learning)

---

## Citation

If you use this code, please cite:

```bibtex
@article{neili2025icbhimamba,
  title={Mamba-GRU: Hardware‑Agnostic Selective State Space Modeling for Respiratory Sound Classification},
  author={Neili, Zakaria and Sundaraj, Kenneth},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026}
}
```

## Acknowledgments

- [ICBHI 2017 Challenge](https://bhichallenge.med.auth.gr/) for the respiratory sound database
- [Audio-Mamba (AuM)](https://github.com/marmot-xy/Audio-Mamba) for SSM-audio design inspiration
- [Mamba](https://github.com/state-spaces/mamba) for the selective state space model framework

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
