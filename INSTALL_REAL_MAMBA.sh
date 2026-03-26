#!/bin/bash
# =============================================================================
# STEP-BY-STEP: Install Real Mamba SSM on WSL (RTX 4070)
# =============================================================================
# Run these commands ONE BY ONE in your WSL terminal
# =============================================================================

echo "============================================"
echo "  Step A: Check GPU & CUDA"
echo "============================================"

# 1. Verify GPU is visible
nvidia-smi

# 2. Check CUDA version
nvcc --version

# If nvcc not found, install CUDA toolkit:
# sudo apt install nvidia-cuda-toolkit

echo "============================================"
echo "  Step B: Create/Activate Environment"
echo "============================================"

# Option 1: If you already have icbhi_mamba env
conda activate icbhi_mamba

# Option 2: Fresh environment
# conda create -n icbhi_mamba python=3.10 -y
# conda activate icbhi_mamba
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "============================================"
echo "  Step C: Install mamba-ssm + causal-conv1d"
echo "============================================"

# CRITICAL: mamba-ssm requires:
#   - Linux (NOT Windows, WSL is fine)
#   - NVIDIA GPU with CUDA
#   - PyTorch with CUDA support
#   - Python 3.8-3.11

# Install causal-conv1d first (dependency)
pip install causal-conv1d>=1.0.0

# Install mamba-ssm
pip install mamba-ssm>=1.0.1

# Verify installation
python -c "
from mamba_ssm import Mamba
import torch
x = torch.randn(1, 64, 384).cuda()
model = Mamba(d_model=384, d_state=16, d_conv=4, expand=2).cuda()
y = model(x)
print(f'Input:  {x.shape}')
print(f'Output: {y.shape}')
print('✅ Real Mamba SSM working!')
"

echo "============================================"
echo "  Step D: Install other dependencies"
echo "============================================"

pip install timm einops scikit-learn matplotlib tqdm soundfile pyyaml

echo "============================================"
echo "  Step E: Run the experiments"
echo "============================================"

cd ~/ICBHIMamba-JBHI

# Train all 8 models (including real Mamba)
python scripts/step3_train_all_models.py

# Or train just the real Mamba to compare
python scripts/step3_train_all_models.py --models mamba_real

# Or train real Mamba + GRU Mamba to compare both
python scripts/step3_train_all_models.py --models mamba mamba_real

echo "============================================"
echo "  Step F: Generate figures with 8 models"
echo "============================================"

python scripts/step4_generate_figures.py
python scripts/step6_generate_latex.py

echo "============================================"
echo "  DONE!"
echo "============================================"
