#!/bin/bash
# =============================================================================
# ICBHIMambaNet - Environment Setup
# ICBHI 2017 Respiratory Sound Classification
# =============================================================================

set -e

echo "============================================"
echo "  ICBHIMambaNet Environment Setup"
echo "============================================"

# 1. Create conda environment
conda create -n icbhi_mamba python=3.9 -y
conda activate icbhi_mamba

# 2. Install PyTorch (choose ONE)
# -- GPU (CUDA 11.8) --
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# -- GPU (CUDA 12.1) --
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# -- CPU only --
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
print('Setup complete!')
"
