"""
Real Mamba SSM Model for ICBHI Classification
==============================================
Uses the actual mamba-ssm CUDA selective scan kernel (S6),
NOT a GRU approximation. Requires: Linux + NVIDIA GPU + mamba-ssm.

Architecture:
  Mel-Spectrogram → Patch Embed → N × Real Mamba Block → Classify

Each Mamba block uses input-dependent B, C, Δ parameters
with hardware-efficient selective scan — the core innovation
of Gu & Dao (2023).

Install: pip install mamba-ssm causal-conv1d
"""

import torch
import torch.nn as nn


def check_mamba_available():
    """Check if mamba-ssm is installed."""
    try:
        from mamba_ssm import Mamba
        return True
    except ImportError:
        return False


class RealMambaBlock(nn.Module):
    """
    A Mamba block using the REAL selective state space mechanism.
    Uses mamba_ssm.Mamba which implements:
      - Input-dependent B_k, C_k, Δ_k
      - Hardware-efficient selective scan (S6 kernel)
      - Linear-time sequence processing
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        from mamba_ssm import Mamba

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return residual + self.drop(x)


class RealMambaNet(nn.Module):
    """
    ICBHIMambaNet with REAL Mamba selective scan blocks.

    This is the genuine SSM architecture — not a GRU approximation.
    Uses the same patch embedding and classification head as
    ICBHIMambaNet (GRU), making the comparison fair: the ONLY
    difference is GRU vs real selective scan inside each block.

    Architecture:
        Mel-spectrogram [B, 512, 128]
            → Patch Embed (16×16) → [B, 256, d_model]
            → + Positional Embedding
            → N × RealMambaBlock (selective scan S6)
            → LayerNorm → Global Average Pool
            → Linear → 4-class logits
    """

    def __init__(self,
                 num_classes: int = 4,
                 d_model: int = 384,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 n_layers: int = 12,
                 fbank_size: tuple = (512, 128),
                 patch_size: tuple = (16, 16),
                 dropout: float = 0.2):
        super().__init__()

        T, F = fbank_size
        pt, pf = patch_size
        n_patches = (T // pt) * (F // pf)

        # Patch embedding (identical to GRU version)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        self.n_patches = n_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Real Mamba blocks with selective scan
        self.blocks = nn.Sequential(*[
            RealMambaBlock(d_model, d_state=d_state, d_conv=d_conv,
                           expand=expand, dropout=dropout * 0.5)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, T, F]
        x = x.unsqueeze(1)                  # [B, 1, T, F]
        x = self.patch_embed(x)             # [B, d_model, N]
        x = x.transpose(1, 2)              # [B, N, d_model]
        x = x + self.pos_embed
        x = self.drop(x)

        x = self.blocks(x)                 # [B, N, d_model]

        x = self.norm(x)
        x = x.mean(dim=1)                  # global average pool
        x = self.head(x)
        return x
