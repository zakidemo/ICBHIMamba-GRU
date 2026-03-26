"""
ICBHI AuM Model
===============
Wraps the Audio-Mamba (AuM) backbone for 4-class lung sound classification
(normal / crackle / wheeze / both).

Also provides a lightweight CNN-Mamba hybrid for comparison and for users
who cannot install the full mamba-ssm CUDA extension.
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# AuM backbone wrapper
# ─────────────────────────────────────────────────────────────────────────────

def get_aum_model(num_classes: int = 4,
                  model_size:  str = "small",
                  imagenet_pretrained: bool = False):
    """
    Build an Audio-Mamba (AuM) model for ICBHI.

    Parameters
    ----------
    num_classes : int
        Number of output classes (4 for ICBHI).
    model_size : str
        'tiny' | 'small' | 'base'
    imagenet_pretrained : bool
        Load ImageNet-pretrained weights (requires network access or
        local checkpoint).

    Returns
    -------
    nn.Module
    """
    # Add project root to sys.path so imports inside AuM work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vim_path = os.path.join(project_root, "vim_mamba_ssm")
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if vim_path not in sys.path:
        sys.path.insert(0, vim_path)

    from src.models.mamba_models import VisionMamba

    # Model configs mirroring AuM paper
    configs = {
        "tiny":  dict(patch_size=16, embed_dim=192,  depth=24, rms_norm=True,
                      residual_in_fp32=True, fused_add_norm=True,
                      if_abs_pos_embed=True, if_cls_token=True,
                      if_bidirectional=False, use_middle_cls_token=True),
        "small": dict(patch_size=16, embed_dim=384,  depth=24, rms_norm=True,
                      residual_in_fp32=True, fused_add_norm=True,
                      if_abs_pos_embed=True, if_cls_token=True,
                      if_bidirectional=False, use_middle_cls_token=True),
        "base":  dict(patch_size=16, embed_dim=768,  depth=24, rms_norm=True,
                      residual_in_fp32=True, fused_add_norm=True,
                      if_abs_pos_embed=True, if_cls_token=True,
                      if_bidirectional=False, use_middle_cls_token=True),
    }

    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")

    cfg = configs[model_size]
    model = VisionMamba(
        num_classes=num_classes,
        **cfg
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight CNN + Mamba-like SSM  (pure-PyTorch, no CUDA extension needed)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleMambaBlock(nn.Module):
    """
    A simplified SSM-inspired block implemented in plain PyTorch.
    Uses a 1-D convolution + GRU as a recurrent sequence mixer,
    capturing the spirit of Mamba without the selective-scan CUDA kernel.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, d_inner * 2)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1,
                                  groups=d_inner)
        self.act      = nn.SiLU()
        self.ssm      = nn.GRU(d_inner, d_state, batch_first=True)
        self.ssm_proj = nn.Linear(d_state, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.drop     = nn.Dropout(0.1)

    def forward(self, x):
        # x : [B, L, D]
        B, L, D = x.shape
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)                # [B, L, 2*d_inner]
        x_, z = xz.chunk(2, dim=-1)         # each [B, L, d_inner]
        x_ = x_.transpose(1, 2)             # [B, d_inner, L]
        x_ = self.conv1d(x_)
        x_ = x_.transpose(1, 2)             # [B, L, d_inner]
        x_ = self.act(x_)
        y, _ = self.ssm(x_)                 # [B, L, d_state]
        y  = self.ssm_proj(y)               # [B, L, d_inner]
        y  = y * self.act(z)
        y  = self.out_proj(y)               # [B, L, D]
        return residual + self.drop(y)


class PatchEmbedAudio(nn.Module):
    """Split a 2-D mel-spectrogram into patches and project to d_model."""

    def __init__(self, fbank_size=(512, 128), patch_size=(16, 16), d_model=256):
        super().__init__()
        T, F   = fbank_size
        pt, pf = patch_size
        self.n_patches = (T // pt) * (F // pf)
        self.proj = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),          # [B, d_model, N]
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.unsqueeze(1)              # [B, 1, T, F]
        x = self.proj(x)               # [B, d_model, N]
        x = x.transpose(1, 2)          # [B, N, d_model]
        return x


class ICBHIMambaNet(nn.Module):
    """
    Pure-PyTorch Mamba-inspired model for ICBHI 4-class classification.

    Architecture:
        mel-spectrogram → patch embed → N × SimpleMambaBlock → classify
    """

    def __init__(self,
                 num_classes:  int   = 4,
                 d_model:      int   = 256,
                 n_layers:     int   = 8,
                 fbank_size:   tuple = (512, 128),
                 patch_size:   tuple = (16, 16),
                 dropout:      float = 0.2):
        super().__init__()

        self.patch_embed = PatchEmbedAudio(fbank_size, patch_size, d_model)
        n_patches = self.patch_embed.n_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            SimpleMambaBlock(d_model) for _ in range(n_layers)
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
        x = self.patch_embed(x)         # [B, N, D]
        x = x + self.pos_embed
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)              # global average pool over patches
        x = self.head(x)
        return x
