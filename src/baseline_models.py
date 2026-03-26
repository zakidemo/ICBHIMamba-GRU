"""
Baseline Models for ICBHI Comparative Study
=============================================
All models use IDENTICAL input conditions:
  - Same mel-spectrogram [512 × 128]
  - Same train/test split
  - Same normalization
  - Same augmentation pipeline
  - Same loss function (FocalLoss + class weights)
  - Same optimizer (AdamW + cosine LR)

Models implemented:
  1. MFCCBaseline     – Hand-crafted MFCC features + MLP (traditional approach)
  2. CNN2D            – Custom 2D CNN on mel-spectrogram
  3. AlexNet1D        – AlexNet-inspired adapted for spectrograms
  4. VGG16Transfer    – VGG16 with ImageNet pretrained weights (transfer learning)
  5. ResNet50Transfer – ResNet50 with ImageNet pretrained weights
  6. ViTSmall         – Vision Transformer (ViT-Small, scratch)
  7. ICBHIMambaNet    – Our proposed SSM/Mamba model (from icbhi_model.py)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MFCC Baseline  (hand-crafted features + MLP)
# ─────────────────────────────────────────────────────────────────────────────

class MFCCBaseline(nn.Module):
    """
    Traditional hand-crafted approach:
      - 40 MFCC coefficients + delta + delta-delta = 120 features
      - Statistical pooling (mean, std, max, min) over time → 480-dim vector
      - 3-layer MLP classifier

    This represents the pre-deep-learning standard in respiratory sound analysis.
    """
    def __init__(self, num_classes=4, n_mfcc=40, hidden_dim=512, dropout=0.4):
        super().__init__()
        # Input: 120 features (mfcc + delta + delta2) × 4 stats = 480
        input_dim = n_mfcc * 3 * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, num_classes),
        )
        self.n_mfcc = n_mfcc

    def extract_mfcc_features(self, fbank):
        """
        Extract MFCC features from mel-filterbank using DCT.
        fbank: [B, T, F] mel-spectrogram (already computed)
        Returns: [B, n_mfcc*3*4] hand-crafted feature vector
        """
        B, T, F = fbank.shape
        # Apply DCT to get real MFCC coefficients from mel-filterbank
        # DCT-II: standard MFCC extraction from log-mel spectrum
        log_fbank = torch.log(torch.clamp(fbank, min=1e-8))
        # Create DCT matrix
        n = torch.arange(F, dtype=fbank.dtype, device=fbank.device)
        k = torch.arange(self.n_mfcc, dtype=fbank.dtype, device=fbank.device)
        dct_matrix = torch.cos(np.pi * k.unsqueeze(1) * (2*n.unsqueeze(0)+1) / (2*F))
        dct_matrix = dct_matrix * np.sqrt(2.0 / F)
        # Apply DCT: [B, T, F] x [n_mfcc, F]^T -> [B, T, n_mfcc]
        mfcc = torch.matmul(log_fbank, dct_matrix.T)  # [B, T, n_mfcc]

        # Compute deltas (first-order difference)
        delta  = torch.zeros_like(mfcc)
        delta[:, 1:-1, :] = mfcc[:, 2:, :] - mfcc[:, :-2, :]
        delta2 = torch.zeros_like(mfcc)
        delta2[:, 1:-1, :] = delta[:, 2:, :] - delta[:, :-2, :]

        # Stack: [B, T, n_mfcc*3]
        features = torch.cat([mfcc, delta, delta2], dim=2)

        # Statistical pooling over time: mean, std, max, min → [B, n_mfcc*3*4]
        mean = features.mean(dim=1)
        std  = features.std(dim=1)
        mx   = features.max(dim=1).values
        mn   = features.min(dim=1).values
        out  = torch.cat([mean, std, mx, mn], dim=1)   # [B, n_mfcc*3*4]
        return out

    def forward(self, x):
        # x: [B, T, F] mel-spectrogram
        features = self.extract_mfcc_features(x)
        return self.mlp(features)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom 2D CNN
# ─────────────────────────────────────────────────────────────────────────────

class CNN2D(nn.Module):
    """
    Custom 2D CNN treating mel-spectrogram as a grayscale image.
    5 convolutional blocks with progressive downsampling.
    """
    def __init__(self, num_classes=4, dropout=0.4):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1,  32),    # [B,  32, 256, 64]
            conv_block(32, 64),    # [B,  64, 128, 32]
            conv_block(64, 128),   # [B, 128,  64, 16]
            conv_block(128, 256),  # [B, 256,  32,  8]
            conv_block(256, 512),  # [B, 512,  16,  4]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, T, F] → [B, 1, T, F]
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. AlexNet (adapted for spectrograms)
# ─────────────────────────────────────────────────────────────────────────────

class AlexNetSpectrogram(nn.Module):
    """
    AlexNet architecture adapted for mel-spectrogram input.
    Original AlexNet (Krizhevsky 2012) with modified kernel sizes
    for time-frequency feature extraction.
    """
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: large kernel to capture broad time-frequency patterns
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3-5: smaller kernels for fine-grained features
            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 16, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)     # [B, 1, T, F]
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. VGG16 (transfer learning from ImageNet)
# ─────────────────────────────────────────────────────────────────────────────

class VGG16Transfer(nn.Module):
    """
    VGG16 with ImageNet pretrained weights, adapted for grayscale spectrogram.
    First conv layer modified to accept 1-channel input.
    Classifier head replaced for 4-class ICBHI.
    """
    def __init__(self, num_classes=4, dropout=0.5, pretrained=True):
        super().__init__()
        import torchvision.models as models

        vgg = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)

        # Adapt first conv: 3-channel → 1-channel
        # Average the 3-channel weights to initialise 1-channel
        orig_weight = vgg.features[0].weight.data   # [64, 3, 3, 3]
        new_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        new_conv.weight.data = orig_weight.mean(dim=1, keepdim=True)
        vgg.features[0] = new_conv

        self.features    = vgg.features
        self.avgpool     = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)      # [B, 1, T, F]
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ResNet50 (transfer learning from ImageNet)
# ─────────────────────────────────────────────────────────────────────────────

class ResNet50Transfer(nn.Module):
    """
    ResNet50 with ImageNet pretrained weights.
    First conv adapted for 1-channel spectrogram input.
    """
    def __init__(self, num_classes=4, dropout=0.4, pretrained=True):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)

        # Adapt first conv: 3-channel → 1-channel
        orig_w = resnet.conv1.weight.data   # [64, 3, 7, 7]
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                             bias=False)
        new_conv.weight.data = orig_w.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

        # Replace final FC
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

        self.model = resnet

    def forward(self, x):
        x = x.unsqueeze(1)     # [B, 1, T, F]
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Vision Transformer (ViT-Small, from scratch)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv     = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj    = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x), attn


class ViTBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim    = int(d_model * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x, attn_weights


class ViTSmall(nn.Module):
    """
    Vision Transformer (ViT-Small) trained from scratch on mel-spectrograms.
    d_model=384, n_heads=6, depth=12 — matching ViT-Small paper config.
    Patch size 16×16, same as our Mamba model for fair comparison.
    """
    def __init__(self, num_classes=4, fbank_size=(512, 128),
                 patch_size=(16, 16), d_model=384, n_heads=6,
                 depth=12, dropout=0.1):
        super().__init__()
        T, F   = fbank_size
        pt, pf = patch_size
        self.n_patches  = (T // pt) * (F // pf)
        self.patch_size = patch_size
        self.d_model    = d_model

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_drop = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList([
            ViTBlock(d_model, n_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attn=False):
        # x: [B, T, F]
        B = x.shape[0]
        x = x.unsqueeze(1)                              # [B,1,T,F]
        x = self.patch_embed(x).transpose(1, 2)        # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)               # [B, N+1, D]
        x   = self.pos_drop(x + self.pos_embed)

        attn_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attn_weights.append(attn)

        x = self.norm(x)
        cls_out = x[:, 0]                              # CLS token
        logits  = self.head(cls_out)

        if return_attn:
            return logits, attn_weights
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def get_model(name: str, num_classes: int = 4,
              fbank_size: tuple = (512, 128)) -> nn.Module:
    """
    Factory function — returns model by name.

    Names: 'mfcc_mlp', 'cnn2d', 'alexnet', 'vgg16',
           'resnet50', 'vit_small', 'mamba', 'mamba_real'
    """
    name = name.lower()

    if name == "mfcc_mlp":
        return MFCCBaseline(num_classes=num_classes)

    elif name == "cnn2d":
        return CNN2D(num_classes=num_classes)

    elif name == "alexnet":
        return AlexNetSpectrogram(num_classes=num_classes)

    elif name == "vgg16":
        return VGG16Transfer(num_classes=num_classes, pretrained=True)

    elif name == "resnet50":
        return ResNet50Transfer(num_classes=num_classes, pretrained=True)

    elif name == "vit_small":
        return ViTSmall(num_classes=num_classes, fbank_size=fbank_size)

    elif name == "mamba":
        from src.icbhi_model import ICBHIMambaNet
        return ICBHIMambaNet(
            num_classes=num_classes,
            d_model=384, n_layers=12,
            fbank_size=fbank_size,
            patch_size=(16, 16),
            dropout=0.3,
        )

    elif name == "mamba_real":
        from src.real_mamba_model import RealMambaNet, check_mamba_available
        if not check_mamba_available():
            raise RuntimeError(
                "mamba-ssm not installed. Install with:\n"
                "  pip install mamba-ssm causal-conv1d\n"
                "Requires: Linux + NVIDIA GPU + CUDA"
            )
        return RealMambaNet(
            num_classes=num_classes,
            d_model=384, d_state=16, d_conv=4, expand=2,
            n_layers=12,
            fbank_size=fbank_size,
            patch_size=(16, 16),
            dropout=0.2,
        )

    else:
        raise ValueError(f"Unknown model: {name}. "
                         f"Choose from: mfcc_mlp, cnn2d, alexnet, "
                         f"vgg16, resnet50, vit_small, mamba, mamba_real")


def count_parameters(model: nn.Module) -> float:
    """Return total trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad) / 1e6
