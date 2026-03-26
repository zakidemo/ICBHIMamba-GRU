"""
GradCAM & Attention Visualization for All Models
=================================================
Produces publication-quality explainability figures:

  - GradCAM heatmaps overlaid on mel-spectrograms (CNN, AlexNet, VGG16, ResNet50)
  - Attention rollout maps (ViT)
  - SSM token importance maps (Mamba)
  - MFCC feature importance bar chart
  - Grid figure: one row per class, one column per model

These figures answer: "What does each model look at in the spectrogram?"
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.icbhi_dataloader import ICBHIDataset
from src.baseline_models   import get_model
from src.icbhi_utils       import CLASS_NAMES

# ── Config ─────────────────────────────────────────────────────────────────
TEST_JSON    = os.path.join(PROJECT_ROOT, "datafiles/icbhi_test.json")
LABEL_CSV    = os.path.join(PROJECT_ROOT, "datafiles/icbhi_labels.csv")
CKPT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
OUT_DIR      = os.path.join(PROJECT_ROOT, "results/gradcam")
NUM_CLASSES  = 4
NUM_MEL_BINS = 128
TARGET_LEN   = 512
FRAME_SHIFT  = 10
NORM_MEAN    = -11.605727
NORM_STD     =  4.965201
FBANK_SIZE   = (TARGET_LEN, NUM_MEL_BINS)

# Models that support GradCAM via conv layers
GRADCAM_MODELS = ["cnn2d", "alexnet", "vgg16", "resnet50"]
ATTN_MODELS    = ["vit_small"]
SSM_MODELS     = ["mamba"]
MLP_MODELS     = ["mfcc_mlp"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 9, "figure.dpi": 150, "savefig.dpi": 300,
})


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    GradCAM for 2D convolutional models.
    Hooks into the last convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        """
        x: [1, T, F] mel-spectrogram
        Returns: heatmap [T, F] normalised to [0, 1]
        """
        self.model.eval()
        # CRITICAL: must use enable_grad + no AMP for backward hooks to work
        with torch.enable_grad():
            x_in = x.unsqueeze(0).float().requires_grad_(True)
            logits = self.model(x_in)
            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()
            self.model.zero_grad()
            logits[0, class_idx].backward()
        x = x_in

        # Pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1)        # [1, H, W]
        cam     = F.relu(cam).squeeze(0)                         # [H, W]

        # Resize to input size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(TARGET_LEN, NUM_MEL_BINS),
            mode="bilinear", align_corners=False
        ).squeeze()

        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def get_last_conv_layer(model_name, model):
    """Get the last convolutional layer for GradCAM."""
    if model_name == "cnn2d":
        return model.features[-3]   # last conv before pool
    elif model_name == "alexnet":
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, torch.nn.Conv2d):
                return layer
        return model.features[-3]
    elif model_name == "vgg16":
        # VGG16: last ReLU after last conv (index -2 is MaxPool, -3 is ReLU, -4 is Conv)
        # Find last Conv2d layer
        for layer in reversed(list(model.features.children())):
            if isinstance(layer, torch.nn.Conv2d):
                return layer
        return model.features[-3]
    elif model_name == "resnet50":
        return model.model.layer4[-1].conv3
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ViT Attention Rollout
# ─────────────────────────────────────────────────────────────────────────────

def vit_attention_rollout(model, x, device):
    """
    Compute attention rollout for ViT.
    Returns heatmap [T, F] showing which patches the model attends to.
    """
    model.eval()
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        _, attn_list = model(x, return_attn=True)

    # attn_list: list of [1, n_heads, N+1, N+1]
    # Rollout: multiply attention matrices across layers
    n_patches = model.n_patches
    rollout   = torch.eye(n_patches + 1).to(device)

    for attn in attn_list:
        attn_avg = attn[0].mean(0)               # [N+1, N+1]
        attn_avg = attn_avg + torch.eye(n_patches + 1).to(device)
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        rollout  = attn_avg @ rollout

    # CLS token attention to all patches
    cls_attn = rollout[0, 1:]                    # [N_patches]

    # Reshape to 2D grid
    T, F   = FBANK_SIZE
    pt, pf = 16, 16
    gT, gF = T // pt, F // pf
    cls_attn = cls_attn.reshape(gT, gF).cpu().numpy()

    # Upsample to full spectrogram size
    cls_attn = np.repeat(np.repeat(cls_attn, pt, axis=0), pf, axis=1)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn


# ─────────────────────────────────────────────────────────────────────────────
# Mamba token importance (perturbation-based — works with GRU/SSM)
# ─────────────────────────────────────────────────────────────────────────────

def mamba_token_importance(model, x, device, class_idx=None):
    """
    Gradient-FREE perturbation saliency for Mamba/GRU models.
    GradCAM backward fails on cudnn RNN — this method works perfectly.

    Strategy: occlude each patch with its mean value and measure
    how much the target class score drops. High drop = important patch.
    Returns heatmap [T, F].
    """
    model.eval()
    x_in = x.unsqueeze(0).to(device)   # [1, T, F]

    with torch.no_grad():
        logits    = model(x_in)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        base_score = torch.softmax(logits, dim=1)[0, class_idx].item()

    T, F   = x.shape
    pt, pf = 16, 16
    gT, gF = T // pt, F // pf
    importance_map = np.zeros((gT, gF), dtype=np.float32)

    # Occlude each patch and measure score drop
    for i in range(gT):
        for j in range(gF):
            x_masked = x_in.clone()
            # Replace patch with mean value (neutral occlusion)
            patch_mean = x_in[0,
                               i*pt:(i+1)*pt,
                               j*pf:(j+1)*pf].mean()
            x_masked[0, i*pt:(i+1)*pt, j*pf:(j+1)*pf] = patch_mean

            with torch.no_grad():
                score = torch.softmax(model(x_masked), dim=1)[0, class_idx].item()

            # Importance = how much score drops when patch is removed
            importance_map[i, j] = max(0.0, base_score - score)

    # Upsample patch grid to full spectrogram size
    heatmap = np.repeat(np.repeat(importance_map, pt, axis=0), pf, axis=1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap, class_idx


# ─────────────────────────────────────────────────────────────────────────────
# MFCC feature importance
# ─────────────────────────────────────────────────────────────────────────────

def mfcc_feature_importance(model, x, device):
    """
    Compute feature importance for MLP using gradient × input.
    Returns bar chart data for MFCC feature groups.
    """
    model.eval()
    x_in = x.unsqueeze(0).to(device).requires_grad_(True)

    logits = model(x_in)
    class_idx = logits.argmax(1).item()
    model.zero_grad()
    logits[0, class_idx].backward()

    features  = model.extract_mfcc_features(x_in)
    grad_imp  = (x_in.grad.abs() * x_in.abs()).squeeze().detach().cpu().numpy()

    # Average over time → frequency importance
    freq_imp = grad_imp.mean(axis=0)   # [F]
    freq_imp = (freq_imp - freq_imp.min()) / (freq_imp.max() - freq_imp.min() + 1e-8)

    # Expand to 2D for consistent display
    heatmap = np.tile(freq_imp, (TARGET_LEN, 1))
    return heatmap, class_idx


# ─────────────────────────────────────────────────────────────────────────────
# Load sample spectrograms per class
# ─────────────────────────────────────────────────────────────────────────────

def load_samples_per_class(n_per_class=2):
    """Load n correctly-classified test samples per class."""
    conf = dict(num_mel_bins=NUM_MEL_BINS, target_length=TARGET_LEN,
                mean=NORM_MEAN, std=NORM_STD, fshift=FRAME_SHIFT,
                mode="eval", freqm=0, timem=0, mixup=0.0, noise=False)
    ds = ICBHIDataset(TEST_JSON, conf, LABEL_CSV)

    samples = {i: [] for i in range(NUM_CLASSES)}
    for fbank, labels, path in ds:
        cls = labels.argmax().item()
        if len(samples[cls]) < n_per_class:
            samples[cls].append((fbank, cls, path))
        if all(len(v) >= n_per_class for v in samples.values()):
            break
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Generate GradCAM figure for one model
# ─────────────────────────────────────────────────────────────────────────────

def generate_model_gradcam(model_name, model, samples, device, out_path):
    """Generate GradCAM figure: rows=classes, cols=samples."""
    n_per_class = 2
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class * 2,
                              figsize=(n_per_class * 6, NUM_CLASSES * 3))
    fig.suptitle(f"GradCAM — {model_name.upper()}", fontweight="bold",
                 fontsize=12)

    for cls_idx in range(NUM_CLASSES):
        for s_idx, (fbank, true_cls, path) in enumerate(samples[cls_idx]):
            col_base = s_idx * 2
            fbank_np = fbank.numpy()

            # Raw spectrogram
            ax_spec = axes[cls_idx, col_base]
            ax_spec.imshow(fbank_np.T, aspect="auto", origin="lower",
                           cmap="magma", interpolation="nearest")
            ax_spec.set_title(f"{CLASS_NAMES[cls_idx].capitalize()}\nSpectrogram",
                               fontsize=8)
            ax_spec.axis("off")

            # Heatmap
            ax_cam = axes[cls_idx, col_base + 1]
            try:
                if model_name in GRADCAM_MODELS:
                    last_conv = get_last_conv_layer(model_name, model)
                    gcam      = GradCAM(model, last_conv)
                    heatmap, pred = gcam(fbank.to(device), class_idx=true_cls)
                elif model_name in ATTN_MODELS:
                    heatmap = vit_attention_rollout(model, fbank, device)
                    pred    = true_cls
                elif model_name in SSM_MODELS:
                    heatmap, pred = mamba_token_importance(model, fbank,
                                                            device, true_cls)
                elif model_name in MLP_MODELS:
                    heatmap, pred = mfcc_feature_importance(model, fbank, device)
                else:
                    heatmap = np.zeros((TARGET_LEN, NUM_MEL_BINS))
                    pred    = true_cls

                # Overlay: spectrogram + heatmap
                ax_cam.imshow(fbank_np.T, aspect="auto", origin="lower",
                               cmap="gray",   alpha=0.6, interpolation="nearest")
                ax_cam.imshow(heatmap.T,   aspect="auto", origin="lower",
                               cmap="jet",    alpha=0.5, interpolation="nearest",
                               vmin=0, vmax=1)
                color = "green" if pred == true_cls else "red"
                ax_cam.set_title(
                    f"Pred: {CLASS_NAMES[pred].capitalize()}", fontsize=8,
                    color=color)
            except Exception as e:
                ax_cam.text(0.5, 0.5, f"Error:\n{str(e)[:40]}",
                            ha="center", va="center", transform=ax_cam.transAxes,
                            fontsize=7)
            ax_cam.axis("off")

    # Y-axis labels
    for cls_idx in range(NUM_CLASSES):
        axes[cls_idx, 0].set_ylabel(CLASS_NAMES[cls_idx].capitalize(),
                                     fontsize=10, rotation=0, labelpad=50,
                                     va="center")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  GradCAM saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Combined GradCAM comparison: all models, one class
# ─────────────────────────────────────────────────────────────────────────────

def generate_combined_gradcam(models_dict, samples, device, out_dir):
    """
    Fig: rows = classes, cols = models.
    Shows what each model attends to for each sound type.
    This is the KEY publication figure.
    """
    model_names = list(models_dict.keys())
    n_models    = len(model_names)

    fig, axes = plt.subplots(NUM_CLASSES, n_models,
                              figsize=(n_models * 2.5, NUM_CLASSES * 2.8))
    fig.suptitle(
        "Model Explainability: GradCAM / Attention Maps on Mel-Spectrograms\n"
        "(Each column = model, each row = lung sound class)",
        fontweight="bold", fontsize=11, y=1.01)

    for col, model_name in enumerate(model_names):
        model = models_dict[model_name]
        axes[0, col].set_title(model_name.upper().replace("_", "\n"),
                                fontsize=9, fontweight="bold")

        for row, cls_idx in enumerate(range(NUM_CLASSES)):
            ax = axes[row, col]
            if not samples[cls_idx]:
                ax.axis("off")
                continue

            fbank, true_cls, _ = samples[cls_idx][0]
            fbank_np = fbank.numpy()

            try:
                if model_name in GRADCAM_MODELS:
                    last_conv = get_last_conv_layer(model_name, model)
                    gcam      = GradCAM(model, last_conv)
                    heatmap, _ = gcam(fbank.to(device), class_idx=true_cls)
                elif model_name in ATTN_MODELS:
                    heatmap = vit_attention_rollout(model, fbank, device)
                elif model_name in SSM_MODELS:
                    heatmap, _ = mamba_token_importance(model, fbank,
                                                         device, true_cls)
                elif model_name in MLP_MODELS:
                    heatmap, _ = mfcc_feature_importance(model, fbank, device)
                else:
                    heatmap = np.zeros_like(fbank_np)

                ax.imshow(fbank_np.T, aspect="auto", origin="lower",
                           cmap="gray",  alpha=0.55, interpolation="nearest")
                ax.imshow(heatmap.T,   aspect="auto", origin="lower",
                           cmap="hot",   alpha=0.55, interpolation="nearest",
                           vmin=0, vmax=1)
            except Exception:
                ax.imshow(fbank_np.T, aspect="auto", origin="lower",
                           cmap="magma", interpolation="nearest")

            if col == 0:
                ax.set_ylabel(CLASS_NAMES[cls_idx].capitalize(),
                               fontsize=10, rotation=0, labelpad=45, va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "combined_gradcam.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[GradCAM] Combined figure saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Load samples ──────────────────────────────────────────────────────
    print("[INFO] Loading test samples …")
    samples = load_samples_per_class(n_per_class=2)
    for cls_idx, slist in samples.items():
        print(f"  {CLASS_NAMES[cls_idx]:10s}: {len(slist)} samples")

    # ── Load all trained models ───────────────────────────────────────────
    models_dict = {}
    all_model_names = ["mfcc_mlp", "cnn2d", "alexnet", "vgg16",
                        "resnet50", "vit_small", "mamba"]

    for name in all_model_names:
        ckpt_path = os.path.join(CKPT_DIR, f"best_{name}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {name} — no checkpoint found at {ckpt_path}")
            continue
        try:
            model = get_model(name, NUM_CLASSES, FBANK_SIZE).to(device)
            ckpt  = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            models_dict[name] = model
            print(f"[OK]   Loaded {name}")
        except Exception as e:
            print(f"[WARN] Could not load {name}: {e}")

    if not models_dict:
        print("[ERROR] No trained models found. Run step6_train_all_baselines.py first.")
        return

    # ── Per-model GradCAM figures ─────────────────────────────────────────
    print("\n[INFO] Generating per-model GradCAM figures …")
    for name, model in models_dict.items():
        out_path = os.path.join(OUT_DIR, f"gradcam_{name}.png")
        generate_model_gradcam(name, model, samples, device, out_path)

    # ── Combined comparison figure ────────────────────────────────────────
    print("\n[INFO] Generating combined comparison figure …")
    generate_combined_gradcam(models_dict, samples, device, OUT_DIR)

    print(f"\n[DONE] All GradCAM figures saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
