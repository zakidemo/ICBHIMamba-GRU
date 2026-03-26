"""
Step 4 – Generate Publication Figures (IEEE JBHI Ready)
========================================================
Generates all figures from the comparative study results:

  Fig 1 – Confusion matrices (all models, normalized)
  Fig 2 – ICBHI score comparison bar chart
  Fig 3 – Per-class recall comparison
  Fig 4 – Training convergence curves
  Fig 5 – Per-class ROC curves (all models)

Usage:
    python scripts/step4_generate_figures.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.icbhi_dataloader import ICBHIDataset
from src.baseline_models  import get_model
from src.icbhi_utils      import CLASS_NAMES

# ── Config ────────────────────────────────────────────────────────────────
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results/comparative_study")
CKPT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
OUT_DIR      = os.path.join(PROJECT_ROOT, "figures")
TEST_JSON    = os.path.join(PROJECT_ROOT, "datafiles/icbhi_test.json")
LABEL_CSV    = os.path.join(PROJECT_ROOT, "datafiles/icbhi_labels.csv")
NUM_CLASSES  = 4
NUM_MEL_BINS = 128
TARGET_LEN   = 512
FRAME_SHIFT  = 10
NORM_MEAN    = -11.605727
NORM_STD     =  4.965201
BATCH_SIZE   = 16
NUM_WORKERS  = 4
FBANK_SIZE   = (TARGET_LEN, NUM_MEL_BINS)

MODEL_DISPLAY = {
    "mfcc_mlp":    "Mel-FB + MLP",
    "cnn2d":       "CNN-2D",
    "alexnet":     "AlexNet",
    "vgg16":       "VGG-16",
    "resnet50":    "ResNet-50",
    "vit_small":   "ViT-Small",
    "mamba":       "ICBHIMamba-GRU (Ours)",
    "mamba_real":  "ICBHIMamba-SSM",
}

COLORS = {
    "mfcc_mlp":    "#9E9E9E",
    "cnn2d":       "#2196F3",
    "alexnet":     "#4CAF50",
    "vgg16":       "#FF9800",
    "resnet50":    "#F44336",
    "vit_small":   "#9C27B0",
    "mamba":       "#795548",
    "mamba_real":  "#E91E63",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_all_results():
    path = os.path.join(RESULTS_DIR, "all_results.json")
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run step3 first.")
        return None
    with open(path) as f:
        results = json.load(f)
    for r in results:
        hist_path = os.path.join(RESULTS_DIR, r["model"], "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                r["history"] = json.load(f)
        else:
            r["history"] = None
    return results


# ── Fig 1: Confusion Matrices ────────────────────────────────────────────────
def fig_confusion_matrices(results, out_path):
    n = len(results)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax  = axes[i]
        cm  = np.array(r["confusion_matrix"], dtype=float)
        cm_n = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
        im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(MODEL_DISPLAY.get(r["model"], r["model"]),
                     fontweight="bold", fontsize=10)
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels([c[:3].upper() for c in CLASS_NAMES], fontsize=8)
        ax.set_yticklabels([c[:3].upper() for c in CLASS_NAMES], fontsize=8)
        for ii in range(NUM_CLASSES):
            for jj in range(NUM_CLASSES):
                col = "white" if cm_n[ii, jj] > 0.5 else "black"
                ax.text(jj, ii, f"{cm_n[ii,jj]:.2f}",
                        ha="center", va="center", color=col, fontsize=8)
        ax.set_xlabel(
            f"Predicted\nICBHI={r['icbhi_score']*100:.1f}%", fontsize=8)
        ax.set_ylabel("True", fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Confusion Matrices — All Models (Normalised by True Class)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Fig 1] Saved: {out_path}")


# ── Fig 2: Main Comparison Bar Chart ─────────────────────────────────────────
def fig_main_comparison(results, out_path):
    results_sorted = sorted(results, key=lambda x: x["icbhi_score"])
    names  = [MODEL_DISPLAY.get(r["model"], r["model"]) for r in results_sorted]
    icbhi  = [r["icbhi_score"]  * 100 for r in results_sorted]
    se     = [r["sensitivity"]  * 100 for r in results_sorted]
    sp     = [r["specificity"]  * 100 for r in results_sorted]
    acc    = [r["accuracy"]     * 100 for r in results_sorted]

    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 1.5*w, icbhi, w, label="ICBHI Score", color="#9C27B0",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x - 0.5*w, se,    w, label="Sensitivity", color="#FF9800",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + 0.5*w, sp,    w, label="Specificity", color="#4CAF50",
           edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + 1.5*w, acc,   w, label="Accuracy",    color="#2196F3",
           edgecolor="white", linewidth=0.5, zorder=3)

    # Annotate ICBHI bars
    for i_bar, val in enumerate(icbhi):
        ax.text(x[i_bar] - 1.5*w, val + 0.3, f"{val:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#9C27B0")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("ICBHI 2017 Classification — Comparative Results",
                 fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", zorder=0)

    # Highlight our model
    mamba_idx = next((i for i, r in enumerate(results_sorted)
                      if r["model"] == "mamba"), None)
    if mamba_idx is None:
        mamba_idx = next((i for i, r in enumerate(results_sorted)
                          if r["model"] == "mamba"), None)
    if mamba_idx is not None:
        ax.axvspan(mamba_idx - 0.5, mamba_idx + 0.5,
                   alpha=0.08, color="#E91E63", zorder=0)
        ax.text(mamba_idx, 2, "★ Proposed", ha="center",
                fontsize=8, color="#E91E63", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Fig 2] Saved: {out_path}")


# ── Fig 3: Per-Class Recall ──────────────────────────────────────────────────
def fig_per_class_recall(results, out_path):
    results_sorted = sorted(results, key=lambda x: -x["icbhi_score"])
    n = len(results_sorted)
    x = np.arange(NUM_CLASSES)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, r in enumerate(results_sorted):
        recall_key = "per_class_recall" if "per_class_recall" in r else "per_class_acc"
        recalls = [v * 100 for v in r[recall_key]]
        color   = COLORS.get(r["model"], "#607D8B")
        offset  = (i - n/2 + 0.5) * w
        ax.bar(x + offset, recalls, w,
               label=MODEL_DISPLAY.get(r["model"], r["model"]),
               color=color, edgecolor="white", linewidth=0.3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES], fontsize=12)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Per-Class Recall Comparison — All Models", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Fig 3] Saved: {out_path}")


# ── Fig 4: Training Convergence ──────────────────────────────────────────────
def fig_convergence(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for r in results:
        if r.get("history") is None:
            continue
        h     = r["history"]
        color = COLORS.get(r["model"], "#607D8B")
        label = MODEL_DISPLAY.get(r["model"], r["model"])
        ep    = range(1, len(h["icbhi_score"]) + 1)
        axes[0].plot(ep, [v*100 for v in h["icbhi_score"]],
                     color=color, lw=1.5, label=label)
        axes[1].plot(ep, [v*100 for v in h["sensitivity"]],
                     color=color, lw=1.5, linestyle="--")
        axes[1].plot(ep, [v*100 for v in h["specificity"]],
                     color=color, lw=1.5)

    axes[0].set_title("ICBHI Score During Training", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ICBHI Score (%)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_title("Sensitivity (--) & Specificity (—)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    plt.suptitle("Training Convergence — All Models", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Fig 4] Saved: {out_path}")


# ── Fig 5: ROC Curves ────────────────────────────────────────────────────────
@torch.no_grad()
def get_scores(model_name, device):
    ckpt_path = os.path.join(CKPT_DIR, f"best_{model_name}.pth")
    if not os.path.exists(ckpt_path):
        return None, None
    model = get_model(model_name, NUM_CLASSES, FBANK_SIZE).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    conf = dict(num_mel_bins=NUM_MEL_BINS, target_length=TARGET_LEN,
                mean=NORM_MEAN, std=NORM_STD, fshift=FRAME_SHIFT,
                mode="eval", freqm=0, timem=0, mixup=0.0, noise=False)
    loader = DataLoader(ICBHIDataset(TEST_JSON, conf, LABEL_CSV),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)
    all_true, all_scores = [], []
    for fb, labels, _ in loader:
        fb = fb.to(device)
        logits = model(fb)
        scores = torch.softmax(logits, 1).cpu().numpy()
        trues  = labels.argmax(1).cpu().numpy()
        all_true.extend(trues)
        all_scores.append(scores)
    return np.array(all_true), np.vstack(all_scores)


def fig_roc_curves(results, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(NUM_CLASSES * 4, 4))

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ax = axes[cls_idx]
        ax.plot([0,1],[0,1],"k--", lw=0.8, alpha=0.4)
        for r in sorted(results, key=lambda x: -x["icbhi_score"]):
            y_true, y_scores = get_scores(r["model"], device)
            if y_true is None:
                continue
            y_bin = (y_true == cls_idx).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_scores[:, cls_idx])
            roc_auc     = auc(fpr, tpr)
            color = COLORS.get(r["model"], "#607D8B")
            label = MODEL_DISPLAY.get(r["model"], r["model"])
            lw = 2.5 if r["model"] == "mamba" else 1.2
            ax.plot(fpr, tpr, color=color, lw=lw,
                    label=f"{label} ({roc_auc:.3f})")
        ax.set_title(f"{cls_name.capitalize()}", fontweight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR" if cls_idx == 0 else "")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.suptitle("Per-Class ROC Curves — All Models",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Fig 5] Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = load_all_results()
    if results is None:
        return

    print(f"[INFO] Loaded results for {len(results)} models")

    fig_confusion_matrices(results,
        os.path.join(OUT_DIR, "fig1_confusion_matrices.png"))
    fig_main_comparison(results,
        os.path.join(OUT_DIR, "fig2_main_comparison.png"))
    fig_per_class_recall(results,
        os.path.join(OUT_DIR, "fig3_per_class_recall.png"))
    fig_convergence(results,
        os.path.join(OUT_DIR, "fig4_convergence.png"))
    fig_roc_curves(results,
        os.path.join(OUT_DIR, "fig5_roc_curves.png"))

    print(f"\n[DONE] All figures saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
