"""
Step 6 – Generate LaTeX Tables for Paper
==========================================
Generates ready-to-paste LaTeX tables from results.

Usage:
    python scripts/step6_generate_latex.py
"""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.icbhi_utils import CLASS_NAMES

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/comparative_study")
OUT_DIR     = os.path.join(PROJECT_ROOT, "figures")
NUM_CLASSES = 4

MODEL_DISPLAY = {
    "mfcc_mlp":    "MFCC + MLP",
    "cnn2d":       "CNN-2D",
    "alexnet":     "AlexNet",
    "vgg16":       "VGG-16",
    "resnet50":    "ResNet-50",
    "vit_small":   "ViT-Small",
    "mamba":       "ICBHIMamba-GRU",
    "mamba_real":  "ICBHIMamba-SSM (Ours)",
}


def latex_main_table(results, out_path):
    results_sorted = sorted(results, key=lambda x: -x["icbhi_score"])

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Comparative Results on ICBHI 2017 Respiratory Sound Database. "
        r"Best result per column in \textbf{bold}. "
        r"$\dagger$ = ImageNet pretrained.}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{l|cccc|cccc|cc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Acc.} & \textbf{Se} & \textbf{Sp} & "
        r"\textbf{ICBHI} & \textbf{Normal} & \textbf{Crackle} & "
        r"\textbf{Wheeze} & \textbf{Both} & \textbf{Params} & \textbf{ms/s} \\",
        r" & (\%) & (\%) & (\%) & (\%) & Recall(\%) & Recall(\%) & "
        r"Recall(\%) & Recall(\%) & (M) & \\",
        r"\hline",
    ]

    # Find best values for bolding
    best = {
        "accuracy":    max(r["accuracy"]    for r in results_sorted),
        "sensitivity": max(r["sensitivity"] for r in results_sorted),
        "specificity": max(r["specificity"] for r in results_sorted),
        "icbhi_score": max(r["icbhi_score"] for r in results_sorted),
    }
    recall_key = "per_class_recall" if "per_class_recall" in results_sorted[0] else "per_class_acc"
    best_pca = [max(r[recall_key][i] for r in results_sorted)
                for i in range(NUM_CLASSES)]

    def fmt(val, best_val, pct=True):
        s = f"{val*100:.2f}" if pct else f"{val:.2f}"
        return f"\\textbf{{{s}}}" if abs(val - best_val) < 1e-4 else s

    for r in results_sorted:
        pca  = r[recall_key]
        name = MODEL_DISPLAY.get(r["model"], r["model"])
        if r["model"] in ["vgg16", "resnet50"]:
            name += r"$^\dagger$"
        if r["model"] == "mamba_real":
            name = r"\textbf{" + name + "}"

        row = (f"  {name} & "
               f"{fmt(r['accuracy'],    best['accuracy'])} & "
               f"{fmt(r['sensitivity'], best['sensitivity'])} & "
               f"{fmt(r['specificity'], best['specificity'])} & "
               f"{fmt(r['icbhi_score'], best['icbhi_score'])} & "
               f"{fmt(pca[0], best_pca[0])} & "
               f"{fmt(pca[1], best_pca[1])} & "
               f"{fmt(pca[2], best_pca[2])} & "
               f"{fmt(pca[3], best_pca[3])} & "
               f"{r.get('params_M', 0):.2f} & "
               f"{r.get('infer_ms', 0):.1f} \\\\")
        lines.append(row)

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[LaTeX] Main table saved: {out_path}")
    print("\n" + text)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    path = os.path.join(RESULTS_DIR, "all_results.json")
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run step3 first.")
        return

    with open(path) as f:
        results = json.load(f)

    latex_main_table(results,
        os.path.join(OUT_DIR, "table1_main_results.tex"))

    print(f"\n[DONE] LaTeX tables saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
