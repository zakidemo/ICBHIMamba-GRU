"""
ICBHI Training Utilities
=========================
Metrics and helper classes for the ICBHI 4-class lung sound classification task.

Metrics computed:
  - Sensitivity  (Se) : macro-average recall across all 4 classes
  - Specificity  (Sp) : macro-average specificity across all 4 classes
  - ICBHI Score        : (Se + Sp) / 2
  - Overall Accuracy   : fraction of correctly classified samples
  - Per-class Recall   : individual class recall (TP / (TP + FN))

Note: Our Se/Sp computation follows the macro-average convention where
both metrics are averaged across ALL 4 classes, including Normal.
This differs from some ICBHI implementations that define Se as
pathological-only average and Sp as Normal recall.
"""

import numpy as np
import csv
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
)


CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]


# ─────────────────────────────────────────────────────────────────────────────
class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ─────────────────────────────────────────────────────────────────────────────
def icbhi_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4):
    """
    Compute ICBHI challenge metrics.

    Parameters
    ----------
    y_true : 1-D int array of ground truth class indices
    y_pred : 1-D int array of predicted class indices
    num_classes : int

    Returns
    -------
    dict with keys:
        accuracy, sensitivity, specificity, icbhi_score,
        f1_weighted, per_class_recall, per_class_se, per_class_sp,
        confusion_matrix
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class_recall = []
    sensitivities    = []
    specificities    = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        se = TP / (TP + FN + 1e-8)   # sensitivity / recall for class i
        sp = TN / (TN + FP + 1e-8)   # specificity for class i

        sensitivities.append(se)
        specificities.append(sp)
        per_class_recall.append(se)   # recall = sensitivity per class

    sensitivity = float(np.mean(sensitivities))
    specificity = float(np.mean(specificities))
    icbhi_score = (sensitivity + specificity) / 2.0
    accuracy    = float(accuracy_score(y_true, y_pred))
    f1          = float(f1_score(y_true, y_pred, average="weighted",
                                 zero_division=0))

    return {
        "accuracy":          accuracy,
        "sensitivity":       sensitivity,
        "specificity":       specificity,
        "icbhi_score":       icbhi_score,
        "f1_weighted":       f1,
        "per_class_recall":  per_class_recall,
        "per_class_acc":     per_class_recall,   # backward compat alias
        "per_class_se":      sensitivities,
        "per_class_sp":      specificities,
        "confusion_matrix":  cm.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
def print_metrics(metrics: dict):
    """Pretty-print metric results."""
    print("\n" + "=" * 60)
    print(f"  Accuracy    : {metrics['accuracy']     * 100:.2f}%")
    print(f"  Sensitivity : {metrics['sensitivity']  * 100:.2f}%  (Se, macro-avg)")
    print(f"  Specificity : {metrics['specificity']  * 100:.2f}%  (Sp, macro-avg)")
    print(f"  ICBHI Score : {metrics['icbhi_score']  * 100:.2f}%  (Se+Sp)/2")
    print(f"  F1 Weighted : {metrics['f1_weighted']  * 100:.2f}%")
    print("-" * 60)
    print("  Per-class recall:")
    recall_key = "per_class_recall" if "per_class_recall" in metrics else "per_class_acc"
    for name, acc in zip(CLASS_NAMES, metrics[recall_key]):
        print(f"    {name:10s}: {acc * 100:.2f}%")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix_csv(cm, out_path: str):
    """Save confusion matrix to a CSV file."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + [int(cm[i][j]) for j in range(len(CLASS_NAMES))])
    print(f"[INFO] Confusion matrix saved to {out_path}")
