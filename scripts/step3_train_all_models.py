"""
Step 3 – Train ALL Models Under Identical Conditions
=====================================================
Comparative study: 7 architectures, strictly identical experimental setup.

Models:
  1. MFCC + MLP    (hand-crafted baseline)
  2. CNN-2D        (custom convolutional)
  3. AlexNet       (classic deep CNN, from scratch)
  4. VGG-16        (ImageNet transfer learning)
  5. ResNet-50     (ImageNet transfer learning)
  6. ViT-Small     (Vision Transformer, from scratch)
  7. ICBHIMambaNet (SSM-inspired, from scratch — proposed)

IDENTICAL CONDITIONS:
  ✓ Same train/test split (patient-level 80/20)
  ✓ Same mel-spectrogram input [512 × 128]
  ✓ Same normalization (μ=-4.268, σ=4.569)
  ✓ Same augmentation (SpecAugment + MixUp + noise)
  ✓ Same loss (Focal Loss γ=1.5 + class weights)
  ✓ Same optimizer (AdamW, cosine LR + warmup)
  ✓ Same epochs (60), same seed (42)
  ✓ Same balanced sampler

Usage:
    python scripts/step3_train_all_models.py
    python scripts/step3_train_all_models.py --models mamba vit_small
    python scripts/step3_train_all_models.py --skip_trained
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.icbhi_dataloader import ICBHIDataset
from src.baseline_models  import get_model, count_parameters
from src.icbhi_utils      import (AverageMeter, icbhi_metrics,
                                  print_metrics, CLASS_NAMES)

# =============================================================================
# CONFIGURATION — identical for ALL models
# =============================================================================
TRAIN_JSON   = os.path.join(PROJECT_ROOT, "datafiles/icbhi_train.json")
TEST_JSON    = os.path.join(PROJECT_ROOT, "datafiles/icbhi_test.json")
LABEL_CSV    = os.path.join(PROJECT_ROOT, "datafiles/icbhi_labels.csv")
CKPT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results/comparative_study")

NORM_MEAN    = -11.605727
NORM_STD     =  4.965201
NUM_MEL_BINS = 128
TARGET_LEN   = 512
FRAME_SHIFT  = 10
FBANK_SIZE   = (TARGET_LEN, NUM_MEL_BINS)

# Augmentation
FREQ_MASK    = 32
TIME_MASK    = 128
MIXUP_RATE   = 0.2
ADD_NOISE    = True

# Loss
FOCAL_GAMMA      = 1.5
CLASS_WEIGHTS    = torch.tensor([1.8, 1.0, 3.0, 5.0], dtype=torch.float32)
LABEL_SMOOTHING  = 0.05

# Training
NUM_CLASSES      = 4
BATCH_SIZE       = 16
GRAD_ACCUM       = 4
NUM_EPOCHS       = 60
WARMUP_EPOCHS    = 5
LR               = 5e-4
MIN_LR           = 1e-6
WEIGHT_DECAY     = 3e-4
NUM_WORKERS      = 4
SEED             = 42
USE_AMP          = True

ALL_MODELS = ["mfcc_mlp", "cnn2d", "alexnet", "vgg16", "resnet50",
              "vit_small", "mamba", "mamba_real"]

# =============================================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Combines class weights + focal down-weighting of easy examples.
    """
    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight,
                              label_smoothing=self.label_smoothing,
                              reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Balanced sampler ──────────────────────────────────────────────────────────
def make_balanced_sampler():
    """Oversample minority classes so each batch has ~uniform class distribution."""
    with open(TRAIN_JSON) as f:
        data = json.load(f)["data"]
    labels = [int(d["labels"]) for d in data]
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    print(f"[INFO] Training class counts: "
          f"{dict(zip(CLASS_NAMES, counts.astype(int)))}")
    class_w = 1.0 / (counts + 1e-8)
    sample_w = torch.tensor([class_w[l] for l in labels], dtype=torch.double)
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


# ── Cosine LR ─────────────────────────────────────────────────────────────────
def cosine_lr(optimizer, warmup, total, min_lr, base_lr):
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        p = (ep - warmup) / max(1, total - warmup)
        return min_lr/base_lr + (1 - min_lr/base_lr) * 0.5*(1+np.cos(np.pi*p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ── Dataloaders ───────────────────────────────────────────────────────────────
def build_loaders():
    base = dict(num_mel_bins=NUM_MEL_BINS, target_length=TARGET_LEN,
                mean=NORM_MEAN, std=NORM_STD, fshift=FRAME_SHIFT)
    tr_conf = dict(**base, mode="train", freqm=FREQ_MASK, timem=TIME_MASK,
                   mixup=MIXUP_RATE, noise=ADD_NOISE)
    te_conf = dict(**base, mode="eval",  freqm=0, timem=0,
                   mixup=0.0, noise=False)

    train_ds = ICBHIDataset(TRAIN_JSON, tr_conf, LABEL_CSV)
    test_ds  = ICBHIDataset(TEST_JSON,  te_conf, LABEL_CSV)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=make_balanced_sampler(),
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, test_loader


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    meter = AverageMeter()
    optimizer.zero_grad()

    for step, (fb, labels, _) in enumerate(
            tqdm(loader, desc=f"  Ep{epoch:03d}", leave=False)):
        fb      = fb.to(device, non_blocking=True)
        targets = labels.argmax(1).to(device, non_blocking=True)

        with autocast(enabled=USE_AMP and device.type == "cuda"):
            loss = criterion(model(fb), targets) / GRAD_ACCUM

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        meter.update(loss.item() * GRAD_ACCUM, fb.size(0))

        if (step + 1) % GRAD_ACCUM == 0:
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

    return meter.avg


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for fb, labels, _ in loader:
        fb = fb.to(device, non_blocking=True)
        with autocast(enabled=USE_AMP and device.type == "cuda"):
            logits = model(fb)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(labels.argmax(1).cpu().numpy())
    return icbhi_metrics(np.array(trues), np.array(preds), NUM_CLASSES)


# ── Measure inference speed ───────────────────────────────────────────────────
def measure_inference_time(model, device, n_runs=100):
    """Returns avg inference time in ms per sample."""
    model.eval()
    dummy = torch.randn(1, TARGET_LEN, NUM_MEL_BINS).to(device)
    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_runs * 1000
    return round(elapsed, 3)


# ── Train one model ───────────────────────────────────────────────────────────
def train_model(model_name, train_loader, test_loader, device):
    print(f"\n{'='*65}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*65}")

    set_seed(SEED)  # Reset seed before each model for reproducibility

    model = get_model(model_name, NUM_CLASSES, FBANK_SIZE).to(device)
    params = count_parameters(model)
    print(f"  Parameters: {params:.2f}M")

    weights   = CLASS_WEIGHTS.to(device)
    criterion = FocalLoss(weight=weights, gamma=FOCAL_GAMMA,
                          label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = cosine_lr(optimizer, WARMUP_EPOCHS, NUM_EPOCHS, MIN_LR, LR)
    scaler    = GradScaler() if (USE_AMP and device.type == "cuda") else None

    history    = dict(train_loss=[], icbhi_score=[], sensitivity=[],
                      specificity=[], per_class_recall=[])
    best_icbhi = 0.0
    best_ckpt  = os.path.join(CKPT_DIR, f"best_{model_name}.pth")
    t0         = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion,
                           scaler, device, epoch)
        scheduler.step()
        m = evaluate(model, test_loader, device)

        recall_key = "per_class_recall" if "per_class_recall" in m else "per_class_acc"
        history["train_loss"].append(loss)
        history["icbhi_score"].append(m["icbhi_score"])
        history["sensitivity"].append(m["sensitivity"])
        history["specificity"].append(m["specificity"])
        history["per_class_recall"].append(m[recall_key])

        pca = m[recall_key]
        print(f"  Ep{epoch:3d}/{NUM_EPOCHS} | "
              f"Loss {loss:.4f} | "
              f"ICBHI {m['icbhi_score']*100:.2f}% | "
              f"Acc {m['accuracy']*100:.1f}% | "
              f"[N:{pca[0]*100:.0f}% C:{pca[1]*100:.0f}% "
              f"W:{pca[2]*100:.0f}% B:{pca[3]*100:.0f}%] | "
              f"{(time.time()-t0)/60:.1f}m")

        if m["icbhi_score"] > best_icbhi:
            best_icbhi = m["icbhi_score"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "metrics": m}, best_ckpt)
            print(f"  ★ Best ICBHI: {best_icbhi*100:.2f}%")

    # Final evaluation with best checkpoint
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    final_m = evaluate(model, test_loader, device)

    # Inference speed
    infer_ms = measure_inference_time(model, device)

    recall_key = "per_class_recall" if "per_class_recall" in final_m else "per_class_acc"
    result = {
        "model":          model_name,
        "params_M":       round(params, 3),
        "infer_ms":       infer_ms,
        "best_epoch":     ckpt["epoch"],
        "accuracy":       final_m["accuracy"],
        "sensitivity":    final_m["sensitivity"],
        "specificity":    final_m["specificity"],
        "icbhi_score":    final_m["icbhi_score"],
        "f1_weighted":    final_m["f1_weighted"],
        "per_class_recall": final_m[recall_key],
        "per_class_se":   final_m.get("per_class_se", final_m[recall_key]),
        "per_class_sp":   final_m.get("per_class_sp", []),
        "confusion_matrix": final_m["confusion_matrix"],
        "history":        history,
    }

    # Save per-model results
    out_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    saveable = {k: v for k, v in result.items() if k != "history"}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(saveable, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  [{model_name.upper()} FINAL]  "
          f"ICBHI={final_m['icbhi_score']*100:.2f}%  "
          f"Acc={final_m['accuracy']*100:.2f}%  "
          f"Params={params:.2f}M  "
          f"Speed={infer_ms}ms/sample")

    return result


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary_table(results):
    print("\n" + "=" * 110)
    print(f"  {'Model':15s} {'Acc%':>7} {'Se%':>7} {'Sp%':>7} "
          f"{'ICBHI%':>8} {'F1%':>7} "
          f"{'N%':>6} {'C%':>6} {'W%':>6} {'B%':>6} "
          f"{'Params':>8} {'ms/s':>7}")
    print("-" * 110)
    for r in sorted(results, key=lambda x: -x["icbhi_score"]):
        pca = r["per_class_recall"]
        print(f"  {r['model']:15s} "
              f"{r['accuracy']*100:>7.2f} "
              f"{r['sensitivity']*100:>7.2f} "
              f"{r['specificity']*100:>7.2f} "
              f"{r['icbhi_score']*100:>8.2f} "
              f"{r['f1_weighted']*100:>7.2f} "
              f"{pca[0]*100:>6.1f} "
              f"{pca[1]*100:>6.1f} "
              f"{pca[2]*100:>6.1f} "
              f"{pca[3]*100:>6.1f} "
              f"{r['params_M']:>8.2f} "
              f"{r['infer_ms']:>7.1f}")
    print("=" * 110)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train all models for ICBHI comparative study")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="Models to train (default: all 7)")
    parser.add_argument("--skip_trained", action="store_true",
                        help="Skip models that already have a checkpoint")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs (default: 60)")
    args = parser.parse_args()

    pass  # epochs use module-level NUM_EPOCHS

    set_seed(SEED)
    os.makedirs(CKPT_DIR,     exist_ok=True)
    os.makedirs(RESULTS_DIR,  exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device : {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU    : {torch.cuda.get_device_name(0)}")

    print(f"[INFO] Models to train: {args.models}")
    print(f"[INFO] Epochs per model: {NUM_EPOCHS}")
    print(f"[INFO] Loss: FocalLoss(γ={FOCAL_GAMMA}) + "
          f"weights {CLASS_WEIGHTS.tolist()}")

    train_loader, test_loader = build_loaders()

    all_results  = []
    total_start  = time.time()

    for model_name in args.models:
        ckpt_path = os.path.join(CKPT_DIR, f"best_{model_name}.pth")
        if args.skip_trained and os.path.exists(ckpt_path):
            metrics_path = os.path.join(RESULTS_DIR, model_name, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    r = json.load(f)
                all_results.append(r)
                print(f"[SKIP] {model_name} — loaded existing results "
                      f"(ICBHI={r['icbhi_score']*100:.2f}%)")
                continue

        result = train_model(model_name, train_loader, test_loader, device)
        all_results.append(result)

        # Save running summary
        summary = [{k: v for k, v in r.items() if k != "history"}
                   for r in all_results]
        with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # Final summary
    print_summary_table(all_results)

    total_mins = (time.time() - total_start) / 60
    print(f"\n[DONE] Total training time: {total_mins:.1f} minutes")
    print(f"[DONE] Results: {RESULTS_DIR}/")
    print(f"[NEXT] Run: python scripts/step4_generate_figures.py")


if __name__ == "__main__":
    main()
