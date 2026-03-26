"""
Step 2 – Compute Normalization Statistics
==========================================
Computes the mean and standard deviation of the mel-spectrogram features
over the training set.  These values are needed to normalise model inputs.

Run AFTER step1_prepare_data.py.

Usage:
    python step2_get_norm_stats.py

Output:
    datafiles/icbhi_norm_stats.npy   (array [mean, std])
    Prints the values to console so you can paste them into step3_train.py
"""

import os
import json
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────────
# Configuration – must match training settings
# ─────────────────────────────────────────────
PROJECT_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_JSON       = os.path.join(PROJECT_ROOT, "datafiles/icbhi_train.json")
LABEL_CSV        = os.path.join(PROJECT_ROOT, "datafiles/icbhi_labels.csv")
NUM_MEL_BINS     = 128
TARGET_LENGTH    = 512       # time frames (≈ 5 s at 10 ms frame shift)
FRAME_SHIFT      = 10        # ms
SAMPLE_RATE      = 22050
BATCH_SIZE       = 32
NUM_WORKERS      = 4

# ─────────────────────────────────────────────
# Minimal dataset class (no augmentation)
# ─────────────────────────────────────────────
class PlainAudioDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            self.data = json.load(f)["data"]

    def __len__(self):
        return len(self.data)

    def _load_fbank(self, wav_path):
        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception:
            return None
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=NUM_MEL_BINS,
            dither=0.0,
            frame_shift=FRAME_SHIFT,
        )
        # pad / trim to fixed length
        n = fbank.shape[0]
        p = TARGET_LENGTH - n
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        else:
            fbank = fbank[:TARGET_LENGTH, :]
        return fbank   # [T, F]

    def __getitem__(self, idx):
        datum = self.data[idx]
        fbank = None
        tries = 0
        while fbank is None and tries < 5:
            fbank = self._load_fbank(datum["wav"])
            tries += 1
            if fbank is None:
                idx = (idx + 1) % len(self.data)
                datum = self.data[idx]
        if fbank is None:
            fbank = torch.zeros(TARGET_LENGTH, NUM_MEL_BINS)
        return fbank


def compute_stats():
    print("[INFO] Loading training data …")
    dataset = PlainAudioDataset(TRAIN_JSON)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                         shuffle=False, pin_memory=False)

    running_sum  = 0.0
    running_sum2 = 0.0
    n_total      = 0

    for batch in tqdm(loader, desc="Computing stats"):
        # batch: [B, T, F]
        vals = batch.reshape(-1).numpy()
        running_sum  += vals.sum()
        running_sum2 += (vals ** 2).sum()
        n_total      += vals.size

    mean = running_sum  / n_total
    std  = np.sqrt(running_sum2 / n_total - mean ** 2)

    print(f"\n[RESULT] mean = {mean:.6f},  std = {std:.6f}")

    out_path = os.path.join(PROJECT_ROOT, "datafiles/icbhi_norm_stats.npy")
    np.save(out_path, np.array([mean, std]))
    print(f"[INFO]  Saved to {out_path}")
    print("\n[INFO] These values are used in configs/default.yaml:")
    print(f"  norm_mean: {mean:.6f}")
    print(f"  norm_std:  {std:.6f}")

    return mean, std


if __name__ == "__main__":
    os.makedirs(os.path.join(PROJECT_ROOT, "datafiles"), exist_ok=True)
    compute_stats()
