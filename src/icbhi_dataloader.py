"""
ICBHI Dataloader
================
Adapted from Audio-Mamba (AuM) AudiosetDataset for 4-class lung sound classification.

Classes:
  0 – normal
  1 – crackle
  2 – wheeze
  3 – both (crackle + wheeze)
"""

import os
import csv
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import random


# ─────────────────────────────────────────────────────────────────────────────
# Utility: label CSV parsing
# ─────────────────────────────────────────────────────────────────────────────

def make_index_dict(label_csv):
    """mid → index mapping."""
    mapping = {}
    with open(label_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["mid"]] = row["index"]
    return mapping


def make_name_dict(label_csv):
    """index → display_name mapping."""
    mapping = {}
    with open(label_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["index"]] = row["display_name"]
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ICBHIDataset(Dataset):
    """
    Dataset for ICBHI lung sound classification.

    audio_conf keys:
        num_mel_bins   : number of Mel filterbanks (default 128)
        target_length  : number of time frames to pad/trim to (default 512)
        freqm          : frequency mask max width (0 = off)
        timem          : time mask max width (0 = off)
        mixup          : mixup probability (0 = off)
        mean           : dataset mean for normalisation
        std            : dataset std  for normalisation
        noise          : bool – add gaussian noise augmentation
        mode           : 'train' or 'eval'
        fshift         : frame shift in ms (default 10)
    """

    def __init__(self, dataset_json_file: str, audio_conf: dict, label_csv: str):
        with open(dataset_json_file) as fp:
            data_json = json.load(fp)
        self.data = data_json["data"]

        self.audio_conf   = audio_conf
        self.melbins      = audio_conf.get("num_mel_bins",  128)
        self.target_len   = audio_conf.get("target_length", 512)
        self.freqm        = audio_conf.get("freqm",          0)
        self.timem        = audio_conf.get("timem",           0)
        self.mixup        = audio_conf.get("mixup",         0.0)
        self.norm_mean    = audio_conf.get("mean",          -4.2677393)
        self.norm_std     = audio_conf.get("std",            4.5689974)
        self.noise        = audio_conf.get("noise",         False)
        self.mode         = audio_conf.get("mode",         "train")
        self.fshift       = audio_conf.get("fshift",         10)
        self.skip_norm    = audio_conf.get("skip_norm",     False)

        self.index_dict   = make_index_dict(label_csv)
        self.label_num    = len(self.index_dict)

        print(f"[Dataloader – {self.mode}] {len(self.data)} samples, "
              f"{self.label_num} classes, "
              f"freqm={self.freqm}, timem={self.timem}, mixup={self.mixup}")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _wav2fbank(self, path1, path2=None):
        """Convert wav file(s) to mel-filterbank. Returns (fbank, mix_lambda)."""
        try:
            w1, sr = torchaudio.load(path1)
        except Exception:
            return None, 0.0

        w1 = w1 - w1.mean()

        if path2 is not None:
            try:
                w2, _ = torchaudio.load(path2)
            except Exception:
                path2 = None

        if path2 is not None:
            w2 = w2 - w2.mean()
            if w1.shape[1] != w2.shape[1]:
                if w1.shape[1] > w2.shape[1]:
                    pad = torch.zeros(1, w1.shape[1] - w2.shape[1])
                    w2 = torch.cat([w2, pad], dim=1)
                else:
                    w2 = w2[:, : w1.shape[1]]
            lam      = np.random.beta(10, 10)
            waveform = lam * w1 + (1.0 - lam) * w2
            waveform = waveform - waveform.mean()
        else:
            waveform = w1
            lam      = 0.0

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.melbins,
            dither=0.0,
            frame_shift=self.fshift,
        )

        n = fbank.shape[0]
        p = self.target_len - n
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        else:
            fbank = fbank[: self.target_len, :]

        return fbank, lam   # [T, F], scalar

    def _make_label(self, datum, mix_lambda=None, mix_datum=None):
        """Build a one-hot (or soft) label tensor."""
        label = np.zeros(self.label_num, dtype=np.float32)

        if mix_datum is not None and mix_lambda is not None:
            lam = mix_lambda
            for ls in datum["labels"].split(","):
                label[int(self.index_dict[ls.strip()])] += lam
            for ls in mix_datum["labels"].split(","):
                label[int(self.index_dict[ls.strip()])] += 1.0 - lam
        else:
            for ls in datum["labels"].split(","):
                label[int(self.index_dict[ls.strip()])] = 1.0

        return torch.FloatTensor(label)

    # ─────────────────────────────────────────────────────────────────────
    # __getitem__
    # ─────────────────────────────────────────────────────────────────────

    def __getitem__(self, index):
        use_mixup = (random.random() < self.mixup) and (self.mode == "train")

        fbank = None
        attempts = 0
        while fbank is None:
            datum = self.data[index]

            if use_mixup:
                mix_idx   = random.randint(0, len(self.data) - 1)
                mix_datum = self.data[mix_idx]
                fbank, lam = self._wav2fbank(datum["wav"], mix_datum["wav"])
            else:
                mix_datum = None
                fbank, lam = self._wav2fbank(datum["wav"])

            if fbank is None:
                index   = random.randint(0, len(self.data) - 1)
                attempts += 1
                if attempts > 10:
                    fbank = torch.zeros(self.target_len, self.melbins)
                    lam   = 0.0

        label_indices = self._make_label(datum,
                                         mix_lambda=lam if use_mixup else None,
                                         mix_datum=mix_datum if use_mixup else None)

        # ── SpecAugment (train only) ──────────────────────────────────────
        fbank = torch.transpose(fbank, 0, 1).unsqueeze(0)   # [1, F, T]
        if self.mode == "train":
            if self.freqm > 0:
                fbank = torchaudio.transforms.FrequencyMasking(self.freqm)(fbank)
            if self.timem > 0:
                fbank = torchaudio.transforms.TimeMasking(self.timem)(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)                # [T, F]

        # ── Normalise ─────────────────────────────────────────────────────
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        # ── Noise + time-shift augmentation (train only) ──────────────────
        # Adds uniform random noise and applies circular time shift
        if self.noise and self.mode == "train":
            fbank = fbank + torch.randn_like(fbank) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        return fbank, label_indices, datum["wav"]

    def __len__(self):
        return len(self.data)
