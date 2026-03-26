"""
ICBHI 2017 Lung Sound Dataset Preparation
==========================================
Processes the raw ICBHI dataset:
1. Reads all .wav audio files and their annotation .txt files
2. Splits recordings into individual respiratory cycles
3. Labels each cycle: normal, crackle, wheeze, both
4. Creates train/test JSON using the OFFICIAL ICBHI 60/40 patient split
5. Creates label CSV mapping

OFFICIAL SPLIT: The ICBHI 2017 challenge defines a 60/40 patient-level
train/test split via patient_list_foldwise.txt (test = fold 5).

Usage:
    python scripts/step1_prepare_data.py
"""

import os
import sys
import json
import csv
import random
import numpy as np
import torchaudio
import torch
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ICBHI_DIR        = os.path.join(PROJECT_ROOT, "data/ICBHI_final_database")
OUTPUT_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data/cycles")
DATAFILES_DIR    = os.path.join(PROJECT_ROOT, "datafiles")
SAMPLE_RATE      = 22050
CYCLE_DURATION   = 5.0
SEED             = 42

CLASSES = {
    (0, 0): ("normal", 0),
    (1, 0): ("crackle", 1),
    (0, 1): ("wheeze",  2),
    (1, 1): ("both",    3),
}

random.seed(SEED)
np.random.seed(SEED)


def load_official_split(icbhi_dir):
    """
    Load patient_list_foldwise.txt — the official ICBHI split file.
    Format: patient_id<TAB>fold_number
    Test fold = 4 (folds are 0-4) (following RespireNet/LungAttn convention).
    Returns (train_patients, test_patients) sets, or None if file not found.
    """
    for candidate in [
        os.path.join(icbhi_dir, "patient_list_foldwise.txt"),
        os.path.join(os.path.dirname(icbhi_dir), "patient_list_foldwise.txt"),
        os.path.join(PROJECT_ROOT, "data", "patient_list_foldwise.txt"),
    ]:
        if os.path.exists(candidate):
            train_patients, test_patients = set(), set()
            with open(candidate, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        pid, fold = parts[0], int(parts[1])
                        (test_patients if fold == 4 else train_patients).add(pid)
            print(f"[INFO] Loaded official split: {candidate}")
            print(f"       Train: {len(train_patients)} patients, "
                  f"Test: {len(test_patients)} patients")
            return train_patients, test_patients
    return None


def parse_annotation(txt_path):
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                cycles.append((float(parts[0]), float(parts[1]),
                               int(parts[2]), int(parts[3])))
    return cycles


def load_audio(wav_path, target_sr=SAMPLE_RATE):
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform, target_sr


def extract_cycle(waveform, sr, start, end, target_duration=CYCLE_DURATION):
    start_sample = int(start * sr)
    end_sample   = int(end * sr)
    cycle = waveform[:, start_sample:end_sample]
    target_len = int(target_duration * sr)
    if cycle.shape[1] < target_len:
        cycle = torch.cat([cycle, torch.zeros(1, target_len - cycle.shape[1])], dim=1)
    else:
        cycle = cycle[:, :target_len]
    return cycle


def process_dataset():
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)
    os.makedirs(DATAFILES_DIR, exist_ok=True)

    icbhi_path = Path(ICBHI_DIR)
    if not icbhi_path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] ICBHI dataset not found at '{ICBHI_DIR}'.\n"
            "Place ICBHI_final_database inside data/\n")

    wav_files = sorted(icbhi_path.glob("*.wav"))
    print(f"[INFO] Found {len(wav_files)} wav files")

    # ── Load official split ─────────────────────────────────────────
    split_result = load_official_split(ICBHI_DIR)

    # ── Extract all cycles ──────────────────────────────────────────
    all_records = []
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    all_patients = set()

    for wav_file in wav_files:
        txt_file = wav_file.with_suffix(".txt")
        if not txt_file.exists():
            continue
        try:
            waveform, sr = load_audio(str(wav_file))
        except Exception as e:
            print(f"[WARN] Could not load {wav_file.name}: {e}")
            continue

        cycles = parse_annotation(str(txt_file))
        stem = wav_file.stem
        patient_id = stem.split("_")[0]
        all_patients.add(patient_id)

        for i, (start, end, crackle, wheeze) in enumerate(cycles):
            class_name, class_idx = CLASSES[(crackle, wheeze)]
            out_fname = f"{stem}_cycle{i:03d}_{class_name}.wav"
            out_path = os.path.join(OUTPUT_AUDIO_DIR, out_fname)

            # Only extract if not already done
            if not os.path.exists(out_path):
                cycle_wav = extract_cycle(waveform, sr, start, end)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torchaudio.save(out_path, cycle_wav, sr)

            class_counts[class_idx] += 1
            all_records.append({
                "wav": os.path.abspath(out_path),
                "labels": str(class_idx),
                "class_name": class_name,
                "patient": patient_id,
            })

    print(f"\n[INFO] Total cycles: {len(all_records)}")
    for (_, _), (name, cidx) in CLASSES.items():
        print(f"  {name:10s}: {class_counts[cidx]}")

    # ── Split data ──────────────────────────────────────────────────
    if split_result is not None:
        train_patients, test_patients = split_result
    else:
        # Fallback: 60/40 random patient split (not recommended)
        print("[WARN] Official split file not found — using random 60/40 split")
        patients = sorted(all_patients)
        random.shuffle(patients)
        split_idx = int(len(patients) * 0.6)
        train_patients = set(patients[:split_idx])
        test_patients = set(patients[split_idx:])

    train_data = [r for r in all_records if r["patient"] in train_patients]
    test_data = [r for r in all_records if r["patient"] in test_patients]

    # Handle unassigned patients
    assigned = train_patients | test_patients
    unassigned = [r for r in all_records if r["patient"] not in assigned]
    if unassigned:
        print(f"[WARN] {len(unassigned)} cycles from unrecognized patients → train")
        train_data.extend(unassigned)

    print(f"\n[INFO] Split: Train={len(train_data)}, Test={len(test_data)}")
    for name, data in [("Train", train_data), ("Test", test_data)]:
        counts = {}
        for r in data:
            counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
        print(f"  {name}: {counts}")

    # ── Write outputs ───────────────────────────────────────────────
    def write_json(records, path):
        with open(path, "w") as f:
            json.dump({"data": [{"wav": r["wav"], "labels": r["labels"]}
                                for r in records]}, f, indent=2)
        print(f"[INFO] Saved {path} ({len(records)} samples)")

    write_json(train_data, os.path.join(DATAFILES_DIR, "icbhi_train.json"))
    write_json(test_data, os.path.join(DATAFILES_DIR, "icbhi_test.json"))

    label_csv = os.path.join(DATAFILES_DIR, "icbhi_labels.csv")
    with open(label_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "mid", "display_name"])
        w.writeheader()
        for (_, _), (name, idx) in CLASSES.items():
            w.writerow({"index": idx, "mid": str(idx), "display_name": name})

    print(f"\n[DONE] Dataset ready (official 60/40 split)")


if __name__ == "__main__":
    process_dataset()
