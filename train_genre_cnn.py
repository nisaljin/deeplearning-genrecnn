#!/usr/bin/env python3
"""Train genre classification CNN with notebook-aligned workflow.

Features synced from `genre_classifier_workflow.ipynb`:
- official or stratified splits
- rare-class merge/drop policy based on train support
- precompute full-track mel cache + skip broken decodes
- train-time random mel crops + SpecAugment
- weighted sampler for imbalance
- ResCNN (default) or comparison CNN2 architecture
- multi-crop validation/test evaluation
- early stopping and checkpointing on macro metric
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SPLIT_TRAIN = "training"
SPLIT_VAL = "validation"
SPLIT_TEST = "test"


@dataclass
class TrackRecord:
    track_id: int
    genre: str
    split: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FMA genre CNN (notebook-synced workflow)")
    p.add_argument("--audio-dir", type=Path, default=Path("fma_large"))
    p.add_argument("--metadata-path", type=Path, default=Path("fma_metadata/tracks.csv"))
    p.add_argument("--cache-dir", type=Path, default=Path(".cache/mels_full_29s"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/high_recall_precision_run"))

    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--precompute-duration", type=float, default=29.0)
    p.add_argument("--train-crop-duration", type=float, default=5.0)
    p.add_argument("--eval-crop-duration", type=float, default=5.0)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)

    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)

    p.add_argument("--split-strategy", type=str, default="stratified", choices=["official", "stratified"])
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--min-class-count-for-split", type=int, default=10)

    p.add_argument("--min-train-count-for-model", type=int, default=50)
    p.add_argument("--merge-rare-classes", action="store_true", default=True)
    p.add_argument("--no-merge-rare-classes", action="store_false", dest="merge_rare_classes")
    p.add_argument("--rare-class-name", type=str, default="Other")

    p.add_argument("--min-audio-file-bytes", type=int, default=4096)
    p.add_argument("--exclude-bad-audio-scan", action="store_true", default=True)
    p.add_argument("--no-exclude-bad-audio-scan", action="store_false", dest="exclude_bad_audio_scan")
    p.add_argument("--bad-audio-scan-path", type=Path, default=Path("outputs/bad_audio_scan.csv"))

    p.add_argument("--use-balanced-sampler", action="store_true", default=True)
    p.add_argument("--no-use-balanced-sampler", action="store_false", dest="use_balanced_sampler")
    p.add_argument("--use-focal-loss", action="store_true", default=False)
    p.add_argument("--focal-gamma", type=float, default=1.5)

    p.add_argument("--use-aug", action="store_true", default=True)
    p.add_argument("--no-use-aug", action="store_false", dest="use_aug")
    p.add_argument("--time-mask-max", type=int, default=24)
    p.add_argument("--freq-mask-max", type=int, default=12)

    p.add_argument("--val-multi-crops", type=int, default=5)
    p.add_argument("--test-multi-crops", type=int, default=7)

    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--metric-for-best", type=str, default="val_f1_macro", choices=["val_f1_macro", "val_recall_macro"])

    p.add_argument("--model-arch", type=str, default="res_cnn", choices=["res_cnn", "their_cnn2"])
    p.add_argument("--their-recipe-lr", type=float, default=1e-4)
    p.add_argument("--their-recipe-disable-spec-aug", action="store_true", default=True)
    p.add_argument("--no-their-recipe-disable-spec-aug", action="store_false", dest="their_recipe_disable_spec_aug")

    p.add_argument("--precompute-overwrite", action="store_true", default=False)
    p.add_argument("--eval-log-every", type=int, default=500)
    p.add_argument("--posthoc-tune", action="store_true", default=True)
    p.add_argument("--no-posthoc-tune", action="store_false", dest="posthoc_tune")
    p.add_argument(
        "--posthoc-objective",
        type=str,
        default="f1_macro",
        choices=[
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
        ],
    )
    p.add_argument("--posthoc-temp-values", type=str, default="0.8,1.0,1.2,1.4")
    p.add_argument("--posthoc-prior-tau-values", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--posthoc-bias-iters", type=int, default=2)
    p.add_argument("--posthoc-bias-step", type=float, default=0.2)
    p.add_argument("--posthoc-bias-step-decay", type=float, default=0.5)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested cuda but CUDA is unavailable.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested mps but MPS is unavailable.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def optimize_runtime(device: torch.device) -> Dict[str, object]:
    use_amp = False
    amp_dtype = None
    use_channels_last = False
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        use_amp = True
        amp_dtype = torch.float16
        use_channels_last = True

    return {"use_amp": use_amp, "amp_dtype": amp_dtype, "use_channels_last": use_channels_last}


def track_path(audio_dir: Path, track_id: int) -> Path:
    return audio_dir / f"{track_id:06d}"[:3] / f"{track_id:06d}.mp3"


def cache_key(track_id: int, sample_rate: int, duration: float, n_mels: int, n_fft: int, hop_length: int) -> str:
    return f"{track_id:06d}_sr{sample_rate}_dur{duration}_mel{n_mels}_fft{n_fft}_hop{hop_length}.npy"


def load_records(metadata_path: Path, audio_dir: Path, min_audio_file_bytes: int = 0) -> List[TrackRecord]:
    df = pd.read_csv(metadata_path, header=[0, 1], index_col=0, low_memory=False)
    df.index = df.index.astype(int)

    subset_mask = df[("set", "subset")] == "large"
    labeled_mask = df[("track", "genre_top")].notna()
    filtered = df[subset_mask & labeled_mask]

    out: List[TrackRecord] = []
    skipped_missing = 0
    skipped_tiny = 0

    for track_id, row in filtered.iterrows():
        ap = track_path(audio_dir, int(track_id))
        if not ap.exists():
            skipped_missing += 1
            continue
        if min_audio_file_bytes > 0 and ap.stat().st_size < min_audio_file_bytes:
            skipped_tiny += 1
            continue
        out.append(TrackRecord(int(track_id), str(row[("track", "genre_top")]), str(row[("set", "split")])))

    print(f"Loaded labeled records={len(out)} (skipped missing={skipped_missing}, tiny<{min_audio_file_bytes}B={skipped_tiny})")
    return out


def drop_known_bad_audio(records: Sequence[TrackRecord], bad_audio_scan_path: Path, enabled: bool) -> List[TrackRecord]:
    if not enabled:
        return list(records)
    if not bad_audio_scan_path.exists():
        print(f"Bad-audio scan not found: {bad_audio_scan_path}; continuing without extra filtering.")
        return list(records)

    bad = pd.read_csv(bad_audio_scan_path)
    if "track_id" not in bad.columns:
        print(f"Bad-audio scan missing track_id column: {bad_audio_scan_path}; continuing without extra filtering.")
        return list(records)

    bad_ids = set(bad["track_id"].astype(int).tolist())
    filtered = [r for r in records if r.track_id not in bad_ids]
    print(f"Dropped known bad-audio tracks from scan: {len(records) - len(filtered)}")
    return filtered


def split_records_official(records: Sequence[TrackRecord]) -> Tuple[List[TrackRecord], List[TrackRecord], List[TrackRecord]]:
    train = [TrackRecord(r.track_id, r.genre, SPLIT_TRAIN) for r in records if r.split == SPLIT_TRAIN]
    val = [TrackRecord(r.track_id, r.genre, SPLIT_VAL) for r in records if r.split == SPLIT_VAL]
    test = [TrackRecord(r.track_id, r.genre, SPLIT_TEST) for r in records if r.split == SPLIT_TEST]
    return train, val, test


def split_records_stratified(
    records: Sequence[TrackRecord],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    min_class_count: int,
) -> Tuple[List[TrackRecord], List[TrackRecord], List[TrackRecord], Dict[str, int]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total}")

    df = pd.DataFrame([{"track_id": r.track_id, "genre": r.genre, "split": r.split} for r in records])
    counts = df["genre"].value_counts()

    keep_genres = counts[counts >= min_class_count].index.tolist()
    dropped = counts[counts < min_class_count].to_dict()
    df = df[df["genre"].isin(keep_genres)].copy()

    if len(df) == 0:
        raise RuntimeError("No records left after min_class_count filtering; lower --min-class-count-for-split.")

    idx = np.arange(len(df))
    y = df["genre"].to_numpy()

    train_idx, hold_idx = train_test_split(
        idx,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    hold_y = y[hold_idx]
    hold_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        hold_idx,
        test_size=hold_test_ratio,
        random_state=seed + 1,
        shuffle=True,
        stratify=hold_y,
    )

    def build(ix: np.ndarray, split_name: str) -> List[TrackRecord]:
        chunk = df.iloc[ix]
        return [TrackRecord(int(r.track_id), str(r.genre), split_name) for r in chunk.itertuples(index=False)]

    train = build(train_idx, SPLIT_TRAIN)
    val = build(val_idx, SPLIT_VAL)
    test = build(test_idx, SPLIT_TEST)
    return train, val, test, dropped


def maybe_limit(records: List[TrackRecord], max_items: int | None, seed: int) -> List[TrackRecord]:
    if max_items is None or len(records) <= max_items:
        return records
    rng = random.Random(seed)
    x = records.copy()
    rng.shuffle(x)
    return x[:max_items]


def apply_train_support_policy(
    train_records: Sequence[TrackRecord],
    val_records: Sequence[TrackRecord],
    test_records: Sequence[TrackRecord],
    min_train_count: int,
    merge_rare: bool,
    rare_class_name: str,
) -> Tuple[List[TrackRecord], List[TrackRecord], List[TrackRecord], Dict[str, object]]:
    train = list(train_records)
    val = list(val_records)
    test = list(test_records)

    def counts_of(rs: Sequence[TrackRecord]) -> pd.Series:
        if len(rs) == 0:
            return pd.Series(dtype="int64")
        return pd.Series([r.genre for r in rs]).value_counts().sort_index()

    def relabel(rs: Sequence[TrackRecord], remap: Dict[str, str]) -> List[TrackRecord]:
        out: List[TrackRecord] = []
        for r in rs:
            g = remap.get(r.genre, r.genre)
            out.append(TrackRecord(r.track_id, g, r.split))
        return out

    info: Dict[str, object] = {
        "min_train_count": int(min_train_count),
        "initial_train_counts": counts_of(train).astype(int).to_dict(),
        "rare_classes": {},
        "merged_to": None,
        "dropped_after_policy": {},
        "final_train_counts": {},
    }

    train_counts = counts_of(train)
    rare = train_counts[train_counts < min_train_count]
    info["rare_classes"] = rare.astype(int).to_dict()

    if merge_rare and len(rare) > 0:
        remap = {g: rare_class_name for g in rare.index.tolist()}
        train = relabel(train, remap)
        val = relabel(val, remap)
        test = relabel(test, remap)
        info["merged_to"] = rare_class_name

    post_counts = counts_of(train)
    keep = set(post_counts[post_counts >= min_train_count].index.tolist())
    dropped = post_counts[~post_counts.index.isin(keep)]
    info["dropped_after_policy"] = dropped.astype(int).to_dict()

    if len(dropped) > 0:
        train = [r for r in train if r.genre in keep]
        val = [r for r in val if r.genre in keep]
        test = [r for r in test if r.genre in keep]

    final_counts = counts_of(train)
    if len(final_counts) == 0:
        raise RuntimeError("No classes left after train-support policy; lower --min-train-count-for-model.")
    info["final_train_counts"] = final_counts.astype(int).to_dict()
    return train, val, test, info


def print_split_support(train_records: Sequence[TrackRecord], val_records: Sequence[TrackRecord], test_records: Sequence[TrackRecord], min_train_count: int) -> None:
    for split_name, rs in [("train", train_records), ("val", val_records), ("test", test_records)]:
        counts = pd.Series([r.genre for r in rs]).value_counts().sort_index()
        print(
            f"\n[{split_name}] classes={len(counts)} min={int(counts.min())} "
            f"median={float(counts.median()):.1f} max={int(counts.max())}"
        )
        print(counts.to_string())

    all_counts = pd.Series([r.genre for r in train_records]).value_counts().sort_index()
    tiny = all_counts[all_counts < min_train_count]
    if len(tiny) > 0:
        print(f"\nWarning: very low-support classes in train (<{min_train_count} samples).")
        print(tiny.to_string())


def mel_from_audio(y: np.ndarray, sample_rate: int, target_samples: int, n_mels: int, n_fft: int, hop_length: int) -> np.ndarray:
    if y.shape[0] < target_samples:
        y = np.pad(y, (0, target_samples - y.shape[0]))
    elif y.shape[0] > target_samples:
        y = y[:target_samples]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=sample_rate // 2,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mean = float(mel_db.mean())
    std = float(mel_db.std())
    return (mel_db - mean) / (std + 1e-6)


def precompute_cache(
    records: Sequence[TrackRecord],
    audio_dir: Path,
    cache_dir: Path,
    sample_rate: int,
    precompute_duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    overwrite: bool,
    log_every: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_samples = int(sample_rate * precompute_duration)
    written = skipped = failed = 0

    for i, rec in enumerate(records, start=1):
        out = cache_dir / cache_key(rec.track_id, sample_rate, precompute_duration, n_mels, n_fft, hop_length)
        if out.exists() and not overwrite:
            skipped += 1
            continue

        ap = track_path(audio_dir, rec.track_id)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
                warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
                y, _ = librosa.load(
                    str(ap),
                    sr=sample_rate,
                    mono=True,
                    duration=precompute_duration,
                    res_type="kaiser_fast",
                )
            if y is None or y.shape[0] == 0:
                raise ValueError("Decoded empty audio array")
            mel = mel_from_audio(y, sample_rate, target_samples, n_mels, n_fft, hop_length)
            np.save(out, mel)
            written += 1
        except Exception as exc:
            failed += 1
            if out.exists():
                out.unlink()
            if failed <= 10 or failed % 50 == 0:
                print(f"Warning: failed {rec.track_id} ({ap}): {exc}; skipping this track.")

        if i % log_every == 0 or i == len(records):
            print(f"[{i}/{len(records)}] written={written} skipped={skipped} failed={failed}")

    print(f"Precompute done: written={written} skipped={skipped} failed={failed} total={len(records)}")


def cache_status(
    records: Sequence[TrackRecord],
    cache_dir: Path,
    sample_rate: int,
    precompute_duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> Tuple[int, int, List[int]]:
    missing: List[int] = []
    for rec in records:
        p = cache_dir / cache_key(rec.track_id, sample_rate, precompute_duration, n_mels, n_fft, hop_length)
        if not p.exists():
            missing.append(rec.track_id)
    return len(records), len(missing), missing[:10]


def apply_spec_augment(x: torch.Tensor, time_mask_max: int, freq_mask_max: int) -> torch.Tensor:
    x = x.clone()
    _, f, t = x.shape
    if time_mask_max > 0 and t > 2:
        w = random.randint(0, min(time_mask_max, max(1, t // 4)))
        if w > 0:
            s = random.randint(0, t - w)
            x[:, :, s : s + w] = 0
    if freq_mask_max > 0 and f > 2:
        w = random.randint(0, min(freq_mask_max, max(1, f // 4)))
        if w > 0:
            s = random.randint(0, f - w)
            x[:, s : s + w, :] = 0
    return x


class MelCropDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TrackRecord],
        cache_dir: Path,
        label_to_idx: Dict[str, int],
        crop_duration: float,
        sample_rate: int,
        hop_length: int,
        precompute_duration: float,
        n_mels: int,
        n_fft: int,
        train_mode: bool,
        use_aug: bool,
        time_mask_max: int,
        freq_mask_max: int,
    ) -> None:
        self.records = list(records)
        self.cache_dir = cache_dir
        self.label_to_idx = label_to_idx
        self.crop_frames = int(round((crop_duration * sample_rate) / hop_length))
        self.sample_rate = sample_rate
        self.precompute_duration = precompute_duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.train_mode = train_mode
        self.use_aug = use_aug
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        p = self.cache_dir / cache_key(
            rec.track_id,
            self.sample_rate,
            self.precompute_duration,
            self.n_mels,
            self.n_fft,
            self.hop_length,
        )
        mel = np.load(p).astype(np.float32)
        t = mel.shape[1]
        c = self.crop_frames

        if t < c:
            pad = np.zeros((mel.shape[0], c - t), dtype=np.float32)
            crop = np.concatenate([mel, pad], axis=1)
        elif t == c:
            crop = mel
        else:
            if self.train_mode:
                start = random.randint(0, t - c)
            else:
                start = (t - c) // 2
            crop = mel[:, start : start + c]

        x = torch.from_numpy(crop).unsqueeze(0)
        if self.train_mode and self.use_aug:
            x = apply_spec_augment(x, self.time_mask_max, self.freq_mask_max)
        y = self.label_to_idx[rec.genre]
        return x, y


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.proj = None
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class GenreResCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 192, stride=2)
        self.layer4 = ResidualBlock(192, 256, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


class GenreTheirCNN2(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 1.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()


def class_weights(records: Sequence[TrackRecord], label_to_idx: Dict[str, int], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(label_to_idx), dtype=np.float64)
    for r in records:
        counts[label_to_idx[r.genre]] += 1
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def class_priors(records: Sequence[TrackRecord], label_to_idx: Dict[str, int]) -> np.ndarray:
    counts = np.zeros(len(label_to_idx), dtype=np.float64)
    for r in records:
        counts[label_to_idx[r.genre]] += 1
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()


def parse_float_list(spec: str) -> List[float]:
    out: List[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("Expected at least one float value.")
    return out


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict[str, float]:
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
    }


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler | None,
    use_channels_last: bool,
) -> Tuple[float, List[int], List[int]]:
    model.train(True)
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    for xb, yb in loader:
        if use_channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        yb = yb.to(device, non_blocking=(device.type == "cuda"))

        if amp_enabled:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True):
                logits = model(xb)
                loss = criterion(logits, yb)
        else:
            logits = model(xb)
            loss = criterion(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(yb.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return total_loss / max(1, len(loader.dataset)), y_true, y_pred


def predict_multicrop(
    model: nn.Module,
    rec: TrackRecord,
    cache_dir: Path,
    device: torch.device,
    sample_rate: int,
    precompute_duration: float,
    eval_crop_duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    num_crops: int,
    use_channels_last: bool,
) -> torch.Tensor:
    p = cache_dir / cache_key(rec.track_id, sample_rate, precompute_duration, n_mels, n_fft, hop_length)
    mel = np.load(p).astype(np.float32)
    t = mel.shape[1]
    c = int(round((eval_crop_duration * sample_rate) / hop_length))

    if t <= c:
        starts = [0]
    else:
        starts = np.linspace(0, t - c, num=max(1, num_crops), dtype=int).tolist()

    batch = []
    for s in starts:
        crop = mel[:, s : s + c]
        if crop.shape[1] < c:
            pad = np.zeros((mel.shape[0], c - crop.shape[1]), dtype=np.float32)
            crop = np.concatenate([crop, pad], axis=1)
        batch.append(torch.from_numpy(crop).unsqueeze(0))

    xb = torch.stack(batch, dim=0)
    if use_channels_last:
        xb = xb.contiguous(memory_format=torch.channels_last)
    xb = xb.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(xb)
        mean_logits = logits.mean(dim=0)
    return mean_logits


def predict_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    class_bias: np.ndarray | None = None,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = logits / float(temperature)
    if class_bias is not None:
        z = z + class_bias[None, :]
    return z.argmax(axis=1).astype(np.int64)


def metrics_from_logits(
    logits: np.ndarray,
    y_true: Sequence[int],
    num_classes: int,
    temperature: float = 1.0,
    class_bias: np.ndarray | None = None,
) -> Tuple[Dict[str, float], List[int]]:
    pred = predict_from_logits(logits, temperature=temperature, class_bias=class_bias)
    metrics = compute_metrics(y_true, pred.tolist(), num_classes)
    return metrics, pred.tolist()


def tune_posthoc(
    logits: np.ndarray,
    y_true: Sequence[int],
    objective: str,
    temp_values: Sequence[float],
    prior_tau_values: Sequence[float],
    log_priors: np.ndarray,
    bias_iters: int,
    bias_step: float,
    bias_step_decay: float,
) -> Tuple[float, float, np.ndarray, Dict[str, float]]:
    num_classes = logits.shape[1]
    best_temp = 1.0
    best_tau = 0.0
    best_bias = np.zeros(num_classes, dtype=np.float32)
    best_metrics, _ = metrics_from_logits(logits, y_true, num_classes)
    best_score = best_metrics[objective]

    for t in temp_values:
        for tau in prior_tau_values:
            bias = (float(tau) * log_priors).astype(np.float32)
            m, _ = metrics_from_logits(logits, y_true, num_classes, temperature=t, class_bias=bias)
            s = m[objective]
            if s > best_score:
                best_score = s
                best_temp = float(t)
                best_tau = float(tau)
                best_bias = bias
                best_metrics = m

    step = float(bias_step)
    current_bias = best_bias.copy()
    for _ in range(max(0, bias_iters)):
        improved = False
        for c in range(num_classes):
            for delta in (step, -step):
                cand = current_bias.copy()
                cand[c] += float(delta)
                m, _ = metrics_from_logits(logits, y_true, num_classes, temperature=best_temp, class_bias=cand)
                s = m[objective]
                if s > best_score:
                    best_score = s
                    best_bias = cand
                    current_bias = cand
                    best_metrics = m
                    improved = True
        step *= float(bias_step_decay)
        if not improved and step < 1e-4:
            break

    return best_temp, best_tau, best_bias, best_metrics


def collect_multicrop_logits(
    model: nn.Module,
    records: Sequence[TrackRecord],
    label_to_idx: Dict[str, int],
    cache_dir: Path,
    device: torch.device,
    sample_rate: int,
    precompute_duration: float,
    eval_crop_duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    num_crops: int,
    use_channels_last: bool,
    log_every: int,
) -> Tuple[np.ndarray, List[int]]:
    logits_rows: List[np.ndarray] = []
    y_true: List[int] = []

    for i, rec in enumerate(records, start=1):
        mean_logits = predict_multicrop(
            model,
            rec,
            cache_dir,
            device,
            sample_rate,
            precompute_duration,
            eval_crop_duration,
            n_mels,
            n_fft,
            hop_length,
            num_crops,
            use_channels_last,
        )
        logits_rows.append(mean_logits.detach().cpu().numpy().astype(np.float32))
        y_true.append(label_to_idx[rec.genre])
        if i % log_every == 0 or i == len(records):
            print(f"eval progress: {i}/{len(records)}")

    return np.stack(logits_rows, axis=0), y_true


def evaluate_multicrop(
    model: nn.Module,
    records: Sequence[TrackRecord],
    label_to_idx: Dict[str, int],
    cache_dir: Path,
    device: torch.device,
    sample_rate: int,
    precompute_duration: float,
    eval_crop_duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    num_crops: int,
    use_channels_last: bool,
    log_every: int,
    temperature: float = 1.0,
    class_bias: np.ndarray | None = None,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    logits, y_true = collect_multicrop_logits(
        model,
        records,
        label_to_idx,
        cache_dir,
        device,
        sample_rate,
        precompute_duration,
        eval_crop_duration,
        n_mels,
        n_fft,
        hop_length,
        num_crops,
        use_channels_last,
        log_every,
    )
    metrics, y_pred = metrics_from_logits(
        logits,
        y_true,
        len(label_to_idx),
        temperature=temperature,
        class_bias=class_bias,
    )
    return metrics, y_true, y_pred


def save_curves(history: pd.DataFrame, out_dir: Path) -> None:
    if history.empty:
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["epoch"], history["train_f1_macro"], label="train_f1_macro")
    ax[0].plot(history["epoch"], history["val_f1_macro"], label="val_f1_macro")
    ax[0].set_title("Macro F1")
    ax[0].legend()

    ax[1].plot(history["epoch"], history["val_precision_macro"], label="val_precision_macro")
    ax[1].plot(history["epoch"], history["val_recall_macro"], label="val_recall_macro")
    ax[1].set_title("Macro Precision / Recall")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "metric_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = choose_device(args.device)
    runtime = optimize_runtime(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(runtime["use_amp"] and device.type == "cuda"))

    records = load_records(args.metadata_path, args.audio_dir, min_audio_file_bytes=args.min_audio_file_bytes)
    records = drop_known_bad_audio(records, args.bad_audio_scan_path, enabled=args.exclude_bad_audio_scan)
    if not records:
        raise RuntimeError("No labeled tracks found after filtering.")

    if args.split_strategy == "official":
        train_records, val_records, test_records = split_records_official(records)
        dropped_classes: Dict[str, int] = {}
    else:
        train_records, val_records, test_records, dropped_classes = split_records_stratified(
            records,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_class_count=args.min_class_count_for_split,
        )

    train_records = maybe_limit(train_records, args.max_train, args.seed)
    val_records = maybe_limit(val_records, args.max_val, args.seed + 1)
    test_records = maybe_limit(test_records, args.max_test, args.seed + 2)

    train_records, val_records, test_records, support_policy = apply_train_support_policy(
        train_records,
        val_records,
        test_records,
        min_train_count=args.min_train_count_for_model,
        merge_rare=args.merge_rare_classes,
        rare_class_name=args.rare_class_name,
    )

    print(f"split_strategy={args.split_strategy}")
    print(f"train={len(train_records)} val={len(val_records)} test={len(test_records)}")
    if dropped_classes:
        print("Dropped low-count classes for stratified splitting:")
        print(pd.Series(dropped_classes).sort_values().to_string())
    if support_policy.get("rare_classes"):
        print(f"Rare classes in train (<{args.min_train_count_for_model}):")
        print(pd.Series(support_policy["rare_classes"]).sort_values().to_string())
    if support_policy.get("merged_to") is not None:
        print(f"Merged rare classes into: {support_policy['merged_to']}")
    if support_policy.get("dropped_after_policy"):
        print("Dropped classes after train-support policy:")
        print(pd.Series(support_policy["dropped_after_policy"]).sort_values().to_string())

    print_split_support(train_records, val_records, test_records, args.min_train_count_for_model)

    precompute_cache(
        [*train_records, *val_records, *test_records],
        args.audio_dir,
        args.cache_dir,
        args.sample_rate,
        args.precompute_duration,
        args.n_mels,
        args.n_fft,
        args.hop_length,
        overwrite=args.precompute_overwrite,
        log_every=args.eval_log_every,
    )

    for split_name, rs in [("train", train_records), ("val", val_records), ("test", test_records)]:
        total, miss, prev = cache_status(
            rs,
            args.cache_dir,
            args.sample_rate,
            args.precompute_duration,
            args.n_mels,
            args.n_fft,
            args.hop_length,
        )
        print(f"{split_name}: missing={miss} / {total}; preview={prev}")

    present = set(int(p.name.split("_")[0]) for p in args.cache_dir.glob("*.npy"))
    train_records = [r for r in train_records if r.track_id in present]
    val_records = [r for r in val_records if r.track_id in present]
    test_records = [r for r in test_records if r.track_id in present]

    print("\nAfter removing missing-cache tracks:")
    print(f"train={len(train_records)} val={len(val_records)} test={len(test_records)}")
    print_split_support(train_records, val_records, test_records, args.min_train_count_for_model)

    all_genres = sorted({r.genre for r in [*train_records, *val_records, *test_records]})
    label_to_idx = {g: i for i, g in enumerate(all_genres)}
    idx_to_label = {i: g for g, i in label_to_idx.items()}

    for split_name, rs in [("train", train_records), ("val", val_records), ("test", test_records)]:
        total, miss, prev = cache_status(
            rs,
            args.cache_dir,
            args.sample_rate,
            args.precompute_duration,
            args.n_mels,
            args.n_fft,
            args.hop_length,
        )
        if miss > 0:
            raise FileNotFoundError(f"Cache incomplete: {split_name} missing={miss}, preview={prev}")

    effective_use_aug = args.use_aug
    if args.model_arch == "their_cnn2" and args.their_recipe_disable_spec_aug:
        effective_use_aug = False

    train_ds = MelCropDataset(
        train_records,
        args.cache_dir,
        label_to_idx,
        args.train_crop_duration,
        args.sample_rate,
        args.hop_length,
        args.precompute_duration,
        args.n_mels,
        args.n_fft,
        train_mode=True,
        use_aug=effective_use_aug,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
    )

    if args.use_balanced_sampler:
        counts = np.zeros(len(label_to_idx), dtype=np.float64)
        for r in train_records:
            counts[label_to_idx[r.genre]] += 1
        sample_weights = [1.0 / counts[label_to_idx[r.genre]] for r in train_records]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_records), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    if args.model_arch == "res_cnn":
        model = GenreResCNN(num_classes=len(all_genres)).to(device)
    else:
        model = GenreTheirCNN2(num_classes=len(all_genres)).to(device)
    if runtime["use_channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    cw = class_weights(train_records, label_to_idx, device)

    if args.use_balanced_sampler:
        loss_alpha = None
        ce_weight = None
    else:
        loss_alpha = cw
        ce_weight = cw

    criterion = FocalLoss(alpha=loss_alpha, gamma=args.focal_gamma) if args.use_focal_loss else nn.CrossEntropyLoss(weight=ce_weight)
    effective_lr = args.their_recipe_lr if args.model_arch == "their_cnn2" else args.lr
    optimizer = optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    history: List[Dict[str, float]] = []
    best_score = -1.0
    best_ckpt = args.output_dir / "best_model.pt"
    no_improve = 0

    print("Using device:", device)
    print("Runtime:", runtime)
    print("Classes:", len(all_genres))
    print("MODEL_ARCH:", args.model_arch)
    print("effective_lr:", effective_lr)
    print("effective_use_aug:", effective_use_aug)

    for epoch in range(1, args.epochs + 1):
        tr_loss, ytr, ptr = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            runtime["use_amp"],
            runtime["amp_dtype"],
            scaler,
            runtime["use_channels_last"],
        )
        tr_metrics = compute_metrics(ytr, ptr, len(all_genres))

        val_metrics, _, _ = evaluate_multicrop(
            model,
            val_records,
            label_to_idx,
            args.cache_dir,
            device,
            args.sample_rate,
            args.precompute_duration,
            args.eval_crop_duration,
            args.n_mels,
            args.n_fft,
            args.hop_length,
            args.val_multi_crops,
            runtime["use_channels_last"],
            args.eval_log_every,
        )

        scheduler.step(val_metrics["f1_macro"])

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_accuracy": tr_metrics["accuracy"],
            "train_precision_macro": tr_metrics["precision_macro"],
            "train_recall_macro": tr_metrics["recall_macro"],
            "train_f1_macro": tr_metrics["f1_macro"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_precision_weighted": val_metrics["precision_weighted"],
            "val_recall_weighted": val_metrics["recall_weighted"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        score = row[args.metric_for_best]
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_f1={row['train_f1_macro']:.4f} "
            f"val_f1={row['val_f1_macro']:.4f} "
            f"val_prec={row['val_precision_macro']:.4f} "
            f"val_rec={row['val_recall_macro']:.4f}"
        )

        if score > best_score:
            best_score = score
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "config": {
                        "sample_rate": args.sample_rate,
                        "precompute_duration": args.precompute_duration,
                        "eval_crop_duration": args.eval_crop_duration,
                        "n_mels": args.n_mels,
                        "n_fft": args.n_fft,
                        "hop_length": args.hop_length,
                        "val_multi_crops": args.val_multi_crops,
                        "test_multi_crops": args.test_multi_crops,
                        "model_arch": args.model_arch,
                        "use_aug": effective_use_aug,
                    },
                    "epoch": epoch,
                    "best_metric": args.metric_for_best,
                    "best_score": best_score,
                    "val_metrics": val_metrics,
                },
                best_ckpt,
            )
        else:
            no_improve += 1

        if no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(args.output_dir / "history.csv", index=False)
    save_curves(hist_df, args.output_dir)
    print("Training complete. Best score:", best_score)

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    ckpt_label_to_idx = ckpt["label_to_idx"]
    ckpt_idx_to_label = ckpt.get("idx_to_label", {v: k for k, v in ckpt_label_to_idx.items()})
    if ckpt_label_to_idx != label_to_idx:
        raise RuntimeError("Label mapping mismatch between current session and checkpoint.")

    val_metrics, _, _ = evaluate_multicrop(
        model,
        val_records,
        ckpt_label_to_idx,
        args.cache_dir,
        device,
        args.sample_rate,
        args.precompute_duration,
        args.eval_crop_duration,
        args.n_mels,
        args.n_fft,
        args.hop_length,
        args.val_multi_crops,
        runtime["use_channels_last"],
        args.eval_log_every,
    )
    test_metrics, y_te, p_te = evaluate_multicrop(
        model,
        test_records,
        ckpt_label_to_idx,
        args.cache_dir,
        device,
        args.sample_rate,
        args.precompute_duration,
        args.eval_crop_duration,
        args.n_mels,
        args.n_fft,
        args.hop_length,
        args.test_multi_crops,
        runtime["use_channels_last"],
        args.eval_log_every,
    )

    cm = confusion_matrix(y_te, p_te, labels=list(range(len(ckpt_label_to_idx))))
    np.save(args.output_dir / "test_confusion_matrix.npy", cm)

    summary = {
        "best_metric": ckpt["best_metric"],
        "best_score": ckpt["best_score"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(best_ckpt),
    }

    if args.posthoc_tune:
        val_logits, y_val = collect_multicrop_logits(
            model,
            val_records,
            ckpt_label_to_idx,
            args.cache_dir,
            device,
            args.sample_rate,
            args.precompute_duration,
            args.eval_crop_duration,
            args.n_mels,
            args.n_fft,
            args.hop_length,
            args.val_multi_crops,
            runtime["use_channels_last"],
            args.eval_log_every,
        )
        test_logits, y_test = collect_multicrop_logits(
            model,
            test_records,
            ckpt_label_to_idx,
            args.cache_dir,
            device,
            args.sample_rate,
            args.precompute_duration,
            args.eval_crop_duration,
            args.n_mels,
            args.n_fft,
            args.hop_length,
            args.test_multi_crops,
            runtime["use_channels_last"],
            args.eval_log_every,
        )

        train_priors = class_priors(train_records, ckpt_label_to_idx)
        log_priors = np.log(np.clip(train_priors, 1e-8, 1.0)).astype(np.float32)
        temp_values = parse_float_list(args.posthoc_temp_values)
        prior_tau_values = parse_float_list(args.posthoc_prior_tau_values)

        best_temp, best_tau, best_bias, posthoc_val_metrics = tune_posthoc(
            val_logits,
            y_val,
            objective=args.posthoc_objective,
            temp_values=temp_values,
            prior_tau_values=prior_tau_values,
            log_priors=log_priors,
            bias_iters=args.posthoc_bias_iters,
            bias_step=args.posthoc_bias_step,
            bias_step_decay=args.posthoc_bias_step_decay,
        )
        posthoc_test_metrics, p_te_posthoc = metrics_from_logits(
            test_logits,
            y_test,
            len(ckpt_label_to_idx),
            temperature=best_temp,
            class_bias=best_bias,
        )

        print("POSTHOC_VAL:", posthoc_val_metrics)
        print("POSTHOC_TEST:", posthoc_test_metrics)
        print(f"POSTHOC_PARAMS: temp={best_temp:.4f} prior_tau={best_tau:.4f}")
        print("Classification report (test, posthoc):")
        print(
            classification_report(
                y_test,
                p_te_posthoc,
                labels=list(range(len(ckpt_label_to_idx))),
                target_names=[ckpt_idx_to_label[i] for i in range(len(ckpt_label_to_idx))],
                digits=4,
                zero_division=0,
            )
        )

        cm_posthoc = confusion_matrix(y_test, p_te_posthoc, labels=list(range(len(ckpt_label_to_idx))))
        np.save(args.output_dir / "test_confusion_matrix_posthoc.npy", cm_posthoc)

        summary["posthoc"] = {
            "objective": args.posthoc_objective,
            "temperature": best_temp,
            "prior_tau": best_tau,
            "class_bias": [float(x) for x in best_bias.tolist()],
            "val_metrics": posthoc_val_metrics,
            "test_metrics": posthoc_test_metrics,
        }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("VAL:", val_metrics)
    print("TEST:", test_metrics)
    print("Classification report (test):")
    print(
        classification_report(
            y_te,
            p_te,
            labels=list(range(len(ckpt_label_to_idx))),
            target_names=[ckpt_idx_to_label[i] for i in range(len(ckpt_label_to_idx))],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
