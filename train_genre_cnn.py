#!/usr/bin/env python3
"""Train a genre classification CNN on FMA-large using official metadata splits.

This script is designed for reproducible coursework experiments:
- deterministic seed setup
- fixed train/validation/test split from FMA metadata
- weighted loss for class imbalance
- baseline comparison (majority class)
- metrics: accuracy, precision, recall, F1
- saved logs/plots/checkpoints
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset


SPLIT_TRAIN = "training"
SPLIT_VAL = "validation"
SPLIT_TEST = "test"


@dataclass
class TrackRecord:
    track_id: int
    genre: str
    split: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN on FMA-large genre labels")
    parser.add_argument("--audio-dir", type=Path, default=Path("fma_large"))
    parser.add_argument("--metadata-path", type=Path, default=Path("fma_metadata/tracks.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path, default=None)

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--clip-duration", type=float, default=29.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("Requested --device mps, but MPS is not available.")
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

    # Global matmul optimization for modern PyTorch backends.
    torch.set_float32_matmul_precision("high")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        use_amp = True
        amp_dtype = torch.float16
        use_channels_last = True
    elif device.type == "mps":
        use_amp = False
        amp_dtype = None
        use_channels_last = True

    return {
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
        "use_channels_last": use_channels_last,
    }


def track_path(audio_dir: Path, track_id: int) -> Path:
    return audio_dir / f"{track_id:06d}"[:3] / f"{track_id:06d}.mp3"


def load_records(metadata_path: Path, audio_dir: Path) -> List[TrackRecord]:
    df = pd.read_csv(metadata_path, header=[0, 1], index_col=0, low_memory=False)
    df.index = df.index.astype(int)

    subset_mask = df[("set", "subset")] == "large"
    labeled_mask = df[("track", "genre_top")].notna()
    split_mask = df[("set", "split")].isin([SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST])
    filtered = df[subset_mask & labeled_mask & split_mask]

    records: List[TrackRecord] = []
    for track_id, row in filtered.iterrows():
        audio_path = track_path(audio_dir, int(track_id))
        if not audio_path.exists():
            continue
        records.append(
            TrackRecord(
                track_id=int(track_id),
                genre=str(row[("track", "genre_top")]),
                split=str(row[("set", "split")]),
            )
        )
    return records


class FMAMelDataset(Dataset):
    def __init__(
        self,
        records: Sequence[TrackRecord],
        audio_dir: Path,
        label_to_idx: Dict[str, int],
        sample_rate: int,
        clip_duration: float,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        cache_dir: Path | None,
    ) -> None:
        self.records = list(records)
        self.audio_dir = audio_dir
        self.label_to_idx = label_to_idx
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.target_samples = int(sample_rate * clip_duration)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cache_dir = cache_dir

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _cache_key(self, track_id: int) -> str:
        return (
            f"{track_id:06d}_sr{self.sample_rate}_dur{self.clip_duration}_"
            f"mel{self.n_mels}_fft{self.n_fft}_hop{self.hop_length}.npy"
        )

    def _feature_from_audio(self, y: np.ndarray) -> np.ndarray:
        if y.shape[0] < self.target_samples:
            y = np.pad(y, (0, self.target_samples - y.shape[0]))
        elif y.shape[0] > self.target_samples:
            y = y[: self.target_samples]

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=20,
            fmax=self.sample_rate // 2,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db.astype(np.float32)

        mean = float(mel_db.mean())
        std = float(mel_db.std())
        mel_norm = (mel_db - mean) / (std + 1e-6)
        return mel_norm

    def _load_feature(self, rec: TrackRecord) -> np.ndarray:
        audio_path = track_path(self.audio_dir, rec.track_id)

        if self.cache_dir is not None:
            cache_path = self.cache_dir / self._cache_key(rec.track_id)
            if cache_path.exists():
                return np.load(cache_path)

        y, _ = librosa.load(
            str(audio_path),
            sr=self.sample_rate,
            mono=True,
            duration=self.clip_duration,
            res_type="kaiser_fast",
        )
        feature = self._feature_from_audio(y)

        if self.cache_dir is not None:
            np.save(cache_path, feature)

        return feature

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        feature = self._load_feature(rec)
        x = torch.from_numpy(feature).unsqueeze(0)  # [1, n_mels, time]
        y = self.label_to_idx[rec.genre]
        return x, y


class GenreCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def split_records(records: Sequence[TrackRecord]) -> Tuple[List[TrackRecord], List[TrackRecord], List[TrackRecord]]:
    train = [r for r in records if r.split == SPLIT_TRAIN]
    val = [r for r in records if r.split == SPLIT_VAL]
    test = [r for r in records if r.split == SPLIT_TEST]
    return train, val, test


def maybe_limit(records: List[TrackRecord], max_items: int | None, seed: int) -> List[TrackRecord]:
    if max_items is None or len(records) <= max_items:
        return records
    rng = random.Random(seed)
    sampled = records.copy()
    rng.shuffle(sampled)
    return sampled[:max_items]


def class_weights(train_records: Sequence[TrackRecord], label_to_idx: Dict[str, int], device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(label_to_idx), dtype=np.float64)
    for rec in train_records:
        counts[label_to_idx[rec.genre]] += 1
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler | None,
    use_channels_last: bool,
) -> Tuple[float, List[int], List[int]]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            if use_channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))

            amp_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                if amp_enabled
                else nullcontext()
            )
            with amp_ctx:
                logits = model(xb)
                loss = criterion(logits, yb)

            if train_mode:
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

    mean_loss = total_loss / max(1, len(loader.dataset))
    return mean_loss, y_true, y_pred


def evaluate_majority_baseline(
    train_records: Sequence[TrackRecord],
    eval_records: Sequence[TrackRecord],
    label_to_idx: Dict[str, int],
) -> Dict[str, float]:
    label_counts: Dict[str, int] = {}
    for rec in train_records:
        label_counts[rec.genre] = label_counts.get(rec.genre, 0) + 1
    majority_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
    majority_idx = label_to_idx[majority_label]

    y_true = [label_to_idx[r.genre] for r in eval_records]
    y_pred = [majority_idx] * len(eval_records)
    return compute_metrics(y_true, y_pred, num_classes=len(label_to_idx))


def save_curves(history: List[Dict[str, float]], out_dir: Path) -> None:
    if not history:
        return

    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_accuracy"] for h in history], label="train_acc")
    plt.plot(epochs, [h["val_accuracy"] for h in history], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curves.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = choose_device(args.device)
    runtime = optimize_runtime(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(runtime["use_amp"] and device.type == "cuda"))

    records = load_records(args.metadata_path, args.audio_dir)
    if not records:
        raise RuntimeError("No labeled tracks found. Check dataset paths and metadata.")

    genres = sorted({r.genre for r in records})
    label_to_idx = {label: i for i, label in enumerate(genres)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    train_records, val_records, test_records = split_records(records)

    train_records = maybe_limit(train_records, args.max_train, seed=args.seed)
    val_records = maybe_limit(val_records, args.max_val, seed=args.seed + 1)
    test_records = maybe_limit(test_records, args.max_test, seed=args.seed + 2)

    print(f"Using device: {device}")
    print(
        "Runtime optimizations:",
        {
            "amp": runtime["use_amp"],
            "amp_dtype": str(runtime["amp_dtype"]),
            "channels_last": runtime["use_channels_last"],
        },
    )
    print(f"Classes ({len(genres)}): {genres}")
    print(f"Samples: train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    with open(args.output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f, indent=2)

    train_ds = FMAMelDataset(
        records=train_records,
        audio_dir=args.audio_dir,
        label_to_idx=label_to_idx,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        cache_dir=args.cache_dir,
    )
    val_ds = FMAMelDataset(
        records=val_records,
        audio_dir=args.audio_dir,
        label_to_idx=label_to_idx,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        cache_dir=args.cache_dir,
    )
    test_ds = FMAMelDataset(
        records=test_records,
        audio_dir=args.audio_dir,
        label_to_idx=label_to_idx,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        cache_dir=args.cache_dir,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = GenreCNN(num_classes=len(genres)).to(device)
    if runtime["use_channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    weights = class_weights(train_records, label_to_idx, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    history: List[Dict[str, float]] = []
    best_val_f1 = -math.inf
    best_ckpt_path = args.output_dir / "best_model.pt"

    baseline_val = evaluate_majority_baseline(train_records, val_records, label_to_idx)
    baseline_test = evaluate_majority_baseline(train_records, test_records, label_to_idx)
    print("Majority baseline (val):", baseline_val)
    print("Majority baseline (test):", baseline_test)

    for epoch in range(1, args.epochs + 1):
        train_loss, y_tr, p_tr = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            amp_enabled=bool(runtime["use_amp"]),
            amp_dtype=runtime["amp_dtype"],
            scaler=scaler,
            use_channels_last=bool(runtime["use_channels_last"]),
        )
        val_loss, y_va, p_va = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            amp_enabled=bool(runtime["use_amp"]),
            amp_dtype=runtime["amp_dtype"],
            scaler=None,
            use_channels_last=bool(runtime["use_channels_last"]),
        )

        train_metrics = compute_metrics(y_tr, p_tr, num_classes=len(genres))
        val_metrics = compute_metrics(y_va, p_va, num_classes=len(genres))
        scheduler.step(val_metrics["f1_macro"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "train_precision_macro": train_metrics["precision_macro"],
            "val_precision_macro": val_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_f1={train_metrics['f1_macro']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_ckpt_path,
            )

    with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    save_curves(history, args.output_dir)

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, y_te, p_te = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer=None,
        device=device,
        amp_enabled=bool(runtime["use_amp"]),
        amp_dtype=runtime["amp_dtype"],
        scaler=None,
        use_channels_last=bool(runtime["use_channels_last"]),
    )
    test_metrics = compute_metrics(y_te, p_te, num_classes=len(genres))

    print(f"Test loss: {test_loss:.4f}")
    print("Test metrics:", test_metrics)
    print("Classification report:\n")
    print(
        classification_report(
            y_te,
            p_te,
            labels=list(range(len(genres))),
            target_names=[idx_to_label[i] for i in range(len(genres))],
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_te, p_te, labels=list(range(len(genres))))
    np.save(args.output_dir / "test_confusion_matrix.npy", cm)

    summary = {
        "best_val_f1_macro": best_val_f1,
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "best_checkpoint": str(best_ckpt_path),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
