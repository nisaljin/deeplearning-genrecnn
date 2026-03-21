#!/usr/bin/env python3
"""Batch inference script for trained FMA genre CNN checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn


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
        return self.classifier(self.features(x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict genre from an mp3 file")
    p.add_argument("audio_path", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--clip-duration", type=float, default=29.0)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_feature(
    audio_path: Path,
    sample_rate: int,
    clip_duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    target_samples = int(sample_rate * clip_duration)
    y, _ = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True,
        duration=clip_duration,
        res_type="kaiser_fast",
    )
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
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    model = GenreCNN(num_classes=len(label_to_idx)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = build_feature(
        audio_path=args.audio_path,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    ).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(args.top_k, probs.shape[0])
    top_probs, top_idx = torch.topk(probs, k=k)

    output = []
    for p, i in zip(top_probs.tolist(), top_idx.tolist()):
        output.append({"genre": idx_to_label[i], "probability": float(p)})

    print(json.dumps({"audio": str(args.audio_path), "predictions": output}, indent=2))


if __name__ == "__main__":
    main()
