#!/usr/bin/env python3
"""Batch inference script for trained FMA genre CNN checkpoints."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GenreStandardCNN(nn.Module):
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


def build_model_from_checkpoint(config: dict, state_dict: dict, num_classes: int) -> nn.Module:
    arch = (config or {}).get("model_arch")
    if arch == "residual_cnn":
        return GenreResCNN(num_classes=num_classes)
    if arch == "standard_cnn":
        return GenreStandardCNN(num_classes=num_classes)

    # Backward-compatible fallback if model_arch is missing.
    if any(k.startswith("stem.") for k in state_dict.keys()):
        return GenreResCNN(num_classes=num_classes)
    if any(k.startswith("head.") for k in state_dict.keys()) or any(
        k.startswith("features.") for k in state_dict.keys()
    ):
        return GenreStandardCNN(num_classes=num_classes)
    return GenreCNN(num_classes=num_classes)


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


def optimize_runtime(device: torch.device) -> dict:
    torch.set_float32_matmul_precision("high")
    use_amp = False
    amp_dtype = None
    use_channels_last = False

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        use_amp = True
        amp_dtype = torch.float16
        use_channels_last = True
    elif device.type == "mps":
        use_channels_last = True

    return {
        "use_amp": use_amp,
        "amp_dtype": amp_dtype,
        "use_channels_last": use_channels_last,
    }


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
    runtime = optimize_runtime(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    label_to_idx = ckpt["label_to_idx"]
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    model = build_model_from_checkpoint(
        config=ckpt.get("config", {}),
        state_dict=ckpt["model_state_dict"],
        num_classes=len(label_to_idx),
    ).to(device)
    if runtime["use_channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x = build_feature(
        audio_path=args.audio_path,
        sample_rate=args.sample_rate,
        clip_duration=args.clip_duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    if runtime["use_channels_last"]:
        x = x.contiguous(memory_format=torch.channels_last)
    x = x.to(device, non_blocking=(device.type == "cuda"))

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=runtime["amp_dtype"], enabled=runtime["use_amp"])
        if runtime["use_amp"]
        else nullcontext()
    )
    with torch.inference_mode():
        with amp_ctx:
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
