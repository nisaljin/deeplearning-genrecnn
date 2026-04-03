#!/usr/bin/env python3
"""Precompute mel spectrogram cache files for FMA-large tracks.

Output cache file names are intentionally aligned with train_genre_cnn.py:
  {track_id:06d}_sr{sample_rate}_dur{clip_duration}_mel{n_mels}_fft{n_fft}_hop{hop_length}.npy
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd


SPLIT_TRAIN = "training"
SPLIT_VAL = "validation"
SPLIT_TEST = "test"


@dataclass
class TrackRecord:
    track_id: int
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute mel cache for FMA-large tracks")
    parser.add_argument("--audio-dir", type=Path, default=Path("fma_large"))
    parser.add_argument("--metadata-path", type=Path, default=Path("fma_metadata/tracks.csv"))
    parser.add_argument("--cache-dir", type=Path, required=True)

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--clip-duration", type=float, default=29.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)

    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute mel files even if cache files already exist.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Optional cap for debugging/smoke tests.",
    )
    return parser.parse_args()


def track_path(audio_dir: Path, track_id: int) -> Path:
    return audio_dir / f"{track_id:06d}"[:3] / f"{track_id:06d}.mp3"


def cache_name(
    track_id: int,
    sample_rate: int,
    clip_duration: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> str:
    return (
        f"{track_id:06d}_sr{sample_rate}_dur{clip_duration}_"
        f"mel{n_mels}_fft{n_fft}_hop{hop_length}.npy"
    )


def load_records(metadata_path: Path, audio_dir: Path) -> List[TrackRecord]:
    df = pd.read_csv(metadata_path, header=[0, 1], index_col=0, low_memory=False)
    df.index = df.index.astype(int)

    subset_mask = df[("set", "subset")] == "large"
    labeled_mask = df[("track", "genre_top")].notna()
    split_mask = df[("set", "split")].isin([SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST])
    filtered = df[subset_mask & labeled_mask & split_mask]

    records: List[TrackRecord] = []
    for track_id, row in filtered.iterrows():
        audio = track_path(audio_dir, int(track_id))
        if audio.exists():
            records.append(TrackRecord(track_id=int(track_id), split=str(row[("set", "split")])))
    return records


def mel_from_audio(
    y: np.ndarray,
    sample_rate: int,
    target_samples: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
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


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    target_samples = int(args.sample_rate * args.clip_duration)

    records = load_records(args.metadata_path, args.audio_dir)
    if args.max_tracks is not None:
        records = records[: args.max_tracks]

    if not records:
        raise RuntimeError("No tracks found from metadata/audio path.")

    print(f"Precomputing mel cache into: {args.cache_dir}")
    print(
        f"Config: sr={args.sample_rate} dur={args.clip_duration} "
        f"n_mels={args.n_mels} n_fft={args.n_fft} hop={args.hop_length}"
    )
    print(f"Tracks to process: {len(records)}")

    started = time.time()
    done = 0
    skipped = 0
    failed = 0

    for i, rec in enumerate(records, start=1):
        audio_path = track_path(args.audio_dir, rec.track_id)
        out_path = args.cache_dir / cache_name(
            rec.track_id,
            args.sample_rate,
            args.clip_duration,
            args.n_mels,
            args.n_fft,
            args.hop_length,
        )

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
                warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
                y, _ = librosa.load(
                    str(audio_path),
                    sr=args.sample_rate,
                    mono=True,
                    duration=args.clip_duration,
                    res_type="kaiser_fast",
                )
            if y is None or y.shape[0] == 0:
                raise ValueError("Decoded empty audio array")
            mel = mel_from_audio(
                y=y,
                sample_rate=args.sample_rate,
                target_samples=target_samples,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            )
            np.save(out_path, mel)
            done += 1
        except Exception as exc:
            failed += 1
            if failed <= 10 or failed % 50 == 0:
                print(f"Warning: failed track {rec.track_id} ({audio_path}): {exc}")

        if i % args.log_every == 0 or i == len(records):
            elapsed = time.time() - started
            rate = i / max(elapsed, 1e-9)
            remaining = (len(records) - i) / max(rate, 1e-9)
            print(
                f"[{i}/{len(records)}] written={done} skipped={skipped} failed={failed} "
                f"elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m"
            )

    elapsed = time.time() - started
    print(
        f"Completed in {elapsed/60:.1f}m. "
        f"written={done} skipped={skipped} failed={failed} total={len(records)}"
    )


if __name__ == "__main__":
    main()
