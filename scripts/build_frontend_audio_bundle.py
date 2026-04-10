#!/usr/bin/env python3
"""Build a deterministic, genre-balanced frontend validation audio bundle.

This script curates a small starter set of validation tracks from FMA-large,
copies (or optionally transcodes) them into `frontend/public/validation-audio`,
and regenerates `manifest.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class TrackRecord:
    track_id: int
    genre: str
    split: str
    source_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic frontend audio bundle from FMA validation split.")
    p.add_argument("--audio-dir", type=Path, default=Path("fma_large"))
    p.add_argument("--metadata-path", type=Path, default=Path("fma_metadata/tracks.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("frontend/public/validation-audio"))
    p.add_argument("--max-tracks", type=int, default=24)
    p.add_argument("--max-per-genre", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-bad-audio-scan", action="store_true", default=True)
    p.add_argument("--no-exclude-bad-audio-scan", action="store_false", dest="exclude_bad_audio_scan")
    p.add_argument("--bad-audio-scan-path", type=Path, default=Path("outputs/bad_audio_scan.csv"))
    p.add_argument("--trim-seconds", type=float, default=0.0, help="If > 0, trim each output clip to this duration.")
    p.add_argument(
        "--normalize-loudness",
        action="store_true",
        default=False,
        help="If set, apply ffmpeg loudnorm (I=-16 LUFS) during transcode.",
    )
    p.add_argument(
        "--mp3-bitrate-kbps",
        type=int,
        default=0,
        help="If > 0, transcode audio with this MP3 bitrate. Requires ffmpeg.",
    )
    return p.parse_args()


def track_path(audio_dir: Path, track_id: int) -> Path:
    return audio_dir / f"{track_id:06d}"[:3] / f"{track_id:06d}.mp3"


def load_validation_candidates(args: argparse.Namespace) -> List[TrackRecord]:
    df = pd.read_csv(args.metadata_path, header=[0, 1], index_col=0, low_memory=False)
    df.index = df.index.astype(int)

    subset_mask = df[("set", "subset")] == "large"
    split_mask = df[("set", "split")] == "validation"
    labeled_mask = df[("track", "genre_top")].notna()
    filtered = df[subset_mask & split_mask & labeled_mask]

    bad_ids = set()
    if args.exclude_bad_audio_scan and args.bad_audio_scan_path.exists():
        bad = pd.read_csv(args.bad_audio_scan_path)
        if "track_id" in bad.columns:
            bad_ids = set(bad["track_id"].astype(int).tolist())

    out: List[TrackRecord] = []
    for track_id, row in filtered.iterrows():
        tid = int(track_id)
        if tid in bad_ids:
            continue
        src = track_path(args.audio_dir, tid)
        if not src.exists():
            continue
        out.append(
            TrackRecord(
                track_id=tid,
                genre=str(row[("track", "genre_top")]),
                split=str(row[("set", "split")]),
                source_path=src,
            )
        )
    return out


def select_balanced(records: List[TrackRecord], max_tracks: int, max_per_genre: int, seed: int) -> List[TrackRecord]:
    if max_tracks <= 0:
        return []

    by_genre: Dict[str, List[TrackRecord]] = defaultdict(list)
    for rec in records:
        by_genre[rec.genre].append(rec)

    rng = random.Random(seed)
    for genre in by_genre:
        by_genre[genre].sort(key=lambda r: r.track_id)
        rng.shuffle(by_genre[genre])

    genre_order = sorted(by_genre.keys())
    rng.shuffle(genre_order)
    picks_per_genre: Dict[str, int] = {g: 0 for g in genre_order}
    selected: List[TrackRecord] = []

    while len(selected) < max_tracks:
        made_progress = False
        for genre in genre_order:
            if len(selected) >= max_tracks:
                break
            if picks_per_genre[genre] >= max_per_genre:
                continue
            idx = picks_per_genre[genre]
            candidates = by_genre[genre]
            if idx >= len(candidates):
                continue
            selected.append(candidates[idx])
            picks_per_genre[genre] += 1
            made_progress = True
        if not made_progress:
            break

    selected.sort(key=lambda r: r.track_id)
    return selected


def run_ffmpeg(src: Path, dst: Path, args: argparse.Namespace) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn"]
    if args.trim_seconds > 0:
        cmd.extend(["-t", str(args.trim_seconds)])

    filters = []
    if args.normalize_loudness:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    if filters:
        cmd.extend(["-af", ",".join(filters)])

    bitrate = args.mp3_bitrate_kbps if args.mp3_bitrate_kbps > 0 else 128
    cmd.extend(["-codec:a", "libmp3lame", "-b:a", f"{bitrate}k", str(dst)])
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def should_transcode(args: argparse.Namespace) -> bool:
    return args.trim_seconds > 0 or args.normalize_loudness or args.mp3_bitrate_kbps > 0


def build_bundle(args: argparse.Namespace) -> int:
    candidates = load_validation_candidates(args)
    if not candidates:
        raise RuntimeError("No usable validation candidates found. Check dataset paths and metadata.")

    selected = select_balanced(candidates, args.max_tracks, args.max_per_genre, args.seed)
    if not selected:
        raise RuntimeError("No tracks selected. Check --max-tracks/--max-per-genre values.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    expected_filenames = set()
    transcode = should_transcode(args)

    if transcode and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for trim/normalize/transcode options but was not found in PATH.")

    for rec in selected:
        filename = f"{rec.track_id:06d}.mp3"
        expected_filenames.add(filename)
        dst = args.out_dir / filename
        if transcode:
            run_ffmpeg(rec.source_path, dst, args)
        else:
            shutil.copy2(rec.source_path, dst)
        manifest.append(
            {
                "trackId": rec.track_id,
                "split": rec.split,
                "genre": rec.genre,
                "filename": filename,
                "audioPath": f"/validation-audio/{filename}",
            }
        )

    for old_file in args.out_dir.glob("*.mp3"):
        if old_file.name not in expected_filenames:
            old_file.unlink()

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_bytes = sum(p.stat().st_size for p in args.out_dir.glob("*.mp3"))
    print(
        json.dumps(
            {
                "tracks_selected": len(manifest),
                "total_bytes": total_bytes,
                "out_dir": str(args.out_dir),
                "seed": args.seed,
                "max_per_genre": args.max_per_genre,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    return build_bundle(args)


if __name__ == "__main__":
    raise SystemExit(main())
