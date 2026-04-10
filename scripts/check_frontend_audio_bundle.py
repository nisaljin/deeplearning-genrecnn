#!/usr/bin/env python3
"""Validate frontend bundled audio limits and manifest consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check frontend validation audio bundle guardrails.")
    p.add_argument("--bundle-dir", type=Path, default=Path("frontend/public/validation-audio"))
    p.add_argument("--max-bytes", type=int, default=25 * 1024 * 1024)
    p.add_argument("--max-files", type=int, default=24)
    return p.parse_args()


def fail(msg: str) -> int:
    print(f"ERROR: {msg}")
    return 1


def main() -> int:
    args = parse_args()
    bundle_dir = args.bundle_dir
    manifest_path = bundle_dir / "manifest.json"

    if not bundle_dir.exists():
        return fail(f"Bundle directory does not exist: {bundle_dir}")
    if not manifest_path.exists():
        return fail(f"Missing manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"Invalid manifest JSON: {exc}")

    if not isinstance(manifest, list):
        return fail("Manifest must be a JSON array.")

    audio_files = sorted(bundle_dir.glob("*.mp3"))
    audio_names = {p.name for p in audio_files}
    total_bytes = sum(p.stat().st_size for p in audio_files)

    if total_bytes > args.max_bytes:
        return fail(f"Bundle size {total_bytes} exceeds max-bytes {args.max_bytes}.")
    if len(audio_files) > args.max_files:
        return fail(f"Bundle file count {len(audio_files)} exceeds max-files {args.max_files}.")
    if len(manifest) != len(audio_files):
        return fail(f"Manifest entries ({len(manifest)}) do not match mp3 files ({len(audio_files)}).")

    seen_track_ids = set()
    manifest_names = set()
    for idx, entry in enumerate(manifest):
        if not isinstance(entry, dict):
            return fail(f"Manifest entry at index {idx} is not an object.")
        track_id = entry.get("trackId")
        filename = entry.get("filename")
        genre = entry.get("genre")
        split = entry.get("split")

        if not isinstance(track_id, int):
            return fail(f"Manifest entry {idx} has invalid trackId: {track_id}")
        if track_id in seen_track_ids:
            return fail(f"Duplicate trackId in manifest: {track_id}")
        seen_track_ids.add(track_id)

        if not isinstance(filename, str) or not filename.endswith(".mp3"):
            return fail(f"Manifest entry {idx} has invalid filename: {filename}")
        if filename not in audio_names:
            return fail(f"Manifest references missing file: {filename}")
        manifest_names.add(filename)

        if split != "validation":
            return fail(f"Manifest entry {idx} split must be 'validation', got: {split}")
        if not isinstance(genre, str) or not genre.strip():
            return fail(f"Manifest entry {idx} has invalid genre: {genre}")

    extras = audio_names - manifest_names
    if extras:
        return fail(f"Bundle contains unreferenced mp3 files: {sorted(extras)}")

    print(
        json.dumps(
            {
                "status": "ok",
                "bundle_dir": str(bundle_dir),
                "tracks": len(audio_files),
                "total_bytes": total_bytes,
                "max_bytes": args.max_bytes,
                "max_files": args.max_files,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
