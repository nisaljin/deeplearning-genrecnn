#!/usr/bin/env python3
"""HTTP inference API for the trained FMA genre CNN checkpoints."""

from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from predict_genre import (
    build_feature,
    build_model_from_checkpoint,
    choose_device,
    optimize_runtime,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve genre inference as an HTTP API")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--clip-duration", type=float, default=29.0)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--hop-length", type=int, default=512)
    p.add_argument("--top-k-default", type=int, default=5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--cors-origins",
        type=str,
        default="*",
        help='Comma-separated allowed origins. Use "*" for all.',
    )
    return p.parse_args()


def resolve_checkpoint(cli_checkpoint: Path | None) -> Path:
    if cli_checkpoint is not None:
        if not cli_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cli_checkpoint}")
        return cli_checkpoint

    preferred = Path("outputs/high_recall_precision_run/best_model.pt")
    if preferred.exists():
        return preferred

    fallback = Path("outputs/best_model.pt")
    if fallback.exists():
        return fallback

    candidates = sorted(Path("outputs").glob("**/best_model.pt"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No checkpoint found. Pass --checkpoint or place a checkpoint at "
        "'outputs/high_recall_precision_run/best_model.pt'."
    )


def create_app(args: argparse.Namespace) -> FastAPI:
    device = choose_device(args.device)
    runtime = optimize_runtime(device)
    checkpoint = resolve_checkpoint(args.checkpoint)
    ckpt = torch.load(checkpoint, map_location=device)
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

    app = FastAPI(title="Genre Detection Inference API", version="1.0.0")

    allowed_origins = [origin.strip() for origin in args.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": str(device),
            "num_classes": len(label_to_idx),
            "checkpoint": str(checkpoint),
        }

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), top_k: int | None = None) -> dict:
        filename = file.filename or ""
        if not filename.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        k = args.top_k_default if top_k is None else top_k
        if k <= 0:
            raise HTTPException(status_code=400, detail="top_k must be >= 1.")

        suffix = Path(filename).suffix or ".audio"
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty upload.")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(payload)
            tmp.flush()

            try:
                x = build_feature(
                    audio_path=Path(tmp.name),
                    sample_rate=args.sample_rate,
                    clip_duration=args.clip_duration,
                    n_mels=args.n_mels,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to decode audio: {exc}") from exc

            if runtime["use_channels_last"]:
                x = x.contiguous(memory_format=torch.channels_last)
            x = x.to(device, non_blocking=(device.type == "cuda"))

            amp_ctx = (
                torch.autocast(
                    device_type=device.type,
                    dtype=runtime["amp_dtype"],
                    enabled=runtime["use_amp"],
                )
                if runtime["use_amp"]
                else nullcontext()
            )
            with torch.inference_mode():
                with amp_ctx:
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).squeeze(0)

        k = min(k, probs.shape[0])
        top_probs, top_idx = torch.topk(probs, k=k)
        predictions = [
            {"genre": idx_to_label[i], "probability": float(p)}
            for p, i in zip(top_probs.tolist(), top_idx.tolist())
        ]
        return {"filename": filename, "predictions": predictions}

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
