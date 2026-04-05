#!/usr/bin/env bash
set -euo pipefail

# CNN-only comparison runs under the same protocol as genre_classifier_workflow.ipynb.
# This keeps results directly comparable to notebook-scale metrics.

COMMON_ARGS=(
  --split-strategy stratified
  --train-ratio 0.8
  --val-ratio 0.1
  --test-ratio 0.1
  --min-class-count-for-split 10
  --min-train-count-for-model 120
  --no-merge-rare-classes
  --seed 42
  --epochs 35
  --batch-size 64
  --lr 7e-4
  --weight-decay 1e-4
  --num-workers 0
  --train-crop-duration 5
  --eval-crop-duration 5
  --val-multi-crops 5
  --test-multi-crops 7
  --use-balanced-sampler
  --use-focal-loss
  --focal-gamma 2.0
  --time-mask-max 24
  --freq-mask-max 12
  --posthoc-tune
  --posthoc-objective f1_macro
  --metric-for-best val_f1_macro
  --cache-dir .cache/mels_full_29s
)

echo "[1/3] Running cnn_standard_base_fullprotocol..."
python train_genre_cnn.py \
  "${COMMON_ARGS[@]}" \
  --model-arch standard_cnn \
  --no-use-aug \
  --standard-recipe-lr 1e-4 \
  --output-dir outputs/cnn_standard_base_fullprotocol

echo "[2/3] Running cnn_residual_fullprotocol..."
python train_genre_cnn.py \
  "${COMMON_ARGS[@]}" \
  --model-arch residual_cnn \
  --use-aug \
  --output-dir outputs/cnn_residual_fullprotocol

echo "[3/3] Running cnn_standard_aug_fullprotocol..."
python train_genre_cnn.py \
  "${COMMON_ARGS[@]}" \
  --model-arch standard_cnn \
  --use-aug \
  --standard-recipe-lr 1e-4 \
  --no-standard-recipe-disable-spec-aug \
  --output-dir outputs/cnn_standard_aug_fullprotocol

echo
echo "Done. Compare these files:"
echo "  outputs/cnn_standard_base_fullprotocol/summary.json"
echo "  outputs/cnn_residual_fullprotocol/summary.json"
echo "  outputs/cnn_standard_aug_fullprotocol/summary.json"
