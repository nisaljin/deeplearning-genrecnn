# FMA Genre Classification (CNN)

This project builds a deep-learning music genre classifier on `fma_large` using labels/splits from `fma_metadata/tracks.csv`.
It includes:

- a CNN training workflow (`genre_classifier_workflow.ipynb`, `train_genre_cnn.py`)
- a batch inference artifact (`predict_genre.py`)

## Dataset Layout

Expected local structure:

- `fma_large/<###>/<######>.mp3`
- `fma_metadata/tracks.csv`

Large datasets and local training artifacts are intentionally excluded from git via `.gitignore`.

## Download FMA Data

This project expects the extracted FMA archives to live at the repository root.
The official FMA repository is here: https://github.com/mdeff/fma

1. Create a temporary download folder in the project directory:

```bash
mkdir -p data/fma_downloads
cd data/fma_downloads
```

2. Download the archives you need from the official FMA repository:

```bash
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_large.zip
```

3. Unzip the archives so the extracted folders are named exactly as expected:

```bash
unzip fma_metadata.zip
unzip fma_large.zip
```

4. Move the extracted folders into the project root:

```bash
mv fma_metadata ../..
mv fma_large ../..
cd ../..
```

After this, the repository should contain:

- `fma_metadata/tracks.csv`
- `fma_large/<###>/<######>.mp3`

If you downloaded the archives somewhere else, the important part is that the extracted directories end up at the repository root with those exact names.

## Environment Setup (venv)

Use a virtual environment for all project commands.

### Option A: One-command setup script

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

### Option B: Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

To deactivate:

```bash
deactivate
```

## Quick Smoke Run

Useful for checking your pipeline before full training:

```bash
python train_genre_cnn.py \
  --max-train 300 --max-val 100 --max-test 100 \
  --epochs 2 --batch-size 16 --num-workers 0 \
  --output-dir outputs/smoke
```

## Full Training Example

```bash
python train_genre_cnn.py \
  --epochs 25 --batch-size 32 --num-workers 4 \
  --cache-dir .cache/mels \
  --output-dir outputs/full_run_seed42 \
  --seed 42
```

## Precompute Mel Cache (Recommended)

Run once to build mel `.npy` files for all labeled FMA-large tracks:

```bash
python precompute_mels.py \
  --cache-dir .cache/mels \
  --clip-duration 5 \
  --log-every 500
```

Then train from cache only (no MP3 decode during training):

```bash
python train_genre_cnn.py \
  --clip-duration 5 \
  --cache-dir .cache/mels \
  --require-cache \
  --epochs 25 --batch-size 32 --num-workers 4 \
  --output-dir outputs/full_run_seed42_cache_only \
  --seed 42
```

## Inference Demo (Deployment Artifact)

```bash
python predict_genre.py fma_large/002/002003.mp3 \
  --checkpoint outputs/full_run_seed42_cache_only/best_model.pt \
  --top-k 5
```

## Inference API (Frontend Demo)

Run an HTTP inference server:

```bash
python infer_api.py \
  --host 0.0.0.0 \
  --port 8000
```

By default it auto-loads `outputs/high_recall_precision_run/best_model.pt`.
You can still override with `--checkpoint <path>`.

Health check:

```bash
curl http://localhost:8000/health
```

Predict from an uploaded audio file:

```bash
curl -X POST "http://localhost:8000/predict?top_k=5" \
  -F "file=@fma_large/002/002003.mp3"
```

For browser apps, CORS is enabled by default (`*`). Restrict it in production:

```bash
python infer_api.py \
  --checkpoint outputs/high_recall_precision_run/best_model.pt \
  --cors-origins "http://localhost:3000,https://your-demo.example"
```

## Reproducibility Notes

- Uses official FMA split from `set.split`: `training` / `validation` / `test`.
- Uses only `set.subset == large` and non-null `track.genre_top`.
- Auto-detects accelerator with `--device auto` (prefers `cuda`, then `mps`, then `cpu`) and applies backend-specific runtime optimizations.
- Saves:
  - `best_model.pt`
  - `history.csv` and `history.json`
  - `loss_curves.png`, `accuracy_curves.png`
  - `summary.json`
  - `test_confusion_matrix.npy`
  - `label_mapping.json`

## Evaluation Metrics

The script reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

Recommended for report consistency:

- report both validation and test metrics
- include class-wise performance and confusion matrix discussion
- include at least one simple baseline (for example majority-class predictor)

## Suggested Hyperparameter Grid

Run and compare by macro-F1 on validation:

- `lr`: `1e-3`, `5e-4`, `1e-4`
- `batch-size`: `16`, `32`
- `clip-duration`: `15`, `29`
- `n-mels`: `64`, `128`

Use the best validation configuration and report test metrics once.

## Model Comparison Plan

Use CNN-only model comparison (valid for project requirements) with:

- `cnn_standard_base` (simpler settings)
- `cnn_residual` (ResCNN-style variant)
- `cnn_standard_aug` (`standard_cnn` architecture)

You can run all three using:

```bash
bash scripts/run_cnn_comparison.sh
```

Minimum comparison table columns:

- model name
- key hyperparameters
- validation macro-F1
- test macro-F1
- test macro-precision
- test macro-recall
- train/inference cost notes

For each run, read metrics from:

- `<output_dir>/summary.json`
- `<output_dir>/history.csv`

## Rubric-Aligned Submission Docs

- Submission readiness checklist: `docs/SUBMISSION_CHECKLIST.md`
- 10-15 page report template: `docs/REPORT_TEMPLATE.md`

## Deployment + Monitoring Note (for report)

Current artifact is a batch inference script (`predict_genre.py`).
If deployed as a service, track:

- latency (P50/P95/P99)
- throughput
- prediction drift (class distribution shift over time)
- data drift (audio duration/sample-rate/loudness distribution changes)
