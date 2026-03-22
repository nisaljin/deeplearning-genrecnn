# FMA Genre Classification (CNN)

This project trains a genre classification CNN on `fma_large` using labels/splits from `fma_metadata/tracks.csv`.

## Dataset Layout

Expected local structure:

- `fma_large/<###>/<######>.mp3`
- `fma_metadata/tracks.csv`

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

## Inference Demo (Deployment Artifact)

```bash
python predict_genre.py fma_large/002/002003.mp3 \
  --checkpoint outputs/full_run_seed42/best_model.pt \
  --top-k 5
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

## Metrics and Baseline

The script reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1

It also reports a **majority-class baseline** on validation/test for comparison.

## Suggested Hyperparameter Tuning Grid

Run and compare by macro-F1 on validation:

- `lr`: `1e-3`, `5e-4`, `1e-4`
- `batch-size`: `16`, `32`
- `clip-duration`: `15`, `29`
- `n-mels`: `64`, `128`

Use the best validation configuration and report test metrics once.

## Rubric Mapping (From `project.md`)

- Coding + reproducibility:
  - `requirements.txt`
  - deterministic seed (`--seed`)
  - explicit commands above for smoke/full runs
  - saved logs, plots, checkpoint, label mapping
- Performance & evaluation:
  - validation-based model selection
  - test evaluation with accuracy/precision/recall/F1
  - majority-class baseline comparison
- Delivery:
  - `predict_genre.py` demonstrates how a trained artifact is consumed

Recommended for your report:
- Include an end-to-end pipeline diagram (data -> preprocessing -> training -> validation tuning -> test -> inference).
- Add failure analysis examples using confusion matrix (`test_confusion_matrix.npy`).
- Run 3-5 hyperparameter settings and compare validation macro-F1 in a table.
