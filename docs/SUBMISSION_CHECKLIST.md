# Submission Checklist (61.502 Deep Learning for Enterprise, Y2026)

Use this checklist before final submission on eDimension/GitHub.

Target deadline in project brief: **April 17, 2026, 11:59pm**.

## 1) Repository Quality

- [ ] Repo is public and accessible.
- [ ] `README.md` explains setup, training, evaluation, and inference.
- [ ] Code is organized (notebook + reusable `.py` modules/scripts).
- [ ] No dataset files committed to git (`fma_large`, `fma_metadata`, cache, heavy outputs).

## 2) Environment + Reproducibility

- [ ] `requirements.txt` (or `environment.yml` / `pyproject.toml`) is present and correct.
- [ ] Seed is fixed and documented for final results.
- [ ] Exact commands are documented for: `smoke run`, `full training run`, `evaluation/test run`, `inference demo`.
- [ ] Hardware/runtime notes included (CPU/GPU, runtime expectations).
- [ ] Every figure/table in report has a clear reproduction command/path.

## 3) Technical Implementation (50%)

- [ ] Problem is clearly a deep learning task and relevant.
- [ ] Main model architecture is explained with enough detail to reproduce.
- [ ] Training pipeline uses train/validation/test split.
- [ ] Metrics for classification are reported: accuracy, precision, recall, F1 score.
- [ ] Hyperparameter tuning performed (at least 3-5 settings).
- [ ] At least one baseline is included and compared.
- [ ] At least two model variants are compared (for example: CNN vs transfer-learning model).

## 4) Visualization + Failure Analysis

- [ ] Training curves included (loss and/or accuracy/F1 curves).
- [ ] Confusion matrix or class-wise breakdown included.
- [ ] Failure cases shown (misclassifications) with plausible reasons.
- [ ] Limitations are explicit (class imbalance, noisy labels, short clips, domain shift, etc.).

## 5) Deployment / Consumption

- [ ] Lightweight consumption artifact provided (batch script or API).
- [ ] Clear example command shown with expected output format.
- [ ] Monitoring note included (latency, drift, quality checks if productionized).

## 6) Report (30%)

- [ ] Report PDF is in repo root (or clearly linked) and is 10-15 pages excluding appendix/references.
- [ ] Contains required sections: executive summary (<=1 page), background/introduction, related work (if applicable), problem formulation and solution overview, data description, methods/hyperparameters, evaluation/plots, discussion, recommendations, limitations/future work.
- [ ] Includes pipeline diagram (end-to-end ML flow).
- [ ] Includes link to code repository and dataset/weights locations.
- [ ] Includes group members and explicit contribution split.

## 7) Submission Packaging

- [ ] eDimension submission contains code/notebooks + report PDF (no heavy dataset files).
- [ ] GitHub repository includes the same code/report and clear run instructions.
- [ ] If weights are large: upload to Drive/Dropbox and link in README/report.

## 8) Presentation (20%)

- [ ] Slides/demo prepared for Week 13 presentation.
- [ ] Presentation covers objective, method, key results, impact/value.
- [ ] Demo is runnable or a fallback recorded video is ready.
