# Project Report Template (10-15 Pages)

Use this as a writing template for the final PDF report.

## Title Page

- Project title
- Course: 61.502 Deep Learning for Enterprise (Y2026)
- Team members (full names, IDs)
- Date
- GitHub repository link

## 1. Executive Summary (<=1 page)

- Problem in one paragraph
- What you built
- Best result (main metric numbers)
- Main recommendation/business value

## 2. Background and Introduction

- Why the problem matters
- Stakeholders / users
- Impact if solved
- Scope and constraints

## 3. Related Work (if applicable)

- Prior approaches/models
- Why your approach is different
- Tradeoffs vs prior work

## 4. Problem Formulation and Solution Overview

- Input/output definition
- Task type (single-label classification)
- Success criteria
- High-level architecture overview

## 5. Data Description

- Dataset source and access link
- Dataset size and class distribution
- Split policy (train/validation/test)
- Any cleaning/filtering decisions
- Data limitations/bias risks

## 6. Methodology and Implementation Details

- Model A (CNN): architecture, training setup, hyperparameters
- Model B (Transfer model): architecture, finetuning setup, hyperparameters
- Loss, optimizer, scheduler
- Augmentation and regularization
- Runtime/hardware configuration
- Reproducibility details: package versions, seed, exact commands used.

## 7. Experiments and Evaluation

- Baseline definition and results
- Hyperparameter search setup
- Model comparison table (same split, same metrics)
- Validation and test metrics: Accuracy, Precision (macro), Recall (macro), F1 (macro).
- Plots: training curves, confusion matrix, optional class-wise bar charts.
- Bias/fairness audit note (class imbalance, underperforming classes)

## 8. Discussion

- What worked and why
- What failed and why
- Error analysis with concrete failure examples
- Interpretation of model behavior

## 9. Recommendations

- Which model to deploy and under what conditions
- Practical deployment guidance
- Risk mitigation steps

## 10. Limitations and Future Work

- Current limitations/caveats
- Improvements that are feasible in next iteration
- Longer-term research directions

## 11. Reproducibility Appendix

- Environment setup commands
- Train/eval/inference commands
- Output artifacts and where each figure comes from
- Links to dataset and model weights (Drive/Dropbox if large)
- Team member contribution table

## Required Figures/Tables Checklist

- Pipeline diagram (end-to-end ML flow)
- Model comparison table
- Hyperparameter tuning summary table
- Train/validation curves
- Test confusion matrix
- Failure-case examples

## Suggested Team Contribution Table

| Member | Responsibility | Deliverables |
|---|---|---|
| Member A | Data + training pipeline | preprocessing, baseline, run scripts |
| Member B | Modeling + tuning | model variants, metrics, ablations |
| Member C | Deployment + reporting | inference demo, report, presentation |
