# Course Project: Simulator + CV Validation

Lean, self-contained project for:
- training `spatial` and `spatial_transformer`
- rollout p-ratio validation (sides metric)
- autoregressive rollout analysis (including 100-step R2)
- CV vs global p-ratio analysis

This folder is independent from `simulator/`.
Only external project dependency is `graph_utils`.

## Setup

```bash
pip install -r course_project/requirements.txt
```

## Run from Notebook (recommended)

Use:
- `course_project/notebooks/01_training_and_rollout.ipynb`
- `course_project/notebooks/02_cv_vs_global_pratio.ipynb`

All experiment configs are defined inline in the notebook cells (no JSON config files required).
These call Python methods directly (`run_experiment`) and do not rely on CLI calls.

## Outputs per run

`course_project/results/<run_name>/`
- `final_checkpoint.pt`
- `train_stats.pt`
- `metrics.json`
- `rollout_predictions.csv`
- `autoregressive_rollout_r2.csv`
- `cv_vs_pratio.csv`
