# Methods

## Models
Two standalone simulator models are implemented under `course_project/src/course_project/models/`:
- `spatial`
- `spatial_transformer`

These are full simulator-style models (with `BaseSimulator`, `BaseModelInputs`, `forward`, `update`, `loss`).
No runtime dependency on `simulator/` is used from the course package.

## Training
- Sliding window over trajectory (`history` frames), node features from graph windows (`velocity` by default)
- Loss: model-native simulator loss (`model.loss`)
- Optimizer: Adam
- LR scheduler: ExponentialLR

## Rollout Evaluation (sides p-ratio)
- Run autoregressive rollout from initial history frames
- Compare predicted vs target p-ratio with:
  - `graph_utils.calc_p_ratio_rollout_sides`
- Report: `rollout_r2`, `rollout_pos_mse`

## Autoregressive Horizon Curve
- For horizons `1..autoregressive_max_steps` (default 100)
- Compute p-ratio R2 per horizon using sides metric
- Report includes `autoregressive_r2_h100`

## CV Evaluation
(Transformer model)
- Extract CV2 over early frames
- Compare mean CV2 per simulation vs final global p-ratio from:
  - `graph_utils.calc_p_ratio_box`
- Report: absolute Pearson r and linear-fit R2
