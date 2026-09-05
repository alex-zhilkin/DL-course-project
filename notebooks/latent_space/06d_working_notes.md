# 06d working notes

## Current setup

- Notebook: `06d_noisy_lj_history_rollout.ipynb`.
- The shared AE is trained/loaded once at
  `notebooks/results/06d_noisy_lj_history_rollout/06d_06b_matched_all_source_ae.pt`.
- AE recipe matches 06b: 4D attention AE, 32 latent tokens, hidden size 96,
  compact stored edges, normalized-delta input/target, 30 epochs maximum,
  patience 3, and validation p-ratio evaluation at steps 10, 50, and 100.
- AE checkpoint selection maximizes the minimum, across sources, of each
  source's summed validation p-ratio R² at steps 10, 50, and 100.
- Source-wise AE training counts are controlled by `AE_TRAIN_TRAJECTORIES`.
  Current values are Reid 20, dePablo low-T 20, dePablo mixed-T 20, and
  noisy-LJ 80. Validation uses 20 trajectories per source.
- `FORCE_TRAIN_AE` controls fresh AE training; otherwise its cache is loaded
  (or created if missing). `FORCE_TRAIN_PROPAGATOR` independently controls
  the propagator cache.

## Current propagator screen

- One experiment only: `noisy_lj_history_truncated_5`.
- Model: `history_mlp`, which receives three consecutive latent states plus
  the AE encoding of that trajectory's frame-0 reference structure.
- It uses equal-weight first-step normalized delta-Z loss and fifth-step
  normalized cumulative displacement loss (from the true starting latent).
  The intermediate autoregressive states are detached, so the fifth-step loss
  does not backpropagate through steps 1–4.
- No temperature, source-ID, or classifier input is used.
- The propagator trains on `TRAIN_TRAJECTORIES` noisy-LJ trajectories and
  reuses the one shared, frozen AE above.
- Latent normalization is one global set of training-split statistics. There
  is no source-specific latent normalization: the dataset is unknown at
  inference, so the model receives only observed graph-derived inputs.
- The notebook reports validation and held-out test source-wise normalized Δz
  MSE, source-classification accuracy/probability, and step-100 rollout
  p-ratio R².

## Diagnostics

- Immediately after AE training, the notebook displays source-wise validation
  AE p-ratio R² at step 100, source-wise validation reconstruction position
  MSE at step 100, and source-wise summed p-ratio R² at steps 10, 50, and 100,
  all against AE epoch.
- The R² panel is clamped to a lower y-limit of 0.
- The notebook also includes held-out noisy-LJ latent-slope versus final
  p-ratio scatters for z0–z3.

## Recent fixes

- The AE callback now records per-source validation reconstruction position
  MSE at the same step-100 cadence as the p-ratio R².
- The 06d setup reloads `lss.latent.experiment`, so notebook reruns use local
  source edits without requiring a kernel restart.
- Stride-aware rollout evaluation now aligns requested horizons relative to
  the last fixed observed frame, avoiding incompatible horizon rounding for
  fixed-history rollouts.

No new experiment results have been recorded yet.

## Latest AE run

- The latest all-source AE run reached its best minimum source sum at epoch 8.
- At epoch 8, noisy-LJ validation p-ratio R² was 0.225 at step 10, 0.543 at
  step 50, and 0.645 at step 100 (sum 1.413).
- Noisy-LJ step-100 R² later peaked at 0.683 at epoch 10, but its step-10 R²
  was negative there, giving a lower three-step sum (0.863).
- The step-10 p-ratio R² is highly unstable for noisy-LJ, including very large
  negative values at some epochs. Treat raw step-10 R² carefully: at such
  early deformation, its target variance can be very small, making R²
  ill-conditioned.
