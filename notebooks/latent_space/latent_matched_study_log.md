# Shared latent simulator results

## AE-unseen mixed-T transfer — running

The next controlled question excludes mixed-T from AE training and AE checkpoint
selection, as well as propagator supervision and checkpoint selection. Nine
Reid/low-T/LJ AEs (dimensions 2/6/8, seeds 123/456/786) use the existing matched
representation recipe and split. Each feeds acceleration, fixed-velocity, and
plain delta propagators with the corresponding model seed and forecast origin
at frame 5. All four validation sources are evaluated, including mixed-T AE
reconstruction and rollout. No mixed-T response label selects a checkpoint.
This crosses dimension and dynamics model; seeds vary AE and dynamics together,
so their variability cannot be attributed to either component separately.
The original all-four-AE 8D comparisons remain the fixed-AE baseline.
Nine PBS pipelines automatically train the AE then its three propagators;
results are under `notebooks/results/latent_no_mixed/`. Reserved test is untouched.

## Matched representation screen — complete

The screen compares dimensions 2/4/6/8 for shared Reid + both dePablo
sources, shared all-four-source, and individual-source AEs. Each condition has
seeds 123/456/786. Every run uses the same topology-disjoint split: 30 training
and 20 validation trajectories per source. Reserved test trajectories remain
unseen.

The AE is an attention model with width 128 and 32 decoder tokens, trained on
frames 0–100 with normalized displacement input and target, physical reference
context, compact 5D edges, equal source-weighted graph MSE, and response-blind
checkpoint selection by worst-source validation reconstruction error.

Frame-100 validation p-ratio R² for the 8D models (mean ± sample SD across
three seeds):

| Training scope | Reid | dePablo low-T | dePablo mixed-T | noisy-LJ |
|---|---:|---:|---:|---:|
| Individual | 0.930 ± 0.003 | 0.934 ± 0.001 | 0.759 ± 0.240 | -0.136 ± 0.069 |
| Shared three-source | 0.932 ± 0.021 | 0.964 ± 0.021 | 0.707 ± 0.253 | — |
| Shared four-source | **0.962 ± 0.006** | **0.982 ± 0.008** | **0.882 ± 0.105** | **0.351 ± 0.083** |

Adding noisy-LJ generally improves established-source response reconstruction in
this screen; this is not a guarantee for every seed or dimension. Across
all four dimensions and matched seeds, shared-four minus shared-three R² is
`+0.032 ± 0.055` for Reid, `+0.020 ± 0.025` for low-T, and
`+0.099 ± 0.166` for mixed-T. The corresponding field MSEs remain essentially
unchanged. For noisy-LJ, the shared model improves over its individual AE by
`+0.382`, `-0.313`, `+0.356`, and `+0.487` R² at dimensions 2, 4, 6, and 8.
The 4D noisy-LJ result is unstable; 2D, 6D, and 8D all give positive mean R².

The results are consistent with beneficial sharing. Regularization by the
other sources is a hypothesis, not a mechanism established by this comparison:
shared-four also has more total training examples and updates per epoch.
Noisy-LJ still has the lowest response reconstruction despite field MSE near
`8e-6`, because its transverse-strain variation is small relative to node-level
error. A constant latent and one global PC fail. The full 2D shared AE works,
while a two-PC projection of the 8D AE is unreliable for noisy-LJ, so the useful
low-dimensional structure is not always a global linear subspace.

The frozen rollout AE is seed 123 of the 8D shared-four
condition. Its source-wise validation R² is Reid `0.965`, low-T `0.987`,
mixed-T `0.929`, and noisy-LJ `0.421`.

## Dynamics screen — 33 runs complete, validation only

The dynamics screen keeps that AE, split, 30/20 trajectories per source,
frames 0–100, physical mean graph context, and all observations through frame
5 fixed. It compares three dynamics seeds for:

- shared versus four source-specific two-frame velocity-residual propagators;
- shared propagation with and without explicit source ID;
- shared propagation with temperature, and temperature plus source ID;
- two-frame velocity-residual versus four-frame latent history;
- one-step versus eight-step closed-loop training.

All shared batches and losses are source-balanced. Checkpoints use the minimum
source-wise frame-100 validation p-ratio R². Only validation trajectories are
available to these runs; final test evaluation waits until the dynamics recipe
is fixed.

Frame-100 endpoint graph-utils p-ratio R², mean across three dynamics seeds
(20 validation trajectories/source/seed):

| Propagator | Reid | low-T | mixed-T | noisy-LJ |
|---|---:|---:|---:|---:|
| Separate velocity models | 0.946 | 0.993 | -0.010 | 0.102 |
| Shared velocity, one-step | 0.846 | 0.915 | -0.170 | -0.354 |
| Shared velocity + source ID | 0.891 | 0.888 | -0.324 | -0.151 |
| Shared velocity + temperature | 0.813 | 0.868 | -0.317 | -0.184 |
| Shared velocity + source ID + temperature | 0.726 | 0.876 | -0.254 | -0.135 |
| Shared velocity, eight-step | 0.935 | 0.979 | -0.425 | 0.068 |
| Shared fixed window, one-step | 0.135 | 0.226 | 0.226 | -0.191 |
| Shared fixed window, eight-step | 0.452 | 0.335 | -0.009 | -0.389 |

No recipe succeeds across all four sources. Mixed-T fails even with a separate
velocity model; sharing alone cannot explain that failure. The fixed window
helps mixed-T but loses Reid/low-T accuracy. Metadata conditioning does not
resolve the failure in this recipe. These are trained-model comparisons, not
proof that the observed prefix lacks predictive information.

Exact dynamics recipe: frozen shared-four 8D seed-123 AE; hidden width 96,
mean static context projected to 16 dimensions; batch 512; AdamW lr 1e-4,
weight decay 1e-4; maximum 30 epochs, patience 6; validation endpoint selection
every two epochs. Velocity observes (1,5); the fixed window observes (0,1,3,5).
Both forecast from frame 5 to frame 100. Eight-step training supervises horizons
1–8. The per-source row budget is 2850, with resampling for eight-step training
whose available start positions are fewer. Separate models also fit separate
latent normalizers, so their contrast includes scaling as well as sharing.

Implementation audit after completion found an additional confound: the
vectorized fixed-history one-step path always adds Gaussian noise with scale
0.25 times training latent-increment SD to current and observed latents. The
eight-step path does not use that augmentation. Thus these results cannot
isolate training horizon alone. Preserve them as diagnostic results; align
noise, row sampling, and normalization before a confirmatory horizon ablation.

## Mixed-T transfer follow-up

The historical 06b result achieved mixed-T rollout R² 0.901 while excluding
mixed-T from propagator training; its AE included mixed-T. It used a plain
delta MLP, not the fixed-velocity model in the recent screen. Nine follow-up
runs use the current frozen AE and split, with seeds 123/456/786: velocity
excluding mixed-T, plain delta excluding mixed-T, and plain delta on all four.
Transfer runs exclude mixed-T from training rows, latent-stat fitting, validation
loss, and checkpoint selection, but report it in final validation evaluation.
The delta model starts its forecast at observed frame 5, matching the velocity
forecast origin. This tests transfer under the current AE, not exact historical
reproduction. Results are pending under the corresponding `*_transfer3_*` and
`delta_all4_*` directories; the frozen implementation is `code_v2`.

## Rolling acceleration follow-up

Rolling acceleration follow-up: six additional runs (all four training sources
versus Reid/low-T/LJ only; seeds 123/456/786) use `history_mlp` with one-step
`history_one_step` training, otherwise the same width-96, context-16, batch-512
optimizer and selection recipe. The forecast initializes from frames (3,4,5)
and reference frame 0, then updates rolling velocity and acceleration from
predicted states. Training targets stay within frame 100. Transfer runs exclude
mixed-T from both propagator fitting and checkpoint selection. Frozen code is
`code_v3`; results are pending in `acceleration_all4_*` and
`acceleration_transfer3_*`. This is a matched-AE follow-up, not a reproduction
of the historical notebook-08 AE or optimization sweep.

All 15 transfer/acceleration follow-ups completed (48 dynamics runs total).
Frame-100 validation endpoint graph-utils p-ratio R², mean across seeds:

| Model / propagator training sources | Reid | low-T | mixed-T | noisy-LJ |
|---|---:|---:|---:|---:|
| Velocity / Reid + low-T + LJ | 0.936 | 0.992 | -0.321 | 0.202 |
| Delta / Reid + low-T + LJ | 0.282 | 0.253 | 0.053 | -0.299 |
| Delta / all four | 0.151 | 0.008 | 0.023 | -0.223 |
| Acceleration / Reid + low-T + LJ | 0.957 | 0.990 | -2.471 | 0.370 |
| Acceleration / all four | -0.824 | -0.444 | -1.252 | -2.174 |

The three-source acceleration model improves noisy-LJ and retains Reid/low-T.
Its seed SDs are 0.001 / 0.002 / 0.045 / 0.003 in table source order.
However, it does not recover the historical mixed-T transfer result. Its
mixed-T AE reconstruction R² is 0.929 on the same validation trajectories,
so the observed rollout failure is downstream of accurate endpoint
reconstruction. The matched graph-utils AE noisy-LJ R² is 0.411 (distinct
from the conventional strain-ratio value 0.421 reported above).

Noisy-LJ acceleration-transfer R² at frames 10/25/50/75/100 is
-5.004 / -0.486 / 0.132 / 0.294 / 0.370. Thus the endpoint improvement is not
uniformly good trajectory response prediction. All reported endpoints retain
20/20 trajectories. Including mixed-T changes training latent statistics and
checkpoint selection as well as supervision; this comparison alone does not
establish gradient interference. Next diagnose source scaling and compare
untrained kinematic baselines before attributing the improvement to learned
acceleration or expanding the architecture search. Reserved test is untouched.

## Initial response coordinates (validation analysis)

The saved coordinates were joined to frame-100 validation response by source
and manifest trajectory identity. All raw axes and training-fitted PCs are
reported in `initial_coordinate_response_correlations.csv`; none is selected
as a predictive readout on these validation labels.

For shared-four 2D seed 123, frame-zero z1 Pearson correlations are Reid
-0.776, low-T 0.689, mixed-T 0.554, and noisy-LJ 0.118. Results vary strongly
with model seed. For noisy-LJ, absolute raw-coordinate correlations remain
below 0.30 across all three 2D seeds despite positive endpoint reconstruction.
This distinguishes initial structural response information from reconstruction
using an observed deformed state. These 30-trajectory/source results do not
replicate or refute the historical one-trajectory/source experiment, whose
budget, temporal support, and split differ. That matched low-data comparison
and training-calibrated held-out probes remain necessary for the paper.

Machine-readable recipes, checkpoints, per-trajectory metrics, and aggregate
tables are under `notebooks/results/latent_matched_study/` and
`notebooks/results/latent_matched_rollouts/`.

## Full-workload AE CPU timing sweep — submitted 2026-09-05

PBS job `4669053.zeus-master` benchmarks the existing
`shared_no_mixed_d8_s123/recipe.json` using its frozen `latent_no_mixed/code_v1`
implementation: Reid/low-T/LJ, 30 training and 20 validation trajectories per
source, frames 0–100, batch size 32, attention AE width 128, 32 latent tokens,
8D latent, source-mean gradients, and seed 123. Only the epoch budget and
patience change to three, with separate benchmark checkpoint paths.

Thread counts 1/2/4/8/16/32/64 each run twice in fresh processes, in a fixed
shuffled order on one exclusively reserved node (64 requested CPUs, 64 GB,
24-hour limit). PyTorch intra-op, OMP, MKL, OpenBLAS, and NumExpr thread counts
match each setting; PyTorch inter-op is one. The benchmark snapshots source
code and recipe and records code hashes, host, CPU model, affinity, individual
training/validation epoch times, full experiment elapsed time, and AE histories.
Ranking prioritizes median training-plus-validation epoch wall time across
epochs 2–3 of both repeats; first-epoch and setup costs remain in total times.
This tests full-data epoch throughput, not time to convergence or rollout speed.
Results: `notebooks/results/latent_full_ae_benchmark/4669053.zeus-master/`.
No timing winner or scientific result is available yet. Production CPU settings
remain at eight pending the measurements. Python compilation and PBS shell
syntax checks passed before submission.

### Multi-node timing sweep — 2026-09-05

At the user's request, replaced the sequential job `4669053.zeus-master`
with three concurrent, exclusively reserved node jobs:

- `4669055.zeus-master`, dm01: 1/2/8 threads.
- `4669056.zeus-master`, dm02: 4/8/16 threads.
- `4669057.zeus-master`, dm04: 8/32/64 threads.

Each setting still has two independent three-epoch runs. All jobs copy the
original sweep's frozen code and recipe; each node includes an eight-thread
baseline. Compare within-node speedups and confirm finalists on a common node
before changing production defaults. The original job was cancelled with its
partial output preserved. No completed timing or scientific results are claimed.
`scripts/summarize_latent_full_ae_benchmark.py` collects available results and
reports speedups against the local eight-thread baseline.

## CPU timing result — 2026-09-06

All 18 full-data timing runs completed. Explicit host requests were rejected
by the scheduler; before execution the three jobs were changed to scheduler-selected
exclusive nodes. Eight threads was fastest in each within-node comparison.

| Job | Actual host | Threads: median epoch seconds |
|---|---|---|
| 4669055.zeus-master | n153.zeus.technion.ac.il | 8: 117.08; 2: 177.94; 1: 252.24 |
| 4669056.zeus-master | n150.zeus.technion.ac.il | 8: 124.50; 16: 157.95; 4: 168.88 |
| 4669057.zeus-master | n151.zeus.technion.ac.il | 8: 120.79; 32: 123.55; 64: 158.78 |

Retain eight threads for AE jobs. The 32-thread result is only about 2.3%
slower than its local eight-thread baseline; no strong separation is claimed.

## Scientific follow-ups — implemented and initial jobs running, 2026-09-06

Diagnostics use saved shared-four 8D AE transfer checkpoints for velocity, delta,
and rolling acceleration, seeds 123/456/786. Forecast origin is frame 5.
Compare free rollout and teacher-forced local prediction, with oracle AE,
constant latent, velocity from (1,5), velocity from (4,5), and discrete constant
acceleration from (3,4,5). Report source-wise field/strain/response errors at
frames 10/25/50/75/100 and raw and normalized latent error at every predicted
frame. Teacher forcing is a diagnostic using true subsequent observations,
not an early-prefix forecast. Require free predictions to reproduce saved
rollout p-ratios before marking results complete.

Low-data runs use shared-three/shared-four 2D AEs, seeds 123/456/786, the first
predeclared training trajectory per source, and the existing 20 validation
trajectories/source. Preserve the matched frames 0–100, 40-epoch maximum and
patience 8; reduce rows/source to 101. This is a matched-budget comparison,
not update-matched training or exact historical frames-0–199 reproduction.
Fit source-specific fixed-alpha-1 ridge probes using the other 29 training
networks/source for both the one- and thirty-trajectory AEs. Compare a constant,
training-calibrated initial PC1, full initial z, and prefix features through
frame 5. All PCA/scaling/regression fits exclude validation labels. Report
source-wise validation metrics and initial correlations; reserved test remains
untouched. The calibration budget is separate from the AE training budget.

Scripts are frozen under `notebooks/results/latent_science_followup/code_v1/`.
Job ledger: `notebooks/results/latent_science_followup/jobs.jsonl`. Each job
requests one exclusive node, eight CPUs, 48 GB, and 24 hours. Initial jobs:
`4669148.zeus-master` (seed-123 diagnostics), `4669149.zeus-master`
(shared-three seed-123 low-data). Compilation and PBS syntax checks passed.

All nine science follow-up jobs are submitted (`4669148`–`4669156`);
the ledger records their task/scope/seed mapping. These remain pending results.

## AE-unseen mixed-T transfer — completed 2026-09-06

All nine AE-plus-dynamics pipelines and 27 rollout runs completed using the
recipe recorded above (frozen `latent_no_mixed/code_v1`, dimensions 2/6/8,
seeds 123/456/786, 30 training and 20 validation trajectories/source,
AE excludes mixed-T, propagator fitting/statistics/checkpoint selection exclude
mixed-T, forecast origin 5). Full recipes are in each AE and dynamics output
directory; source-wise metrics at all five horizons are in
`latent_no_mixed/sourcewise_results.csv` and `sourcewise_aggregate.csv`.
Frame-100 validation p-ratio R², mean ± sample SD over three seeds:

| Dimension | Dynamics | Reid | low-T | mixed-T | noisy-LJ |
|---|---|---|---|---|---|
| 2 | acceleration_transfer3 | 0.923 ± 0.024 | 0.960 ± 0.020 | -1.486 ± 1.026 | 0.413 ± 0.096 |
| 2 | delta_transfer3 | 0.082 ± 0.138 | 0.195 ± 0.154 | 0.189 ± 0.084 | -0.229 ± 0.150 |
| 2 | velocity_transfer3 | 0.909 ± 0.036 | 0.949 ± 0.018 | -1.196 ± 0.235 | 0.234 ± 0.035 |
| 6 | acceleration_transfer3 | 0.528 ± 0.680 | 0.770 ± 0.331 | -1.481 ± 0.821 | 0.214 ± 0.231 |
| 6 | delta_transfer3 | -0.139 ± 0.375 | -0.156 ± 0.701 | -0.276 ± 0.444 | -0.335 ± 0.218 |
| 6 | velocity_transfer3 | 0.622 ± 0.526 | 0.813 ± 0.250 | -1.367 ± 0.716 | 0.133 ± 0.151 |
| 8 | acceleration_transfer3 | 0.910 ± 0.060 | 0.972 ± 0.006 | -1.811 ± 0.510 | 0.418 ± 0.040 |
| 8 | delta_transfer3 | 0.081 ± 0.040 | 0.131 ± 0.049 | 0.168 ± 0.238 | -0.163 ± 0.187 |
| 8 | velocity_transfer3 | 0.926 ± 0.023 | 0.971 ± 0.002 | -1.514 ± 0.889 | 0.252 ± 0.072 |

Every reported endpoint retains 20/20 trajectories per source per seed.
Mixed-T transfer remains unresolved. These are validation results, not reserved
final-test results. No pooled success claim is supported.

## Completed rollout diagnostics — 2026-09-06

All nine transfer diagnostics completed and reproduced saved free-rollout
p-ratios within rtol=1e-4, atol=1e-5. Recipe and causal prefix are as specified
in the scientific follow-up entry above. These use the shared-four 8D AE,
not the AE-unseen mixed-T AEs. Frame-100 validation R² averaged across three
dynamics seeds (same fixed AE and 20 networks/source):

| Dynamics | Evaluation | Reid | low-T | mixed-T | noisy-LJ |
|---|---|---|---|---|---|
| acceleration_transfer3 | free_rollout | 0.957 | 0.990 | -2.471 | 0.370 |
| acceleration_transfer3 | teacher_forced | 0.965 | 0.988 | 0.682 | 0.410 |
| delta_transfer3 | free_rollout | 0.282 | 0.253 | 0.053 | -0.299 |
| delta_transfer3 | teacher_forced | 0.965 | 0.987 | 0.896 | 0.409 |
| velocity_transfer3 | free_rollout | 0.936 | 0.992 | -0.321 | 0.202 |
| velocity_transfer3 | teacher_forced | 0.966 | 0.987 | 0.843 | 0.415 |
| shared control | oracle_ae | 0.965 | 0.988 | 0.929 | 0.411 |
| shared control | constant_latent | 0.387 | 0.459 | -0.893 | -4.708 |
| shared control | constant_velocity | 0.939 | 0.971 | -0.855 | -0.792 |
| shared control | constant_velocity_recent | 0.939 | 0.915 | -1.768 | -0.549 |
| shared control | constant_acceleration | 0.032 | 0.590 | -1.638 | -26.543 |

Controls are identical across the three dynamics-seed comparisons; they are
not independent AE repetitions. Complete per-source/frame/seed metrics, valid
counts, strain/field errors and latent drift are saved in the diagnostic run
directories and `latent_science_followup/diagnostics_sourcewise.csv`. Teacher
forcing uses subsequent true observations and is not a forecast from frame 5.
Its improvement over free rollout is consistent with accumulated rollout error;
it does not by itself establish the cause or a successful intervention.

Low-data status at this check: shared-three seed 123 complete; four runs active
and shared-four seed 786 queued. Preliminary initial-z ridge probe validation
R² for the one-trajectory AE is Reid 0.288 / low-T 0.974 / mixed-T 0.921,
using 29 labeled calibration networks/source. Prefix-through-5 probes give
0.930 / 0.970 / 0.936. This single seed is not a confirmed multi-seed finding.
Its frame-100 AE reconstruction graph-utils R² is -0.249 / -0.263 / -0.210;
initial-coordinate predictive information and response reconstruction must be
reported separately. Remaining low-data results are pending.

## Low-data comparison — all six runs complete

Same one-trajectory/source AE recipe and separate 29-network/source calibration
budget recorded above; all six runs completed, three model seeds per scope.
Validation initial-z probe R² (mean ± sample SD):

| Scope | AE budget | Source | R² |
|---|---|---|---|
| shared3 | one_trajectory | depablo_low_temp | 0.979 ± 0.006 |
| shared3 | one_trajectory | depablo_mixed_temp | 0.934 ± 0.015 |
| shared3 | one_trajectory | reid | 0.411 ± 0.445 |
| shared3 | thirty_trajectories | depablo_low_temp | 0.108 ± 0.170 |
| shared3 | thirty_trajectories | depablo_mixed_temp | 0.173 ± 0.266 |
| shared3 | thirty_trajectories | reid | 0.080 ± 0.082 |
| shared4 | one_trajectory | depablo_low_temp | 0.873 ± 0.045 |
| shared4 | one_trajectory | depablo_mixed_temp | 0.837 ± 0.044 |
| shared4 | one_trajectory | lj_noisy | -0.137 ± 0.105 |
| shared4 | one_trajectory | reid | 0.896 ± 0.004 |
| shared4 | thirty_trajectories | depablo_low_temp | 0.779 ± 0.092 |
| shared4 | thirty_trajectories | depablo_mixed_temp | 0.779 ± 0.075 |
| shared4 | thirty_trajectories | lj_noisy | -0.123 ± 0.075 |
| shared4 | thirty_trajectories | reid | 0.510 ± 0.222 |

All probe rows retain 20/20 validation networks. Source-specific regressions
use response labels only from calibration networks; these are not zero-label
predictors. One-trajectory AE response reconstruction at frame 100:

| Scope | Source | R² |
|---|---|---|
| shared3 | depablo_low_temp | -0.194 ± 0.060 |
| shared3 | depablo_mixed_temp | -0.122 ± 0.078 |
| shared3 | reid | 0.002 ± 0.217 |
| shared4 | depablo_low_temp | -0.202 ± 0.056 |
| shared4 | depablo_mixed_temp | -0.111 ± 0.025 |
| shared4 | lj_noisy | -15.797 ± 5.780 |
| shared4 | reid | 0.026 ± 0.118 |

Initial response coordinates and reconstruction behave differently. The
40-epoch low-data recipe uses only 101 rows/source/epoch versus 3030 in the
original screen; update budget is a confound. Prepared (not submitted) a
3030-row resampling control and a controlled acceleration horizon comparison
(1 vs 8, identical starts 5–92, no history noise, shared-four frozen 8D AE,
all-four vs three-source fitting, seeds 123/456/786). Index checks verify 88
starts/trajectory for both horizons and latent-stat fitting, final target 100.

At user request, audit materialized normalized datasets before further training.
Audit jobs and results: `notebooks/results/normalized_data_audit/`. The saved
manifest contains retained-file entries and stale workstation paths, not
validation evidence. Current raw-data experiments already request reference-box
normalization; verify numerical equivalence before interpreting a file-path
change as a new normalization experiment. No normalized-data training has been
submitted pending that audit.

## In-memory normalization verified; controlled follow-ups — 2026-09-06

At the user's direction, retain raw data with in-memory `position_normalization`.
Independent affine/geometry checks passed on all 802 trajectories and 160,400
frames: Reid 102/20,400, low-T 200/40,000, mixed-T 300/60,000, LJ 200/40,000.
All frames use the frame-zero center and per-axis half-width, with evolving
periodic boxes. Position mapping, minimum-image edge vectors/lengths, unchanged
stiffness/scalar edge fields, preserved physical reference context, and marker
checks passed. Box-width differences from independently computed values were
at most 2.4e-7. Repeat normalization was checked at frame zero for all real
trajectories; an independent two-frame anisotropic test also checked all fields
on both frames, affine strain preservation, velocity scaling, and idempotence.
`tests/test_reference_box_current_convention.py`: 1 passed. Historical tests in
`test_data_edges.py` still describe the earlier isotropic/stiffness-scaled
convention and were not used as evidence for this current per-axis convention.
No normalization implementation was changed.

Machine-readable audit: `notebooks/results/in_memory_normalization_audit/`.
The prior materialized-file comparison found missing/different reference-edge
context channels and differing LJ frames. Those files are not being substituted
into training. Their provenance discrepancy is not an in-memory audit failure.

After all four in-memory reports passed, submit the following predeclared
matrix through `scripts/submit_latent_controlled_followup.py`, with append-only
ledger `notebooks/results/latent_controlled_followup/jobs.jsonl`:

- Twelve acceleration runs: frozen shared-four 8D seed-123 AE; all-four versus
  Reid/low-T/LJ propagator fitting; horizons 1 versus 1–8; dynamics seeds
  123/456/786. Same starts 5–92 and latent-stat fitting starts, 88 per trajectory,
  2640 rows/source/epoch, history noise 0, width 96, mean context 16, batch 512,
  AdamW 1e-4/weight-decay 1e-4, maximum 30 epochs/patience 6. Existing source-wise
  endpoint validation selection every two epochs, frame-5 forecast origin,
  response evaluation at 10/25/50/75/100. This controls within-scope horizon
  comparisons; all-four vs transfer still changes fitted source statistics and
  checkpoint-selection sources. Frozen code: `latent_stability/code_v1`.
- Six 2D low-data AE repeats: shared-three/shared-four, seeds 123/456/786,
  one predeclared training network/source, but 3030 resampled rows/source/epoch
  to match the original thirty-trajectory screen's per-epoch sample/update
  budget. Same 40-epoch cap/patience 8; actual early stopping can differ. Keep
  the same 29-network/source response calibration and 20-network/source
  validation probes. Results: `latent_low_data_update_matched`; scripts frozen
  in that directory's `code_v1/scripts`, using the original matched AE code.

Each job requests `mendels_q`, one exclusive scheduler-selected node, eight CPUs,
48 GB, 24 hours. No host pins. No reserved final-test evaluation. Configuration
preparation, compilation, PBS syntax, and equal-start-population checks passed.
These follow-ups have no results yet; they test hypotheses, not established fixes.

### Scheduling correction: share nodes

At the user's request, changed PBS templates from `place=excl` to explicit
`place=free:shared`. Training still uses one node and eight CPUs per job, with
no host pin. Queued controlled follow-ups were changed in place. PBS rejected
placement changes on active jobs, so exclusive jobs 4669263/4669264/4669265/
4669270 were cancelled and resubmitted with shared placement. Partial output
was preserved under `latent_controlled_followup/interrupted_exclusive/` and
replacement IDs appended to the job ledger. Completed results were retained.
The user's unrelated job array was not modified. CPU and memory requests are
unchanged by this scheduling correction. Historical CPU benchmark timings were
measured under exclusive placement; shared-node elapsed times may differ.

## Controlled follow-ups — all 18 runs complete

All 12 acceleration comparisons and six update-budget AE controls completed.
Recipes, frozen code, sample population and normalization are specified in the
preceding entries. Aggregation uses completed run directories only, excluding
interrupted exclusive-placement attempts. Full per-seed/source metrics are in
`latent_controlled_followup/{stability,reconstruction,probes}_sourcewise.csv`;
aggregates retain three seeds and sample SD. No final-test results are used.

Frame-100 validation rollout p-ratio R² (mean ± sample SD):

| Fitting scope | Horizon | Reid | low-T | mixed-T | noisy-LJ |
|---|---|---|---|---|---|
| all4 | 1 | -0.615 ± 0.415 | -0.214 ± 0.597 | -1.289 ± 0.869 | -1.773 ± 1.380 |
| all4 | 8 | 0.700 ± 0.097 | 0.897 ± 0.093 | -0.821 ± 0.537 | -0.395 ± 0.264 |
| transfer3 | 1 | 0.956 ± 0.000 | 0.992 ± 0.001 | -2.500 ± 0.038 | 0.352 ± 0.023 |
| transfer3 | 8 | 0.955 ± 0.001 | 0.992 ± 0.000 | -2.423 ± 0.119 | 0.379 ± 0.007 |

Every source retains 20/20 trajectories at all reported horizons. Eight-step
training improves the all-four model but does not establish all-source success.
The transfer model preserves Reid/low-T and modestly improves LJ; mixed-T
remains strongly negative. Do not interpret the all-four/transfer contrast as
isolating gradient interference: normalizers and selection sources also differ.

One-trajectory/source AE reconstruction, with 3030 sampled rows/source/epoch,
frame-100 graph-utils p-ratio R²:

| Scope | Source | R² mean ± SD |
|---|---|---|
| shared3 | depablo_low_temp | -0.364 ± 0.070 |
| shared3 | depablo_mixed_temp | -0.271 ± 0.094 |
| shared3 | reid | -0.530 ± 0.418 |
| shared4 | depablo_low_temp | -0.599 ± 0.037 |
| shared4 | depablo_mixed_temp | -0.520 ± 0.078 |
| shared4 | lj_noisy | -1.712 ± 0.157 |
| shared4 | reid | -0.403 ± 0.270 |

All six update-budget controls stopped at nine epochs, so the per-epoch
update budget is matched but total training updates are not equalized across
early-stopped conditions. Repeating the same trajectory did not recover
accurate response reconstruction. Initial-z probes with 29 calibration
networks/source remain distinct from reconstruction:

| Scope | Source | Probe R² mean ± SD |
|---|---|---|
| shared3 | depablo_low_temp | 0.934 ± 0.064 |
| shared3 | depablo_mixed_temp | 0.905 ± 0.071 |
| shared3 | reid | 0.792 ± 0.096 |
| shared4 | depablo_low_temp | 0.861 ± 0.011 |
| shared4 | depablo_mixed_temp | 0.851 ± 0.046 |
| shared4 | lj_noisy | -0.147 ± 0.126 |
| shared4 | reid | 0.800 ± 0.080 |

These controls do not resolve mixed-T rollout transfer or initial noisy-LJ
response prediction. The next targeted diagnosis should separate source-scale
and checkpoint-selection effects before expanding architecture or horizon
sweeps. No further training was launched during this result collection.

## Return to historical 4D baseline

User requested reproducing successful mixed-T rollout before building further,
with 4D explicitly required. Submitted three 4D recipe reconstructions, seeds
3456456/123/456, AE all four sources, delta propagator Reid + low-T only.
Historical artifact/recipe gaps and protected-split differences are documented
in `06b_experiment_log.md` and per-run recipes. This is not an exact historical
replay. Results and ledger: `notebooks/results/06b_4d_reconstruction/`.
All jobs permit shared-node placement in `mendels_q`. Pending results; no further
architecture or dimensionality sweep submitted.

## Persistent experiment policy and dimensionality comparison

Updated repository guidance to favor broad, hypothesis-driven matrices,
multiple seeds, saved exact recipes/code identities and source-wise outcomes,
and consultation of accumulated evidence before new tests. Added
`experiment_results_index.md` as the entry point for existing results and
pending work. The 06b reconstruction now includes matched 2D/4D across five
seeds (ten total runs); 4D is a candidate, not an assumed requirement. See
`06b_experiment_log.md` for exact matrix and reproduction limitations.

## Reconstructed 06b dimensionality comparison — ten runs complete

All 2D/4D × five-seed runs completed on the protected matched validation
split. Exact recipes and deliberate historical differences remain as above.
AE sees all four sources; propagator supervision and selection use Reid and
low-T only. Seeds 3456456/123/456/786/2026. All metrics below retain 20/20
validation networks per source per seed. Frame-100 p-ratio R² (mean ± SD):

| Evaluation | Dimension | Reid | low-T | mixed-T | noisy-LJ |
|---|---|---|---|---|---|
| ae | 2 | 0.926 ± 0.045 | 0.966 ± 0.011 | 0.893 ± 0.032 | -0.144 ± 0.651 |
| ae | 4 | 0.938 ± 0.026 | 0.971 ± 0.010 | 0.932 ± 0.026 | -0.641 ± 1.584 |
| rollout | 2 | 0.814 ± 0.044 | 0.947 ± 0.011 | 0.904 ± 0.012 | -10.549 ± 3.505 |
| rollout | 4 | 0.698 ± 0.084 | 0.923 ± 0.036 | 0.877 ± 0.042 | -8.219 ± 2.619 |

Strong mixed-T propagator transfer is recovered across five seeds under this
reconstructed recipe. This is new validation evidence, not an exact replay of
the historical held-out 0.901 result. 4D is not necessary: 2D has better mean
rollout R² for Reid/low-T/mixed-T, despite slightly lower AE reconstruction.
All five 2D mixed-T rollout scores fall between 0.895 and 0.923.

Noisy-LJ remains unresolved. It has no propagator supervision here, and its
AE response reconstruction is unstable already. Do not attribute its entire
rollout failure to the propagator. Five seeds share the same validation
networks; SD measures training randomness, not uncertainty across new datasets.
No initial-coordinate predictive probe was run on these new checkpoints, so
initial-response claims cannot be transferred from earlier low-data AEs.

Adopt 2D as the working baseline for the established three sources. Preserve
these checkpoints and split identities. Next isolate adding LJ supervision
on a frozen AE from repairing LJ representation; keep source scaling and
checkpoint-selection changes explicit. Compare learned motion with the same
causal-prefix baselines. Do not change dimension, AE recipe, dynamics recipe
and source mixture together. Reserve final-test evaluation until selection
is fixed. No new training launched during this synthesis.

## Successful 2D recipe: mixed-T AE exposure test

Submitted five paired-seed exclusions of mixed-T from AE fitting/statistics/
checkpoint selection. Baselines are the completed 2D 06b reconstruction runs;
propagator remains Reid + low-T only in both conditions. Mixed-T enters only
post-training evaluation on the same validation networks. Exact recipe,
known sample-budget/RNG qualifications, provenance and paths are recorded in
`06b_experiment_log.md`. Results: `06b_ae_mixed_ablation/`; pending.

## Mixed-T AE exposure ablation — five paired seeds complete

All five exclusions completed. The included conditions reuse the completed
successful 2D runs. Recipe, identical retained-source split IDs, post-training
loading of excluded mixed-T, and sample-budget/RNG qualifications are recorded
above. No mixed-T training, statistics fitting, or checkpoint-selection exposure
occurs in the exclusion condition. Propagator supervision is Reid + low-T in
both. Frame-100 validation p-ratio R² (five-seed mean ± sample SD):

| AE exposure | Evaluation | Reid | low-T | mixed-T | noisy-LJ |
|---|---|---|---|---|---|
| exclude_mixed | ae | 0.903 ± 0.051 | 0.958 ± 0.015 | 0.753 ± 0.102 | -0.011 ± 0.342 |
| exclude_mixed | rollout | 0.802 ± 0.093 | 0.913 ± 0.039 | 0.868 ± 0.033 | -12.195 ± 5.034 |
| include_mixed | ae | 0.926 ± 0.045 | 0.966 ± 0.011 | 0.893 ± 0.032 | -0.144 ± 0.651 |
| include_mixed | rollout | 0.814 ± 0.044 | 0.947 ± 0.011 | 0.904 ± 0.012 | -10.549 ± 3.505 |

All source/horizon/seed rows retain 20/20 validation trajectories. Mixed-T
AE exposure is not necessary for strong endpoint response rollout in this
recipe: excluded-source scores span 0.832–0.909 across the five seeds.
Exclusion-minus-inclusion paired endpoint change is -0.0365 ± 0.0439.
Exposure improves mean endpoint accuracy but does not establish necessity.
Five seeds share the same evaluation networks; this is development validation,
not final held-out-test confirmation or independent dataset replication.

Mixed-T exclusion rollout R² at frames 5/10/25/50/75/100 is
0.071/0.296/0.357/0.547/0.669/0.868, versus
0.079/0.311/0.404/0.574/0.732/0.904 with AE exposure. Early-time response
prediction is substantially weaker than the endpoint. AE reconstruction on
excluded mixed-T averages 0.753 at frame 100, below the rollout score; response
metric differences alone do not prove a learned denoising mechanism.

Noisy-LJ remains poor in both rollout conditions, with no propagator LJ
supervision and unstable AE response reconstruction. Next prioritize matched
simple kinematic/reference-only baselines on the successful 2D checkpoints
and controlled LJ-supervision/representation comparisons; preserve this
fully unseen mixed-T baseline. No additional training launched while collecting
these results. Exact per-source/seed metrics and paired comparisons are under
`06b_ae_mixed_ablation/`.

## Noisy-LJ AE must be repaired before adding dynamics supervision

Current mixed-T-unseen 2D AE has noisy-LJ endpoint response reconstruction R²
-0.011 ± 0.342, not adequate evidence of good mechanical reconstruction.
At user request, hold off on LJ propagator training. Submitted 40 AE-only
comparisons (eight recipes, five seeds) with no mixed-T fitting or selection.
See `06b_experiment_log.md` for exact matrix and label-use qualifications;
`notebooks/results/lj_ae_repair/` for recipes, hashes, job ledger and results.
Promote a representation only after source-wise response/strain/field evidence,
not low pooled position error or a single favorable seed.

## Partial AE reconstruction review — 28/40 completed

Collected existing completed runs while remaining jobs run; no dynamics launched.
Three recipes have all five seeds: dimension4_response, lj_edges, and
lj_edges_response. Endpoint validation response R² (mean ± seed SD):

| Recipe | Reid | Low-T | LJ |
|---|---:|---:|---:|
| dimension4_response | 0.870 ± 0.168 | 0.968 ± 0.011 | 0.210 ± 0.285 |
| lj_edges | 0.950 ± 0.016 | 0.949 ± 0.025 | -0.165 ± 0.229 |
| lj_edges_response | 0.933 ± 0.030 | 0.954 ± 0.016 | 0.113 ± 0.030 |

All source/seed endpoints have 20/20 valid predictions. Other recipes are
incomplete; their means cannot establish a winner. These are AE reconstructions,
not rollouts, and no recipe yet establishes adequate LJ response preservation.

New saved-prediction diagnostics (`scripts/diagnose_lj_ae_predictions.py`)
write per-seed/source/frame and aggregate CSVs plus input SHA-256 provenance
under `lj_ae_repair/prediction_diagnostics*`. No calibration is fitted. For
completed dimension4_response, mean per-seed LJ bias-squared/MSE is 0.030,
response correlation 0.553, predicted/true response SD 0.726, and mean absolute
axial/transverse strain error divided by mean absolute target strain is
0.090/0.124 at frame100. Thus constant bias alone does not explain failure.
For lj_edges_response the SD ratio is 0.423 (compressed response variation),
correlation 0.492, and relative strain errors 0.103/0.119.

LJ response R² at frames5/10/25/50/75/100 is
-17.255/-4.991/-0.450/0.117/0.231/0.210 for dimension4_response and
-3.452/-1.201/-0.149/-0.069/0.114/0.113 for lj_edges_response.
Early small-strain ratios are especially weak. These descriptive results motivate
a controlled strain-aware reconstruction objective (avoid directly dividing by
near-zero axial strain) and comparison across frames, rather than assuming
more latent dimensions or endpoint checkpoint selection resolves the problem.
This is a follow-up hypothesis, not a verified mechanism or launched experiment.
Mixed-T remains post-selection diagnostic only; no final-test data touched.
The automatic collector now refreshes these diagnostics when the matrix ends.

## Dynamics-only requirement and matched 6D/8D capacity study

User clarified the project-wide scientific requirement: learn from network
states/trajectories and structure, never add p-ratio, strain, or expert-derived
supervision. Expert observables remain post-training diagnostics, not training
inputs, objectives, or checkpoint/model-selection criteria. This supersedes the
previous proposed strain-aware objective; no such objective was implemented.
Historical response-selected AE and propagator runs remain recorded, but are
not compliant evidence of fully observable-blind learning. Future propagator
baselines also need state/trajectory-based selection.

New AE capacity matrix: dimension6 and dimension8, each seeds
3456456/123/456/786/2026. Matched against equal_source (2D) and dimension4:
width96, 32 tokens, four stored edge channels, normalized displacement input
and target, equal-source/graph MSE, worst-source validation reconstruction
checkpoint, 40 epochs/patience8, batch32, lr1e-4, weight decay1e-5, frames0–100,
20 train/20 validation each Reid/low-T/LJ. Mixed-T excluded until post-fit
validation; final test untouched. Disable the diagnostic per-epoch p-ratio
callback; it never selected checkpoints in the matched 2D/4D controls.
No additional expert features or LJ edge augmentation. No propagator training.
Prior 8D evidence used different width/edges/splits/source exposure and cannot
substitute for this matched comparison. Dimensional capacity remains a hypothesis.

Recipes/results/job ledger: `notebooks/results/lj_ae_capacity/`; frozen source
and hashes in `code_v1`. Runner/submitter/collector:
`scripts/{run,submit,aggregate}_lj_ae_capacity.py`. Configuration assertions
passed for all ten variants/seeds; shared mendels_q, 8 CPUs/32GB, no host pins.
Existing and new collectors rank only state reconstruction among eligible
non-response-selected recipes; p-ratio/strain tables remain diagnostic only.

## Completed AE collection and pipeline audit — 2026-09-07

All40 lj_ae_repair and all10 lj_ae_capacity runs completed; aggregates refreshed.
Matched LJ endpoint reconstruction diagnostic R² mean±SD: D2 -.202±1.435,
D4 .092±.235, D6 -.089±.391, D8 -.170±.620; five seeds and20/20 valid each.
Position MSE remains approximately8–9e-6. Full source-wise coordinate and
response summary: lj_ae_capacity/matched_dimension_summary.csv. More dimensions
alone have not resolved this recipe; no conclusion of dimensional impossibility.
All use historical uncorrected edge orientation; 6/8 disable diagnostic callback
while2/4 had callback without response-based selection. Expert metrics are
post-fit diagnostics only, not model-selection criteria.

Pipeline audit saved at docs/research/latent_pipeline_audit.md. Default cache
configuration matching is now enabled and regression-tested; current study
force-training means stale reuse does not explain its outcomes. Targeted active
pipeline suite:17 passed plus6 subtests, including batch-vs-single parity,
normalization inversion and future-data independence at fixed latent. Three
separate legacy simulator tests failed on stale argument signatures; recorded
rather than hidden. Edge-reversal corrected controls and architecture study
remain delegated to Terra, with smoke failures and retries preserved.
