# Latent simulator: evidence and next experiments

2026-09-05. Repository audit of saved notebook outputs, CSVs, experiment logs,
and implementation. No notebooks, training, inference, or numerical experiments
were executed. Saved outputs are historical evidence, not fresh reproduction
of the current working tree. Existing user changes were preserved.

## What the evidence supports

The promising paper question is: **When does a low-dimensional representation
that organizes mechanical response also contain enough information to predict
its evolution across different interaction laws?**

There are three distinct claims to establish: a structural response coordinate
at the initial frame; a representation that reconstructs subsequent deformation;
and a causal dynamical model that evolves that representation. Success at one
does not establish the others.

### The existing 2D narrative

The saved output of `06_mixed_dataset_shared_latent_space.ipynb` identifies a
2D shared AE trained on three trajectories total: one each from Reid, dePablo
low-T, and dePablo mixed-T. The displayed cache is
`../results/06_mixed_dataset_shared_latent_space_boxnorm/position_normalized_3source_e7e5a33f064d.pt`.
Current configuration: seed 786, ten validation trajectories/source,
normalized displacement input/target, compact 4D edges, physical reference
context, frames through 199. This is one trajectory **per source**, not one
trajectory total, and each trajectory supplies many training frames.

The held-out initial-coordinate correlations are:

| Source | Test networks | r(z0 at frame 0, final p-ratio) | r(z1 at frame 0, final p-ratio) |
|---|---:|---:|---:|
| Reid | 91 | 0.875 | 0.872 |
| dePablo low-T | 189 | 0.431 | -0.898 |
| dePablo mixed-T | 289 | 0.391 | -0.893 |

Source: [descriptor correlations](../results/06_mixed_dataset_shared_latent_space_boxnorm/network_latent_global_descriptor_correlations.csv)
and notebook cell 17, counting cells from zero.

Thus one coordinate, z1, orders response within all three families, with a
different orientation for Reid. This supports structural response information
in the initial latent. It does not establish a universally calibrated scalar.
The shared linear readout using only initial z has test R² -0.427 / 0.929 /
0.924 for Reid / low-T / mixed-T. Its pooled R² of 0.813 hides Reid's failure.
Adding trajectory descriptors gives 0.898 / 0.960 / 0.947, but those descriptors
include observed latent changes/slopes and therefore do not demonstrate
initial-frame forecasting.

Source: [readout feature comparison](../results/06_mixed_dataset_shared_latent_space_boxnorm/shared_probe_feature_set_scores.csv).
Readouts use labeled validation networks for calibration: distinguish that
label budget from the AE's one-trajectory/source training budget. Notebook
cell 27 also reports a family-conditioned static readout with test R²
0.623 / 0.971 / 0.960; this is a different, calibrated readout claim.

The same notebook reports r=0.816 between temperature and detrended latent
fluctuation magnitude on 289 held-out mixed-T networks. This is useful
descriptive evidence for a response-plus-fluctuations narrative, but does not
by itself show that temperature is identifiable from the initial structure.

### What happens above two dimensions

The 06b log records an all-four-source 4D AE with validation reconstruction
p-ratio R² at frame 100 of 0.967 / 0.944 / 0.776 / 0.663 for Reid / low-T /
mixed-T / noisy-LJ. Recipe: seed 3456456, 20 train and 20 validation
trajectories/source, frames 0–100. This establishes that a compact shared
representation can preserve response in all four sources in at least one run.
It is validation evidence, not a matched dimension sweep or a rollout result.

In contrast, the shared 8D and 16D failures in [07b's log](07b_experiment_log.md)
show that increasing the bottleneck is insufficient on its own. Splits,
geometry, objective, training budgets, and conditioning changed between runs;
their differences cannot be attributed solely to latent dimension.

The saved `09_four_source_standard_ae_pca.ipynb` run is actually **three-source**:
mixed-T is commented out in both configuration and displayed split coverage.
Its recipe is seed 123, 6D attention AE, hidden width 128, 32 decoder latent
tokens, batch size 32, 30/30/60 training trajectories for Reid/low-T/LJ,
30 validation/source, 100 training frames, ordinary mean-gradient optimization,
unbalanced mixed batches, normalized displacement input/target, and 5D edges
with LJ graph-distance augmentation through three spring hops. Saved cache:
`latent_cd8d92300974ab08.pt`; best epoch 9, normalized validation loss 0.223656.

| Source | Test networks | AE p-ratio R², frame 100 | AE p-ratio R², frame 150 | Best raw initial-coordinate r with final p-ratio |
|---|---:|---:|---:|---:|
| Reid | 42 | 0.892 | 0.732 | -0.511 (z4) |
| dePablo low-T | 140 | 0.977 | 0.930 | 0.797 (z0) |
| noisy-LJ | 110 | 0.468 | -2.407 | 0.140 (z2) |

These are saved cells 4 and 6. Best-coordinate correlations were selected on
the same test sample and are exploratory, not independently validated probe
performance. Frame 150 lies outside the configured AE training-frame range.

Cell 5 reports PC1=0.7838196 and PC2=0.21585436 of latent variance: together
99.9674%. This is a nominally 6D latent concentrated near a linear plane in
that pooled sample. It neither proves intrinsic dimension two nor proves the
remaining directions irrelevant: latent scale and decoder sensitivity matter,
and pooling sources/frames can obscure within-source structure. This PCA was
fit on test frames, suitable for exploratory visualization but not a frozen
training-derived predictive representation.

### The strongest newer shared rollout inspected

`08_four_source_nash_mtl_autoencoder.ipynb` currently uses ordinary mean-gradient
training and three sources, despite its filename and some headings. Saved
cell 20 contains this matched AE-versus-rollout comparison on 30 test
trajectories/source:

| Source | AE R² at step 100 | Rollout R² at step 50 | Rollout R² at step 100 | Rollout R² at step 150 |
|---|---:|---:|---:|---:|
| Reid | 0.9568 | 0.9333 | 0.9417 | 0.8820 |
| dePablo low-T | 0.9769 | 0.9508 | 0.9633 | 0.9451 |
| noisy-LJ | 0.6296 | 0.3294 | 0.0088 | -0.4275 |
| dePablo mixed-T | not included | not included | not included | not included |

Current source specifies seed 456456, a 6D AE, width 192, 46 decoder tokens,
30/30/60 train and 30 validation trajectories, 150 AE training frames,
batch 32, normalized displacement, physical reference context, and the same
5D augmented edge scheme. The propagator is a width-96 `history_mlp` with
three-frame history, mean 16D static context, lr 1e-4, pooled source loss,
and no source-ID or temperature input. AE selection uses the minimum source
sum of validation p-ratio R² at 50 and 150; propagator selection uses its
validation loss. The code has `multistep_horizons=[1]`; a printed
`training_unroll: 5` is stale and must not be used as recipe evidence.
Saved history contains 40 propagator epochs while current configuration allows
50, so an exact replication still requires reconciliation with saved bundle
parameters. No bundle was deserialized in this audit.

Noisy-LJ's AE retains substantially more endpoint response than its propagated
state on this same subset. That is evidence to investigate dynamics as well
as representation. AE reconstruction is an oracle-state diagnostic, not a
strict mathematical upper bound on any possible propagated prediction.

The AE-only cell 10 uses larger test samples and reports LJ frame-100 R²
0.4681. Do not substitute that number for the 30-trajectory matched AE value
0.6296 when measuring the propagator gap. The latent correlation cells combine
train, validation, and test frames; they are descriptive and not held-out probes.

## Mechanistic questions worth testing

1. **Does the latent encode deformation more readily than response?** In 08,
   noisy-LJ's strongest reported single-coordinate correlation with y-strain is
   0.945, but with final p-ratio only 0.369 across all frames and splits. In 09,
   its best initial response correlation is 0.140. Inspect transverse and axial
   strain separately, at fixed frames and within source, before interpreting
   a colored latent trajectory as a mechanical response coordinate.
2. **Is the reduced state missing motion or interaction information?** Compare
   held-out predictions of the next latent increment from current z, z plus
   observed history, and z plus graph-derived context. If adding history lowers
   conditional prediction error, that supports an incomplete instantaneous
   state. Long-rollout drift alone does not distinguish that from accumulated
   approximation error or extrapolation.
3. **Does graph preprocessing describe the relevant LJ interactions?**
   `src/lss/data.py::append_lj_edge_indicator` constructs pairs at spring-graph
   distances two and three and updates their geometry. This is not a physical
   distance-cutoff neighbor list. Audit coverage of actual force-bearing pairs
   before concluding that the encoder sees the necessary physics. Verify the
   original simulation's exclusion and noise rules; the dataset's name alone
   does not establish stochastic dynamics or an irreducible noise floor.
4. **How much information bypasses z?** `NodeDeltaAttentionAutoEncoder.decode`
   consumes nodewise reference embeddings h0 as well as z. The architecture is
   a low-dimensional evolving state conditioned on the reference graph. It
   does not compress the entire graph into two or six numbers. A training-mean
   z control with each graph's own h0 measures how much reconstruction can come
   from reference context alone.

Memory after reduction has precedent in [Data-Driven Discovery of Closure
Models](https://epubs.siam.org/doi/10.1137/18M1177263). Noise-corrupted training
inputs improved long-rollout behavior in [Learning to Simulate Complex Physics
with Graph Networks](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html).
These motivate controls; neither establishes the cause of this project's LJ
failure or novelty of the proposed paper.

## Next experiment sequence

### 1. Establish the representation result with a matched comparison

Use all four sources explicitly, fixed trajectory identities, and independent
split/model seeds. Start with 30 training and 20 validation trajectories/source,
frames 0–100, and latent dimensions 2, 4, 6, 8. Hold the AE architecture,
optimizer, normalization, interaction schema, and checkpoint criterion fixed;
use equal source weighting. Use seed 123 for an initial screen, then repeat
surviving comparisons with model seeds 456 and 786 on the same split. This is
a proposed recipe, not a completed experiment or a claim of optimal settings.

For each dimension compare the shared model with the same source-specific
training budgets. First establish the cost of sharing; only then isolate extra
LJ data or gradient arbitration. Use validation for selection and preserve a
fresh confirmatory test partition, since historical test scores have already
guided many choices. Group by underlying network identity if datasets contain
multiple temperatures/noise realizations of the same network.

Report source-wise displacement error, axial/transverse strain error, p-ratio
R²/MAE, and valid/total counts at 10, 50, and 100. Keep negative R² visible.
P-ratio has denominator sensitivity at small deformation: report the strain
threshold and target variance alongside early-frame results. Do not choose an
AE solely from pooled reconstruction loss. Declare any p-ratio-based checkpoint
selection as use of response labels even when the gradient objective is pure
reconstruction.

### 2. Identify the mechanical coordinates before fitting a bigger propagator

Fit PCA and readouts on training/calibration networks; freeze them for held-out
evaluation. Examine frame-zero structure separately from temporal increments.
Compare one fixed scalar projection, the full initial z, and observed-prefix
features. Select coordinates on validation, report Pearson/Spearman and
calibrated prediction R² separately, and bootstrap networks rather than frames.
Control for source, frame/progress, temperature, and simple reference-graph
descriptors. Show within-temperature results for mixed-T.

Decode with only the first k training-derived PCs while preserving each graph's
reference context, then measure source-wise reconstruction and response. This
tests whether low-variance latent directions carry useful information. Include
the constant-z/reference-only control. Test a local, in-distribution latent
edit against decoded response before calling an axis a controllable mechanical
coordinate; a correlation alone is insufficient.

### 3. Separate missing state from rollout instability

Freeze a shared AE that preserves every source adequately. On identical splits
and with the same causal prefix through frame 5, compare a simple delta MLP,
the existing two-observation (1,5) residual formulation, and a rolling-history
model. The 07b AQ result makes (1,5) a useful control, not a guaranteed LJ fix.
Compare each shared propagator with source-specific propagators on the same
frozen AE. If LJ fails even alone, interference is not sufficient to explain it.

Then change only the training horizon (1 versus 5); separately test a
noise-perturbed-state objective with amplitude selected on validation. Measure
teacher-forced increment error, autoregressive latent drift, decoded strain,
and p-ratio on the same trajectories. Select by source-wise rollout validation
with a worst-source criterion; evaluate final test trajectories only after
selection. Keep the initialization frame, forecast duration, endpoint frame,
physical sampling interval, and imposed strain schedule explicit.

Only if these diagnostics show insufficient motion information should the
representation objective change to preserve a causal velocity/history state.
If the original simulation is stochastic, independently repeated trajectories
at fixed structure/conditions are needed to estimate predictable response
versus realization variability. That could motivate a distributional simulator,
but the current audit does not establish the need for one.

## Paper shape

The current evidence supports an opening result about mechanical response
emerging in a small, reference-conditioned latent with very few training
trajectories. The LJ extension can ask whether the coordinates that reconstruct
deformation are also sufficient for transferable dynamics. A successful
extension needs all-four-source results, a matched dimension comparison, a
mechanistic diagnosis of the missing information, and matched simulator
baselines with both response and field errors. Report speed only after timed
measurements including initialization/encoding/decoding costs.

Possible figure sequence: initial response ordering and calibration; shared
representation versus latent dimension; source-wise geometry with controlled
mechanical probes; matched oracle-AE and rollout curves; the intervention that
repairs LJ and its controls. Use inline Editorial styling, no saved figure
files unless requested. Universal all-four-source rollout success and a strong
initial noisy-LJ response coordinate remain open results.
