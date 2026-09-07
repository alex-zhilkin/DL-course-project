# 06b experiment log

## 2026-08-21 — AE all sources; propagator Reid + low-T

- Notebook: `06b_mixed_dataset_shared_latent_rollout.ipynb`
- Seed: `3456456`; 4D latent; 20 train and 20 validation trajectories per source; frames `0..100`.
- AE training sources: Reid, de Pablo low-T, de Pablo mixed-T, noisy LJ. AE stopped at epoch 11 of 14. Validation step-100 p-ratio R² at that epoch: Reid `0.967`, low-T `0.944`, mixed-T `0.776`, noisy LJ `0.663`.
- Propagator: one-step shared Δz MLP, hidden size 64, static 16D mean-pooled reference context, equal per-source loss; training sources only Reid and de Pablo low-T. It stopped at epoch 3 of 6.
- Held-out step-100 rollout R² (30 trajectories/source): Reid `0.319`, low-T `0.943`.
- Propagator-unseen mixed-T step-100 rollout R² (30 held-out trajectories): `0.901`.
- Gradient cosine (one fixed 32-transition probe batch after training): Reid ↔ low-T `-0.149`.

Noisy LJ and mixed-T were excluded from propagator supervision. Mixed-T remained an AE-trained, propagator-zero-shot evaluation source; noisy LJ was AE-only in this run.

## 2026-08-21 — frozen all-source AE; Reid + low-T + noisy-LJ propagator with stride-5 PCGrad

- Notebook: `06b_mixed_dataset_shared_latent_rollout.ipynb`
- Frozen AE: `model_compact_edges_v8_ae_all_sources_prop_reid_lowT_lj.pt` (the all-four-source 4D AE above); no AE retraining in this experiment.
- Propagator: shared one-step Δz MLP (hidden size 64), 16D mean-pooled static reference context, equal source loss, genuinely balanced source-mixed batches, source-specific latent/Δz standardization fitted on propagator-training trajectories only, and PCGrad.
- Training sources: Reid, de Pablo low-T, and noisy LJ; 20 training and 20 validation trajectories/source; frames `0..100`; stride `5`; seed `3456456`.
- Checkpoint selection used the macro source-wise validation rollout R². Best epoch: 5 of 8; validation step-100 R²: Reid `0.720`, low-T `0.897`, noisy LJ `-0.378`.
- Held-out step-100 rollout R² (30 trajectories/source): Reid `0.749`, low-T `0.903`, noisy LJ `-0.032`.
- Propagator-unseen mixed-T held-out step-100 R²: `0.777`.
- Raw source-gradient cosine probes (32 batches): Reid ↔ low-T mean `-0.042`, negative fraction `0.500`; Reid ↔ noisy-LJ mean `0.060`, negative fraction `0.406`; low-T ↔ noisy-LJ mean `-0.117`, negative fraction `0.625`.

PCGrad projected conflicting gradients during training, but it did not make noisy-LJ's shared rollout viable. The remaining failure is source-specific noisy-LJ dynamics, rather than an unbalanced batch or an inactive PCGrad path.

## 4D historical-recipe reconstruction — submitted

The original v8 checkpoint is absent from this workspace. Git history preserves
older 2D and later 32D notebooks, not the exact 4D run. Therefore these jobs are
explicitly recipe reconstructions, not exact replays of the held-out 0.901 score.

Documented core: 4D attention AE trained on all four sources, 20 train/20
validation trajectories/source, frames 0–100; width-64 one-step delta MLP with
16D physical mean reference context, trained and selected on Reid + low-T only.
AE cap 14 epochs and propagator cap 6. Seeds 3456456 (historical), 123, 456.

Reconstruction assumptions: AE width 96/32 tokens from older notebook lineage;
batch 32, AdamW lr 1e-4, weight decay 1e-5, patience 3; compact four-channel
stored edges; no LJ edge augmentation/indicator; current audited per-axis box
normalization; forecast origin 0, endpoint p-ratio; macro-source rollout
checkpoint selection. Use a single fresh AE-plus-propagator call without an
intermediate reseed. These settings are not all recoverable from the old log.

Deliberate split difference: first 20 current matched training IDs and the same
20 validation IDs per source. Explicit empty test split protects the reserved
final-test partition. Report every source separately, including mixed-T and LJ,
at frames 5/10/25/50/75/100. No pooled success criterion.

Frozen source and scripts: `notebooks/results/06b_4d_reconstruction/code_v1/`.
Exact per-run recipes and completion markers are saved under `d4_s<seed>/`.
Job ledger: `notebooks/results/06b_4d_reconstruction/jobs.jsonl`. Jobs use
`mendels_q`, shared placement, eight CPUs, 32 GB, no host pins. Compilation and
PBS syntax checks passed. Results pending. Establish this 4D baseline before
adding LJ supervision or changing latent dimension; no 8D substitution.

## Matched 2D versus 4D expansion

User clarified that 4D is not a requirement: test whether 2D suffices and use
available shared cluster capacity for broader useful comparisons. Expand to
2D/4D × seeds 3456456/123/456/786/2026 (ten conditions). Retain the three
already-submitted 4D runs and submit only seven missing conditions. All source
and training settings remain fixed apart from latent dimension/model seed.
`code_v2` generalizes the runner's dimension argument; its underlying `src/`
SHA-256 hashes were verified identical to `code_v1`. Hash manifests are stored
with each code snapshot. The original-recipe limitations still apply.

All outcomes, including negative results, remain reusable under
`notebooks/results/06b_4d_reconstruction/` (directory name retained for existing
runs). `aggregate_06b_dimension_reconstruction.py` writes completion status,
source-wise raw/aggregate results, and within-seed 2D-minus-4D contrasts.
Do not infer dimensional necessity from a single seed or pooled score. Prefer
2D when source-wise validation supports it; do not use reserved test for choice.
New jobs use `mendels_q`, shared placement, eight CPUs, 32 GB, no host pins.

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

## Mixed-T AE exposure ablation — submitted

Question: does the successful 2D reconstruction still transfer to mixed-T when
mixed-T is excluded from AE exposure as well as propagator supervision?
Reuse five completed includes-mixed-T baselines; add five excludes-mixed-T runs
with seeds 3456456/123/456/786/2026. Each new run copies its paired baseline
recipe, removes only mixed-T from the fitting mixture, and changes the output
path/labels. Underlying frozen training/model code hashes match `code_v2` of
`06b_4d_reconstruction`. Retained-source split IDs and every training
configuration value were verified unchanged (apart from cache path).

AE: 2D, width 96, 32 tokens, 20 train/20 validation networks per retained
source (Reid/low-T/LJ), frames 0–100, maximum 14 epochs, patience 3, batch 32,
lr 1e-4, weight decay 1e-5. Propagator: unchanged width-64 delta MLP, mean
physical context 16, only Reid + low-T supervision and selection, max 6 epochs,
patience 3, same optimizer/seed/forecast origin and evaluation horizons.
Raw inputs with audited in-memory normalization; no reserved test data.

The excluded mixed-T source is loaded only after AE and propagator fitting and
checkpoint selection finish. Mixed evaluation has zero training/test rows and
the same 20 validation IDs as the baseline. It cannot affect AE statistics,
weights, or checkpoint selection. Combined CSVs contain all four sources;
`bundle.pt` is saved by the training call before the extra mixed-T evaluation,
so its cached evaluation rows cover only the retained sources. The full mixed-T
evaluation specification is preserved in the run recipe.

Qualification: source removal reduces examples/updates per epoch and changes
AE-fitted normalizers and random-number consumption. Equal model seeds do not
ensure identical propagator initialization after different AE runs. This tests
whether the same exclusion recipe works, not universal necessity or an isolated
mechanism for any failure. All results must remain source-wise with counts.

Results/ledger: `notebooks/results/06b_ae_mixed_ablation/`. Frozen scripts/source
and SHA-256 manifest in `code_v1/`. Collector:
`scripts/aggregate_06b_ae_mixed_ablation.py` reports source-wise AE/rollout scores
and paired excluded-minus-included contrasts. Five new shared-node PBS jobs,
`mendels_q`, eight CPUs, 32 GB, no host pins. Compilation, shell syntax, exact
configuration/split assertions and source-code hash checks passed. Results pending.

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

## Noisy-LJ representation repair before dynamics — AE-only matrix submitted

User required repairing noisy-LJ AE reconstruction before adding LJ propagator
supervision. The successful mixed-T-unseen 2D recipe's LJ AE endpoint R² is
-0.0108 ± 0.3422 over five seeds (all 20/20 validation networks), versus Reid
0.9028 and low-T 0.9581. It does not establish adequate LJ response preservation.
No new LJ propagator training is part of this study.

Eight variants × seeds 3456456/123/456/786/2026 = 40 AE-only runs. Every run
uses the corresponding excluded-mixed-T recipe and retained-source split IDs:
20 training/20 validation networks each from Reid, low-T, LJ; frames 0–100;
batch32, lr1e-4, weight decay1e-5, physical reference context, normalized
in-memory coordinates. Mixed-T is loaded only after checkpoint selection,
with zero training/test entries and the same 20 validation networks.

| Variant | Dimension / hidden | Change / selection |
|---|---|---|
| longer | 2 / 96 | Original node-pooled loss and val-loss selection; cap14→40, patience3→8 |
| equal_source | 2 / 96 | Equal source/graph reconstruction objective; worst-source reconstruction selection |
| response_selection | 2 / 96 | As equal_source; worst-source frame-100 p-ratio R² selection |
| wider | 2 / 128 | As equal_source, increase hidden width only |
| dimension4 | 4 / 96 | As equal_source, increase latent dimension only |
| dimension4_response | 4 / 96 | As dimension4; response selection |
| lj_edges | 2 / 96 | As equal_source; fifth LJ edge-indicator channel and LJ graph-distance-three augmentation |
| lj_edges_response | 2 / 96 | As lj_edges; response selection |

All new runs use cap40/patience8; early stopping may differ. Response-selection
variants explicitly use retained-source validation response labels for checkpoint
selection; gradient targets remain displacement reconstruction. Other variants
select using reconstruction loss. Mixed-T is not a selection source. The
`equal_source` comparison changes objective weighting and checkpoint criterion
together, so it is a recipe comparison rather than an isolated attribution.

Save per-trajectory/source/frame position MSE, axial/transverse strain errors,
p-ratio R²/MAE/target variance, valid/total counts at 5/10/25/50/75/100. Rank
fully completed five-seed variants on worst retained-source endpoint mean R²,
requiring all 20/20 valid predictions; preserve all per-source/frame results.
An improved rank alone is not proof the AE is fixed. Evaluate whether LJ
response improves reliably without losing Reid/low-T, and inspect earlier
frames and field/strain errors before advancing to dynamics. Mixed-T results
remain post-selection diagnostics.

Exact recipes, frozen training source/scripts and code SHA-256 manifest:
`notebooks/results/lj_ae_repair/`. Job ledger `jobs.jsonl`; collector
`scripts/aggregate_lj_ae_repair.py`. Jobs use `mendels_q`, shared single-node
placement, eight CPUs, 32 GB, no host pin. Source exclusion, retained split IDs,
configuration expansion, compilation and PBS syntax checks passed. No model
implementation was modified; these use existing supported AE training options.
No repair success claimed yet; results pending.

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

Deeper architecture audit: `docs/research/shared_dynamics_deeper_audit.md`. Saved history diagnostics suggest LJ training error is also high (~0.30 normalized MSE vs ~0.33 validation); no irreducible-noise claim. Identified nonzero-mean standardized-edge reversal inconsistency in frozen model; algebraic check `scripts/audit_ae_edge_reversal.py`, downstream impact pending. Terra notified to isolate correction from architecture changes. Proposed encoder-vs-decoder latent optimization, latent-use controls, and predictive temporal objectives without expert supervision.

## Automatic AE reconstruction collection

Collected 40/40 completed runs. Source-wise results, seed SDs, counts,
field/strain errors and a continuation packet are under `notebooks/results/lj_ae_repair/`.
Read `review_packet.md` and the underlying recipes before making a scientific claim
or advancing to LJ dynamics. Automatic collection is not a declaration of success.

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
