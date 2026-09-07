# Experiment results index

Research objective: compact shared mechanical-response coordinates and reliable
source-wise rollouts, including noisy-LJ and mixed-T transfer. Initial response
prediction, deformation reconstruction, and causal rollout are separate claims.
Latent dimension is open: explicitly compare 2D and 4D rather than assuming 4D.

## Read before choosing experiments

- `latent_matched_study_log.md`: current controlled studies, recipes, source-wise results, confounds, and scheduling changes.
- `06b_experiment_log.md`: historical mixed-T successes and current reconstruction attempts. Historical 0.901 mixed-T rollout used a 4D all-four AE but Reid + low-T-only delta propagator. Original checkpoint/full recipe unavailable; recent reconstructions are not exact replays.
- `07b_experiment_log.md`: historical source-wise dynamics/AE ablations, including positive mixed-T results and invalidated experiments. Read before re-testing an old idea.
- `latent_simulator_research_audit.md`: claim boundaries, historical evidence, and research gaps.

## Current machine-readable results

Paths below are relative to `notebooks/results/`.

| Study | Status at index update | Results / ledger | Key use |
|---|---|---|---|
| Matched representation screen | Complete | `latent_matched_study/` | Dimensions 2/4/6/8, shared vs separate AEs, three seeds |
| Matched dynamics and transfer | Complete | `latent_matched_rollouts/` | Frozen 8D AE; source-wise rollout failures and successes |
| AE-unseen mixed-T | 27 rollouts complete | `latent_no_mixed/sourcewise_aggregate.csv` | Excluding mixed-T from AE and propagator did not recover its rollout |
| CPU timing | Complete | `latent_full_ae_benchmark/multinode_summary.json` | Eight threads fastest observed full-data AE setting; shared-node timing can differ |
| In-memory normalization | Passed all 160,400 frames | `in_memory_normalization_audit/*.json` | Retain raw files with audited in-memory normalization |
| Motion baselines and drift | Nine complete | `latent_science_followup/diagnostics_aggregate.csv` | Learned acceleration beats LJ extrapolation; teacher forcing improves mixed-T |
| Low-data AE and probes | Six complete | `latent_science_followup/probes_aggregate.csv` | Separate 29-network/source calibration; initial information does not imply reconstruction |
| Matched horizon and AE update budget | Eighteen complete | `latent_controlled_followup/{stability,reconstruction,probes}_aggregate.csv` | Longer horizon/repeated samples did not solve all-source rollout/reconstruction |
| 06b reconstruction and dimensionality | Ten runs complete | `06b_4d_reconstruction/jobs.jsonl`, `d{dimension}_s{seed}/` | 2D mixed-T rollout 0.904 ± 0.012; 4D not necessary; LJ unresolved |

## Interpretation and operation

Read per-run `recipe.json` and `completed.json` before interpreting aggregates.
Job ledgers may include cancelled jobs and replacements: count completed run
directories, not submission lines. Preserve seed SD and source-wise valid/total
counts; do not treat repeated fixed-AE controls as independent AE seeds.

Use `mendels_q` and shared single-node placement; no explicit host pinning.
Run broader controlled matrices when justified, save all outcomes, and avoid
repeating resolved questions without identifying the new factor. Keep reserved
final-test data untouched during selection. Do not claim missing historical
artifacts have been exactly reproduced.

## Current working baseline

Reconstructed 06b 2D AE-all4 / delta-propagator-Reid+low-T gives validation
rollout R² 0.814/0.947/0.904 for Reid/low-T/mixed-T over five seeds. Preserve
it when adding LJ. This is propagator transfer, not AE-unseen mixed-T transfer.
LJ AE response reconstruction and rollout remain poor. Exact historical replay
is still unavailable, but strong mixed-T behavior is recovered on the current
validation split. See latest 06b log entry before proposing more experiments.

## Completed AE-exposure test

`06b_ae_mixed_ablation/`: five completed 2D excludes-mixed-T AE runs paired with the five
completed successful 2D includes-mixed-T baselines. Propagator always trains
only on Reid + low-T. Mixed-T is loaded only after model selection; source-wise
AE/rollout and paired contrasts confirm successful exclusion (mixed-T endpoint rollout
R² 0.868 ± 0.033) versus inclusion (0.904 ± 0.012). See 06b log for sample-budget and RNG
qualifications. AE exposure is not necessary for strong endpoint transfer in this recipe;
early-time performance and noisy-LJ remain unresolved.

## Active priority: repair noisy-LJ AE before dynamics

User explicitly requires adequate LJ reconstruction first. The successful
mixed-T-unseen AE's LJ frame-100 response R² is -0.011 ± 0.342; it is not fixed.
`lj_ae_repair/`: 40 AE-only runs, eight supported recipes × five seeds, covering
longer training, equal-source objective/selection, response-aware checkpoint
selection, width, 2D/4D, and LJ edge information. Mixed-T remains unavailable
until after fitting/selection. Full matrix and comparison qualifications are
in the latest 06b log entry. Preserve the successful transfer baselines and
advance to LJ propagator supervision only after representation evidence.

Automatic collection job `4670235.zeus-master` waits with `afterany` dependencies
on all 40 AE-study jobs. It writes aggregates, incomplete-run status and
`lj_ae_repair/review_packet.md`, plus a collection entry in the 06b log. This
schedules result collection even if some runs fail; it does not reopen the
assistant conversation or launch further training autonomously.

Partial AE review: 28/40 completed; three full five-seed recipes still have weak LJ response reconstruction (best of these: 0.210 ± 0.285). Saved `lj_ae_repair/prediction_diagnostics*` quantify bias, correlation, response spread and relative strain errors at all six frames. See latest 06b log. Strain-aware training is the next hypothesis; no LJ dynamics launched.

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

Architecture research and overnight expansion delegated to Terra (`terra_shared_ae`) at user request. Durable instructions and primary-source hypotheses: `docs/research/shared_dynamics_architecture_brief.md`. Priorities: nonlinear neighbor message passing, pooling/capacity controls, followed conditionally by learned temporal state and rollout robustness. Broad queued AE experiments authorized after smoke checks; submission/results will be recorded by the executing agent.

Deeper architecture audit: `docs/research/shared_dynamics_deeper_audit.md`. Saved history diagnostics suggest LJ training error is also high (~0.30 normalized MSE vs ~0.33 validation); no irreducible-noise claim. Identified nonzero-mean standardized-edge reversal inconsistency in frozen model; algebraic check `scripts/audit_ae_edge_reversal.py`, downstream impact pending. Terra notified to isolate correction from architecture changes. Proposed encoder-vs-decoder latent optimization, latent-use controls, and predictive temporal objectives without expert supervision.

Consolidated execution plan: `docs/research/shared_dynamics_execution_plan.md`, assigned to Terra at user request. Stages A–F cover correctness controls, information-bottleneck diagnostics, broad spatial architectures, temporal representations/history, source interference and rollout robustness. Early independent work can queue after smoke checks; later stages depend on recorded state-based evidence.

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
