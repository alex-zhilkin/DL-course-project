# Shared dynamics execution plan

Owner: Terra (`terra_shared_ae`). User authorizes implementation and broad queued
experiments. This consolidates shared_dynamics_architecture_brief.md and
shared_dynamics_deeper_audit.md. Execute stages, record evidence, and advance
without repeated approval within this scope. No promise that queued work wakes
an assistant; PBS collectors collect results, while conditional scientific
choices require an active agent to review them.

## Common protocol

- Read AGENTS.md and experiment_results_index.md first. Inventory running and
  completed jobs; preserve old snapshots and reuse exactly equivalent controls.
- Fit Reid/low-T/LJ only. Mixed-T is post-fit transfer evaluation. Reserved final
  test is untouched. Separate any later all-four supervised experiment.
- Inputs/targets are observed states, causal histories and network structure.
  No expert response labels, strain losses, analytic forces, source-ID routing,
  temperature inputs, or expert-based checkpoint/model selection.
- Use the existing topology-disjoint split IDs and raw data with audited
  in-memory normalization. Fit all statistics on training samples only.
- Primary state metric: equal-network mean squared coordinate error, reported
  per source and frame, in normalized training coordinates and physical/current
  coordinate convention separately. Select with worst retained-source mean
  normalized reconstruction over the declared frame grid, not expert metrics.
  If reproducing existing controls, preserve their exact full-frame criterion
  and distinguish it from any new sparse-grid criterion.
- Five paired seeds: 3456456,123,456,786,2026. Screen diagnostics on a fixed small
  training subset first; confirm conclusions across seeds. Report valid/total
  counts, mean/SD and paired differences; mixed-T never decides advancement.
- Predeclare intended matrices and metrics before submission. Save code/data/
  split hashes, recipes, parameter and total latent-state counts, optimizer
  updates, wall time, peak memory, PBS IDs and completion/failure artifacts.
- Shared mendels_q, no node pins/exclusivity. Start at 8 CPUs/32GB; change only
  with measurement. Smoke-test largest representative graphs before queuing.
- A scientific gain must exceed observed seed variation and preserve each
  retained source. Report tradeoffs instead of claiming a pooled win. Define
  numerical noninferiority tolerances from state-error baselines before looking
  at candidate outcomes; never derive them from p-ratio.

## A. Correctness and frozen baseline inventory — first

1. Audit standardized edge reversal using saved statistics. Test raw reversal
   followed by normalization against model endpoint features; include canonical
   versus reciprocal edge lists, endpoint swapping and node permutations.
2. Implement a consistent correction as default behavior in new code, preserve
   frozen historical behavior, and isolate correction-only controls from
   architecture changes. Tests cover forward/backward, empty edges, dtype,
   checkpoint reload and no future-state information in decoder context.
3. Queue corrected baseline D=2/8 × five seeds (10 runs) unless equivalent
   verified jobs already exist. Existing uncorrected 6D/8D runs remain useful
   historical capacity controls, not corrected architecture controls.

Exit: correctness tests and real-data smoke pass. Failure blocks dependent
architecture submission, not independent offline diagnostics.

## B. Identify the information bottleneck — alongside A

B1. Reevaluate frozen checkpoints on a declared training subset and validation
subset at frames5/10/25/50/75/100. Distinguish online training history from actual
frozen-checkpoint training error.

B2. Freeze decoder and optimize per-frame latent codes on training examples,
starting from encoded codes plus multiple reproducible starts with fixed step
budgets. Compare reconstruction before/after, per source and D. Save fitted
codes as diagnostic artifacts only; they are neither forecasts nor unbiased
validation predictions. A positive optimization gap supports an encoder issue;
a small gap is inconclusive about decoder capacity due to local minima.

B3. Compare original codes, wrong-time codes within a trajectory, same-source/
same-time codes permuted across trajectories, and training-mean codes. Use
seeded permutations and the same target graphs. Report effect sizes, temporal
vs between-network variance, attention diversity and effective rank. Do not
infer collapse from pooled PCA or uniform attention alone.

B4. Tiny training-set overfit check using one fixed network per retained source:
existing AE versus a higher-capacity coordinate decoder. This diagnoses fitting
ability, not generalization. Evaluate normalized coordinate residuals and
convergence; no test set or expert observable needed.

Exit: diagnostic report assigns supported/unsupported/unresolved status to
encoder loss, decoder capacity, unused latent, and fitting/optimization limits.

## C. Broad matched spatial architecture matrix — after A

- Corrected baseline from A.
- Message passing depth2/4 × D2/8 × five seeds: 20 runs.
- Existing direct/single-stage pooling versus pyramid, D2/8 × five seeds:
  10 runs per chosen distinct pooling family, preferably one initially.
- Keep decoder, frame/update budget, source weighting and normalization fixed.
  Report unavoidable parameter-count/runtime changes. Do not combine all axes
  immediately. Existing controls can be reused only when exact.

B can motivate one bounded additional branch: richer decoder if code fitting
cannot recover fields; or a learned spatial-token AE if global compression is
limiting. For tokens report total evolving scalar count, not just token width.
Use five-seed confirmations and a matched parameter/budget control where useful.

Exit: collect all outcomes, identify state-reconstruction Pareto frontier, and
assess retained-source noninferiority. A top rank alone does not mean adequate.

## D. Temporal representation and state sufficiency — conditional

Prepare scripts while C runs; submit after AE assessment. If representation
remains poor, temporal AE training is a targeted representation experiment,
not permission to claim successful LJ rollout. Explicitly label this branch.

D1. Compare reconstruction-only AE, frozen-AE future-state predictor, and joint
reconstruction plus future-coordinate prediction. Start with lags1 and5 in
separate controlled recipes, fixed target/update budgets and train-derived
scales. Keep instantaneous reconstruction in the objective and report it.
D2. Compare current encoded state with causal windows of1/3/5 observed frames
or one matched recurrent model. Use a common forecast origin for fair history
comparisons and separately preserve initial-frame-only forecasts. Include a
full-state predictor control to distinguish compression from missing history.
D3. Select checkpoints by retained-source state prediction/reconstruction,
never response scores. Report teacher-forced and free predictions separately.

Exit: establish whether temporal training/history helps state prediction without
sacrificing AE reconstruction. Do not assert discovery of the slowest mode.

## E. Source interference — conditional on B/C

Compare shared versus LJ-only models with matched LJ data and optimizer exposure.
Measure source-wise gradient agreement as a diagnostic, not proof of conflict.
Only if interference is supported, test a shared trunk with learned routing
from states/graph features, without source labels. Compare matched total
parameters and report routing/capacity. Shared models should still be evaluated
on every retained source; LJ-only is a diagnostic control, not the objective.

## F. Rollout robustness — after representation assessment

On selected compliant representations, compare frozen-AE one-step baseline,
short decoded-coordinate unroll training, and controlled input perturbation.
Keep forecast origins and sampling/update budgets matched. Start one factor at
a time; five seeds for confirmations. Preserve initial/reference-only and
history-conditioned claims separately. If residual uncertainty remains after
adequate state/history modeling, compare a stochastic transition with a
likelihood/calibration evaluation; do not assume noisy-LJ is irreducibly random
or train away genuine fluctuations. No expert or force-based losses.

## Collection, decisions and handoff

Queue A/C once correctness and smoke gates pass; queue B diagnostics independently.
Schedule failure-aware PBS collectors with afterany dependencies. Collectors
must not mark partial matrices complete or auto-promote candidates. D/E/F are
conditional branches, not a promise to launch every cross-product tonight.
Keep a stage status file with planned/submitted/running/complete/blocked and
concrete reasons; save negative results and exact resumable next commands.
Update the result index and relevant logs after each collection. Terra should
report the actual launched matrix, checks, job IDs, pending branches, and next
decision evidence; do not end with only an implementation proposal when queued
work is already authorized and verified.
