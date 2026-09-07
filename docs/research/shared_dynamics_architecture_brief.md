# Shared AE + propagator: architecture research and first implementation

## Objective and constraints

Learn a shared simulator for Reid, dePablo low-T and noisy-LJ, with mixed-T
held out from fitting and selection for the primary transfer experiment.
All-four-source fitting is a separate supervised comparison, never transfer.
Learn from observed states/trajectories and existing network structure only.
No p-ratio, strain, temperature/source labels, analytic forces, conservation
penalties, or other expert-derived inputs/objectives/selection criteria.
Expert observables are post-fit diagnostics only. Do not infer that p-ratio is
already established to be a slow mode. Preserve the initial-structure-only
forecast question separately from forecasts with observed motion history.

Read AGENTS.md, experiment_results_index.md, latest 06b and matched-study logs,
and relevant 07b history before implementation. Preserve dirty user files and
frozen snapshots. Current 6D/8D five-seed jobs are already submitted under
notebooks/results/lj_ae_capacity; do not duplicate or cancel them.
Historical response-selected successes are useful diagnostics, not compliant
baselines. Re-select/retrain using state errors for compliant comparisons.

## Evidence and plausible ideas (hypotheses, not established solutions)

1. Nonlinear neighbor message passing before compression. Current
NodeDeltaAttentionAutoEncoder aggregates incident edge attributes by a mean
and a linear projection, then pools nodes; it does not iteratively exchange
neighbor node states. This may discard local configuration information even
before the 2D/4D bottleneck. Compare the existing encoder against residual
message-passing encoders with modest depth (e.g. 2 and 4), keeping the latent,
decoder, data, loss and selection fixed. Use existing edges and observed node
states; do not insert an LJ-specific interaction law. Watch dense-edge memory;
chunk messages or benchmark representative largest graphs. Permutation
consistency, edge orientation, empty graphs and finite gradients need checks.
Primary motivation: https://arxiv.org/abs/2002.09405 and
https://arxiv.org/abs/2010.03409. These papers support learned graph simulators,
not a guarantee that this particular AE will improve.

2. Larger or structured latent capacity. Complete the running 6D/8D controls.
Then consider a small learned set of spatial latent tokens rather than one
vector if global compression remains limiting. Report total evolving latent
scalar count, static decoder context, parameter count and runtime. Decoder
expansion into 32 tokens is NOT 32 independently encoded dynamic tokens.
A local/global or multiscale representation can preserve heterogeneous local
motion. Do not claim compact 2D dynamics from a two-dimensional visualization
of a much larger state. Reference:
https://proceedings.mlr.press/v202/cao23a.html (multiscale graph communication).

3. Learned temporal state and prediction-aware representations. A snapshot AE
may discard information needed for the next state. Compare fixed-AE prediction
against joint reconstruction plus future-coordinate prediction, with causal
short history or a learned recurrent state. Targets are observed positions,
not expert response labels. Start with frozen-AE controls after AE assessment;
maintain reconstruction terms to avoid sacrificing other sources. Explicitly
report observed warmup frames, parameter/update budgets, forecast origin,
free rollout and teacher forcing separately. Earlier history/multistep runs
failed in other recipes: identify the changed factor before repeating.
https://planetrl.github.io/ supports recurrent latent-state modeling; adapt
only state modeling, not rewards/actions or task-specific supervision.

4. Rollout robustness and stochastic residuals, later. Once representation
works, test controlled input perturbations and short decoded-state rollout
losses against clean one-step controls. Do not remove real fluctuations by
assuming they are measurement noise. A stochastic latent transition may be
appropriate if history-conditioned residual uncertainty remains; evaluate
state likelihood/calibration as well as mean trajectories. No claim of
irreducible stochasticity from a deterministic model's error alone.
https://arxiv.org/abs/2002.09405 discusses noise during simulator training;
https://research.google/pubs/learning-latent-dynamics-for-planning-from-pixels/
describes deterministic and stochastic latent dynamics. Neither establishes
that noisy-LJ in this repository needs a stochastic model.

## Terra assignment and deliverables

Independently assess these ideas against the actual implementation and saved
results. Write a concise ranked research memo with linked primary sources,
mechanisms, falsifiable tests, costs and risks. Implement ONE first architecture
experiment: nonlinear residual neighbor message passing for the AE, unless
inspection reveals a concrete reason this is unsuitable (document it and
choose a comparably bounded AE-first experiment). Use the existing model
registry to preserve experimental controls; no unnecessary new feature flags.

Prepare reproducible baseline-vs-architecture controls with five paired seeds,
at least a matched small/large latent comparison (2D and 8D), same split IDs,
train frames and source-balanced coordinate loss. Reuse completed equivalent
controls; label parameter-count/runtime confounds. Verify configuration expands
correctly, forward/backward and permutation/edge handling, then benchmark a
small real-data smoke run before submitting a broader matrix. Do not submit a
large matrix on failed smoke checks. CPU-only shared mendels_q, no host pins,
8 CPUs/32GB initial request with evidence-based changes if necessary.

Save exact recipes, source/split hashes, frozen code, job IDs, machine-readable
source-wise state reconstruction across frames, counts, failures and timing.
Select checkpoints and models only with state/trajectory reconstruction error
on retained sources, not expert metrics or mixed-T. Inspect post-fit diagnostic
response separately. No propagator supervision expansion until the AE evidence
is assessed. Keep final-test data untouched. Update result index and experiment
logs with facts and remaining work. Deliver code paths, validation evidence,
submitted job IDs or a concrete blocker; no promises of guaranteed success.

## Overnight expansion authorized by user

The user explicitly asks for many queued tests spanning a wide variety while
away. After representative smoke checks, expand the architecture study to
message-passing depths 2/4 × latent2/8 × five seeds, with a separate existing
pooling-family comparison when technically supported. Reuse equivalent
baselines. This suggested matrix is adjustable based on implementation and
runtime evidence, not an obligation to launch redundant or broken jobs.
Persist an intended-matrix manifest before submission and distinguish queued,
running, completed, failed and missing runs. Collect after any terminal outcome,
not only success. Do not assume a ranking declares scientific adequacy.

For honest model selection, predeclare an observable-blind coordinate criterion
across retained sources and validation frames before viewing expert diagnostics.
Report physical/in-memory coordinate MSE and train-only normalized coordinate
MSE separately. Do not tune source/frame weighting from p-ratio or strain.
Parameter counts, training updates and wall-clock budgets must accompany
architecture comparisons. Spatial tokens and recurrent history increase total
state/information budgets; disclose these rather than treating nominal latent
width as the whole capacity. Ensure decoder reference context contains only
information available at forecast initialization. Predictive architectures must
not receive true future edges/positions during free rollout.
