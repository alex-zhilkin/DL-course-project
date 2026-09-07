# Deeper information and learning-objective audit

## Findings from the current evidence

The original frozen AE does not perform iterative neighbor-state message
passing. Its incident-edge aggregation is linear after a mean; its pyramid
pools N nodes to100 to20 toD through attention with linear values and no
inter-stage nonlinear feed-forward blocks. Attention is nonlinear, so this is
not a claim the whole model is linear. Near-uniform repeated pooling can lose
variation; whether trained models actually do so requires attention diversity
and sensitivity measurements. Increasing D cannot recover information already
lost upstream. Current working code may include Terra's new experiments;
these observations describe lj_ae_repair/code_v1.

Training-history audit: source-balanced 2D (four completed seeds) has normalized
LJ train/validation MSE .310/.339; 4D (four) .300/.327; wider (four) .295/.337;
extra edges (five) .303/.330. Source-wise CSVs are saved in
lj_ae_repair/training_gap_diagnostic*. Training losses are online epoch averages,
not frozen-checkpoint reevaluations. This suggests fitting/compression/objective
limitations deserve priority over explaining everything as overfitting. It does
not establish noise or an irreducible floor. Cross-source normalized losses
still depend on the shared training normalizers.

An implementation issue also needs an isolated control: endpoint reversal is
applied after standardization in the frozen AE. For directional feature e,
u=(e-mu)/sigma, the reversed normalized feature is -u-2mu/sigma, not -u.
Saved equal_source_s456 reference directional means are .1502/.2788 and SDs
.8870/.8539, so the difference is nonzero. The data normalization audit can pass
while this later model operation is inconsistent. Algebraic reproduction:
scripts/audit_ae_edge_reversal.py and ae_information_audit/edge_reversal.json.
Downstream effect is unmeasured. Terra has been asked to separate the correction
from architecture comparisons and preserve old snapshots.

## Experiments that distinguish explanations

1. **Representation destroyed before the bottleneck.** Compare matched
nonlinear neighbor encoders and direct pooling against the pyramid. Inspect
attention entropy AND diversity (uniform attention alone is not proof of
failure). Test edge-orientation and node-permutation consistency. At fixed D,
improved train and validation state reconstruction supports richer encoding;
no improvement argues for looking elsewhere. Graph-aggregation expressivity
motivation: https://arxiv.org/abs/1810.00826. General expressivity theory does
not guarantee improved simulation on these data.

2. **Decoder cannot express the displacement field.** For diagnostic-only
training trajectories, hold decoder fixed and optimize a separate latent code
per observed frame against coordinates. Compare optimized codes with encoder
codes. A large improvement implicates encoding/inference; little improvement
implicates decoder/capacity/optimization, with local-minimum caveats. Separately
fit a high-capacity per-network coordinate decoder on a tiny training subset
as an overfit check. These diagnostic fitted codes are never forecast results
or replacements for held-out encoder predictions.

3. **The latent carries only progress along an average deformation.** The
decoder sees per-node initial graph features outside D. Compare real codes,
same-trajectory wrong-time codes, within-source same-time shuffled codes and
train-mean codes. Score coordinates per network/frame. This separates dynamic
state information from static graph reconstruction. Zeroing a code alone is
an out-of-distribution perturbation and is insufficient. Existing old constant
latent failures do not settle this question on the new recipe. Inspect
within-network temporal and between-network code variation separately, not
pooled PCA alone. Count static context honestly in claims of compression.

4. **Snapshot reconstruction rewards the wrong information for prediction.**
A small AE may allocate capacity to high-amplitude fluctuations rather than
persistent predictive structure. Compare reconstruction-only training against
reconstruction plus observed future-position prediction at several lags, using
causal inputs. No p-ratio/strain targets. A time-lagged objective can encourage
predictive structure without naming a slow mode, but cannot guarantee the
slowest mode: https://arxiv.org/abs/1710.11239 and the important limitation
https://arxiv.org/abs/1906.00325. Preserve instantaneous reconstruction as a
separate outcome; improved forecast means can coexist with lost fluctuations.
This follows AE assessment rather than prematurely broadening LJ propagation.

5. **The compressed state is not sufficient to predict its future.** Compare
z_t-only with a short observed history under matched prediction budgets; also
compare full-state/history baselines. Better history-conditioned prediction
supports missing state information but changes the observation budget. Keep
initial-structure-only prediction separate. A recurrent model should learn its
memory from data, without inserting velocities, forces or a physical law as
expert supervision. Distinguish one-step ease from free-rollout stability.

6. **Sources compete for limited shared capacity.** Compare matched shared
and LJ-only models with equal LJ samples/updates and a small shared trunk with
learned routing from graph/state features only, if simpler tests support
interference. Do not gate with source identity or temperature. Log source-wise
gradient agreement, but negative cosine alone does not prove harmful transfer.
Historical larger source-specific models did not automatically solve LJ, so
this is lower priority than fixing information loss and checking decoder fit.

## Priority

First establish orientation-consistent preprocessing and meaningful tiny-data
fit. Then distinguish encoder vs decoder limits and complete matched capacity
and pooling/message-passing tests. Those results determine whether the next
major change should be spatial latent tokens, a richer decoder, or temporal
prediction training. Avoid launching every combination of these factors before
knowing which component is limiting. Broad experiments should answer different
questions, not merely produce a large count of jobs.

All primary model selection remains based on retained-source observed-state
errors. Mixed-T and expert observables are post-fit diagnostics; final test is
reserved. These are hypotheses and test instructions, not completed outcomes.
