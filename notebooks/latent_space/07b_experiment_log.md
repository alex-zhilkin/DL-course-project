# 07b shared Reid + dePablo + noisy-LJ rollout log

## 2026-09-05 — saved-evidence research audit (no new runs)

Reviewed this log alongside saved outputs in notebooks 06, 08, and 09 for
the paper restart. No training, inference, or notebook execution was performed;
no new experiment or recomputed metric is being reported. The broader evidence,
recipe caveats, and proposed matched experiments are recorded in
[latent_simulator_research_audit.md](latent_simulator_research_audit.md).
In particular, the newer 08/09 saved runs exclude mixed-T despite four-source
filenames, and their promising higher-dimensional reconstructions should not
be recorded as successful all-four-source rollouts. Existing 07b source-wise
results and validation/test distinctions below remain historical evidence.

## Goal

Train one shared autoencoder and one shared latent propagator across Reid,
dePablo low-T, dePablo mixed-T, and noisy-LJ. The model must roll out without
future-frame inputs and perform well for every source at 100 steps; an aggregate
score is not sufficient when one source fails.

All reported values below are held-out test p-ratio R² at rollout step 100,
recomputed source-wise from each saved bundle's `rollout_rows`. Negative values
are retained. Trajectory splits are disjoint, but experiments with different
train/validation counts use different held-out test subsets and are therefore
diagnostic comparisons rather than perfectly matched benchmarks.

## Reference configuration

- Normalized position/delta domain with stored 13D edge features.
- Source-balanced training rows.
- Shared attention AE, normally 8D unless stated otherwise.
- Evaluation is a 100-step autoregressive rollout.

## Main completed experiments

| Experiment | Training recipe | Reid | dePablo low-T | dePablo mixed-T | noisy-LJ | Takeaway |
|---|---|---:|---:|---:|---:|---|
| `B_balanced_ae_stored13_train50` | 50/source; fixed velocity residual; observed `(1,5)`; mean static context; 1-step | 0.739 | 0.823 | 0.402 | -0.051 | Best balanced reference. Three sources work; noisy-LJ remains the limiting source. |
| `B_balanced_reprop50_lr2e4_batch1024` | Same reference AE; propagator retrain, lr `2e-4`, batch 1024 | 0.777 | 0.883 | 0.515 | -0.208 | Improves Reid/dePablo but worsens noisy-LJ. |
| `A_balanced_no_context_batch512` | Reference architecture without static context | 0.416 | 0.905 | -0.975 | -0.103 | Context matters for Reid and mixed-T; removing it is not a universal fix. |
| `B_more_data_stored13_train70` | 70/source; fixed velocity residual; mean context | -1.501 | 0.515 | -5.298 | -0.699 | More trajectories alone did not stabilize the propagator. |
| `B_learned_attention_context` | Learned-attention static context | 0.793 | 0.949 | -1.046 | -0.073 | Better on Reid/low-T, worse on mixed-T; attention is not the shared solution. |
| `B_source_conditioned_reprop50_lr2e4` | Source-conditioned propagator, 50/source | 0.823 | 0.950 | 0.325 | -0.114 | Source conditioning does not solve noisy-LJ. |

## Early-history experiments

| Experiment | History / objective | Reid | low-T | mixed-T | noisy-LJ | Takeaway |
|---|---|---:|---:|---:|---:|---|
| `A_four_frame_no_context_train70` | Direct window `(0,1,2,3)`; 1-step; no context | -0.781 | 0.049 | -0.369 | -1.056 | Four raw latent frames without an inductive anchor are unstable. |
| `B_four_frame_mean_context_train70` | Direct window `(0,1,2,3)`; 1-step; mean context | -0.770 | 0.064 | 0.197 | -1.018 | Context helps mixed-T modestly, but not the core failure. |
| `A_four_frame_velocity_no_context_train70` | Four-frame window plus local velocity-residual anchor | -1.600 | 0.116 | -3.018 | -23.944 | Rejected: early local velocity is an unstable anchor. |
| `A_four_frame_multistep_no_context_train20_latent6` | Direct four-frame window; 5-step closed-loop loss; no context | -0.109 | 0.744 | -1.674 | -1.823 | Closed-loop training helps low-T but does not yet generalize across sources. |
| `B_four_frame_multistep_mean_context_train20_latent6` | Direct four-frame window; 5-step closed-loop loss; mean static context | -0.152 | 0.737 | -0.857 | -1.856 | Context improves mixed-T over A, but all but low-T still fail. |
| `A_raw_four_frame_history_train20_latent6` | Raw observed `(0,1,2,3)`; one-step; no static context | -0.069 | 0.779 | -10.951 | -1.639 | Failed: removing static context does not recover Reid and causes severe mixed-T divergence. |
| `B_learned_motion_context_train20_latent6` | 6D GRU context from observed `(0,1,2,3)`; one-step; no static context | -0.068 | 0.770 | -13.414 | -1.669 | Failed: learned prefix context is nearly identical on Reid/low-T/noisy-LJ and worse on mixed-T. |
| `C_history_gated_graph_context_unbalanced_lj70` | 20/20/20/70 train; 10/source val; unbalanced; 6D GRU history; mean graph context with history gate | -1.858 | 0.413 | 0.830 | -0.667 | Only both dePablo sets work. Noisy-LJ improves versus A/B but remains negative; Reid collapses. |
| `C_history_gated_graph_context_latent10_lj40` | 20/20/20/40 train; 15/source val; 10D; 150 frames; unbalanced; 6D GRU history gate | 0.068 | 0.170 | -0.260 | 0.092 | Close but incomplete: Reid and noisy-LJ are weakly positive, mixed-T remains negative. |
| `B_multistep_train20` | Two-frame velocity residual; 16-step closed-loop loss; mean context | 0.154 | 0.537 | -0.718 | -21.687 | Long unrolls were unstable in this earlier recipe. |

## AE finding

Noisy-LJ is not only a propagator issue. In the 50/source shared-8D reference,
the AE's own noisy-LJ p-ratio reconstruction at step 100 was poor (about -0.39
on that test subset), whereas the same AE reconstructed the other families well.
Some 70/source AE runs reconstructed noisy-LJ much better on their own split,
so AE information retention remains a prerequisite to a successful noisy-LJ
rollout.

For the cached 10D AE reused by `S_velocity_mean_context_screen50`, its saved
held-out reconstruction p-ratio R² values were: Reid `0.339`, low-T `0.865`,
mixed-T `0.453`, noisy-LJ `-0.191` (75 evaluated trajectories/source). Thus
the new screen's noisy-LJ rollout failure is consistent with information already
being absent from this representation, not just a propagator training failure.

## Latest notebook experiment

The completed diagnostic used:

- 20 train trajectories for Reid and each dePablo set, 70 for noisy-LJ, and
  20 validation trajectories per source;
- 6D shared attention AE;
- direct fixed latent window `(0,1,2,3)`;
- one-step rollout training;
- mixed, unbalanced AE and propagator batches (no source balancing);
- a 6D GRU early-motion context from `(0,1,2,3)`;
- mean static graph context multiplied by a learned scalar gate derived only
  from the early-motion context.

All later steps are autoregressive. The gate began at 0.5 and was learned from
the early prefix only.

### Latest completed learned-history result — test p-ratio R² at step 100

| Source | Raw four-frame (A) | Learned 6D history (B) | Outcome |
|---|---:|---:|---|
| Reid | -0.069 | -0.068 | Both fail; no learned-history gain |
| dePablo low-T | 0.779 | 0.770 | Both work; B is marginally worse |
| dePablo mixed-T | -10.951 | -13.414 | Both fail severely; B is worse |
| noisy-LJ | -1.639 | -1.669 | Both fail; no learned-history gain |

### Latest completed history-gated, unbalanced result — test p-ratio R² at step 100

| Source | C: history-gated graph context | Outcome |
|---|---:|---|
| Reid | -1.858 | Failed; worse than the prior balanced reference |
| dePablo low-T | 0.413 | Works |
| dePablo mixed-T | 0.830 | Works |
| noisy-LJ | -0.667 | Failed, although less negative than the recent no-context runs |

This run evaluated only 10 held-out trajectories per source because its final
evaluation was explicitly capped at 10/source. The notebook now removes that
cap for future runs, so final metrics and endpoint scatters use every remaining
test trajectory. This run remains unmatched to the earlier 20/source
diagnostics.

### Latest completed 10D history-gated result — test p-ratio R² at step 100

| Source | R² | Outcome |
|---|---:|---|
| Reid | 0.068 | Weakly positive |
| dePablo low-T | 0.170 | Positive |
| dePablo mixed-T | -0.260 | Failed |
| noisy-LJ | 0.092 | Weakly positive |

This run used the old in-memory notebook settings: `batch_graphs=1024`, rollout
validation every epoch, and a final-test cap of 15/source. It is not D and does
not test the new 2048-batch, every-five-epochs, full-test configuration.

## Pending experiment

## Latest completed 3-source kinematic-target diagnostic (validation only)

`U_ae_kinematic_target_prefix3_history3_physical_context_reid_depablo` used
15 train and 15 validation trajectories per source (Reid, dePablo low-T,
dePablo mixed-T), seed `44545`, 101 frames, a 10D attention AE, compact stored
edges, and mixed unbalanced batches. The AE encoder received the current
normalized displacement plus a fixed normalized prefix `(d0, d1, d2)`. Its
six-channel standardized target was normalized `[displacement, velocity,
acceleration]`. Both AE and mean-pool static context used the preserved
physical reference graph; the propagator was a 3-frame fixed-window MLP with
observed frames `(0,1,2)`. The result is validation-only, so it is not
comparable as a held-out test benchmark.

| Source | AE p-ratio R² at 100 | Rollout p-ratio R² at 100 |
|---|---:|---:|
| Reid | 0.053 | 0.013 |
| dePablo low-T | 0.887 | 0.811 |
| dePablo mixed-T | 0.296 | 0.439 |

Relative to `S_history3_normalized_context_reid_depablo`, AE reconstruction
fell for Reid (`0.215 → 0.053`) and mixed-T (`0.777 → 0.296`), while low-T
was similar (`0.862 → 0.887`). The rollout comparison points in a different
direction: mixed-T improved (`-0.058 → 0.439`), low-T improved (`0.021 →
0.811`), and Reid slightly declined (`0.081 → 0.013`). These are **not a
physical-context ablation**: seed, encoder input, and AE target all changed
along with reference context.

### Completed controlled reference-context comparison — AE training only

`V_ae_kinematic_target_prefix3_history3_normalized_context_reid_depablo`
repeated U with the same seed `44545`, 15/15 train/validation trajectories per
source, 101 frames, 10D AE, prefix-3 input, and six-channel kinematic target.
The only changed setting was `static_context_use_physical_reference=False`:
static reference positions and edge geometry were normalized rather than
preserved in physical units.

V early-stopped at epoch 14 (best epoch 8) with normalized validation loss
`1.16792`; U's corresponding best loss was `1.15930`. Thus physical versus
normalized static context is not the cause of the poor AE fit at seed `44545`.
No V propagator or source-wise p-ratio evaluation was run, so no rollout R² is
reported for this AE-only completed run.

For comparison, T used the same normalized-context, prefix-3, kinematic-target
AE recipe but seed `456`, and reached a much lower best validation loss
`0.67683`. Comparing saved parameter dictionaries confirms that, apart from
cache path, the only T-versus-V configuration change was the split/model seed
(`456` versus `44545`). The large difference therefore comes from the selected
15-trajectory train/validation split and/or seed-dependent initialization, not
from the static-reference-context switch.

### Pending AE p-ratio learning trace

`W_ae_kinematic_target_prefix3_history3_normalized_context_reid_depablo_pratio_trace`
uses the current seed `456` normalized-context kinematic-target recipe. During
AE training it evaluates decoded validation deformation at step 100 every two
epochs and saves overall plus source-wise p-ratio R² in the AE history and
`*_val_ae_p_ratio_r2_by_epoch.csv`. The notebook plots the three source curves
after AE training. Record the completed source-wise final values and learning
curve behaviour here after the run.

### Completed physical static-graph embedding trace — AE only

`X_ae_kinematic_target_prefix3_history3_physical_static_embedding_reid_depablo_pratio_trace`
used seed `456`, 15 train and 15 validation trajectories per source, 101
frames, a **2D** attention AE, compact stored edges, normalized prefix-3 input,
and the six-channel standardized `[displacement, velocity, acceleration]`
target. `static_context_use_physical_reference=True` supplied original
reference node positions and reference edge vector, length, and stiffness to
the full graph encoder; its learned mean-pooled static embedding has dimension
16. Dynamic positions/deformation remained normalized. Validation AE p-ratio
R² was evaluated every two epochs at step 100.

The restored AE checkpoint was epoch 28 (best normalized validation loss
`0.98003`). Its source-wise validation reconstruction p-ratio R² at step 100
was:

| Source | AE p-ratio R² |
|---|---:|
| Reid | 0.923 |
| dePablo low-T | 0.971 |
| dePablo mixed-T | 0.835 |

The learning trace rose from epoch 2 values of `-0.011`, `0.241`, and `0.168`
to strong, stable values by epochs 20–30. This is a successful AE result.

X's subsequently trained plain fixed-window delta propagator used the same
physical 16D mean static embedding and observed frames `(0,1,2)`. Its
source-wise **validation** rollout p-ratio R² at step 100 was Reid `-0.101`,
low-T `0.301`, and mixed-T `0.659`. Thus the AE is not the limiting component,
but the shared propagator still fails Reid. These are validation results; no X
test rollout has been evaluated.

### Completed CPU propagator screen — fixed-window velocity residual

`Y_xae_fixed_window_velocity_residual_cpu_val` reused frozen X with the same
seed `456`, 20 train and 20 validation trajectories/source, normalized dynamic
state, physical 16D static graph embedding, and observed prefix `(0,1,2)`. It
replaced X's plain fixed-window delta MLP with a fixed-window velocity-residual
MLP: the last observed latent velocity was added as a constant-velocity anchor,
and the network predicted only a residual. It used one-step training, 20 CPU
epochs, and source-wise validation rollout evaluation at step 100 every two
epochs. No test trajectories were evaluated.

| Source | Validation rollout p-ratio R² at 100 |
|---|---:|
| Reid | -0.461 |
| dePablo low-T | 0.807 |
| dePablo mixed-T | -0.794 |

Rejected. The selected checkpoint was epoch 18 by the minimum source-wise
validation metric. The velocity anchor helps low-T but strongly harms Reid and
mixed-T, so it is not the way to repair X's rollout.

### Completed CPU propagator screen — fixed-window progress input

`Z_xae_fixed_window_progress_cpu_val` reused frozen X and exactly the same
20/20/source validation screen as Y. It returned to the plain fixed-window
delta MLP but appended normalized rollout progress to its causal input. This
is a time/phase signal only; it exposes no future trajectory values. It used
one-step training for up to 20 CPU epochs and selected from source-wise
validation step-100 rollout R².

| Source | Validation rollout p-ratio R² at 100 |
|---|---:|
| Reid | -0.134 |
| dePablo low-T | 0.322 |
| dePablo mixed-T | 0.642 |

Rejected. The selected checkpoint was epoch 6. Progress modestly changed the
dePablo trade-off but did not make Reid positive, and therefore did not improve
on X's plain fixed-window baseline for the all-source objective.

### Completed CPU diagnostic — history-gated static context

`AA_xae_history_gated_static_context_cpu_screen` reused frozen X and tested the
fixed-window history-gated-context MLP. It learns a gate from the causal
three-frame latent prefix before applying the physical 16D static graph
embedding. To keep CPU iteration feasible, it used 30 transitions/trajectory,
at most 8 epochs, and 5 validation trajectories/source for checkpoint rollout
evaluation; it is a screening result, not a benchmark.

| Source | Validation rollout p-ratio R² at 100 |
|---|---:|
| Reid | -0.228 |
| dePablo low-T | -0.128 |
| dePablo mixed-T | -0.194 |

Rejected. This more expressive context mechanism collapses the successful
dePablo behavior, so it should not be escalated to a full validation/test run.

`D_history_gated_graph_context_unbalanced_lj40` is configured but has no
results yet:

- 20/20/20/40 train trajectories (Reid / low-T / mixed-T / noisy-LJ), with
  15 validation trajectories per source;
- 10D shared attention AE; 150 frames per trajectory;
- mixed, unbalanced AE and propagator batches;
- four-frame learned history plus history-gated mean graph context;
- `batch_graphs=2048`;
- 100-step validation rollout every 5 propagator epochs;
- final R² and scatter use all remaining test trajectories, without a cap.

The CPU runner is authorized for this experiment; this changes execution
hardware only, not the data split, causal inputs, or evaluation protocol.

Run it from a CUDA-visible terminal with
`bash scripts/run_07b_cuda.sh`. The notebook itself checks CUDA availability
and aborts instead of silently using CPU.

The same exact D recipe can run without a notebook kernel on CPU with
`python scripts/run_07b_experiment.py --device cpu`; it writes source-wise
rollout R² CSV files next to the experiment bundle.

## Completed small validation screen

`S_velocity_mean_context_screen50` reused the completed 10D shared AE and
trained only a two-frame fixed-velocity-residual propagator with mean static
context: 5 train + 5 validation trajectories/source, 30 transitions/trajectory,
at most 8 epochs. It was validation-only: no test rows were evaluated.

### Step-50 validation p-ratio R²

| Source | R² | Outcome |
|---|---:|---|
| Reid | 0.341 | Positive |
| dePablo low-T | 0.765 | Positive |
| dePablo mixed-T | -0.978 | Failed |
| noisy-LJ | -7.437 | Failed severely |

This is a deliberately tiny screen, so its magnitudes are not a benchmark.
It does reject the simple explanation that returning to the fixed-velocity
residual alone restores all four sources when paired with the current 10D AE.
The numerical report is
`notebooks/results/07b_mixed_reid_depablo_lj_context_ablation/S_velocity_mean_context_screen50_val_step50_p_ratio_r2.csv`.

## Incomplete compute-limited attempt (2026-08-20 to 2026-08-21)

`S_ae16_unbalanced_lj30_screen50` was a validation-only, shared 16D attention
AE screen: 12/12/12/30 mixed, unbalanced training trajectories and 8 validation
trajectories/source, 51 frames/trajectory, `batch_graphs=256`, CPU only. It was
allowed to run for the agreed five-hour budget, then stopped before a cache was
saved; consequently no rollout or test result exists for it.

The cleaned `07b_mixed_reid_depablo_lj_context_ablation.ipynb` now reproduces
this exact screen, defaults to CUDA when available, and reports only source-wise
validation reconstruction R² at step 50. It deliberately trains no propagator
until the shared AE cache has been completed.

Its reported normalized validation reconstruction losses were:

| Epoch | Validation loss |
|---:|---:|
| 1 | 0.787365 |
| 2 | 0.690963 |
| 3 | 0.632904 |
| 5 | 0.630630 |
| 6 | 0.624741 |

The decreasing loss shows the architecture was optimizing, but it is not
evidence of source-wise reconstruction quality or shared rollout success. The
planned frozen-AE propagator screen is implemented in
`scripts/run_07b_ae16_velocity_screen.py`, but must wait for a completed AE
checkpoint.

## Completed small validation screen — 16D, 12/12/12/30

`S_ae16_velocity_mean_context_source_id_screen50` trained a 16D shared
attention AE for 20 epochs on 12/12/12/30 trajectories, with 51 frames per
trajectory. Its propagator used the fixed `(1,5)` velocity residual with mean
static context and a source-ID input. It evaluated only 8 validation
trajectories/source at step 50; no test rows were used.

| Source | AE R² | Rollout R² |
|---|---:|---:|
| Reid | -1.551 | -2.372 |
| dePablo low-T | 0.820 | -0.893 |
| dePablo mixed-T | -11.107 | -3.017 |
| noisy-LJ | -19.721 | -5.269 |

Rejected: the compact 16D AE failed to preserve three source families, so the
rollout result cannot test the static-context hypothesis fairly.

## Matched small 06b-style validation comparison

Both runs used the 06b-style 2D shared attention AE, compact 4D stored edges,
16D mean static context, and one-step `delta_mlp` propagator. Each had 12
training and 10 validation trajectories per included source, 100 transitions,
and validation-only step-100 reporting. These are not held-out test results.

| Source | Reid + dePablo AE R² | All-four AE R² | Reid + dePablo rollout R² | All-four rollout R² |
|---|---:|---:|---:|---:|
| Reid | 0.590 | -0.201 | -0.323 | -0.386 |
| dePablo low-T | 0.979 | 0.603 | 0.068 | 0.289 |
| dePablo mixed-T | 0.819 | -0.850 | 0.084 | -3.106 |
| noisy-LJ | — | -3.340 | — | -2.228 |

Adding noisy-LJ to this small shared run substantially degraded the AE for Reid
and mixed-T. Low-T AE quality declined, though its rollout improved. This is
an AE interference effect, not a uniform small propagator degradation.

### Latest completed A/B result — test p-ratio R² at step 100

| Source | No context (A) | Mean context (B) | Outcome |
|---|---:|---:|---|
| Reid | -0.109 | -0.152 | Failed |
| dePablo low-T | 0.744 | 0.737 | Works |
| dePablo mixed-T | -1.674 | -0.857 | Failed; B is less bad |
| noisy-LJ | -1.823 | -1.856 | Failed |

Neither current A nor B is a shared solution: only dePablo low-T has positive
R² at 100 steps. The source-wise endpoint table is also saved in the results
directory for quick inspection.

Read `val_rollout_source_*_p_ratio_r2` and
`val_rollout_min_source_endpoint_p_ratio_r2` during training. Do not select a
model from pooled `val_rollout_p_ratio_r2` alone.

The notebook saves the full numerical test curve to
`notebooks/results/07b_mixed_reid_depablo_lj_context_ablation/mixed_context_ablation_rollout_p_ratio_r2.csv`
and a step-100 source-wise table to
`notebooks/results/07b_mixed_reid_depablo_lj_context_ablation/mixed_context_ablation_endpoint_p_ratio_r2.csv`.

## Completed CPU diagnostic — no propagator static context

`AD_xae_fixed_window_no_static_cpu_screen` reused the frozen successful X AE
(2D, kinematic target, prefix frames 0–2). It trained a one-step delta-z
`fixed_window_mlp` propagator from the same observed frames `(0, 1, 2)`, but
removed its static graph context entirely. The screen used 20 train and 20
validation trajectories per source, 30 transitions per training trajectory,
8 epochs, patience 3, learning rate `1e-4`, hidden size 64, mixed unbalanced
sources, and validation-only rollout evaluation at step 100.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | -0.228 |
| dePablo low-T | -0.113 |
| dePablo mixed-T | -0.100 |

Rejected: removing context makes every source negative, including low-T. This
does not support the hypothesis that physical static context is the primary
cause of the X rollout failure; it is necessary information for this frozen AE
and setup. These are validation values, not test values.

## Completed CPU diagnostic — larger fixed-window propagator

`AE_xae_fixed_window_static_hp_capacity_cpu_val` reused frozen X and restored
the physical 16D mean static context. It otherwise retained the three-frame
one-step delta-z `fixed_window_mlp`, but used 100 transitions per training
trajectory, hidden size 128, learning rate `3e-4`, 20 epochs, and patience 6.
The split was 20 train/20 validation trajectories per source; reporting is
validation-only at step 100.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | -0.363 |
| dePablo low-T | 0.200 |
| dePablo mixed-T | 0.384 |

Rejected: extra width and a larger learning rate harm every source relative to
the original X fixed-window result. This is not an obvious capacity shortage.

## Completed CPU diagnostic — matched 06b one-step delta MLP

`AF_xae_06b_onestep_delta_mlp_static_cpu_val` reused frozen X and the same
physical 16D mean context, but changed only the propagator to the 06b-style
plain `one_step` `delta_mlp` (hidden size 64, learning rate `1e-4`). It used
100 transitions per training trajectory, up to 30 epochs, patience 6, and the
same 20 train/20 validation trajectories per source. It early-stopped at epoch
16 (checkpoint epoch 10); values below are the full 20-per-source validation
rollout at step 100.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | -0.172 |
| dePablo low-T | 0.427 |
| dePablo mixed-T | 0.833 |

This reproduces the strong dePablo mixed-T behavior of 06b and improves both
dePablo sources over X, but still does not solve Reid. The fixed-window
architecture was therefore not the cause of the Reid failure, while the plain
one-step delta MLP is the better dePablo propagator for this frozen AE.

## Completed CPU diagnostic — learned-attention static graph pooling

`AG_xae_06b_onestep_delta_mlp_attention_static_cpu_val` repeated AF exactly,
except the physical static graph was pooled by learned attention rather than a
mean before conditioning the 06b-style one-step `delta_mlp`. It early-stopped
at epoch 22 (checkpoint epoch 16). The split remained 20 train/20 validation
trajectories per source, with 100 transitions per training trajectory and
validation-only step-100 rollout reporting.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | -0.238 |
| dePablo low-T | 0.349 |
| dePablo mixed-T | 0.857 |

Rejected: learned attention preserves the strong mixed-T result but degrades
Reid and low-T versus mean pooling. The issue is therefore not that mean
pooling alone discards a recoverable Reid cue in this setup.

## Completed CPU diagnostic — 4D static-context bottleneck

`AI_xae_06b_onestep_delta_mlp_static_context4_cpu_val` reused frozen X with
the 06b-style one-step `delta_mlp`, mean physical static graph context, hidden
size 64, learning rate `1e-4`, and seed 456. The only changed quantity from AF
was the learned static-context bottleneck: 4D instead of 16D. It used 20
train/20 validation trajectories per source, 100 transitions per training
trajectory, and early-stopped at epoch 16 (checkpoint epoch 10). Values are
the full validation rollout at step 100.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | -0.135 |
| dePablo low-T | 0.229 |
| dePablo mixed-T | 0.815 |

Rejected as the shared configuration: compressing the context slightly
improves Reid but materially harms both dePablo sources. The mean 16D context
remains the stronger simple baseline overall.

## Completed matched propagator baseline — three-frame residual window

`AP_xae_fixed_window_velocity_residual_012_cpu_val` reused frozen X (no AE
training) with the current 07b three-observation velocity-residual propagator:
`fixed_window_velocity_residual_mlp`, observed frames `(0, 1, 2)`, hidden size
128, 16D mean physical static context, one-step delta-z loss, 20 train/20
validation trajectories per source, 100 transitions per training trajectory,
20 epochs, and seed 456. Values are full validation rollouts at step 100.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | 0.472 |
| dePablo low-T | 0.685 |
| dePablo mixed-T | -1.339 |

This is the matched baseline for the 06b-history ablation. It captures Reid
and low-T, but its three-frame residual state is unstable for mixed-T.

## Completed propagator ablation — 06b two-frame residual state

`AQ_xae_06b_fixed_velocity_residual_15_cpu_val` changed only the compatible
history-state formulation relative to AP: it used
`fixed_velocity_residual_mlp` with two observed latent frames `(1, 5)`, hence
the anchored velocity `(z(5)-z(1))/4`. Frozen X, the 16D mean physical context,
2D latent, optimizer, split, transition budget, hidden size 128, and
validation-only step-100 evaluation were otherwise unchanged.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | 0.408 |
| dePablo low-T | 0.814 |
| dePablo mixed-T | 0.578 |

This is the first matched frozen-X propagator run that is positive for every
source. The two-frame `(1,5)` residual state trades a small amount of Reid
performance for a large mixed-T stability improvement.

## Completed factor separation — early two-frame residual state

`AR_xae_two_frame_velocity_residual_02_cpu_val` kept AQ's exact two-frame
`fixed_velocity_residual_mlp` architecture but changed its observations to
`(0, 2)`. It therefore tests the two-frame residual formulation without the
later frame-5 initialization. All other AQ settings were unchanged.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | -0.829 |
| dePablo low-T | 0.830 |
| dePablo mixed-T | -0.154 |

Rejected: the two-frame architecture alone is not sufficient. An early
two-frame velocity estimate is too weak for Reid and does not stabilize mixed-T.

## Completed factor separation — later three-frame residual window

`AS_xae_fixed_window_velocity_residual_135_cpu_val` kept AP's three-frame
`fixed_window_velocity_residual_mlp` architecture but shifted its three causal
observations to `(1, 3, 5)`. Its learned velocity anchor is thus derived from
the last two observed frames, while all X, optimization, context, and split
settings remain identical.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | -0.164 |
| dePablo low-T | 0.650 |
| dePablo mixed-T | -1.641 |

Rejected: later observations alone do not fix the current three-frame window
model. The successful AQ behavior requires the specific two-frame `(1,5)`
residual-state formulation, not just a later initialization window.

## Completed width control — current three-frame residual state

`AT_xae_fixed_window_velocity_residual_012_hidden64_cpu_val` repeated AP with
only its hidden width changed from 128 to 64. It is the original current-07b
three-frame residual state `(0,1,2)` with frozen X, 16D mean physical context,
and the matched 20/20-per-source validation protocol.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | -0.461 |
| dePablo low-T | 0.807 |
| dePablo mixed-T | -0.794 |

Width 128 helps Reid in this unstable three-frame architecture, but does not
resolve its negative mixed-T rollout.

## Completed width control — 06b two-frame residual state

`AU_xae_06b_fixed_velocity_residual_15_hidden64_cpu_val` repeated AQ with
only its hidden width changed from 128 to 64. It retained `(1,5)`, the anchored
velocity `(z(5)-z(1))/4`, frozen X, and all matched physical-context and split
settings.

| Source | Held-out validation rollout p-ratio R² at step 100 |
|---|---:|
| Reid | 0.488 |
| dePablo low-T | 0.816 |
| dePablo mixed-T | 0.157 |

The two-frame state alone is enough to recover Reid and low-T at 64 units;
raising width to 128 chiefly improves mixed-T (AQ: 0.578). The required shared
combination is therefore the `(1,5)` two-frame residual state plus adequate
capacity for mixed-T, not context or loss changes.

## Decision rules

1. Do not promote the fixed-velocity residual with the current 10D AE as a
   shared solution: the small validation screen rejects it for mixed-T and
   noisy-LJ. The next small screen should instead isolate AE representation
   quality or use a causal source-aware dynamics signal.
2. If noisy-LJ AE reconstruction remains poor, improve the shared AE before
   adding more propagator complexity.
3. If noisy-LJ reconstruction is good but its rollout remains poor after the
   gate, investigate source-aware dynamics without future rollout information.
4. Record every completed run here with its exact cache name, split size,
   history/objective, and four source-wise test R² values.

## Completed current-code single-call control — no-history 06b-style baseline

`AW_current_code_single_call_one_step_delta_reid_depablo` is the direct
control for the AV cached-AE run. It used seed `78564`, 15 train and 15
validation trajectories per source, 101 frames, compact stored 4D dynamic
edges, normalized displacement node input/AE target, a 2D attention AE, and
the preserved physical reference graph with a mean-pooled 16D static context.
The propagator was the 128-unit `delta_mlp`, trained with one-step Δz loss for
up to 30 epochs. Crucially, AE and propagator were trained in **one fresh
`run_latent_experiment` call**: no pretrained-AE cache, no refitting or reuse
of external normalizers, and no reseed between AE training and propagator
initialization.

| Source | Validation rollout p-ratio R² at 100 (n=15) | Held-out test rollout p-ratio R² at 100 (n=30) |
|---|---:|---:|
| Reid | -0.691 | 0.139 |
| dePablo low-T | 0.804 | 0.816 |
| dePablo mixed-T | 0.619 | 0.597 |

The AE itself remained informative (test reconstruction p-ratio R²: Reid
`0.835`, low-T `0.958`, mixed-T `0.923`), but the Reid propagator collapsed
to a narrow output range. This is nearly the same result as AV's cached-AE
run (Reid test `0.122`), so the separate-cache/reseed flow is **not** the
cause of the bad Reid rollout. The remaining mismatch is between the old
successful 06b learned representation/training state and current-code
training, not a train/validation split or normalizer mismatch.

## Completed corrected evolving-box geometry — 06b-style baseline

`AY_evolving_box_06b_ae_recipe_seed456_reid_depablo` corrected normalized
periodic edge geometry: frame zero is `[-1,1]^2`, while later frames use their
own transformed, evolving periodic box for minimum-image edge vectors and
lengths. It used seed `456`, 15 train and 15 validation trajectories/source,
101 frames, the 2D normalized-displacement attention AE, compact stored edges,
the physical 16D mean static context, and the 128-unit one-step Δz `delta_mlp`.
AE and propagator were trained jointly in one call.

| Source | Validation AE p-ratio R² at 100 | Test AE p-ratio R² at 100 | Validation rollout R² at 100 | Held-out test rollout R² at 100 |
|---|---:|---:|---:|---:|
| Reid | 0.900 | 0.918 | 0.105 | 0.264 |
| dePablo low-T | 0.942 | 0.902 | 0.376 | 0.338 |
| dePablo mixed-T | 0.808 | 0.683 | 0.745 | 0.699 |

The geometry correction materially improved the AE (AW Reid test `0.835` to
AY `0.918`; validation loss `0.644` to `0.421`) and made every rollout source
positive. It does not yet recover the high low-T/Reid rollout values of the
historical v5 checkpoint, so the remaining limitation is propagator capacity
or state formulation rather than AE information loss or edge normalization.

## Invalidated static Δz run — mismatched split seed

`AZ_frozen_AY_static_delta_h64_seed456_rollout.pt` loaded the frozen AY AE but
was actually run with `split_seed = model_seed = 45634562345`, rather than
AY's seed `456`. It therefore used different trajectories and is not a valid
frozen-AE propagator comparison.

| Source | Validation rollout p-ratio R² at 100 (n=15) | Held-out test rollout p-ratio R² at 100 (n=30) |
|---|---:|---:|
| Reid | 0.052 | 0.046 |
| dePablo low-T | 0.655 | 0.641 |
| dePablo mixed-T | 0.938 | 0.820 |

The notebook now uses `SEED = 456` for both the frozen AY AE split and the
propagator, with a new cache name ending in `seed456_matched_split`.
