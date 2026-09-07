# Latent pipeline audit — 2026-09-07

Scope: active normalized_delta / compact_stored AE path and current experiment
entry point. Frozen historical studies are preserved. This is not a claim that
every legacy architecture/objective has been verified.

## Confirmed issues and impact

1. Standardized directional edges were reversed by sign alone despite nonzero
   training means. Correct reversal is -u-2mu/sigma. Reproduced algebraically
   from a saved checkpoint. Terra owns isolated corrected-baseline and MP
   implementation/smokes; downstream accuracy impact is pending. Separate this
   correction from architecture attribution. See ae_information_audit outputs.
2. Explicit cache paths were trusted by default despite configuration mismatch.
   Changed cache_require_matching_config default to true in experiment.py;
   existing explicit opt-out remains available for intentional legacy reuse.
   Regression tests verify matching loads and mismatching retrains. Recent
   studies use force_train and new run directories, so this is not an explanation
   for their failures. Config fingerprints do not themselves prove unchanged raw
   data or code; frozen snapshots/data hashes remain necessary.
3. Architecture smoke tests found nested configuration naming and normalizer
   shape errors before broad submission. Terra preserved failures and retried.
   These were new experiment integration failures, not evidence the earlier
   successful runs used those new models.

## Verified active-path contracts

17 tests plus 6 subtests passed in the targeted suite:
- fixed reference-box normalization and idempotence;
- explicit disjoint mixture splits and config expansion/cache fingerprint;
- source-weighted reconstruction loss;
- batched training forward versus single-frame encode/decode;
- normalized target roundtrip with physical reference context deliberately
  different from the model coordinate origin;
- future positions/edge attributes cannot change initial encoding or a decoded
  prediction at fixed latent, for normalized_delta / compact_stored;
- recipe-matching cache loading regression.

Command: OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/lss-pipeline-audit .venv/bin/python
-m pytest -q tests/test_latent_pipeline_contract.py
 tests/test_latent_cache_and_sources.py tests/test_reference_box_current_convention.py
 tests/test_explicit_mixture_splits.py tests/test_source_mean_ae_loss.py

Code inspection: AE statistics are fitted on train_data/train_frames; latent
transition statistics on train_data/latent_stat_rows. Explicit split IDs retain
order and enforce disjointness. Best checkpoint weights are cloned and restored.
Decoder uses the initial graph and inverts target normalization in model
coordinates, rather than adding normalized displacement to physical context.
These tests use constructed graphs; they supplement the earlier complete raw
normalization audit, not a new full real-data replay or all-objective rollout test.

## Other paths and limitations

- Legacy prefix3 node features read frames1/2 even when encoding frame0.
  They require an explicitly observed prefix and cannot support a frame0-only
  forecast claim. Active normalized_delta does not use this branch.
- Recomputed-stored edges are supported by batching but not the inspected
  single-frame encode/decode branches. Active compact_stored is covered.
- Three tests in test_static_structure_rollout.py fail because a separate older
  simulator helper no longer accepts pratio_loss_weight. The tested helper is
  outside this active AE path; failures are retained/documented, not counted as
  passing or treated as proof of a current latent failure.
- Historical p-ratio-selected models do not meet the new expert-blind selection
  requirement. Preserve their observations without promoting them as compliant.
- Epoch scalar loss is a mean of batch losses; per-source metrics separately
  accumulate per-graph errors. For exact cross-batch comparisons use saved
  per-network state errors, particularly when the last batch is shorter.

No evidence so far of a target-coordinate inversion or basic train/inference
mismatch in the active path. The edge reversal issue is the strongest confirmed
candidate for affecting learned representations; its scientific impact still
requires corrected controls. Broader initial-state-only free-rollout checks,
real-data reload parity and advanced history modes remain future audit coverage.
