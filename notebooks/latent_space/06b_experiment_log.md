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
