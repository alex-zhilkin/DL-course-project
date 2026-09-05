import torch
from pathlib import Path

from lss.latent.capacity import load_experiment_bundle
from lss.latent.simulation import make_transition_index
from lss.latent.training import fit_latent_step_stats

cache = Path(
    "notebooks/results/07b_mixed_reid_depablo_lj_context_ablation/"
    "mixed4source_A_no_context_latent8_attention_stored13_unbalancedAE_train20_100frames.pt"
)
cfg = {
    "dataset_mixture": [
        {"name": "reid", "path": "data/reid_200_frames.pt", "train_count": 20, "val_count": 10},
        {"name": "depablo_low_temp", "path": "data/depablo-near-zero-temp.pt", "train_count": 20, "val_count": 10},
        {"name": "depablo_mixed_temp", "path": "data/depablo-10k-mix-temp.pt", "train_count": 20, "val_count": 10},
        {"name": "lj_noisy", "path": "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt", "train_count": 20, "val_count": 10},
    ],
    "dataset_name": "mixed_reid_depablo_lj", "pos_dim": 2, "device": "cuda",
    "batch_graphs": 256, "frame_skip": 1, "train_frame_start_order": 0,
    "edge_mode": "stored", "coordinate_normalization": "position_normalization",
    "edge_feature_schema": "physical_static_normalized_edge_changes_v3",
    "ae_config": {"latent_dim": 8, "latent_tokens": 32, "hidden_size": 96, "model": "attention", "edge_feature_dim": 13, "node_feature_mode": "normalized_delta", "target_mode": "normalized_delta"},
    "propagator_config": {"objective": "one_step", "loss": "delta", "model": "delta_mlp", "hidden_size": 64},
}
r = load_experiment_bundle(cache, cfg=cfg, device=torch.device("cuda"))
all_sims = r["train_data"]
for source in ["reid", "depablo_low_temp", "depablo_mixed_temp", "lj_noisy"]:
    sims = [sim for sim in all_sims if str(getattr(sim[0], "source_name", "")) == source]
    rows = make_transition_index(sims, frame_skip=1, max_frames_per_sim=100)
    stats = fit_latent_step_stats(r["ae"], sims, rows, batch_graphs=256, pos_dim=2, node_feature_mode="normalized_delta", normalizers=r["normalizers"], device=torch.device("cuda"))
    print(source, "z_std", stats.z_std.squeeze().detach().cpu().numpy(), "dz_std", stats.dz_std.squeeze().detach().cpu().numpy(), "dz_mean", stats.dz_mean.squeeze().detach().cpu().numpy(), flush=True)
