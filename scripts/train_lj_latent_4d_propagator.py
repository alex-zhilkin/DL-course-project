"""Train the normal 4D LJ latent-space simulator after fixing the AE ceiling."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.utils import resolve_device


DATA = (
    ROOT
    / "data"
    / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"
)
VARIANT = os.environ.get("LJ_PROPAGATOR_VARIANT", "baseline")
LATENT_DIM = int(os.environ.get("LJ_LATENT_DIM", "4"))
DYN_TRANSITIONS = int(os.environ.get("LJ_DYN_TRANSITIONS", "199"))
CURRICULUM_HORIZONS = [
    int(value)
    for value in os.environ.get("LJ_CURRICULUM_HORIZONS", "1,4,8,16,32").split(",")
]
CURRICULUM_EPOCHS = [
    int(value)
    for value in os.environ.get("LJ_CURRICULUM_EPOCHS", "4,5,7,9,11").split(",")
]
if len(CURRICULUM_HORIZONS) != len(CURRICULUM_EPOCHS):
    raise ValueError("LJ curriculum horizons and epochs must have equal lengths")
OUTPUT = ROOT / (
    "notebooks/results/07_lj_4d_propagator"
    if VARIANT == "baseline"
    else f"notebooks/results/07_lj_4d_propagator_{VARIANT}"
)
MODEL = OUTPUT / (
    f"models/lj_cv{LATENT_DIM}_ae200_dyn{DYN_TRANSITIONS}_seed20260726.pt"
)
PRETRAINED_AE = os.environ.get("LJ_PRETRAINED_AE")
NETWORK_VARIATION_WEIGHT = float(
    os.environ.get("LJ_NETWORK_VARIATION_WEIGHT", "0")
)
PROPAGATOR_P_RATIO_WEIGHT = float(
    os.environ.get("LJ_PROPAGATOR_P_RATIO_WEIGHT", "0")
)
INITIAL_VELOCITY = os.environ.get("LJ_INITIAL_VELOCITY", "zero")
SEED = 20260726


def experiment(device):
    mixture = [
        {
            "name": "lj_noisy",
            "label": "LJ noisy",
            "path": str(DATA),
            "train_count": 100,
            "val_count": 20,
            "edge_multiplicity": 1,
        }
    ]
    cfg = {
        "dataset_name": "lj_noisy",
        "split_seed": SEED,
        "model_seed": SEED,
        "device": str(device),
        "pos_dim": 2,
        "batch_graphs": 16,
        "frame_skip": 1,
        "edge_multiplicity": 1,
        "edge_vector_dim": 2,
        "edge_mode": "stored",
        "latent_dim": LATENT_DIM,
        "latent_tokens": 32,
        "hidden_size": 96,
        "autoencoder_model": "attention",
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        # AE: every frame, ordinary reconstruction plus the successful
        # two-frame geometric p-ratio preservation term.
        "ae_max_train_frames_per_sim": 200,
        "ae_max_epochs": 15,
        "ae_patience": 4,
        "ae_lr": 1e-4,
        "ae_weight_decay": 1e-5,
        "ae_strain_loss_weight": 0.0,
        "ae_p_ratio_loss_weight": 10.0,
        "ae_p_ratio_minimum_driven_strain": 1e-3,
        # Propagator: use every available transition. Generalization is across
        # held-out networks, not an artificial time extrapolation split.
        "dyn_max_train_transitions_per_sim": DYN_TRANSITIONS,
        "dyn_max_epochs": sum(CURRICULUM_EPOCHS),
        "dyn_patience": 6,
        "dyn_lr": 1e-4,
        "dyn_weight_decay": 1e-5,
        "propagator_hidden_size": 128,
        "propagator_objective": "kinematic_multistep",
        "propagator_model": "kinematic_mlp",
        "propagator_loss": "next_z",
        "propagator_standardize_latent": True,
        "propagator_curriculum_horizons": CURRICULUM_HORIZONS,
        "propagator_curriculum_epochs": CURRICULUM_EPOCHS,
        "propagator_use_static_context": True,
        "propagator_context_pool": "learned_attention",
        "graph_context_dim": 64,
        # "observed" seeds the second-order state with encoded frames 0 and 1.
        # Training already uses consecutive ground-truth latents at each start.
        "initial_velocity": INITIAL_VELOCITY,
        "propagator_position_loss_weight": 0.0,
        "propagator_p_ratio_loss_weight": PROPAGATOR_P_RATIO_WEIGHT,
        "propagator_p_ratio_minimum_driven_strain": 1e-3,
        "propagator_p_ratio_boundary_fraction": 0.10,
        "propagator_network_variation_weight": NETWORK_VARIATION_WEIGHT,
        "propagator_network_variation_floor_fraction": 0.05,
        "early_stop_min_delta": 1e-5,
        "rollout_steps_grid": [25, 50, 99, 125, 150, 199],
        "rollout_eval_max_sims_per_split": None,
        # LJ is evaluated by the agreed two-frame endpoint ratio.
        "temperature_pratio_estimator": "endpoint",
        "should_rollout": True,
        "should_train_propagator": True,
        "cache_path": str(MODEL),
        "force_train": True,
        "repeat_idx": 1,
        "dataset_mixture": mixture,
    }
    if PRETRAINED_AE:
        cfg["pretrained_ae_cache_path"] = PRETRAINED_AE
    source = {
        **cfg,
        "source_name": "LJ noisy",
        "label": (
            f"LJ noisy | {LATENT_DIM}D AE + all-frame kinematic propagator"
        ),
        "path": str(DATA),
    }
    return source, cfg


def save_results(result):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result["ae_history"].to_csv(OUTPUT / "ae_history.csv", index=False)
    result["dyn_history"].to_csv(OUTPUT / "propagator_history.csv", index=False)
    result["ae_reconstruction_stats"].to_csv(
        OUTPUT / "ae_ceiling.csv", index=False
    )
    result["rollout_stats"].to_csv(OUTPUT / "rollout_stats.csv", index=False)
    result["rollout_rows"].to_csv(OUTPUT / "rollout_rows.csv", index=False)

    ceiling = result["ae_reconstruction_stats"].query("split == 'test'").copy()
    rollout = result["rollout_stats"].query("split == 'test'").copy()
    comparison = pd.concat(
        [
            ceiling.assign(result="AE ceiling"),
            rollout.assign(result="Propagator rollout"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(OUTPUT / "test_pratio_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    for label, group in comparison.groupby("result", sort=False):
        ax.plot(
            group["rollout_steps"],
            group["p_ratio_r2"],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Ground-truth frame / rollout step",
        ylabel=r"Endpoint p-ratio $R^2$",
        title=f"Noisy LJ: {LATENT_DIM}D AE ceiling and latent rollout",
    )
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "ae_ceiling_vs_rollout.png", dpi=220)
    fig.savefig(OUTPUT / "ae_ceiling_vs_rollout.pdf")
    plt.close(fig)

    print("\nTest AE ceiling")
    print(
        ceiling[
            ["rollout_steps", "used", "p_ratio_r2", "p_ratio_pearson"]
        ].to_string(index=False)
    )
    print("\nTest propagator rollout")
    print(
        rollout[
            ["rollout_steps", "used", "p_ratio_r2", "p_ratio_pearson"]
        ].to_string(index=False)
    )


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    seed_everything(SEED)
    device = resolve_device("auto")
    source, cfg = experiment(device)
    result = run_latent_experiment(source, cfg, device=device)
    save_results(result)


if __name__ == "__main__":
    main()
