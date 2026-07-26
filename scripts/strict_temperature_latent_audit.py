"""Strict temperature holdout audit for the one-dimensional latent autoencoder."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch

from lss.data import load_dataset, simulation_temperature
from lss.latent.experiment import find_project_root, ground_truth_p_ratio, seed_everything
from lss.latent.models import NodeDeltaAttentionAutoEncoder
from lss.latent.simulation import (
    fit_ae_target_stats,
    fit_edge_stats,
    fit_node_feature_stats,
    make_frame_index,
    pearson_r,
)
from lss.latent.training import TrainingConfig, encode_frame_latent, train_autoencoder
from lss.utils import resolve_device


SEED = 20260623
TRAIN_TEMPERATURES = (1.0, 5.0)
TRAIN_PER_TEMPERATURE = 1
VAL_PER_TEMPERATURE = 10


def select_splits(simulations):
    rng = np.random.default_rng(SEED)
    groups = {}
    for sim in simulations:
        groups.setdefault(simulation_temperature(sim), []).append(sim)
    for temperature in groups:
        order = rng.permutation(len(groups[temperature]))
        groups[temperature] = [groups[temperature][idx] for idx in order]

    train_data = []
    val_data = []
    test_data = []
    for temperature, group in sorted(groups.items()):
        if temperature in TRAIN_TEMPERATURES:
            train_stop = TRAIN_PER_TEMPERATURE
            val_stop = train_stop + VAL_PER_TEMPERATURE
            train_data.extend(group[:train_stop])
            val_data.extend(group[train_stop:val_stop])
            test_data.extend(group[val_stop:])
        else:
            test_data.extend(group)
    return train_data, val_data, test_data


def train_model(train_data, val_data, device):
    pos_dim = 2
    batch_graphs = 4
    node_mode = "normalized_delta"
    target_mode = "normalized_delta"
    train_frames = make_frame_index(
        train_data,
        frame_skip=1,
        max_frames_per_sim=80,
        include_last=True,
        start_frame_order=0,
    )
    val_frames = make_frame_index(
        val_data,
        frame_skip=1,
        max_frames_per_sim=80,
        include_last=True,
        start_frame_order=0,
    )
    target_mean, target_std = fit_ae_target_stats(
        train_data,
        train_frames,
        pos_dim=pos_dim,
        batch_graphs=batch_graphs,
        device=device,
        target_mode=target_mode,
    )
    node_mean, node_std = fit_node_feature_stats(
        train_data,
        train_frames,
        pos_dim=pos_dim,
        batch_graphs=batch_graphs,
        device=device,
        node_feature_mode=node_mode,
    )
    edge_mean, edge_std = fit_edge_stats(
        train_data,
        train_frames,
        pos_dim=pos_dim,
        batch_graphs=batch_graphs,
        device=device,
    )
    normalizers = {
        "target_mean": target_mean,
        "target_std": target_std,
        "node_feature_mean": node_mean,
        "node_feature_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
    }
    model = NodeDeltaAttentionAutoEncoder(
        pos_dim=pos_dim,
        edge_dim=int(edge_mean.numel()),
        hidden_size=96,
        latent_dim=1,
        latent_tokens=32,
    ).to(device)
    result = train_autoencoder(
        model,
        train_data,
        val_data,
        train_frames,
        val_frames,
        batch_graphs=batch_graphs,
        pos_dim=pos_dim,
        node_feature_mode=node_mode,
        ae_target_mode=target_mode,
        normalizers=normalizers,
        device=device,
        config=TrainingConfig(
            max_epochs=300,
            patience=6,
            learning_rate=5e-5,
            weight_decay=1e-5,
            min_delta=1e-5,
            log_every=10,
        ),
    )
    return result.model, normalizers, result.history


def trajectory_metrics(model, normalizers, simulations, device):
    rows = []
    model.eval()
    with torch.no_grad():
        for sim_idx, sim in enumerate(simulations):
            z0 = np.asarray(
                [
                    float(
                        encode_frame_latent(
                            model,
                            sim,
                            frame_idx,
                            pos_dim=2,
                            node_feature_mode="normalized_delta",
                            normalizers=normalizers,
                            device=device,
                        )[0].cpu()
                    )
                    for frame_idx in range(len(sim))
                ]
            )
            time = np.arange(len(z0), dtype=float)
            slope, intercept = np.polyfit(time, z0, deg=1)
            rows.append(
                {
                    "sim_idx": sim_idx,
                    "temperature": simulation_temperature(sim),
                    "final_p_ratio": ground_truth_p_ratio(
                        sim,
                        dataset_name="depablo_mixed_temp",
                        cfg={
                            "temperature_pratio_estimator": "robust",
                            "temperature_pratio_min_fit_frames": 8,
                            "temperature_pratio_min_driven_strain_range": 1e-3,
                            "temperature_pratio_smooth_window": 5,
                        },
                    ),
                    "z0_initial": float(z0[0]),
                    "z0_slope": float(slope),
                    "z0_step_std": float(np.std(np.diff(z0))),
                    "z0_residual_std": float(np.std(z0 - (slope * time + intercept))),
                }
            )
    return pd.DataFrame(rows)


def main():
    seed_everything(SEED + 1009)
    device = resolve_device("auto")
    project_root = find_project_root()
    output_dir = (
        project_root
        / "notebooks"
        / "results"
        / "04a_latent_space_analysis"
        / f"strict_temperature_holdout_cv1_seed{SEED}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "strict_temperature_model.pt"

    simulations = load_dataset(project_root / "data" / "depablo-10k-mix-temp.pt")
    train_data, val_data, test_data = select_splits(simulations)
    if cache_path.exists():
        bundle = torch.load(cache_path, map_location=device, weights_only=False)
        normalizers = {key: value.to(device) for key, value in bundle["normalizers"].items()}
        model = NodeDeltaAttentionAutoEncoder(
            pos_dim=2,
            edge_dim=int(normalizers["edge_mean"].numel()),
            hidden_size=96,
            latent_dim=1,
            latent_tokens=32,
        ).to(device)
        model.load_state_dict(bundle["model_state"])
        history = pd.DataFrame(bundle["history"])
    else:
        model, normalizers, history = train_model(train_data, val_data, device)
        torch.save(
            {
                "model_state": model.state_dict(),
                "normalizers": {key: value.detach().cpu() for key, value in normalizers.items()},
                "history": history.to_dict(orient="records"),
            },
            cache_path,
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metrics = trajectory_metrics(model, normalizers, test_data, device)
    unseen = metrics[~metrics["temperature"].isin(TRAIN_TEMPERATURES)]
    summary = pd.DataFrame(
        [
            {
                "scope": "all test trajectories",
                "n": len(metrics),
                "slope_pratio_r": pearson_r(metrics["z0_slope"], metrics["final_p_ratio"]),
                "jitter_temperature_r": pearson_r(
                    metrics["z0_step_std"], metrics["temperature"]
                ),
            },
            {
                "scope": "unseen temperature levels",
                "n": len(unseen),
                "slope_pratio_r": pearson_r(unseen["z0_slope"], unseen["final_p_ratio"]),
                "jitter_temperature_r": pearson_r(
                    unseen["z0_step_std"], unseen["temperature"]
                ),
            },
        ]
    )
    history.to_csv(output_dir / "ae_history.csv", index=False)
    metrics.to_csv(output_dir / "test_trajectory_metrics.csv", index=False)
    summary.to_csv(output_dir / "audit_summary.csv", index=False)
    print("train temperatures:", [simulation_temperature(sim) for sim in train_data])
    print("val temperatures:", sorted({simulation_temperature(sim) for sim in val_data}))
    print("test counts:", metrics["temperature"].value_counts().sort_index().to_dict())
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
