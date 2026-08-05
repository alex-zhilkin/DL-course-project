"""Measure whether a trained exact pyramid latent organizes around p-ratio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from internal_09_transformer_search import make_inputs
from lss.data import load_dataset
from lss.latent.experiment import ground_truth_p_ratio
from lss.models.attention_pyramid_simulator import AttentionPyramidSimulator


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-key", choices=("lj_noisy", "depablo_low_temp"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--train-networks", type=int, default=100)
    parser.add_argument("--val-networks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def p_ratio(sim, dataset_key, estimator):
    cfg = {
        "temperature_pratio_estimator": estimator,
        "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5,
    }
    return ground_truth_p_ratio(sim, dataset_name=dataset_key, cfg=cfg)


def ridge_fit_predict(train_x, train_y, test_x, penalty=1e-3):
    mean = train_x.mean(0)
    scale = train_x.std(0)
    scale[scale < 1e-8] = 1
    x = (train_x - mean) / scale
    xt = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    test_design = np.column_stack([np.ones(len(xt)), xt])
    regularizer = penalty * np.eye(design.shape[1])
    regularizer[0, 0] = 0
    weights = np.linalg.solve(
        design.T @ design + regularizer, design.T @ train_y
    )
    return test_design @ weights


def r2(true, predicted):
    denominator = np.square(true - true.mean()).sum()
    return float(1 - np.square(true - predicted).sum() / denominator)


def main():
    args = arguments()
    paths = {
        "lj_noisy": ROOT
        / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt",
        "depablo_low_temp": ROOT / "data/depablo-near-zero-temp.pt",
    }
    sims = load_dataset(paths[args.dataset_key], edge_multiplicity=1)
    order = np.random.default_rng(args.seed).permutation(len(sims))
    val_ids = order[
        args.train_networks : args.train_networks + args.val_networks
    ]
    test_ids = order[args.train_networks + args.val_networks :]
    state = torch.load(args.state, map_location="cpu", weights_only=False)
    model = AttentionPyramidSimulator(
        node_dim=7,
        edge_dim=13,
        hidden_size=64,
        pyramid_tokens=(32, 16),
        heads=4,
        bottleneck_layers=2,
        latent_dim=args.latent_dim,
    )
    model.load_state_dict(state["model_state"])
    model.eval()
    common = {
        "target_mean": state["target_mean"],
        "target_std": state["target_std"],
        "accel_mean": state["accel_mean"],
        "accel_std": state["accel_std"],
        "edge_mean": state["edge_mean"],
        "edge_std": state["edge_std"],
        "length_scale": float(state["length_scale"]),
        "distance_floor": -8.0 if args.dataset_key == "lj_noisy" else -6.0,
        "velocity_skip": False,
        "dual_kinematic": False,
        "boundary_features": False,
        "boundary_weight": 1.0,
        "node_count_feature": False,
        "target_mode": "next_state",
        "edge_mode": "complete",
        "undirected_edges": True,
        "device": "cpu",
    }

    def describe(sim_id, split):
        sim = sims[int(sim_id)]
        frame_ids = sorted(set([*range(min(20, len(sim))), len(sim) - 1]))
        latents = {}
        with torch.no_grad():
            for t in frame_ids:
                previous = sim[t - 1] if t > 0 else sim[t]
                node, edge, edge_index, prior, *_ = make_inputs(
                    sim, t, sim[t], previous, with_target=False, **common
                )
                latents[t] = model.encode_latent(
                    node, edge, edge_index, attention_bias=prior
                ).numpy()
        row = {
            "split": split,
            "sim_id": int(sim_id),
            "p_ratio_robust": p_ratio(sim, args.dataset_key, "robust"),
            "p_ratio_endpoint": p_ratio(sim, args.dataset_key, "endpoint"),
        }
        for coordinate in range(args.latent_dim):
            row[f"initial_z{coordinate}"] = latents[0][coordinate]
            row[f"final_z{coordinate}"] = latents[len(sim) - 1][coordinate]
            for window, final_frame in ((5, 4), (10, 9), (20, 19)):
                times = np.arange(window, dtype=float)
                values = np.stack([latents[t] for t in range(window)])
                slope = np.polyfit(times, values[:, coordinate], 1)[0]
                row[f"slope{window}_z{coordinate}"] = slope
                row[f"delta{window}_z{coordinate}"] = (
                    latents[final_frame][coordinate] - latents[0][coordinate]
                )
        return row

    rows = [
        *(describe(sim_id, "val") for sim_id in val_ids),
        *(describe(sim_id, "test") for sim_id in test_ids),
    ]
    table = pd.DataFrame(rows).dropna()
    test = table[table.split.eq("test")]
    correlation_rows = []
    targets = ("p_ratio_robust", "p_ratio_endpoint")
    features = [
        column
        for column in table
        if column.startswith(("initial_", "final_", "slope", "delta"))
    ]
    for target in targets:
        for feature in features:
            correlation_rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson": test[[target, feature]].corr().iloc[0, 1],
                    "n": len(test),
                }
            )
    correlations = pd.DataFrame(correlation_rows)

    probe_rows = []
    val = table[table.split.eq("val")]
    feature_groups = {
        "initial": [f"initial_z{i}" for i in range(args.latent_dim)],
        "final": [f"final_z{i}" for i in range(args.latent_dim)],
    }
    for window in (5, 10, 20):
        feature_groups[f"slope{window}"] = [
            f"slope{window}_z{i}" for i in range(args.latent_dim)
        ]
        feature_groups[f"delta{window}"] = [
            f"delta{window}_z{i}" for i in range(args.latent_dim)
        ]
    for target in targets:
        for name, columns in feature_groups.items():
            predicted = ridge_fit_predict(
                val[columns].to_numpy(),
                val[target].to_numpy(),
                test[columns].to_numpy(),
            )
            probe_rows.append(
                {
                    "target": target,
                    "descriptor": name,
                    "test_r2": r2(test[target].to_numpy(), predicted),
                    "test_pearson": np.corrcoef(
                        test[target].to_numpy(), predicted
                    )[0, 1],
                }
            )
    probes = pd.DataFrame(probe_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "latent_descriptors.csv", index=False)
    correlations.to_csv(args.output / "latent_correlations.csv", index=False)
    probes.to_csv(args.output / "latent_linear_probes.csv", index=False)
    summary = {
        "dataset": args.dataset_key,
        "state": str(args.state),
        "latent_dim": args.latent_dim,
        "n_val": len(val),
        "n_test": len(test),
        "strongest_correlations": correlations.iloc[
            correlations.pearson.abs().sort_values(ascending=False).index[:12]
        ].to_dict(orient="records"),
        "linear_probes": probes.to_dict(orient="records"),
    }
    (args.output / "latent_summary.json").write_text(json.dumps(summary, indent=2))
    print(correlations.iloc[
        correlations.pearson.abs().sort_values(ascending=False).index[:12]
    ].to_string(index=False))
    print(probes.to_string(index=False))


if __name__ == "__main__":
    main()
