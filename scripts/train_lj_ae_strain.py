"""Train and audit a 2D noisy-LJ AE with global strain preservation.

The experiment is AE-only: no latent propagator is created or evaluated.
Networks are split before frames, and every frame of each training/validation
trajectory is used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides

from lss.graph import clone_graph
from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.latent.simulation import pearson_r, r2_score
from lss.latent.training import decode_latent_to_graph, encode_frame_latent
from lss.utils import resolve_device


DATA = (
    ROOT
    / "data"
    / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"
)
OUTPUT = ROOT / "notebooks/results/07_lj_ae_strain"
SEED = 20260726


def make_experiment(args, device):
    common = {
        "dataset_name": "lj_noisy",
        "split_seed": SEED,
        "model_seed": SEED,
        "device": str(device),
        "pos_dim": 2,
        "batch_graphs": args.batch_graphs,
        "frame_skip": 1,
        "train_frame_start_order": 0,
        "latent_dim": args.latent_dim,
        "latent_tokens": 32,
        "hidden_size": args.hidden_size,
        "autoencoder_model": "attention",
        "edge_mode": "stored",
        "edge_feature_dim": 12,
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "ae_max_train_frames_per_sim": 200,
        "dyn_max_train_transitions_per_sim": 0,
        "ae_max_epochs": args.epochs,
        "ae_patience": args.patience,
        "ae_lr": args.learning_rate,
        "ae_weight_decay": 1e-5,
        "ae_strain_loss_weight": args.strain_weight,
        "ae_strain_boundary_fraction": 0.10,
        "ae_p_ratio_loss_weight": args.p_ratio_weight,
        "ae_p_ratio_minimum_driven_strain": 1e-3,
        "early_stop_min_delta": 1e-5,
        "should_rollout": False,
        "should_train_propagator": False,
        "force_train": True,
        # Deliberately do not cache models while this objective is being tested.
        "cache_path": None,
        "repeat_idx": 1,
    }
    mixture = [
        {
            "name": "lj_noisy",
            "label": "LJ noisy",
            "path": str(DATA),
            "train_count": args.train_networks,
            "val_count": args.val_networks,
        }
    ]
    cfg = {**common, "dataset_mixture": mixture}
    source = {
        **common,
        "source_name": "LJ noisy",
        "label": (
            f"LJ AE CV{args.latent_dim} all frames, strain={args.strain_weight:g}, "
            f"p-ratio={args.p_ratio_weight:g}"
        ),
        "path": str(DATA),
        "dataset_mixture": mixture,
    }
    return source, cfg


def endpoint_p_ratio(first, final) -> float:
    return float(
        calc_p_ratio_rollout_sides(
            [clone_graph(first).cpu(), clone_graph(final).cpu()], -1
        )
    )


@torch.no_grad()
def evaluate(result, device, horizons):
    cfg = result["params"]
    rows = []
    result["ae"].eval()
    for split in ("train", "val", "test"):
        simulations = result[f"{split}_data"]
        for sim_index, sim in enumerate(simulations):
            for frame in horizons:
                if frame >= len(sim):
                    continue
                z = encode_frame_latent(
                    result["ae"],
                    sim,
                    frame,
                    pos_dim=2,
                    node_feature_mode="normalized_delta",
                    normalizers=result["normalizers"],
                    device=device,
                )
                decoded = decode_latent_to_graph(
                    result["ae"],
                    sim,
                    z,
                    frame,
                    pos_dim=2,
                    ae_target_mode="normalized_delta",
                    normalizers=result["normalizers"],
                    device=device,
                )
                rows.append(
                    {
                        "split": split,
                        "sim_index": sim_index,
                        "frame": frame,
                        "true_p_ratio": endpoint_p_ratio(sim[0], sim[frame]),
                        "pred_p_ratio": endpoint_p_ratio(sim[0], decoded),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = []
    for (split, frame), group in raw.groupby(["split", "frame"], sort=False):
        finite = np.isfinite(group["true_p_ratio"] * group["pred_p_ratio"])
        true = group.loc[finite, "true_p_ratio"].to_numpy(float)
        pred = group.loc[finite, "pred_p_ratio"].to_numpy(float)
        summary.append(
            {
                "split": split,
                "frame": int(frame),
                "used": int(finite.sum()),
                "p_ratio_r2": r2_score(true, pred),
                "p_ratio_pearson": pearson_r(true, pred),
                "p_ratio_mae": float(np.mean(np.abs(true - pred))),
                "true_std": float(np.std(true)),
                "pred_std": float(np.std(pred)),
            }
        )
    return raw, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-networks", type=int, default=100)
    parser.add_argument("--val-networks", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-graphs", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument(
        "--latent-dim", type=int, choices=tuple(range(1, 17)), default=2
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--strain-weight", type=float, default=100.0)
    parser.add_argument("--p-ratio-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--save-bundle",
        type=Path,
        help="Save the frozen AE immediately after training for propagator work.",
    )
    args = parser.parse_args()

    if not DATA.exists():
        raise FileNotFoundError(DATA)
    seed_everything(SEED)
    device = resolve_device(args.device)
    source, cfg = make_experiment(args, device)
    result = run_latent_experiment(source, cfg, device=device)
    if args.save_bundle is not None:
        args.save_bundle.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "spec": dict(source),
                "params": dict(result["params"]),
                "ae_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in result["ae"].state_dict().items()
                },
                "dyn_state_dict": {},
                "stats": result["stats"],
                "ae_history": result["ae_history"],
                "dyn_history": pd.DataFrame(),
            },
            args.save_bundle,
        )
        print(f"saved AE bundle: {args.save_bundle}", flush=True)
    raw, summary = evaluate(result, device, [25, 50, 99, 125, 150, 199])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"cv{args.latent_dim}_train{args.train_networks}"
        f"_strain{args.strain_weight:g}_pratio{args.p_ratio_weight:g}"
    ).replace(".", "p")
    raw.to_csv(OUTPUT / f"endpoint_rows_{suffix}.csv", index=False)
    summary.to_csv(OUTPUT / f"endpoint_summary_{suffix}.csv", index=False)
    result["ae_history"].to_csv(
        OUTPUT / f"history_{suffix}.csv", index=False
    )
    print("\nEndpoint AE oracle")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
