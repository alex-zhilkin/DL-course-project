"""Matched AE screen with explicit splits, source-wise metrics, and frozen PCA controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box
from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.latent.training import encode_frame_latent, decode_latent_to_graph


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strains(reference, graph):
    sides = directional_side_indices_from_box(reference)
    def extent(g):
        p = g.x[:, :2].detach().cpu().numpy()
        return np.array([p[sides["right"], 0].mean() - p[sides["left"], 0].mean(),
                         p[sides["top"], 1].mean() - p[sides["bottom"], 1].mean()])
    initial = extent(reference)
    return (extent(graph) - initial) / initial


def score(y, pred):
    denominator = np.square(y - y.mean()).sum()
    return float(1 - np.square(y - pred).sum() / denominator) if len(y) > 1 and denominator > 0 else float("nan")


def evaluate(result, output, manifest):
    model, params, norms = result["ae"], result["params"], result["normalizers"]
    device = torch.device("cpu")
    def encode(sim, frame):
        return encode_frame_latent(model, sim, frame, pos_dim=2,
                                   node_feature_mode=params["node_feature_mode"],
                                   normalizers=norms, device=device)
    # PCA is fit only on training networks, at eleven predeclared frames.
    latents, latent_rows, source_strains = [], [], {}
    with torch.no_grad():
        for i, sim in enumerate(result["train_data"]):
            source = sim[0].source_name
            source_strains.setdefault(source, []).append(np.abs(strains(sim[0], sim[100])))
            for frame in range(0, 101, 10):
                z = encode(sim, frame).cpu().numpy().reshape(-1)
                latents.append(z)
                latent_rows.append({"split": "train", "source": source, "sim_idx": i,
                                    "frame": frame, **{f"z{j}": float(v) for j, v in enumerate(z)}})
        z_train = np.stack(latents)
        center = z_train.mean(axis=0)
        _, singular, components = np.linalg.svd(z_train - center, full_matrices=False)
        axes = {source: int(np.argmax(np.median(values, axis=0))) for source, values in source_strains.items()}
        np.savez(output / "training_pca.npz", center=center, components=components,
                 singular_values=singular)
        (output / "metric_definition.json").write_text(json.dumps({
            "driven_axis_by_source": axes, "axis_selection": "Largest median absolute training frame-100 side strain",
            "p_ratio": "-transverse strain / driven strain, using the same training-selected axis for true and predicted fields",
            "min_abs_driven_strain": 1e-5, "side_quantile": 0.1,
            "legacy_p_ratio": "graph_utils.calc_p_ratio_rollout_sides, also retained for historical comparisons",
            "PCA_fit_frames": list(range(0, 101, 10)), "split": "validation only",
        }, indent=2) + "\n")
        rows = []
        source_offsets = {source: 0 for source in axes}
        for i, sim in enumerate(result["val_data"]):
            source = sim[0].source_name
            original_index = manifest["sources"][source]["split_indices"]["val"][source_offsets[source]]
            source_offsets[source] += 1
            for frame in (0, 10, 50, 100):
                z = encode(sim, frame)
                vector = z.cpu().numpy().reshape(-1)
                latent_rows.append({"split": "val", "source": source, "sim_idx": i,
                                    "frame": frame, **{f"z{j}": float(v) for j, v in enumerate(vector)}})
                variants = {"ae": vector, "constant_training_mean": center}
                for k in (1, 2):
                    if k < len(vector):
                        basis = components[:k]
                        variants[f"training_pca_{k}"] = center + (vector - center) @ basis.T @ basis
                true_strain = strains(sim[0], sim[frame])
                axis = axes[source]
                for name, values in variants.items():
                    pred = decode_latent_to_graph(model, sim,
                        torch.as_tensor(values, dtype=z.dtype, device=device).reshape_as(z), frame,
                        pos_dim=2, ae_target_mode=params["ae_target_mode"], normalizers=norms, device=device)
                    pred_strain = strains(sim[0], pred)
                    difference = pred.x[:, :2] - sim[frame].x[:, :2].cpu()
                    true_pr = -true_strain[1-axis] / true_strain[axis] if abs(true_strain[axis]) >= 1e-5 else np.nan
                    pred_pr = -pred_strain[1-axis] / pred_strain[axis] if abs(pred_strain[axis]) >= 1e-5 else np.nan
                    rows.append({"source": source, "original_index": original_index, "frame": frame,
                        "control": name, "position_mse": float(difference.square().mean()),
                        "x_mse": float(difference[:, 0].square().mean()), "y_mse": float(difference[:, 1].square().mean()),
                        "true_strain_x": true_strain[0], "true_strain_y": true_strain[1],
                        "pred_strain_x": pred_strain[0], "pred_strain_y": pred_strain[1],
                        "true_p_ratio": true_pr, "pred_p_ratio": pred_pr,
                        "legacy_true_p_ratio": calc_p_ratio_rollout_sides(sim, frame),
                        "legacy_pred_p_ratio": calc_p_ratio_rollout_sides([sim[0], pred], -1)})
            print(f"Evaluated validation {source} index={original_index}", flush=True)
    pd.DataFrame(latent_rows).to_csv(output / "latent_coordinates.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "validation_rows.csv", index=False)
    summaries = []
    for (source, frame, control), group in raw.groupby(["source", "frame", "control"]):
        summary = {"source": source, "frame": frame, "control": control, "total": len(group),
                   "position_mse": group.position_mse.mean(), "x_mse": group.x_mse.mean(), "y_mse": group.y_mse.mean()}
        for prefix in ("", "legacy_"):
            y, pred = group[f"{prefix}true_p_ratio"].to_numpy(), group[f"{prefix}pred_p_ratio"].to_numpy()
            valid = np.isfinite(y) & np.isfinite(pred)
            summary.update({f"{prefix}valid": int(valid.sum()), f"{prefix}true_valid": int(np.isfinite(y).sum()),
                f"{prefix}p_ratio_r2": score(y[valid], pred[valid]) if valid.any() else np.nan,
                f"{prefix}p_ratio_mae": float(np.abs(y[valid] - pred[valid]).mean()) if valid.any() else np.nan,
                f"{prefix}target_variance": float(np.var(y[np.isfinite(y)])) if np.isfinite(y).any() else np.nan})
        for axis in ("x", "y"):
            summary[f"strain_{axis}_mae"] = (group[f"pred_strain_{axis}"] - group[f"true_strain_{axis}"]).abs().mean()
        summaries.append(summary)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "validation_summary.csv", index=False)
    print(summary.loc[(summary.frame == 100) & (summary.control == "ae")].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, choices=(2, 4, 6, 8), required=True)
    parser.add_argument("--sources", choices=("shared3", "shared4", "shared_no_mixed", "reid", "depablo_low_temp", "depablo_mixed_temp", "lj_noisy"), required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    base = ROOT / "notebooks/results/latent_matched_study"
    manifest_path = CODE / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selected = list(manifest["sources"])
    if args.sources == "shared3":
        selected.remove("lj_noisy")
    elif args.sources == "shared_no_mixed":
        selected.remove("depablo_mixed_temp")
    elif args.sources != "shared4":
        selected = [args.sources]
    output = base / f"{args.sources}_d{args.dim}_s{args.seed}"
    output.mkdir(exist_ok=False)
    started = time.time()
    mixture = []
    for name in selected:
        spec = manifest["sources"][name]
        if sha256(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Dataset changed since split freeze: {name}")
        indices = {**spec["split_indices"], "test": []}
        mixture.append({"name": name, "label": name, "path": spec["path"],
            "train_count": 30, "val_count": 20, "split_indices": indices,
            "append_lj_indicator": True, "lj_max_graph_distance": 3 if name == "lj_noisy" else None})
    source = {"dataset_name": "latent_matched_study", "source_name": args.sources,
              "label": args.sources, "path": mixture[0]["path"], "dataset_mixture": mixture}
    cfg = {"split_seed": 123, "model_seed": args.seed, "pos_dim": 2,
        "batch_graphs": 32, "frame_skip": 1, "coordinate_normalization": "position_normalization",
        "edge_mode": "compact_stored", "static_context_use_physical_reference": True,
        "should_rollout": False, "should_train_propagator": False,
        "force_train": True, "cache_path": str(output / "ae.pt"), "early_stop_min_delta": 1e-5,
        "ae_config": {"model": "attention", "latent_dim": args.dim, "latent_tokens": 32,
            "hidden_size": 128, "target_mode": "normalized_delta", "node_feature_mode": "normalized_delta",
            "edge_feature_dim": 5, "max_train_frames_per_sim": 101, "max_val_frames_per_sim": 101,
            "train_frame_skip": 1, "val_frame_skip": 1, "max_epochs": 40, "patience": 8,
            "lr": 1e-4, "weight_decay": 1e-5, "mix_sources": True, "balance_sources": True,
            "train_rows_per_source": 3030,
            "gradient_method": "source_mean", "checkpoint_metric": "val_max_source_reconstruction",
            "checkpoint_mode": "min"}}
    (output / "recipe.json").write_text(json.dumps({"source": source, "config": cfg,
        "manifest_sha256": sha256(manifest_path), "code_root": str(CODE), "job_id": os.environ.get("PBS_JOBID"),
        "host": platform.node(), "torch": torch.__version__}, indent=2) + "\n")
    seed_everything(args.seed)
    result = run_latent_experiment(source, cfg, device=torch.device("cpu"))
    if result["test_data"]:
        raise AssertionError("Representation screens must not expose reserved test trajectories")
    result["ae_history"].to_csv(output / "ae_history.csv", index=False)
    evaluate(result, output, manifest)
    (output / "completed.json").write_text(json.dumps({"seconds": time.time() - started,
        "job_id": os.environ.get("PBS_JOBID"), "status": "completed", "split": "validation"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
