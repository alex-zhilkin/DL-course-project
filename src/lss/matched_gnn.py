"""Matched spatial-GNN baselines for latent-rollout notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .metrics import evaluate_rollout_pratio_sides
from .runner import _build_model_and_data, run_experiment


def _config(
    *,
    case_key: str,
    path,
    seed: int,
    train_count: int,
    val_count: int,
    train_frames: int,
    epochs: int,
    selection_target_frame: int,
    device,
    output_dir,
) -> ExperimentConfig:
    history = 1
    return ExperimentConfig(
        run_name=f"{case_key}_spatial_gnn_train{train_count}_seed{seed}",
        model_type="spatial",
        dataset_path=str(path),
        train_count=int(train_count),
        val_count=int(val_count),
        output_root=str(Path(output_dir) / "matched_gnn"),
        pos_dim=2,
        history=history,
        limit=int(train_frames),
        hidden_size=64,
        n_layers=3,
        model_extras={
            "num_mlp": 2,
            "use_skip": True,
            "final_decoder_local_skip": True,
        },
        learning_rate=1e-4,
        learning_rate_decay=0.995,
        weight_decay=1e-6,
        epochs=int(epochs),
        val_every=5,
        rollout_every=5,
        cv_eval_every=0,
        freeze_normalizers_after_epoch=10,
        device=str(device),
        seed=int(seed),
        # The graph evaluator observes frames 0..history. Its argument counts
        # steps after that warm start, so subtract history+1 to target the same
        # physical frame reported by the latent simulator.
        rollout_steps=int(selection_target_frame) - history - 1,
        split_seed=int(seed),
        shuffle_dataset_within_source=True,
        stratify_temperature=False,
        node_features="positions",
        edge_multiplicity=1,
        edge_vector_dim=12,
    )


def run_matched_spatial_gnns(
    *,
    cases: dict,
    latent_results: dict,
    case_keys,
    all_case_keys,
    base_seed: int,
    train_count: int,
    val_count: int,
    train_frames: int,
    rollout_steps,
    eval_count: int | None,
    device,
    output_dir,
    epochs: int = 80,
    force_train: bool = False,
) -> pd.DataFrame:
    """Train/load GNNs and return a matched latent-versus-GNN metric table."""

    gnn_parts = []
    latent_parts = []
    for case_key in case_keys:
        case = cases[case_key]
        seed = int(base_seed) + 101 * (list(all_case_keys).index(case_key) + 1)
        cfg = _config(
            case_key=case_key,
            path=case["path"],
            seed=seed,
            train_count=train_count,
            val_count=val_count,
            train_frames=train_frames,
            epochs=epochs,
            selection_target_frame=100,
            device=device,
            output_dir=output_dir,
        )
        run_dir = Path(cfg.output_root) / cfg.run_name
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "final_checkpoint.pt"
        if force_train or not (metrics_path.exists() and checkpoint_path.exists()):
            run_experiment(cfg)
        else:
            json.loads(metrics_path.read_text())
            print(f"loading matched GNN: {checkpoint_path}")

        resolved_device, _, _, _, model_inputs_cls, model = _build_model_and_data(cfg)
        bundle = torch.load(
            checkpoint_path,
            map_location=resolved_device,
            weights_only=False,
        )
        model.load_checkpoint(bundle["selected_checkpoint"]["path"])
        model.eval()
        model.freeze_normalizers = True

        test_sims = latent_results[case_key]["test_data"]
        if eval_count is not None:
            test_sims = test_sims[: int(eval_count)]
        rows = []
        with torch.no_grad():
            for target_frame in rollout_steps:
                target_frame = int(target_frame)
                if target_frame <= cfg.history:
                    continue
                metric = evaluate_rollout_pratio_sides(
                    model,
                    test_sims,
                    cfg.history,
                    target_frame - cfg.history - 1,
                    cfg.pos_dim,
                    resolved_device,
                    model_inputs_cls,
                    node_features=cfg.node_features,
                )
                rows.append(
                    {
                        "case": case_key,
                        "dataset_label": case["label"],
                        "model": "spatial GNN",
                        "rollout_steps": target_frame,
                        "used": metric["used"],
                        "p_ratio_r2": metric["rollout_r2"],
                        "p_ratio_pearson": metric["rollout_pearson_r"],
                        "position_mse": metric["rollout_pos_mse"],
                    }
                )
        gnn_parts.append(pd.DataFrame(rows))

        latent = latent_results[case_key]["rollout_stats"].query(
            "split == 'test'"
        ).copy()
        latent = latent[latent["rollout_steps"].isin(list(rollout_steps))]
        latent["case"] = case_key
        latent["dataset_label"] = case["label"]
        latent["model"] = "latent simulator"
        latent["position_mse"] = (
            latent["final_pos_mse"] if "final_pos_mse" in latent else np.nan
        )
        latent_parts.append(
            latent[
                [
                    "case",
                    "dataset_label",
                    "model",
                    "rollout_steps",
                    "used",
                    "p_ratio_r2",
                    "p_ratio_pearson",
                    "position_mse",
                ]
            ]
        )

    return pd.concat([*latent_parts, *gnn_parts], ignore_index=True)
