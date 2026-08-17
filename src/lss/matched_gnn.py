"""Matched spatial-GNN baselines for latent-rollout notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .graph import rollout
from .metrics import evaluate_rollout_pratio_sides
from .runner import _build_model_and_data, run_experiment
from .latent.experiment import evaluate_rollout_horizons, temperature_p_ratio
from .latent.simulation import pearson_r, r2_score


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


def _assert_same_split(reference_sims, candidate_sims, *, split_name: str) -> None:
    """Verify that independently loaded deterministic splits contain the same trajectories."""

    if len(reference_sims) != len(candidate_sims):
        raise ValueError(
            f"Matched GNN {split_name} split has {len(candidate_sims)} trajectories; "
            f"the latent split has {len(reference_sims)}."
        )
    for index, (reference, candidate) in enumerate(zip(reference_sims, candidate_sims)):
        reference_source = str(getattr(reference[0], "source_name", "unknown"))
        candidate_source = str(getattr(candidate[0], "source_name", "unknown"))
        if (
            reference_source != candidate_source
            or len(reference) != len(candidate)
            or not torch.equal(
                reference[0].x.detach().cpu(),
                candidate[0].x.detach().cpu(),
            )
        ):
            raise ValueError(
                f"Matched GNN {split_name} trajectory {index} does not match "
                "the cached latent split."
            )


def run_matched_shared_spatial_gnn(
    *,
    latent_result: dict,
    rollout_steps,
    eval_count_per_source: int | None,
    device,
    output_dir,
    epochs: int = 40,
    force_train: bool = False,
    history: int = 0,
    reid_trajectory_pratio: bool = False,
) -> pd.DataFrame:
    """Compare a shared latent propagator with one shared spatial GNN.

    The GNN uses the cached latent experiment's exact dataset mixture, split
    seed, train/validation trajectories, evaluation cohort, and target frames.
    Both methods are scored with the same p-ratio definition. When
    ``reid_trajectory_pratio`` is enabled, Reid uses the validation-selected
    cumulative directional-side strain/time fit used by notebook 06b.
    """

    params = latent_result["params"]
    mixture = [dict(item) for item in params.get("dataset_mixture", [])]
    if not mixture:
        raise ValueError("The shared matched-GNN comparison requires dataset_mixture.")
    history = int(history)
    if history < 0:
        raise ValueError("history must be non-negative.")
    train_frames = int(params["dyn_max_train_transitions_per_sim"])
    train_per_source = [int(item["train_count"]) for item in mixture]
    val_per_source = [int(item["val_count"]) for item in mixture]
    if len(set(train_per_source)) != 1 or len(set(val_per_source)) != 1:
        raise ValueError(
            "The paper comparison expects equal train and validation counts per source."
        )
    source_labels = {
        str(item["name"]): str(item.get("label", item["name"])) for item in mixture
    }
    base_seed = int(params["split_seed"])
    selection_target = 100
    run_name = (
        f"shared_spatial_gnn_{len(mixture)}src_train{train_per_source[0]}each_"
        f"frames{train_frames}_h{history}_seed{base_seed}_epochs{int(epochs)}"
    )
    cfg = ExperimentConfig(
        run_name=run_name,
        model_type="spatial",
        dataset_path=str(mixture[0]["path"]),
        dataset_mixture=mixture,
        train_count=sum(train_per_source),
        val_count=sum(val_per_source),
        output_root=str(Path(output_dir) / "matched_gnn"),
        pos_dim=int(params.get("pos_dim", 2)),
        history=history,
        limit=train_frames,
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
        seed=base_seed,
        rollout_steps=selection_target - history - 1,
        split_seed=base_seed,
        shuffle_dataset_within_source=True,
        stratify_temperature=False,
        node_features="positions",
        edge_multiplicity=int(params.get("edge_multiplicity", 1)),
        edge_vector_dim=int(params.get("edge_feature_dim", 12)),
        save_plots=False,
    )
    run_dir = Path(cfg.output_root) / cfg.run_name
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "final_checkpoint.pt"
    if force_train or not (metrics_path.exists() and checkpoint_path.exists()):
        run_experiment(cfg)
    else:
        print(f"loading matched shared GNN: {checkpoint_path}")

    resolved_device, _, train_data, val_data, model_inputs_cls, model = (
        _build_model_and_data(cfg)
    )
    _assert_same_split(latent_result["train_data"], train_data, split_name="train")
    _assert_same_split(latent_result["val_data"], val_data, split_name="validation")
    bundle = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_checkpoint(bundle["selected_checkpoint"]["path"])
    model.eval()
    model.freeze_normalizers = True

    selected_by_source: dict[str, list[tuple[int, object]]] = {
        source: [] for source in source_labels
    }
    for sim_idx, sim in enumerate(latent_result["test_data"]):
        source = str(getattr(sim[0], "source_name", "unknown"))
        if source not in selected_by_source:
            continue
        if (
            eval_count_per_source is None
            or len(selected_by_source[source]) < int(eval_count_per_source)
        ):
            selected_by_source[source].append((sim_idx, sim))

    reid_pratio_cfg = {
        **params,
        "p_ratio_estimator": "strain_gated_trajectory",
        "p_ratio_min_fit_frames": 4,
        "p_ratio_min_driven_strain_range": 1e-4,
        "p_ratio_side_quantile": 0.10,
    }

    gnn_rows = []
    with torch.no_grad():
        for source, indexed_sims in selected_by_source.items():
            sims = [sim for _, sim in indexed_sims]
            for target_frame in rollout_steps:
                target_frame = int(target_frame)
                if target_frame <= history:
                    continue
                if source == "reid" and reid_trajectory_pratio:
                    pred_values = []
                    true_values = []
                    position_values = []
                    for sim in sims:
                        if target_frame >= len(sim):
                            continue
                        predicted_path = rollout(
                            model=model,
                            input_graphs=[sim[index] for index in range(history + 1)],
                            num_steps=target_frame - history,
                            history=history,
                            pos_dim=cfg.pos_dim,
                            device=resolved_device,
                            model_inputs_cls=model_inputs_cls,
                            node_features=cfg.node_features,
                        )
                        pred_pr = temperature_p_ratio(
                            predicted_path,
                            cfg=reid_pratio_cfg,
                            last_index=-1,
                        )
                        true_pr = temperature_p_ratio(
                            sim,
                            cfg=reid_pratio_cfg,
                            last_index=target_frame,
                        )
                        pred_pos = predicted_path[-1].x[:, : cfg.pos_dim].cpu()
                        true_pos = sim[target_frame].x[:, : cfg.pos_dim].cpu()
                        pos_mse = float(
                            torch.nn.functional.mse_loss(pred_pos, true_pos).item()
                        )
                        if not all(np.isfinite([pred_pr, true_pr, pos_mse])):
                            continue
                        pred_values.append(pred_pr)
                        true_values.append(true_pr)
                        position_values.append(pos_mse)
                    metric = {
                        "used": len(pred_values),
                        "rollout_r2": r2_score(true_values, pred_values),
                        "rollout_pearson_r": pearson_r(true_values, pred_values),
                        "rollout_pos_mse": (
                            float(np.mean(position_values))
                            if position_values
                            else float("nan")
                        ),
                    }
                else:
                    metric = evaluate_rollout_pratio_sides(
                        model,
                        sims,
                        history,
                        target_frame - history - 1,
                        cfg.pos_dim,
                        resolved_device,
                        model_inputs_cls,
                        node_features=cfg.node_features,
                    )
                gnn_rows.append(
                    {
                        "source": source,
                        "dataset_label": source_labels[source],
                        "model": "spatial GNN",
                        "rollout_steps": target_frame,
                        "used": int(metric["used"]),
                        "p_ratio_r2": float(metric["rollout_r2"]),
                        "p_ratio_pearson": float(metric["rollout_pearson_r"]),
                        "position_mse": float(metric["rollout_pos_mse"]),
                    }
                )

    latent_raw = latent_result["rollout_rows"].query("split == 'test'").copy()
    true_column = (
        "endpoint_true_p_ratio"
        if "endpoint_true_p_ratio" in latent_raw
        else "true_p_ratio"
    )
    pred_column = (
        "endpoint_pred_p_ratio"
        if "endpoint_pred_p_ratio" in latent_raw
        else "pred_p_ratio"
    )
    latent_rows = []
    for source, indexed_sims in selected_by_source.items():
        if source == "reid" and reid_trajectory_pratio:
            reid_sims = [sim for _, sim in indexed_sims]
            _, reid_stats = evaluate_rollout_horizons(
                latent_result["ae"],
                latent_result["dyn"],
                reid_sims,
                latent_result["latent_stats"],
                cfg=reid_pratio_cfg,
                normalizers=latent_result["normalizers"],
                dataset=latent_result["label"],
                split_name="test",
                rollout_steps=rollout_steps,
                device=device,
            )
            for _, metric in reid_stats.iterrows():
                latent_rows.append(
                    {
                        "source": source,
                        "dataset_label": source_labels[source],
                        "model": "latent propagator",
                        "rollout_steps": int(metric["rollout_steps"]),
                        "used": int(metric["used"]),
                        "p_ratio_r2": float(metric["p_ratio_r2"]),
                        "p_ratio_pearson": float(metric["p_ratio_pearson"]),
                        "position_mse": float(metric["final_pos_mse"]),
                    }
                )
            continue
        selected_indices = {index for index, _ in indexed_sims}
        source_frame = latent_raw[
            latent_raw["source"].eq(source)
            & latent_raw["sim_idx"].isin(selected_indices)
        ]
        for target_frame in rollout_steps:
            group = source_frame[
                source_frame["rollout_steps"].eq(int(target_frame))
            ].replace([np.inf, -np.inf], np.nan)
            valid = group[[true_column, pred_column]].dropna()
            position = group["final_pos_mse"].dropna()
            latent_rows.append(
                {
                    "source": source,
                    "dataset_label": source_labels[source],
                    "model": "latent propagator",
                    "rollout_steps": int(target_frame),
                    "used": int(len(valid)),
                    "p_ratio_r2": (
                        r2_score(valid[true_column], valid[pred_column])
                        if len(valid) >= 2
                        else float("nan")
                    ),
                    "p_ratio_pearson": (
                        pearson_r(valid[true_column], valid[pred_column])
                        if len(valid) >= 2
                        else float("nan")
                    ),
                    "position_mse": (
                        float(position.mean()) if len(position) else float("nan")
                    ),
                }
            )

    return pd.concat(
        [pd.DataFrame(latent_rows), pd.DataFrame(gnn_rows)],
        ignore_index=True,
    )
