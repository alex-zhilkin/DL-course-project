"""Shared-AE latent propagator comparison workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from ..data import resolve_dataset_splits
from .experiment import build_autoencoder, rollout_steps_for_sims, seed_everything
from .simulation import (
    fit_ae_target_stats,
    fit_edge_stats,
    fit_node_feature_stats,
    make_frame_index,
    make_transition_index,
)
from .training import (
    LatentNormalizer,
    TrainingConfig,
    fit_latent_step_stats,
    latent_rollout_eval,
    latent_velocity_rollout_eval,
    make_multistep_transition_index,
    make_propagator,
    make_velocity_transition_index,
    train_autoencoder,
    train_propagator,
)


def run_propagator_comparison(
    dataset_spec: dict,
    cfg: dict,
    propagator_specs: list[dict],
    *,
    seed: int,
    device,
) -> dict[str, object]:
    """Train one AE, compare propagators, and return histories and rollout tables."""

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data, val_data, test_data, split_info = resolve_dataset_splits(
        dataset_spec["path"],
        train_count=int(dataset_spec["train_count"]),
        val_count=int(dataset_spec["val_count"]),
        split_seed=cfg["split_seed"],
        shuffle_within_source=True,
    )
    rollout_grid = rollout_steps_for_sims(
        val_data + test_data,
        cfg["rollout_steps_grid"],
        frame_skip=cfg["frame_skip"],
    )

    train_frames = make_frame_index(
        train_data,
        frame_skip=cfg["frame_skip"],
        max_frames_per_sim=cfg["ae_train_frames_per_sim"],
        include_last=True,
        start_frame_order=cfg["train_frame_start_order"],
    )
    val_frames = make_frame_index(
        val_data,
        frame_skip=cfg["frame_skip"],
        max_frames_per_sim=cfg["ae_train_frames_per_sim"],
        include_last=True,
        start_frame_order=cfg["train_frame_start_order"],
    )
    train_steps = make_transition_index(
        train_data,
        frame_skip=cfg["frame_skip"],
        max_frames_per_sim=cfg["dyn_train_transitions_per_sim"],
    )
    val_steps = make_transition_index(
        val_data,
        frame_skip=cfg["frame_skip"],
        max_frames_per_sim=cfg["dyn_train_transitions_per_sim"],
    )
    train_multistep = make_multistep_transition_index(
        train_data,
        horizons=cfg["multistep_horizons"],
        frame_skip=cfg["frame_skip"],
        max_starts_per_sim=cfg["multistep_max_starts_per_sim"],
    )
    val_multistep = make_multistep_transition_index(
        val_data,
        horizons=cfg["multistep_horizons"],
        frame_skip=cfg["frame_skip"],
        max_starts_per_sim=cfg["multistep_max_starts_per_sim"],
    )
    train_velocity = make_velocity_transition_index(
        train_data,
        frame_skip=cfg["frame_skip"],
        max_frames_per_sim=cfg["dyn_train_transitions_per_sim"],
    )
    val_velocity = make_velocity_transition_index(
        val_data,
        frame_skip=cfg["frame_skip"],
        max_frames_per_sim=cfg["dyn_train_transitions_per_sim"],
    )

    target_mean, target_std = fit_ae_target_stats(
        train_data,
        train_frames,
        pos_dim=cfg["pos_dim"],
        batch_graphs=cfg["batch_graphs"],
        device=device,
        target_mode=cfg["ae_target_mode"],
    )
    node_mean, node_std = fit_node_feature_stats(
        train_data,
        train_frames,
        pos_dim=cfg["pos_dim"],
        batch_graphs=cfg["batch_graphs"],
        device=device,
        node_feature_mode=cfg["node_feature_mode"],
    )
    edge_mean, edge_std = fit_edge_stats(
        train_data,
        train_frames,
        pos_dim=cfg["pos_dim"],
        batch_graphs=cfg["batch_graphs"],
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
    edge_dim = int(edge_mean.numel())
    ae_path = output_dir / (
        f"shared_ae_{cfg['dataset_name']}_cv{cfg['latent_dim']:02d}"
        f"_nets{dataset_spec['train_count']:03d}"
        f"_frames{cfg['ae_train_frames_per_sim']:03d}.pt"
    )
    if ae_path.exists() and not cfg["force_train_ae"]:
        bundle = torch.load(ae_path, map_location=device, weights_only=False)
        ae_model = build_autoencoder(cfg, edge_dim=edge_dim, device=device)
        ae_model.load_state_dict(bundle["ae_state_dict"])
        ae_history = pd.DataFrame(bundle.get("ae_history", []))
    else:
        ae_result = train_autoencoder(
            build_autoencoder(cfg, edge_dim=edge_dim, device=device),
            train_data,
            val_data,
            train_frames,
            val_frames,
            batch_graphs=cfg["batch_graphs"],
            pos_dim=cfg["pos_dim"],
            node_feature_mode=cfg["node_feature_mode"],
            ae_target_mode=cfg["ae_target_mode"],
            normalizers=normalizers,
            device=device,
            config=TrainingConfig(
                max_epochs=cfg["ae_max_epochs"],
                patience=cfg["ae_patience"],
                learning_rate=cfg["ae_lr"],
                weight_decay=cfg["ae_weight_decay"],
                min_delta=cfg["early_stop_min_delta"],
            ),
        )
        ae_model = ae_result.model
        ae_history = ae_result.history
        torch.save(
            {
                "cfg": dict(cfg),
                "source_spec": dict(dataset_spec),
                "normalizers": {key: value.detach().cpu() for key, value in normalizers.items()},
                "ae_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in ae_model.state_dict().items()
                },
                "ae_history": ae_history.to_dict("records"),
            },
            ae_path,
        )
    ae_model.eval()
    for parameter in ae_model.parameters():
        parameter.requires_grad_(False)

    latent_stats = fit_latent_step_stats(
        ae_model,
        train_data,
        train_steps,
        batch_graphs=cfg["batch_graphs"],
        pos_dim=cfg["pos_dim"],
        node_feature_mode=cfg["node_feature_mode"],
        normalizers=normalizers,
        device=device,
    ).to(device)

    histories = []
    models = {}
    train_config = TrainingConfig(
        max_epochs=cfg["propagator_max_epochs"],
        patience=cfg["propagator_patience"],
        learning_rate=cfg["propagator_lr"],
        weight_decay=cfg["propagator_weight_decay"],
        min_delta=cfg["early_stop_min_delta"],
    )
    for spec_idx, spec in enumerate(propagator_specs):
        objective = spec.get("train_objective", "one_step")
        if objective == "velocity":
            train_rows, val_rows = train_velocity, val_velocity
        elif objective == "multistep":
            train_rows, val_rows = train_multistep, val_multistep
        else:
            train_rows, val_rows = train_steps, val_steps
        for repeat_idx in range(1, int(cfg["propagator_repeats"]) + 1):
            repeat_seed = int(seed + 9176 * repeat_idx + 104729 * (spec_idx + 1))
            seed_everything(repeat_seed)
            name = spec["name"]
            path = output_dir / (
                f"prop_{name}_{cfg['dataset_name']}_cv{cfg['latent_dim']:02d}"
                f"_nets{dataset_spec['train_count']:03d}"
                f"_frames{cfg['dyn_train_transitions_per_sim']:03d}"
                f"_rep{repeat_idx:02d}.pt"
            )
            model = make_propagator(
                cfg["latent_dim"],
                int(spec.get("hidden_size", cfg["hidden_size"])),
                loss_mode=spec["loss_mode"],
                model_type=spec.get("model_type"),
            ).to(device)
            if path.exists() and not cfg["force_train_propagators"]:
                bundle = torch.load(path, map_location=device, weights_only=False)
                model.load_state_dict(bundle["state_dict"])
                history = pd.DataFrame(bundle.get("history", []))
            else:
                trained = train_propagator(
                    model,
                    ae_model,
                    train_data,
                    val_data,
                    train_rows,
                    val_rows,
                    latent_stats,
                    batch_graphs=cfg["batch_graphs"],
                    pos_dim=cfg["pos_dim"],
                    node_feature_mode=cfg["node_feature_mode"],
                    normalizers=normalizers,
                    device=device,
                    loss_mode=spec["loss_mode"],
                    objective=objective,
                    horizons=cfg["multistep_horizons"] if objective == "multistep" else None,
                    config=train_config,
                )
                model = trained.model
                history = trained.history
                torch.save(
                    {
                        "spec": dict(spec),
                        "repeat_idx": repeat_idx,
                        "repeat_seed": repeat_seed,
                        "state_dict": {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        },
                        "history": history.to_dict("records"),
                    },
                    path,
                )
            history = history.assign(
                propagator=name,
                repeat_idx=repeat_idx,
                repeat_seed=repeat_seed,
                loss_mode=spec["loss_mode"],
                train_objective=objective,
            )
            histories.append(history)
            models[(name, repeat_idx)] = (model.eval(), spec, repeat_seed)

    raw_parts = []
    stats_parts = []
    for (name, repeat_idx), (model, spec, repeat_seed) in models.items():
        for split_name, sims in (("val", val_data), ("test", test_data)):
            for steps in rollout_grid:
                for method in cfg["p_ratio_methods"]:
                    evaluator = (
                        latent_velocity_rollout_eval
                        if spec.get("train_objective") == "velocity"
                        else latent_rollout_eval
                    )
                    kwargs = {
                        "dataset": f"{dataset_spec['label']} | {name}",
                        "split_name": split_name,
                        "rollout_steps": int(steps),
                        "pos_dim": cfg["pos_dim"],
                        "ae_target_mode": cfg["ae_target_mode"],
                        "node_feature_mode": cfg["node_feature_mode"],
                        "normalizers": normalizers,
                        "device": device,
                        "p_ratio_method": method,
                    }
                    if evaluator is latent_velocity_rollout_eval:
                        kwargs["initial_velocity"] = "mean"
                    else:
                        kwargs["loss_mode"] = spec["loss_mode"]
                    raw, stats = evaluator(
                        ae_model,
                        model,
                        sims,
                        latent_stats,
                        **kwargs,
                    )
                    metadata = {
                        "propagator": name,
                        "repeat_idx": repeat_idx,
                        "repeat_seed": repeat_seed,
                        "loss_mode": spec["loss_mode"],
                        "train_objective": spec.get("train_objective", "one_step"),
                        "p_ratio_method": method,
                    }
                    for key, value in metadata.items():
                        raw[key] = value
                        stats[key] = value
                    raw_parts.append(raw)
                    stats_parts.append(pd.DataFrame([stats]))

    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "split_info": pd.DataFrame(split_info),
        "rollout_steps_grid": rollout_grid,
        "ae_model": ae_model,
        "ae_history": ae_history,
        "normalizers": normalizers,
        "latent_stats": latent_stats,
        "propagator_history": pd.concat(histories, ignore_index=True),
        "rollout_raw": pd.concat(raw_parts, ignore_index=True),
        "rollout_stats": pd.concat(stats_parts, ignore_index=True),
    }


__all__ = ["run_propagator_comparison"]
