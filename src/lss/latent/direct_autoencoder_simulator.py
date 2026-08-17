"""Direct next-frame autoencoder simulator used by notebook 10."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from graph_utils import calc_p_ratio_rollout_sides

from lss.data import resolve_dataset_splits
from lss.graph import clone_graph

from .models import (
    NodeDeltaAttentionAutoEncoder,
    NodeDeltaDirectAttentionAutoEncoder,
    NodeDeltaMLPAutoEncoder,
    NodeDeltaSingleStageAttentionAutoEncoder,
)
from .simulation import (
    ae_target_tensor,
    batch_delta_graphs,
    fit_ae_target_stats,
    fit_edge_stats,
    fit_node_feature_stats,
    iter_batches,
    make_transition_index,
    pearson_r,
    r2_score,
)
from .training import _source_mixed_rows


def _model_class(model_type: str):
    normalized = str(model_type).lower()
    if normalized in {"single_stage_attention", "direct_latent_attention"}:
        return NodeDeltaSingleStageAttentionAutoEncoder
    if normalized in {"direct_attention", "node_to_latent_attention"}:
        return NodeDeltaDirectAttentionAutoEncoder
    if normalized in {"mlp", "mean_mlp", "mean_pool"}:
        return NodeDeltaMLPAutoEncoder
    if normalized in {"attention", "pyramid_attention"}:
        return NodeDeltaAttentionAutoEncoder
    raise ValueError(f"Unknown direct autoencoder model type: {model_type}")


def _evenly_limited_transitions(sims, limit: int | None) -> list[tuple[int, int, int]]:
    rows = make_transition_index(sims, max_frames_per_sim=None)
    if limit is None:
        return rows
    limit = int(limit)
    if limit < 1:
        raise ValueError("transitions_per_trajectory must be positive.")
    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for row in rows:
        grouped.setdefault(int(row[0]), []).append(row)
    selected = []
    for sim_idx in sorted(grouped):
        sim_rows = grouped[sim_idx]
        if len(sim_rows) <= limit:
            selected.extend(sim_rows)
        elif limit == 1:
            selected.append(sim_rows[0])
        else:
            indices = [
                round(index * (len(sim_rows) - 1) / (limit - 1))
                for index in range(limit)
            ]
            selected.extend(sim_rows[index] for index in indices)
    return selected


def _normalizers_to_device(normalizers: dict, device) -> dict:
    return {key: value.to(device) for key, value in normalizers.items()}


def _build_model(cfg: dict, normalizers: dict, device):
    model_cls = _model_class(cfg["autoencoder_model"])
    model = model_cls(
        pos_dim=int(cfg["pos_dim"]),
        node_feature_dim=int(normalizers["node_feature_mean"].numel()),
        edge_dim=int(normalizers["edge_mean"].numel()),
        hidden_size=int(cfg["hidden_size"]),
        latent_dim=int(cfg["latent_dim"]),
        latent_tokens=int(cfg["latent_tokens"]),
        reconstruction_dim=int(normalizers["target_mean"].numel()),
    ).to(device)
    model.edge_mode = str(cfg.get("edge_mode", "recomputed_stored"))
    return model


def _direct_target_tensor(input_batch, target_batch, target_mode: str):
    """Return the one-step target used by the direct simulator.

    ``normalized_step_delta`` deliberately predicts only the update from t to
    t+1.  This removes the very strong identity shortcut present when a model
    is asked to reconstruct the full displacement at t+1 from displacement at
    t, while remaining a strictly one-step (teacher-forced) objective.
    """

    if target_mode == "normalized_step_delta":
        return target_batch["normalized_delta"] - input_batch["normalized_delta"]
    return ae_target_tensor(target_batch, target_mode)


def _fit_direct_target_stats(
    sims,
    rows,
    *,
    cfg: dict,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg["target_mode"] != "normalized_step_delta":
        target_rows = [(sim_idx, t1) for sim_idx, _, t1 in rows]
        return fit_ae_target_stats(
            sims,
            target_rows,
            pos_dim=int(cfg["pos_dim"]),
            batch_graphs=int(cfg["batch_graphs"]),
            device=device,
            target_mode=cfg["target_mode"],
            node_feature_mode=cfg["node_feature_mode"],
        )

    chunks = []
    for row_batch in iter_batches(
        rows, int(cfg["batch_graphs"]), shuffle=False
    ):
        input_batch = batch_delta_graphs(
            sims,
            [(sim_idx, t0) for sim_idx, t0, _ in row_batch],
            pos_dim=int(cfg["pos_dim"]),
            device=device,
            node_feature_mode=cfg["node_feature_mode"],
            edge_mode=cfg["edge_mode"],
        )
        target_batch = batch_delta_graphs(
            sims,
            [(sim_idx, t1) for sim_idx, _, t1 in row_batch],
            pos_dim=int(cfg["pos_dim"]),
            device=device,
            node_feature_mode=cfg["node_feature_mode"],
            edge_mode=cfg["edge_mode"],
        )
        chunks.append(
            _direct_target_tensor(
                input_batch, target_batch, cfg["target_mode"]
            ).detach()
        )
    targets = torch.cat(chunks, dim=0)
    return (
        targets.mean(dim=0, keepdim=True),
        targets.std(dim=0, keepdim=True).clamp_min(1e-6),
    )


def _transition_epoch(
    model,
    sims,
    rows,
    *,
    cfg: dict,
    normalizers: dict,
    device,
    optimizer=None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    epoch_rows = (
        _source_mixed_rows(sims, rows, shuffle=True)
        if is_train and bool(cfg.get("mix_sources", False))
        else rows
    )
    for row_batch in iter_batches(
        epoch_rows,
        int(cfg["batch_graphs"]),
        shuffle=is_train and not bool(cfg.get("mix_sources", False)),
    ):
        input_rows = [(sim_idx, t0) for sim_idx, t0, _ in row_batch]
        target_rows = [(sim_idx, t1) for sim_idx, _, t1 in row_batch]
        input_batch = batch_delta_graphs(
            sims,
            input_rows,
            pos_dim=int(cfg["pos_dim"]),
            device=device,
            node_feature_mode=cfg["node_feature_mode"],
            edge_mode=cfg["edge_mode"],
        )
        target_batch = batch_delta_graphs(
            sims,
            target_rows,
            pos_dim=int(cfg["pos_dim"]),
            device=device,
            node_feature_mode=cfg["node_feature_mode"],
            edge_mode=cfg["edge_mode"],
        )
        node_norm = (
            input_batch["node_feature"] - normalizers["node_feature_mean"]
        ) / normalizers["node_feature_std"]
        edge_norm = (
            input_batch["edge_attr"] - normalizers["edge_mean"]
        ) / normalizers["edge_std"]
        ref_edge_norm = (
            input_batch["ref_edge_attr"] - normalizers["edge_mean"]
        ) / normalizers["edge_std"]
        target = _direct_target_tensor(
            input_batch, target_batch, cfg["target_mode"]
        )
        target_norm = (
            target - normalizers["target_mean"]
        ) / normalizers["target_std"]
        prediction_norm, _ = model(
            node_norm,
            input_batch["ref_pos"],
            edge_norm,
            ref_edge_norm,
            input_batch["edge_index"],
            input_batch["batch"],
        )
        loss = F.mse_loss(prediction_norm, target_norm)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def _cache_key(dataset_spec: dict, cfg: dict) -> str:
    payload = {
        "dataset": {**dataset_spec, "path": str(dataset_spec["path"])},
        "config": {
            key: value
            for key, value in cfg.items()
            if key not in {"device", "force_train", "cache_path"}
        },
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def run_direct_autoencoder_case(dataset_spec: dict, cfg: dict, *, device) -> dict:
    """Train or load an individual or mixed-source next-frame AE simulator."""

    cfg = dict(cfg)
    cache_path = Path(cfg["cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    key = _cache_key(dataset_spec, cfg)
    dataset_mixture = dataset_spec.get("dataset_mixture")
    train_data, val_data, test_data, split_info = resolve_dataset_splits(
        dataset_spec.get(
            "path",
            dataset_mixture[0]["path"] if dataset_mixture else None,
        ),
        train_count=int(cfg.get("train_count", 0)),
        val_count=int(cfg.get("val_count", 0)),
        dataset_mixture=dataset_mixture,
        split_seed=int(cfg["split_seed"]),
        shuffle_within_source=True,
        stratify_temperature=False,
        edge_multiplicity=int(dataset_spec.get("edge_multiplicity", 1)),
        edge_vector_dim=int(dataset_spec.get("edge_vector_dim", 2)),
    )
    train_rows = _evenly_limited_transitions(
        train_data, cfg.get("transitions_per_trajectory")
    )
    val_rows = _evenly_limited_transitions(
        val_data, cfg.get("validation_transitions_per_trajectory")
    )

    if cache_path.exists() and not bool(cfg.get("force_train", False)):
        bundle = torch.load(cache_path, map_location=device, weights_only=False)
        if bundle.get("cache_key") == key:
            normalizers = _normalizers_to_device(bundle["normalizers"], device)
            model = _build_model(cfg, normalizers, device)
            model.load_state_dict(bundle["model_state_dict"])
            model.eval()
            print(f"loading direct autoencoder simulator: {cache_path}")
            return {
                "model": model,
                "normalizers": normalizers,
                "history": pd.DataFrame(bundle["history"]),
                "train_data": train_data,
                "val_data": val_data,
                "test_data": test_data,
                "split_info": pd.DataFrame(split_info),
                "train_rows": train_rows,
                "val_rows": val_rows,
                "config": cfg,
                "dataset_spec": dataset_spec,
            }
        print(f"ignoring stale direct-simulator cache: {cache_path}")

    input_train_rows = [(sim_idx, t0) for sim_idx, t0, _ in train_rows]
    target_mean, target_std = _fit_direct_target_stats(
        train_data,
        train_rows,
        cfg=cfg,
        device=device,
    )
    node_mean, node_std = fit_node_feature_stats(
        train_data,
        input_train_rows,
        pos_dim=int(cfg["pos_dim"]),
        batch_graphs=int(cfg["batch_graphs"]),
        device=device,
        node_feature_mode=cfg["node_feature_mode"],
    )
    edge_mean, edge_std = fit_edge_stats(
        train_data,
        input_train_rows,
        pos_dim=int(cfg["pos_dim"]),
        batch_graphs=int(cfg["batch_graphs"]),
        device=device,
        edge_mode=cfg["edge_mode"],
    )
    normalizers = {
        "target_mean": target_mean,
        "target_std": target_std,
        "node_feature_mean": node_mean,
        "node_feature_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
    }
    model = _build_model(cfg, normalizers, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        train_loss = _transition_epoch(
            model,
            train_data,
            train_rows,
            cfg=cfg,
            normalizers=normalizers,
            device=device,
            optimizer=optimizer,
        )
        with torch.no_grad():
            val_loss = _transition_epoch(
                model,
                val_data,
                val_rows,
                cfg=cfg,
                normalizers=normalizers,
                device=device,
            )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )
        print(
            f"direct-AE epoch={epoch:03d} train={train_loss:.6g} "
            f"val={val_loss:.6g}",
            flush=True,
        )
        if np.isfinite(val_loss) and val_loss < best_val - float(cfg["min_delta"]):
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(cfg["patience"]):
            break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "cache_key": key,
            "config": cfg,
            "dataset_spec": dataset_spec,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "normalizers": {
                key: value.detach().cpu() for key, value in normalizers.items()
            },
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        cache_path,
    )
    return {
        "model": model,
        "normalizers": normalizers,
        "history": pd.DataFrame(history),
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "split_info": pd.DataFrame(split_info),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "config": cfg,
        "dataset_spec": dataset_spec,
    }


def predict_next_graph(
    result: dict,
    reference_graph,
    current_graph,
    *,
    device,
    previous_graph=None,
):
    """Predict one next graph from reference, previous, and current states."""

    model = result["model"]
    cfg = result["config"]
    normalizers = result["normalizers"]
    # Three entries make frame_node_feature(t=2) compute velocity from the
    # previous and current frames.  Repeating the current frame gives a clean
    # zero-velocity state for the first prediction.
    previous_graph = current_graph if previous_graph is None else previous_graph
    working_sim = [reference_graph, previous_graph, current_graph]
    batch = batch_delta_graphs(
        [working_sim],
        [(0, 2)],
        pos_dim=int(cfg["pos_dim"]),
        device=device,
        node_feature_mode=cfg["node_feature_mode"],
        edge_mode=cfg["edge_mode"],
    )
    node_norm = (
        batch["node_feature"] - normalizers["node_feature_mean"]
    ) / normalizers["node_feature_std"]
    edge_norm = (
        batch["edge_attr"] - normalizers["edge_mean"]
    ) / normalizers["edge_std"]
    ref_edge_norm = (
        batch["ref_edge_attr"] - normalizers["edge_mean"]
    ) / normalizers["edge_std"]
    prediction_norm, _ = model(
        node_norm,
        batch["ref_pos"],
        edge_norm,
        ref_edge_norm,
        batch["edge_index"],
        batch["batch"],
    )
    predicted_target = (
        prediction_norm * normalizers["target_std"]
        + normalizers["target_mean"]
    )
    if cfg["target_mode"] == "normalized_step_delta":
        predicted_position = (
            batch["cur_pos"] + predicted_target * batch["position_scale"]
        )
    elif cfg["target_mode"] in {
        "normalized_delta",
        "self_normalized_delta",
        "relative_delta",
    }:
        predicted_position = (
            batch["ref_pos"] + predicted_target * batch["position_scale"]
        )
    elif cfg["target_mode"] in {"delta", "displacement"}:
        predicted_position = batch["ref_pos"] + predicted_target
    elif cfg["target_mode"] in {"position", "positions"}:
        predicted_position = predicted_target
    else:
        raise ValueError(f"Unsupported rollout target mode: {cfg['target_mode']}")
    predicted = clone_graph(current_graph)
    predicted.x = predicted.x.to(device).clone()
    predicted.x[:, : int(cfg["pos_dim"])] = predicted_position
    return predicted


def evaluate_direct_autoencoder_rollout(
    result: dict,
    *,
    rollout_steps,
    device,
    max_sims: int | None = None,
    max_sims_per_source: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Autoregressively roll the direct autoencoder and summarize held-out errors."""

    horizons = sorted({int(step) for step in rollout_steps if int(step) > 0})
    sims = result["test_data"]
    if max_sims_per_source is not None:
        selected = []
        source_counts: dict[str, int] = {}
        for sim in sims:
            source = str(getattr(sim[0], "source_name", result["dataset_spec"]["name"]))
            count = source_counts.get(source, 0)
            if count >= int(max_sims_per_source):
                continue
            source_counts[source] = count + 1
            selected.append(sim)
        sims = selected
    elif max_sims is not None:
        sims = sims[: int(max_sims)]
    rows = []
    result["model"].eval()
    with torch.no_grad():
        for sim_idx, sim in enumerate(sims):
            source = str(
                getattr(sim[0], "source_name", result["dataset_spec"]["name"])
            )
            source_labels = result["dataset_spec"].get("source_labels", {})
            source_label = str(
                source_labels.get(source, result["dataset_spec"].get("label", source))
            )
            available = len(sim) - 1
            active_horizons = [step for step in horizons if step <= available]
            if not active_horizons:
                continue
            reference = clone_graph(sim[0]).to(device)
            current = clone_graph(sim[0]).to(device)
            previous = clone_graph(sim[0]).to(device)
            for step in range(1, max(active_horizons) + 1):
                predicted = predict_next_graph(
                    result,
                    reference,
                    current,
                    device=device,
                    previous_graph=previous,
                )
                previous, current = current, predicted
                if step not in active_horizons:
                    continue
                target = sim[step]
                pred_position = current.x[:, : int(result["config"]["pos_dim"])].cpu()
                target_position = target.x[:, : int(result["config"]["pos_dim"])].cpu()
                rows.append(
                    {
                        "dataset": source,
                        "dataset_label": source_label,
                        "sim_idx": sim_idx,
                        "rollout_steps": step,
                        "true_p_ratio": float(calc_p_ratio_rollout_sides(sim, step)),
                        "pred_p_ratio": float(
                            calc_p_ratio_rollout_sides(
                                [
                                    clone_graph(sim[0]).cpu(),
                                    clone_graph(current).cpu(),
                                ],
                                -1,
                            )
                        ),
                        "position_mse": float(
                            F.mse_loss(pred_position, target_position).item()
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    summaries = []
    for (source, source_label, step), group in frame.groupby(
        ["dataset", "dataset_label", "rollout_steps"]
    ):
        valid = group.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["true_p_ratio", "pred_p_ratio"]
        )
        summaries.append(
            {
                "dataset": source,
                "dataset_label": source_label,
                "rollout_steps": int(step),
                "used": len(valid),
                "p_ratio_r2": r2_score(valid["true_p_ratio"], valid["pred_p_ratio"]),
                "p_ratio_pearson": pearson_r(
                    valid["true_p_ratio"], valid["pred_p_ratio"]
                ),
                "position_mse": float(group["position_mse"].mean()),
            }
        )
    return frame, pd.DataFrame(summaries)


__all__ = [
    "evaluate_direct_autoencoder_rollout",
    "predict_next_graph",
    "run_direct_autoencoder_case",
]
