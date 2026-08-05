"""Internal search for Notebook 09's full-attention simulator."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lss.data import load_dataset
from lss.graph import box_tensor, clone_graph
from lss.latent.experiment import ground_truth_p_ratio, seed_everything
from lss.latent.simulation import (
    complete_graph_edge_data,
    pearson_r,
    r2_score,
    stored_graph_edge_data,
    undirected_complete_graph_edge_data,
    undirected_stored_graph_edge_data,
)
from lss.models.complete_graph_attention_simulator import (
    CompleteGraphAttentionSimulator,
)
from lss.models.complete_graph_transformer_simulator import (
    CompleteGraphTransformerSimulator,
)
from lss.models.one_shot_edge_attention_simulator import (
    OneShotUndirectedEdgeAttentionSimulator,
)
from lss.models.simple_edge_mlp_simulator import (
    SimpleUndirectedEdgeMLPSimulator,
)
from lss.models.attention_pyramid_simulator import AttentionPyramidSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-key",
        choices=("depablo_low_temp", "reid", "lj_noisy", "lj_noisy_50"),
        default="depablo_low_temp",
    )
    parser.add_argument("--train-networks", type=int, default=20)
    parser.add_argument("--val-networks", type=int, default=20)
    parser.add_argument("--transitions", type=int, default=100)
    parser.add_argument(
        "--target-mode",
        choices=("increment", "next_state"),
        default="increment",
        help=(
            "Predict a frame increment, or the next displacement from the "
            "static reference configuration."
        ),
    )
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=1,
        help="Closed-loop training steps. One preserves ordinary teacher forcing.",
    )
    parser.add_argument(
        "--unroll-batches",
        type=int,
        default=0,
        help="Training chunks per epoch for multi-step training; 0 uses --train-batches.",
    )
    parser.add_argument(
        "--one-step-warmup",
        type=int,
        default=0,
        help="Teacher-forced epochs before closed-loop multi-step training.",
    )
    parser.add_argument(
        "--startup-fraction",
        type=float,
        default=0.0,
        help="Fraction of training samples that start at the static frame.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-batches", type=int, default=400)
    parser.add_argument("--val-batches", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--edge-mode",
        choices=("complete", "stored"),
        default="complete",
    )
    parser.add_argument(
        "--model-kind",
        choices=(
            "transformer",
            "edge_attention",
            "one_shot",
            "simple_mlp",
            "pyramid",
        ),
        default="transformer",
    )
    parser.add_argument(
        "--pyramid-tokens",
        type=str,
        default="32,16",
        help="Strictly decreasing token counts for the attention pyramid.",
    )
    parser.add_argument("--bottleneck-layers", type=int, default=2)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=0,
        help="Exact scalar bottleneck; 0 keeps the token bottleneck.",
    )
    parser.add_argument(
        "--message-layers",
        type=int,
        default=4,
        help="Sparse edge-attention layers; ignored by the Transformer model.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--rollout", type=int, default=150)
    parser.add_argument("--test-networks", type=int, default=20)
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--distance-floor", type=float, default=-6.0)
    parser.add_argument(
        "--edge-aggregation",
        choices=("softmax", "gated_sum"),
        default="softmax",
    )
    parser.add_argument("--velocity-skip", action="store_true")
    parser.add_argument("--dual-kinematic", action="store_true")
    parser.add_argument("--boundary-features", action="store_true")
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--node-count-feature", action="store_true")
    parser.add_argument(
        "--no-progress-feature",
        action="store_true",
        help="Omit explicit time progress and infer updates only from the current state.",
    )
    parser.add_argument(
        "--global-context",
        action="store_true",
        help="Broadcast a learned mean-pooled graph state to every node decoder.",
    )
    parser.add_argument(
        "--static-structure-only",
        action="store_true",
        help="Use frame-0 topology, edge properties, and box at every model call.",
    )
    parser.add_argument(
        "--ignore-box",
        action="store_true",
        help="Do not expose box dimensions to geometric edge construction.",
    )
    parser.add_argument("--select-rollout-networks", type=int, default=0)
    parser.add_argument("--select-rollout-horizon", type=int, default=49)
    parser.add_argument(
        "--selection-every",
        type=int,
        default=1,
        help="Measure the expensive validation rollout every N epochs.",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument(
        "--init-state",
        type=Path,
        help="Warm-start training from a compatible saved state.",
    )
    parser.add_argument(
        "--init-local-state",
        type=Path,
        help="For a pyramid, initialize its high-resolution path from one-shot state.",
    )
    parser.add_argument("--load-state", type=Path)
    return parser.parse_args()


def reference_geometry(sim, device: str | torch.device):
    ref = sim[0].x[:, :2].to(device).float()
    low, high = ref.min(0).values, ref.max(0).values
    scale = (high - low).clamp_min(1e-6)
    center = 0.5 * (low + high)
    normalized = 2 * (ref - center) / scale
    return ref, normalized, scale, center


def transition_rows(sims, transitions: int):
    return [
        (sim_index, t)
        for sim_index, sim in enumerate(sims)
        for t in range(min(int(transitions), len(sim) - 1))
    ]


def target_stats(sims, rows, target_mode: str):
    chunks = []
    for sim_index, t in rows:
        sim = sims[sim_index]
        ref, _, scale, _ = reference_geometry(sim, "cpu")
        next_pos = sim[t + 1].x[:, :2].float()
        if target_mode == "increment":
            value = (next_pos - sim[t].x[:, :2].float()) / scale
        elif target_mode == "next_state":
            value = (next_pos - ref) / scale
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
        chunks.append(value)
    values = torch.cat(chunks)
    return values.mean(0, keepdim=True), values.std(0, keepdim=True).clamp_min(1e-7)


def acceleration_stats(sims, rows):
    chunks = []
    for sim_index, t in rows:
        if t <= 0:
            continue
        sim = sims[sim_index]
        _, _, scale, _ = reference_geometry(sim, "cpu")
        current_velocity = sim[t].x[:, :2].float() - sim[t - 1].x[:, :2].float()
        next_velocity = sim[t + 1].x[:, :2].float() - sim[t].x[:, :2].float()
        chunks.append((next_velocity - current_velocity) / scale)
    if not chunks:
        raise ValueError("Acceleration statistics require transitions with t > 0")
    values = torch.cat(chunks)
    return values.mean(0, keepdim=True), values.std(0, keepdim=True).clamp_min(1e-9)


def edge_stats(
    sims,
    rows,
    edge_mode: str,
    *,
    undirected_edges: bool,
    sample_count: int = 80,
):
    chosen = np.linspace(0, len(rows) - 1, min(sample_count, len(rows)), dtype=int)
    sums = torch.zeros(13)
    sums2 = torch.zeros(13)
    count = 0
    for index in chosen:
        sim_index, t = rows[int(index)]
        sim = sims[sim_index]
        if undirected_edges:
            edge_builder = (
                undirected_complete_graph_edge_data
                if edge_mode == "complete"
                else undirected_stored_graph_edge_data
            )
        else:
            edge_builder = (
                complete_graph_edge_data
                if edge_mode == "complete"
                else stored_graph_edge_data
            )
        _, edge, _ = edge_builder(sim[0], sim[t], pos_dim=2, device="cpu")
        sums += edge.sum(0)
        sums2 += edge.square().sum(0)
        count += edge.size(0)
    mean = sums / max(count, 1)
    var = (sums2 / max(count, 1) - mean.square()).clamp_min(1e-12)
    return mean, var.sqrt()


def make_inputs(
    sim,
    t: int,
    current_graph,
    previous_graph,
    *,
    target_mean,
    target_std,
    accel_mean,
    accel_std,
    edge_mean,
    edge_std,
    length_scale: float,
    distance_floor: float,
    velocity_skip: bool,
    dual_kinematic: bool,
    boundary_features: bool,
    boundary_weight: float,
    node_count_feature: bool,
    target_mode: str,
    edge_mode: str,
    undirected_edges: bool,
    device: str,
    with_target: bool,
    include_progress: bool = True,
    include_velocity_feature: bool = False,
    warm_start_frames: int = 0,
    static_structure_only: bool = False,
    ignore_box: bool = False,
):
    ref, ref_normalized, scale, center = reference_geometry(sim, device)
    current_pos = current_graph.x[:, :2].to(device).float()
    previous_pos = previous_graph.x[:, :2].to(device).float()
    current_normalized = 2 * (current_pos - center) / scale
    current_delta = (current_pos - ref) / scale
    progress = torch.full(
        (ref.size(0), 1), (t + 1) / (len(sim) - 1), device=device
    )
    previous_displacement = (current_pos - previous_pos) / scale
    node_parts = [current_normalized, current_delta, ref_normalized]
    if include_progress:
        node_parts.append(progress)
    if include_velocity_feature or velocity_skip or dual_kinematic:
        node_parts.append(previous_displacement)
    side_flags = torch.zeros((ref.size(0), 4), device=device, dtype=ref.dtype)
    side_count = max(1, int(np.ceil(0.10 * ref.size(0))))
    side_flags[torch.topk(ref_normalized[:, 0], side_count, largest=False).indices, 0] = 1
    side_flags[torch.topk(ref_normalized[:, 0], side_count, largest=True).indices, 1] = 1
    side_flags[torch.topk(ref_normalized[:, 1], side_count, largest=False).indices, 2] = 1
    side_flags[torch.topk(ref_normalized[:, 1], side_count, largest=True).indices, 3] = 1
    if boundary_features:
        node_parts.append(side_flags)
    if node_count_feature:
        node_parts.append(
            torch.full(
                (ref.size(0), 1),
                float(ref.size(0)) / 200.0,
                device=device,
                dtype=ref.dtype,
            )
        )
    node = torch.cat(node_parts, dim=1)
    if undirected_edges:
        edge_builder = (
            undirected_complete_graph_edge_data
            if edge_mode == "complete"
            else undirected_stored_graph_edge_data
        )
    else:
        edge_builder = (
            complete_graph_edge_data
            if edge_mode == "complete"
            else stored_graph_edge_data
        )
    reference_edge_graph = sim[0]
    edge_graph = current_graph
    if static_structure_only:
        # Preserve predicted/current positions while making all structural
        # metadata exactly the metadata available in the single input frame.
        edge_graph = clone_graph(current_graph)
        edge_graph.box = sim[0].box
        static_box = box_tensor(
            sim[0], device=current_pos.device, dtype=current_pos.dtype
        )
        if static_box is not None:
            edge_graph.box_tensor = static_box
        edge_graph.edge_index = sim[0].edge_index
        edge_graph.edge_attr = sim[0].edge_attr
    if ignore_box:
        reference_edge_graph = clone_graph(sim[0])
        edge_graph = clone_graph(edge_graph)
        reference_edge_graph.box = None
        edge_graph.box = None
        if "box_tensor" in reference_edge_graph:
            del reference_edge_graph.box_tensor
        if "box_tensor" in edge_graph:
            del edge_graph.box_tensor
    edge_index, raw_edge, _ = edge_builder(
        reference_edge_graph, edge_graph, pos_dim=2, device=device
    )
    current_distance = raw_edge[:, 5]
    prior = -0.5 * (current_distance / length_scale).square()
    if distance_floor is not None:
        prior = prior.clamp_min(float(distance_floor))
    edge = (raw_edge - edge_mean) / edge_std
    target = None
    if with_target:
        next_pos = sim[t + 1].x[:, :2].to(device).float()
        if target_mode == "increment":
            target_value = (next_pos - current_pos) / scale
        elif target_mode == "next_state":
            target_value = (next_pos - ref) / scale
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
        target = (target_value - target_mean) / target_std
    boundary_mask = side_flags.amax(dim=1)
    return (
        node,
        edge,
        edge_index,
        prior,
        target,
        scale,
        previous_displacement,
        boundary_mask,
    )


def epoch_pass(
    model,
    sims,
    rows,
    *,
    optimizer,
    max_batches,
    seed,
    common,
):
    training = optimizer is not None
    model.train(training)
    indices = np.arange(len(rows))
    if training:
        np.random.default_rng(seed).shuffle(indices)
    if max_batches is not None and len(indices) > max_batches:
        indices = (
            indices[:max_batches]
            if training
            else np.linspace(0, len(indices) - 1, max_batches, dtype=int)
        )
    losses = []
    predictions = []
    targets = []
    for index in indices:
        sim_index, t = rows[int(index)]
        sim = sims[sim_index]
        previous_graph = sim[t - 1] if t > 0 else sim[t]
        (
            node,
            edge,
            edge_index,
            prior,
            target,
            scale,
            previous_displacement,
            boundary_mask,
        ) = make_inputs(
            sim, t, sim[t], previous_graph, with_target=True, **common
        )
        prediction = model(node, edge, edge_index, attention_bias=prior)
        if common["dual_kinematic"]:
            direct_prediction = prediction[:, :2]
            direct_node_loss = (direct_prediction - target).square().mean(dim=1)
            weights = 1.0 + (common["boundary_weight"] - 1.0) * boundary_mask
            direct_loss = (direct_node_loss * weights).sum() / weights.sum()
            if t > 0:
                acceleration = (
                    target * common["target_std"]
                    + common["target_mean"]
                    - previous_displacement
                )
                acceleration_target = (
                    acceleration - common["accel_mean"]
                ) / common["accel_std"]
                acceleration_loss = F.mse_loss(
                    prediction[:, 2:], acceleration_target
                )
                loss = acceleration_loss + 0.25 * direct_loss
            else:
                loss = direct_loss
            prediction = direct_prediction
        elif common["velocity_skip"]:
            prediction = prediction + (
                previous_displacement - common["target_mean"]
            ) / common["target_std"]
            node_loss = (prediction - target).square().mean(dim=1)
            weights = 1.0 + (common["boundary_weight"] - 1.0) * boundary_mask
            loss = (node_loss * weights).sum() / weights.sum()
        else:
            node_loss = (prediction - target).square().mean(dim=1)
            weights = 1.0 + (common["boundary_weight"] - 1.0) * boundary_mask
            loss = (node_loss * weights).sum() / weights.sum()
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach()))
        predictions.append(
            (prediction * common["target_std"] + common["target_mean"])
            .detach()
            .cpu()
            .reshape(-1)
        )
        targets.append(
            (target * common["target_std"] + common["target_mean"])
            .detach()
            .cpu()
            .reshape(-1)
        )
    pred = torch.cat(predictions).numpy()
    true = torch.cat(targets).numpy()
    return float(np.mean(losses)), r2_score(true, pred)


def predicted_position(prediction, current_graph, scale, ref, common):
    """Decode a standardized model output into the next physical position."""
    value = prediction * common["target_std"] + common["target_mean"]
    if common["target_mode"] == "increment":
        return current_graph.x[:, :2].to(common["device"]).float() + value * scale
    if common["target_mode"] == "next_state":
        return ref + value * scale
    raise ValueError(f"Unknown target mode: {common['target_mode']}")


def make_predicted_graph(template_graph, position):
    """Keep scheduled box/topology metadata while replacing only positions."""
    graph = clone_graph(template_graph).to(position.device)
    graph.x = graph.x.clone().float()
    graph.x[:, :2] = position
    return graph


def multi_step_epoch(
    model,
    sims,
    *,
    optimizer,
    transitions,
    unroll_steps,
    max_batches,
    seed,
    common,
    startup_fraction=0.0,
):
    """Train through short, fully autoregressive position rollouts."""
    model.train(True)
    chunks = [
        (sim_index, start)
        for sim_index, sim in enumerate(sims)
        for start in range(
            max(0, min(int(transitions), len(sim) - 1) - int(unroll_steps) + 1)
        )
    ]
    if startup_fraction > 0:
        if not 0 <= startup_fraction < 1:
            raise ValueError("startup_fraction must be in [0, 1)")
        startup = [row for row in chunks if row[1] == 0]
        ordinary = [row for row in chunks if row[1] != 0]
        repeats = int(
            np.ceil(
                startup_fraction
                * len(ordinary)
                / max((1.0 - startup_fraction) * len(startup), 1.0)
            )
        )
        chunks = ordinary + startup * max(1, repeats)
    rng = np.random.default_rng(seed)
    rng.shuffle(chunks)
    if max_batches is not None and len(chunks) > int(max_batches):
        chunks = chunks[: int(max_batches)]
    losses = []
    for sim_index, start in chunks:
        sim = sims[sim_index]
        current = clone_graph(sim[start]).to(common["device"])
        previous = (
            clone_graph(sim[start - 1]).to(common["device"])
            if start > 0
            else current
        )
        step_losses = []
        for offset in range(int(unroll_steps)):
            t = start + offset
            (
                node,
                edge,
                edge_index,
                prior,
                target,
                scale,
                previous_displacement,
                boundary_mask,
            ) = make_inputs(
                sim, t, current, previous, with_target=True, **common
            )
            prediction = model(node, edge, edge_index, attention_bias=prior)
            if common["dual_kinematic"]:
                raise ValueError("Multi-step mode does not support dual_kinematic")
            if common["velocity_skip"]:
                prediction = prediction + (
                    previous_displacement - common["target_mean"]
                ) / common["target_std"]
            node_loss = (prediction - target).square().mean(dim=1)
            weights = 1.0 + (common["boundary_weight"] - 1.0) * boundary_mask
            step_loss = (node_loss * weights).sum() / weights.sum()
            ref, _, _, _ = reference_geometry(sim, common["device"])
            position = predicted_position(prediction, current, scale, ref, common)
            # Later free-running states matter slightly more than the first
            # teacher-initialized step.
            step_losses.append(step_loss * (1.0 + offset / max(1, unroll_steps)))
            previous = current
            template = (
                sim[0] if common.get("static_structure_only", False) else sim[t + 1]
            )
            current = make_predicted_graph(template, position)
        loss = torch.stack(step_losses).sum() / sum(
            1.0 + offset / max(1, unroll_steps)
            for offset in range(int(unroll_steps))
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def rollout(model, sim, horizon: int, common):
    warm_start_frames = min(
        int(common.get("warm_start_frames", 0)), int(horizon), len(sim) - 1
    )
    predicted = [clone_graph(graph).cpu() for graph in sim[: warm_start_frames + 1]]
    current = predicted[-1]
    with torch.no_grad():
        for next_step in range(warm_start_frames + 1, horizon + 1):
            previous = predicted[-2] if len(predicted) > 1 else current
            (
                node,
                edge,
                edge_index,
                prior,
                _,
                scale,
                previous_displacement,
                _,
            ) = make_inputs(
                sim,
                next_step - 1,
                current,
                previous,
                with_target=False,
                **common,
            )
            pred_standard = model(node, edge, edge_index, attention_bias=prior)
            if common["dual_kinematic"]:
                if next_step == 1:
                    displacement = (
                        pred_standard[:, :2] * common["target_std"]
                        + common["target_mean"]
                    )
                else:
                    acceleration = (
                        pred_standard[:, 2:] * common["accel_std"]
                        + common["accel_mean"]
                    )
                    displacement = previous_displacement + acceleration
            elif common["velocity_skip"]:
                pred_standard = pred_standard + (
                    previous_displacement - common["target_mean"]
                ) / common["target_std"]
                displacement = (
                    pred_standard * common["target_std"] + common["target_mean"]
                )
            else:
                displacement = pred_standard * common["target_std"] + common["target_mean"]
            if common["dual_kinematic"] or common["velocity_skip"]:
                position = (
                    current.x[:, :2].to(common["device"]).float()
                    + displacement * scale
                )
            else:
                ref, _, _, _ = reference_geometry(sim, common["device"])
                position = predicted_position(
                    pred_standard, current, scale, ref, common
                )
            template = (
                sim[0]
                if common.get("static_structure_only", False)
                else sim[next_step]
            )
            next_graph = make_predicted_graph(template, position).cpu()
            predicted.append(next_graph)
            current = next_graph
    return predicted


def main():
    args = parse_args()
    if args.velocity_skip and args.dual_kinematic:
        raise ValueError("--velocity-skip and --dual-kinematic are mutually exclusive")
    if args.target_mode == "next_state" and (args.velocity_skip or args.dual_kinematic):
        raise ValueError("next_state mode is incompatible with kinematic residual heads")
    if args.unroll_steps < 1:
        raise ValueError("--unroll-steps must be positive")
    torch.set_num_threads(max(1, args.threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cpu"
    seed_everything(args.seed)
    dataset_paths = {
        "depablo_low_temp": ROOT / "data" / "depablo-near-zero-temp.pt",
        "reid": ROOT / "data" / "reid_200_frames.pt",
        "lj_noisy": ROOT
        / "data"
        / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt",
        "lj_noisy_50": ROOT
        / "data"
        / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_1348sims_50frames.pt",
    }
    sims = load_dataset(
        dataset_paths[args.dataset_key],
        edge_multiplicity=1,
    )
    order = np.random.default_rng(args.seed).permutation(len(sims))
    train_ids = order[: args.train_networks]
    val_ids = order[args.train_networks : args.train_networks + args.val_networks]
    test_ids = order[args.train_networks + args.val_networks :]
    train_sims = [sims[i] for i in train_ids]
    val_sims = [sims[i] for i in val_ids]
    test_sims = [sims[i] for i in test_ids]
    train_rows = transition_rows(train_sims, args.transitions)
    stats_train_rows = train_rows
    val_rows = transition_rows(val_sims, args.transitions)
    if args.startup_fraction > 0:
        if not 0 <= args.startup_fraction < 1:
            raise ValueError("--startup-fraction must be in [0, 1)")
        startup = [row for row in train_rows if row[1] == 0]
        ordinary = [row for row in train_rows if row[1] != 0]
        repeats = int(
            np.ceil(
                args.startup_fraction
                * len(ordinary)
                / max((1.0 - args.startup_fraction) * len(startup), 1.0)
            )
        )
        train_rows = ordinary + startup * max(1, repeats)

    target_mean, target_std = target_stats(
        train_sims, stats_train_rows, args.target_mode
    )
    accel_mean, accel_std = acceleration_stats(train_sims, stats_train_rows)
    undirected_edges = args.model_kind in {"one_shot", "simple_mlp", "pyramid"}
    edge_mean, edge_std = edge_stats(
        train_sims,
        stats_train_rows,
        args.edge_mode,
        undirected_edges=undirected_edges,
    )
    bond_lengths = torch.cat(
        [torch.linalg.vector_norm(sim[0].edge_attr[:, :2].float(), dim=1) for sim in train_sims]
    )
    length_scale = float(bond_lengths.median().clamp_min(1e-6))
    common = {
        "target_mean": target_mean.to(device),
        "target_std": target_std.to(device),
        "accel_mean": accel_mean.to(device),
        "accel_std": accel_std.to(device),
        "edge_mean": edge_mean.to(device),
        "edge_std": edge_std.to(device),
        "length_scale": length_scale,
        "distance_floor": args.distance_floor,
        "velocity_skip": bool(args.velocity_skip),
        "dual_kinematic": bool(args.dual_kinematic),
        "boundary_features": bool(args.boundary_features),
        "boundary_weight": float(args.boundary_weight),
        "node_count_feature": bool(args.node_count_feature),
        "include_progress": not bool(args.no_progress_feature),
        "static_structure_only": bool(args.static_structure_only),
        "ignore_box": bool(args.ignore_box),
        "target_mode": str(args.target_mode),
        "edge_mode": str(args.edge_mode),
        "undirected_edges": bool(undirected_edges),
        "device": device,
    }
    node_dim = (
        (7 if common["include_progress"] else 6)
        + (2 if (args.velocity_skip or args.dual_kinematic) else 0)
        + (4 if args.boundary_features else 0)
        + (1 if args.node_count_feature else 0)
    )
    if args.model_kind == "transformer":
        model = CompleteGraphTransformerSimulator(
            node_dim=node_dim,
            edge_dim=13,
            hidden_size=args.hidden,
            transformer_layers=args.layers,
            transformer_heads=args.heads,
            edge_aggregation=args.edge_aggregation,
            output_dim=4 if args.dual_kinematic else 2,
        ).to(device)
    elif args.model_kind == "edge_attention":
        model = CompleteGraphAttentionSimulator(
            node_dim=node_dim,
            edge_dim=13,
            hidden_size=args.hidden,
            layers=args.message_layers,
            output_dim=4 if args.dual_kinematic else 2,
        ).to(device)
    elif args.model_kind == "one_shot":
        model = OneShotUndirectedEdgeAttentionSimulator(
            node_dim=node_dim,
            edge_dim=13,
            hidden_size=args.hidden,
            output_dim=4 if args.dual_kinematic else 2,
            global_context=args.global_context,
            edge_aggregation=args.edge_aggregation,
        ).to(device)
    elif args.model_kind == "simple_mlp":
        model = SimpleUndirectedEdgeMLPSimulator(
            node_dim=node_dim,
            edge_dim=13,
            hidden_size=args.hidden,
            output_dim=4 if args.dual_kinematic else 2,
        ).to(device)
    else:
        pyramid_tokens = tuple(
            int(value.strip())
            for value in args.pyramid_tokens.split(",")
            if value.strip()
        )
        model = AttentionPyramidSimulator(
            node_dim=node_dim,
            edge_dim=13,
            hidden_size=args.hidden,
            pyramid_tokens=pyramid_tokens,
            heads=args.heads,
            bottleneck_layers=args.bottleneck_layers,
            latent_dim=args.latent_dim,
            output_dim=4 if args.dual_kinematic else 2,
        ).to(device)
    best_state = None
    best_val = float("inf")
    best_rollout_r2 = float("-inf")
    history = []
    pr_cfg = {
        "temperature_pratio_estimator": "robust",
        "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5,
    }
    metric_dataset_name = (
        "lj_noisy" if args.dataset_key.startswith("lj_noisy") else args.dataset_key
    )
    state_path = args.load_state or args.init_state
    if state_path is not None:
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        saved_common = state.get("common", state)
        common.update(
            {
                "target_mean": saved_common.get("target_mean", target_mean).to(device),
                "target_std": saved_common.get("target_std", target_std).to(device),
                "accel_mean": saved_common.get("accel_mean", accel_mean).to(device),
                "accel_std": saved_common.get("accel_std", accel_std).to(device),
                "edge_mean": saved_common.get("edge_mean", edge_mean).to(device),
                "edge_std": saved_common.get("edge_std", edge_std).to(device),
                "length_scale": float(saved_common.get("length_scale", length_scale)),
            }
        )
        if args.load_state is not None:
            best_state = deepcopy(model.state_dict())
            best_val = float("nan")
    if args.init_local_state is not None:
        if args.model_kind != "pyramid":
            raise ValueError("--init-local-state is only valid for --model-kind pyramid")
        local_state = torch.load(
            args.init_local_state, map_location=device, weights_only=False
        )
        source = local_state["model_state"]
        translated = {}
        for key, value in source.items():
            target_key = (
                "local_decoder" + key[len("decoder") :]
                if key.startswith("decoder.")
                else key
            )
            if target_key in model.state_dict() and model.state_dict()[target_key].shape == value.shape:
                translated[target_key] = value
        missing, unexpected = model.load_state_dict(translated, strict=False)
        print(
            f"initialized pyramid local path from {args.init_local_state} "
            f"({len(translated)} tensors; {len(missing)} pyramid tensors new)",
            flush=True,
        )
        common.update(
            {
                "target_mean": local_state["target_mean"].to(device),
                "target_std": local_state["target_std"].to(device),
                "accel_mean": local_state.get("accel_mean", accel_mean).to(device),
                "accel_std": local_state.get("accel_std", accel_std).to(device),
                "edge_mean": local_state["edge_mean"].to(device),
                "edge_std": local_state["edge_std"].to(device),
                "length_scale": float(local_state["length_scale"]),
            }
        )
    if args.load_state is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        stale = 0
        for epoch in range(1, args.epochs + 1):
            use_unroll = (
                args.unroll_steps > 1 and epoch > int(args.one_step_warmup)
            )
            if use_unroll:
                train_loss = multi_step_epoch(
                    model,
                    train_sims,
                    optimizer=optimizer,
                    transitions=args.transitions,
                    unroll_steps=args.unroll_steps,
                    max_batches=(args.unroll_batches or args.train_batches),
                    seed=args.seed + epoch,
                    common=common,
                    startup_fraction=args.startup_fraction,
                )
                with torch.no_grad():
                    _, train_r2 = epoch_pass(
                        model,
                        train_sims,
                        train_rows,
                        optimizer=None,
                        max_batches=args.val_batches,
                        seed=args.seed,
                        common=common,
                    )
            else:
                train_loss, train_r2 = epoch_pass(
                    model,
                    train_sims,
                    train_rows,
                    optimizer=optimizer,
                    max_batches=args.train_batches,
                    seed=args.seed + epoch,
                    common=common,
                )
            with torch.no_grad():
                val_loss, val_r2 = epoch_pass(
                    model,
                    val_sims,
                    val_rows,
                    optimizer=None,
                    max_batches=args.val_batches,
                    seed=args.seed,
                    common=common,
                )
                val_rollout_r2 = float("nan")
                selection_epoch = (
                    (
                        epoch % max(1, int(args.selection_every)) == 0
                        or epoch == args.epochs
                    )
                    and (
                        args.unroll_steps <= 1
                        or epoch > int(args.one_step_warmup)
                    )
                )
                if args.select_rollout_networks > 0 and selection_epoch:
                    true_ratios = []
                    predicted_ratios = []
                    selection_horizon = min(
                        int(args.select_rollout_horizon), len(val_sims[0]) - 1
                    )
                    for sim in val_sims[: int(args.select_rollout_networks)]:
                        predicted_path = rollout(
                            model, sim, selection_horizon, common
                        )
                        true_ratios.append(
                            ground_truth_p_ratio(
                                sim[: selection_horizon + 1],
                                dataset_name=metric_dataset_name,
                                cfg=pr_cfg,
                            )
                        )
                        predicted_ratios.append(
                            ground_truth_p_ratio(
                                predicted_path,
                                dataset_name=metric_dataset_name,
                                cfg=pr_cfg,
                            )
                        )
                    val_rollout_r2 = r2_score(true_ratios, predicted_ratios)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_displacement_r2": train_r2,
                    "val_displacement_r2": val_r2,
                    "val_rollout_pratio_r2": val_rollout_r2,
                }
            )
            print(
                f"epoch={epoch:03d} train={train_loss:.5g} val={val_loss:.5g} "
                f"train_r2={train_r2:.4f} val_r2={val_r2:.4f} "
                f"rollout_r2={val_rollout_r2:.4f}",
                flush=True,
            )
            rollout_selection = args.select_rollout_networks > 0
            improved = (
                np.isfinite(val_rollout_r2)
                and val_rollout_r2 > best_rollout_r2 + 1e-4
                if rollout_selection
                else val_loss < best_val - 1e-5
            )
            if best_state is None:
                improved = True
            if improved:
                best_val = val_loss
                if np.isfinite(val_rollout_r2):
                    best_rollout_r2 = val_rollout_r2
                best_state = deepcopy(model.state_dict())
                stale = 0
            elif not rollout_selection or np.isfinite(val_rollout_r2):
                stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No valid model")
    model.load_state_dict(best_state)
    model.eval()
    if args.state_output is not None:
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "target_mean": target_mean,
                "target_std": target_std,
                "accel_mean": accel_mean,
                "accel_std": accel_std,
                "edge_mean": edge_mean,
                "edge_std": edge_std,
                "length_scale": length_scale,
            },
            args.state_output,
        )

    horizons = sorted(
        {h for h in [25, 49, 50, 75, 100, 125, 150, args.rollout] if h <= args.rollout}
    )
    prediction_rows = []
    evaluation_pool = val_sims if args.eval_split == "val" else test_sims
    selected_test = evaluation_pool[: args.test_networks]
    for test_index, sim in enumerate(selected_test):
        path = rollout(model, sim, args.rollout, common)
        for horizon in horizons:
            prediction_rows.append(
                {
                    "test_index": test_index,
                    "rollout_step": horizon,
                    "true_p_ratio": ground_truth_p_ratio(
                        sim[: horizon + 1], dataset_name=metric_dataset_name, cfg=pr_cfg
                    ),
                    "pred_p_ratio": ground_truth_p_ratio(
                        path[: horizon + 1], dataset_name=metric_dataset_name, cfg=pr_cfg
                    ),
                }
            )
        print(f"rollout={test_index + 1}/{len(selected_test)}", flush=True)
    predictions = pd.DataFrame(
        prediction_rows,
        columns=("test_index", "rollout_step", "true_p_ratio", "pred_p_ratio"),
    ).dropna()
    metrics = []
    for horizon, group in predictions.groupby("rollout_step"):
        metrics.append(
            {
                "rollout_step": int(horizon),
                "n": len(group),
                "p_ratio_r2": r2_score(group.true_p_ratio, group.pred_p_ratio),
                "p_ratio_pearson": pearson_r(group.true_p_ratio, group.pred_p_ratio),
            }
        )
    payload = {
        "args": vars(args)
        | {
            "output": str(args.output),
            "state_output": str(args.state_output) if args.state_output else None,
            "init_state": str(args.init_state) if args.init_state else None,
            "init_local_state": (
                str(args.init_local_state) if args.init_local_state else None
            ),
            "load_state": str(args.load_state) if args.load_state else None,
        },
        "best_val_loss": best_val,
        "best_val_rollout_pratio_r2": best_rollout_r2,
        "history": history,
        "prediction_rows": predictions.to_dict(orient="records"),
        "rollout_metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
