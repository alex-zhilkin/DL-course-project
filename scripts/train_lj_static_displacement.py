"""Static-only LJ rollout by predicting a node-wise response field."""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from lss.data import load_dataset

from lss.latent.experiment import ground_truth_p_ratio, temperature_p_ratio
from lss.latent.simulation import pearson_r, r2_score
from lss.models.transformer_simulator import TwoStageDownUpTransformer
from lss.utils import resolve_device
from train_lj_static_velocity import DATA, OUT, MessageLayer, graph_features


SEED = 20260716
FIELD_OUT = OUT.parent / "11_lj_static_displacement"


def static_physics_features(graph):
    """Initial WCA force and elastic fabric, both determined by the static graph."""

    pos = graph.x[:, :2].float()
    n = pos.size(0)
    delta = pos[None, :, :] - pos[:, None, :]
    box = getattr(graph, "box", None)
    if box is not None:
        size = torch.tensor([box.x2 - box.x1, box.y2 - box.y1], dtype=pos.dtype)
        delta = delta - torch.round(delta / size) * size
    distance2 = delta.square().sum(-1)
    sigma = float(getattr(graph, "lj_sigma", 1.0))
    epsilon = float(getattr(graph, "lj_epsilon", .01))
    cutoff = float(getattr(graph, "lj_cutoff", 2 ** (1 / 6)))
    mask = (distance2 > 1e-10) & (distance2 < cutoff ** 2)
    safe_r2 = distance2.clamp_min(1e-4)
    inv_r2 = sigma ** 2 / safe_r2
    inv_r6 = inv_r2 ** 3
    coefficient = 24 * epsilon * (2 * inv_r6.square() - inv_r6) / safe_r2
    coefficient = torch.where(mask, coefficient.clamp(max=1e12), torch.zeros_like(coefficient))
    force = -(coefficient[..., None] * delta).sum(1)
    force = torch.sign(force) * torch.log1p(force.abs())
    force_magnitude = torch.log1p(force.square().sum(1, keepdim=True).sqrt())

    source, target = graph.edge_index.long()
    edge_delta = pos[target] - pos[source]
    edge_length = edge_delta.norm(dim=1).clamp_min(1e-8)
    unit = edge_delta / edge_length[:, None]
    weight = graph.edge_attr[:, 3].float()
    fabric_edge = weight[:, None] * torch.stack(
        [unit[:, 0].square(), unit[:, 1].square(), unit[:, 0] * unit[:, 1]], dim=1
    )
    fabric = torch.zeros(n, 3, dtype=pos.dtype)
    count = torch.zeros(n, 1, dtype=pos.dtype)
    fabric.index_add_(0, target, fabric_edge)
    count.index_add_(0, target, torch.ones(len(target), 1))
    fabric = fabric / count.clamp_min(1)
    return torch.cat([force, force_magnitude, fabric], dim=1)


def add_displacement_targets(rows, sims):
    enriched = []
    for row, sim in zip(rows, sims):
        data = row.clone()
        data.x = torch.cat([data.x, static_physics_features(sim[0])], dim=1)
        reference = sim[0].x[:, :2].float()
        scale = (reference.max(0).values - reference.min(0).values).clamp_min(1e-6)
        data.displacement = ((sim[-1].x[:, :2].float() - reference) / scale).float()
        enriched.append(data)
    return enriched


class StaticDisplacementGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 96):
        super().__init__()
        self.node_in = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.edge_in = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(MessageLayer(hidden) for _ in range(6))
        self.out = nn.Sequential(
            nn.Linear(hidden + node_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
            nn.GELU(), nn.Linear(hidden, 2),
        )

    def forward(self, data):
        h, edge_h = self.node_in(data.x), self.edge_in(data.edge_attr)
        for layer in self.layers:
            h = layer(h, edge_h, data.edge_index)
        return self.out(torch.cat([h, data.x], dim=1))


def weighted_graph_mean(values, weights, batch, graphs):
    numerator = torch.zeros(graphs, dtype=values.dtype, device=values.device)
    denominator = torch.zeros_like(numerator)
    numerator.index_add_(0, batch, values * weights)
    denominator.index_add_(0, batch, weights)
    return numerator / denominator.clamp_min(1e-6)


def side_response(displacement, data):
    # graph_features stores left, right, bottom, top weights in columns 5:9.
    left, right, bottom, top = [data.x[:, k] for k in range(5, 9)]
    graphs = int(data.num_graphs)
    dx = weighted_graph_mean(displacement[:, 0], right, data.batch, graphs) - weighted_graph_mean(
        displacement[:, 0], left, data.batch, graphs
    )
    dy = weighted_graph_mean(displacement[:, 1], top, data.batch, graphs) - weighted_graph_mean(
        displacement[:, 1], bottom, data.batch, graphs
    )
    return dx, dy


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    losses = []
    for data in loader:
        data = data.to(device)
        pred = model(data)
        pred_dx, pred_dy = side_response(pred, data)
        true_dx, true_dy = side_response(data.displacement, data)
        node_loss = F.mse_loss(pred, data.displacement)
        side_loss = F.mse_loss(pred_dx, true_dx) + F.mse_loss(pred_dy, true_dy)
        # The loading is x-compression; this stable clamp prevents an early ratio singularity.
        pred_ratio = -pred_dy / pred_dx.clamp(max=-2e-3)
        ratio_loss = F.mse_loss(pred_ratio, data.p_ratio.reshape(-1))
        loss = node_loss + 4.0 * side_loss + 0.02 * ratio_loss
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def fit(rows, train_idx, val_idx, device, max_epochs, model_type="gnn"):
    if model_type == "transformer":
        model = TwoStageDownUpTransformer(
            in_node_dim=rows[0].x.size(1), in_edge_dim=rows[0].edge_attr.size(1),
            hidden_size=64, pos_dim=2, transformer_layers=3, transformer_heads=4,
            transformer_dropout=.05, num_mlp=2,
        ).to(device)
    else:
        model = StaticDisplacementGNN(rows[0].x.size(1), rows[0].edge_attr.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    train_loader = DataLoader([rows[i] for i in train_idx], batch_size=24, shuffle=True)
    val_loader = DataLoader([rows[i] for i in val_idx], batch_size=48)
    best, best_val, stale, history = None, float("inf"), 0, []
    for step in range(1, max_epochs + 1):
        train_loss = epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = epoch(model, val_loader, device)
        history.append({"epoch": step, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 2e-5:
            best_val, stale = val_loss, 0
            best = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if step == 1 or step % 20 == 0:
            print(f"field epoch {step:03d} train={train_loss:.6f} val={val_loss:.6f} stale={stale}", flush=True)
        if stale >= 25:
            break
    model.load_state_dict(best)
    return model.eval(), pd.DataFrame(history), best_val


def evaluate(model, rows, sims, indices, device, train_count):
    loader = DataLoader([rows[i] for i in indices], batch_size=48)
    final_fields = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            sizes = (data.ptr[1:] - data.ptr[:-1]).cpu().tolist()
            final_fields.extend(model(data).cpu().split(sizes))
    summary, details = [], []
    for horizon in (12, 25, 49):
        pred_mse, initial_mse, pred_p, true_p = [], [], [], []
        for local, sim_index in enumerate(indices):
            sim = sims[sim_index]
            reference = sim[0].x[:, :2].float()
            scale = (reference.max(0).values - reference.min(0).values).clamp_min(1e-6)
            predicted_path = []
            for frame in range(horizon + 1):
                graph = sim[frame].clone()
                graph.x = graph.x.clone().float()
                graph.x[:, :2] = reference + (frame / 49.0) * final_fields[local] * scale
                predicted_path.append(graph)
            target = sim[horizon].x[:, :2].float()
            predicted = predicted_path[-1].x[:, :2]
            pred_mse.append(float(F.mse_loss(predicted, target)))
            initial_mse.append(float(F.mse_loss(reference, target)))
            pred_p.append(temperature_p_ratio(predicted_path, cfg={
                "temperature_pratio_estimator": "robust", "temperature_pratio_min_fit_frames": 8,
                "temperature_pratio_min_driven_strain_range": 1e-3, "temperature_pratio_smooth_window": 5,
            }))
            true_p.append(ground_truth_p_ratio(sim, horizon, dataset_name="lj_noisy", cfg={
                "temperature_pratio_estimator": "robust", "temperature_pratio_min_fit_frames": 8,
                "temperature_pratio_min_driven_strain_range": 1e-3, "temperature_pratio_smooth_window": 5,
            }))
            details.append({"train_networks": train_count, "sim_index": sim_index, "horizon": horizon,
                "pred_p_ratio": pred_p[-1], "true_p_ratio": true_p[-1],
                "position_mse": pred_mse[-1], "initial_mse": initial_mse[-1]})
        pred_p, true_p = np.asarray(pred_p), np.asarray(true_p)
        finite = np.isfinite(pred_p) & np.isfinite(true_p)
        summary.append({"train_networks": train_count, "horizon": horizon, "test_networks": len(indices),
            "rollout_position_r2": max(0.0, 1 - np.mean(pred_mse) / np.mean(initial_mse)),
            "rollout_p_ratio_r2": r2_score(true_p[finite], pred_p[finite]),
            "rollout_p_ratio_pearson": pearson_r(true_p[finite], pred_p[finite]),
            "p_ratio_used": int(finite.sum())})
    return pd.DataFrame(summary), pd.DataFrame(details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-counts", type=int, nargs="+", default=[20, 200, 1000])
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--model", choices=["gnn", "transformer"], default="gnn")
    args = parser.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = resolve_device("auto")
    sims = load_dataset(DATA, edge_multiplicity=1)
    base_rows = torch.load(OUT / "static_velocity_targets.pt", map_location="cpu", weights_only=False)
    rows = add_displacement_targets(base_rows, sims)
    test_idx = list(range(len(rows) - 200, len(rows)))
    pool = np.arange(len(rows) - 200); np.random.default_rng(SEED).shuffle(pool)
    val_idx, train_pool = pool[:148].tolist(), pool[148:].tolist()
    FIELD_OUT.mkdir(parents=True, exist_ok=True)
    summaries, details = [], []
    for count in args.train_counts:
        print(f"\n=== static displacement, train={count} ===", flush=True)
        model, history, best_val = fit(
            rows, train_pool[:count], val_idx, device, args.epochs, args.model
        )
        result, detail = evaluate(model, rows, sims, test_idx, device, count)
        result["best_val_loss"] = best_val
        summaries.append(result); details.append(detail)
        history.to_csv(FIELD_OUT / f"history_{args.model}_train{count}.csv", index=False)
        torch.save({"state_dict": model.state_dict(), "train_indices": train_pool[:count],
            "val_indices": val_idx, "test_indices": test_idx},
            FIELD_OUT / f"static_displacement_{args.model}_train{count}.pt")
        result["model"] = args.model
        detail["model"] = args.model
        print(result.to_string(index=False), flush=True)
    pd.concat(summaries, ignore_index=True).to_csv(FIELD_OUT / f"summary_{args.model}.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(FIELD_OUT / f"test_predictions_{args.model}.csv", index=False)


if __name__ == "__main__":
    main()
