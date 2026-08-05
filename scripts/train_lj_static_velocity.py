"""Predict LJ latent response from an undeformed graph, with no trajectory warm-start."""

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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool

from lss.data import load_dataset
from lss.latent.capacity import load_experiment_bundle
from lss.latent.experiment import ground_truth_p_ratio
from lss.latent.simulation import pearson_r, r2_score
from lss.latent.training import decode_latent_positions, encode_frame_latent
from lss.utils import resolve_device


DATA = ROOT / "data" / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_1348sims_50frames.pt"
BASELINE = (
    ROOT / "notebooks" / "results" / "08_lj_train1_vs20" / "models"
    / "lj_noisy_train20_seed20260716.pt"
)
OUT = ROOT / "notebooks" / "results" / "10_lj_static_velocity"
SEED = 20260716


def summary(a: np.ndarray) -> list[float]:
    a = np.asarray(a, dtype=float)
    return [float(a.mean()), float(a.std()), *np.quantile(a, [0, .1, .25, .5, .75, .9, 1]).tolist()]


def graph_features(graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Boundary-aware node/edge features plus inexpensive global rigidity descriptors."""

    pos = graph.x[:, :2].float()
    center = pos.mean(0)
    scale = (pos.max(0).values - pos.min(0).values).clamp_min(1e-6)
    xy = 2.0 * (pos - center) / scale
    source, target = graph.edge_index.long()
    degree = torch.bincount(
        torch.cat([source, target]), minlength=pos.size(0)
    ).float()
    degree_norm = degree / degree.mean().clamp_min(1.0)
    boundary = torch.stack(
        [
            torch.exp(-4 * (xy[:, 0] + 1).abs()),
            torch.exp(-4 * (xy[:, 0] - 1).abs()),
            torch.exp(-4 * (xy[:, 1] + 1).abs()),
            torch.exp(-4 * (xy[:, 1] - 1).abs()),
        ], dim=1,
    )
    node_x = torch.cat(
        [xy, xy.square(), degree_norm[:, None], boundary], dim=1
    )

    delta = pos[target] - pos[source]
    delta_scaled = delta / scale
    length = delta.norm(dim=1).clamp_min(1e-8)
    unit = delta / length[:, None]
    raw_weight = graph.edge_attr[:, 3:4].float()
    edge_x = torch.cat(
        [delta_scaled, unit, (length / scale.prod().sqrt())[:, None], raw_weight], dim=1
    )

    angle = torch.atan2(delta[:, 1], delta[:, 0]).cpu().numpy()
    length_np = length.cpu().numpy()
    weight_np = raw_weight[:, 0].cpu().numpy()
    degree_np = degree.cpu().numpy()
    global_x = [
        pos.size(0) / 200.0,
        source.numel() / max(pos.size(0), 1),
        float(scale[0] / scale[1]),
    ]
    global_x += summary(degree_np) + summary(length_np) + summary(weight_np)
    for harmonic in (1, 2, 3, 4, 6, 8):
        global_x += [
            float(np.cos(harmonic * angle).mean()),
            float(np.sin(harmonic * angle).mean()),
        ]
    for axis in (0, 1):
        coord = xy[:, axis].cpu().numpy()
        for side in (-1, 1):
            mask = coord * side > .7
            global_x += [
                float(degree_np[mask].mean()) if mask.any() else 0.0,
                float(mask.mean()),
            ]
    return node_x, edge_x, torch.tensor(global_x, dtype=torch.float32)


class MessageLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, node_h, edge_h, edge_index):
        source, target = edge_index
        message = self.message(torch.cat([node_h[source], node_h[target], edge_h], dim=1))
        aggregate = torch.zeros_like(node_h)
        count = torch.zeros(node_h.size(0), 1, dtype=node_h.dtype, device=node_h.device)
        aggregate.index_add_(0, target, message)
        count.index_add_(0, target, torch.ones(message.size(0), 1, device=node_h.device))
        return node_h + self.update(torch.cat([node_h, aggregate / count.clamp_min(1)], dim=1))


class StaticVelocityGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, global_dim: int, hidden: int = 96):
        super().__init__()
        self.node_in = nn.Sequential(nn.Linear(node_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.edge_in = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList(MessageLayer(hidden) for _ in range(4))
        self.global_in = nn.Sequential(nn.Linear(global_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.trunk = nn.Sequential(nn.Linear(4 * hidden, 2 * hidden), nn.GELU(), nn.Dropout(.1), nn.Linear(2 * hidden, hidden), nn.GELU())
        self.velocity = nn.Linear(hidden, 2)
        self.p_ratio = nn.Linear(hidden, 1)

    def forward(self, data):
        node_h, edge_h = self.node_in(data.x), self.edge_in(data.edge_attr)
        for layer in self.layers:
            node_h = layer(node_h, edge_h, data.edge_index)
        mean = global_mean_pool(node_h, data.batch)
        maximum = global_max_pool(node_h, data.batch)
        variance = global_mean_pool(node_h.square(), data.batch) - mean.square()
        global_h = self.global_in(data.u.reshape(mean.size(0), -1))
        h = self.trunk(torch.cat([mean, maximum, variance.clamp_min(0).sqrt(), global_h], dim=1))
        return self.velocity(h), self.p_ratio(h)


def prepare(bundle, sims, cfg, device, cache_path: Path):
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    rows = []
    with torch.no_grad():
        for index, sim in enumerate(sims):
            z0 = encode_frame_latent(bundle["ae"], sim, 0, pos_dim=2,
                node_feature_mode="normalized_delta", normalizers=bundle["normalizers"], device=device)
            z1 = encode_frame_latent(bundle["ae"], sim, len(sim) - 1, pos_dim=2,
                node_feature_mode="normalized_delta", normalizers=bundle["normalizers"], device=device)
            node_x, edge_x, global_x = graph_features(sim[0])
            rows.append(Data(
                x=node_x, edge_index=sim[0].edge_index.long(), edge_attr=edge_x,
                u=global_x[None], velocity=((z1 - z0) / (len(sim) - 1)).cpu()[None],
                z0=z0.cpu()[None], p_ratio=torch.tensor([[ground_truth_p_ratio(
                    sim, dataset_name="lj_noisy", cfg=cfg)]], dtype=torch.float32),
                registry_p_ratio=torch.tensor([[float(sim[0].registry_poisson_ratio)]]),
                sim_index=torch.tensor([index]),
            ))
            if index == 0 or (index + 1) % 100 == 0:
                print(f"prepared {index + 1}/{len(sims)}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, cache_path)
    return rows


def target_stats(rows, indices):
    velocity = torch.cat([rows[i].velocity for i in indices])
    p_ratio = torch.cat([rows[i].p_ratio for i in indices])
    return {
        "v_mean": velocity.mean(0, keepdim=True),
        "v_std": velocity.std(0, keepdim=True, unbiased=False).clamp_min(1e-7),
        "p_mean": p_ratio.mean(0, keepdim=True),
        "p_std": p_ratio.std(0, keepdim=True, unbiased=False).clamp_min(1e-7),
    }


def epoch(model, loader, stats, device, optimizer=None):
    model.train(optimizer is not None)
    losses = []
    for data in loader:
        data = data.to(device)
        pred_v, pred_p = model(data)
        target_v = (data.velocity - stats["v_mean"]) / stats["v_std"]
        target_p = (data.p_ratio - stats["p_mean"]) / stats["p_std"]
        loss_v = F.mse_loss(pred_v, target_v)
        loss_p = F.mse_loss(pred_p, target_p)
        loss = loss_v + .5 * loss_p
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def fit_model(rows, train_idx, val_idx, device, max_epochs=250):
    stats = {k: v.to(device) for k, v in target_stats(rows, train_idx).items()}
    model = StaticVelocityGNN(rows[0].x.size(1), rows[0].edge_attr.size(1), rows[0].u.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=3e-4)
    train_loader = DataLoader([rows[i] for i in train_idx], batch_size=32, shuffle=True)
    val_loader = DataLoader([rows[i] for i in val_idx], batch_size=64)
    best, best_val, stale, history = None, float("inf"), 0, []
    for step in range(1, max_epochs + 1):
        train_loss = epoch(model, train_loader, stats, device, optimizer)
        with torch.no_grad():
            val_loss = epoch(model, val_loader, stats, device)
        history.append({"epoch": step, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-4:
            best_val, stale = val_loss, 0
            best = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if step == 1 or step % 20 == 0:
            print(f"epoch {step:03d} train={train_loss:.4f} val={val_loss:.4f} stale={stale}", flush=True)
        if stale >= 30:
            break
    model.load_state_dict(best)
    return model.eval(), stats, pd.DataFrame(history), best_val


def predict(model, stats, rows, indices, device):
    predictions = []
    loader = DataLoader([rows[i] for i in indices], batch_size=64)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            velocity, p_ratio = model(data)
            predictions.extend(zip(
                (velocity * stats["v_std"] + stats["v_mean"]).cpu(),
                (p_ratio * stats["p_std"] + stats["p_mean"]).cpu(),
            ))
    return predictions


def evaluate(model, stats, rows, sims, test_idx, bundle, device, train_count):
    predictions = predict(model, stats, rows, test_idx, device)
    true_p = np.asarray([float(rows[i].p_ratio) for i in test_idx])
    pred_p = np.asarray([float(p[1]) for p in predictions])
    true_v = np.stack([rows[i].velocity.numpy().squeeze(0) for i in test_idx])
    pred_v = np.stack([p[0].numpy() for p in predictions])
    summary_rows = [{
        "train_networks": train_count, "horizon": 0, "test_networks": len(test_idx),
        "direct_p_ratio_r2": r2_score(true_p, pred_p),
        "direct_p_ratio_pearson": pearson_r(true_p, pred_p),
        "velocity_r2_z0": r2_score(true_v[:, 0], pred_v[:, 0]),
        "velocity_r2_z1": r2_score(true_v[:, 1], pred_v[:, 1]),
    }]
    row_details = []
    for horizon in (12, 25, 49):
        pred_mse, initial_mse, decoded_p, endpoint_true_p = [], [], [], []
        with torch.no_grad():
            for local, sim_index in enumerate(test_idx):
                sim = sims[sim_index]
                z = rows[sim_index].z0.squeeze(0).to(device) + horizon * torch.as_tensor(pred_v[local], device=device)
                position = decode_latent_positions(bundle["ae"], sim, z, horizon, pos_dim=2,
                    ae_target_mode="normalized_delta", normalizers=bundle["normalizers"], device=device).cpu()
                target = sim[horizon].x[:, :2].float()
                reference = sim[0].x[:, :2].float()
                pred_mse.append(float(F.mse_loss(position, target)))
                initial_mse.append(float(F.mse_loss(reference, target)))
                # Endpoint p-ratio from the same side strips used by the rollout metric.
                x0 = reference.numpy(); x1 = position.numpy()
                qx_lo, qx_hi = np.quantile(x0[:, 0], [.1, .9]); qy_lo, qy_hi = np.quantile(x0[:, 1], [.1, .9])
                left, right = x0[:, 0] <= qx_lo, x0[:, 0] >= qx_hi
                bottom, top = x0[:, 1] <= qy_lo, x0[:, 1] >= qy_hi
                w0, w1 = x0[right, 0].mean() - x0[left, 0].mean(), x1[right, 0].mean() - x1[left, 0].mean()
                h0, h1 = x0[top, 1].mean() - x0[bottom, 1].mean(), x1[top, 1].mean() - x1[bottom, 1].mean()
                decoded_p.append(abs(((h1 - h0) / h0) / (((w1 - w0) / w0) + 1e-12)))
                endpoint_true_p.append(ground_truth_p_ratio(sim, horizon, dataset_name="lj_noisy", cfg=bundle["params"]))
                row_details.append({"train_networks": train_count, "sim_index": sim_index,
                    "horizon": horizon, "true_p_ratio": true_p[local], "pred_p_ratio": pred_p[local],
                    "decoded_endpoint_p_ratio": decoded_p[-1]})
        position_r2 = max(0.0, 1.0 - np.mean(pred_mse) / np.mean(initial_mse))
        summary_rows.append({
            "train_networks": train_count, "horizon": horizon, "test_networks": len(test_idx),
            "rollout_position_r2": position_r2,
            "decoded_endpoint_p_ratio_r2": r2_score(np.asarray(endpoint_true_p), np.asarray(decoded_p)),
            "decoded_endpoint_p_ratio_pearson": pearson_r(np.asarray(endpoint_true_p), np.asarray(decoded_p)),
            "direct_p_ratio_r2": r2_score(true_p, pred_p), "direct_p_ratio_pearson": pearson_r(true_p, pred_p),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(row_details)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-counts", type=int, nargs="+", default=[20, 200, 1000])
    parser.add_argument("--epochs", type=int, default=250)
    args = parser.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = resolve_device("auto")
    baseline_raw = torch.load(BASELINE, map_location="cpu", weights_only=False)
    cfg = dict(baseline_raw["params"])
    bundle = load_experiment_bundle(BASELINE, cfg=cfg, device=device)
    sims = load_dataset(DATA, edge_multiplicity=1)
    rows = prepare(bundle, sims, cfg, device, OUT / "static_velocity_targets.pt")
    # The final 200 registry entries were never available to the original AE training.
    test_idx = list(range(len(rows) - 200, len(rows)))
    pool = np.arange(len(rows) - 200)
    rng = np.random.default_rng(SEED); rng.shuffle(pool)
    val_idx, train_pool = pool[:148].tolist(), pool[148:].tolist()
    all_summary, all_details = [], []
    for count in args.train_counts:
        print(f"\n=== static-only model, train={count} ===", flush=True)
        model, stats, history, best_val = fit_model(rows, train_pool[:count], val_idx, device, args.epochs)
        summary_frame, detail_frame = evaluate(model, stats, rows, sims, test_idx, bundle, device, count)
        summary_frame["best_val_loss"] = best_val
        all_summary.append(summary_frame); all_details.append(detail_frame)
        history.to_csv(OUT / f"history_train{count}.csv", index=False)
        torch.save({"state_dict": model.state_dict(), "stats": {k: v.cpu() for k, v in stats.items()},
            "train_indices": train_pool[:count], "val_indices": val_idx, "test_indices": test_idx}, OUT / f"static_velocity_train{count}.pt")
        print(summary_frame.to_string(index=False), flush=True)
    summary_frame = pd.concat(all_summary, ignore_index=True)
    detail_frame = pd.concat(all_details, ignore_index=True)
    summary_frame.to_csv(OUT / "summary.csv", index=False)
    detail_frame.to_csv(OUT / "test_predictions.csv", index=False)
    registry = np.asarray([float(rows[i].registry_p_ratio) for i in test_idx])
    robust = np.asarray([float(rows[i].p_ratio) for i in test_idx])
    print(f"held-out registry-vs-trajectory r={pearson_r(registry, robust):.4f}")


if __name__ == "__main__":
    main()
