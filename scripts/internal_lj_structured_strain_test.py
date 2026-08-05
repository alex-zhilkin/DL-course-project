"""Quick structured 2D strain-latent test for the 200x200 noisy LJ dataset."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from graph_utils import directional_side_indices_from_box
from torch import nn
from torch_geometric.utils import softmax

from lss.data import load_dataset
from lss.latent.simulation import pearson_r, r2_score


DATA = ROOT / "data" / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"
SEED = 20260726
GRID = 8
FRAME_SKIP = 2
TRAIN_NETWORKS = 100
VAL_NETWORKS = 20
TRAIN_STEPS = 50  # sampled steps: raw frames 0 through 100
CURRICULUM = [(1, 8), (4, 10), (8, 12), (16, 16)]


def strain_path(sim) -> np.ndarray:
    """Return observable [x strain, y strain] using fixed reference-side nodes."""

    side = directional_side_indices_from_box(sim[0], quantile=0.10, eps=1e-12)

    def dimensions(graph):
        pos = graph.x[:, :2].detach().cpu().numpy()
        width = pos[side["right"], 0].mean() - pos[side["left"], 0].mean()
        height = pos[side["top"], 1].mean() - pos[side["bottom"], 1].mean()
        return float(width), float(height)

    width0, height0 = dimensions(sim[0])
    frames = list(range(0, len(sim), FRAME_SKIP))
    values = []
    for frame in frames:
        width, height = dimensions(sim[frame])
        values.append([(width - width0) / width0, (height - height0) / height0])
    return np.asarray(values, dtype=np.float32)


def static_grid_descriptor(sim) -> np.ndarray:
    """Rasterize static node and edge structure without using trajectory targets."""

    graph = sim[0]
    pos = graph.x[:, :2].detach().cpu().numpy().astype(np.float64)
    edge_index = graph.edge_index.detach().cpu().numpy()
    edge_attr = graph.edge_attr.detach().cpu().numpy().astype(np.float64)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    scale = np.maximum(hi - lo, 1e-8)
    normalized = np.clip((pos - lo) / scale, 0.0, 1.0 - 1e-9)
    node_cell = np.floor(normalized * GRID).astype(int)

    first, second = edge_index
    stiffness = edge_attr[:, -1]
    vector = edge_attr[:, :2]
    length = np.linalg.norm(vector, axis=1).clip(1e-8)
    degree = np.bincount(np.concatenate([first, second]), minlength=len(pos))
    incident_stiffness = np.zeros(len(pos))
    np.add.at(incident_stiffness, first, stiffness)
    np.add.at(incident_stiffness, second, stiffness)

    channels = np.zeros((8, GRID, GRID), dtype=np.float64)
    np.add.at(channels[0], (node_cell[:, 1], node_cell[:, 0]), 1.0)
    np.add.at(channels[1], (node_cell[:, 1], node_cell[:, 0]), degree)
    np.add.at(
        channels[2],
        (node_cell[:, 1], node_cell[:, 0]),
        incident_stiffness,
    )

    midpoint = np.clip(
        0.5 * (normalized[first] + normalized[second]), 0.0, 1.0 - 1e-9
    )
    edge_cell = np.floor(midpoint * GRID).astype(int)
    angle = np.arctan2(vector[:, 1], vector[:, 0])
    np.add.at(channels[3], (edge_cell[:, 1], edge_cell[:, 0]), 1.0)
    np.add.at(channels[4], (edge_cell[:, 1], edge_cell[:, 0]), stiffness)
    np.add.at(channels[5], (edge_cell[:, 1], edge_cell[:, 0]), length)
    np.add.at(
        channels[6],
        (edge_cell[:, 1], edge_cell[:, 0]),
        stiffness * np.cos(2 * angle),
    )
    np.add.at(
        channels[7],
        (edge_cell[:, 1], edge_cell[:, 0]),
        stiffness * np.sin(2 * angle),
    )
    channels[0] /= max(1, len(pos))
    channels[1:3] /= max(1, len(pos))
    channels[3:] /= max(1, len(stiffness))

    global_features = np.asarray(
        [
            len(pos) / 200,
            len(stiffness) / 500,
            degree.mean(),
            degree.std(),
            stiffness.mean(),
            stiffness.std(),
            *np.quantile(stiffness, [0.1, 0.5, 0.9]),
            length.mean(),
            length.std(),
            *np.quantile(length, [0.1, 0.5, 0.9]),
        ],
        dtype=np.float64,
    )
    return np.concatenate([channels.reshape(-1), global_features]).astype(np.float32)


def rolling_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return values.copy()
    half = window // 2
    return np.asarray(
        [
            values[max(0, index - half) : min(len(values), index + half + 1)].mean(
                axis=0
            )
            for index in range(len(values))
        ]
    )


def p_ratio_from_strains(strains: np.ndarray, horizon: int) -> float:
    """Match the robust p-ratio definition, operating directly on the 2D latent."""

    values = rolling_mean(np.asarray(strains[: horizon + 1], dtype=float), 5)
    meaningful = (np.abs(values[:, 0]) >= 1e-5) | (np.abs(values[:, 1]) >= 1e-5)
    values = values[meaningful]
    if len(values) < 8:
        return float("nan")
    ranges = np.ptp(values, axis=0)
    if float(ranges.max()) < 1e-3:
        return float("nan")
    driven = int(ranges[1] > ranges[0])
    transverse = 1 - driven
    slopes = []
    x, y = values[:, driven], values[:, transverse]
    for first in range(len(values) - 1):
        dx = x[first + 1 :] - x[first]
        valid = np.abs(dx) > 1e-12
        slopes.extend(((y[first + 1 :] - y[first])[valid] / dx[valid]).tolist())
    return float(-np.median(slopes)) if slopes else float("nan")


class StructuredStrainPropagator(nn.Module):
    def __init__(self, descriptor_dim: int, hidden: int = 192, context: int = 96):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, context),
            nn.GELU(),
            nn.LayerNorm(context),
        )
        self.step = nn.Sequential(
            nn.Linear(context + 2 + 2 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.xavier_uniform_(self.step[-1].weight)
        self.step[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.step[-1].bias)

    def encode_static(self, descriptor):
        return self.context(descriptor)

    def advance(self, q, previous_q, progress, context):
        velocity = q - previous_q
        update = self.step(torch.cat([q, velocity, progress, context], dim=-1))
        return q + update


def complete_attention_graph(sim):
    """Build one canonical all-pairs edge with static distance/bond features."""

    graph = sim[0]
    pos = graph.x[:, :2].detach().cpu().float()
    lo, hi = pos.amin(dim=0), pos.amax(dim=0)
    scale = (hi - lo).amax().clamp_min(1e-6)
    center = 0.5 * (hi + lo)
    pos_norm = (pos - center) / scale
    count = pos.size(0)

    stored = graph.edge_index.detach().cpu().long()
    stored_first = torch.minimum(stored[0], stored[1])
    stored_second = torch.maximum(stored[0], stored[1])
    stored_key = stored_first * count + stored_second
    stiffness_matrix = torch.zeros((count, count), dtype=torch.float32)
    stiffness = graph.edge_attr[:, -1].detach().cpu().float()
    stiffness_matrix[stored_first, stored_second] = stiffness
    stiffness_matrix[stored_second, stored_first] = stiffness
    degree = torch.bincount(
        torch.cat([stored_first, stored_second]), minlength=count
    ).float()
    incident = torch.zeros(count)
    incident.index_add_(0, stored_first, stiffness)
    incident.index_add_(0, stored_second, stiffness)

    boundary = torch.zeros((count, 4))
    side_count = max(1, int(np.ceil(0.10 * count)))
    boundary[torch.topk(pos_norm[:, 0], side_count, largest=False).indices, 0] = 1
    boundary[torch.topk(pos_norm[:, 0], side_count, largest=True).indices, 1] = 1
    boundary[torch.topk(pos_norm[:, 1], side_count, largest=False).indices, 2] = 1
    boundary[torch.topk(pos_norm[:, 1], side_count, largest=True).indices, 3] = 1
    node = torch.cat(
        [
            pos_norm,
            boundary,
            (degree / max(1, count)).reshape(-1, 1),
            (incident / degree.clamp_min(1)).reshape(-1, 1),
            torch.full((count, 1), count / 200),
        ],
        dim=-1,
    )

    edge_index = torch.triu_indices(count, count, offset=1)
    first, second = edge_index
    vector = pos_norm[second] - pos_norm[first]
    distance = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    pair_stiffness = stiffness_matrix[first, second].reshape(-1, 1)
    stored_flag = (pair_stiffness != 0).float()
    centers = torch.linspace(0.0, 1.5, 8).reshape(1, -1)
    radial = torch.exp(-((distance - centers) / 0.18).square())
    edge = torch.cat(
        [vector, distance, pair_stiffness, stored_flag, radial], dim=-1
    )
    # A geometric prior only; there is deliberately no permanent-bond boost.
    prior = (-0.5 * (distance.squeeze(-1) / 0.18).square()).clamp_min(-8.0)
    return node, edge, edge_index, prior


class FullEdgeAttentionStrainPropagator(nn.Module):
    """One-shot full-edge attention followed by 2D strain dynamics."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden: int = 96,
        context_dim: int = 96,
        context_tokens: int = 8,
    ):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.edge_value = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.node_fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.pool_queries = nn.Parameter(
            torch.randn(context_tokens, hidden) * 0.02
        )
        self.pool_keys = nn.Linear(hidden, hidden)
        self.pool_values = nn.Linear(hidden, hidden)
        self.context_fuse = nn.Sequential(
            nn.Linear(context_tokens * hidden, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        self.step = nn.Sequential(
            nn.Linear(context_dim + 5, 192),
            nn.GELU(),
            nn.Linear(192, 192),
            nn.GELU(),
            nn.Linear(192, 192),
            nn.GELU(),
            nn.Linear(192, 2),
        )
        nn.init.zeros_(self.edge_score[-1].weight)
        nn.init.zeros_(self.edge_score[-1].bias)
        nn.init.xavier_uniform_(self.step[-1].weight)
        self.step[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.step[-1].bias)
        self.score_scale = hidden**0.5

    def encode_graph(self, node, edge, edge_index, prior):
        node_h = self.node_encoder(node)
        edge_h = self.edge_encoder(edge)
        first, second = edge_index
        reverse_edge = edge.clone()
        reverse_edge[:, :2] *= -1
        reverse_edge_h = self.edge_encoder(reverse_edge)

        forward_logits = self.edge_score(
            torch.cat([node_h[first], node_h[second], edge_h], dim=-1)
        ).squeeze(-1)
        reverse_logits = self.edge_score(
            torch.cat([node_h[second], node_h[first], reverse_edge_h], dim=-1)
        ).squeeze(-1)
        endpoints = torch.cat([second, first])
        # Scale only the learned dot-product-like score.  The distance prior is
        # already a logit, so scaling it would make the initial attention nearly
        # uniform over the complete graph.
        logits = torch.cat([
            forward_logits / self.score_scale + prior,
            reverse_logits / self.score_scale + prior,
        ])
        attention = softmax(logits, endpoints, num_nodes=len(node))
        forward_value = self.edge_value(
            torch.cat([node_h[first], edge_h], dim=-1)
        )
        reverse_value = self.edge_value(
            torch.cat([node_h[second], reverse_edge_h], dim=-1)
        )
        values = torch.cat([forward_value, reverse_value])
        aggregate = torch.zeros_like(node_h)
        aggregate.index_add_(0, endpoints, values * attention.unsqueeze(-1))
        node_token = node_h + self.node_fuse(
            torch.cat([node_h, aggregate], dim=-1)
        )

        keys = self.pool_keys(node_token)
        values = self.pool_values(node_token)
        scores = self.pool_queries @ keys.transpose(0, 1) / self.score_scale
        tokens = torch.softmax(scores, dim=-1) @ values
        return self.context_fuse(tokens.reshape(1, -1)).squeeze(0)

    def advance(self, q, previous_q, progress, context):
        velocity = q - previous_q
        return q + self.step(torch.cat([q, velocity, progress, context], dim=-1))


def sequence_loss(
    model,
    context,
    paths,
    local_sim,
    start,
    *,
    horizon,
    z_scale,
    time_variation,
):
    q = paths[local_sim, start] / z_scale
    previous_q = paths[local_sim, (start - 1).clamp_min(0)] / z_scale
    row_context = context[local_sim]
    step_losses = []
    for step in range(1, horizon + 1):
        target_index = start + step
        progress = target_index.float().unsqueeze(-1) / (paths.size(1) - 1)
        prediction = model.advance(q, previous_q, progress, row_context)
        target = paths[local_sim, target_index] / z_scale
        base = (prediction - target).square().mean(dim=-1)
        variation = time_variation[target_index].clamp_min(0.05)
        network_specific = (
            (prediction - target) / variation
        ).square().mean(dim=-1)
        weight = 1.0 + (step - 1) / max(1, horizon)
        step_losses.append((base + 0.02 * network_specific) * weight)
        previous_q, q = q, prediction
    return torch.stack(step_losses).sum(dim=0).mean() / sum(
        1.0 + step / max(1, horizon) for step in range(horizon)
    )


def graph_epoch(
    model,
    graphs,
    paths,
    *,
    horizon,
    z_scale,
    time_variation,
    optimizer=None,
    networks_per_batch=10,
):
    training = optimizer is not None
    model.train(training)
    network_order = np.arange(len(graphs))
    if training:
        np.random.shuffle(network_order)
    losses = []
    for offset in range(0, len(graphs), networks_per_batch):
        selected = network_order[offset : offset + networks_per_batch]
        context = torch.stack(
            [model.encode_graph(*graphs[int(index)]) for index in selected]
        )
        local_sim, starts = [], []
        for local_index in range(len(selected)):
            for start in range(TRAIN_STEPS - horizon + 1):
                local_sim.append(local_index)
                starts.append(start)
        local_sim = torch.tensor(local_sim, dtype=torch.long, device=paths.device)
        starts = torch.tensor(starts, dtype=torch.long, device=paths.device)
        loss = sequence_loss(
            model,
            context,
            paths[torch.as_tensor(selected, dtype=torch.long, device=paths.device)],
            local_sim,
            starts,
            horizon=horizon,
            z_scale=z_scale,
            time_variation=time_variation,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def graph_rollout(model, graphs, steps, z_scale, device):
    model.eval()
    contexts = torch.stack([model.encode_graph(*graph) for graph in graphs])
    q = torch.zeros((len(graphs), 2), device=device)
    previous_q = q.clone()
    output = [q * z_scale]
    for step in range(1, steps + 1):
        progress = torch.full((len(graphs), 1), step / steps, device=device)
        prediction = model.advance(q, previous_q, progress, contexts)
        previous_q, q = q, prediction
        output.append(q * z_scale)
    return torch.stack(output, dim=1).cpu().numpy()


def run_full_attention(simulations, paths_np, device):
    print("building canonical complete-edge attention graphs", flush=True)
    raw_graphs = [complete_attention_graph(sim) for sim in simulations]
    node_all = torch.cat([graph[0] for graph in raw_graphs[:TRAIN_NETWORKS]])
    edge_all = torch.cat([graph[1] for graph in raw_graphs[:TRAIN_NETWORKS]])
    node_mean, node_std = node_all.mean(0), node_all.std(0).clamp_min(1e-5)
    edge_mean, edge_std = edge_all.mean(0), edge_all.std(0).clamp_min(1e-5)
    graphs = [
        (
            ((node - node_mean) / node_std).to(device),
            ((edge - edge_mean) / edge_std).to(device),
            edge_index.to(device),
            prior.to(device),
        )
        for node, edge, edge_index, prior in raw_graphs
    ]
    paths = torch.from_numpy(paths_np).to(device)
    train_paths = paths[:TRAIN_NETWORKS]
    z_scale = train_paths[:, : TRAIN_STEPS + 1].std(dim=(0, 1)).clamp_min(1e-5)
    time_variation = (
        train_paths.std(dim=0, unbiased=False) / z_scale.reshape(1, 2)
    ).clamp_min(0.02)
    model = FullEdgeAttentionStrainPropagator(
        graphs[0][0].size(1), graphs[0][1].size(1)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    epoch_offset = 0
    for horizon, max_epochs in CURRICULUM:
        best_state, best_val, stale = None, float("inf"), 0
        for stage_epoch in range(1, max_epochs + 1):
            train_loss = graph_epoch(
                model,
                graphs[:100],
                paths[:100],
                horizon=horizon,
                z_scale=z_scale,
                time_variation=time_variation,
                optimizer=optimizer,
            )
            with torch.no_grad():
                val_loss = graph_epoch(
                    model,
                    graphs[100:120],
                    paths[100:120],
                    horizon=horizon,
                    z_scale=z_scale,
                    time_variation=time_variation,
                )
            improved = val_loss < best_val - 1e-5
            if improved:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            print(
                f"attention h={horizon:02d} epoch={epoch_offset + stage_epoch:03d} "
                f"train={train_loss:.6g} val={val_loss:.6g} stale={stale}",
                flush=True,
            )
            if stale >= 4:
                break
        model.load_state_dict(best_state)
        epoch_offset += stage_epoch
    predicted = graph_rollout(model, graphs, paths.size(1) - 1, z_scale, device)
    horizons = [12, 25, 50, 75, 99]
    print("\nattention train:", metrics(paths_np[:100], predicted[:100], horizons))
    print("attention val:", metrics(paths_np[100:120], predicted[100:120], horizons))
    print("attention test:", metrics(paths_np[120:], predicted[120:], horizons))


def chunks(count: int, horizon: int):
    return np.asarray(
        [
            (sim_index, start)
            for sim_index in range(count)
            for start in range(TRAIN_STEPS - horizon + 1)
        ],
        dtype=np.int64,
    )


def epoch(
    model,
    descriptors,
    paths,
    rows,
    *,
    horizon,
    z_scale,
    time_variation,
    optimizer=None,
    batch_size=256,
):
    training = optimizer is not None
    model.train(training)
    order = np.arange(len(rows))
    if training:
        np.random.shuffle(order)
    losses = []
    for offset in range(0, len(order), batch_size):
        selected = rows[order[offset : offset + batch_size]]
        sim_idx = torch.as_tensor(selected[:, 0], dtype=torch.long)
        start = torch.as_tensor(selected[:, 1], dtype=torch.long)
        context = model.encode_static(descriptors[sim_idx])
        q = paths[sim_idx, start] / z_scale
        previous_index = (start - 1).clamp_min(0)
        previous_q = paths[sim_idx, previous_index] / z_scale
        step_losses = []
        for step in range(1, int(horizon) + 1):
            target_index = start + step
            progress = target_index.float().unsqueeze(-1) / (paths.size(1) - 1)
            prediction = model.advance(q, previous_q, progress, context)
            target = paths[sim_idx, target_index] / z_scale
            base = (prediction - target).square().mean(dim=-1)
            variation = time_variation[target_index].clamp_min(0.05)
            network_specific = (
                (prediction - target) / variation
            ).square().mean(dim=-1)
            weight = 1.0 + (step - 1) / max(1, horizon)
            step_losses.append((base + 0.02 * network_specific) * weight)
            previous_q, q = q, prediction
        loss = torch.stack(step_losses).sum(dim=0).mean() / sum(
            1.0 + step / max(1, horizon) for step in range(horizon)
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def rollout(model, descriptors, steps: int, z_scale):
    model.eval()
    context = model.encode_static(descriptors)
    q = torch.zeros((len(descriptors), 2), device=descriptors.device)
    previous_q = q.clone()
    output = [q * z_scale]
    for step in range(1, steps + 1):
        progress = torch.full(
            (len(descriptors), 1),
            step / steps,
            device=descriptors.device,
        )
        prediction = model.advance(q, previous_q, progress, context)
        previous_q, q = q, prediction
        output.append(q * z_scale)
    return torch.stack(output, dim=1).cpu().numpy()


def metrics(true_paths, predicted_paths, horizons):
    rows = []
    for horizon in horizons:
        true = np.asarray([p_ratio_from_strains(path, horizon) for path in true_paths])
        predicted = np.asarray(
            [p_ratio_from_strains(path, horizon) for path in predicted_paths]
        )
        rows.append(
            (
                horizon * FRAME_SKIP,
                int(np.isfinite(true * predicted).sum()),
                r2_score(true, predicted),
                pearson_r(true, predicted),
            )
        )
    return rows


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulations = load_dataset(DATA, edge_multiplicity=1)
    if os.environ.get("LJ_SHUFFLE_SPLIT", "0") == "1":
        order = np.random.default_rng(SEED).permutation(len(simulations))
        simulations = [simulations[int(index)] for index in order]
        print("using a deterministic randomized 100/20/80 split", flush=True)
    paths_np = np.stack([strain_path(sim) for sim in simulations])
    if os.environ.get("LJ_ENCODER", "grid") == "full_attention":
        run_full_attention(simulations, paths_np, device)
        return
    descriptors_np = np.stack([static_grid_descriptor(sim) for sim in simulations])
    descriptor_mean = descriptors_np[:TRAIN_NETWORKS].mean(axis=0, keepdims=True)
    descriptor_std = descriptors_np[:TRAIN_NETWORKS].std(axis=0, keepdims=True)
    descriptors_np = (descriptors_np - descriptor_mean) / np.maximum(
        descriptor_std, 1e-5
    )

    descriptors = torch.from_numpy(descriptors_np).to(device)
    paths = torch.from_numpy(paths_np).to(device)
    train_paths = paths[:TRAIN_NETWORKS]
    z_scale = train_paths[:, : TRAIN_STEPS + 1].std(dim=(0, 1)).clamp_min(1e-5)
    time_variation = (
        train_paths.std(dim=0, unbiased=False) / z_scale.reshape(1, 2)
    ).clamp_min(0.02)
    model = StructuredStrainPropagator(descriptors.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    train_desc = descriptors[:TRAIN_NETWORKS]
    val_desc = descriptors[TRAIN_NETWORKS : TRAIN_NETWORKS + VAL_NETWORKS]
    val_paths = paths[TRAIN_NETWORKS : TRAIN_NETWORKS + VAL_NETWORKS]
    epoch_offset = 0
    for horizon, max_epochs in CURRICULUM:
        train_rows = chunks(TRAIN_NETWORKS, horizon)
        val_rows = chunks(VAL_NETWORKS, horizon)
        best_state, best_val, stale = None, float("inf"), 0
        for stage_epoch in range(1, max_epochs + 1):
            train_loss = epoch(
                model,
                train_desc,
                train_paths,
                train_rows,
                horizon=horizon,
                z_scale=z_scale,
                time_variation=time_variation,
                optimizer=optimizer,
            )
            with torch.no_grad():
                val_loss = epoch(
                    model,
                    val_desc,
                    val_paths,
                    val_rows,
                    horizon=horizon,
                    z_scale=z_scale,
                    time_variation=time_variation,
                )
            improved = val_loss < best_val - 1e-5
            if improved:
                best_val = val_loss
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            print(
                f"h={horizon:02d} epoch={epoch_offset + stage_epoch:03d} "
                f"train={train_loss:.6g} val={val_loss:.6g} stale={stale}",
                flush=True,
            )
            if stale >= 4:
                break
        model.load_state_dict(best_state)
        epoch_offset += stage_epoch

    predicted = rollout(model, descriptors, paths.size(1) - 1, z_scale)
    horizons = [12, 25, 50, 75, 99]
    print("\ntrain:", metrics(paths_np[:100], predicted[:100], horizons))
    print("val:", metrics(paths_np[100:120], predicted[100:120], horizons))
    print("test:", metrics(paths_np[120:], predicted[120:], horizons))


if __name__ == "__main__":
    main()
