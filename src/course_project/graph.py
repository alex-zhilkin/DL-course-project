from __future__ import annotations

import torch
from torch_geometric.data import Data


def clone_graph(graph: Data) -> Data:
    out = Data(
        x=graph.x.clone(),
        edge_index=graph.edge_index.clone(),
        edge_attr=graph.edge_attr.clone() if graph.edge_attr is not None else None,
        box=graph.box if hasattr(graph, "box") else None,
        t=graph.t if hasattr(graph, "t") else None,
        batch=graph.batch.clone() if hasattr(graph, "batch") and graph.batch is not None else None,
    )
    if hasattr(graph, "vel_state"):
        out.vel_state = graph.vel_state.clone()
    return out


def build_graph(input_graphs: list[Data]) -> Data:
    """Build model input graph using positions as node features."""
    base_graph = input_graphs[-1]
    data = clone_graph(base_graph)
    if not hasattr(data, "t"):
        data.t = None
    if hasattr(base_graph, "vel_state"):
        data.vel_state = base_graph.vel_state
    return data


def rollout(
    model,
    input_graphs: list[Data],
    num_steps: int,
    history: int,
    pos_dim: int,
    device: str,
    model_inputs_cls,
):
    rollout_graphs = [clone_graph(g).cpu() for g in input_graphs]
    model.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            frames = [clone_graph(g).to(device) for g in rollout_graphs[-(history + 1) :]]
            input_graph = build_graph(input_graphs=frames).to(device)
            cur_graph = clone_graph(frames[-1]).to(device)
            prev_graph = clone_graph(frames[-2] if len(frames) > 1 else frames[-1]).to(device)
            if len(frames) > 1:
                cur_graph.vel_state = cur_graph.x[:, :pos_dim] - prev_graph.x[:, :pos_dim]
            model_inputs = model_inputs_cls(prev_graph, cur_graph, None, pos_dim)
            model_output = model(input_graph, is_training=False)
            predicted_graph = model.update(model_inputs, model_output)
            rollout_graphs.append(predicted_graph.cpu())
    return rollout_graphs
