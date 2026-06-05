from __future__ import annotations

import torch
from torch_geometric.data import Data


def clone_graph(graph: Data) -> Data:
    batch = graph.batch.clone() if getattr(graph, 'batch', None) is not None else torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
    out = Data(
        x=graph.x.clone(),
        edge_index=graph.edge_index.clone(),
        edge_attr=graph.edge_attr.clone(),
        box=graph.box,
        t=getattr(graph, 't', None),
        batch=batch,
    )
    if hasattr(graph, "vel_state"):
        out.vel_state = graph.vel_state.clone()
    return out


def build_graph(input_graphs: list[Data], node_features: str = "positions") -> Data:
    """Build model input graph using positions, velocity history, or both."""
    if len(input_graphs) < 1:
        raise ValueError("input_graphs must contain at least one graph")

    base_graph = input_graphs[-1]
    match node_features:
        case "positions":
            return clone_graph(base_graph)
        case "velocity":
            if len(input_graphs) == 1:
                x = torch.zeros_like(base_graph.x)
            else:
                x = torch.column_stack(
                    [
                        input_graphs[i].x - input_graphs[i - 1].x
                        for i in range(len(input_graphs) - 1, 0, -1)
                    ]
                )
        case "combined":
            if len(input_graphs) == 1:
                velocity = torch.zeros_like(base_graph.x)
            else:
                velocity = torch.column_stack(
                    [
                        input_graphs[i].x - input_graphs[i - 1].x
                        for i in range(len(input_graphs) - 1, 0, -1)
                    ]
                )
            x = torch.column_stack([velocity, base_graph.x])
        case _:
            raise ValueError("node_features must be 'positions', 'velocity', or 'combined'")

    out = Data(
        x=x,
        edge_index=base_graph.edge_index,
        edge_attr=base_graph.edge_attr,
        box=base_graph.box if hasattr(base_graph, "box") else None,
        t=getattr(base_graph, "t", None),
        batch=base_graph.batch.clone() if getattr(base_graph, "batch", None) is not None else torch.zeros(base_graph.x.size(0), dtype=torch.long, device=base_graph.x.device),
    )
    if hasattr(base_graph, "vel_state"):
        out.vel_state = base_graph.vel_state.clone()
    return out


def rollout(
    model,
    input_graphs: list[Data],
    num_steps: int,
    history: int,
    pos_dim: int,
    device: str,
    model_inputs_cls,
    node_features: str = "positions",
):
    rollout_graphs = [clone_graph(g).cpu() for g in input_graphs]
    model.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            frames = [clone_graph(g).to(device) for g in rollout_graphs[-(history + 1) :]]
            input_graph = build_graph(input_graphs=frames, node_features=node_features).to(device)
            cur_graph = clone_graph(frames[-1]).to(device)
            prev_graph = clone_graph(frames[-2] if len(frames) > 1 else frames[-1]).to(device)
            if len(frames) > 1:
                cur_graph.vel_state = cur_graph.x[:, :pos_dim] - prev_graph.x[:, :pos_dim]
            model_inputs = model_inputs_cls(prev_graph, cur_graph, cur_graph, pos_dim)
            model_output = model(input_graph, is_training=False)
            predicted_graph = model.update(model_inputs, model_output)
            rollout_graphs.append(predicted_graph.cpu())
    return rollout_graphs
