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


def build_graph(input_graphs: list[Data], node_features: str) -> Data:
    if len(input_graphs) < 1:
        raise ValueError("no graphs provided")
    if not all(isinstance(obj, Data) for obj in input_graphs):
        raise TypeError("all input_graphs must be torch_geometric.data.Data")

    base_graph = input_graphs[-1]

    if node_features == "positions":
        data = clone_graph(base_graph)
        if not hasattr(data, "t"):
            data.t = None
        if hasattr(base_graph, "vel_state"):
            data.vel_state = base_graph.vel_state
        return data
    if node_features == "velocity":
        if len(input_graphs) == 1:
            velocity = base_graph.x.new_zeros(base_graph.x.shape)
        else:
            chunks = [
                input_graphs[i].x - input_graphs[i - 1].x
                for i in range(len(input_graphs) - 1, 0, -1)
            ]
            velocity = chunks[0] if len(chunks) == 1 else torch.column_stack(chunks)
        data = Data(
            x=velocity,
            edge_index=base_graph.edge_index,
            edge_attr=base_graph.edge_attr,
            box=base_graph.box if hasattr(base_graph, "box") else None,
            t=base_graph.t if hasattr(base_graph, "t") else None,
            dtype=base_graph.x.dtype,
        )
        if hasattr(base_graph, "vel_state"):
            data.vel_state = base_graph.vel_state
        return data
    if node_features == "combined":
        if len(input_graphs) == 1:
            velocity = base_graph.x.new_zeros(base_graph.x.shape)
            x = torch.column_stack([velocity, base_graph.x])
        else:
            chunks = [
                input_graphs[i].x - input_graphs[i - 1].x
                for i in range(len(input_graphs) - 1, 0, -1)
            ]
            velocity = chunks[0] if len(chunks) == 1 else torch.column_stack(chunks)
            x = torch.column_stack([velocity, base_graph.x])
        data = Data(
            x=x,
            edge_index=base_graph.edge_index,
            edge_attr=base_graph.edge_attr,
            box=base_graph.box if hasattr(base_graph, "box") else None,
            t=base_graph.t if hasattr(base_graph, "t") else None,
            dtype=base_graph.x.dtype,
        )
        if hasattr(base_graph, "vel_state"):
            data.vel_state = base_graph.vel_state
        return data
    raise ValueError("node_features must be one of: velocity, positions, combined")


def rollout(
    model,
    input_graphs: list[Data],
    num_steps: int,
    history: int,
    pos_dim: int,
    device: str,
    node_features: str,
    model_inputs_cls,
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
            model_inputs = model_inputs_cls(prev_graph, cur_graph, None, pos_dim)
            model_output = model(input_graph, is_training=False)
            predicted_graph = model.update(model_inputs, model_output)
            rollout_graphs.append(predicted_graph.cpu())
    return rollout_graphs
