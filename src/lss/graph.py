from __future__ import annotations

import torch
from torch_geometric.data import Data


def box_tensor(graph: Data, *, device=None, dtype=None) -> torch.Tensor | None:
    if hasattr(graph, "box_tensor") and isinstance(graph.box_tensor, torch.Tensor):
        out = graph.box_tensor
        if device is not None or dtype is not None:
            out = out.to(device=device or out.device, dtype=dtype or out.dtype)
        return out
    if not hasattr(graph, "box") or graph.box is None:
        return None
    box = graph.box
    if hasattr(box, "x") and hasattr(box, "y"):
        values = [float(box.x), float(box.y)]
    elif all(hasattr(box, key) for key in ("x1", "x2", "y1", "y2")):
        values = [float(box.x2) - float(box.x1), float(box.y2) - float(box.y1)]
    else:
        return None
    return torch.tensor(
        values,
        device=device or graph.x.device,
        dtype=dtype or graph.x.dtype,
    )


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
    if hasattr(graph, "pos") and isinstance(graph.pos, torch.Tensor):
        out.pos = graph.pos.clone()
    else:
        out.pos = out.x[:, :2].clone()
    if hasattr(graph, "box_tensor") and isinstance(graph.box_tensor, torch.Tensor):
        out.box_tensor = graph.box_tensor.clone()
    else:
        maybe_box = box_tensor(graph)
        if maybe_box is not None:
            out.box_tensor = maybe_box.clone()
    if hasattr(graph, "vel_state"):
        out.vel_state = graph.vel_state.clone()
    return out


def inverse_design_velocity_graph(input_graphs: list[Data]) -> Data:
    """Build the residual-velocity input used by GNNInverseDesign."""

    base_graph = input_graphs[-1]
    if len(input_graphs) == 1:
        x = torch.zeros_like(base_graph.x[:, :2])
    else:
        residual_velocities = []
        for i in range(len(input_graphs) - 1, 0, -1):
            target_graph = input_graphs[i]
            cur_graph = input_graphs[i - 1]
            cur_box = box_tensor(cur_graph, device=cur_graph.x.device, dtype=cur_graph.x.dtype)
            target_box = box_tensor(
                target_graph,
                device=target_graph.x.device,
                dtype=target_graph.x.dtype,
            )
            if cur_box is None or target_box is None:
                affine_velocity = torch.zeros_like(cur_graph.x[:, :2])
            else:
                strain = (cur_box[0] - target_box[0]) / cur_box[0].clamp_min(1e-12)
                affine_velocity_x = cur_graph.x[:, 0] * (1 - strain) - cur_graph.x[:, 0]
                affine_velocity = torch.column_stack(
                    [affine_velocity_x, torch.zeros_like(affine_velocity_x)]
                )
            global_velocity = target_graph.x[:, :2] - cur_graph.x[:, :2]
            residual_velocities.append(global_velocity - affine_velocity)
        x = torch.column_stack(residual_velocities)

    out = Data(
        x=x,
        pos=base_graph.x[:, :2].clone(),
        edge_index=base_graph.edge_index,
        edge_attr=base_graph.edge_attr,
        box=base_graph.box if hasattr(base_graph, "box") else None,
        t=getattr(base_graph, "t", None),
        batch=base_graph.batch.clone() if getattr(base_graph, "batch", None) is not None else torch.zeros(base_graph.x.size(0), dtype=torch.long, device=base_graph.x.device),
    )
    maybe_box = box_tensor(base_graph, device=base_graph.x.device, dtype=base_graph.x.dtype)
    if maybe_box is not None:
        out.box_tensor = maybe_box
    return out


def build_graph(input_graphs: list[Data], node_features: str = "positions") -> Data:
    """Build model input graph using positions, velocity history, or both."""
    if len(input_graphs) < 1:
        raise ValueError("input_graphs must contain at least one graph")

    base_graph = input_graphs[-1]
    match node_features:
        case "positions":
            return clone_graph(base_graph)
        case "inverse_design_velocity":
            return inverse_design_velocity_graph(input_graphs)
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
