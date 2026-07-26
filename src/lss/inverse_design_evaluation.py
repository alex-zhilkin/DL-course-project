from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from .graph import box_tensor, clone_graph, inverse_design_velocity_graph
from .inverse_design_barostat import (
    estimate_initial_box_vel_y,
    estimate_initial_box_vel_y_accurate,
    update_box_y_thermodynamic,
)
from .models.base import BaseModelInputs


def prepare_inverse_design_graph(graph: Data) -> Data:
    out = clone_graph(graph)
    out.pos = out.x[:, :2].clone()
    maybe_box = box_tensor(out, device=out.x.device, dtype=out.x.dtype)
    if maybe_box is None:
        raise AttributeError("GNNInverseDesign rollout requires box dimensions.")
    out.box_tensor = maybe_box
    return out


def prepare_inverse_design_trajectory(sim: list[Data]) -> list[Data]:
    return [prepare_inverse_design_graph(graph) for graph in sim]


def recalculate_edge_attr(graph: Data) -> Tensor:
    row, col = graph.edge_index
    delta = graph.pos[row] - graph.pos[col]
    box = graph.box_tensor.view(1, 2)
    delta = delta - torch.round(delta / box) * box
    distances = torch.norm(delta, dim=1)
    stiffness = graph.edge_attr[:, -1]
    return torch.column_stack([delta, distances, stiffness])


def compute_total_stress(
    graph: Data,
    *,
    r0: Tensor,
    temperature: float = 1e-7,
) -> Tensor:
    kb_metal = 8.6173303e-5
    row, col = graph.edge_index
    delta = graph.pos[row] - graph.pos[col]
    box = graph.box_tensor.view(1, 2)
    delta = delta - torch.round(delta / box) * box
    distance = torch.norm(delta, dim=1) + 1e-12
    stiffness = graph.edge_attr[:, -1]
    force_magnitude = -2.0 * stiffness * (distance - r0.to(distance.device))
    ratio = force_magnitude / distance
    area = graph.box_tensor[0] * graph.box_tensor[1]
    p_xx = 0.5 * torch.sum(ratio * delta[:, 0].square()) / area
    p_yy = 0.5 * torch.sum(ratio * delta[:, 1].square()) / area
    kinetic = graph.num_nodes * kb_metal * temperature / area
    return torch.stack([p_xx + kinetic, p_yy + kinetic])


def box_p_ratio(trajectory: list[Data], last_index: int = -1) -> Tensor:
    initial = trajectory[0].box_tensor
    final = trajectory[last_index].box_tensor
    strain_x = (final[0] - initial[0]) / initial[0]
    strain_y = (final[1] - initial[1]) / initial[1]
    return -strain_y / (strain_x + 1e-8)


def inverse_design_rollout(
    input_graphs: list[Data],
    model,
    *,
    num_steps: int,
    history: int,
    barostat_config: dict,
    device: str,
) -> list[Data]:
    """Port of GNNInverseDesign/utils.py:get_rollout."""

    rollout = [prepare_inverse_design_graph(graph).cpu() for graph in input_graphs]
    r0 = rollout[0].edge_attr[:, -2].clone()
    stride_dt = float(barostat_config["default_skip"]) * float(barostat_config["dt"])
    box_compression_factor = (
        rollout[-1].box_tensor[0] / rollout[-2].box_tensor[0]
    )
    if history >= 2:
        current_box_vel_y = estimate_initial_box_vel_y_accurate(
            rollout[-3].to(device),
            rollout[-2].to(device),
            rollout[-1].to(device),
            stride_dt,
        )
    else:
        current_box_vel_y = estimate_initial_box_vel_y(
            rollout[-2].to(device),
            rollout[-1].to(device),
            stride_dt,
        )

    particle_count = rollout[0].num_nodes
    piston_mass = (
        float(barostat_config["C_coupling"])
        * particle_count
        * stride_dt**2
    )
    damping = (
        float(barostat_config["damping"])
        * particle_count
        * stride_dt
    )

    model.eval()
    with torch.no_grad():
        for _ in range(int(num_steps) + 1):
            window = [
                graph.to(device)
                for graph in rollout[-(int(history) + 1) :]
            ]
            input_graph = inverse_design_velocity_graph(window).to(device)
            inputs = BaseModelInputs(window[-2], window[-1], window[-1], 2)
            output = model(input_graph, is_training=False)
            predicted = model.update(inputs, output)

            new_lx = predicted.box_tensor[0] * box_compression_factor
            new_ly, current_box_vel_y = update_box_y_thermodynamic(
                positions=predicted.pos,
                edge_index=inputs.cur_graph.edge_index,
                edge_attr=inputs.cur_graph.edge_attr,
                current_box=inputs.cur_graph.box_tensor,
                r0=r0.to(predicted.pos.device),
                box_vel_y=current_box_vel_y,
                W_y=piston_mass,
                damping=damping,
                stride_dt=stride_dt,
                target_pressure=float(barostat_config["target_pressure"]),
                temperature=float(barostat_config["temperature"]),
            )
            predicted.box_tensor = torch.stack([new_lx, new_ly])
            predicted.edge_attr = recalculate_edge_attr(predicted)
            rollout.append(predicted.detach().cpu())
    return rollout
