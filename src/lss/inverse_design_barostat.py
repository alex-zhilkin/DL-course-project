from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data


def update_box_y_thermodynamic(
    positions: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    current_box: Tensor,
    r0: Tensor,
    box_vel_y: Tensor | float,
    W_y: float,
    damping: float,
    stride_dt: float,
    target_pressure: float = 0.0,
    temperature: float = 1e-7,
) -> tuple[Tensor, Tensor]:
    """Differentiable y-box update from GNNInverseDesign/barostat_utils.py."""

    kb_metal = 8.6173303e-5
    row, col = edge_index
    lx, ly = current_box[0], current_box[1]
    volume = lx * ly
    num_particles = positions.shape[0]

    dy = positions[row, 1] - positions[col, 1]
    dy = dy - ly * torch.round(dy / ly)
    dx = positions[row, 0] - positions[col, 0]
    dx = dx - lx * torch.round(dx / lx)

    dist = torch.norm(torch.stack([dx, dy], dim=1), dim=1)
    stiffness = edge_attr[:, -1]
    force_mag = -2.0 * stiffness * (dist - r0.to(positions.device))
    virial_term = force_mag * (dy**2) / (dist + 1e-12)
    virial_sum_y = 0.5 * torch.sum(virial_term)

    kinetic_pressure_val = (
        0.5 * num_particles * kb_metal * temperature
    ) / (volume + 1e-12)
    virial_pressure_val = virial_sum_y / (volume + 1e-12)
    p_yy_total = virial_pressure_val + kinetic_pressure_val

    box_vel_y = torch.as_tensor(
        box_vel_y,
        dtype=positions.dtype,
        device=positions.device,
    )
    driving_force = (p_yy_total - target_pressure) * lx
    total_force = driving_force - damping * box_vel_y
    box_acc = total_force / W_y
    new_box_vel_y = box_vel_y + box_acc * stride_dt
    new_ly = ly * torch.exp(new_box_vel_y * stride_dt)
    return new_ly, new_box_vel_y


def estimate_initial_box_vel_y(prev_graph: Data, curr_graph: Data, stride_dt: float) -> Tensor:
    return torch.log(curr_graph.box_tensor[1] / prev_graph.box_tensor[1]) / stride_dt


def estimate_initial_box_vel_y_accurate(
    prev_prev_graph: Data,
    prev_graph: Data,
    curr_graph: Data,
    stride_dt: float,
) -> Tensor:
    log_c = torch.log(curr_graph.box_tensor[1])
    log_p = torch.log(prev_graph.box_tensor[1])
    log_pp = torch.log(prev_prev_graph.box_tensor[1])
    return (3 * log_c - 4 * log_p + log_pp) / (2 * stride_dt)


STIFF_OPTIMIZED_BAROSTAT = {
    "dt": 0.01,
    "default_skip": 200,
    "C_coupling": 0.5608636158379727,
    "damping": 0.10083014689462362,
    "temperature": 1e-7,
    "target_pressure": 0.0,
}


NODE_OPTIMIZED_BAROSTAT = {
    "dt": 0.01,
    "default_skip": 200,
    "C_coupling": 1.1761735924144685,
    "damping": 0.28052825855550206,
    "temperature": 1e-7,
    "target_pressure": 0.0,
}
