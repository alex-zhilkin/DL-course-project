from __future__ import annotations

import torch
from torch_geometric.data import Data

from lss.latent.experiment import ground_truth_p_ratio, temperature_p_ratio


def _linear_strain_trajectory(*, p_ratio: float = 0.3, frames: int = 8):
    reference = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    trajectory = []
    for step in range(frames):
        driven_strain = 0.01 * step
        scale = torch.tensor(
            [1.0 + driven_strain, 1.0 - p_ratio * driven_strain]
        )
        trajectory.append(
            Data(
                x=reference * scale,
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 4)),
                box={"x1": -2.0, "x2": 2.0, "y1": -2.0, "y2": 2.0},
                box_tensor=torch.tensor([4.0, 4.0]),
            )
        )
    return trajectory


def test_strain_gated_trajectory_estimator_recovers_linear_p_ratio() -> None:
    trajectory = _linear_strain_trajectory()
    cfg = {
        "p_ratio_estimator": "strain_gated_trajectory",
        "p_ratio_min_fit_frames": 4,
        "p_ratio_min_driven_strain_range": 1e-4,
    }

    estimated = temperature_p_ratio(trajectory, cfg=cfg)
    ground_truth = ground_truth_p_ratio(
        trajectory, dataset_name="reid", cfg=cfg
    )

    assert abs(estimated - 0.3) < 1e-5
    assert abs(ground_truth - 0.3) < 1e-5
