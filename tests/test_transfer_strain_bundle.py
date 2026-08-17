import copy

import numpy as np
import torch
from torch_geometric.data import Data

from lss.latent.transfer_strain_bundle import (
    DirectionalStrainTransferBundle,
    directional_strain_latents,
)


class Box:
    x1, x2, y1, y2 = -1.0, 1.0, -1.0, 1.0


def trajectory(rate_x=-0.002, rate_y=0.01, frames=20):
    grid = torch.tensor(
        [[x, y] for x in (-1.0, -0.5, 0.5, 1.0) for y in (-1.0, -0.5, 0.5, 1.0)]
    )
    out = []
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    for index in range(frames):
        graph = Data(
            x=grid * torch.tensor([1 + rate_x * index, 1 + rate_y * index]),
            edge_index=edge_index,
            edge_attr=torch.ones(edge_index.size(1), 1),
        )
        graph.box = Box()
        out.append(graph)
    return out


def test_directional_strain_roundtrip_and_progress_free_rollout():
    source = [trajectory(-0.002 - 0.0001 * index) for index in range(4)]
    target = trajectory(-0.003)
    bundle = DirectionalStrainTransferBundle.fit(source, max_transitions_per_sim=10)
    rollout = bundle.rollout(target)
    direct = bundle.direct_autoencoder_trajectory(target)
    assert np.allclose(directional_strain_latents(direct), directional_strain_latents(target), atol=1e-6)
    predicted = directional_strain_latents(rollout)
    assert np.allclose(predicted[5], directional_strain_latents(target)[5], atol=1e-7)
    assert np.allclose(predicted[-1], directional_strain_latents(target)[-1], atol=5e-3)
    saved = copy.deepcopy(bundle.state_dict())
    assert saved["weights"].shape == (7, 2)
