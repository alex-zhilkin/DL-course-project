from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

from lss.latent.direct_autoencoder_simulator import (
    _direct_target_tensor,
    predict_next_graph,
)


def _graph(positions: list[list[float]]) -> Data:
    return Data(
        x=torch.tensor(positions, dtype=torch.float32),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0, 0.0, 1.0, 2.0]]),
        # PyG drops attributes assigned as None, while clone_graph expects the
        # real datasets' box attribute to be present.
        box=torch.tensor([10.0, 10.0]),
    )


def test_normalized_step_target_is_t_plus_one_minus_t() -> None:
    input_batch = {"normalized_delta": torch.tensor([[0.1, -0.2]])}
    target_batch = {"normalized_delta": torch.tensor([[0.4, 0.3]])}

    target = _direct_target_tensor(
        input_batch, target_batch, "normalized_step_delta"
    )

    torch.testing.assert_close(target, torch.tensor([[0.3, 0.5]]))


class _CaptureZeroStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_node_features = None

    def forward(
        self,
        node_features,
        ref_pos,
        edge_attr,
        ref_edge_attr,
        edge_index,
        batch,
    ):
        self.seen_node_features = node_features.detach().clone()
        return torch.zeros_like(ref_pos), torch.zeros((1, 2))


def test_rollout_uses_previous_frame_velocity_and_integrates_step() -> None:
    reference = _graph([[0.0, 0.0], [2.0, 1.0]])
    previous = _graph([[0.1, 0.0], [2.1, 1.0]])
    current = _graph([[0.3, 0.0], [2.3, 1.0]])
    model = _CaptureZeroStep()
    result = {
        "model": model,
        "config": {
            "pos_dim": 2,
            "node_feature_mode": "normalized_delta_velocity",
            "target_mode": "normalized_step_delta",
            "edge_mode": "recomputed_stored",
        },
        "normalizers": {
            "node_feature_mean": torch.zeros((1, 4)),
            "node_feature_std": torch.ones((1, 4)),
            "edge_mean": torch.zeros((1, 13)),
            "edge_std": torch.ones((1, 13)),
            "target_mean": torch.zeros((1, 2)),
            "target_std": torch.ones((1, 2)),
        },
    }

    predicted = predict_next_graph(
        result,
        reference,
        current,
        previous_graph=previous,
        device="cpu",
    )

    # Zero predicted increment leaves the current state unchanged.
    torch.testing.assert_close(predicted.x[:, :2], current.x[:, :2])
    # Reference range is [2, 1], so the observed velocity is [0.2/2, 0/1].
    torch.testing.assert_close(
        model.seen_node_features[:, 2:],
        torch.tensor([[0.1, 0.0], [0.1, 0.0]]),
    )
