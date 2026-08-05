from __future__ import annotations

import torch
from graph_utils.box import Box
from torch_geometric.data import Data

from scripts.internal_09_transformer_search import make_inputs
from lss.models.simple_edge_mlp_simulator import (
    SimpleUndirectedEdgeMLPSimulator,
)


def graph(box_half_width: float) -> Data:
    return Data(
        x=torch.tensor([[-0.9, 0.0], [0.9, 0.0], [0.0, 0.5]]),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        edge_attr=torch.tensor(
            [[1.8, 0.0, 1.8, 1.0], [0.9, 0.5, 1.03, 1.0]]
        ),
        box=Box(
            -box_half_width,
            box_half_width,
            -box_half_width,
            box_half_width,
            -0.1,
            0.1,
        ),
    )


def inputs(sim, *, static_structure_only: bool, ignore_box: bool = False):
    return make_inputs(
        sim,
        0,
        sim[1],
        sim[0],
        target_mean=torch.zeros(1, 2),
        target_std=torch.ones(1, 2),
        accel_mean=torch.zeros(1, 2),
        accel_std=torch.ones(1, 2),
        edge_mean=torch.zeros(13),
        edge_std=torch.ones(13),
        length_scale=1.0,
        distance_floor=None,
        velocity_skip=False,
        dual_kinematic=False,
        boundary_features=False,
        boundary_weight=1.0,
        pratio_loss_weight=0.0,
        node_count_feature=False,
        target_mode="increment",
        edge_mode="stored",
        undirected_edges=True,
        device="cpu",
        with_target=False,
        static_structure_only=static_structure_only,
        ignore_box=ignore_box,
    )


def test_static_structure_mode_ignores_future_box() -> None:
    reference = graph(1.0)
    future_small = graph(1.0)
    future_large = graph(2.0)

    edge_small = inputs(
        [reference, future_small], static_structure_only=True
    )[1]
    edge_large = inputs(
        [reference, future_large], static_structure_only=True
    )[1]

    torch.testing.assert_close(edge_small, edge_large)


def test_legacy_mode_exposes_future_box_difference() -> None:
    reference = graph(1.0)
    future_small = graph(1.0)
    future_large = graph(2.0)

    edge_small = inputs(
        [reference, future_small], static_structure_only=False
    )[1]
    edge_large = inputs(
        [reference, future_large], static_structure_only=False
    )[1]

    assert not torch.allclose(edge_small, edge_large)


def test_ignore_box_mode_is_independent_of_reference_box() -> None:
    small = [graph(1.0), graph(1.0)]
    large = [graph(2.0), graph(2.0)]

    edge_small = inputs(
        small, static_structure_only=True, ignore_box=True
    )[1]
    edge_large = inputs(
        large, static_structure_only=True, ignore_box=True
    )[1]

    torch.testing.assert_close(edge_small, edge_large)


def test_simple_edge_mlp_supports_variable_graph_sizes() -> None:
    model = SimpleUndirectedEdgeMLPSimulator(
        node_dim=6, edge_dim=13, hidden_size=16
    )
    for node_count, edge_index in (
        (3, torch.tensor([[0, 0], [1, 2]])),
        (5, torch.tensor([[0, 0, 1, 3], [1, 2, 4, 4]])),
    ):
        output = model(
            torch.randn(node_count, 6),
            torch.randn(edge_index.size(1), 13),
            edge_index,
        )
        assert output.shape == (node_count, 2)
        assert torch.isfinite(output).all()
