from __future__ import annotations

import torch
from torch_geometric.data import Data

from lss.data import canonicalize_graph_edges, load_dataset


def reciprocal_graph() -> Data:
    return Data(
        x=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 2], [1, 0, 0]]),
        edge_attr=torch.tensor(
            [
                [1.0, 0.0, 1.0, 2.0],
                [-1.0, 0.0, 1.0, 2.0],
                [0.0, -1.0, 1.0, 3.0],
            ]
        ),
    )


def test_one_edge_per_pair_canonicalizes_and_coalesces() -> None:
    graph = canonicalize_graph_edges(reciprocal_graph(), edge_multiplicity=1)

    assert graph.edge_index.tolist() == [[0, 0], [1, 2]]
    torch.testing.assert_close(
        graph.edge_attr,
        torch.tensor([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 3.0]]),
    )
    assert graph.edge_multiplicity == 1
    assert graph.edges_are_undirected is True


def test_two_edges_per_pair_reverses_only_vector_features() -> None:
    graph = canonicalize_graph_edges(reciprocal_graph(), edge_multiplicity=2)

    assert graph.edge_index.tolist() == [[0, 0, 1, 2], [1, 2, 0, 0]]
    torch.testing.assert_close(graph.edge_attr[2:, :2], -graph.edge_attr[:2, :2])
    torch.testing.assert_close(graph.edge_attr[2:, 2:], graph.edge_attr[:2, 2:])
    assert graph.edge_multiplicity == 2


def test_load_dataset_accepts_single_trajectory(tmp_path) -> None:
    path = tmp_path / "trajectory.pt"
    torch.save([reciprocal_graph(), reciprocal_graph()], path)

    trajectory = load_dataset(path)

    assert len(trajectory) == 2
    assert trajectory[0].edge_index.size(1) == 2
