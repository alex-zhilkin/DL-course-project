from __future__ import annotations

from copy import deepcopy

import torch
from graph_utils.box import Box
from torch_geometric.data import Data

from lss.data import (
    canonicalize_graph_edges,
    load_dataset,
    normalize_trajectory_to_reference_box,
    resolve_dataset_splits,
)
from lss.latent.simulation import batch_delta_graphs, set_reference_context_mode
from lss.latent.training import (
    decode_latent_positions,
    encode_frame_latent,
    fit_latent_step_stats,
)


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


def test_load_dataset_adds_runtime_two_hop_lj_edges_without_changing_file(
    tmp_path,
) -> None:
    edge_index = torch.tensor([[0, 1], [1, 2]])
    positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    vector = positions[edge_index[1]] - positions[edge_index[0]]
    graph = Data(
        x=positions,
        edge_index=edge_index,
        edge_attr=torch.column_stack(
            [vector, torch.linalg.vector_norm(vector, dim=-1), torch.ones(2)]
        ),
        box=Box(-3, 3, -3, 3, -0.1, 0.1),
    )
    path = tmp_path / "two_hop.pt"
    torch.save([[graph]], path)

    simulations = load_dataset(
        path,
        append_lj_indicator=True,
        add_lj_two_hop_edges=True,
    )
    augmented = simulations[0][0]

    assert augmented.edge_index.tolist() == [[0, 1, 0], [1, 2, 2]]
    assert augmented.edge_attr.shape == (3, 5)
    torch.testing.assert_close(augmented.edge_attr[:2, -1], torch.zeros(2))
    torch.testing.assert_close(augmented.edge_attr[2:, 3], torch.zeros(1))
    torch.testing.assert_close(augmented.edge_attr[2:, -1], torch.ones(1))
    assert augmented.lj_two_hop_edges_added == 1

    untouched = load_dataset(path)
    assert untouched[0][0].edge_index.size(1) == 2
    assert untouched[0][0].edge_attr.size(1) == 4

    batch = batch_delta_graphs(
        simulations,
        [(0, 0)],
        pos_dim=2,
        device="cpu",
        edge_mode="compact_stored",
    )
    assert batch["edge_attr"].shape == (3, 5)
    assert batch["ref_edge_attr"].shape == (3, 5)
    torch.testing.assert_close(batch["edge_attr"][:, -1], augmented.edge_attr[:, -1])
    torch.testing.assert_close(batch["ref_edge_attr"][:, -1], augmented.edge_attr[:, -1])


def test_load_dataset_applies_stiffness_exponent_before_normalization(tmp_path) -> None:
    graph = reciprocal_graph()
    graph.box = Box(-2, 2, -2, 2, -0.1, 0.1)
    path = tmp_path / "inverse_square.pt"
    torch.save([[graph]], path)

    simulations = load_dataset(
        path,
        coordinate_normalization="position_normalization",
        edge_stiffness_length_exponent=2,
    )

    normalized = simulations[0][0]
    assert normalized.edge_stiffness_length_exponent == 2
    torch.testing.assert_close(
        normalized.edge_attr[:, -1],
        torch.tensor([8.0, 12.0]),
    )


def test_reference_box_normalization_uses_one_fixed_trajectory_map() -> None:
    edge_index = torch.tensor([[0, 0], [1, 2]])

    def frame(x, box, stiffness):
        vector = x[edge_index[1]] - x[edge_index[0]]
        return Data(
            x=x,
            pos=x.clone(),
            edge_index=edge_index.clone(),
            edge_attr=torch.column_stack(
                [vector, torch.linalg.vector_norm(vector, dim=-1), stiffness]
            ),
            box=box,
        )

    reference = frame(
        torch.tensor([[-5.0, -2.0], [5.0, -2.0], [-5.0, 2.0]]),
        Box(-10, 10, -5, 5, -0.1, 0.1),
        torch.tensor([0.1, 0.25]),
    )
    current = frame(
        torch.tensor([[-4.0, -2.4], [4.0, -2.4], [-4.0, 2.4]]),
        Box(-8, 8, -6, 6, -0.1, 0.1),
        torch.tensor([0.1, 0.25]),
    )
    raw_simulation = deepcopy([reference, current])

    normalize_trajectory_to_reference_box([reference, current])

    length_scale = torch.sqrt(torch.tensor(10.0 * 5.0))
    torch.testing.assert_close(
        reference.x,
        torch.tensor([[-5.0, -2.0], [5.0, -2.0], [-5.0, 2.0]])
        / length_scale,
    )
    torch.testing.assert_close(
        current.x,
        torch.tensor([[-4.0, -2.4], [4.0, -2.4], [-4.0, 2.4]])
        / length_scale,
    )
    torch.testing.assert_close(
        reference.box_tensor, torch.tensor([20.0, 10.0]) / length_scale
    )
    torch.testing.assert_close(
        current.box_tensor, torch.tensor([16.0, 12.0]) / length_scale
    )
    torch.testing.assert_close(
        current.edge_attr[:, 2], torch.tensor([8.0, 4.8]) / length_scale
    )
    torch.testing.assert_close(
        current.edge_attr[:, 3], torch.tensor([0.1, 0.25]) * length_scale
    )
    torch.testing.assert_close(
        reference.edge_attr[:, 2] * reference.edge_attr[:, 3], torch.ones(2)
    )

    raw_batch = batch_delta_graphs(
        [raw_simulation],
        [(0, 1)],
        pos_dim=2,
        device="cpu",
        node_feature_mode="normalized_delta",
    )
    normalized_batch = batch_delta_graphs(
        [[reference, current]],
        [(0, 1)],
        pos_dim=2,
        device="cpu",
        node_feature_mode="normalized_delta",
    )
    # The physical reference channel is invariant, while the evolving edge
    # channel deliberately stays in normalized coordinates.
    for key in ("ref_pos", "ref_edge_attr", "node_feature", "normalized_delta"):
        torch.testing.assert_close(raw_batch[key], normalized_batch[key])
    assert not torch.allclose(raw_batch["edge_attr"], normalized_batch["edge_attr"])
    initial_batch = batch_delta_graphs(
        [[reference, current]],
        [(0, 0)],
        pos_dim=2,
        device="cpu",
        node_feature_mode="normalized_delta",
    )
    torch.testing.assert_close(
        initial_batch["edge_attr"], torch.zeros_like(initial_batch["edge_attr"])
    )


def test_reference_box_normalization_supports_inverse_square_stiffness() -> None:
    edge_index = torch.tensor([[0, 0], [1, 2]])
    positions = torch.tensor([[-5.0, -2.0], [5.0, -2.0], [-5.0, 2.0]])
    vector = positions[edge_index[1]] - positions[edge_index[0]]
    stiffness = torch.tensor([0.01, 0.0625])
    graph = Data(
        x=positions,
        edge_index=edge_index,
        edge_attr=torch.column_stack(
            [vector, torch.linalg.vector_norm(vector, dim=-1), stiffness]
        ),
        box=Box(-10, 10, -5, 5, -0.1, 0.1),
        edge_stiffness_length_exponent=2,
    )

    normalize_trajectory_to_reference_box([graph])

    length_scale = torch.sqrt(torch.tensor(10.0 * 5.0))
    torch.testing.assert_close(graph.edge_attr[:, 3], stiffness * length_scale**2)
    torch.testing.assert_close(
        graph.edge_attr[:, 3] * graph.edge_attr[:, 2].pow(2),
        torch.ones(2),
    )
    assert graph.edge_stiffness_length_exponent == 2

    batch = batch_delta_graphs([[graph]], [(0, 0)], pos_dim=2, device="cpu")
    torch.testing.assert_close(batch["ref_edge_attr"][:, -1], stiffness)


def test_compact_stored_edges_remove_only_exactly_redundant_channels() -> None:
    reference = reciprocal_graph()
    current = deepcopy(reference)
    current.x = current.x.clone()
    current.x[1, 0] += 0.1
    current.edge_attr = current.edge_attr.clone()
    current.edge_attr[:, :2] = current.x[current.edge_index[1]] - current.x[current.edge_index[0]]
    current.edge_attr[:, 2] = torch.linalg.vector_norm(
        current.edge_attr[:, :2], dim=-1
    )

    batch = batch_delta_graphs(
        [[reference, current]],
        [(0, 1)],
        pos_dim=2,
        device="cpu",
        edge_mode="compact_stored",
    )

    assert batch["edge_attr"].shape[1] == 4
    assert batch["ref_edge_attr"].shape[1] == 4
    expected_vector_change = current.edge_attr[:, :2] - reference.edge_attr[:, :2]
    expected_stretch = current.edge_attr[:, 2:3] - reference.edge_attr[:, 2:3]
    torch.testing.assert_close(batch["edge_attr"][:, :2], expected_vector_change)
    torch.testing.assert_close(batch["edge_attr"][:, 2:3], expected_stretch)
    torch.testing.assert_close(
        batch["edge_attr"][:, 3:4],
        expected_stretch / reference.edge_attr[:, 2:3],
    )
    torch.testing.assert_close(
        batch["ref_edge_attr"],
        reference.edge_attr[:, [0, 1, 2, 3]],
    )


def test_single_frame_encoder_supports_compact_stored_edges() -> None:
    reference = reciprocal_graph()
    current = deepcopy(reference)
    current.x = current.x + torch.tensor([0.05, -0.02])

    class Encoder:
        edge_mode = "compact_stored"

        def encode(self, node, ref_pos, edge, ref_edge, edge_index, batch):
            assert edge.shape[1] == 4
            assert ref_edge.shape[1] == 4
            return torch.zeros((1, 2)), None

    normalizers = {
        "node_feature_mean": torch.zeros(2),
        "node_feature_std": torch.ones(2),
        "edge_mean": torch.zeros(4),
        "edge_std": torch.ones(4),
        "ref_edge_mean": torch.zeros(4),
        "ref_edge_std": torch.ones(4),
    }
    latent = encode_frame_latent(
        Encoder(),
        [reference, current],
        1,
        pos_dim=2,
        node_feature_mode="normalized_delta",
        normalizers=normalizers,
        device="cpu",
    )
    assert latent.shape == (2,)

    stats = fit_latent_step_stats(
        Encoder(),
        [[reference, current]],
        [(0, 0, 1)],
        batch_graphs=1,
        pos_dim=2,
        node_feature_mode="normalized_delta",
        normalizers=normalizers,
        device="cpu",
    )
    assert stats.z_mean.shape[-1] == 2


def test_physical_reference_context_does_not_change_decoded_coordinate_system() -> None:
    class ZeroDisplacementDecoder(torch.nn.Module):
        edge_mode = "stored"

        def encode_reference_graph(self, ref_pos, ref_edge_attr, edge_index):
            # The static encoder really does see reconstructed physical positions.
            assert float(ref_pos.abs().max()) > 2.0
            return torch.zeros((ref_pos.size(0), 4), dtype=ref_pos.dtype)

        def decode(self, z, h0, batch):
            return torch.zeros((h0.size(0), 2), dtype=h0.dtype)

    graph = reciprocal_graph()
    graph.x = graph.x / 5.0
    graph.pos = graph.x.clone()
    graph.reference_length_scale = 5.0
    graph.reference_box_center = torch.tensor([3.0, -2.0])
    graph.edge_attr[:, :3] = graph.edge_attr[:, :3] / 5.0
    graph.edge_attr[:, 3] = graph.edge_attr[:, 3] * 5.0
    expected = graph.x.clone()
    normalizers = {
        "target_mean": torch.zeros(2),
        "target_std": torch.ones(2),
        "edge_mean": torch.zeros(13),
        "edge_std": torch.ones(13),
        "ref_edge_mean": torch.zeros(13),
        "ref_edge_std": torch.ones(13),
    }

    decoded = decode_latent_positions(
        ZeroDisplacementDecoder(),
        [[graph][0]],
        torch.zeros(2),
        0,
        pos_dim=2,
        ae_target_mode="normalized_delta",
        normalizers=normalizers,
        device="cpu",
    )
    torch.testing.assert_close(decoded, expected)


def test_normalized_reference_context_does_not_reconstruct_physical_scale() -> None:
    graph = reciprocal_graph()
    graph.x = graph.x / 5.0
    graph.pos = graph.x.clone()
    graph.reference_length_scale = 5.0
    graph.reference_box_center = torch.tensor([3.0, -2.0])
    graph.edge_attr[:, :3] = graph.edge_attr[:, :3] / 5.0
    graph.edge_attr[:, 3] = graph.edge_attr[:, 3] * 5.0
    simulation = [graph]
    set_reference_context_mode([simulation], "normalized")

    batch = batch_delta_graphs(
        [simulation], [(0, 0)], pos_dim=2, device="cpu"
    )
    torch.testing.assert_close(batch["ref_pos"], graph.x)
    assert graph.reference_context_mode == "normalized"
    assert float(batch["ref_pos"].abs().max()) <= 0.200001


def test_recomputed_stored_edges_ignore_current_serialized_geometry() -> None:
    reference = Data(
        x=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.tensor([[1.0, 0.0, 1.0, 2.0]]),
    )
    current_a = Data(
        x=torch.tensor([[0.0, 0.0], [1.2, 0.1]]),
        edge_index=reference.edge_index.clone(),
        edge_attr=torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
    )
    current_b = Data(
        x=current_a.x.clone(),
        edge_index=reference.edge_index.clone(),
        edge_attr=torch.tensor([[-7.0, -8.0, -9.0, -10.0]]),
    )

    batch_a = batch_delta_graphs(
        [[reference, current_a]],
        [(0, 1)],
        pos_dim=2,
        device="cpu",
        edge_mode="recomputed_stored",
    )
    batch_b = batch_delta_graphs(
        [[reference, current_b]],
        [(0, 1)],
        pos_dim=2,
        device="cpu",
        edge_mode="recomputed_stored",
    )

    torch.testing.assert_close(batch_a["edge_attr"], batch_b["edge_attr"])
    assert batch_a["edge_attr"][0, -1].item() == 2.0


def test_mixture_source_seed_matches_individual_source_split(tmp_path) -> None:
    paths = []
    for source_offset in (0.0, 100.0):
        trajectories = []
        for trajectory_index in range(8):
            graph = reciprocal_graph()
            graph.x = graph.x + source_offset + trajectory_index
            trajectories.append([graph])
        path = tmp_path / f"source_{int(source_offset)}.pt"
        torch.save(trajectories, path)
        paths.append(path)

    mixture = [
        {
            "name": f"source_{index}",
            "path": str(path),
            "train_count": 2,
            "val_count": 2,
            "split_seed": 41 + index,
        }
        for index, path in enumerate(paths)
    ]
    mixed_train, mixed_val, _, _ = resolve_dataset_splits(
        paths[0],
        train_count=0,
        val_count=0,
        dataset_mixture=mixture,
        split_seed=999,
        shuffle_within_source=True,
    )

    for source_index, path in enumerate(paths):
        individual_train, individual_val, _, _ = resolve_dataset_splits(
            path,
            train_count=2,
            val_count=2,
            split_seed=41 + source_index,
            shuffle_within_source=True,
        )
        mixed_source_train = [
            sim
            for sim in mixed_train
            if sim[0].source_name == f"source_{source_index}"
        ]
        mixed_source_val = [
            sim
            for sim in mixed_val
            if sim[0].source_name == f"source_{source_index}"
        ]
        assert [sim[0].x.tolist() for sim in mixed_source_train] == [
            sim[0].x.tolist() for sim in individual_train
        ]
        assert [sim[0].x.tolist() for sim in mixed_source_val] == [
            sim[0].x.tolist() for sim in individual_val
        ]
