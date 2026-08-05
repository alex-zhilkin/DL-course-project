import torch

from lss.models.attention_pyramid_simulator import AttentionPyramidSimulator


def test_attention_pyramid_exact_four_dimensional_bottleneck():
    nodes = 9
    edges = torch.combinations(torch.arange(nodes), r=2).T
    model = AttentionPyramidSimulator(
        node_dim=7,
        edge_dim=13,
        hidden_size=32,
        pyramid_tokens=(8, 4),
        heads=4,
        latent_dim=4,
    )
    prediction = model(
        torch.randn(nodes, 7),
        torch.randn(edges.size(1), 13),
        edges,
        attention_bias=torch.randn(edges.size(1)),
    )
    assert prediction.shape == (nodes, 2)
    prediction.square().mean().backward()
    assert model.latent_down[-1].out_features == 4


def test_attention_pyramid_rejects_directed_duplicate_edges():
    model = AttentionPyramidSimulator(
        node_dim=7,
        edge_dim=13,
        hidden_size=32,
        pyramid_tokens=(4, 2),
        heads=4,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]])
    try:
        model(
            torch.randn(2, 7),
            torch.randn(2, 13),
            edge_index,
        )
    except ValueError as error:
        assert "canonical" in str(error)
    else:
        raise AssertionError("directed duplicate edges should be rejected")
