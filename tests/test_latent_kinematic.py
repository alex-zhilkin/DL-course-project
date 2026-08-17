import torch

from lss.latent.models import NodeDeltaMLPAutoEncoder, make_latent_propagator
from lss.latent.training import (
    LatentNormalizer,
    latent_step_fixed_history,
    latent_step_kinematic,
)


def test_one_undirected_edge_matches_reciprocal_endpoint_aggregation():
    model = NodeDeltaMLPAutoEncoder(
        pos_dim=2,
        edge_dim=4,
        hidden_size=8,
        latent_dim=4,
        latent_tokens=4,
    )
    one_index = torch.tensor([[0], [1]])
    one_attr = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    reciprocal_index = torch.tensor([[0, 1], [1, 0]])
    reciprocal_attr = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, 3.0, 4.0]]
    )

    one = model.aggregate_edges(one_attr, one_index, num_nodes=2)
    reciprocal = model.aggregate_edges(
        reciprocal_attr, reciprocal_index, num_nodes=2
    )

    torch.testing.assert_close(one, reciprocal)


def test_kinematic_attention_context_is_differentiable():
    latent_dim, context_dim = 4, 12
    model = make_latent_propagator(
        latent_dim,
        16,
        model_type="kinematic_mlp",
        context_dim=context_dim,
        graph_context_dim=8,
        context_pool_mode="learned_attention",
    )
    stats = LatentNormalizer(
        z_mean=torch.zeros(1, latent_dim),
        z_std=torch.ones(1, latent_dim),
        dz_mean=torch.zeros(1, latent_dim),
        dz_std=torch.ones(1, latent_dim),
    )
    z0 = torch.zeros(latent_dim)
    context = torch.randn(7, context_dim)

    predicted = latent_step_kinematic(
        model,
        z0,
        z0,
        z0,
        stats,
        progress=0.1,
        context=context,
    )
    assert predicted.shape == (latent_dim,)
    assert torch.isfinite(predicted).all()

    predicted.square().sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.context_projection.parameters()
    )


def test_fixed_history_step_has_no_progress_input():
    latent_dim = 4
    model = make_latent_propagator(
        latent_dim,
        16,
        model_type="fixed_history_mlp",
    )
    stats = LatentNormalizer(
        z_mean=torch.zeros(1, latent_dim),
        z_std=torch.ones(1, latent_dim),
        dz_mean=torch.zeros(1, latent_dim),
        dz_std=torch.ones(1, latent_dim),
    )

    predicted = latent_step_fixed_history(
        model,
        torch.randn(latent_dim),
        torch.randn(latent_dim),
        torch.randn(latent_dim),
        stats,
    )

    assert predicted.shape == (latent_dim,)
    assert model.net[0].in_features == 3 * latent_dim


def test_fixed_velocity_residual_starts_from_observed_velocity():
    latent_dim = 2
    model = make_latent_propagator(
        latent_dim,
        8,
        model_type="fixed_velocity_residual_mlp",
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    stats = LatentNormalizer(
        z_mean=torch.zeros(1, latent_dim),
        z_std=torch.ones(1, latent_dim),
        dz_mean=torch.zeros(1, latent_dim),
        dz_std=torch.ones(1, latent_dim),
    )
    z1 = torch.tensor([1.0, 2.0])
    z5 = torch.tensor([5.0, -2.0])
    current = torch.tensor([7.0, 3.0])

    predicted = latent_step_fixed_history(
        model,
        current,
        z1,
        z5,
        stats,
        observed_frame_gap=4,
    )

    torch.testing.assert_close(predicted, current + (z5 - z1) / 4)
