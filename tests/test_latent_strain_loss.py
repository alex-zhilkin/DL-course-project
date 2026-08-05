import torch

from lss.latent.training import (
    endpoint_p_ratios_from_strains,
    side_strains_from_positions,
)


def test_side_strains_recover_anisotropic_affine_deformation():
    reference = torch.tensor(
        [[-1.0, -2.0], [-1.0, 2.0], [1.0, -2.0], [1.0, 2.0]]
    )
    expected = torch.tensor([0.10, -0.20])
    current = reference * (1.0 + expected)
    batch = torch.zeros(4, dtype=torch.long)

    actual = side_strains_from_positions(
        current, reference, batch, boundary_fraction=0.25
    )

    torch.testing.assert_close(actual, expected.reshape(1, 2))


def test_side_strains_are_differentiable():
    reference = torch.tensor(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    current = reference.clone().requires_grad_(True)
    batch = torch.zeros(4, dtype=torch.long)

    loss = side_strains_from_positions(
        current, reference, batch, boundary_fraction=0.25
    ).square().sum()
    loss.backward()

    assert current.grad is not None
    assert torch.isfinite(current.grad).all()


def test_endpoint_p_ratio_uses_larger_strain_as_driven_axis():
    strains = torch.tensor([[0.10, -0.03], [-0.02, 0.08], [1e-5, 2e-5]])

    ratio, valid = endpoint_p_ratios_from_strains(
        strains, minimum_driven_strain=1e-3
    )

    torch.testing.assert_close(ratio[:2], torch.tensor([0.30, 0.25]))
    assert valid.tolist() == [True, True, False]
