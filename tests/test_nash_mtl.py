from __future__ import annotations

import torch

from lss.latent.training import (
    _nash_mtl_combined_gradients,
    _solve_nash_mtl_coefficients,
)


def test_nash_solver_recovers_orthogonal_inverse_norm_weights() -> None:
    gram = torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64))

    alpha, residual, projections = _solve_nash_mtl_coefficients(gram)

    torch.testing.assert_close(
        alpha,
        torch.tensor([1.0, 0.5, 1.0 / 3.0], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-7,
    )
    assert residual < 1e-6
    assert torch.all(projections > 0)


def test_nash_solver_satisfies_conflicting_feasible_system() -> None:
    gram = torch.tensor(
        [[1.0, -0.2, 0.1], [-0.2, 1.0, 0.15], [0.1, 0.15, 1.0]],
        dtype=torch.float64,
    )

    alpha, residual, projections = _solve_nash_mtl_coefficients(gram)

    assert residual < 1e-6
    assert torch.all(alpha > 0)
    assert torch.all(projections > 0)
    torch.testing.assert_close(gram @ alpha, alpha.reciprocal(), rtol=1e-6, atol=1e-7)


def test_nash_solver_handles_recorded_strongly_conflicting_batch() -> None:
    # Regression for a real AE batch on which SLSQP abandoned a feasible
    # convex subproblem with "Positive directional derivative for linesearch".
    gram = torch.tensor(
        [
            [0.3853, -0.2696, -0.6007],
            [-0.2696, 0.4573, 0.5668],
            [-0.6007, 0.5668, 1.6194],
        ],
        dtype=torch.float64,
    )

    alpha, residual, projections = _solve_nash_mtl_coefficients(gram)

    assert residual < 1e-6
    assert torch.all(alpha > 0)
    assert torch.all(projections > 0)
    torch.testing.assert_close(gram @ alpha, alpha.reciprocal(), rtol=1e-6, atol=1e-7)


def test_nash_direction_is_invariant_to_task_gradient_scaling() -> None:
    generator = torch.Generator().manual_seed(808)
    gradients = torch.randn(3, 32, generator=generator, dtype=torch.float64)
    alpha, _, _ = _solve_nash_mtl_coefficients(gradients @ gradients.T)
    direction = alpha @ gradients

    scaled_gradients = gradients.clone()
    scaled_gradients[0] *= 100.0
    scaled_alpha, _, _ = _solve_nash_mtl_coefficients(
        scaled_gradients @ scaled_gradients.T
    )
    scaled_direction = scaled_alpha @ scaled_gradients

    cosine = torch.nn.functional.cosine_similarity(
        direction.reshape(1, -1), scaled_direction.reshape(1, -1)
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=1e-7, atol=1e-7)


def test_nash_solver_rejects_no_common_descent_direction() -> None:
    gram = torch.tensor(
        [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    try:
        _solve_nash_mtl_coefficients(gram)
    except RuntimeError:
        return
    raise AssertionError("Expected an infeasible Nash batch to be rejected.")


def test_nash_solver_accepts_positive_warm_start() -> None:
    gram = torch.tensor(
        [[1.0, -0.3, 0.2], [-0.3, 1.0, 0.1], [0.2, 0.1, 1.0]],
        dtype=torch.float64,
    )
    warm_start = torch.tensor([4.0, 0.25, 2.0], dtype=torch.float64)

    alpha, residual, _ = _solve_nash_mtl_coefficients(
        gram, initial_alpha=warm_start
    )

    assert residual < 1e-6
    assert torch.all(alpha > 0)
    torch.testing.assert_close(gram @ alpha, alpha.reciprocal(), rtol=1e-6, atol=1e-7)


def test_nash_failure_uses_normalized_gradient_average() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    losses = {
        "first": parameter[0],
        "second": -parameter[0],
        "third": parameter[1],
    }

    combined, _, residual, _, cosines, used_fallback, solved_alpha = (
        _nash_mtl_combined_gradients(losses, [parameter])
    )

    assert used_fallback
    assert solved_alpha is None
    assert torch.isnan(torch.tensor(residual))
    assert cosines["first__second"] == -1.0
    torch.testing.assert_close(combined[0], torch.tensor([0.0, 1.0 / 3.0]))
