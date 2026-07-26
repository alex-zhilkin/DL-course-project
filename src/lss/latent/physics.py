"""Differentiable mechanical-energy losses for latent trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..graph import box_tensor


@dataclass(frozen=True)
class PhysicsLossConfig:
    """Weights and fallbacks for the elastic-network implicit-Euler loss."""

    lambda_phys: float = 0.0
    lambda_mse: float = 1.0
    inertial_weight: float = 1.0
    spring_weight: float = 1.0
    external_weight: float = 1.0
    boundary_weight: float = 1.0
    box_weight: float = 0.0
    spring_strain_margin: float = 0.0
    default_mass: float = 1.0
    dt: float | None = 1.0
    normalize_by_speed: bool = False
    speed_epsilon: float = 1e-3
    latent_noise_std: float = 0.0


def _minimum_image(delta: torch.Tensor, box: torch.Tensor | None) -> torch.Tensor:
    if box is None:
        return delta
    box = box.to(device=delta.device, dtype=delta.dtype).reshape(1, -1)
    return delta - box * torch.round(delta / box.clamp_min(1e-12))


def _node_value(graph, names, *, count: int, dim: int | None, default: float,
                device, dtype) -> torch.Tensor:
    for name in names:
        value = getattr(graph, name, None)
        if isinstance(value, torch.Tensor):
            value = value.to(device=device, dtype=dtype)
            if dim is None:
                return value.reshape(count)
            return value.reshape(count, dim)
    shape = (count,) if dim is None else (count, dim)
    return torch.full(shape, float(default), device=device, dtype=dtype)


def _first_tensor(graph, names) -> torch.Tensor | None:
    for name in names:
        value = getattr(graph, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _box_bounds(graph, *, dim: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor] | None:
    box = getattr(graph, "box", None)
    if box is None:
        return None
    if dim < 2 or not all(hasattr(box, key) for key in ("x1", "x2", "y1", "y2")):
        return None
    low = torch.tensor([float(box.x1), float(box.y1)], device=device, dtype=dtype)
    high = torch.tensor([float(box.x2), float(box.y2)], device=device, dtype=dtype)
    if dim > 2:
        if not all(hasattr(box, key) for key in ("z1", "z2")):
            return None
        low = torch.cat([low, torch.tensor([float(box.z1)], device=device, dtype=dtype)])
        high = torch.cat([high, torch.tensor([float(box.z2)], device=device, dtype=dtype)])
    return low[:dim], high[:dim]


def elastic_implicit_euler_energy(
    x_pred: torch.Tensor,
    x_prev: torch.Tensor,
    x_prev_prev: torch.Tensor,
    *,
    reference_graph,
    target_graph,
    config: PhysicsLossConfig,
) -> dict[str, torch.Tensor]:
    """Return size-normalized implicit-Euler energy components.

    The stored elastic graphs use directed duplicate edges, so each physical
    spring is counted once. Rest length and stiffness are the final two
    reference edge attributes. Optional ``mass``, ``external_force``/``force``,
    ``boundary_mask`` and ``boundary_target`` graph fields are honored when
    present; the current dePablo data only provide the elastic edge fields.
    """

    device, dtype = x_pred.device, x_pred.dtype
    count, dim = x_pred.shape
    dt = float(config.dt if config.dt is not None else 1.0)
    masses = _node_value(
        target_graph, ("mass", "masses", "node_mass"), count=count, dim=None,
        default=config.default_mass, device=device, dtype=dtype,
    )
    free_flight = 2.0 * x_prev - x_prev_prev
    inertial = 0.5 * masses * ((x_pred - free_flight) ** 2).sum(dim=-1) / max(dt * dt, 1e-12)
    inertial = inertial.mean()

    edge_index = reference_graph.edge_index.to(device).long()
    # dePablo stores both i->j and j->i. This mask gives one term per spring.
    unique = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, unique]
    ref_edge = reference_graph.edge_attr.to(device=device, dtype=dtype)[unique]
    source, target = edge_index[0], edge_index[1]
    delta = x_pred[target] - x_pred[source]
    box = box_tensor(target_graph, device=device, dtype=dtype)
    current_length = torch.linalg.vector_norm(_minimum_image(delta, box), dim=-1)
    rest_length = ref_edge[:, -2]
    stiffness = ref_edge[:, -1]
    stretch = (current_length - rest_length).abs()
    if config.spring_strain_margin > 0:
        allowed = float(config.spring_strain_margin) * rest_length.clamp_min(1e-12)
        stretch = (stretch - allowed).clamp_min(0.0)
    spring = (0.5 * stiffness * stretch.square()).mean()

    forces = _node_value(
        target_graph, ("external_force", "external_forces", "force", "forces"),
        count=count, dim=dim, default=0.0, device=device, dtype=dtype,
    )
    external = -(forces * x_pred).sum(dim=-1).mean()

    boundary = torch.zeros((), device=device, dtype=dtype)
    mask = _first_tensor(
        target_graph,
        (
            "boundary_mask",
            "driven_mask",
            "pinned_mask",
            "fixed_mask",
            "kinematic_mask",
            "dirichlet_mask",
        ),
    )
    target_position = _first_tensor(
        target_graph,
        (
            "boundary_target",
            "driven_target",
            "target_position",
            "target_pos",
            "target_x",
            "prescribed_position",
            "prescribed_pos",
        ),
    )
    if isinstance(mask, torch.Tensor):
        mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
        if mask.any():
            if isinstance(target_position, torch.Tensor):
                target_position = target_position.to(device=device, dtype=dtype).reshape(count, dim)
            else:
                target_position = target_graph.x[:, :dim].to(device=device, dtype=dtype)
            boundary = ((x_pred[mask] - target_position[mask]) ** 2).sum(dim=-1).mean()

    box_violation = torch.zeros((), device=device, dtype=dtype)
    bounds = _box_bounds(target_graph, dim=dim, device=device, dtype=dtype)
    if bounds is not None:
        low, high = bounds
        below = (low.reshape(1, -1) - x_pred).clamp_min(0.0)
        above = (x_pred - high.reshape(1, -1)).clamp_min(0.0)
        box_violation = (below.square() + above.square()).sum(dim=-1).mean()

    physical = (
        config.inertial_weight * inertial
        + config.spring_weight * spring
        + config.external_weight * external
        + config.boundary_weight * boundary
        + config.box_weight * box_violation
    )
    speed = torch.linalg.vector_norm(x_prev - x_prev_prev, dim=-1).mean() / max(dt, 1e-12)
    if config.normalize_by_speed:
        physical = physical / (speed.detach() + float(config.speed_epsilon))
    return {
        "physical": physical,
        "inertial": inertial,
        "spring": spring,
        "external": external,
        "boundary": boundary,
        "box": box_violation,
        "previous_speed": speed,
    }


__all__ = ["PhysicsLossConfig", "elastic_implicit_euler_energy"]
