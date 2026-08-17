from __future__ import annotations

import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Literal
import warnings

import torch
from graph_utils.box import Box

EdgeMultiplicity = Literal[1, 2]

POSITION_NORMALIZATION = "position_normalization"


def normalize_trajectory_to_reference_box(
    simulation: list,
    *,
    pos_dim: int = 2,
) -> list:
    """Express a trajectory in one isotropically normalized length unit.

    The scalar unit is the geometric mean of the initial box half-width and
    half-height. The map is fixed at frame zero and applied to every frame,
    preserving aspect ratio and angles. Geometry, stiffness, velocities, and
    interaction length parameters are converted using that same unit.
    ``edge_stiffness_length_exponent`` selects the source convention
    ``k = material_factor / length**exponent`` and defaults to one.
    """

    if not simulation:
        return simulation
    if int(pos_dim) != 2:
        raise ValueError("Reference-box normalization currently requires pos_dim=2.")
    if all(
        str(getattr(graph, "coordinate_normalization", "")).strip().lower()
        == POSITION_NORMALIZATION
        for graph in simulation
    ):
        return simulation

    reference = simulation[0]
    reference_box = getattr(reference, "box", None)
    if reference_box is not None and all(
        hasattr(reference_box, key) for key in ("x1", "x2", "y1", "y2")
    ):
        lower = torch.tensor(
            [float(reference_box.x1), float(reference_box.y1)],
            dtype=reference.x.dtype,
            device=reference.x.device,
        )
        upper = torch.tensor(
            [float(reference_box.x2), float(reference_box.y2)],
            dtype=reference.x.dtype,
            device=reference.x.device,
        )
    else:
        reference_pos = reference.x[:, :2]
        lower = reference_pos.amin(dim=0)
        upper = reference_pos.amax(dim=0)
    half_extent = ((upper - lower) / 2).clamp_min(1e-8)
    center = (upper + lower) / 2
    reference_edge_attr = getattr(reference, "edge_attr", None)
    stiffness_length_exponent = int(
        getattr(reference, "edge_stiffness_length_exponent", 1)
    )
    if stiffness_length_exponent < 0:
        raise ValueError("edge_stiffness_length_exponent must be non-negative.")
    edge_material_factor = None
    if (
        isinstance(reference_edge_attr, torch.Tensor)
        and reference_edge_attr.ndim == 2
        and reference_edge_attr.size(1) >= 4
    ):
        # Preserve the material factor for k=w/l or k=w/l**2 source data.
        # Existing datasets default to exponent one; Real Reid stores two.
        edge_material_factor = (
            reference_edge_attr[:, 2].pow(stiffness_length_exponent)
            * reference_edge_attr[:, -1]
        ).clone()
    normalized_reference_stiffness = None
    length_scale = torch.sqrt(half_extent.prod()).clamp_min(1e-8)

    for graph in simulation:
        graph.x = graph.x.clone()
        graph.x[:, :2] = (graph.x[:, :2] - center.to(graph.x)) / length_scale.to(graph.x)
        if hasattr(graph, "pos") and isinstance(graph.pos, torch.Tensor):
            graph.pos = graph.pos.clone()
            graph.pos[:, :2] = (
                graph.pos[:, :2] - center.to(graph.pos)
            ) / length_scale.to(graph.pos)
        if hasattr(graph, "vel_state") and isinstance(graph.vel_state, torch.Tensor):
            graph.vel_state = graph.vel_state.clone()
            graph.vel_state[:, :2] = (
                graph.vel_state[:, :2] / length_scale.to(graph.vel_state)
            )

        current_box = getattr(graph, "box", None)
        if current_box is not None and all(
            hasattr(current_box, key) for key in ("x1", "x2", "y1", "y2")
        ):
            x1 = (float(current_box.x1) - float(center[0])) / float(length_scale)
            x2 = (float(current_box.x2) - float(center[0])) / float(length_scale)
            y1 = (float(current_box.y1) - float(center[1])) / float(length_scale)
            y2 = (float(current_box.y2) - float(center[1])) / float(length_scale)
            z1 = float(getattr(current_box, "z1", -0.1))
            z2 = float(getattr(current_box, "z2", 0.1))
            graph.box = Box(x1, x2, y1, y2, z1, z2)
            normalized_box = torch.tensor(
                [abs(x2 - x1), abs(y2 - y1)],
                dtype=graph.x.dtype,
                device=graph.x.device,
            )
        else:
            normalized_box = torch.full(
                (2,), 2.0, dtype=graph.x.dtype, device=graph.x.device
            )
        graph.box_tensor = normalized_box

        edge_attr = getattr(graph, "edge_attr", None)
        edge_index = getattr(graph, "edge_index", None)
        if (
            isinstance(edge_attr, torch.Tensor)
            and edge_attr.ndim == 2
            and edge_attr.size(1) >= 3
            and isinstance(edge_index, torch.Tensor)
        ):
            source, target = edge_index.long()
            vector = graph.x[target, :2] - graph.x[source, :2]
            vector = vector - torch.round(
                vector / normalized_box.reshape(1, 2)
            ) * normalized_box.reshape(1, 2)
            graph.edge_attr = edge_attr.clone()
            graph.edge_attr[:, :2] = vector.to(graph.edge_attr)
            graph.edge_attr[:, 2] = torch.linalg.vector_norm(
                vector, dim=-1
            ).to(graph.edge_attr)
            if edge_material_factor is not None:
                if graph.edge_attr.size(0) != edge_material_factor.numel():
                    raise ValueError(
                        "Every frame must retain the reference edge ordering for "
                        "stiffness normalization."
                    )
                if normalized_reference_stiffness is None:
                    normalized_reference_stiffness = (
                        edge_material_factor.to(graph.edge_attr)
                        / graph.edge_attr[:, 2]
                        .clamp_min(1e-8)
                        .pow(stiffness_length_exponent)
                    )
                graph.edge_attr[:, -1] = normalized_reference_stiffness

        # LJ sigma and cutoff are lengths, so convert them with the same unit.
        for attribute in ("lj_sigma", "lj_cutoff"):
            value = getattr(graph, attribute, None)
            if isinstance(value, (int, float)):
                setattr(graph, attribute, float(value) / float(length_scale))

        graph.coordinate_normalization = POSITION_NORMALIZATION
        graph.reference_box_center = center.detach().cpu().clone()
        graph.reference_box_half_extent = half_extent.detach().cpu().clone()
        graph.reference_length_scale = length_scale.detach().cpu().clone()
        graph.edge_stiffness_length_exponent = stiffness_length_exponent
        if graph is reference and edge_material_factor is not None:
            graph.edge_material_factor = edge_material_factor.detach().cpu().clone()
    return simulation


def normalize_dataset_coordinates(
    simulations: list,
    *,
    mode: str | None,
    pos_dim: int = 2,
) -> list:
    """Apply an explicitly requested coordinate convention in-place."""

    normalized_mode = str(mode or "none").strip().lower()
    if normalized_mode in {"", "none", "raw", "physical"}:
        return simulations
    if normalized_mode != POSITION_NORMALIZATION:
        raise ValueError(f"Unknown coordinate_normalization: {mode}")
    trajectories = (
        [simulations]
        if simulations and hasattr(simulations[0], "edge_index")
        else simulations
    )
    for simulation in trajectories:
        normalize_trajectory_to_reference_box(simulation, pos_dim=pos_dim)
    return simulations


def _install_legacy_auxetic_box_alias() -> None:
    """Map legacy MetaForge pickles to the shared graph_utils Box class."""
    auxetic_module = sys.modules.setdefault("auxetic", ModuleType("auxetic"))
    network_module = sys.modules.setdefault("auxetic.network", ModuleType("auxetic.network"))
    network_module.Box = Box
    auxetic_module.network = network_module
    # Some standalone trajectory generators serialized the same Box type as
    # ``network.Box`` rather than ``auxetic.network.Box``.
    legacy_network_module = sys.modules.setdefault("network", ModuleType("network"))
    legacy_network_module.Box = Box


def canonicalize_graph_edges(
    graph,
    *,
    edge_multiplicity: EdgeMultiplicity = 1,
    edge_vector_dim: int = 2,
):
    """Normalize a graph to one or two entries per undirected node pair.

    ``edge_multiplicity=1`` stores one canonical ``i < j`` edge. With
    ``edge_multiplicity=2``, the canonical edge is followed by its reverse
    orientation. Reciprocal/repeated input entries are coalesced before either
    representation is produced.

    The project's edge convention stores the oriented relative-position vector
    in the first ``edge_vector_dim`` columns of ``edge_attr``. Those columns are
    sign-corrected when an edge orientation is reversed; scalar features such
    as length and stiffness are unchanged.
    """
    if int(edge_multiplicity) not in (1, 2):
        raise ValueError("edge_multiplicity must be 1 or 2")
    edge_multiplicity = int(edge_multiplicity)
    edge_vector_dim = int(edge_vector_dim)
    if edge_vector_dim < 0:
        raise ValueError("edge_vector_dim must be non-negative")
    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        return graph

    edge_index = graph.edge_index.long()
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"Expected edge_index shape [2, E], found {tuple(edge_index.shape)}"
        )
    original_edges = int(edge_index.size(1))
    if original_edges == 0:
        graph.edge_multiplicity = edge_multiplicity
        graph.edges_are_undirected = True
        return graph

    source, target = edge_index
    first = torch.minimum(source, target)
    second = torch.maximum(source, target)
    keep = first != second
    first, second = first[keep], second[keep]
    source = source[keep]
    num_nodes = int(graph.x.size(0)) if hasattr(graph, "x") else int(edge_index.max()) + 1
    keys = first * num_nodes + second
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    canonical_first = torch.div(unique_keys, num_nodes, rounding_mode="floor")
    canonical_second = unique_keys.remainder(num_nodes)
    canonical_index = torch.stack([canonical_first, canonical_second], dim=0)

    edge_attr = getattr(graph, "edge_attr", None)
    canonical_attr = None
    if isinstance(edge_attr, torch.Tensor):
        if edge_attr.size(0) != original_edges:
            raise ValueError(
                "edge_attr must have one row per input edge; "
                f"found {edge_attr.size(0)} rows for {original_edges} edges"
            )
        original_rank = edge_attr.ndim
        if original_rank == 1:
            edge_attr = edge_attr.reshape(-1, 1)
        elif original_rank != 2:
            raise ValueError(
                f"Expected edge_attr rank 1 or 2, found shape {tuple(edge_attr.shape)}"
            )
        oriented_attr = edge_attr[keep].clone()
        vector_width = min(edge_vector_dim, oriented_attr.size(1))
        flipped = source != first
        if vector_width and flipped.any():
            oriented_attr[flipped, :vector_width] *= -1
        if not oriented_attr.is_floating_point():
            oriented_attr = oriented_attr.float()
        canonical_attr = torch.zeros(
            (unique_keys.numel(), oriented_attr.size(1)),
            device=oriented_attr.device,
            dtype=oriented_attr.dtype,
        )
        counts = torch.zeros(
            (unique_keys.numel(), 1),
            device=oriented_attr.device,
            dtype=oriented_attr.dtype,
        )
        canonical_attr.index_add_(0, inverse, oriented_attr)
        counts.index_add_(
            0,
            inverse,
            torch.ones(
                (oriented_attr.size(0), 1),
                device=oriented_attr.device,
                dtype=oriented_attr.dtype,
            ),
        )
        canonical_attr = canonical_attr / counts.clamp_min(1)
        if original_rank == 1:
            canonical_attr = canonical_attr.reshape(-1)

    if edge_multiplicity == 1:
        graph.edge_index = canonical_index
        if canonical_attr is not None:
            graph.edge_attr = canonical_attr
    else:
        graph.edge_index = torch.cat([canonical_index, canonical_index.flip(0)], dim=1)
        if canonical_attr is not None:
            reverse_attr = canonical_attr.clone()
            if reverse_attr.ndim == 1:
                # A one-dimensional edge attribute is scalar, so it is
                # orientation independent.
                pass
            else:
                vector_width = min(edge_vector_dim, reverse_attr.size(1))
                if vector_width:
                    reverse_attr[:, :vector_width] *= -1
            graph.edge_attr = torch.cat([canonical_attr, reverse_attr], dim=0)

    graph.edge_multiplicity = edge_multiplicity
    graph.edges_are_undirected = True
    return graph


def normalize_dataset_edges(
    simulations: list,
    *,
    edge_multiplicity: EdgeMultiplicity = 1,
    edge_vector_dim: int = 2,
) -> list:
    """Apply the shared edge representation to every trajectory frame.

    Both a full ``list[trajectory]`` dataset and a single ``list[frame]``
    trajectory are accepted and returned with the same outer structure.
    """
    if not simulations:
        return simulations
    trajectories = (
        [simulations]
        if hasattr(simulations[0], "edge_index")
        else simulations
    )
    for simulation in trajectories:
        for graph in simulation:
            canonicalize_graph_edges(
                graph,
                edge_multiplicity=edge_multiplicity,
                edge_vector_dim=edge_vector_dim,
            )
    return simulations


def load_dataset(
    path: str | Path,
    *,
    edge_multiplicity: EdgeMultiplicity = 1,
    edge_vector_dim: int = 2,
    map_location: str | torch.device = "cpu",
    coordinate_normalization: str | None = None,
    pos_dim: int = 2,
    edge_stiffness_length_exponent: int | None = None,
) -> list:
    """Load a trajectory dataset using the project-wide edge convention.

    The default is one canonical undirected edge per node pair. Callers that
    explicitly require both orientations can request ``edge_multiplicity=2``.
    The source file is never modified.
    """
    _install_legacy_auxetic_box_alias()
    simulations = list(
        torch.load(
            Path(path),
            weights_only=False,
            map_location=map_location,
        )
    )
    if edge_stiffness_length_exponent is not None:
        exponent = int(edge_stiffness_length_exponent)
        if exponent < 0:
            raise ValueError("edge_stiffness_length_exponent must be non-negative.")
        for simulation in simulations:
            for graph in simulation:
                graph.edge_stiffness_length_exponent = exponent
    normalize_dataset_edges(
        simulations,
        edge_multiplicity=edge_multiplicity,
        edge_vector_dim=edge_vector_dim,
    )
    return normalize_dataset_coordinates(
        simulations,
        mode=coordinate_normalization,
        pos_dim=pos_dim,
    )


def simulation_temperature(sim, default: float = float("nan")) -> float:
    """Return optional simulation temperature metadata as a finite float."""

    if not sim:
        return float(default)
    value = getattr(sim[0], "temperature", default)
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return float(default)
    return temperature if math.isfinite(temperature) else float(default)


def tag_simulation_source(sim, source_name: str):
    """Attach source metadata to each frame in a simulation."""

    for frame in sim:
        frame.source_name = str(source_name)
    return sim


def split_dataset(
    path: str | Path,
    *,
    train_count: int,
    val_count: int,
    edge_multiplicity: EdgeMultiplicity = 1,
    edge_vector_dim: int = 2,
) -> tuple[list, list, list]:
    sims = load_dataset(
        path,
        edge_multiplicity=edge_multiplicity,
        edge_vector_dim=edge_vector_dim,
    )
    train_count = int(train_count)
    val_count = int(val_count)
    train_data = sims[:train_count]
    val_data = sims[train_count : train_count + val_count]
    test_data = sims[train_count + val_count :]
    return train_data, val_data, test_data


def resolve_dataset_splits(
    dataset_path: str | Path,
    *,
    train_count: int,
    val_count: int,
    dataset_mixture: list[dict] | None = None,
    split_seed: int | None = None,
    shuffle_within_source: bool = False,
    stratify_temperature: bool = False,
    mix_holdout_across_sources: bool = False,
    edge_multiplicity: EdgeMultiplicity = 1,
    edge_vector_dim: int = 2,
    coordinate_normalization: str | None = None,
    pos_dim: int = 2,
    edge_stiffness_length_exponent: int | None = None,
):
    if not dataset_mixture:
        sims = load_dataset(
            dataset_path,
            edge_multiplicity=edge_multiplicity,
            edge_vector_dim=edge_vector_dim,
            coordinate_normalization=coordinate_normalization,
            pos_dim=pos_dim,
            edge_stiffness_length_exponent=edge_stiffness_length_exponent,
        )
        if stratify_temperature:
            temperatures = [simulation_temperature(sim) for sim in sims]
            if not temperatures or not all(math.isfinite(value) for value in temperatures):
                warnings.warn(
                    "Temperature stratification was requested, but this dataset has no "
                    "complete temperature metadata. Using a normal shuffled split.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                stratify_temperature = False
        if stratify_temperature:
            generator = torch.Generator()
            generator.manual_seed(0 if split_seed is None else int(split_seed))
            groups: dict[float, list] = {}
            for sim in sims:
                temperature = simulation_temperature(sim)
                groups.setdefault(temperature, []).append(sim)
            for temperature, group in groups.items():
                order = torch.randperm(len(group), generator=generator).tolist()
                groups[temperature] = [group[idx] for idx in order]

            temperatures = sorted(groups)

            def take_balanced(count: int) -> list:
                selected = []
                while len(selected) < int(count):
                    active = [temperature for temperature in temperatures if groups[temperature]]
                    if not active:
                        raise ValueError(
                            f"Requested {count} stratified samples but the dataset was exhausted."
                        )
                    order = torch.randperm(len(active), generator=generator).tolist()
                    for idx in order:
                        if len(selected) >= int(count):
                            break
                        selected.append(groups[active[idx]].pop())
                return selected

            train_data = take_balanced(train_count)
            val_data = take_balanced(val_count)
            test_data = [sim for temperature in temperatures for sim in groups[temperature]]
            if test_data:
                order = torch.randperm(len(test_data), generator=generator).tolist()
                test_data = [test_data[idx] for idx in order]

            def counts(split) -> dict[str, int]:
                return {
                    f"temperature_{temperature:g}": sum(
                        simulation_temperature(sim) == temperature
                        for sim in split
                    )
                    for temperature in temperatures
                }

            split_info = [
                {
                    "source": Path(dataset_path).stem,
                    "path": str(dataset_path),
                    "total": len(sims),
                    "train": len(train_data),
                    "val": len(val_data),
                    "test": len(test_data),
                    "stratified_by": "temperature",
                    **{f"train_{key}": value for key, value in counts(train_data).items()},
                    **{f"val_{key}": value for key, value in counts(val_data).items()},
                    **{f"test_{key}": value for key, value in counts(test_data).items()},
                }
            ]
            return train_data, val_data, test_data, split_info
        if shuffle_within_source:
            generator = torch.Generator()
            generator.manual_seed(0 if split_seed is None else int(split_seed))
            order = torch.randperm(len(sims), generator=generator).tolist()
            sims = [sims[i] for i in order]
        train_count = int(train_count)
        val_count = int(val_count)
        train_data = sims[:train_count]
        val_data = sims[train_count : train_count + val_count]
        test_data = sims[train_count + val_count :]
        split_info = [
            {
                "source": Path(dataset_path).stem,
                "path": str(dataset_path),
                "total": len(train_data) + len(val_data) + len(test_data),
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
            }
        ]
        return train_data, val_data, test_data, split_info

    generator = None
    if shuffle_within_source:
        generator = torch.Generator()
        generator.manual_seed(0 if split_seed is None else int(split_seed))

    train_data: list = []
    val_data: list = []
    test_data: list = []
    split_info: list[dict] = []
    pooled_holdout: list[tuple[str, object]] = []

    for source_idx, spec in enumerate(dataset_mixture):
        path = Path(spec["path"])
        source_name = str(spec.get("name", f"source_{source_idx + 1}"))
        sims = load_dataset(
            path,
            edge_multiplicity=spec.get("edge_multiplicity", edge_multiplicity),
            edge_vector_dim=int(spec.get("edge_vector_dim", edge_vector_dim)),
            coordinate_normalization=spec.get(
                "coordinate_normalization", coordinate_normalization
            ),
            pos_dim=int(spec.get("pos_dim", pos_dim)),
            edge_stiffness_length_exponent=spec.get(
                "edge_stiffness_length_exponent",
                edge_stiffness_length_exponent,
            ),
        )
        sims = [tag_simulation_source(sim, source_name) for sim in sims]
        if generator is not None:
            source_generator = generator
            if spec.get("split_seed") is not None:
                source_generator = torch.Generator()
                source_generator.manual_seed(int(spec["split_seed"]))
            order = torch.randperm(
                len(sims), generator=source_generator
            ).tolist()
            sims = [sims[i] for i in order]
        src_train = int(spec["train_count"])
        holdout_train_count = int(spec.get("holdout_train_count", src_train))
        if src_train > holdout_train_count:
            raise ValueError(
                "train_count cannot exceed holdout_train_count for a dataset source."
            )
        src_train_data = sims[:src_train]
        train_data.extend(src_train_data)
        if mix_holdout_across_sources:
            for sim in sims[holdout_train_count:]:
                pooled_holdout.append((source_name, sim))
            src_val_data = []
            src_test_data = []
        else:
            src_val = int(spec["val_count"])
            src_val_data = sims[
                holdout_train_count : holdout_train_count + src_val
            ]
            src_test_data = sims[holdout_train_count + src_val :]
            val_data.extend(src_val_data)
            test_data.extend(src_test_data)
        split_info.append(
            {
                "source": source_name,
                "path": str(path),
                "total": len(sims),
                "train": len(src_train_data),
                "reserved_train_pool": holdout_train_count,
                "val": len(src_val_data),
                "test": len(src_test_data),
            }
        )

    if mix_holdout_across_sources:
        if generator is not None:
            order = torch.randperm(len(pooled_holdout), generator=generator).tolist()
            pooled_holdout = [pooled_holdout[i] for i in order]
        holdout_sims = [sim for _, sim in pooled_holdout]
        val_data = holdout_sims[: int(val_count)]
        test_data = holdout_sims[int(val_count) :]
        val_source_counts: dict[str, int] = {}
        test_source_counts: dict[str, int] = {}
        for source_name, _sim in pooled_holdout[: int(val_count)]:
            val_source_counts[source_name] = val_source_counts.get(source_name, 0) + 1
        for source_name, _sim in pooled_holdout[int(val_count) :]:
            test_source_counts[source_name] = test_source_counts.get(source_name, 0) + 1
        for row in split_info:
            row["val"] = int(val_source_counts.get(row["source"], 0))
            row["test"] = int(test_source_counts.get(row["source"], 0))

    return train_data, val_data, test_data, split_info
