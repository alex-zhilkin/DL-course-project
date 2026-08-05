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
    return normalize_dataset_edges(
        simulations,
        edge_multiplicity=edge_multiplicity,
        edge_vector_dim=edge_vector_dim,
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
):
    if not dataset_mixture:
        sims = load_dataset(
            dataset_path,
            edge_multiplicity=edge_multiplicity,
            edge_vector_dim=edge_vector_dim,
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
        )
        sims = [tag_simulation_source(sim, source_name) for sim in sims]
        if generator is not None:
            order = torch.randperm(len(sims), generator=generator).tolist()
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
