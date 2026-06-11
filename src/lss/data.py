from __future__ import annotations

import math
from pathlib import Path
import sys
from types import ModuleType
import warnings

import torch
from graph_utils.box import Box


def _install_legacy_auxetic_box_alias() -> None:
    """Map legacy MetaForge pickles to the shared graph_utils Box class."""
    auxetic_module = sys.modules.setdefault("auxetic", ModuleType("auxetic"))
    network_module = sys.modules.setdefault("auxetic.network", ModuleType("auxetic.network"))
    network_module.Box = Box
    auxetic_module.network = network_module


def load_dataset(path: str | Path) -> list:
    _install_legacy_auxetic_box_alias()
    return list(torch.load(Path(path), weights_only=False))


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


def split_dataset(path: str | Path, *, train_count: int, val_count: int) -> tuple[list, list, list]:
    sims = load_dataset(path)
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
):
    if not dataset_mixture:
        sims = load_dataset(dataset_path)
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
        sims = load_dataset(path)
        if generator is not None:
            order = torch.randperm(len(sims), generator=generator).tolist()
            sims = [sims[i] for i in order]
        src_train = int(spec["train_count"])
        src_train_data = sims[:src_train]
        train_data.extend(src_train_data)
        if mix_holdout_across_sources:
            for sim in sims[src_train:]:
                pooled_holdout.append((source_name, sim))
            src_val_data = []
            src_test_data = []
        else:
            src_val = int(spec["val_count"])
            src_val_data = sims[src_train : src_train + src_val]
            src_test_data = sims[src_train + src_val :]
            val_data.extend(src_val_data)
            test_data.extend(src_test_data)
        split_info.append(
            {
                "source": source_name,
                "path": str(path),
                "total": len(sims),
                "train": len(src_train_data),
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
