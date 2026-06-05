from __future__ import annotations

from pathlib import Path
import torch


def load_dataset(path: str | Path) -> list:
    return list(torch.load(Path(path), weights_only=False))


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
    mix_holdout_across_sources: bool = False,
):
    if not dataset_mixture:
        sims = load_dataset(dataset_path)
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
