"""Freeze a conservative topology-grouped validation screen from inventory metadata."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "notebooks/results/latent_matched_study"


def main():
    inventory_path = DIRECTORY / "dataset_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    groups = defaultdict(list)
    for source, data in inventory["sources"].items():
        for trajectory in data["trajectories"]:
            groups[trajectory["topology_sha256"]].append((source, trajectory))
    rng = random.Random(123)
    order = sorted(groups)
    rng.shuffle(order)
    used = set()
    splits = {source: {"train": [], "val": [], "test": []} for source in inventory["sources"]}
    for partition, count in (("train", 30), ("val", 20)):
        temperatures = sorted({t["metadata"]["temperature"] for t in inventory["sources"]["depablo_mixed_temp"]["trajectories"]})
        quotas = {temp: count // len(temperatures) + (i < count % len(temperatures))
                  for i, temp in enumerate(temperatures)}
        for group in order:
            if group in used:
                continue
            candidates = list(groups[group])
            rng.shuffle(candidates)
            selected_sources = set()
            for source, t in candidates:
                if source in selected_sources or len(splits[source][partition]) >= count:
                    continue
                temperature = t["metadata"]["temperature"]
                if source == "depablo_mixed_temp" and quotas[temperature] == 0:
                    continue
                splits[source][partition].append(t["index"])
                selected_sources.add(source)
                if source == "depablo_mixed_temp":
                    quotas[temperature] -= 1
            if selected_sources:
                used.add(group)
        if any(len(s[partition]) != count for s in splits.values()):
            raise ValueError(f"Insufficient disjoint groups for {partition}: {splits}")
    for group in order:
        if group not in used:
            for source, t in groups[group]:
                splits[source]["test"].append(t["index"])
    partition_groups = {name: set() for name in ("train", "val", "test")}
    for source, parts in splits.items():
        trajectories = inventory["sources"][source]["trajectories"]
        for name, indices in parts.items():
            partition_groups[name].update(trajectories[i]["topology_sha256"] for i in indices)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert partition_groups[a].isdisjoint(partition_groups[b])
    manifest = {
        "split_seed": 123, "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "grouping": "Node-labelled undirected spring topology across all sources; conservative, not graph-isomorphism invariant.",
        "selection": "At most one trajectory per source/topology in each training or validation partition; mixed-T temperature quotas 5 each train and 4/4/3/3/3/3 val. Unselected members of used groups are excluded.",
        "test_policy": "Reserved, never evaluated by representation screens; not guaranteed historically unseen.",
        "sources": {source: {"path": data["path"], "sha256": data["sha256"],
                              "split_indices": splits[source]}
                    for source, data in inventory["sources"].items()},
    }
    with (DIRECTORY / "split_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print({source: {k: len(v) for k, v in parts.items()} for source, parts in splits.items()})


if __name__ == "__main__":
    main()
