"""Inventory raw trajectory identities on a compute node before freezing splits."""
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lss.data import _install_legacy_auxetic_box_alias
from latent_cluster_preflight import DATASETS


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else None
    return value if isinstance(value, (int, float, str, bool)) else None


def main():
    destination = ROOT / "notebooks/results/latent_matched_study/dataset_inventory.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    _install_legacy_auxetic_box_alias()
    result = {"job_id": os.environ.get("PBS_JOBID"), "sources": {}}
    references = defaultdict(list)
    for source, filename in DATASETS.items():
        path = ROOT / "data" / filename
        print(f"Loading {source}", flush=True)
        simulations = torch.load(path, map_location="cpu", weights_only=False)
        trajectories = []
        for index, sim in enumerate(simulations):
            g = sim[0]
            # Node-labelled reference identity; does not claim graph-isomorphism detection.
            edges = g.edge_index.cpu().long()
            pairs = sorted(set(tuple(sorted(pair)) for pair in edges.t().tolist()))
            topology = hashlib.sha256(json.dumps([g.num_nodes, pairs]).encode()).hexdigest()
            positions = torch.round(g.x[:, :2].cpu().double() * 1e6).long()
            reference = hashlib.sha256(topology.encode() + positions.numpy().tobytes()).hexdigest()
            references[reference].append([source, index])
            metadata = {key: scalar(getattr(g, key, None)) for key in (
                "source_sim_id", "sim_id", "network_id", "temperature", "time",
                "source_frame_index", "coordinate_normalization")}
            trajectories.append({
                "index": index, "frames": len(sim), "nodes": g.num_nodes,
                "edge_shape": list(g.edge_attr.shape), "keys": sorted(g.keys()),
                "topology_sha256": topology, "reference_sha256": reference,
                "metadata": metadata,
                "sampling": {key: [scalar(getattr(sim[i], key, None))
                                    for i in (0, 1, 5, 10, 50, 100) if i < len(sim)]
                             for key in ("time", "source_frame_index")},
            })
        result["sources"][source] = {
            "path": str(path), "sha256": digest_file(path),
            "trajectory_count": len(trajectories),
            "frame_counts": dict(Counter(t["frames"] for t in trajectories)),
            "trajectories": trajectories,
        }
        print(source, len(trajectories), result["sources"][source]["frame_counts"], flush=True)
        del simulations, sim, g
        gc.collect()
    result["repeated_reference_groups"] = [v for v in references.values() if len(v) > 1]
    result["identity_scope"] = "Node-labelled undirected topology plus initial xy rounded to 1e-6; metadata retained for further grouping checks."
    with destination.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Saved {destination}; repeated reference groups: {len(result['repeated_reference_groups'])}", flush=True)


if __name__ == "__main__":
    main()
