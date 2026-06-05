from __future__ import annotations

import numpy as np
import torch

from .graph import build_graph

# Utils code, unrelated to Deep Learning and used as a benchmark to prove that the model learned something mechnically meaningful 

def make_r0_dict(ref_graph, pos_dim: int = 2) -> dict[tuple[int, int], float]:
    x0 = ref_graph.x[:, :pos_dim].detach().cpu().numpy()
    edge_index = ref_graph.edge_index.detach().cpu().numpy()
    r0: dict[tuple[int, int], float] = {}
    for e in range(edge_index.shape[1]):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in r0:
            continue
        r0[(a, b)] = float(np.linalg.norm(x0[a] - x0[b]))
    return r0


def build_spring_hessian_2d(graph, r0_dict: dict[tuple[int, int], float], pos_dim: int = 2) -> np.ndarray:
    x = graph.x[:, :pos_dim].detach().cpu().numpy()
    edge_index = graph.edge_index.detach().cpu().numpy()
    edge_attr = graph.edge_attr.detach().cpu().numpy()

    n = x.shape[0]
    dim = pos_dim
    hessian = np.zeros((dim * n, dim * n), dtype=np.float64)
    seen: set[tuple[int, int]] = set()

    for e in range(edge_index.shape[1]):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        if i == j:
            continue

        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))

        rij = x[i] - x[j]
        r = float(np.linalg.norm(rij))
        if r < 1e-12:
            continue

        u = rij / r
        k = float(edge_attr[e, -1])
        if (not np.isfinite(k)) or k <= 0.0:
            continue
        if (a, b) not in r0_dict:
            continue
        r0 = float(r0_dict[(a, b)])

        alpha = k * (r - r0) / r
        beta = k * r0 / r
        kij = alpha * np.eye(dim) + beta * np.outer(u, u)

        si = slice(dim * i, dim * (i + 1))
        sj = slice(dim * j, dim * (j + 1))
        hessian[si, si] += kij
        hessian[sj, sj] += kij
        hessian[si, sj] -= kij
        hessian[sj, si] -= kij

    return 0.5 * (hessian + hessian.T)


def collect_cv_lambda5_points(
    model,
    sims,
    *,
    history: int,
    pos_dim: int,
    device: str,
    max_sims: int = 100,
    cv_index: int = 1,
) -> dict[str, np.ndarray]:
    sim_ids: list[int] = []
    frames: list[int] = []
    cv_vals: list[float] = []
    lam5_vals: list[float] = []

    model.eval()
    with torch.no_grad():
        for sim_idx, sim in enumerate(sims[:max_sims]):
            n_local = len(sim) - 1
            r0_dict = make_r0_dict(sim[0], pos_dim=pos_dim)
            for t in range(history, n_local):
                frame_graphs = [sim[i].to(device) for i in range(t - history, t + 1)]
                if history == 0 and t > 0:
                    prev_pos = sim[t - 1].to(device).x[:, :pos_dim]
                    cur_pos = frame_graphs[-1].x[:, :pos_dim]
                    frame_graphs[-1].vel_state = cur_pos - prev_pos

                graph_in = build_graph(frame_graphs).to(device)

                cv = model.extract_cv(graph_in, is_training=False).squeeze(0).detach().cpu().numpy()
                cv = np.asarray(cv, dtype=float).reshape(-1)
                cv_idx = cv_index if cv.shape[0] > cv_index else 0
                cv_value = float(cv[cv_idx])

                hessian = build_spring_hessian_2d(graph_in, r0_dict=r0_dict, pos_dim=pos_dim)
                evals = np.linalg.eigvalsh(hessian)
                evals.sort()
                lambda5 = float(evals[4]) if evals.shape[0] > 4 else float("nan")

                sim_ids.append(sim_idx)
                frames.append(t)
                cv_vals.append(cv_value)
                lam5_vals.append(lambda5)

    return {
        "sim_idx": np.asarray(sim_ids, dtype=int),
        "frame": np.asarray(frames, dtype=int),
        "cv": np.asarray(cv_vals, dtype=float),
        "lambda5": np.asarray(lam5_vals, dtype=float),
    }
