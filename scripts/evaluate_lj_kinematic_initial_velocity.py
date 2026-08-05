"""Test whether two observed frames rescue noisy-LJ latent rollout.

The saved kinematic propagators were already trained with consecutive latent
states.  This script changes only rollout initialization:

* ``zero``: frame 0 only, so latent velocity is initialized to zero.
* ``observed``: encode frames 0 and 1, use z1-z0 as latent velocity, and
  autonomously predict frames 2 onward.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from lss.latent.capacity import evaluate_experiment, load_experiment_bundle
from lss.utils import resolve_device


RUNS = {
    2: ROOT
    / "notebooks/results/07_lj_4d_propagator_allframes_cv2_h16/models/"
    "lj_cv2_ae200_dyn199_seed20260726.pt",
    4: ROOT
    / "notebooks/results/07_lj_4d_propagator_allframes_cv4_h16/models/"
    "lj_cv4_ae200_dyn199_seed20260726.pt",
}
OUTPUT = ROOT / "notebooks/results/11_lj_kinematic_initial_velocity"
HORIZONS = [25, 50, 99, 125, 150, 199]


def main() -> None:
    device = resolve_device("auto")
    metric_parts = []
    row_parts = []

    for latent_dim, checkpoint in RUNS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        result = load_experiment_bundle(checkpoint, cfg={}, device=device)
        for initialization in ("zero", "observed"):
            result["params"]["initial_velocity"] = initialization
            evaluated = evaluate_experiment(
                result,
                {"rollout_steps_grid": HORIZONS},
                device=device,
            )
            metrics = evaluated["rollout_stats"].copy()
            metrics["latent_dim"] = latent_dim
            metrics["initial_velocity"] = initialization
            metric_parts.append(metrics)

            rows = evaluated["rollout_rows"].copy()
            rows["latent_dim"] = latent_dim
            rows["initial_velocity"] = initialization
            row_parts.append(rows)

    metrics = pd.concat(metric_parts, ignore_index=True)
    rows = pd.concat(row_parts, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "test_metrics.csv", index=False)
    rows.to_csv(OUTPUT / "test_rows.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    for (latent_dim, initialization), group in metrics.groupby(
        ["latent_dim", "initial_velocity"], sort=False
    ):
        label = (
            f"{latent_dim}D, frames 0+1"
            if initialization == "observed"
            else f"{latent_dim}D, frame 0"
        )
        ax.plot(
            group["rollout_steps"],
            group["p_ratio_r2"],
            marker="o",
            label=label,
        )
    ax.axhline(0, color="0.3", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Rollout frame",
        ylabel=r"Endpoint p-ratio $R^2$",
        title="Does observed initial latent velocity rescue noisy-LJ rollout?",
    )
    ax.legend(frameon=False, ncol=2)
    fig.savefig(OUTPUT / "initial_velocity_control.pdf")
    fig.savefig(OUTPUT / "initial_velocity_control.png", dpi=300)
    plt.close(fig)

    columns = [
        "latent_dim",
        "initial_velocity",
        "rollout_steps",
        "used",
        "p_ratio_r2",
        "p_ratio_pearson",
    ]
    print(metrics[columns].to_string(index=False))


if __name__ == "__main__":
    main()
