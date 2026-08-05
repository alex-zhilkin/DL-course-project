"""Plot the matched noisy-LJ AE endpoint p-ratio sweep for latent dims 1--6."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/results/07_lj_ae_strain"


def summary_path(latent_dim: int) -> Path:
    current = OUTPUT / (
        f"endpoint_summary_cv{latent_dim}_train100_strain0_pratio10.csv"
    )
    if current.exists():
        return current
    if latent_dim == 2:
        legacy = OUTPUT / "endpoint_summary_train100_strain0_pratio10.csv"
        if legacy.exists():
            return legacy
    raise FileNotFoundError(current)


frames = []
for latent_dim in range(1, 7):
    frame = pd.read_csv(summary_path(latent_dim))
    frame.insert(0, "latent_dim", latent_dim)
    frames.append(frame)

all_results = pd.concat(frames, ignore_index=True)
test = all_results.loc[all_results["split"].eq("test")].copy()
all_results.to_csv(OUTPUT / "latent_dim_1_to_6_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
colors = plt.cm.viridis([0.02, 0.20, 0.38, 0.56, 0.74, 0.92])
for color, (latent_dim, group) in zip(
    colors, test.groupby("latent_dim", sort=True)
):
    ax.plot(
        group["frame"],
        group["p_ratio_r2"],
        marker="o",
        linewidth=2,
        markersize=5,
        color=color,
        label=f"{latent_dim}D",
    )

ax.axhline(0, color="0.35", linewidth=1, linestyle="--")
ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
ax.set_yticks([-20, -10, -3, -1, 0, 0.5, 1])
ax.set_yticklabels(["−20", "−10", "−3", "−1", "0", "0.5", "1"])
ax.set(
    xlabel="Ground-truth frame",
    ylabel=r"Unseen-network endpoint p-ratio $R^2$",
    title="Noisy LJ autoencoder: latent-dimension sweep",
)
ax.grid(alpha=0.22)
ax.legend(title="Latent", ncol=2, frameon=False)
fig.savefig(OUTPUT / "latent_dim_1_to_6_r2_vs_frame.png", dpi=220)
fig.savefig(OUTPUT / "latent_dim_1_to_6_r2_vs_frame.pdf")
plt.close(fig)

final = (
    test.loc[test["frame"].eq(199)]
    .sort_values("latent_dim")
    [["latent_dim", "p_ratio_r2", "p_ratio_pearson", "p_ratio_mae"]]
)
final.to_csv(OUTPUT / "latent_dim_1_to_6_frame199.csv", index=False)
print(final.to_string(index=False))
