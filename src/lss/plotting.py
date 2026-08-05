from __future__ import annotations

import matplotlib.pyplot as plt


PAPER_COLORS = {
    # Darkened Okabe-Ito derivatives: color-vision-safe and sufficiently
    # contrasted against the white paper background for lines and markers.
    "ink": "#25282A",
    "blue": "#0072B2",
    "red": "#C84E00",
    "green": "#008A64",
    "orange": "#B87500",
    "gold": "#8A741C",
    "purple": "#A85D91",
    "sky": "#397EA8",
    "slate": "#626C72",
    "light": "#B7C1C6",
    "grid": "#D8DEE1",
}

DATASET_COLORS = {
    "depablo_low_temp": PAPER_COLORS["blue"],
    "depablo": PAPER_COLORS["blue"],
    "De Pablo": PAPER_COLORS["blue"],
    "de Pablo": PAPER_COLORS["blue"],
    "reid": PAPER_COLORS["red"],
    "Reid": PAPER_COLORS["red"],
    "depablo_mixed_temp": PAPER_COLORS["green"],
    "de Pablo mixed-T": PAPER_COLORS["green"],
    "lj_noisy": PAPER_COLORS["purple"],
    "Noisy LJ": PAPER_COLORS["purple"],
    "unknown": PAPER_COLORS["slate"],
}

# Backwards-compatible name used by older notebooks.
SOURCE_COLORS = DATASET_COLORS

LATENT_DIMENSION_COLORS = {
    1: PAPER_COLORS["purple"],
    2: PAPER_COLORS["blue"],
    4: PAPER_COLORS["orange"],
    8: PAPER_COLORS["green"],
}

MEASUREMENT_STYLES = {
    "AE ceiling": {"linestyle": "-", "marker": "o"},
    "Latent propagator": {"linestyle": "--", "marker": "s"},
    "Latent rollout": {"linestyle": "--", "marker": "s"},
    "latent simulator": {"linestyle": "--", "marker": "s"},
    "spatial GNN": {"linestyle": "-", "marker": "o"},
}

NETWORK_NODE_COLOR = PAPER_COLORS["red"]
NETWORK_EDGE_COLOR = PAPER_COLORS["slate"]

STATE_COLORS = {
    "folded": PAPER_COLORS["blue"],
    "unfolded": PAPER_COLORS["red"],
}


def apply_editorial_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "figure.constrained_layout.use": False,
            "axes.facecolor": "white",
            "axes.edgecolor": PAPER_COLORS["ink"],
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": PAPER_COLORS["ink"],
            "axes.titlecolor": PAPER_COLORS["ink"],
            "axes.linewidth": 0.9,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": PAPER_COLORS["ink"],
            "ytick.color": PAPER_COLORS["ink"],
            "legend.fontsize": 9,
            "legend.frameon": False,
            "grid.color": PAPER_COLORS["grid"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "font.family": "DejaVu Sans",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": plt.cycler(
                color=[
                    PAPER_COLORS["blue"], PAPER_COLORS["red"],
                    PAPER_COLORS["green"], PAPER_COLORS["purple"],
                    PAPER_COLORS["orange"], PAPER_COLORS["sky"],
                ]
            ),
        }
    )


def paper_caption(text: str) -> None:
    print(text)


def style_axes(ax, *, xlabel=None, ylabel=None, legend: bool = False) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False, loc="best")


def dataset_color(name: str) -> str:
    """Return the canonical paper color for a dataset name or label."""

    return DATASET_COLORS.get(str(name), PAPER_COLORS["slate"])


def latent_dimension_color(latent_dim: int) -> str:
    """Return the canonical paper color for a latent capacity."""

    return LATENT_DIMENSION_COLORS.get(int(latent_dim), PAPER_COLORS["slate"])


def measurement_style(name: str) -> dict:
    """Return a copy of the canonical line/marker style for a measurement."""

    return dict(MEASUREMENT_STYLES.get(str(name), {"linestyle": "-", "marker": "o"}))
