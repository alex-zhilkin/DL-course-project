from __future__ import annotations

import matplotlib.pyplot as plt


PAPER_COLORS = {
    "ink": "#111827",
    "blue": "#1d4ed8",
    "red": "#b91c1c",
    "green": "#047857",
    "orange": "#c2410c",
    "gold": "#b45309",
    "slate": "#475569",
    "light": "#94a3b8",
}


def apply_editorial_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PAPER_COLORS["ink"],
            "axes.linewidth": 0.9,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.24,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
        }
    )


def paper_caption(text: str) -> None:
    print(text)


def style_axes(ax, *, xlabel=None, ylabel=None, legend: bool = False) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(frameon=False, loc="best")
