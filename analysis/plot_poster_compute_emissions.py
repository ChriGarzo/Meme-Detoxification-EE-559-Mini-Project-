"""
Generate the poster-ready compute/emissions figure.

The values match the compute/emissions table in the poster draft, but are
rendered as compact log-scale lollipop charts so inference costs remain visible
beside teacher-generation costs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROWS = [
    {
        "label": "Stage 0\nOCR + CLIP",
        "kind": "preprocess",
        "gpu_h": 2.8,
        "gpu_label": "2.8 h",
        "energy": 0.76,
        "energy_label": "0.76 kWh",
        "co2_g": 26.7,
        "co2_label": "26.7 g",
    },
    {
        "label": "Stage 1A\nLLaVA explain",
        "kind": "teacher",
        "gpu_h": 14.1,
        "gpu_label": "14.1 h",
        "energy": 6.61,
        "energy_label": "6.61 kWh",
        "co2_g": 230.3,
        "co2_label": "230.3 g",
    },
    {
        "label": "Stage 1B\nLLaVA rewrite",
        "kind": "teacher",
        "gpu_h": 14.5,
        "gpu_label": "14.5 h",
        "energy": 5.78,
        "energy_label": "5.78 kWh",
        "co2_g": 201.3,
        "co2_label": "201.3 g",
    },
    {
        "label": "Stage 2\nBART LoRA",
        "kind": "student",
        "gpu_h": 2.1,
        "gpu_label": "2.1 h",
        "energy": 0.82,
        "energy_label": "0.82 kWh",
        "co2_g": 28.4,
        "co2_label": "28.4 g",
    },
    {
        "label": "BART\ninference",
        "kind": "student",
        "gpu_h": 12.3 / 60,
        "gpu_label": "12.3 min",
        "energy": 0.04,
        "energy_label": "0.04 kWh",
        "co2_g": 1.5,
        "co2_label": "1.5 g",
    },
    {
        "label": "Proxy\ninference",
        "kind": "student",
        "gpu_h": 3.4 / 60,
        "gpu_label": "3.4 min",
        "energy": 0.01,
        "energy_label": "0.01 kWh",
        "co2_g": 0.5,
        "co2_label": "0.5 g",
    },
]

COLORS = {
    "preprocess": "#6B7280",
    "teacher": "#D55E00",
    "student": "#0072B2",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot_panel(ax, key, labels_key, title, xlabel, xmin, xmax):
    y = np.arange(len(ROWS))[::-1]
    values = np.array([row[key] for row in ROWS])
    colors = [COLORS[row["kind"]] for row in ROWS]

    for yi, value, color, row in zip(y, values, colors, ROWS):
        ax.hlines(yi, xmin, value, color=color, linewidth=2.4, alpha=0.82)
        ax.scatter(value, yi, s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(
            value * 1.12,
            yi,
            row[labels_key],
            ha="left",
            va="center",
            fontsize=7.5,
            color="#2F2F2F",
        )

    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in ROWS])
    ax.grid(axis="x", color="#D6D6D6", linewidth=0.55, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", length=3, color="#555555")
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#555555")


def main() -> None:
    output_dir = Path("poster_template_DL/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.35), sharey=True)
    plot_panel(axes[0], "gpu_h", "gpu_label", "GPU time", "hours, log scale", 0.035, 30)
    plot_panel(axes[1], "energy", "energy_label", "Energy", "kWh, log scale", 0.006, 12)
    plot_panel(axes[2], "co2_g", "co2_label", "Emissions", "g CO$_2$, log scale", 0.28, 420)

    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[2].tick_params(axis="y", left=False, labelleft=False)

    handles = [
        plt.Line2D([0], [0], color=COLORS["teacher"], lw=3, marker="o", markersize=5, label="LLaVA teacher"),
        plt.Line2D([0], [0], color=COLORS["student"], lw=3, marker="o", markersize=5, label="Student / deployment"),
        plt.Line2D([0], [0], color=COLORS["preprocess"], lw=3, marker="o", markersize=5, label="Preprocessing"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.58, 1.04), ncol=3, frameon=False)
    fig.text(
        0.02,
        0.012,
        "Stage 1 dominates the one-time distillation footprint; deployed student inference is two orders of magnitude cheaper.",
        fontsize=8.0,
        color="#3B3B3B",
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.0)

    for suffix in ("pdf", "png", "svg"):
        out = output_dir / f"compute_emissions_lollipop.{suffix}"
        fig.savefig(out, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
