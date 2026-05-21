"""
Generate the poster-ready compute/emissions figure.

The figure has two rows:
  1. Total GPU time and emissions for the pipeline stages.
  2. Per-image GPU time and emissions for the compared inference systems.

Values are rendered as compact log-scale lollipop charts so small deployed
models remain visible beside the teacher-generation costs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PIPELINE_ROWS = [
    {
        "label": "Stage 0",
        "kind": "preprocess",
        "gpu_h": 2.8,
        "gpu_label": "2.8 h",
        "energy": 0.76,
        "energy_label": "0.76 kWh",
        "co2_g": 26.7,
        "co2_label": "26.7 g",
    },
    {
        "label": "Stage 1A",
        "kind": "teacher",
        "gpu_h": 14.1,
        "gpu_label": "14.1 h",
        "energy": 6.61,
        "energy_label": "6.61 kWh",
        "co2_g": 230.3,
        "co2_label": "230.3 g",
    },
    {
        "label": "Stage 1B",
        "kind": "teacher",
        "gpu_h": 14.5,
        "gpu_label": "14.5 h",
        "energy": 5.78,
        "energy_label": "5.78 kWh",
        "co2_g": 201.3,
        "co2_label": "201.3 g",
    },
    {
        "label": "Stage 2",
        "kind": "student",
        "gpu_h": 2.1,
        "gpu_label": "2.1 h",
        "energy": 0.82,
        "energy_label": "0.82 kWh",
        "co2_g": 28.4,
        "co2_label": "28.4 g",
    },
]

COLORS = {
    "preprocess": "#6B7280",
    "teacher": "#D55E00",
    "student": "#0072B2",
    "baseline": "#E69F00",
    "proxy": "#56B4E9",
}

# Per-image benchmark values. LLaVA and DetoxLLM come from the
# single-inference benchmark table in README.md. Proxy+BART is derived from the
# reported 280-example proxy run: 3.4 min and 0.5 g CO2 over 280 examples.
INFERENCE_ROWS = [
    {
        "label": "LLaVA\nteacher",
        "kind": "teacher",
        "seconds": 9.376,
        "time_label": "9.38 s",
        "co2_ug": 34340,
        "co2_mg": 34.34,
        "co2_label": "34.3 mg",
    },
    {
        "label": "DetoxLLM",
        "kind": "baseline",
        "seconds": 1.989,
        "time_label": "1.99 s",
        "co2_ug": 5646,
        "co2_mg": 5.646,
        "co2_label": "5.65 mg",
    },
    {
        "label": "CLIP Proxy\n+ BART FT full",
        "kind": "proxy",
        "seconds": 3.4 * 60 / 280,
        "time_label": "0.729 s",
        "co2_ug": 0.5 * 1e6 / 280,
        "co2_mg": 0.5 * 1000 / 280,
        "co2_label": "1.79 mg",
    },
]


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


def plot_panel(ax, rows, key, labels_key, title, xlabel, xmin, xmax, *, show_ylabels=True):
    y = np.arange(len(rows))[::-1]
    values = np.array([row[key] for row in rows])
    colors = [COLORS[row["kind"]] for row in rows]

    for yi, value, color, row in zip(y, values, colors, rows):
        ax.hlines(yi, xmin, value, color=color, linewidth=2.4, alpha=0.82)
        ax.scatter(value, yi, s=42, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(
            value + (xmax - xmin) * 0.03,
            yi,
            row[labels_key],
            ha="left",
            va="center",
            fontsize=7.5,
            color="#2F2F2F",
        )

    ax.set_xlim(xmin, xmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_yticks(y)
    ax.set_yticklabels([row["label"] for row in rows] if show_ylabels else [])
    ax.set_ylim(bottom=-0.7)
    ax.grid(axis="x", color="#D6D6D6", linewidth=0.55, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="both", length=3, color="#555555")
    ax.tick_params(axis="y", left=show_ylabels)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#555555")
    if not show_ylabels:
        ax.spines["left"].set_visible(False)


def main() -> None:
    output_dir = Path("poster_template_DL/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 4.65))
    plot_panel(
        axes[0, 0],
        PIPELINE_ROWS,
        "gpu_h",
        "gpu_label",
        "(a) Pipeline GPU time",
        "hours",
        0,
        19,
    )
    plot_panel(
        axes[0, 1],
        PIPELINE_ROWS,
        "co2_g",
        "co2_label",
        "(b) Pipeline emissions",
        "g CO$_2$",
        0,
        280,
        show_ylabels=False,
    )
    plot_panel(
        axes[1, 0],
        INFERENCE_ROWS,
        "seconds",
        "time_label",
        "(c) Single-image GPU time",
        "seconds",
        0,
        13,
    )
    plot_panel(
        axes[1, 1],
        INFERENCE_ROWS,
        "co2_mg",
        "co2_label",
        "(d) Single-image emissions",
        "mg CO$_2$",
        0,
        43,
        show_ylabels=False,
    )

    handles = [
        plt.Line2D([0], [0], color=COLORS["teacher"], lw=3, marker="o", markersize=5, label="LLaVA teacher"),
        plt.Line2D([0], [0], color=COLORS["student"], lw=3, marker="o", markersize=5, label="BART student"),
        plt.Line2D([0], [0], color=COLORS["proxy"], lw=3, marker="o", markersize=5, label="CLIP Proxy+BART FT full"),
        plt.Line2D([0], [0], color=COLORS["baseline"], lw=3, marker="o", markersize=5, label="DetoxLLM"),
        plt.Line2D([0], [0], color=COLORS["preprocess"], lw=3, marker="o", markersize=5, label="Preprocessing"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 1.025), ncol=5, frameon=False)
    fig.text(
        0.02,
        0.006,
        "Proxy+BART per-image cost is derived from the 280-example proxy run; other per-image values use the single-inference benchmark.",
        fontsize=8.0,
        color="#3B3B3B",
    )
    plt.tight_layout(rect=(0, 0.055, 1, 0.93), w_pad=1.8, h_pad=1.35)

    for suffix in ("pdf", "png", "svg"):
        out = output_dir / f"compute_emissions_lollipop.{suffix}"
        fig.savefig(out, dpi=600 if suffix == "png" else None, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
