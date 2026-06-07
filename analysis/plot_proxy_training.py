"""
Plot training curves for the ExplanationProxy stage.

Reads:
  <proxy_checkpoint_dir>/training_history.json
  <proxy_checkpoint_dir>/eval_results.json          (optional)

Writes:
  <output_dir>/proxy_loss_curves.png
  <output_dir>/proxy_generalization_gap.png
  <output_dir>/proxy_training_summary.png

Usage:
  python analysis/plot_proxy_training.py \
      --proxy_checkpoint_dir /scratch/hmr_proxy_checkpoint \
      --output_dir /scratch/hmr_proxy_training_plots
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Return parsed JSON from path, or None if the file does not exist."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def epochs(values: List[float]) -> List[int]:
    """Return 1-indexed epoch numbers matching a list of per-epoch metric values."""
    return list(range(1, len(values) + 1))


def plot_proxy_curves(history: Dict[str, Any], eval_results: Optional[Dict[str, Any]], output_dir: Path) -> None:
    """Render proxy training curves and write PNG files to output_dir.

    Produces: proxy_loss_curves.png, proxy_generalization_gap.png (if both splits
    available), proxy_training_summary.png.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    train_loss = [float(x) for x in history.get("train_loss", [])]
    val_loss = [float(x) for x in history.get("val_loss", [])]
    max_epochs = max(len(train_loss), len(val_loss))
    if max_epochs == 0:
        raise ValueError("No train_loss or val_loss values found in proxy training history.")

    # ── Main loss curves ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    if train_loss:
        ax.plot(epochs(train_loss), train_loss, color="#1976D2", linewidth=2.2, marker="o", label="Train MSE")
    if val_loss:
        ax.plot(epochs(val_loss), val_loss, color="#D32F2F", linewidth=2.2, marker="s", label="Validation MSE")
        best_epoch = int(np.argmin(val_loss)) + 1
        best_val = min(val_loss)
        ax.scatter([best_epoch], [best_val], color="#111111", s=55, zorder=5, label=f"Best val: epoch {best_epoch}")
        ax.axvline(best_epoch, color="#111111", linestyle="--", linewidth=1, alpha=0.35)
    ax.set_title("Proxy Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "proxy_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Generalization gap ────────────────────────────────────────────────
    if train_loss and val_loss:
        n = min(len(train_loss), len(val_loss))
        gap = [val_loss[i] - train_loss[i] for i in range(n)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs(gap), gap, color="#5E35B1", linewidth=2.2, marker="o")
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
        ax.set_title("Proxy Generalization Gap")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE - Train MSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "proxy_generalization_gap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Text summary PNG (slide/report-friendly) ──────────────────────────
    final_train = train_loss[-1] if train_loss else None
    final_val = val_loss[-1] if val_loss else None
    best_val = min(val_loss) if val_loss else None
    best_epoch = int(np.argmin(val_loss)) + 1 if val_loss else None
    eval_mse = eval_results.get("mse_loss") if eval_results else None
    num_samples = eval_results.get("num_samples") if eval_results else None
    model_name = eval_results.get("model_name") if eval_results else None
    clip_model = eval_results.get("clip_model") if eval_results else None

    summary_lines = [
        "ExplanationProxy Training Summary",
        "",
        f"Epochs: {max_epochs}",
        f"Final train MSE: {final_train:.6g}" if final_train is not None else "Final train MSE: n/a",
        f"Final validation MSE: {final_val:.6g}" if final_val is not None else "Final validation MSE: n/a",
        f"Best validation MSE: {best_val:.6g} (epoch {best_epoch})" if best_val is not None else "Best validation MSE: n/a",
        f"Final eval MSE: {float(eval_mse):.6g}" if eval_mse is not None else "Final eval MSE: n/a",
        f"Eval samples: {num_samples}" if num_samples is not None else "Eval samples: n/a",
        f"BART target: {model_name}" if model_name else "BART target: n/a",
        f"CLIP model: {clip_model}" if clip_model else "CLIP model: n/a",
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(
        0.03,
        0.95,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#F5F7FA", edgecolor="#B0BEC5"),
    )
    fig.tight_layout()
    fig.savefig(output_dir / "proxy_training_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ExplanationProxy training curves")
    parser.add_argument("--proxy_checkpoint_dir", type=Path, default=Path("/scratch/hmr_proxy_checkpoint"))
    parser.add_argument("--output_dir", type=Path, default=Path("/scratch/hmr_proxy_training_plots"))
    args = parser.parse_args()

    history_path = args.proxy_checkpoint_dir / "training_history.json"
    eval_path = args.proxy_checkpoint_dir / "eval_results.json"
    history = load_json(history_path)
    if history is None:
        raise FileNotFoundError(f"Missing proxy training history: {history_path}")
    eval_results = load_json(eval_path)

    plot_proxy_curves(history, eval_results, args.output_dir)
    print(f"Saved proxy plots to {args.output_dir}")
    for path in sorted(args.output_dir.glob("proxy_*.png")):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
