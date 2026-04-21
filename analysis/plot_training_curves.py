"""
Plot training curves from Stage 2 training_history.json files.

Reads the training_history.json saved by train_stage2_phase1.py and
train_stage2_phase2.py and produces publication-ready PNG figures for the
final report.

Outputs (saved to --output_dir):
  phase1_curves.png          — Phase 1 train loss / eval loss / eval rougeL
  phase2_loss_curves.png     — Phase 2 eval loss for all 4 conditions
  phase2_rouge_curves.png    — Phase 2 eval rougeL for all 4 conditions
  phase2_train_loss.png      — Phase 2 training loss for all 4 conditions
  all_phases_summary.png     — Side-by-side overview of every stage

Usage:
    python analysis/plot_training_curves.py \\
        --phase1_dir   /scratch/hmr_stage2_phase1_checkpoint \\
        --phase2_dirs  /scratch/hmr_stage2_phase2_full_checkpoint \\
                       /scratch/hmr_stage2_phase2_target_only_checkpoint \\
                       /scratch/hmr_stage2_phase2_attack_only_checkpoint \\
                       /scratch/hmr_stage2_phase2_none_checkpoint \\
        --output_dir   /scratch/hmr_training_plots

    # Or, if all checkpoints live under a common root:
    python analysis/plot_training_curves.py \\
        --scratch_root /scratch \\
        --output_dir   /scratch/hmr_training_plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── colour palette (colour-blind friendly) ──────────────────────────────────
CONDITION_COLORS = {
    "full":        "#2196F3",   # blue
    "target_only": "#FF9800",   # orange
    "attack_only": "#4CAF50",   # green
    "none":        "#9C27B0",   # purple
    "phase1":      "#E53935",   # red
}
CONDITION_LABELS = {
    "full":        "Full [T+A+M]",
    "target_only": "Target only [T]",
    "attack_only": "Attack only [A]",
    "none":        "No cond. [—]",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_history(checkpoint_dir: str) -> Optional[Dict]:
    path = Path(checkpoint_dir) / "training_history.json"
    if not path.exists():
        # Fall back to HuggingFace's native trainer_state.json
        fallback = Path(checkpoint_dir) / "trainer_state.json"
        if fallback.exists():
            with open(fallback) as f:
                state = json.load(f)
            return {
                "phase": Path(checkpoint_dir).name,
                "condition": None,
                "run_config": {},
                "results": {},
                "log_history": state.get("log_history", []),
            }
        print(f"  [WARN] No training_history.json or trainer_state.json in {checkpoint_dir}")
        return None
    with open(path) as f:
        return json.load(f)


def split_log(log_history: List[Dict]) -> Tuple[List, List]:
    """Separate train-step entries from eval-step entries."""
    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs  = [e for e in log_history if "eval_loss" in e]
    return train_logs, eval_logs


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.1f}h"


# ── individual plot helpers ──────────────────────────────────────────────────

def plot_phase1(history: Dict, out_dir: Path, plt, np):
    train_logs, eval_logs = split_log(history["log_history"])
    if not train_logs and not eval_logs:
        print("  [WARN] Phase 1 log_history is empty — nothing to plot.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Stage 2 — Phase 1: BART ParaDetox Warm-up", fontsize=13, fontweight="bold")

    color = CONDITION_COLORS["phase1"]
    rc = history.get("run_config", {})
    res = history.get("results", {})

    # ── train loss ──
    ax = axes[0]
    if train_logs:
        steps  = [e["step"]  for e in train_logs]
        losses = [e["loss"]  for e in train_logs]
        ax.plot(steps, losses, color=color, linewidth=1.5, alpha=0.8)
        # smoothed trend
        if len(losses) > 10:
            window = max(5, len(losses) // 20)
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax.plot(steps[window-1:], smoothed, color=color, linewidth=2.5, label="smoothed")
        ax.set_title("Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    # ── eval loss ──
    ax = axes[1]
    if eval_logs:
        steps  = [e["step"]       for e in eval_logs]
        losses = [e["eval_loss"]  for e in eval_logs]
        ax.plot(steps, losses, color=color, linewidth=2, marker="o", markersize=4)
        ax.set_title("Validation Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Eval Loss")
        ax.grid(True, alpha=0.3)

    # ── eval rougeL ──
    ax = axes[2]
    rouge_logs = [e for e in eval_logs if "eval_rougeL" in e]
    if rouge_logs:
        steps   = [e["step"]        for e in rouge_logs]
        rouges  = [e["eval_rougeL"] for e in rouge_logs]
        ax.plot(steps, rouges, color=color, linewidth=2, marker="s", markersize=4)
        ax.set_title("Validation ROUGE-L")
        ax.set_xlabel("Step")
        ax.set_ylabel("ROUGE-L")
        ax.grid(True, alpha=0.3)

    # ── eval STA ──
    ax = axes[3]
    sta_logs = [e for e in eval_logs if "eval_sta" in e]
    if sta_logs:
        steps  = [e["step"]     for e in sta_logs]
        scores = [e["eval_sta"] for e in sta_logs]
        ax.plot(steps, scores, color=color, linewidth=2, marker="^", markersize=5)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color="#aaa", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title("Validation STA\n(↑ = more non-toxic outputs)")
        ax.set_xlabel("Step")
        ax.set_ylabel("STA (prop. non-toxic)")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("Validation STA\n(not available)")
        ax.text(0.5, 0.5, "STA not recorded\n(old run)", ha="center", va="center",
                transform=ax.transAxes, color="#aaa", fontsize=10)

    # ── annotation box ──
    info_lines = []
    if rc.get("num_epochs"):     info_lines.append(f"Epochs: {rc['num_epochs']}")
    if rc.get("batch_size"):     info_lines.append(f"Batch:  {rc['batch_size']}")
    if rc.get("learning_rate"):  info_lines.append(f"LR:     {rc['learning_rate']}")
    if rc.get("train_samples"):  info_lines.append(f"Train:  {rc['train_samples']:,}")
    if res.get("training_duration_seconds"):
        info_lines.append(f"Time:   {fmt_duration(res['training_duration_seconds'])}")
    if res.get("best_metric_rougeL") is not None:
        info_lines.append(f"Best rougeL: {res['best_metric_rougeL']:.4f}")
    if info_lines:
        fig.text(0.5, -0.04, "   |   ".join(info_lines),
                 ha="center", fontsize=9, color="#555555",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    out_path = out_dir / "phase1_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_phase2_comparison(histories: List[Dict], out_dir: Path, plt, np):
    """One figure per metric, all 4 conditions overlaid."""
    if not histories:
        return

    for metric_key, metric_label, fname in [
        ("eval_loss",   "Validation Loss",    "phase2_loss_curves.png"),
        ("eval_rougeL", "Validation ROUGE-L", "phase2_rouge_curves.png"),
        ("eval_sta",    "Validation STA (↑ = more non-toxic outputs)", "phase2_sta_curves.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title(f"Stage 2 — Phase 2: {metric_label} by Condition", fontsize=12, fontweight="bold")

        any_data = False
        for hist in histories:
            cond = hist.get("condition") or Path(str(hist.get("run_config", {}).get("phase1_checkpoint", ""))).name
            label = CONDITION_LABELS.get(cond, cond)
            color = CONDITION_COLORS.get(cond, "#607D8B")

            _, eval_logs = split_log(hist["log_history"])
            entries = [e for e in eval_logs if metric_key in e]
            if not entries:
                continue

            steps  = [e["step"]      for e in entries]
            values = [e[metric_key]  for e in entries]
            ax.plot(steps, values, color=color, linewidth=2, marker="o", markersize=4, label=label)
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.set_xlabel("Step")
        ax.set_ylabel(metric_label)
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # ── training loss per condition ──
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Stage 2 — Phase 2: Training Loss by Condition", fontsize=12, fontweight="bold")
    any_data = False
    for hist in histories:
        cond  = hist.get("condition") or "unknown"
        label = CONDITION_LABELS.get(cond, cond)
        color = CONDITION_COLORS.get(cond, "#607D8B")
        train_logs, _ = split_log(hist["log_history"])
        if not train_logs:
            continue
        steps  = [e["step"] for e in train_logs]
        losses = [e["loss"] for e in train_logs]
        if len(losses) > 10:
            window = max(5, len(losses) // 20)
            smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax.plot(steps[window-1:], smoothed, color=color, linewidth=2.5, label=label)
        else:
            ax.plot(steps, losses, color=color, linewidth=2, label=label)
        any_data = True

    if any_data:
        ax.set_xlabel("Step")
        ax.set_ylabel("Training Loss (smoothed)")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / "phase2_train_loss.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_summary(phase1_hist: Optional[Dict], phase2_hists: List[Dict], out_dir: Path, plt, np):
    """4-panel summary: P1 train loss | P1 eval loss | P2 eval loss | P2 rougeL."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Stage 2 Training Overview — All Phases", fontsize=13, fontweight="bold")

    # Panel 0 — Phase 1 train loss
    ax = axes[0]
    ax.set_title("P1 Training Loss")
    if phase1_hist:
        train_logs, _ = split_log(phase1_hist["log_history"])
        if train_logs:
            steps  = [e["step"] for e in train_logs]
            losses = [e["loss"] for e in train_logs]
            ax.plot(steps, losses, color=CONDITION_COLORS["phase1"], linewidth=1, alpha=0.5)
            if len(losses) > 10:
                w = max(5, len(losses)//20)
                sm = np.convolve(losses, np.ones(w)/w, mode="valid")
                ax.plot(steps[w-1:], sm, color=CONDITION_COLORS["phase1"], linewidth=2.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)

    # Panel 1 — Phase 1 eval loss
    ax = axes[1]
    ax.set_title("P1 Validation Loss")
    if phase1_hist:
        _, eval_logs = split_log(phase1_hist["log_history"])
        if eval_logs:
            steps  = [e["step"]      for e in eval_logs]
            losses = [e["eval_loss"] for e in eval_logs]
            ax.plot(steps, losses, color=CONDITION_COLORS["phase1"], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Step"); ax.set_ylabel("Eval Loss"); ax.grid(True, alpha=0.3)

    # Panel 2 — Phase 2 eval loss (all conditions)
    ax = axes[2]
    ax.set_title("P2 Validation Loss (by condition)")
    for hist in phase2_hists:
        cond  = hist.get("condition", "unknown")
        _, eval_logs = split_log(hist["log_history"])
        entries = [e for e in eval_logs if "eval_loss" in e]
        if entries:
            ax.plot([e["step"] for e in entries], [e["eval_loss"] for e in entries],
                    color=CONDITION_COLORS.get(cond, "#607D8B"), linewidth=2, marker="o", markersize=3,
                    label=CONDITION_LABELS.get(cond, cond))
    ax.set_xlabel("Step"); ax.set_ylabel("Eval Loss"); ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)

    # Panel 3 — Phase 2 rougeL (all conditions)
    ax = axes[3]
    ax.set_title("P2 ROUGE-L (by condition)")
    for hist in phase2_hists:
        cond  = hist.get("condition", "unknown")
        _, eval_logs = split_log(hist["log_history"])
        entries = [e for e in eval_logs if "eval_rougeL" in e]
        if entries:
            ax.plot([e["step"] for e in entries], [e["eval_rougeL"] for e in entries],
                    color=CONDITION_COLORS.get(cond, "#607D8B"), linewidth=2, marker="s", markersize=3,
                    label=CONDITION_LABELS.get(cond, cond))
    ax.set_xlabel("Step"); ax.set_ylabel("ROUGE-L"); ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "all_phases_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot Stage 2 training curves")
    parser.add_argument("--phase1_dir",  type=str, default=None,
                        help="Phase 1 checkpoint directory (contains training_history.json)")
    parser.add_argument("--phase2_dirs", type=str, nargs="+", default=None,
                        help="Phase 2 checkpoint directories (one per condition)")
    parser.add_argument("--scratch_root", type=str, default=None,
                        help="Auto-discover checkpoints under this root "
                             "(looks for hmr_stage2_phase*_checkpoint)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where PNG plots are saved")
    args = parser.parse_args()

    # ── auto-discover from scratch root ──────────────────────────────────────
    if args.scratch_root:
        root = Path(args.scratch_root)
        if args.phase1_dir is None:
            candidate = root / "hmr_stage2_phase1_checkpoint"
            if candidate.exists():
                args.phase1_dir = str(candidate)
        if args.phase2_dirs is None:
            args.phase2_dirs = sorted(str(p) for p in root.glob("hmr_stage2_phase2_*checkpoint"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib and numpy are required. Install: pip install matplotlib numpy")
        sys.exit(1)

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    phase1_hist = None
    if args.phase1_dir:
        print(f"\nLoading Phase 1 from: {args.phase1_dir}")
        phase1_hist = load_history(args.phase1_dir)
        if phase1_hist:
            print("  Plotting Phase 1 curves...")
            plot_phase1(phase1_hist, out_dir, plt, np)
        else:
            print("  Skipping Phase 1 plots (no history found)")
    else:
        print("\n[INFO] --phase1_dir not provided; skipping Phase 1 plots")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    phase2_hists = []
    if args.phase2_dirs:
        print(f"\nLoading Phase 2 from {len(args.phase2_dirs)} directories:")
        for d in args.phase2_dirs:
            print(f"  {d}")
            h = load_history(d)
            if h:
                phase2_hists.append(h)
        if phase2_hists:
            print(f"  Plotting Phase 2 comparison curves ({len(phase2_hists)} conditions)...")
            plot_phase2_comparison(phase2_hists, out_dir, plt, np)
        else:
            print("  Skipping Phase 2 plots (no history found)")
    else:
        print("\n[INFO] --phase2_dirs not provided; skipping Phase 2 plots")

    # ── Summary ──────────────────────────────────────────────────────────────
    if phase1_hist or phase2_hists:
        print("\n  Plotting summary overview...")
        plot_summary(phase1_hist, phase2_hists, out_dir, plt, np)

    print(f"\nAll plots saved to: {out_dir}")
    print("Files:")
    for f in sorted(out_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
