"""
Recover training metrics from already-completed runs that predate training_history.json.

HuggingFace Trainer always writes trainer_state.json inside each checkpoint
subdirectory (e.g. checkpoint-1646/trainer_state.json).  This script finds
those files, consolidates the log_history, and produces:

  <checkpoint_dir>/training_history.json   ← same format as our new code writes
  <output_dir>/phase1_curves.png
  <output_dir>/phase2_*.png
  <output_dir>/all_phases_summary.png

Usage — recover everything at once:
    python analysis/recover_training_metrics.py \\
        --scratch_root /scratch \\
        --output_dir   /scratch/hmr_training_plots

Usage — single checkpoint:
    python analysis/recover_training_metrics.py \\
        --checkpoint_dir /scratch/hmr_stage2_phase1_checkpoint \\
        --phase phase1 \\
        --output_dir /scratch/hmr_training_plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ── known checkpoint directory → (phase, condition) mapping ──────────────────
KNOWN_DIRS = {
    "hmr_stage2_phase1_checkpoint":         ("phase1", None),
    "hmr_stage2_phase2_full_checkpoint":    ("phase2", "full"),
    "hmr_stage2_phase2_target_only_checkpoint": ("phase2", "target_only"),
    "hmr_stage2_phase2_visual_only_checkpoint": ("phase2", "visual_only"),
    "hmr_stage2_phase2_none_checkpoint":    ("phase2", "none"),
}

CONDITION_COLORS = {
    # Okabe-Ito inspired, colorblind-safe palette.
    "full":        "#0072B2",
    "target_only": "#E69F00",
    "visual_only": "#009E73",
    "none":        "#CC79A7",
    "phase1":      "#D55E00",
}
CONDITION_LABELS = {
    "full":        "Full (T+V+M)",
    "target_only": "Target only (T)",
    "visual_only": "Visual only (V)",
    "none":        "No condition",
}
LOWER_IS_BETTER = {
    "eval_loss",
    "eval_pred_toxicity_mean",
    "eval_multimodal_hate_prob_mean",
    "eval_copy_rate_high",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_trainer_state(checkpoint_dir: Path) -> Optional[Path]:
    """
    Look for trainer_state.json.
    Priority:
      1. Direct file: <checkpoint_dir>/trainer_state.json
      2. Inside checkpoint subdirs: <checkpoint_dir>/checkpoint-NNNN/trainer_state.json
         (pick the newest file by mtime, so mixed old/new reruns do not select
         a stale high-numbered checkpoint from a previous run)
    """
    direct = checkpoint_dir / "trainer_state.json"
    if direct.exists():
        return direct

    candidates = list(checkpoint_dir.glob("checkpoint-*/trainer_state.json"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return None


def latest_trainer_state_mtime(checkpoint_dir: Path) -> Optional[float]:
    """Return the newest trainer_state.json mtime under a checkpoint directory."""
    paths = list(checkpoint_dir.rglob("trainer_state.json"))
    if not paths:
        return None
    return max(p.stat().st_mtime for p in paths)


def load_trainer_state(path: Path) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_history_data(state: Dict, phase: str, condition: Optional[str],
                       checkpoint_dir: Path) -> Dict:
    """Wrap a raw trainer_state.json into our training_history.json schema."""
    log_history = state.get("log_history", [])

    # Best metric
    best_metric = state.get("best_metric")
    best_checkpoint = state.get("best_model_checkpoint")

    # Infer total steps
    total_steps = state.get("global_step", 0)

    # Try to read training_args.bin → too binary; fall back to best-effort
    run_config = {
        "recovered_from": "trainer_state.json",
        "checkpoint_dir": str(checkpoint_dir),
    }
    if condition:
        run_config["condition"] = condition

    return {
        "phase": f"phase1_paradetox" if phase == "phase1" else "phase2_meme_finetune",
        "condition": condition,
        "run_config": run_config,
        "hardware": {},   # not recoverable post-hoc
        "results": {
            "total_steps": total_steps,
            "best_metric": best_metric,
            "best_model_checkpoint": str(best_checkpoint) if best_checkpoint else None,
            "training_duration_seconds": None,  # not recoverable post-hoc
        },
        "log_history": log_history,
        "recovered": True,
    }


def recover_checkpoint(checkpoint_dir: Path, phase: str, condition: Optional[str]) -> Optional[Dict]:
    print(f"\n  Checkpoint: {checkpoint_dir.name}")

    # If training_history.json already exists, just load it
    existing = checkpoint_dir / "training_history.json"
    if existing.exists():
        latest_state_mtime = latest_trainer_state_mtime(checkpoint_dir)
        if latest_state_mtime is None or existing.stat().st_mtime >= latest_state_mtime:
            print(f"    ✓ training_history.json already exists — loading it")
            with open(existing) as f:
                return json.load(f)
        print(
            "    ! training_history.json is older than a checkpoint trainer_state.json "
            "— rebuilding from newest trainer_state.json"
        )

    # Try to find trainer_state.json
    state_path = find_trainer_state(checkpoint_dir)
    if state_path is None:
        print(f"    ✗ No trainer_state.json found anywhere under {checkpoint_dir}")
        print(f"      The training log_history is not recoverable for this run.")
        return None

    print(f"    ✓ Found trainer_state.json at {state_path.relative_to(checkpoint_dir)}")
    state = load_trainer_state(state_path)
    n_logs = len(state.get("log_history", []))
    print(f"    ✓ log_history has {n_logs} entries")

    if n_logs == 0:
        print(f"    ✗ log_history is empty — no metrics were recorded")
        return None

    history = build_history_data(state, phase, condition, checkpoint_dir)

    # Write training_history.json to the checkpoint dir for future use
    out_path = checkpoint_dir / "training_history.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"    ✓ Saved training_history.json to {checkpoint_dir.name}/")
    except PermissionError:
        print(f"    ! Could not write to {checkpoint_dir} (permission denied) — using in-memory copy")

    return history


def canonicalize_condition(condition: Optional[str]) -> Optional[str]:
    """Normalize condition names."""
    return condition


# ── plotting (shared with plot_training_curves.py) ────────────────────────────

def split_log(log_history):
    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs  = [e for e in log_history if "eval_loss" in e]
    return train_logs, eval_logs


def plot_all(phase1_hist, phase2_hists, out_dir, plt, np):
    """Reproduce all plots from plot_training_curves.py inline."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    def smooth(values, window):
        if len(values) <= window:
            return values
        return list(np.convolve(values, np.ones(window) / window, mode="valid"))

    def smooth_xy(steps, values, window):
        if len(values) <= window:
            return steps, values
        return steps[window - 1:], smooth(values, window)

    def save_figure(fig, path, dpi=600):
        """Save a poster/paper-ready raster plus editable vector versions."""
        path = Path(path)
        outputs = []
        png_path = path.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(png_path.name)
        for suffix in (".pdf", ".svg"):
            vector_path = path.with_suffix(suffix)
            fig.savefig(vector_path, bbox_inches="tight", facecolor="white")
            outputs.append(vector_path.name)
        print(f"  Saved: {path.parent}/{{{', '.join(outputs)}}}")

    def format_step(x, _pos):
        if abs(x) >= 1000:
            return f"{x / 1000:g}k"
        return f"{int(x)}" if float(x).is_integer() else f"{x:g}"

    def style_axes(ax, xlabel="Training step", ylabel=None):
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(format_step))
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.65)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.025)
        ax.tick_params(axis="both", which="major", length=3.5, width=0.8, color="#4A4A4A")
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#4A4A4A")
            ax.spines[spine].set_linewidth(0.8)

    def set_padded_ylim(ax, values, ylim=None):
        if ylim:
            ax.set_ylim(*ylim)
            return
        clean = [float(v) for v in values if np.isfinite(float(v))]
        if not clean:
            return
        lo, hi = min(clean), max(clean)
        if lo == hi:
            pad = max(0.01, abs(lo) * 0.05)
        else:
            pad = (hi - lo) * 0.10
        ax.set_ylim(lo - pad, hi + pad)

    def best_point(steps, values, lower_is_better):
        clean = [(i, float(v)) for i, v in enumerate(values) if np.isfinite(float(v))]
        if not clean:
            return None
        idx, _ = min(clean, key=lambda item: item[1]) if lower_is_better else max(clean, key=lambda item: item[1])
        return steps[idx], values[idx]

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    if phase1_hist and phase1_hist.get("log_history"):
        train_logs, eval_logs = split_log(phase1_hist["log_history"])
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle("Stage 2 — Phase 1: BART ParaDetox Warm-up", fontsize=13, fontweight="bold")
        c = CONDITION_COLORS["phase1"]

        ax = axes[0]
        if train_logs:
            steps  = [e["step"] for e in train_logs]
            losses = [e["loss"] for e in train_logs]
            ax.plot(steps, losses, color=c, linewidth=1, alpha=0.4)
            w = max(5, len(losses) // 20)
            sm_steps, sm_losses = smooth_xy(steps, losses, w)
            ax.plot(sm_steps, sm_losses, color=c, linewidth=2.5, label="smoothed")
        ax.set_title("Training Loss")
        style_axes(ax, ylabel="Loss")

        ax = axes[1]
        if eval_logs:
            ax.plot([e["step"] for e in eval_logs], [e["eval_loss"] for e in eval_logs],
                    color=c, linewidth=2, marker="o", markersize=4)
        ax.set_title("Validation Loss")
        style_axes(ax, ylabel="Eval Loss")

        ax = axes[2]
        rl = [e for e in eval_logs if "eval_rougeL" in e]
        if rl:
            ax.plot([e["step"] for e in rl], [e["eval_rougeL"] for e in rl],
                    color=c, linewidth=2, marker="s", markersize=4)
        ax.set_title("Validation ROUGE-L")
        style_axes(ax, ylabel="ROUGE-L")

        ax = axes[3]
        sta = [e for e in eval_logs if "eval_sta" in e]
        if sta:
            ax.plot([e["step"] for e in sta], [e["eval_sta"] for e in sta],
                    color=c, linewidth=2, marker="^", markersize=5)
            ax.set_ylim(0, 1.05)
            ax.axhline(y=1.0, color="#aaa", linestyle="--", linewidth=1, alpha=0.5)
        else:
            ax.text(0.5, 0.5, "STA not recorded\n(old run)", ha="center", va="center",
                    transform=ax.transAxes, color="#aaa", fontsize=10)
        ax.set_title("Validation STA\n(↑ = more non-toxic outputs)")
        style_axes(ax, ylabel="STA")

        res = phase1_hist.get("results", {})
        info = []
        if res.get("total_steps"):     info.append(f"Steps: {res['total_steps']}")
        if res.get("best_metric"):     info.append(f"Best rougeL: {res['best_metric']:.4f}")
        if phase1_hist.get("recovered"): info.append("(recovered from trainer_state.json)")
        if info:
            fig.text(0.5, -0.03, "   |   ".join(info), ha="center", fontsize=9, color="#555",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

        plt.tight_layout()
        p = out_dir / "phase1_curves.png"
        save_figure(fig, p)
        plt.close(fig)

    def plot_phase2_metric(
        ax,
        metric_key,
        metric_label,
        marker="o",
        ylim=None,
        title=None,
        highlight_best=False,
        legend=True,
    ):
        any_data = False
        all_values = []
        lower_is_better = metric_key in LOWER_IS_BETTER
        for h in phase2_hists:
            cond = canonicalize_condition(h.get("condition", "unknown"))
            _, eval_logs = split_log(h["log_history"])
            entries = [e for e in eval_logs if metric_key in e]
            if entries:
                color = CONDITION_COLORS.get(cond, "#607D8B")
                steps = [e["step"] for e in entries]
                values = [e[metric_key] for e in entries]
                all_values.extend(values)
                ax.plot(
                    steps,
                    values,
                    color=color,
                    linewidth=2.2,
                    marker=marker,
                    markersize=4.8,
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    label=CONDITION_LABELS.get(cond, cond),
                )
                if highlight_best:
                    point = best_point(steps, values, lower_is_better)
                    if point is not None:
                        ax.scatter(
                            [point[0]],
                            [point[1]],
                            color=color,
                            edgecolor="white",
                            linewidth=0.7,
                            marker="*",
                            s=105,
                            zorder=4,
                        )
                any_data = True
        ax.set_title(title or metric_label)
        style_axes(ax, ylabel=metric_label)
        set_padded_ylim(ax, all_values, ylim)
        if legend:
            ax.legend(loc="best", frameon=True, framealpha=0.94, edgecolor="#D6D6D6")
        return any_data

    # ── Phase 2 per-metric comparison ─────────────────────────────────────────
    for metric_key, metric_label, fname, ylim in [
        ("eval_loss",   "Validation Loss",    "phase2_loss_curves.png", None),
        ("eval_rougeL", "Validation ROUGE-L", "phase2_rouge_curves.png", None),
        ("eval_sta",    "Text STA (↑ = more non-toxic outputs)", "phase2_sta_curves.png", (0, 1.05)),
        ("eval_multimodal_sta", "VisualBERT STA (↑ = more non-hateful memes)", "phase2_multimodal_sta_curves.png", (0, 1.05)),
        ("eval_text_toxicity_drop", "Text Toxicity Drop vs Original (↑ better)", "phase2_text_toxicity_drop_curves.png", None),
        ("eval_multimodal_toxicity_drop", "VisualBERT Hate-Prob. Drop vs Original (↑ better)", "phase2_multimodal_toxicity_drop_curves.png", None),
        ("eval_pred_toxicity_mean", "Generated Text Toxicity Prob. (↓ better)", "phase2_text_toxicity_mean_curves.png", (0, 1.05)),
        ("eval_multimodal_hate_prob_mean", "Generated VisualBERT Hate Prob. (↓ better)", "phase2_multimodal_hate_prob_curves.png", (0, 1.05)),
        ("eval_detox_quality", "Detox Quality Selection Score (↑ better)", "phase2_detox_quality_curves.png", None),
        ("eval_copy_rate_high", "High Source-Copy Rate (↓ better)", "phase2_copy_rate_high_curves.png", (0, 1.05)),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        any_data = plot_phase2_metric(
            ax,
            metric_key,
            metric_label,
            ylim=ylim,
            title=metric_label,
            highlight_best=metric_key in {"eval_loss", "eval_detox_quality"},
        )
        if any_data:
            plt.tight_layout()
            p = out_dir / fname
            save_figure(fig, p)
        plt.close(fig)

    # ── Poster/paper figure: the two most important validation curves ──────────
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharex=True)
    focus_specs = [
        ("eval_loss", "Validation loss", "Validation loss (↓)"),
        ("eval_detox_quality", "Detox quality", "Detox quality (↑)"),
    ]
    any_focus_data = False
    handles = labels = None
    for idx, (ax, (metric_key, title, ylabel)) in enumerate(zip(axes, focus_specs)):
        panel_has_data = plot_phase2_metric(
            ax,
            metric_key,
            ylabel,
            title=f"({chr(ord('a') + idx)}) {title}",
            highlight_best=True,
            legend=False,
        )
        any_focus_data = any_focus_data or panel_has_data
        if panel_has_data and handles is None:
            handles, labels = ax.get_legend_handles_labels()
    if any_focus_data:
        if handles and labels:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(4, len(labels)),
                frameon=False,
                handlelength=2.0,
                columnspacing=1.2,
                markerscale=0.85,
                borderaxespad=0.0,
            )
        plt.tight_layout(rect=(0, 0, 1, 0.90))
        save_figure(fig, out_dir / "phase2_validation_loss_detox_quality_publication.png")
    plt.close(fig)

    # ── Phase 2 toxicity dashboard ───────────────────────────────────────────
    toxicity_panels = [
        ("eval_sta", "Text STA\n(↑ non-toxic)", "^", (0, 1.05)),
        ("eval_multimodal_sta", "VisualBERT STA\n(↑ non-hateful)", "^", (0, 1.05)),
        ("eval_text_toxicity_drop", "Text Toxicity Drop\n(↑ detoxified)", "o", None),
        ("eval_multimodal_toxicity_drop", "VisualBERT Hate-Prob. Drop\n(↑ less hateful)", "o", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Stage 2 — Phase 2: Toxicity Metrics by Condition", fontsize=13, fontweight="bold")
    any_toxicity_data = False
    handles = labels = None
    for ax, (metric_key, metric_label, marker, ylim) in zip(axes.ravel(), toxicity_panels):
        panel_has_data = plot_phase2_metric(ax, metric_key, metric_label, marker=marker, ylim=ylim, legend=False)
        any_toxicity_data = any_toxicity_data or panel_has_data
        if panel_has_data and handles is None:
            handles, labels = ax.get_legend_handles_labels()
    if any_toxicity_data:
        if handles and labels:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), framealpha=0.9)
        plt.tight_layout(rect=(0, 0.06, 1, 0.96))
        p = out_dir / "phase2_toxicity_curves.png"
        save_figure(fig, p)
    plt.close(fig)

    # ── Phase 2 training loss ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Stage 2 — Phase 2: Training Loss by Condition", fontsize=12, fontweight="bold")
    any_data = False
    for h in phase2_hists:
        cond = canonicalize_condition(h.get("condition", "unknown"))
        train_logs, _ = split_log(h["log_history"])
        if train_logs:
            steps  = [e["step"] for e in train_logs]
            losses = [e["loss"] for e in train_logs]
            ax.plot(
                steps,
                losses,
                color=CONDITION_COLORS.get(cond, "#607D8B"),
                linewidth=0.8,
                alpha=0.16,
            )
            w = max(5, len(losses) // 20)
            sm_steps, sm = smooth_xy(steps, losses, w)
            ax.plot(sm_steps, sm, color=CONDITION_COLORS.get(cond, "#607D8B"),
                    linewidth=2.5, label=CONDITION_LABELS.get(cond, cond))
            any_data = True
    if any_data:
        style_axes(ax, ylabel="Training Loss (moving average)")
        ax.legend(loc="best", frameon=True, framealpha=0.94, edgecolor="#D6D6D6")
        plt.tight_layout()
        p = out_dir / "phase2_train_loss.png"
        save_figure(fig, p)
    plt.close(fig)

    # ── 4-panel summary ───────────────────────────────────────────────────────
    if phase1_hist or phase2_hists:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.suptitle("Stage 2 Training Overview — All Phases", fontsize=13, fontweight="bold")

        # P1 train loss
        ax = axes[0]; ax.set_title("P1 Training Loss")
        if phase1_hist and phase1_hist.get("log_history"):
            tl, _ = split_log(phase1_hist["log_history"])
            if tl:
                steps = [e["step"] for e in tl]; losses = [e["loss"] for e in tl]
                ax.plot(steps, losses, color=CONDITION_COLORS["phase1"], linewidth=1, alpha=0.4)
                w = max(5, len(losses) // 20)
                sm_steps, sm_losses = smooth_xy(steps, losses, w)
                ax.plot(sm_steps, sm_losses, color=CONDITION_COLORS["phase1"], linewidth=2.5)
        style_axes(ax, ylabel="Loss")

        # P1 eval loss
        ax = axes[1]; ax.set_title("P1 Validation Loss")
        if phase1_hist and phase1_hist.get("log_history"):
            _, el = split_log(phase1_hist["log_history"])
            if el:
                ax.plot([e["step"] for e in el], [e["eval_loss"] for e in el],
                        color=CONDITION_COLORS["phase1"], linewidth=2, marker="o", markersize=4)
        style_axes(ax, ylabel="Eval Loss")

        # P2 eval loss
        ax = axes[2]; ax.set_title("P2 Validation Loss")
        for h in phase2_hists:
            cond = canonicalize_condition(h.get("condition", "unknown"))
            _, el = split_log(h["log_history"])
            entries = [e for e in el if "eval_loss" in e]
            if entries:
                ax.plot([e["step"] for e in entries], [e["eval_loss"] for e in entries],
                        color=CONDITION_COLORS.get(cond, "#607D8B"), linewidth=2, marker="o",
                        markersize=3, label=CONDITION_LABELS.get(cond, cond))
        style_axes(ax, ylabel="Eval Loss")
        ax.legend(fontsize=7, frameon=True, framealpha=0.94, edgecolor="#D6D6D6")

        # P2 rougeL
        ax = axes[3]; ax.set_title("P2 ROUGE-L")
        for h in phase2_hists:
            cond = canonicalize_condition(h.get("condition", "unknown"))
            _, el = split_log(h["log_history"])
            entries = [e for e in el if "eval_rougeL" in e]
            if entries:
                ax.plot([e["step"] for e in entries], [e["eval_rougeL"] for e in entries],
                        color=CONDITION_COLORS.get(cond, "#607D8B"), linewidth=2, marker="s",
                        markersize=3, label=CONDITION_LABELS.get(cond, cond))
        style_axes(ax, ylabel="ROUGE-L")
        ax.legend(fontsize=7, frameon=True, framealpha=0.94, edgecolor="#D6D6D6")

        plt.tight_layout()
        p = out_dir / "all_phases_summary.png"
        save_figure(fig, p)
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recover training metrics from existing checkpoints")
    parser.add_argument("--scratch_root",   type=str, default=None,
                        help="Auto-discover all known checkpoint dirs under this root")
    parser.add_argument("--checkpoint_suffix", type=str, default="",
                        help=(
                            "Optional suffix inserted before _checkpoint when auto-discovering "
                            "phase2 dirs, e.g. _explicit_detox for "
                            "hmr_stage2_phase2_full_explicit_detox_checkpoint"
                        ))
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Single checkpoint directory to recover")
    parser.add_argument("--phase",      type=str, default=None, choices=["phase1", "phase2"],
                        help="Required when using --checkpoint_dir")
    parser.add_argument("--condition",  type=str, default=None,
                        choices=["full", "target_only", "visual_only", "none"],
                        help="Required for phase2 when using --checkpoint_dir")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save PNG plots")
    parser.add_argument("--no_plots",   action="store_true",
                        help="Only write training_history.json, skip plotting")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        if not args.no_plots:
            print("WARNING: matplotlib/numpy not found — running with --no_plots")
            args.no_plots = True
        plt = np = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase1_hist   = None
    phase2_hists  = []

    if args.scratch_root:
        root = Path(args.scratch_root)
        print(f"\nAuto-discovering checkpoints under {root} ...")
        for dir_name, (phase, condition) in KNOWN_DIRS.items():
            if phase == "phase2" and args.checkpoint_suffix:
                dir_name = dir_name.replace("_checkpoint", f"{args.checkpoint_suffix}_checkpoint")
            cdir = root / dir_name
            if not cdir.exists():
                print(f"  [SKIP] {dir_name} — not found")
                continue
            hist = recover_checkpoint(cdir, phase, condition)
            if hist:
                if phase == "phase1":
                    phase1_hist = hist
                else:
                    hist["condition"] = canonicalize_condition(condition)   # ensure it's set
                    phase2_hists.append(hist)

    elif args.checkpoint_dir:
        if not args.phase:
            print("ERROR: --phase is required with --checkpoint_dir")
            sys.exit(1)
        cdir = Path(args.checkpoint_dir)
        hist = recover_checkpoint(cdir, args.phase, args.condition)
        if hist:
            if args.phase == "phase1":
                phase1_hist = hist
            else:
                hist["condition"] = canonicalize_condition(args.condition)
                phase2_hists.append(hist)
    else:
        print("ERROR: provide --scratch_root or --checkpoint_dir")
        sys.exit(1)

    # ── summary of what was recovered ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOVERY SUMMARY")
    print("=" * 60)
    if phase1_hist:
        n = len(phase1_hist.get("log_history", []))
        print(f"  Phase 1:  ✓  ({n} log entries)")
    else:
        print(f"  Phase 1:  ✗  (no recoverable data)")
    for h in phase2_hists:
        n = len(h.get("log_history", []))
        print(f"  Phase 2 [{h.get('condition','?')}]:  ✓  ({n} log entries)")
    for cond in ["full", "target_only", "visual_only", "none"]:
        if not any(h.get("condition") == cond for h in phase2_hists):
            print(f"  Phase 2 [{cond}]:  ✗  (no recoverable data)")

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots and (phase1_hist or phase2_hists):
        if plt is None or np is None:
            print("\nSkipping plots (matplotlib not available)")
        else:
            import matplotlib
            matplotlib.rcParams.update({
                "font.family": "DejaVu Sans",
                "font.size": 9.5,
                "axes.labelsize": 10.5,
                "axes.titlesize": 11,
                "axes.titleweight": "semibold",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "figure.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.dpi": 600,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "lines.solid_capstyle": "round",
                "lines.solid_joinstyle": "round",
            })
            print(f"\nGenerating plots → {out_dir}")
            plot_all(phase1_hist, phase2_hists, out_dir, plt, np)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
