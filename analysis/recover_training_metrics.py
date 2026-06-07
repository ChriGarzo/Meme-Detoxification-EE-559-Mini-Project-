"""
Recover training metrics from already-completed runs and produce training plots.

HuggingFace Trainer always writes trainer_state.json inside each checkpoint
subdirectory (e.g. checkpoint-1646/trainer_state.json).  This script finds
those files, consolidates the log_history, and produces:

  <checkpoint_dir>/training_history.json   ← same format as our new code writes
  <output_dir>/stage2_*.png                (--stage stage2, default)
  <output_dir>/stage3_*.png                (--stage stage3)

Usage — Stage 2 (BART LoRA fine-tuning, all 4 conditions):
    python analysis/recover_training_metrics.py \\
        --stage stage2 \\
        --scratch_root /scratch/stages \\
        --checkpoint_suffix _explicit_detox \\
        --output_dir /scratch/plots/stage_2_training_plots

Usage — Stage 3 (proxy network):
    python analysis/recover_training_metrics.py \\
        --stage stage3 \\
        --scratch_root /scratch/stages \\
        --checkpoint_suffix _explicit_detox \\
        --output_dir /scratch/plots/stage_3_training_plots

Usage — single Stage 2 checkpoint:
    python analysis/recover_training_metrics.py \\
        --stage stage2 \\
        --checkpoint_dir /scratch/stages/hmr_stage2_phase2_full_explicit_detox_checkpoint \\
        --condition full \\
        --output_dir /scratch/plots/stage_2_training_plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ── known Stage 2 checkpoint directory → (phase, condition) mapping ───────────
# dir_name templates; --checkpoint_suffix is injected before _checkpoint at
# runtime, e.g. _explicit_detox → hmr_stage2_phase2_full_explicit_detox_checkpoint
KNOWN_DIRS = {
    "hmr_stage2_phase2_full_checkpoint":         ("stage2", "full"),
    "hmr_stage2_phase2_target_only_checkpoint":  ("stage2", "target_only"),
    "hmr_stage2_phase2_visual_only_checkpoint":  ("stage2", "visual_only"),
    "hmr_stage2_phase2_none_checkpoint":         ("stage2", "none"),
}

CONDITION_COLORS = {
    # Okabe-Ito inspired, colorblind-safe palette.
    "full":        "#0072B2",
    "target_only": "#E69F00",
    "visual_only": "#009E73",
    "none":        "#CC79A7",
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
    "eval_copy_rate_high",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_trainer_state(checkpoint_dir: Path) -> Optional[Path]:
    """Locate trainer_state.json, preferring the checkpoint subdir with the newest mtime.

    Newest-mtime selection avoids picking a stale high-numbered checkpoint from
    a previous run when the directory contains interleaved old and new checkpoints.

    Search order:
      1. <checkpoint_dir>/trainer_state.json  (direct, written by Trainer on completion)
      2. <checkpoint_dir>/checkpoint-NNNN/trainer_state.json  (newest by mtime)
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
    """Wrap a raw trainer_state.json into the training_history.json schema."""
    log_history = state.get("log_history", [])
    best_metric = state.get("best_metric")
    best_checkpoint = state.get("best_model_checkpoint")
    total_steps = state.get("global_step", 0)

    run_config = {
        "recovered_from": "trainer_state.json",
        "checkpoint_dir": str(checkpoint_dir),
    }
    if condition:
        run_config["condition"] = condition

    return {
        "phase": "stage2_meme_finetune",
        "condition": condition,
        "run_config": run_config,
        "hardware": {},
        "results": {
            "total_steps": total_steps,
            "best_metric": best_metric,
            "best_model_checkpoint": str(best_checkpoint) if best_checkpoint else None,
            "training_duration_seconds": None,
        },
        "log_history": log_history,
        "recovered": True,
    }


def recover_checkpoint(checkpoint_dir: Path, phase: str, condition: Optional[str]) -> Optional[Dict]:
    print(f"\n  Checkpoint: {checkpoint_dir.name}")

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

    out_path = checkpoint_dir / "training_history.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"    ✓ Saved training_history.json to {checkpoint_dir.name}/")
    except PermissionError:
        print(f"    ! Could not write to {checkpoint_dir} (permission denied) — using in-memory copy")

    return history


def canonicalize_condition(condition: Optional[str]) -> Optional[str]:
    """Return condition unchanged; kept as an extension point for future remapping."""
    return condition


# ── Plotting — shared style helpers ──────────────────────────────────────────

def _apply_rcparams(matplotlib):
    matplotlib.rcParams.update({
        "font.family":           "DejaVu Sans",
        "font.size":             9.5,
        "axes.labelsize":        10.5,
        "axes.titlesize":        11,
        "axes.titleweight":      "semibold",
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.linewidth":        0.8,
        "legend.fontsize":       9,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "figure.facecolor":      "white",
        "savefig.facecolor":     "white",
        "savefig.dpi":           600,
        "pdf.fonttype":          42,
        "ps.fonttype":           42,
        "svg.fonttype":          "none",
        "lines.solid_capstyle":  "round",
        "lines.solid_joinstyle": "round",
    })


def _save_png(fig, path, dpi=600):
    path = Path(path).with_suffix(".png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")


# ── Plotting — Stage 2 ────────────────────────────────────────────────────────

def split_log(log_history):
    """Partition Trainer log_history into training steps and eval checkpoints."""
    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs  = [e for e in log_history if "eval_loss" in e]
    return train_logs, eval_logs


def plot_all(stage2_hists, out_dir, plt, np):
    """Generate all Stage 2 training and evaluation plots from recovered histories."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    def smooth(values, window):
        if len(values) <= window:
            return values
        return list(np.convolve(values, np.ones(window) / window, mode="valid"))

    def smooth_xy(steps, values, window):
        if len(values) <= window:
            return steps, values
        return steps[window - 1:], smooth(values, window)

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
        pad = (hi - lo) * 0.10 if lo != hi else max(0.01, abs(lo) * 0.05)
        ax.set_ylim(lo - pad, hi + pad)

    def best_point(steps, values, lower_is_better):
        clean = [(i, float(v)) for i, v in enumerate(values) if np.isfinite(float(v))]
        if not clean:
            return None
        idx, _ = min(clean, key=lambda item: item[1]) if lower_is_better else max(clean, key=lambda item: item[1])
        return steps[idx], values[idx]

    def plot_stage2_metric(ax, metric_key, metric_label, ylim=None, highlight_best=False):
        any_data = False
        all_values = []
        lower_is_better = metric_key in LOWER_IS_BETTER
        for h in stage2_hists:
            cond = canonicalize_condition(h.get("condition", "unknown"))
            _, eval_logs = split_log(h["log_history"])
            entries = [e for e in eval_logs if metric_key in e]
            if entries:
                color = CONDITION_COLORS.get(cond, "#607D8B")
                steps = [e["step"] for e in entries]
                values = [e[metric_key] for e in entries]
                all_values.extend(values)
                ax.plot(steps, values, color=color, linewidth=2.2, marker="o",
                        markersize=4.8, markerfacecolor="white", markeredgewidth=1.1,
                        label=CONDITION_LABELS.get(cond, cond))
                if highlight_best:
                    point = best_point(steps, values, lower_is_better)
                    if point is not None:
                        ax.scatter([point[0]], [point[1]], color=color, edgecolor="white",
                                   linewidth=0.7, marker="*", s=105, zorder=4)
                any_data = True
        ax.set_title(metric_label)
        style_axes(ax, ylabel=metric_label)
        set_padded_ylim(ax, all_values, ylim)
        ax.legend(loc="best", frameon=True, framealpha=0.94, edgecolor="#D6D6D6")
        return any_data

    # ── Per-metric comparison panels ──────────────────────────────────────────
    for metric_key, metric_label, fname, ylim in [
        ("eval_loss",   "Validation Loss  (↓ better)",    "stage2_loss_curves.png", None),
        ("eval_rougeL", "Validation ROUGE-L  (↑ better)", "stage2_rouge_curves.png", None),
        ("eval_sta",    "Text STA  (↑ better)", "stage2_sta_curves.png", (0, 1.05)),
        ("eval_text_toxicity_drop", "Text Toxicity Drop vs Original  (↑ better)", "stage2_text_toxicity_drop_curves.png", None),
        ("eval_pred_toxicity_mean", "Generated Text Toxicity Prob.  (↓ better)", "stage2_text_toxicity_mean_curves.png", (0, 1.05)),
        ("eval_detox_quality", "Detox Quality  (↑ better)\nQ = Δtox + 0.10·ROUGE-L − 0.25·copy_rate", "stage2_detox_quality_curves.png", None),
        ("eval_copy_rate_high", "High Source-Copy Rate  (↓ better)", "stage2_copy_rate_high_curves.png", (0, 1.05)),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        any_data = plot_stage2_metric(ax, metric_key, metric_label, ylim=ylim,
                                      highlight_best=metric_key in {"eval_loss", "eval_detox_quality"})
        if any_data:
            plt.tight_layout()
            _save_png(fig, out_dir / fname)
        plt.close(fig)

    # ── Training loss with smoothed overlay ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Stage 2: Training Loss by Condition", fontsize=12, fontweight="bold")
    any_data = False
    for h in stage2_hists:
        cond = canonicalize_condition(h.get("condition", "unknown"))
        train_logs, _ = split_log(h["log_history"])
        if train_logs:
            steps  = [e["step"] for e in train_logs]
            losses = [e["loss"] for e in train_logs]
            ax.plot(steps, losses, color=CONDITION_COLORS.get(cond, "#607D8B"),
                    linewidth=0.8, alpha=0.16)
            w = max(5, len(losses) // 20)
            sm_steps, sm = smooth_xy(steps, losses, w)
            ax.plot(sm_steps, sm, color=CONDITION_COLORS.get(cond, "#607D8B"),
                    linewidth=2.5, label=CONDITION_LABELS.get(cond, cond))
            any_data = True
    if any_data:
        from matplotlib.ticker import FuncFormatter, MaxNLocator
        ax.set_xlabel("Training step")
        ax.set_ylabel("Training Loss (moving average)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{x/1000:g}k" if abs(x) >= 1000 else f"{int(x)}"
        ))
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.65)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.025)
        ax.legend(loc="best", frameon=True, framealpha=0.94, edgecolor="#D6D6D6")
        plt.tight_layout()
        _save_png(fig, out_dir / "stage2_train_loss.png")
    plt.close(fig)


# ── Plotting — Stage 3 (proxy network) ───────────────────────────────────────

def plot_proxy(history, out_dir, plt, np):
    """Generate Stage 3 proxy MSE loss plots (2 panels: train loss, val loss).

    history: dict with keys train_loss and val_loss (lists of float, one per epoch).
    """
    from matplotlib.ticker import MaxNLocator

    def style_axes(ax, ylabel=None):
        ax.set_xlabel("Epoch")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.65)
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.025)
        ax.tick_params(axis="both", which="major", length=3.5, width=0.8, color="#4A4A4A")
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#4A4A4A")
            ax.spines[spine].set_linewidth(0.8)

    def set_padded_ylim(ax, values):
        clean = [float(v) for v in values if np.isfinite(float(v))]
        if not clean:
            return
        lo, hi = min(clean), max(clean)
        pad = (hi - lo) * 0.10 if lo != hi else max(0.001, abs(lo) * 0.05)
        ax.set_ylim(lo - pad, hi + pad)

    color = "#0072B2"

    for key, title, fname in [
        ("train_loss", "Stage 3 Proxy — Training MSE Loss  (↓ better)",   "stage3_train_loss.png"),
        ("val_loss",   "Stage 3 Proxy — Validation MSE Loss  (↓ better)", "stage3_val_loss.png"),
    ]:
        values = history.get(key, [])
        if not values:
            print(f"  [SKIP] {fname} — no data for '{key}'")
            continue
        epochs = list(range(1, len(values) + 1))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title(title)
        ax.plot(epochs, values, color=color, linewidth=2.2, marker="o",
                markersize=4.8, markerfacecolor="white", markeredgewidth=1.1)
        style_axes(ax, ylabel="MSE Loss")
        set_padded_ylim(ax, values)
        plt.tight_layout()
        _save_png(fig, out_dir / fname)
        plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recover training metrics from existing checkpoints")
    parser.add_argument("--stage", type=str, default="stage2", choices=["stage2", "stage3"],
                        help="Which stage to plot: stage2 (BART LoRA) or stage3 (proxy network)")
    parser.add_argument("--scratch_root", type=str, default=None,
                        help="Auto-discover checkpoints under this root directory")
    parser.add_argument("--checkpoint_suffix", type=str, default="",
                        help=(
                            "Suffix inserted before _checkpoint when auto-discovering stage2 dirs, "
                            "e.g. _explicit_detox → hmr_stage2_phase2_full_explicit_detox_checkpoint"
                        ))
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Single Stage 2 checkpoint directory (requires --stage stage2)")
    parser.add_argument("--condition", type=str, default=None,
                        choices=["full", "target_only", "visual_only", "none"],
                        help="Condition label for --checkpoint_dir mode")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save PNG plots")
    parser.add_argument("--no_plots", action="store_true",
                        help="Only write training_history.json files, skip plotting")
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

    # ── Stage 2: BART LoRA ────────────────────────────────────────────────────
    if args.stage == "stage2":
        stage2_hists: List[Dict] = []

        if args.scratch_root:
            root = Path(args.scratch_root)
            print(f"\nAuto-discovering Stage 2 checkpoints under {root} ...")
            for dir_name, (phase, condition) in KNOWN_DIRS.items():
                resolved = dir_name
                if args.checkpoint_suffix:
                    resolved = resolved.replace("_checkpoint", f"{args.checkpoint_suffix}_checkpoint")
                cdir = root / resolved
                if not cdir.exists():
                    print(f"  [SKIP] {resolved} — not found")
                    continue
                hist = recover_checkpoint(cdir, phase, condition)
                if hist:
                    hist["condition"] = canonicalize_condition(condition)
                    stage2_hists.append(hist)

        elif args.checkpoint_dir:
            cdir = Path(args.checkpoint_dir)
            hist = recover_checkpoint(cdir, "stage2", args.condition)
            if hist:
                hist["condition"] = canonicalize_condition(args.condition)
                stage2_hists.append(hist)
        else:
            print("ERROR: provide --scratch_root or --checkpoint_dir")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("RECOVERY SUMMARY — Stage 2")
        print("=" * 60)
        for h in stage2_hists:
            n = len(h.get("log_history", []))
            print(f"  [{h.get('condition','?')}]:  ✓  ({n} log entries)")
        for cond in ["full", "target_only", "visual_only", "none"]:
            if not any(h.get("condition") == cond for h in stage2_hists):
                print(f"  [{cond}]:  ✗  (no recoverable data)")

        if not args.no_plots and stage2_hists:
            if plt is None or np is None:
                print("\nSkipping plots (matplotlib not available)")
            else:
                import matplotlib
                _apply_rcparams(matplotlib)
                print(f"\nGenerating Stage 2 plots → {out_dir}")
                plot_all(stage2_hists, out_dir, plt, np)

    # ── Stage 3: Proxy network ────────────────────────────────────────────────
    elif args.stage == "stage3":
        proxy_history: Optional[Dict] = None

        if args.scratch_root:
            root = Path(args.scratch_root)
            suffix = args.checkpoint_suffix  # e.g. "_explicit_detox"
            history_path = root / f"hmr_proxy_checkpoint{suffix}" / "training_history.json"
            if history_path.exists():
                with open(history_path) as f:
                    proxy_history = json.load(f)
                print(f"  ✓  {history_path}: "
                      f"{len(proxy_history.get('train_loss', []))} epochs")
            else:
                print(f"  ✗  Not found: {history_path}")
        else:
            print("ERROR: --scratch_root is required for --stage stage3")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("RECOVERY SUMMARY — Stage 3 (Proxy)")
        print("=" * 60)
        if proxy_history:
            print(f"  train_loss: {len(proxy_history.get('train_loss', []))} epochs")
            print(f"  val_loss:   {len(proxy_history.get('val_loss', []))} epochs")
        else:
            print("  ✗  No proxy training history found.")

        if not args.no_plots and proxy_history:
            if plt is None or np is None:
                print("\nSkipping plots (matplotlib not available)")
            else:
                import matplotlib
                _apply_rcparams(matplotlib)
                print(f"\nGenerating Stage 3 plots → {out_dir}")
                plot_proxy(proxy_history, out_dir, plt, np)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
