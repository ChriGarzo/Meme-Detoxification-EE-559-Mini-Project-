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
    "hmr_stage2_phase2_attack_only_checkpoint": ("phase2", "attack_only"),
    "hmr_stage2_phase2_none_checkpoint":    ("phase2", "none"),
}

CONDITION_COLORS = {
    "full":        "#2196F3",
    "target_only": "#FF9800",
    "attack_only": "#4CAF50",
    "none":        "#9C27B0",
    "phase1":      "#E53935",
}
CONDITION_LABELS = {
    "full":        "Full [T+A+M]",
    "target_only": "Target only [T]",
    "attack_only": "Attack only [A]",
    "none":        "No cond. [—]",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def find_trainer_state(checkpoint_dir: Path) -> Optional[Path]:
    """
    Look for trainer_state.json.
    Priority:
      1. Direct file: <checkpoint_dir>/trainer_state.json
      2. Inside the latest checkpoint subdir: <checkpoint_dir>/checkpoint-NNNN/trainer_state.json
         (pick the one with the highest step number — that has the full log)
    """
    direct = checkpoint_dir / "trainer_state.json"
    if direct.exists():
        return direct

    # Find all checkpoint-NNN subdirs
    subdirs = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0
    )
    # Walk from highest step downward and return the first one that has trainer_state.json
    for sub in reversed(subdirs):
        candidate = sub / "trainer_state.json"
        if candidate.exists():
            return candidate

    return None


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
        print(f"    ✓ training_history.json already exists — loading it")
        with open(existing) as f:
            return json.load(f)

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


# ── plotting (shared with plot_training_curves.py) ────────────────────────────

def split_log(log_history):
    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs  = [e for e in log_history if "eval_loss" in e]
    return train_logs, eval_logs


def plot_all(phase1_hist, phase2_hists, out_dir, plt, np):
    """Reproduce all plots from plot_training_curves.py inline."""

    def smooth(values, window):
        if len(values) <= window:
            return values
        return list(np.convolve(values, np.ones(window) / window, mode="valid"))

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
            sm_losses = smooth(losses, w)
            ax.plot(steps[w - 1:], sm_losses, color=c, linewidth=2.5, label="smoothed")
        ax.set_title("Training Loss"); ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)

        ax = axes[1]
        if eval_logs:
            ax.plot([e["step"] for e in eval_logs], [e["eval_loss"] for e in eval_logs],
                    color=c, linewidth=2, marker="o", markersize=4)
        ax.set_title("Validation Loss"); ax.set_xlabel("Step"); ax.set_ylabel("Eval Loss"); ax.grid(True, alpha=0.3)

        ax = axes[2]
        rl = [e for e in eval_logs if "eval_rougeL" in e]
        if rl:
            ax.plot([e["step"] for e in rl], [e["eval_rougeL"] for e in rl],
                    color=c, linewidth=2, marker="s", markersize=4)
        ax.set_title("Validation ROUGE-L"); ax.set_xlabel("Step"); ax.set_ylabel("ROUGE-L"); ax.grid(True, alpha=0.3)

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
        ax.set_title("Validation STA\n(↑ = more non-toxic outputs)"); ax.set_xlabel("Step"); ax.set_ylabel("STA"); ax.grid(True, alpha=0.3)

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
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")

    # ── Phase 2 per-metric comparison ─────────────────────────────────────────
    for metric_key, metric_label, fname in [
        ("eval_loss",   "Validation Loss",    "phase2_loss_curves.png"),
        ("eval_rougeL", "Validation ROUGE-L", "phase2_rouge_curves.png"),
        ("eval_sta",    "Validation STA (↑ = more non-toxic outputs)", "phase2_sta_curves.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title(f"Stage 2 — Phase 2: {metric_label} by Condition", fontsize=12, fontweight="bold")
        any_data = False
        for h in phase2_hists:
            cond = h.get("condition", "unknown")
            _, eval_logs = split_log(h["log_history"])
            entries = [e for e in eval_logs if metric_key in e]
            if entries:
                ax.plot([e["step"] for e in entries], [e[metric_key] for e in entries],
                        color=CONDITION_COLORS.get(cond, "#607D8B"), linewidth=2,
                        marker="o", markersize=4, label=CONDITION_LABELS.get(cond, cond))
                any_data = True
        if any_data:
            ax.set_xlabel("Step"); ax.set_ylabel(metric_label)
            ax.legend(loc="best", framealpha=0.9); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = out_dir / fname
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"  Saved: {p}")
        plt.close(fig)

    # ── Phase 2 training loss ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Stage 2 — Phase 2: Training Loss by Condition", fontsize=12, fontweight="bold")
    any_data = False
    for h in phase2_hists:
        cond = h.get("condition", "unknown")
        train_logs, _ = split_log(h["log_history"])
        if train_logs:
            steps  = [e["step"] for e in train_logs]
            losses = [e["loss"] for e in train_logs]
            w = max(5, len(losses) // 20)
            sm = smooth(losses, w)
            ax.plot(steps[w - 1:], sm, color=CONDITION_COLORS.get(cond, "#607D8B"),
                    linewidth=2.5, label=CONDITION_LABELS.get(cond, cond))
            any_data = True
    if any_data:
        ax.set_xlabel("Step"); ax.set_ylabel("Training Loss (smoothed)")
        ax.legend(loc="best", framealpha=0.9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = out_dir / "phase2_train_loss.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved: {p}")
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
                ax.plot(steps[w-1:], smooth(losses, w), color=CONDITION_COLORS["phase1"], linewidth=2.5)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)

        # P1 eval loss
        ax = axes[1]; ax.set_title("P1 Validation Loss")
        if phase1_hist and phase1_hist.get("log_history"):
            _, el = split_log(phase1_hist["log_history"])
            if el:
                ax.plot([e["step"] for e in el], [e["eval_loss"] for e in el],
                        color=CONDITION_COLORS["phase1"], linewidth=2, marker="o", markersize=4)
        ax.set_xlabel("Step"); ax.set_ylabel("Eval Loss"); ax.grid(True, alpha=0.3)

        # P2 eval loss
        ax = axes[2]; ax.set_title("P2 Validation Loss")
        for h in phase2_hists:
            cond = h.get("condition", "unknown")
            _, el = split_log(h["log_history"])
            entries = [e for e in el if "eval_loss" in e]
            if entries:
                ax.plot([e["step"] for e in entries], [e["eval_loss"] for e in entries],
                        color=CONDITION_COLORS.get(cond, "#607D8B"), linewidth=2, marker="o",
                        markersize=3, label=CONDITION_LABELS.get(cond, cond))
        ax.set_xlabel("Step"); ax.set_ylabel("Eval Loss"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # P2 rougeL
        ax = axes[3]; ax.set_title("P2 ROUGE-L")
        for h in phase2_hists:
            cond = h.get("condition", "unknown")
            _, el = split_log(h["log_history"])
            entries = [e for e in el if "eval_rougeL" in e]
            if entries:
                ax.plot([e["step"] for e in entries], [e["eval_rougeL"] for e in entries],
                        color=CONDITION_COLORS.get(cond, "#607D8B"), linewidth=2, marker="s",
                        markersize=3, label=CONDITION_LABELS.get(cond, cond))
        ax.set_xlabel("Step"); ax.set_ylabel("ROUGE-L"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        p = out_dir / "all_phases_summary.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {p}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recover training metrics from existing checkpoints")
    parser.add_argument("--scratch_root",   type=str, default=None,
                        help="Auto-discover all known checkpoint dirs under this root")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Single checkpoint directory to recover")
    parser.add_argument("--phase",      type=str, default=None, choices=["phase1", "phase2"],
                        help="Required when using --checkpoint_dir")
    parser.add_argument("--condition",  type=str, default=None,
                        choices=["full", "target_only", "attack_only", "none"],
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
            cdir = root / dir_name
            if not cdir.exists():
                print(f"  [SKIP] {dir_name} — not found")
                continue
            hist = recover_checkpoint(cdir, phase, condition)
            if hist:
                if phase == "phase1":
                    phase1_hist = hist
                else:
                    hist["condition"] = condition   # ensure it's set
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
                hist["condition"] = args.condition
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
    for cond in ["full", "target_only", "attack_only", "none"]:
        if not any(h.get("condition") == cond for h in phase2_hists):
            print(f"  Phase 2 [{cond}]:  ✗  (no recoverable data)")

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots and (phase1_hist or phase2_hists):
        if plt is None or np is None:
            print("\nSkipping plots (matplotlib not available)")
        else:
            import matplotlib
            matplotlib.rcParams.update({
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.facecolor": "white",
            })
            print(f"\nGenerating plots → {out_dir}")
            plot_all(phase1_hist, phase2_hists, out_dir, plt, np)

    print(f"\nDone. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
