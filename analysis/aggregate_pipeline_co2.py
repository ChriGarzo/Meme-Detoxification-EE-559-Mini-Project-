"""
Aggregate CodeCarbon emission records across all pipeline stages.

Stages lacking direct CodeCarbon coverage (Stage 0, Stage 2 training) are
estimated analytically; all other stages use measured CSV files.

Stage accounting:
  Stage 0  — OCR + CLIP filtering   ESTIMATED: manifest row count × assumed rate.
             Future runs emit emissions_stage0_*.csv and are used when present.
  Stage 1A — LLaVA-Next explanations  (8 shards, CodeCarbon tracked)
  Stage 1B — LLaVA-Next pseudo-rewrites (8 shards, CodeCarbon tracked)
  Stage 2  — BART LoRA training  ESTIMATED: duration × GPU power when no CSV found.
  Stage 3  — BART-base inference (4 conditions, CodeCarbon tracked)
  Stage 3  — BART-finetuned inference (4 conditions, CodeCarbon tracked)
  Proxy    — Proxy inference (optional, CodeCarbon tracked)
  DetoxLLM — DetoxLLM baseline (optional, CodeCarbon tracked)

Outputs: pipeline_co2_summary.json, pipeline_co2_summary.tsv

Usage:
    python analysis/aggregate_pipeline_co2.py \\
        --scratch_dir /scratch \\
        --output_dir /scratch/hmr_co2_summary
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# A100 TDP = 400 W; seq2seq training at batch=8 typically reaches 60-70% TDP
# (~240-280 W). 270 W is a conservative midpoint used only when no CSV is found.
GPU_POWER_TRAINING_W = 270.0

# EasyOCR dominates at ~0.30 s/image; CLIP adds ~0.03 s/image → 0.33 s total.
# GPU load is lighter than LLaVA inference but heavier than BART eval → ~150 W.
# The three datasets ran as independent parallel jobs; energy is summed.
STAGE0_SEC_PER_IMAGE = 0.33
GPU_POWER_STAGE0_W = 150.0

STAGE0_MANIFEST_GLOBS = [
    "hmr_data/harmeme/manifest.csv",
    "hmr_data/mami/manifest.csv",
    "hmr_data/mmhs150k/manifest.csv",
]
# Future filter_meme_images.py runs emit per-dataset CSVs here.
STAGE0_EMISSIONS_GLOBS = [
    "hmr_data/harmeme/emissions_stage0_harmeme.csv",
    "hmr_data/mami/emissions_stage0_mami.csv",
    "hmr_data/mmhs150k/emissions_stage0_mmhs150k.csv",
]

# Each entry: (stage_id, description, list of Path globs relative to scratch_dir, _unused_flag)
STAGE_DEFS = [
    (
        "stage1a_llava_explanations",
        "LLaVA-Next explanations (8 shards)",
        ["hmr_stage1_output/emissions_shard*.csv"],
        False,
    ),
    (
        "stage1b_llava_rewrites",
        "LLaVA-Next pseudo-rewrites (8 shards)",
        ["hmr_stage1_output/emissions_rewrite_only_shard*.csv"],
        False,
    ),
    (
        "stage3_bart_base_eval",
        "BART-base inference / ablation (4 conditions)",
        [
            "hmr_eval_bart_base_full/emissions.csv",
            "hmr_eval_bart_base_target_only/emissions.csv",
            "hmr_eval_bart_base_visual_only/emissions.csv",
            "hmr_eval_bart_base_none/emissions.csv",
        ],
        False,
    ),
    (
        "stage3_bart_finetuned_eval",
        "BART-finetuned inference (4 conditions)",
        [
            "hmr_eval_stage2_full/emissions.csv",
            "hmr_eval_stage2_target_only/emissions.csv",
            "hmr_eval_stage2_visual_only/emissions.csv",
            "hmr_eval_stage2_none/emissions.csv",
        ],
        False,
    ),
    (
        "proxy_inference",
        "Proxy + BART inference (optional)",
        ["hmr_eval_proxy_*/emissions.csv"],
        False,
    ),
    (
        "detoxllm_inference",
        "DetoxLLM baseline inference (optional)",
        ["hmr_eval_detoxllm*/emissions.csv"],
        False,
    ),
]

TRAINING_STAGE_DEF = (
    "stage2_bart_training",
    "BART LoRA training (4 conditions)",
    [
        "hmr_stage2_full_checkpoint/training_history.json",
        "hmr_stage2_target_only_checkpoint/training_history.json",
        "hmr_stage2_visual_only_checkpoint/training_history.json",
        "hmr_stage2_none_checkpoint/training_history.json",
    ],
)

TRAINING_EMISSIONS_GLOBS = [
    "hmr_stage2_full_checkpoint/emissions.csv",
    "hmr_stage2_target_only_checkpoint/emissions.csv",
    "hmr_stage2_visual_only_checkpoint/emissions.csv",
    "hmr_stage2_none_checkpoint/emissions.csv",
]


def read_emissions_csv(path: Path) -> List[Dict]:
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
    return rows


def aggregate_csv_rows(rows: List[Dict]) -> Dict:
    """Sum duration, energy, and CO2 across CodeCarbon CSV rows; average GPU power."""
    total_duration = 0.0
    total_energy = 0.0
    total_co2 = 0.0
    gpu_powers = []

    for row in rows:
        try:
            total_duration += float(row.get("duration") or 0)
            total_energy += float(row.get("energy_consumed") or 0)
            total_co2 += float(row.get("emissions") or 0)
            gp = float(row.get("gpu_power") or 0)
            if gp > 0:
                gpu_powers.append(gp)
        except (ValueError, TypeError):
            pass

    return {
        "duration_s": total_duration,
        "energy_kwh": total_energy,
        "co2_kg": total_co2,
        "mean_gpu_power_w": float(np.mean(gpu_powers)) if gpu_powers else None,
        "num_rows": len(rows),
    }


def derive_carbon_intensity(all_rows: List[Dict]) -> Optional[float]:
    """Derive carbon intensity (kg CO2/kWh) as median over all tracked measurements.

    Median is used to resist outliers from transient grid-mix changes during long jobs.
    """
    intensities = []
    for row in all_rows:
        try:
            e = float(row.get("emissions") or 0)
            en = float(row.get("energy_consumed") or 0)
            if e > 0 and en > 0:
                intensities.append(e / en)
        except (ValueError, TypeError):
            pass
    if not intensities:
        return None
    ci = float(np.median(intensities))
    logger.info(
        "Derived carbon intensity: %.4f kg CO2/kWh (median over %d rows, "
        "location: Switzerland, Vaud — EPFL RCP cluster)",
        ci,
        len(intensities),
    )
    return ci


def estimate_training_co2(
    scratch_dir: Path,
    carbon_intensity: float,
    gpu_power_w: float,
) -> Dict:
    """Return Stage 2 training CO2 from emissions CSVs (preferred) or duration × power fallback."""
    stage_id, description, history_globs = TRAINING_STAGE_DEF

    # ── Path 1: emissions CSVs exist (preferred) ──────────────────────────
    all_rows: List[Dict] = []
    found_files: List[str] = []
    for glob in TRAINING_EMISSIONS_GLOBS:
        p = scratch_dir / glob
        if p.exists():
            rows = read_emissions_csv(p)
            if rows:
                all_rows.extend(rows)
                found_files.append(str(p.relative_to(scratch_dir)))

    if all_rows:
        agg = aggregate_csv_rows(all_rows)
        logger.info(
            "Training: found %d emissions CSV(s), %.0f s, %.6f kg CO2",
            len(found_files), agg["duration_s"], agg["co2_kg"],
        )
        return {
            "stage": stage_id,
            "description": description,
            "tracked": True,
            "estimated": False,
            "found": True,
            "files": found_files,
            "num_csv_rows": agg["num_rows"],
            "duration_s": agg["duration_s"],
            "energy_kwh": agg["energy_kwh"],
            "co2_kg": agg["co2_kg"],
            "mean_gpu_power_w": agg["mean_gpu_power_w"],
        }

    # ── Path 2: fall back to duration × power estimate ────────────────────
    conditions = {}
    total_duration = 0.0

    for glob in history_globs:
        history_path = scratch_dir / glob
        if not history_path.exists():
            cond = history_path.parent.name.replace("hmr_stage2_", "").replace("_checkpoint", "")
            logger.warning("training_history.json not found: %s", history_path)
            conditions[cond] = None
            continue
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
            cond = data.get("condition") or history_path.parent.name
            duration_s = data.get("results", {}).get("training_duration_seconds")
            if duration_s is None:
                logger.warning("No training_duration_seconds in %s", history_path)
                conditions[cond] = None
                continue
            total_duration += float(duration_s)
            conditions[cond] = float(duration_s)
            logger.info("  Training %s: %.0f s = %.1f min", cond, duration_s, duration_s / 60)
        except Exception as exc:
            logger.warning("Could not read %s: %s", history_path, exc)

    if total_duration <= 0:
        return {
            "stage": stage_id,
            "description": description,
            "tracked": False,
            "estimated": False,
            "error": "No training duration data found",
        }

    energy_kwh = total_duration * gpu_power_w / 3_600_000
    co2_kg = energy_kwh * carbon_intensity

    return {
        "stage": stage_id,
        "description": description,
        "tracked": False,
        "estimated": True,
        "estimation_method": (
            f"duration × assumed_gpu_power ({gpu_power_w:.0f} W, "
            f"A100-SXM4-40GB at ~68% TDP) × carbon_intensity "
            f"({carbon_intensity:.4f} kg CO2/kWh)"
        ),
        "conditions": conditions,
        "duration_s": total_duration,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "num_conditions": len([v for v in conditions.values() if v is not None]),
    }


def estimate_stage0_co2(
    scratch_dir: Path,
    carbon_intensity: float,
    sec_per_image: float = STAGE0_SEC_PER_IMAGE,
    gpu_power_w: float = GPU_POWER_STAGE0_W,
) -> Dict:
    """
    Estimate Stage 0 (OCR + CLIP filtering) CO2.

    Priority:
      1. Read CodeCarbon CSVs if they exist (emissions_stage0_*.csv produced
         by filter_meme_images.py on future runs).
      2. Fall back to manifest row-count estimation:
         energy = n_images × sec_per_image × gpu_power_w / 3_600_000 kWh
         co2    = energy × carbon_intensity
    """
    stage_id = "stage0_ocr_clip_filter"
    description = "OCR + CLIP filtering (3 datasets, ESTIMATED)"

    # ── Path 1: direct CodeCarbon CSVs ────────────────────────────────────
    all_rows: List[Dict] = []
    found_files: List[str] = []
    for glob in STAGE0_EMISSIONS_GLOBS:
        p = scratch_dir / glob
        if p.exists():
            rows = read_emissions_csv(p)
            if rows:
                all_rows.extend(rows)
                found_files.append(str(p.relative_to(scratch_dir)))

    if all_rows:
        agg = aggregate_csv_rows(all_rows)
        logger.info(
            "Stage 0: found %d CodeCarbon CSV(s), %.0f s, %.6f kg CO2",
            len(found_files), agg["duration_s"], agg["co2_kg"],
        )
        return {
            "stage": stage_id,
            "description": description,
            "tracked": True,
            "estimated": False,
            "found": True,
            "files": found_files,
            "num_csv_rows": agg["num_rows"],
            "duration_s": agg["duration_s"],
            "energy_kwh": agg["energy_kwh"],
            "co2_kg": agg["co2_kg"],
            "mean_gpu_power_w": agg["mean_gpu_power_w"],
        }

    # ── Path 2: manifest-based estimation ─────────────────────────────────
    total_images = 0
    dataset_counts: Dict[str, int] = {}
    for glob in STAGE0_MANIFEST_GLOBS:
        p = scratch_dir / glob
        if not p.exists():
            name = Path(glob).parent.name
            logger.warning("Stage 0 manifest not found: %s", p)
            dataset_counts[name] = 0
            continue
        try:
            with open(p, newline="", encoding="utf-8") as f:
                n = sum(1 for row in csv.reader(f)) - 1  # subtract header
            name = Path(glob).parent.name
            dataset_counts[name] = max(n, 0)
            total_images += max(n, 0)
            logger.info("  Stage 0 manifest %s: %d images", name, max(n, 0))
        except Exception as exc:
            name = Path(glob).parent.name
            logger.warning("Could not count %s: %s", p, exc)
            dataset_counts[name] = 0

    if total_images == 0:
        return {
            "stage": stage_id,
            "description": description,
            "tracked": False,
            "estimated": False,
            "found": False,
            "error": "No manifest files or emissions CSVs found for Stage 0",
            "duration_s": 0.0,
            "energy_kwh": 0.0,
            "co2_kg": 0.0,
        }

    duration_s = total_images * sec_per_image
    energy_kwh = duration_s * gpu_power_w / 3_600_000
    co2_kg = energy_kwh * carbon_intensity

    logger.info(
        "Stage 0 ESTIMATED: %d images × %.2f s/img × %.0f W → %.1f s, "
        "%.6f kWh, %.6f kg CO2",
        total_images, sec_per_image, gpu_power_w, duration_s, energy_kwh, co2_kg,
    )

    return {
        "stage": stage_id,
        "description": description,
        "tracked": False,
        "estimated": True,
        "found": True,
        "estimation_method": (
            f"{total_images} images × {sec_per_image} s/image × "
            f"{gpu_power_w:.0f} W (EasyOCR+CLIP on A100) × "
            f"{carbon_intensity:.4f} kg CO2/kWh"
        ),
        "dataset_image_counts": dataset_counts,
        "total_images": total_images,
        "duration_s": duration_s,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
    }


def build_stage_summary(
    stage_id: str,
    description: str,
    globs: List[str],
    scratch_dir: Path,
) -> Dict:
    all_rows = []
    found_files = []

    for glob in globs:
        # Support both exact paths and glob patterns
        if "*" in glob:
            matches = sorted((scratch_dir).glob(glob))
        else:
            p = scratch_dir / glob
            matches = [p] if p.exists() else []

        for p in matches:
            rows = read_emissions_csv(p)
            if rows:
                all_rows.extend(rows)
                found_files.append(str(p.relative_to(scratch_dir)))

    if not all_rows:
        return {
            "stage": stage_id,
            "description": description,
            "tracked": True,
            "found": False,
            "files": [],
            "duration_s": 0.0,
            "energy_kwh": 0.0,
            "co2_kg": 0.0,
        }

    agg = aggregate_csv_rows(all_rows)
    return {
        "stage": stage_id,
        "description": description,
        "tracked": True,
        "found": True,
        "files": found_files,
        "num_csv_rows": agg["num_rows"],
        "duration_s": agg["duration_s"],
        "energy_kwh": agg["energy_kwh"],
        "co2_kg": agg["co2_kg"],
        "mean_gpu_power_w": agg["mean_gpu_power_w"],
    }


def fmt_co2(kg: float) -> str:
    if kg >= 1.0:
        return f"{kg:.3f} kg"
    if kg >= 1e-3:
        return f"{kg*1e3:.2f} g"
    return f"{kg*1e6:.2f} mg"


def fmt_time(s: float) -> str:
    if s >= 3600:
        return f"{s/3600:.1f} h"
    if s >= 60:
        return f"{s/60:.1f} min"
    return f"{s:.0f} s"


def print_table(stages: List[Dict]) -> None:
    col_w = [38, 10, 10, 12, 6]
    header = ["Stage", "Duration", "Energy", "CO2", "Est."]
    sep = "  ".join("-" * w for w in col_w)
    row_fmt = "  ".join(f"{{:<{w}}}" for w in col_w)

    print("\n" + row_fmt.format(*header))
    print(sep)

    total_duration = 0.0
    total_energy = 0.0
    total_co2 = 0.0

    for s in stages:
        d = s.get("duration_s", 0.0) or 0.0
        e = s.get("energy_kwh", 0.0) or 0.0
        c = s.get("co2_kg", 0.0) or 0.0
        found = s.get("found", True) and (d > 0 or c > 0)
        est = "yes" if s.get("estimated") else ("no" if found else "—")
        print(row_fmt.format(
            s["description"][:38],
            fmt_time(d) if d > 0 else "—",
            f"{e:.4f} kWh" if e > 0 else "—",
            fmt_co2(c) if c > 0 else "—",
            est,
        ))
        total_duration += d
        total_energy += e
        total_co2 += c

    print(sep)
    print(row_fmt.format(
        "TOTAL",
        fmt_time(total_duration),
        f"{total_energy:.4f} kWh",
        fmt_co2(total_co2),
        "",
    ))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate pipeline CO2 emissions")
    parser.add_argument("--scratch_dir", type=Path, default=Path("/scratch"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--gpu_power_training_w",
        type=float,
        default=GPU_POWER_TRAINING_W,
        help="Assumed GPU power (W) during BART training (default: %(default)s)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.scratch_dir / "hmr_co2_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning emissions CSVs under %s", args.scratch_dir)

    # Pool all tracked rows to derive a robust carbon-intensity estimate
    # before processing individual stages.
    all_rows_global: List[Dict] = []
    for _, _, globs, _ in STAGE_DEFS:
        for glob in globs:
            if "*" in glob:
                for p in sorted(args.scratch_dir.glob(glob)):
                    all_rows_global.extend(read_emissions_csv(p))
            else:
                p = args.scratch_dir / glob
                if p.exists():
                    all_rows_global.extend(read_emissions_csv(p))

    carbon_intensity = derive_carbon_intensity(all_rows_global)
    if carbon_intensity is None:
        logger.error("Could not derive carbon intensity — no valid emissions CSVs found.")
        return 1

    # ── Per-stage summaries ───────────────────────────────────────────────
    stage_summaries = []
    for stage_id, description, globs, _ in STAGE_DEFS:
        summary = build_stage_summary(stage_id, description, globs, args.scratch_dir)
        stage_summaries.append(summary)
        if summary.get("found"):
            logger.info(
                "%-40s  %s  |  %.6f kg CO2",
                description,
                fmt_time(summary["duration_s"]),
                summary["co2_kg"],
            )
        else:
            logger.info("%-40s  (no data found)", description)

    stage0_summary = estimate_stage0_co2(args.scratch_dir, carbon_intensity)
    stage_summaries.insert(0, stage0_summary)

    training_summary = estimate_training_co2(
        args.scratch_dir, carbon_intensity, args.gpu_power_training_w
    )
    # Position 3 preserves chronological order: stage0, stage1a, stage1b, stage2, …
    stage_summaries.insert(3, training_summary)

    # Exclude optional stages that produced no data from the headline total.
    primary_stages = [
        s for s in stage_summaries
        if not (s["stage"] in ("proxy_inference", "detoxllm_inference") and not s.get("found"))
    ]
    total_co2 = sum(s.get("co2_kg", 0.0) or 0.0 for s in primary_stages)
    total_energy = sum(s.get("energy_kwh", 0.0) or 0.0 for s in primary_stages)
    total_duration = sum(s.get("duration_s", 0.0) or 0.0 for s in primary_stages)

    result = {
        "note": (
            "Stage 0 (OCR/CLIP filtering) CO2 is ESTIMATED from image counts × assumed rate "
            "(EasyOCR+CLIP, ~0.33 s/image, ~150 W on A100). "
            "Stage 2 training CO2 is estimated from job duration × assumed GPU power. "
            "All other stages use direct CodeCarbon measurements."
        ),
        "carbon_intensity_kg_per_kwh": carbon_intensity,
        "carbon_intensity_source": "Median across all tracked runs (Switzerland, Vaud — EPFL RCP cluster)",
        "gpu_power_training_w_assumed": args.gpu_power_training_w,
        "stages": stage_summaries,
        "totals": {
            "duration_s": total_duration,
            "energy_kwh": total_energy,
            "co2_kg": total_co2,
        },
    }

    out_json = output_dir / "pipeline_co2_summary.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Full JSON written to %s", out_json)

    # ── TSV for spreadsheet ingestion ────────────────────────────────────
    tsv_lines = ["stage\tdescription\tduration_s\tenergy_kwh\tco2_kg\testimated\tnote"]
    for s in stage_summaries:
        tsv_lines.append("\t".join([
            s.get("stage", ""),
            s.get("description", ""),
            f"{s.get('duration_s', 0) or 0:.1f}",
            f"{s.get('energy_kwh', 0) or 0:.6f}",
            f"{s.get('co2_kg', 0) or 0:.8f}",
            "yes" if s.get("estimated") else "no",
            s.get("estimation_method") or s.get("error") or "",
        ]))
    tsv_lines.append("\t".join([
        "TOTAL", "",
        f"{total_duration:.1f}",
        f"{total_energy:.6f}",
        f"{total_co2:.8f}",
        "", "",
    ]))
    out_tsv = output_dir / "pipeline_co2_summary.tsv"
    out_tsv.write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    logger.info("TSV table written to %s", out_tsv)

    print("\n=== Pipeline CO2 Summary ===")
    print(f"Carbon intensity (EPFL RCP, Vaud, Switzerland): {carbon_intensity*1000:.2f} g CO2/kWh")
    print_table(stage_summaries)
    print(f"  Total estimated CO2 : {fmt_co2(total_co2)}")
    print(f"  Total energy        : {total_energy:.4f} kWh")
    print(f"  Total wall-clock    : {fmt_time(total_duration)}")
    print()
    print("Note: Stage 0 filtering is ESTIMATED (image count × 0.33 s/img × 150 W on A100).")
    print("      Stage 2 training is estimated from duration × assumed GPU power.")
    print("      All other stages use direct CodeCarbon measurements.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
