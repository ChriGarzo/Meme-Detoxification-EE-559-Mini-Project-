"""
Multi-system evaluation harness for hateful meme detoxification.

Compares the LLaVA teacher, non-finetuned BART, LoRA-finetuned BART, the
CLIP-proxy reranker pipeline, and the DetoxLLM text-only baseline.

Systems
-------
  llava_teacher       — Stage 1 pseudo-rewrite JSONL (target_text field).
  bart_base           — inference/run_stage2.py with facebook/bart-large.
  bart_finetuned      — inference/run_stage2.py with a LoRA-merged checkpoint.
  clip_proxy_bart_*   — inference/run_proxy_pipeline.py outputs.
  detoxllm            — baselines/run_detoxllm_baseline.py outputs.

Metrics computed per system
----------------------------
  text_sta       — mean P(non-toxic) from s-nlp/roberta_toxicity_classifier.
  text_sta_delta — text_sta(rewrite) − text_sta(original).
  sim            — BERTScore F1 (roberta-large, rescaled) between original
                   and rewrite text.
  clip           — normalised cosine similarity between image and rewrite
                   text via openai/clip-vit-large-patch14.

Outputs written to --output_dir:
    evaluation_results.json  — full per-system metric dicts.
    evaluation_summary.json  — compact summary dicts.
    evaluation_summary.tsv   — tab-separated summary table.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import (
    compute_clipscore,
    compute_sim,
    compute_sta,
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, debug: bool = False) -> None:
    """Configure root logger to write to both a file and stderr."""
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "evaluate.log"),
            logging.StreamHandler(),
        ],
    )


def _first_existing_jsonl(output_dir: Path) -> Optional[Path]:
    """Return the first JSONL file found under output_dir, or output_dir itself if it is a JSONL file."""
    if output_dir.is_file() and output_dir.suffix == ".jsonl":
        return output_dir
    if not output_dir.exists():
        return None

    preferred_patterns = [
        "stage2_rewrites_*.jsonl",
        "*_rewrites.jsonl",
        "*.jsonl",
    ]
    for pattern in preferred_patterns:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _condition_from_path(path: Path) -> str:
    """Infer the conditioning variant (full/target_only/visual_only/none) from a JSONL path."""
    name = path.stem
    if name.startswith("stage2_rewrites_"):
        return name.replace("stage2_rewrites_", "")
    parent = path.parent.name
    for token in ("full", "target_only", "visual_only", "none"):
        if token in parent or token in name:
            return token
    return "unknown"


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_record(raw: Dict[str, Any], fallback_system: str) -> Optional[Dict[str, Any]]:
    original = raw.get("original_text") or raw.get("text") or ""
    rewrite = raw.get("rewrite") or raw.get("pseudo_rewrite") or raw.get("generated_text") or ""
    image_path = raw.get("image_path") or ""
    if not original or not rewrite:
        return None

    explanation = raw.get("explanation") or {}
    if not isinstance(explanation, dict):
        explanation = {}

    return {
        "id": raw.get("id") or raw.get("idx"),
        "system": raw.get("system") or fallback_system,
        "image_path": image_path,
        "original_text": str(original),
        "rewrite": str(rewrite),
        "target_group": raw.get("target_group") or explanation.get("target_group"),
        "visual_evidence": raw.get("visual_evidence") or explanation.get("visual_evidence"),
        "implicit_meaning": raw.get("implicit_meaning") or explanation.get("implicit_meaning"),
        "original_toxicity": _as_float(raw.get("original_toxicity")),
        "rewrite_toxicity": _as_float(raw.get("rewrite_toxicity")),
        "toxicity_drop": _as_float(raw.get("toxicity_drop")),
    }


def load_validation_teacher_records(path: Path, max_examples: Optional[int]) -> List[Dict[str, Any]]:
    """
    Load Stage 2 validation JSONL as the LLaVA teacher / pseudo-rewrite reference system.

    Uses target_text as the rewrite field; falls back to pseudo_rewrite if absent.
    """
    records: List[Dict[str, Any]] = []
    if not path.exists():
        logger.warning("Missing validation JSONL: %s", path)
        return records

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON in %s line %d: %s", path, line_num, exc)
                continue
            original = raw.get("original_text") or raw.get("text") or ""
            rewrite = raw.get("target_text") or raw.get("pseudo_rewrite") or ""
            if not original or not rewrite:
                continue
            records.append({
                "id": raw.get("id"),
                "system": "llava_teacher",
                "image_path": raw.get("image_path") or "",
                "original_text": str(original),
                "rewrite": str(rewrite),
                "target_group": raw.get("target_group"),
                "visual_evidence": raw.get("visual_evidence"),
                "implicit_meaning": raw.get("implicit_meaning"),
                "original_toxicity": _as_float(raw.get("original_toxicity")),
                "rewrite_toxicity": _as_float(raw.get("rewrite_toxicity")),
                "toxicity_drop": _as_float(raw.get("stage1_toxicity_drop") or raw.get("toxicity_drop")),
            })
            if max_examples and len(records) >= max_examples:
                break

    logger.info("Loaded %d validation teacher records from %s", len(records), path)
    return records


def load_system_records(
    path: Path,
    system_name: str,
    max_examples: Optional[int],
    id_filter: Optional[set] = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        logger.warning("Missing JSONL for %s: %s", system_name, path)
        return records

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON in %s line %d: %s", path, line_num, exc)
                continue
            if not isinstance(raw, dict):
                continue
            if id_filter is not None and str(raw.get("id") or raw.get("idx")) not in id_filter:
                continue
            rec = _extract_record(raw, system_name)
            if rec is not None:
                rec["system"] = rec.get("system") or system_name
                records.append(rec)
            if max_examples and len(records) >= max_examples:
                break

    logger.info("Loaded %d records for %s from %s", len(records), system_name, path)
    return records


def discover_stage2_systems(
    dirs: List[Path],
    prefix: str,
    max_examples: Optional[int],
    id_filter: Optional[set] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Scan output directories for Stage 2 JSONL files; label each by inferred condition."""
    systems: Dict[str, List[Dict[str, Any]]] = {}
    for output_dir in dirs:
        jsonl_path = _first_existing_jsonl(output_dir)
        if jsonl_path is None:
            logger.warning("No JSONL output found in %s", output_dir)
            continue
        condition = _condition_from_path(jsonl_path)
        system_name = f"{prefix}_{condition}"
        systems[system_name] = load_system_records(
            jsonl_path,
            system_name,
            max_examples,
            id_filter=id_filter,
        )
    return systems


def discover_named_systems(
    dirs: List[Path],
    default_name: str,
    max_examples: Optional[int],
    id_filter: Optional[set] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Scan output directories for named-system JSONL files (e.g. proxy pipeline outputs)."""
    systems: Dict[str, List[Dict[str, Any]]] = {}
    for output_dir in dirs:
        jsonl_path = _first_existing_jsonl(output_dir)
        if jsonl_path is None:
            logger.warning("No JSONL output found in %s", output_dir)
            continue
        system_name = default_name
        if "clip_proxy_bart_full" in jsonl_path.stem or "proxy" in output_dir.name:
            system_name = "clip_proxy_bart_full"
        records = load_system_records(
            jsonl_path,
            system_name,
            max_examples,
            id_filter=id_filter,
        )
        if records and records[0].get("system"):
            system_name = records[0]["system"]
        systems[system_name] = records
    return systems


def valid_image_pairs(records: List[Dict[str, Any]], text_key: str) -> Tuple[List[str], List[str]]:
    """Return (image_paths, texts) for records whose image_path resolves to an existing file."""
    images: List[str] = []
    texts: List[str] = []
    for rec in records:
        image_path = rec.get("image_path") or ""
        if image_path and Path(image_path).exists():
            images.append(image_path)
            texts.append(rec.get(text_key, ""))
    return images, texts


def evaluate_system(
    system_name: str,
    records: List[Dict[str, Any]],
    hf_cache: Optional[str],
    compute_clip: bool,
) -> Dict[str, Any]:
    """
    Run all enabled metrics on a single system's records; return a result dict.

    Text STA and BERTScore SIM are always computed. CLIPScore is skipped when
    --skip_clipscore is set or no valid image-text pairs are available.
    """
    logger.info("=" * 80)
    logger.info("Evaluating %s (%d examples)", system_name, len(records))
    logger.info("=" * 80)

    originals = [r["original_text"] for r in records]
    rewrites = [r["rewrite"] for r in records]

    result: Dict[str, Any] = {
        "system": system_name,
        "num_examples": len(records),
    }

    if not records:
        result["error"] = "no_records"
        return result

    # Text-only STA: score both original and rewrite so we can report the delta.
    logger.info("Computing text STA for originals and rewrites...")
    original_sta = compute_sta(originals)
    rewrite_sta = compute_sta(rewrites)
    result["text_sta_original"] = original_sta
    result["text_sta_rewrite"] = rewrite_sta
    result["text_sta_delta"] = float(rewrite_sta["mean"] - original_sta["mean"])

    logger.info("Computing BERTScore SIM...")
    result["sim"] = compute_sim(originals, rewrites)

    images_for_rewrite, rewrite_texts_for_images = valid_image_pairs(records, "rewrite")
    images_for_original, original_texts_for_images = valid_image_pairs(records, "original_text")
    result["num_valid_images"] = len(images_for_rewrite)

    if compute_clip and images_for_rewrite:
        logger.info("Computing CLIPScore...")
        result["clip"] = compute_clipscore(images_for_rewrite, rewrite_texts_for_images)
    else:
        result["clip"] = None

    # Stage 1 toxicity drop stored in the JSONL (pre-computed by the pseudo-rewrite
    # pipeline); reported as a sanity check.
    stage1_drops = [r["toxicity_drop"] for r in records if r.get("toxicity_drop") is not None]
    if stage1_drops:
        result["stage1_toxicity_drop_mean"] = float(np.mean(stage1_drops))

    return result


def compact_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the key per-system scalars from a full evaluate_system result dict."""
    text_sta = result.get("text_sta_rewrite") or {}
    sim = result.get("sim") or {}
    clip = result.get("clip") or {}
    return {
        "system": result.get("system"),
        "n": result.get("num_examples"),
        "valid_images": result.get("num_valid_images"),
        "text_sta": text_sta.get("mean"),
        "text_sta_delta": result.get("text_sta_delta"),
        "sim": sim.get("mean"),
        "clip": clip.get("mean") if clip else None,
    }


def write_summary_table(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """Write a tab-separated summary table of compact_summary dicts."""
    headers = [
        "system",
        "n",
        "valid_images",
        "text_sta",
        "text_sta_delta",
        "sim",
        "clip",
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = ["\t".join(headers)]
    for row in summaries:
        lines.append("\t".join(fmt(row.get(h)) for h in headers))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LLaVA, base BART, and finetuned BART rewrites")
    parser.add_argument(
        "--validation_jsonl",
        type=Path,
        default=None,
        help="Stage 2 held-out validation JSONL. Uses target_text as the LLaVA teacher/reference.",
    )
    parser.add_argument(
        "--llava_rewrites_path",
        type=Path,
        default=None,
        help="Fallback Stage 1 pseudo-rewrite JSONL if --validation_jsonl is not provided.",
    )
    parser.add_argument("--bart_base_output_dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--bart_finetuned_output_dirs", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--proxy_output_dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Optional output dirs from inference/run_proxy_pipeline.py.",
    )
    parser.add_argument(
        "--detoxllm_output_path",
        type=Path,
        default=None,
        help="Optional JSONL output from baselines/run_detoxllm_baseline.py.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hf_cache", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--skip_clipscore", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible metrics")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    setup_logging(args.output_dir, debug=args.debug)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    max_examples = args.max_examples
    if args.debug and max_examples is None:
        max_examples = 32

    systems: Dict[str, List[Dict[str, Any]]] = {}
    validation_ids = None
    if args.validation_jsonl is not None:
        systems["llava_teacher"] = load_validation_teacher_records(args.validation_jsonl, max_examples)
        # Build a validation ID set so all other systems are evaluated on the
        # same examples as the teacher; omit IDs from any system not in this set.
        validation_ids = {
            str(r["id"]) for r in systems["llava_teacher"]
            if r.get("id") is not None
        }
    elif args.llava_rewrites_path is not None:
        systems["llava_teacher"] = load_system_records(
            args.llava_rewrites_path,
            "llava_teacher",
            max_examples,
        )
    else:
        logger.error("Provide either --validation_jsonl or --llava_rewrites_path.")
        return 1

    systems.update(discover_stage2_systems(
        args.bart_base_output_dirs,
        "bart_base",
        max_examples,
        id_filter=validation_ids,
    ))
    systems.update(discover_stage2_systems(
        args.bart_finetuned_output_dirs,
        "bart_finetuned",
        max_examples,
        id_filter=validation_ids,
    ))
    systems.update(discover_named_systems(
        args.proxy_output_dirs,
        "clip_proxy_bart_full",
        max_examples,
        id_filter=validation_ids,
    ))

    if args.detoxllm_output_path is not None:
        detoxllm_jsonl = _first_existing_jsonl(args.detoxllm_output_path)
        if detoxllm_jsonl is not None:
            systems["detoxllm"] = load_system_records(
                detoxllm_jsonl,
                "detoxllm",
                max_examples,
                id_filter=validation_ids,
            )
        else:
            logger.warning("No JSONL found at --detoxllm_output_path: %s", args.detoxllm_output_path)

    if not systems:
        logger.error("No systems were loaded.")
        return 1

    results: List[Dict[str, Any]] = []
    for system_name, records in systems.items():
        result = evaluate_system(
            system_name=system_name,
            records=records,
            hf_cache=args.hf_cache,
            compute_clip=not args.skip_clipscore,
        )
        results.append(result)

    summaries = [compact_summary(r) for r in results]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_path = args.output_dir / "evaluation_results.json"
    summary_path = args.output_dir / "evaluation_summary.json"
    table_path = args.output_dir / "evaluation_summary.tsv"

    full_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_summary_table(summaries, table_path)

    logger.info("Evaluation complete.")
    logger.info("Full results: %s", full_path)
    logger.info("Summary JSON: %s", summary_path)
    logger.info("Summary TSV: %s", table_path)

    print("\nEvaluation summary")
    print(table_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
