"""
Compare fine-tuned vs base BART Stage 2 outputs on aligned examples.

Examples are aligned by `id`; metrics are computed in parallel on the two
output sets and deltas (finetuned − base) are reported.

Default metrics: STA, SIM
Optional (slower): CLIPScore (--with_clip), Rewrite Precision (--with_rp,
  requires Stage 1 explanations JSONL)

Output: a JSON file with per-set metrics, per-metric deltas, and average word lengths.

Example:
  python analysis/compare_stage2_outputs.py \\
    --finetuned_jsonl /scratch/hmr_eval_stage2_visual_only/stage2_rewrites_visual_only.jsonl \\
    --base_jsonl /scratch/hmr_eval_stage2_visual_only_base/stage2_rewrites_visual_only.jsonl \\
    --output_json /scratch/hmr_eval_compare/visual_only_finetuned_vs_base.json
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from evaluation.metrics import (
    compute_clipscore,
    compute_rewrite_precision,
    compute_sim,
    compute_sta,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_by_id(rows: List[Dict], label: str) -> Dict[str, Dict]:
    """Index rows by string id; warn on missing or duplicate ids.

    Duplicate ids keep the last occurrence, matching the usual JSONL append convention.
    """
    idx: Dict[str, Dict] = {}
    missing = 0
    duplicate = 0
    for row in rows:
        rid = row.get("id")
        if rid is None:
            missing += 1
            continue
        rid_s = str(rid)
        if rid_s in idx:
            duplicate += 1
        idx[rid_s] = row
    if missing > 0:
        logger.warning("%s: %d rows missing `id` were ignored", label, missing)
    if duplicate > 0:
        logger.warning("%s: %d duplicate ids found; last occurrence kept", label, duplicate)
    return idx


def align_rows(
    finetuned_rows: List[Dict],
    base_rows: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Return parallel lists of (finetuned, base) rows sharing the same ids.

    finetuned ordering is preserved; base rows are reordered to match.
    """
    ft_idx = index_by_id(finetuned_rows, "finetuned")
    base_idx = index_by_id(base_rows, "base")

    common_ids = [str(r.get("id")) for r in finetuned_rows if r.get("id") is not None and str(r.get("id")) in base_idx]
    # Deduplicate while preserving finetuned order.
    seen = set()
    ordered_ids: List[str] = []
    for cid in common_ids:
        if cid in seen:
            continue
        seen.add(cid)
        ordered_ids.append(cid)

    aligned_ft = [ft_idx[cid] for cid in ordered_ids]
    aligned_base = [base_idx[cid] for cid in ordered_ids]

    logger.info("Aligned examples: %d (finetuned total=%d, base total=%d)", len(aligned_ft), len(finetuned_rows), len(base_rows))
    return aligned_ft, aligned_base


def safe_text(x: Optional[str]) -> str:
    return x if isinstance(x, str) else ""


def load_explanations_by_id(path: Path) -> Dict[str, Dict]:
    rows = load_jsonl(path)
    out: Dict[str, Dict] = {}
    for row in rows:
        rid = row.get("id")
        if rid is None:
            continue
        out[str(rid)] = row.get("explanation", {}) or {}
    logger.info("Loaded %d explanations from %s", len(out), path)
    return out


def summarize(metric: Optional[Dict]) -> Optional[float]:
    """Extract the scalar mean from a metric dict returned by evaluation functions."""
    if metric is None:
        return None
    return float(metric.get("mean")) if "mean" in metric else None


def evaluate_set(
    name: str,
    originals: List[str],
    rewrites: List[str],
    image_paths: List[str],
    explanations: Optional[List[Dict]],
    with_clip: bool,
    with_rp: bool,
    hf_cache: Optional[str],
) -> Dict:
    logger.info("Evaluating set: %s", name)
    out: Dict = {"name": name}

    out["sta"] = compute_sta(rewrites)
    out["sim"] = compute_sim(originals, rewrites)

    if with_clip:
        out["clip"] = compute_clipscore(image_paths, rewrites)
    else:
        out["clip"] = None

    if with_rp:
        if explanations is None:
            raise ValueError("--with_rp requires explanations to be loaded")
        from models.explainer import MemeExplainer
        explainer = MemeExplainer(hf_cache=hf_cache)
        out["rewrite_precision"] = compute_rewrite_precision(image_paths, rewrites, explanations, explainer)
    else:
        out["rewrite_precision"] = None

    return out


def mean_len(texts: List[str]) -> float:
    if not texts:
        return 0.0
    return float(np.mean([len(t.split()) for t in texts]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fine-tuned vs base Stage 2 outputs")
    parser.add_argument("--finetuned_jsonl", type=Path, required=True)
    parser.add_argument("--base_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--hf_cache", type=str, default=None)
    parser.add_argument("--with_clip", action="store_true", help="Also compute CLIPScore")
    parser.add_argument("--with_rp", action="store_true", help="Also compute Rewrite Precision")
    parser.add_argument(
        "--stage1_explanations_jsonl",
        type=Path,
        default=None,
        help="Required if --with_rp is enabled; expected merged Stage 1 explanations JSONL",
    )
    args = parser.parse_args()

    setup_logging()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    ft_rows = load_jsonl(args.finetuned_jsonl)
    base_rows = load_jsonl(args.base_jsonl)
    aligned_ft, aligned_base = align_rows(ft_rows, base_rows)

    if not aligned_ft:
        raise RuntimeError("No overlapping ids between finetuned and base outputs.")

    originals = [safe_text(r.get("original_text")) for r in aligned_ft]
    ft_rewrites = [safe_text(r.get("rewrite")) for r in aligned_ft]
    base_rewrites = [safe_text(r.get("rewrite")) for r in aligned_base]
    image_paths = [safe_text(r.get("image_path")) for r in aligned_ft]

    if args.with_rp and args.stage1_explanations_jsonl is None:
        raise ValueError("--with_rp requires --stage1_explanations_jsonl")

    explanations: Optional[List[Dict]] = None
    if args.with_rp and args.stage1_explanations_jsonl:
        exp_by_id = load_explanations_by_id(args.stage1_explanations_jsonl)
        explanations = [exp_by_id.get(str(r.get("id")), {}) for r in aligned_ft]

    ft_metrics = evaluate_set(
        name="finetuned",
        originals=originals,
        rewrites=ft_rewrites,
        image_paths=image_paths,
        explanations=explanations,
        with_clip=args.with_clip,
        with_rp=args.with_rp,
        hf_cache=args.hf_cache,
    )
    base_metrics = evaluate_set(
        name="base",
        originals=originals,
        rewrites=base_rewrites,
        image_paths=image_paths,
        explanations=explanations,
        with_clip=args.with_clip,
        with_rp=args.with_rp,
        hf_cache=args.hf_cache,
    )

    deltas = {}
    for key in ["sta", "sim", "clip", "rewrite_precision"]:
        a = summarize(ft_metrics.get(key))
        b = summarize(base_metrics.get(key))
        deltas[key] = None if (a is None or b is None) else (a - b)

    result = {
        "num_aligned_examples": len(aligned_ft),
        "finetuned": ft_metrics,
        "base": base_metrics,
        "delta_finetuned_minus_base": deltas,
        "avg_word_length": {
            "finetuned": mean_len(ft_rewrites),
            "base": mean_len(base_rewrites),
            "delta_finetuned_minus_base": mean_len(ft_rewrites) - mean_len(base_rewrites),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    def fmt(x: Optional[float]) -> str:
        return "N/A" if x is None else f"{x:.4f}"

    print("\n=== Fine-tuned vs Base (same examples) ===")
    print(f"Aligned examples: {result['num_aligned_examples']}")
    print(f"STA  : ft={fmt(summarize(ft_metrics['sta']))} | base={fmt(summarize(base_metrics['sta']))} | delta={fmt(deltas['sta'])}")
    print(f"SIM  : ft={fmt(summarize(ft_metrics['sim']))} | base={fmt(summarize(base_metrics['sim']))} | delta={fmt(deltas['sim'])}")
    if args.with_clip:
        print(f"CLIP : ft={fmt(summarize(ft_metrics['clip']))} | base={fmt(summarize(base_metrics['clip']))} | delta={fmt(deltas['clip'])}")
    if args.with_rp:
        print(f"RP   : ft={fmt(summarize(ft_metrics['rewrite_precision']))} | base={fmt(summarize(base_metrics['rewrite_precision']))} | delta={fmt(deltas['rewrite_precision'])}")
    print(f"Avg words: ft={result['avg_word_length']['finetuned']:.2f} | base={result['avg_word_length']['base']:.2f} | delta={result['avg_word_length']['delta_finetuned_minus_base']:.2f}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
