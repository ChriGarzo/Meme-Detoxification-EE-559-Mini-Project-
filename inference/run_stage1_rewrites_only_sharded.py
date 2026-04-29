"""
Stage 1 (rewrites-only, sharded): generate pseudo-rewrites from existing explanations.

This script does NOT compute explanations again.
It reads explanations JSONL and appends only missing rewrite entries.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import tqdm
from codecarbon import EmissionsTracker
from transformers.utils import logging as hf_logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.explainer import MemeExplainer
from utils.bertscore_utils import compute_bertscore_batch, create_bertscore_scorer

# Reuse shared helpers/classes from the combined Stage 1 script.
from run_stage1_multimodal_sharded import (
    VisualBertMultimodalScorer,
    compute_multimodal_hatefulness,
    has_invalid_rewrite_format,
    load_existing_ids,
    sanitize_generated_rewrite,
    set_seed,
    write_jsonl_batch,
)

logger = logging.getLogger(__name__)
SHARD_PATH_RE = re.compile(r"_shard\d+of\d+\.jsonl$")


def load_jsonl_records(path: str) -> List[Dict]:
    records: List[Dict] = []
    if not os.path.exists(path):
        return records
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                logger.warning("Skipping invalid JSONL line %d in %s: %s", i, path, e)
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _normalize_image_key(path: str) -> str:
    if not isinstance(path, str):
        return ""
    return os.path.normpath(path).replace("\\", "/")


def _record_key(record: Dict) -> str:
    rid = str(record.get("id", "")).strip()
    if rid:
        return rid
    return _normalize_image_key(str(record.get("image_path", "")))


def dedupe_records(records: List[Dict]) -> List[Dict]:
    by_key: Dict[str, Dict] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        key = _record_key(rec)
        if not key:
            continue
        # Keep first for deterministic behavior.
        if key not in by_key:
            by_key[key] = rec
    return [by_key[k] for k in sorted(by_key.keys())]


def is_pre_sharded_explanations(path: str) -> bool:
    return bool(SHARD_PATH_RE.search(os.path.basename(path)))


def select_records_for_shard(records: List[Dict], num_shards: int, shard_id: int) -> List[Dict]:
    selected: List[Dict] = []
    for rec in records:
        key = _record_key(rec)
        if not key:
            continue
        bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % num_shards
        if bucket == shard_id:
            selected.append(rec)
    return selected


def quality_reject_reason(
    *,
    passes_sta: bool,
    passes_bertscore: bool,
    passes_toxicity_delta: bool,
    bertscore: float,
    bertscore_max: float,
) -> str:
    if not passes_sta:
        return "low_sta"
    if not passes_toxicity_delta:
        return "low_toxicity_drop"
    if not passes_bertscore:
        if bertscore_max < 1.0 and bertscore >= bertscore_max:
            return "high_bertscore"
        return "low_bertscore"
    return ""


def _safe_float(value: Any, default: float = -1.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def candidate_rank_key(candidate: Dict[str, Any]) -> Tuple[float, ...]:
    format_valid = int(candidate.get("format_valid", True))
    has_text = int(bool(candidate.get("rewrite", "").strip()))
    passes_all = int(candidate.get("passes_all", False))
    passes_safety = int(candidate.get("passes_sta", False) and candidate.get("passes_toxicity_delta", False))
    passes_similarity = int(candidate.get("passes_bertscore", False))
    length_penalty = -abs(len(candidate["rewrite"].split()) - 12)
    return (
        format_valid,
        has_text,
        passes_all,
        passes_safety,
        passes_similarity,
        _safe_float(candidate.get("bertscore"), default=-1.0),
        _safe_float(candidate.get("sta_score"), default=-1.0),
        _safe_float(candidate.get("toxicity_drop"), default=-1.0),
        length_penalty,
        -len(candidate["rewrite"]),
    )


def most_common_reason(reasons: List[str], fallback: str = "empty") -> str:
    counts: Dict[str, int] = {}
    for reason in reasons:
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return fallback
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 (rewrites-only + sharded): Generate pseudo-rewrites from existing explanations"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'train')")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSONL files")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument(
        "--multimodal_model_name",
        type=str,
        default="chiragmittal92/visualbert-hateful-memes-finetuned-model",
        help="HuggingFace model for multimodal (image+text) hatefulness scoring",
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="Load LLaVA in 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--score_batch_size", type=int, default=8, help="Batch size for multimodal scorer")
    parser.add_argument("--num_shards", type=int, default=8, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id in [0, num_shards-1]")
    parser.add_argument("--hateful_only", action="store_true", help="Only process explanation rows where is_hateful=1")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process max 16 examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sta_threshold", type=float, default=0.25, help="Minimum non-hateful score (1 - hatefulness) for keeping rewrites")
    parser.add_argument("--bertscore_min", type=float, default=0.15, help="Minimum BERTScore similarity for keeping rewrites")
    parser.add_argument("--bertscore_max", type=float, default=1.0, help="Maximum BERTScore similarity to avoid near-copy rewrites (set 1.0 to disable)")
    parser.add_argument("--min_lexical_change", type=float, default=0.0, help="Minimum token-level change ratio required between original and rewrite (set 0.0 to disable)")
    parser.add_argument("--max_char_similarity", type=float, default=1.0, help="Maximum normalized char-level similarity allowed between original and rewrite (set 1.0 to disable)")
    parser.add_argument("--min_toxicity_drop", type=float, default=0.05, help="Minimum required hatefulness decrease from original to rewrite (set 0.0 to disable)")
    parser.add_argument("--min_source_toxicity_for_drop", type=float, default=0.20, help="Only enforce min_toxicity_drop when original hatefulness is at least this value")
    parser.add_argument("--rewrite_max_attempts", type=int, default=2, help="Maximum rewrite attempts per example")
    parser.add_argument("--rewrite_candidates_per_attempt", type=int, default=4, help="Number of rewrite candidates to sample per example on each attempt")
    parser.add_argument("--rewrite_temperature", type=float, default=0.75, help="Sampling temperature for rewrite generation")
    parser.add_argument("--rewrite_top_p", type=float, default=0.92, help="Top-p sampling value for rewrite generation")
    parser.add_argument(
        "--explanations_path",
        type=str,
        default="",
        help=(
            "Optional explicit explanations JSONL path. If empty, prefers "
            "<output_dir>/<dataset>_explanations_merged.jsonl, then falls back "
            "to <dataset>_explanations.jsonl, then shard file."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.score_batch_size < 1:
        raise ValueError("--score_batch_size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards-1]")
    if not (0.0 <= args.sta_threshold <= 1.0):
        raise ValueError("--sta_threshold must be in [0,1]")
    if not (0.0 <= args.bertscore_min <= 1.0):
        raise ValueError("--bertscore_min must be in [0,1]")
    if not (0.0 <= args.bertscore_max <= 1.0):
        raise ValueError("--bertscore_max must be in [0,1]")
    if args.bertscore_max <= args.bertscore_min:
        raise ValueError("--bertscore_max must be greater than --bertscore_min")
    if not (0.0 <= args.min_lexical_change <= 1.0):
        raise ValueError("--min_lexical_change must be in [0,1]")
    if not (0.0 <= args.max_char_similarity <= 1.0):
        raise ValueError("--max_char_similarity must be in [0,1]")
    if not (0.0 <= args.min_toxicity_drop <= 1.0):
        raise ValueError("--min_toxicity_drop must be in [0,1]")
    if not (0.0 <= args.min_source_toxicity_for_drop <= 1.0):
        raise ValueError("--min_source_toxicity_for_drop must be in [0,1]")
    if args.rewrite_max_attempts < 1:
        raise ValueError("--rewrite_max_attempts must be >= 1")
    if args.rewrite_candidates_per_attempt < 1:
        raise ValueError("--rewrite_candidates_per_attempt must be >= 1")
    if args.rewrite_temperature <= 0.0:
        raise ValueError("--rewrite_temperature must be > 0")
    if not (0.0 < args.rewrite_top_p <= 1.0):
        raise ValueError("--rewrite_top_p must be in (0,1]")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    shard_tag = f"shard{args.shard_id:02d}of{args.num_shards:02d}"
    stage1_log_path = os.path.join(args.output_dir, f"stage1_rewrite_only_{shard_tag}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(stage1_log_path),
            logging.StreamHandler(),
        ],
    )

    for noisy_logger in ["httpx", "huggingface_hub", "urllib3", "matplotlib", "PIL"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    explanations_path = args.explanations_path.strip()
    if not explanations_path:
        candidates = [
            os.path.join(args.output_dir, f"{args.dataset}_explanations_merged.jsonl"),
            os.path.join(args.output_dir, f"{args.dataset}_explanations.jsonl"),
            os.path.join(args.output_dir, f"{args.dataset}_explanations_{shard_tag}.jsonl"),
        ]
        explanations_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    pseudo_rewrites_path = os.path.join(args.output_dir, f"{args.dataset}_pseudo_rewrites_{shard_tag}.jsonl")

    print(f"\n{'='*60}")
    print("  Stage 1: Rewrites Only")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Images:     {args.images_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  HF cache:   {args.hf_cache}")
    print(f"  4-bit quant:{args.load_in_4bit}")
    print(f"  Scorer:     {args.multimodal_model_name}")
    print(f"  Explanations:{explanations_path}")
    print(f"  Rewrites:   {pseudo_rewrites_path}")
    print(f"  Shard:      {args.shard_id + 1}/{args.num_shards} ({shard_tag})")
    print(f"  Rewrite cfg: attempts={args.rewrite_max_attempts}, candidates/attempt={args.rewrite_candidates_per_attempt}, temp={args.rewrite_temperature}, top_p={args.rewrite_top_p}")
    print(f"  Debug:      {args.debug}")
    print(f"{'='*60}\n")

    logger.info("Starting rewrites-only Stage 1")
    logger.info(f"Arguments: {vars(args)}")

    if not os.path.exists(explanations_path):
        raise FileNotFoundError(f"Explanations file not found: {explanations_path}")

    explanation_records = load_jsonl_records(explanations_path)
    logger.info(f"Loaded explanation records (raw): {len(explanation_records)}")
    explanation_records = dedupe_records(explanation_records)
    logger.info(f"Loaded explanation records (deduped): {len(explanation_records)}")

    if is_pre_sharded_explanations(explanations_path):
        logger.info("Detected pre-sharded explanations file. No additional sharding applied.")
    else:
        before = len(explanation_records)
        explanation_records = select_records_for_shard(
            explanation_records, num_shards=args.num_shards, shard_id=args.shard_id
        )
        logger.info(
            "Applied deterministic shard split to merged explanations: kept %d/%d for shard %d/%d",
            len(explanation_records),
            before,
            args.shard_id,
            args.num_shards,
        )

    if args.hateful_only:
        before = len(explanation_records)
        explanation_records = [r for r in explanation_records if bool(r.get("is_hateful", False))]
        logger.info(f"--hateful_only: kept {len(explanation_records)}/{before} explanation rows")

    if args.debug:
        explanation_records = explanation_records[:16]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        logger.info("No GPU found — running on CPU (will be slow)")

    explainer = MemeExplainer(
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.hf_cache,
        device=device,
        debug=args.debug,
    )

    hate_scorer = VisualBertMultimodalScorer(
        model_name=args.multimodal_model_name,
        device=device,
        cache_dir=args.hf_cache,
    )

    bertscore_scorer = create_bertscore_scorer(device=device)

    processed_rewrite_ids = load_existing_ids(pseudo_rewrites_path)
    logger.info(f"Already processed rewrites: {len(processed_rewrite_ids)}")

    rewrites_batch: List[Dict] = []
    total_examples = 0
    total_pseudo_rewrites = 0
    written_rewrites = 0
    passed_rewrites = 0
    invalid_rewrite_format = 0
    invalid_rewrite_reason_counts: Dict[str, int] = {}
    quality_reject_count = 0
    quality_reject_reason_counts: Dict[str, int] = {}
    rewrite_generation_failures = 0

    emissions_file = f"emissions_rewrite_only_{shard_tag}.csv"
    tracker = EmissionsTracker(log_level="warning", output_dir=args.output_dir, output_file=emissions_file)
    tracker.start()

    try:
        next_stats_log = 50
        next_rewrites_flush = 100

        with tqdm.tqdm(total=len(explanation_records), desc="Processing rewrites") as pbar:
            for start_idx in range(0, len(explanation_records), args.batch_size):
                raw_batch_rows = explanation_records[start_idx:start_idx + args.batch_size]
                pending_rows = []

                for row in raw_batch_rows:
                    example_id = row.get("id")
                    if example_id in processed_rewrite_ids:
                        pbar.update(1)
                        continue
                    pending_rows.append(row)

                if not pending_rows:
                    continue

                rw_ids: List[str] = []
                rw_image_paths: List[str] = []
                rw_original_texts: List[str] = []
                rw_explanations: List[Dict] = []

                for row in pending_rows:
                    example_id = row.get("id")
                    raw_img = str(row.get("image_path", ""))
                    if os.path.isabs(raw_img) and os.path.exists(raw_img):
                        image_path = raw_img
                    else:
                        image_path = os.path.join(args.images_dir, raw_img)

                    original_text = str(row.get("original_text", ""))
                    explanation = row.get("explanation", {})
                    if not isinstance(explanation, dict):
                        explanation = {}

                    rw_ids.append(example_id)
                    rw_image_paths.append(image_path)
                    rw_original_texts.append(original_text)
                    rw_explanations.append(explanation)

                total_examples += len(pending_rows)
                total_pseudo_rewrites += len(pending_rows)

                rw_original_toxicities = compute_multimodal_hatefulness(
                    rw_image_paths,
                    rw_original_texts,
                    hate_scorer,
                    batch_size=args.score_batch_size,
                )

                selected_candidates: Dict[int, Dict[str, Any]] = {}
                retry_feedback_reasons: List[Optional[str]] = [None] * len(pending_rows)
                active_indices = list(range(len(pending_rows)))

                for attempt_idx in range(args.rewrite_max_attempts):
                    if not active_indices:
                        break

                    active_image_paths = [rw_image_paths[i] for i in active_indices]
                    active_original_texts = [rw_original_texts[i] for i in active_indices]
                    active_explanations = [rw_explanations[i] for i in active_indices]
                    active_ids = [rw_ids[i] for i in active_indices]
                    active_feedback = [retry_feedback_reasons[i] for i in active_indices]

                    try:
                        raw_candidate_groups = explainer.batch_rewrite_candidates(
                            active_image_paths,
                            active_original_texts,
                            active_explanations,
                            feedback_reasons=active_feedback,
                            candidates_per_example=args.rewrite_candidates_per_attempt,
                            do_sample=True,
                            temperature=args.rewrite_temperature,
                            top_p=args.rewrite_top_p,
                        )
                    except Exception as e:
                        rewrite_generation_failures += len(active_indices)
                        logger.warning(
                            "Batch rewrite generation failed at attempt %d/%d for %d examples: %s",
                            attempt_idx + 1,
                            args.rewrite_max_attempts,
                            len(active_indices),
                            e,
                        )
                        break

                    candidate_records: List[Dict[str, Any]] = []
                    invalid_reasons_by_slot: Dict[int, List[str]] = {slot: [] for slot in active_indices}
                    generation_failed_slots: set[int] = set()

                    for pos, active_slot in enumerate(active_indices):
                        example_id = active_ids[pos]
                        raw_candidates = raw_candidate_groups[pos] if pos < len(raw_candidate_groups) else []
                        if not raw_candidates:
                            raw_candidates = [""]

                        for raw_rewrite in raw_candidates:
                            if isinstance(raw_rewrite, str) and raw_rewrite.startswith("[REWRITE ERROR:"):
                                rewrite_generation_failures += 1
                                generation_failed_slots.add(active_slot)
                                logger.warning(
                                    "Rewrite generation failed for %s (attempt %d/%d): %s",
                                    example_id,
                                    attempt_idx + 1,
                                    args.rewrite_max_attempts,
                                    raw_rewrite,
                                )
                                continue

                            candidate = sanitize_generated_rewrite(raw_rewrite)
                            is_invalid, reason = has_invalid_rewrite_format(
                                candidate,
                                rw_original_texts[active_slot],
                                min_lexical_change=args.min_lexical_change,
                                max_char_similarity=args.max_char_similarity,
                            )
                            candidate_record = {
                                "slot_idx": active_slot,
                                "id": example_id,
                                "rewrite": candidate,
                                "original_text": rw_original_texts[active_slot],
                                "image_path": rw_image_paths[active_slot],
                                "original_toxicity": rw_original_toxicities[active_slot],
                                "format_valid": not is_invalid,
                                "format_reason": reason if is_invalid else "",
                                "passes_sta": False,
                                "passes_bertscore": False,
                                "passes_toxicity_delta": False,
                                "passes_all": False,
                                "quality_reason": reason if is_invalid else "",
                                "sta_score": None,
                                "rewrite_toxicity": None,
                                "bertscore": None,
                                "toxicity_drop": None,
                            }

                            if is_invalid:
                                invalid_rewrite_format += 1
                                invalid_rewrite_reason_counts[reason] = invalid_rewrite_reason_counts.get(reason, 0) + 1
                                invalid_reasons_by_slot[active_slot].append(reason)
                            candidate_records.append(candidate_record)

                    scorable_candidates = [
                        c for c in candidate_records
                        if c.get("rewrite", "").strip()
                    ]
                    if scorable_candidates:
                        rewrite_toxicities = compute_multimodal_hatefulness(
                            [c["image_path"] for c in scorable_candidates],
                            [c["rewrite"] for c in scorable_candidates],
                            hate_scorer,
                            batch_size=args.score_batch_size,
                        )
                        sta_scores = [1.0 - tox for tox in rewrite_toxicities]
                        bertscores = compute_bertscore_batch(
                            [c["original_text"] for c in scorable_candidates],
                            [c["rewrite"] for c in scorable_candidates],
                            scorer=bertscore_scorer,
                            batch_size=max(32, args.batch_size * 8),
                        )

                        for cand, sta_score, rewrite_toxicity, bertscore in zip(
                            scorable_candidates,
                            sta_scores,
                            rewrite_toxicities,
                            bertscores,
                        ):
                            original_toxicity = cand["original_toxicity"]
                            toxicity_drop = original_toxicity - rewrite_toxicity
                            required_toxicity_drop = (
                                args.min_toxicity_drop
                                if original_toxicity >= args.min_source_toxicity_for_drop
                                else 0.0
                            )
                            passes_sta = sta_score > args.sta_threshold
                            passes_bertscore = bertscore > args.bertscore_min
                            if args.bertscore_max < 1.0:
                                passes_bertscore = passes_bertscore and (bertscore < args.bertscore_max)
                            passes_toxicity_delta = toxicity_drop >= required_toxicity_drop
                            q_reason = ""
                            if cand.get("format_valid", False):
                                q_reason = quality_reject_reason(
                                    passes_sta=passes_sta,
                                    passes_bertscore=passes_bertscore,
                                    passes_toxicity_delta=passes_toxicity_delta,
                                    bertscore=bertscore,
                                    bertscore_max=args.bertscore_max,
                                )
                            cand.update(
                                {
                                    "sta_score": float(sta_score),
                                    "rewrite_toxicity": float(rewrite_toxicity),
                                    "bertscore": float(bertscore),
                                    "toxicity_drop": float(toxicity_drop),
                                    "passes_sta": passes_sta if cand.get("format_valid", False) else False,
                                    "passes_bertscore": passes_bertscore if cand.get("format_valid", False) else False,
                                    "passes_toxicity_delta": passes_toxicity_delta if cand.get("format_valid", False) else False,
                                    "passes_all": (
                                        cand.get("format_valid", False)
                                        and passes_sta
                                        and passes_bertscore
                                        and passes_toxicity_delta
                                    ),
                                    "quality_reason": q_reason if cand.get("format_valid", False) else cand.get("format_reason", ""),
                                }
                            )

                    candidates_by_slot: Dict[int, List[Dict[str, Any]]] = {slot: [] for slot in active_indices}
                    for cand in candidate_records:
                        candidates_by_slot[cand["slot_idx"]].append(cand)

                    unresolved: List[int] = []
                    for active_slot in active_indices:
                        slot_candidates = candidates_by_slot.get(active_slot, [])
                        example_id = rw_ids[active_slot]
                        passing = [cand for cand in slot_candidates if cand["passes_all"]]

                        if passing:
                            best_passing = max(passing, key=candidate_rank_key)
                            best_so_far = selected_candidates.get(active_slot)
                            if best_so_far is None or candidate_rank_key(best_passing) > candidate_rank_key(best_so_far):
                                selected_candidates[active_slot] = best_passing
                            continue

                        unresolved.append(active_slot)
                        if slot_candidates:
                            best_failed = max(slot_candidates, key=candidate_rank_key)
                            best_so_far = selected_candidates.get(active_slot)
                            if best_so_far is None or candidate_rank_key(best_failed) > candidate_rank_key(best_so_far):
                                selected_candidates[active_slot] = best_failed
                            next_reason = best_failed["quality_reason"] or "low_sta"
                            retry_feedback_reasons[active_slot] = next_reason
                            if best_failed.get("format_valid", False):
                                quality_reject_count += 1
                                quality_reject_reason_counts[next_reason] = quality_reject_reason_counts.get(next_reason, 0) + 1
                            logger.info(
                                "No acceptable rewrite for %s on attempt %d/%d; best candidate failed (%s).",
                                example_id,
                                attempt_idx + 1,
                                args.rewrite_max_attempts,
                                next_reason,
                            )
                        else:
                            if active_slot in generation_failed_slots:
                                retry_feedback_reasons[active_slot] = "empty"
                            else:
                                next_reason = most_common_reason(invalid_reasons_by_slot.get(active_slot, []), fallback="empty")
                                retry_feedback_reasons[active_slot] = next_reason
                                logger.info(
                                    "Rejected rewrite candidates for %s (attempt %d/%d, dominant reason=%s)",
                                    example_id,
                                    attempt_idx + 1,
                                    args.rewrite_max_attempts,
                                    next_reason,
                                )

                    active_indices = unresolved

                for slot_idx in range(len(pending_rows)):
                    selected = selected_candidates.get(slot_idx)
                    if selected is None:
                        continue
                    example_id = rw_ids[slot_idx]
                    source_row = pending_rows[slot_idx]
                    explanation = rw_explanations[slot_idx]
                    original_text = rw_original_texts[slot_idx]

                    written_rewrites += 1
                    if selected.get("passes_all", False):
                        passed_rewrites += 1
                    rewrite_record = {
                        "id": example_id,
                        "image_path": source_row.get("image_path"),
                        "original_text": original_text,
                        "explanation": explanation,
                        "pseudo_rewrite": selected["rewrite"],
                        "sta_score": (
                            float(selected["sta_score"])
                            if selected.get("sta_score") is not None else None
                        ),
                        "bertscore": (
                            float(selected["bertscore"])
                            if selected.get("bertscore") is not None else None
                        ),
                        "original_toxicity": float(selected["original_toxicity"]),
                        "rewrite_toxicity": (
                            float(selected["rewrite_toxicity"])
                            if selected.get("rewrite_toxicity") is not None else None
                        ),
                        "toxicity_drop": (
                            float(selected["toxicity_drop"])
                            if selected.get("toxicity_drop") is not None else None
                        ),
                        "passed_stage1_filters": bool(selected.get("passes_all", False)),
                        "passes_sta": bool(selected.get("passes_sta", False)),
                        "passes_bertscore": bool(selected.get("passes_bertscore", False)),
                        "passes_toxicity_delta": bool(selected.get("passes_toxicity_delta", False)),
                        "format_valid": bool(selected.get("format_valid", False)),
                        "format_reason": selected.get("format_reason", ""),
                        "reject_reason": selected.get("quality_reason", ""),
                    }
                    rewrites_batch.append(rewrite_record)
                    processed_rewrite_ids.add(example_id)

                pbar.update(len(pending_rows))

                if total_examples >= next_stats_log:
                    write_rate = 100 * written_rewrites / max(total_pseudo_rewrites, 1)
                    pass_rate = 100 * passed_rewrites / max(total_pseudo_rewrites, 1)
                    logger.info(
                        "[%d/%d] rewrites_written=%d/%d (%.1f%%) | rewrites_passing=%d (%.1f%%) | rewrite_invalid=%d | quality_rejected=%d | rewrite_failures=%d",
                        total_examples,
                        len(explanation_records),
                        written_rewrites,
                        total_pseudo_rewrites,
                        write_rate,
                        passed_rewrites,
                        pass_rate,
                        invalid_rewrite_format,
                        quality_reject_count,
                        rewrite_generation_failures,
                    )
                    next_stats_log = ((total_examples // 50) + 1) * 50

                if rewrites_batch and total_pseudo_rewrites >= next_rewrites_flush:
                    write_jsonl_batch(rewrites_batch, pseudo_rewrites_path)
                    rewrites_batch = []
                    logger.info(f"Wrote rewrites batch at candidate count {total_pseudo_rewrites}")
                    next_rewrites_flush = ((total_pseudo_rewrites // 100) + 1) * 100

        if rewrites_batch:
            write_jsonl_batch(rewrites_batch, pseudo_rewrites_path)
            logger.info(f"Wrote final rewrites batch ({len(rewrites_batch)} items)")

        write_rate = (written_rewrites / max(total_pseudo_rewrites, 1)) * 100
        pass_rate = (passed_rewrites / max(total_pseudo_rewrites, 1)) * 100

        logger.info("\n=== Rewrites-Only Summary ===")
        logger.info(f"Total explanation rows considered: {len(explanation_records)}")
        logger.info(f"Total pending rows processed: {total_examples}")
        logger.info(f"Total pseudo-rewrites attempted: {total_pseudo_rewrites}")
        logger.info(f"Pseudo-rewrites written (best candidate): {written_rewrites}")
        logger.info(f"Write rate: {write_rate:.2f}%")
        logger.info(f"Pseudo-rewrites passing Stage 1 filters: {passed_rewrites}")
        logger.info(f"Pass rate: {pass_rate:.2f}%")
        logger.info(f"Rejected rewrites due to invalid format: {invalid_rewrite_format}")
        if invalid_rewrite_reason_counts:
            logger.info(f"Invalid rewrite reasons: {invalid_rewrite_reason_counts}")
        logger.info(f"Rejected rewrites due to quality filter: {quality_reject_count}")
        if quality_reject_reason_counts:
            logger.info(f"Quality reject reasons: {quality_reject_reason_counts}")
        logger.info(f"Rewrite generation failures: {rewrite_generation_failures}")
        logger.info(f"Explanations JSONL input: {explanations_path}")
        logger.info(f"Pseudo-rewrites JSONL output: {pseudo_rewrites_path}")

    finally:
        emissions = tracker.stop()
        if emissions is not None:
            logger.info(f"Carbon emissions: {emissions:.6f} kg CO2")
            logger.info(f"Emissions saved to: {os.path.join(args.output_dir, emissions_file)}")
        else:
            logger.warning("Carbon emissions could not be measured (CodeCarbon tracking failed)")


if __name__ == "__main__":
    main()
