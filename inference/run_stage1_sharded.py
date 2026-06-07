"""
Stage 1 inference (sharded): LLaVA-Next structured explanations + pseudo-rewrites.

Pipeline position: AFTER filter_meme_images.py (Stage 0); BEFORE build_stage2_dataset.py.

Inputs : Stage 0 manifest CSV, per-dataset image directories.
Outputs: <output_dir>/<dataset>_explanations_shard<N>of<M>.jsonl
         <output_dir>/<dataset>_pseudo_rewrites_shard<N>of<M>.jsonl

Design decisions:
- Sharding strategy: row index % num_shards == shard_id, so any num_shards
  value yields deterministic, balanced, non-overlapping partitions that can
  be merged with merge_stage1_rewrites_shards.py.
- All records (passed and rejected) are written with a passed_stage1_filters
  flag, keeping the output consistent with the training set behaviour in
  build_stage2_dataset.py.
- Quality filters (STA, BERTScore, toxicity drop) are gated on thresholds
  passed via CLI so that the same script covers both strict production runs
  and relaxed research sweeps.
- Text STA uses s-nlp/roberta_toxicity_classifier (not visual) to ensure
  filter consistency with the training-time label used by train_stage2.py.
"""

import argparse
from collections import Counter
import json
import logging
import os
import random
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from codecarbon import EmissionsTracker
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.utils import logging as hf_logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.explainer import MemeExplainer
from utils.bertscore_utils import compute_bertscore_batch, create_bertscore_scorer

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_text_toxicity(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    device: str,
    batch_size: int = 32,
) -> List[float]:
    """Return P(toxic) per text from s-nlp/roberta_toxicity_classifier (class index 1).

    Consistent with the STA logic used by train_stage2.py so that
    passed_stage1_filters matches training-time filter behaviour.
    """
    if not texts:
        return []
    out: List[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            out.extend(probs[:, 1].cpu().tolist())
    return out


def load_existing_ids(jsonl_path: str) -> set:
    """Load IDs already written to a shard JSONL to support resume from interruption."""
    processed = set()
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed.add(data.get("id"))
        except Exception as e:
            logger.warning(f"Could not load existing IDs from {jsonl_path}: {e}")
    return processed


def write_jsonl_batch(data: List[Dict], output_path: str) -> None:
    """Append a list of records to a JSONL file (periodic flush to limit memory)."""
    with open(output_path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
DOMAIN_RE = re.compile(
    r"(?i)\b[a-z0-9][a-z0-9-]{1,62}\.(?:com|org|net|co|io|ai|edu|gov|uk|us|ru|de|fr|it|me|ly|info|biz)(?:/\S*)?\b"
)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
LEADING_LABEL_RE = re.compile(
    r"(?i)^\s*(?:rewrite|rewritten text|rewritten_text|output|answer|response)\s*:\s*"
)


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _token_change_ratio(original_text: str, rewrite_text: str) -> float:
    """
    Compute how much token content changed between original and rewrite.

    Returns value in [0,1], where 0 means "identical bag of tokens"
    and 1 means "no overlap".
    """
    orig_tokens = _normalize_for_compare(original_text).split()
    rew_tokens = _normalize_for_compare(rewrite_text).split()
    if not orig_tokens and not rew_tokens:
        return 0.0
    if not orig_tokens or not rew_tokens:
        return 1.0

    overlap = sum((Counter(orig_tokens) & Counter(rew_tokens)).values())
    denom = max(len(orig_tokens), len(rew_tokens), 1)
    return 1.0 - (overlap / denom)


def sanitize_generated_rewrite(text: str) -> str:
    """Strip LLaVA instruction-template wrappers and metadata artifacts from raw rewrite output."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    if "[/INST]" in cleaned:
        cleaned = cleaned.split("[/INST]")[-1].strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = " ".join(lines).strip()

    cleaned = LEADING_LABEL_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = HASHTAG_RE.sub(" ", cleaned)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = DOMAIN_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\u2022", " ").replace("\ufffd", " ")
    cleaned = re.sub(r"([!?.,;:])\1{2,}", r"\1", cleaned)
    cleaned = re.sub(r"\s+([!?.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("\"'` ").strip()
    return cleaned


def has_invalid_rewrite_format(
    rewrite: str,
    original_text: str,
    min_lexical_change: float = 0.0,
    max_char_similarity: float = 1.0,
) -> tuple[bool, str]:
    """Format-level quality gate: reject rewrites with URLs, artifacts, repetition, or no edit."""
    text = (rewrite or "").strip()
    if not text:
        return True, "empty"

    if URL_RE.search(text) or DOMAIN_RE.search(text):
        return True, "url"
    if MENTION_RE.search(text):
        return True, "mention"
    if HASHTAG_RE.search(text):
        return True, "hashtag"

    tokens = text.split()
    if len(tokens) < 2:
        return True, "too_short"

    if len(text) > 280:
        return True, "too_long"

    lower_tokens = [t.lower() for t in tokens]
    if len(tokens) >= 8:
        # Repetition heuristics applied only to sequences long enough to be meaningful.
        unique_ratio = len(set(lower_tokens)) / max(len(lower_tokens), 1)
        if unique_ratio < 0.35:
            return True, "low_diversity"
        counts = {}
        for tok in lower_tokens:
            counts[tok] = counts.get(tok, 0) + 1
        if (max(counts.values()) / len(lower_tokens)) > 0.45:
            return True, "repetition"

    non_alnum_ratio = sum(
        1 for c in text if (not c.isalnum() and not c.isspace())
    ) / max(len(text), 1)
    if non_alnum_ratio > 0.35:
        return True, "symbol_heavy"

    if _normalize_for_compare(text) == _normalize_for_compare(original_text):
        return True, "no_edit"

    # Lexical- and character-level similarity guards; disabled (= 0.0/1.0) by default.
    if min_lexical_change > 0.0:
        token_change = _token_change_ratio(original_text, text)
        if token_change < min_lexical_change:
            return True, "too_similar"

    if max_char_similarity < 1.0:
        original_norm = _normalize_for_compare(original_text)
        rewrite_norm = _normalize_for_compare(text)
        if len(original_norm) >= 24:
            char_similarity = SequenceMatcher(None, original_norm, rewrite_norm).ratio()
            if char_similarity > max_char_similarity:
                return True, "too_similar"

    return False, ""


def ensure_explanation_non_null(explanation: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """Replace null/missing explanation fields with conservative non-null fallback strings."""
    normalized = dict(explanation) if isinstance(explanation, dict) else {}
    changed = False

    target_group = normalized.get("target_group")
    if not isinstance(target_group, str) or not target_group.strip() or target_group.strip().lower() in {
        "null", "none", "n/a", "na", "unknown"
    }:
        normalized["target_group"] = "other"
        changed = True

    visual_evidence = normalized.get("visual_evidence")
    if (
        not isinstance(visual_evidence, str)
        or not visual_evidence.strip()
        or visual_evidence.strip().lower() in {"null", "none", "n/a", "na", "unknown"}
    ):
        normalized["visual_evidence"] = (
            "A visual cue in the meme is used to frame the target group negatively."
        )
        changed = True

    implicit_meaning = normalized.get("implicit_meaning")
    if (
        not isinstance(implicit_meaning, str)
        or not implicit_meaning.strip()
        or implicit_meaning.strip().lower() in {"null", "none", "n/a", "na", "unknown"}
    ):
        normalized["implicit_meaning"] = (
            "The meme uses both text and visual context to communicate a hateful or derogatory framing toward a target group."
        )
        changed = True

    return normalized, changed


def main():
    parser = argparse.ArgumentParser(description="Stage 1 (multimodal + sharded): Generate explanations and pseudo-rewrites")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'training')")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest CSV from Stage 0 (output of filter_meme_images.py)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSONL files")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load LLaVA in 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_shards", type=int, default=8, help="Total number of shards for parallel processing")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id to process in [0, num_shards-1]")
    parser.add_argument("--hateful_only", action="store_true", help="Only process examples where hateful=1 (skip non-hateful memes)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process max 16 examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sta_threshold", type=float, default=0.45, help="Minimum non-hateful score (1 - hatefulness) for keeping rewrites")
    parser.add_argument("--bertscore_min", type=float, default=0.25, help="Minimum BERTScore similarity for keeping rewrites")
    parser.add_argument("--bertscore_max", type=float, default=1.0, help="Maximum BERTScore similarity to avoid near-copy rewrites (set 1.0 to disable)")
    parser.add_argument("--min_lexical_change", type=float, default=0.0, help="Minimum token-level change ratio required between original and rewrite (set 0.0 to disable)")
    parser.add_argument("--max_char_similarity", type=float, default=1.0, help="Maximum normalized char-level similarity allowed between original and rewrite (set 1.0 to disable)")
    parser.add_argument("--min_toxicity_drop", type=float, default=0.0, help="Minimum required hatefulness decrease from original to rewrite (set 0.0 to disable)")
    parser.add_argument("--min_source_toxicity_for_drop", type=float, default=0.20, help="Only enforce min_toxicity_drop when original hatefulness is at least this value")
    parser.add_argument("--explain_max_retries", type=int, default=0, help="Additional retries for explanation generation (0 => single attempt)")
    parser.add_argument("--rewrite_max_attempts", type=int, default=2, help="Maximum rewrite attempts per example")

    args = parser.parse_args()
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
    if args.explain_max_retries < 0:
        raise ValueError("--explain_max_retries must be >= 0")
    if args.rewrite_max_attempts < 1:
        raise ValueError("--rewrite_max_attempts must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards-1]")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    shard_tag = f"shard{args.shard_id:02d}of{args.num_shards:02d}"
    stage1_log_path = os.path.join(args.output_dir, f"stage1_{shard_tag}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(stage1_log_path),
            logging.StreamHandler()
        ]
    )

    # Suppress noisy third-party loggers so per-example progress is visible.
    for noisy_logger in [
        "httpx",
        "huggingface_hub",
        "urllib3",
        "matplotlib",
        "PIL",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    print(f"\n{'='*60}")
    print(f"  Stage 1: LLaVA Explanations + Pseudo-rewrites")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Images:     {args.images_dir}")
    print(f"  Manifest:   {args.manifest_path}")
    print(f"  Output:     {args.output_dir}")
    print(f"  HF cache:   {args.hf_cache}")
    print(f"  4-bit quant:{args.load_in_4bit}")
    print(f"  Shard:      {args.shard_id + 1}/{args.num_shards} ({shard_tag})")
    print(f"  Debug:      {args.debug}")
    print(f"{'='*60}\n")
    logger.info(f"Starting Stage 1 with dataset={args.dataset}, debug={args.debug}")
    logger.info(f"Arguments: {vars(args)}")

    # ── Load and filter manifest ───────────────────────────────────────────────
    manifest_df = pd.read_csv(args.manifest_path)
    total_in_manifest = len(manifest_df)
    kept_in_manifest = int(manifest_df["kept"].sum()) if "kept" in manifest_df.columns else total_in_manifest
    logger.info(f"Manifest loaded: {total_in_manifest} total rows, {kept_in_manifest} kept by Stage 0")
    manifest_df = manifest_df[manifest_df["kept"] == True] if "kept" in manifest_df.columns else manifest_df
    if args.hateful_only and "hateful" in manifest_df.columns:
        before = len(manifest_df)
        manifest_df = manifest_df[manifest_df["hateful"] == 1]
        logger.info(f"--hateful_only: kept {len(manifest_df)}/{before} hateful examples")
    if args.debug:
        manifest_df = manifest_df.head(16)
    logger.info(f"Manifest rows after filters (before sharding): {len(manifest_df)}")

    # ── Load models ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("No GPU found — running on CPU (will be slow)")

    explainer = MemeExplainer(
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.hf_cache,
        device=device,
        debug=args.debug,
    )

    # BERTScorer is loaded once and reused; per-call reload would re-download weights.
    bertscore_scorer = create_bertscore_scorer(device=device)

    # Text STA classifier (not visual) for quality filter, consistent with train_stage2.py.
    _sta_tok = AutoTokenizer.from_pretrained(
        "s-nlp/roberta_toxicity_classifier", cache_dir=args.hf_cache
    )
    _sta_model = AutoModelForSequenceClassification.from_pretrained(
        "s-nlp/roberta_toxicity_classifier", cache_dir=args.hf_cache
    ).to(device).eval()

    # ── Output paths and resume state ─────────────────────────────────────────
    explanations_path = os.path.join(args.output_dir, f"{args.dataset}_explanations_{shard_tag}.jsonl")
    pseudo_rewrites_path = os.path.join(args.output_dir, f"{args.dataset}_pseudo_rewrites_{shard_tag}.jsonl")

    processed_explanation_ids = load_existing_ids(explanations_path)
    processed_rewrite_ids = load_existing_ids(pseudo_rewrites_path)

    logger.info(f"Already processed explanations: {len(processed_explanation_ids)}")
    logger.info(f"Already processed rewrites: {len(processed_rewrite_ids)}")

    # ── Processing loop ───────────────────────────────────────────────────────
    explanations_batch = []
    rewrites_batch = []
    json_parse_failures = 0
    forced_non_null_explanations = 0
    total_examples = 0
    kept_rewrites = 0
    total_pseudo_rewrites = 0
    invalid_rewrite_format = 0
    invalid_rewrite_reason_counts: Dict[str, int] = {}
    quality_reject_count = 0
    quality_reject_reason_counts: Dict[str, int] = {}
    rewrite_generation_failures = 0

    emissions_file = f"emissions_{shard_tag}.csv"
    tracker = EmissionsTracker(log_level="warning", output_dir=args.output_dir, output_file=emissions_file)
    tracker.start()

    try:
        all_records = manifest_df.to_dict("records")
        # Round-robin partition: deterministic, balanced, no overlap across shards.
        records = [row for idx, row in enumerate(all_records) if idx % args.num_shards == args.shard_id]
        logger.info(
            "Shard %d/%d selected %d/%d rows",
            args.shard_id,
            args.num_shards,
            len(records),
            len(all_records),
        )
        next_stats_log = 50
        next_explanations_flush = 100
        next_rewrites_flush = 100

        with tqdm.tqdm(total=len(records), desc="Processing examples") as pbar:
            for start_idx in range(0, len(records), args.batch_size):
                raw_batch_rows = records[start_idx:start_idx + args.batch_size]
                pending_rows = []

                for row in raw_batch_rows:
                    example_id = row.get("id")
                    if example_id in processed_explanation_ids:
                        pbar.update(1)
                        continue
                    pending_rows.append(row)

                if not pending_rows:
                    continue

                batch_ids: List[str] = []
                batch_image_paths: List[str] = []
                batch_original_texts: List[str] = []
                batch_hateful_flags: List[bool] = []

                for row in pending_rows:
                    example_id = row.get("id")
                    raw_img = str(row.get("image_path", ""))
                    if os.path.isabs(raw_img) and os.path.exists(raw_img):
                        image_path = raw_img
                    else:
                        image_path = os.path.join(args.images_dir, raw_img)
                    original_text = str(row.get("text", "") or row.get("ocr_text", ""))
                    is_hateful = bool(row.get("hateful", False))

                    batch_ids.append(example_id)
                    batch_image_paths.append(image_path)
                    batch_original_texts.append(original_text)
                    batch_hateful_flags.append(is_hateful)

                total_examples += len(pending_rows)

                try:
                    batch_explanations = explainer.batch_explain(
                        batch_image_paths,
                        batch_original_texts,
                        max_retries=args.explain_max_retries,
                    )
                except Exception as e:
                    logger.warning(
                        f"Batch explanation generation failed for rows "
                        f"{start_idx}-{start_idx + len(pending_rows) - 1}: {e}. "
                        f"Falling back to per-example explain."
                    )
                    batch_explanations = []
                    for image_path, original_text, is_hateful in zip(
                        batch_image_paths, batch_original_texts, batch_hateful_flags
                    ):
                        try:
                            explanation = explainer.explain(
                                image_path,
                                original_text,
                                max_retries=args.explain_max_retries,
                            )
                        except Exception as inner_e:
                            if is_hateful:
                                explanation = {
                                    "target_group": "other",
                                    "visual_evidence": (
                                        "A visual cue in the meme is used to frame the target group negatively."
                                    ),
                                    "implicit_meaning": (
                                        "The meme uses both text and visual context to communicate a hateful or derogatory framing toward a target group."
                                    ),
                                    "error": str(inner_e),
                                }
                            else:
                                explanation = {
                                    "target_group": None,
                                    "visual_evidence": None,
                                    "implicit_meaning": None,
                                    "error": str(inner_e),
                                }
                        batch_explanations.append(explanation)

                for i, row in enumerate(pending_rows):
                    example_id = batch_ids[i]
                    explanation = batch_explanations[i]
                    is_hateful = batch_hateful_flags[i]
                    original_text = batch_original_texts[i]

                    if explanation.get("parse_error"):
                        json_parse_failures += 1

                    if is_hateful:
                        explanation, was_forced = ensure_explanation_non_null(explanation)
                        if was_forced:
                            forced_non_null_explanations += 1

                    batch_explanations[i] = explanation

                    explanation_record = {
                        "id": example_id,
                        "image_path": row.get("image_path"),
                        "original_text": original_text,
                        "explanation": explanation,
                        "is_hateful": is_hateful,
                    }
                    explanations_batch.append(explanation_record)
                    processed_explanation_ids.add(example_id)

                rewrite_positions = [
                    i for i, example_id in enumerate(batch_ids)
                    if batch_hateful_flags[i] and example_id not in processed_rewrite_ids
                ]

                if rewrite_positions:
                    total_pseudo_rewrites += len(rewrite_positions)
                    rw_ids = [batch_ids[i] for i in rewrite_positions]
                    rw_image_paths = [batch_image_paths[i] for i in rewrite_positions]
                    rw_original_texts = [batch_original_texts[i] for i in rewrite_positions]
                    rw_explanations = [batch_explanations[i] for i in rewrite_positions]
                    # Text toxicity for original: used for toxicity_drop calculation.
                    rw_original_toxicities = _compute_text_toxicity(
                        rw_original_texts, _sta_model, _sta_tok, device
                    )

                    cleaned_rewrites: List[Optional[str]] = [None] * len(rewrite_positions)
                    active_indices = list(range(len(rewrite_positions)))
                    max_rewrite_attempts = args.rewrite_max_attempts

                    for attempt_idx in range(max_rewrite_attempts):
                        if not active_indices:
                            break

                        active_image_paths = [rw_image_paths[i] for i in active_indices]
                        active_original_texts = [rw_original_texts[i] for i in active_indices]
                        active_explanations = [rw_explanations[i] for i in active_indices]
                        active_ids = [rw_ids[i] for i in active_indices]

                        try:
                            raw_rewrites = explainer.batch_rewrite(
                                active_image_paths,
                                active_original_texts,
                                active_explanations,
                            )
                        except Exception as e:
                            rewrite_generation_failures += len(active_indices)
                            logger.warning(
                                "Batch rewrite generation failed at attempt %d/%d for %d examples: %s",
                                attempt_idx + 1,
                                max_rewrite_attempts,
                                len(active_indices),
                                e,
                            )
                            break

                        unresolved: List[int] = []
                        for pos, active_slot in enumerate(active_indices):
                            example_id = active_ids[pos]
                            raw_rewrite = raw_rewrites[pos] if pos < len(raw_rewrites) else ""

                            if isinstance(raw_rewrite, str) and raw_rewrite.startswith("[REWRITE ERROR:"):
                                rewrite_generation_failures += 1
                                logger.warning(
                                    f"Rewrite generation failed for {example_id} "
                                    f"(attempt {attempt_idx + 1}/{max_rewrite_attempts}): {raw_rewrite}"
                                )
                                unresolved.append(active_slot)
                                continue

                            candidate = sanitize_generated_rewrite(raw_rewrite)
                            is_invalid, reason = has_invalid_rewrite_format(
                                candidate,
                                rw_original_texts[active_slot],
                                min_lexical_change=args.min_lexical_change,
                                max_char_similarity=args.max_char_similarity,
                            )
                            if is_invalid:
                                invalid_rewrite_format += 1
                                invalid_rewrite_reason_counts[reason] = (
                                    invalid_rewrite_reason_counts.get(reason, 0) + 1
                                )
                                logger.info(
                                    f"Rejected rewrite for {example_id} "
                                    f"(attempt {attempt_idx + 1}/{max_rewrite_attempts}, reason={reason})"
                                )
                                unresolved.append(active_slot)
                                continue

                            cleaned_rewrites[active_slot] = candidate

                        active_indices = unresolved

                    kept_slots = [i for i, text in enumerate(cleaned_rewrites) if text]
                    if kept_slots:
                        kept_rewrite_texts = [cleaned_rewrites[i] for i in kept_slots]
                        kept_original_texts = [rw_original_texts[i] for i in kept_slots]
                        kept_image_paths = [rw_image_paths[i] for i in kept_slots]
                        kept_original_toxicities = [rw_original_toxicities[i] for i in kept_slots]

                        # Text toxicity for rewrite: derives sta_score and toxicity_drop.
                        rewrite_text_toxicities = _compute_text_toxicity(
                            kept_rewrite_texts, _sta_model, _sta_tok, device
                        )
                        sta_scores = [1.0 - tox for tox in rewrite_text_toxicities]
                        bertscores = compute_bertscore_batch(
                            kept_original_texts,
                            kept_rewrite_texts,
                            scorer=bertscore_scorer,
                            batch_size=max(32, args.batch_size * 8),
                        )

                        for slot_idx, rewrite, sta_score, rewrite_toxicity, bertscore, original_toxicity in zip(
                            kept_slots,
                            kept_rewrite_texts,
                            sta_scores,
                            rewrite_text_toxicities,
                            bertscores,
                            kept_original_toxicities,
                        ):
                            example_id = rw_ids[slot_idx]
                            row_idx = rewrite_positions[slot_idx]
                            source_row = pending_rows[row_idx]
                            explanation = rw_explanations[slot_idx]
                            original_text = rw_original_texts[slot_idx]

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
                            passed = passes_sta and passes_bertscore and passes_toxicity_delta

                            if passed:
                                kept_rewrites += 1
                            else:
                                quality_reject_count += 1

                            # Write all records (passed and failed) with passed_stage1_filters
                            # so that build_stage2_dataset.py can apply its own threshold.
                            rewrite_record = {
                                "id": example_id,
                                "image_path": source_row.get("image_path"),
                                "original_text": original_text,
                                "explanation": explanation,
                                "pseudo_rewrite": rewrite,
                                "sta_score": float(sta_score),
                                "bertscore": float(bertscore),
                                "original_toxicity": float(original_toxicity),
                                "rewrite_toxicity": float(rewrite_toxicity),
                                "toxicity_drop": float(toxicity_drop),
                                "passed_stage1_filters": passed,
                                "passes_sta": passes_sta,
                                "passes_bertscore": passes_bertscore,
                                "passes_toxicity_delta": passes_toxicity_delta,
                                "reject_reason": (
                                    "" if passed else (
                                        "low_sta" if not passes_sta else
                                        "low_bertscore" if not passes_bertscore else
                                        "low_toxicity_drop"
                                    )
                                ),
                            }
                            rewrites_batch.append(rewrite_record)
                            processed_rewrite_ids.add(example_id)

                            if not passed:
                                if not passes_sta:
                                    q_reason = "low_sta"
                                elif bertscore <= args.bertscore_min:
                                    q_reason = "low_bertscore"
                                elif bertscore >= args.bertscore_max:
                                    q_reason = "high_bertscore"
                                else:
                                    q_reason = "low_toxicity_drop"
                                quality_reject_reason_counts[q_reason] = (
                                    quality_reject_reason_counts.get(q_reason, 0) + 1
                                )

                pbar.update(len(pending_rows))

                if total_examples >= next_stats_log:
                    keep_rate = 100 * kept_rewrites / max(total_pseudo_rewrites, 1)
                    logger.info(
                        f"[{total_examples}/{len(records)}] "
                        f"explanations={total_examples} | "
                        f"rewrites_kept={kept_rewrites}/{total_pseudo_rewrites} ({keep_rate:.1f}%) | "
                        f"json_failures={json_parse_failures} | "
                        f"forced_non_null={forced_non_null_explanations} | "
                        f"rewrite_invalid={invalid_rewrite_format} | "
                        f"quality_rejected={quality_reject_count} | "
                        f"rewrite_failures={rewrite_generation_failures}"
                    )
                    next_stats_log = ((total_examples // 50) + 1) * 50

                if explanations_batch and total_examples >= next_explanations_flush:
                    write_jsonl_batch(explanations_batch, explanations_path)
                    explanations_batch = []
                    logger.info(f"Wrote explanations batch at example {total_examples}")
                    next_explanations_flush = ((total_examples // 100) + 1) * 100

                if rewrites_batch and total_pseudo_rewrites >= next_rewrites_flush:
                    write_jsonl_batch(rewrites_batch, pseudo_rewrites_path)
                    rewrites_batch = []
                    logger.info(f"Wrote rewrites batch at example {total_pseudo_rewrites}")
                    next_rewrites_flush = ((total_pseudo_rewrites // 100) + 1) * 100

        # ── Final flush ───────────────────────────────────────────────────────
        if explanations_batch:
            write_jsonl_batch(explanations_batch, explanations_path)
            logger.info(f"Wrote final explanations batch ({len(explanations_batch)} items)")

        if rewrites_batch:
            write_jsonl_batch(rewrites_batch, pseudo_rewrites_path)
            logger.info(f"Wrote final rewrites batch ({len(rewrites_batch)} items)")

        # ── Shard summary ─────────────────────────────────────────────────────
        json_parse_rate = (json_parse_failures / max(total_examples, 1)) * 100
        keep_rate = (kept_rewrites / max(total_pseudo_rewrites, 1)) * 100

        logger.info(f"\n=== Stage 1 Summary ===")
        logger.info(f"Total examples processed: {total_examples}")
        logger.info(f"JSON parse failures: {json_parse_failures} ({json_parse_rate:.2f}%)")
        logger.info(f"Hateful explanations forced to non-null: {forced_non_null_explanations}")
        logger.info(f"Total pseudo-rewrites generated: {total_pseudo_rewrites}")
        logger.info(f"Pseudo-rewrites kept (passed filters): {kept_rewrites}")
        logger.info(f"Keep rate: {keep_rate:.2f}%")
        logger.info(f"Rejected rewrites due to invalid format: {invalid_rewrite_format}")
        if invalid_rewrite_reason_counts:
            logger.info(f"Invalid rewrite reasons: {invalid_rewrite_reason_counts}")
        logger.info(f"Rejected rewrites due to quality filter: {quality_reject_count}")
        if quality_reject_reason_counts:
            logger.info(f"Quality reject reasons: {quality_reject_reason_counts}")
        logger.info(f"Rewrite generation failures: {rewrite_generation_failures}")
        logger.info(f"Explanations JSONL: {explanations_path}")
        logger.info(f"Pseudo-rewrites JSONL: {pseudo_rewrites_path}")

    finally:
        emissions = tracker.stop()
        if emissions is not None:
            logger.info(f"Carbon emissions: {emissions:.6f} kg CO2")
            logger.info(f"Emissions saved to: {os.path.join(args.output_dir, emissions_file)}")
        else:
            logger.warning("Carbon emissions could not be measured (CodeCarbon tracking failed)")


if __name__ == "__main__":
    main()
