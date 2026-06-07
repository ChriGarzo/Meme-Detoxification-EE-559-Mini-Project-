"""
Stage 2 dataset construction: build BART training data from Stage 1 pseudo-rewrites.

Data source   : train_pseudo_rewrites_merged.jsonl (merged Stage 1 LLaVA outputs).
Output        : train.jsonl, val.jsonl, and optionally test.jsonl in --output_dir,
                plus a dataset_statistics.json manifest.

Quality filter chain applied to each pseudo-rewrite row
--------------------------------------------------------
1. parse_error == False      — drop rows where the Stage 1 JSON parser failed.
2. passed_stage1_filters     — drop rows that did not meet Stage 1 quality gates
                               (STA threshold, BERTScore floor, etc.); disable
                               with --allow_failed_stage1_filters.
3. toxicity_drop > 0         — require a strictly positive text-toxicity reduction;
                               adjustable via --min_toxicity_drop.
4. Rewrite text sanity       — sanitize_rewrite_text() strips artifacts (URLs,
                               mentions, hashtags, LLM preambles), then
                               is_valid_rewrite_text() discards near-copies,
                               repetition-heavy, symbol-heavy, or near-empty outputs.
5. source_similarity ceiling — optionally drop near-verbatim pseudo-rewrites via
                               --max_source_similarity (normalized char similarity).

Val / test sets are built from dedicated Stage 1 output files when provided
(--val_rewrites_path, --test_rewrites_path); otherwise val is an internal
random split of the train data (legacy fallback, not recommended for evaluation).
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Tuple, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.debug import set_seeds


logger = logging.getLogger(__name__)


def make_debug_dataset() -> List[Dict]:
    """Return two hard-coded examples for smoke-testing the build pipeline."""
    debug_examples = [
        {
            "id": "debug_001",
            "original_text": "that's disgusting",
            "target_group": "race_ethnicity",
            "visual_evidence": "person with insulting hand gesture",
            "explanation": "This uses contempt language targeting a racial group",
            "pseudo_rewrite": "I disagree with that viewpoint",
            "bert_score": 0.85
        },
        {
            "id": "debug_002",
            "original_text": "stupid people",
            "target_group": "disability",
            "visual_evidence": "mocking face expression",
            "explanation": "This mocks disabled individuals",
            "pseudo_rewrite": "people with different perspectives",
            "bert_score": 0.72
        }
    ]
    return debug_examples


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_parse_error(row: Dict) -> bool:
    expl = row.get("explanation")
    if isinstance(expl, dict):
        return bool(expl.get("parse_error", False))
    return bool(row.get("parse_error", False))


def _extract_passed_stage1_filters(row: Dict) -> bool:
    return bool(row.get("passed_stage1_filters", False))


def load_merged_rewrites(merged_path: str) -> List[Dict]:
    """Parse a merged pseudo-rewrite JSONL and normalise each row into a flat dict."""
    path = Path(merged_path)
    if not path.exists():
        logger.error(f"Merged pseudo-rewrite file not found: {path}")
        return []

    rows: List[Dict] = []
    dataset_name = path.stem.replace("_pseudo_rewrites_merged", "")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num} in {path.name}: {e}")
                continue
            if not isinstance(raw, dict):
                continue

            expl = raw.get("explanation") or {}
            if not isinstance(expl, dict):
                expl = {}

            normalized = {
                "id": raw.get("id", ""),
                "image_path": raw.get("image_path", ""),
                "original_text": raw.get("original_text", ""),
                "target_group": raw.get("target_group") or expl.get("target_group") or "null",
                "visual_evidence": raw.get("visual_evidence") or expl.get("visual_evidence") or "null",
                "explanation": raw.get("implicit_meaning") or expl.get("implicit_meaning") or "",
                "pseudo_rewrite": raw.get("pseudo_rewrite", ""),
                "toxicity_drop": raw.get("toxicity_drop"),
                "passed_stage1_filters": _extract_passed_stage1_filters(raw),
                "reject_reason": raw.get("reject_reason", ""),
                "sta_score": raw.get("sta_score"),
                "bertscore": raw.get("bertscore"),
                "original_toxicity": raw.get("original_toxicity"),
                "rewrite_toxicity": raw.get("rewrite_toxicity"),
                "parse_error": _extract_parse_error(raw),
                "dataset": raw.get("dataset", dataset_name),
            }
            rows.append(normalized)

    logger.info(f"Loaded {len(rows)} rows from {path}")
    return rows


def filter_rows_for_stage2(
    examples: List[Dict],
    min_toxicity_drop: float = 0.0,
    require_passed_stage1_filters: bool = True,
) -> tuple[List[Dict], Dict[str, int]]:
    """
    Apply quality gates to Stage 1 pseudo-rewrite rows; return (kept, drop_counts).

    Gates applied in order:
      1. parse_error == False
      2. passed_stage1_filters == True (skipped when require_passed_stage1_filters=False)
      3. toxicity_drop is not None
      4. toxicity_drop > min_toxicity_drop  (strictly positive by default)
    """
    kept: List[Dict] = []
    dropped = {
        "failed_stage1_filters": 0,
        "parse_error": 0,
        "missing_toxicity_drop": 0,
        "non_positive_toxicity_drop": 0,
    }

    for ex in examples:
        if bool(ex.get("parse_error", False)):
            dropped["parse_error"] += 1
            continue

        if require_passed_stage1_filters and not bool(ex.get("passed_stage1_filters", False)):
            dropped["failed_stage1_filters"] += 1
            continue

        tox_drop = _safe_float(ex.get("toxicity_drop"))
        if tox_drop is None:
            dropped["missing_toxicity_drop"] += 1
            continue

        if tox_drop <= min_toxicity_drop:
            dropped["non_positive_toxicity_drop"] += 1
            continue

        kept.append(ex)

    logger.info(
        "Stage 2 filtering complete: kept %d / %d rows (dropped: %s)",
        len(kept),
        len(examples),
        dropped,
    )
    return kept, dropped


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


def normalized_char_similarity(a: str, b: str) -> float:
    """Character-level SequenceMatcher ratio after whitespace/punct normalisation; used as a copy-risk score."""
    import difflib

    left = _normalize_for_compare(a)
    right = _normalize_for_compare(b)
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def sanitize_rewrite_text(text: str) -> str:
    """
    Strip LLM generation artifacts and normalise whitespace in a pseudo-rewrite.

    Handles: Llama [/INST] suffixes, markdown code fences, leading "Rewrite:"
    labels, Twitter mentions/hashtags, URLs, bullet/replacement Unicode chars,
    repeated punctuation, and surrounding quotes.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    # Llama-style instruction-following models echo the [/INST] delimiter;
    # keep only the text generated after it.
    if "[/INST]" in cleaned:
        cleaned = cleaned.split("[/INST]")[-1].strip()

    # Strip markdown code fences occasionally emitted by chat-tuned models.
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
    # Collapse runs of the same punctuation mark (e.g. "!!!" \u2192 "!").
    cleaned = re.sub(r"([!?.,;:])\1{2,}", r"\1", cleaned)
    cleaned = re.sub(r"\s+([!?.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("\"'` ").strip()
    return cleaned


def is_valid_rewrite_text(rewrite: str, original_text: str) -> tuple[bool, str]:
    """
    Post-sanitisation validity check; return (True, "") or (False, rejection_reason).

    Rejection criteria (applied in order):
      empty       — zero-length after sanitisation.
      url/mention/hashtag — residual social-media tokens after sanitisation.
      too_short   — fewer than 2 whitespace-delimited tokens.
      too_long    — exceeds 280 characters (Twitter meme register heuristic).
      low_diversity — for sequences >= 8 tokens, type-token ratio < 0.35
                      indicates degenerate repetitive output.
      repetition  — any single token accounts for > 45% of the token sequence.
      symbol_heavy — non-alphanumeric, non-space character ratio > 35%;
                     flags garbled or encoding-corrupted outputs.
      no_edit     — normalised form is identical to the original meme text.
    """
    text = (rewrite or "").strip()
    if not text:
        return False, "empty"

    if URL_RE.search(text) or DOMAIN_RE.search(text):
        return False, "url"
    if MENTION_RE.search(text):
        return False, "mention"
    if HASHTAG_RE.search(text):
        return False, "hashtag"

    tokens = text.split()
    if len(tokens) < 2:
        return False, "too_short"

    if len(text) > 280:
        return False, "too_long"

    lower_tokens = [t.lower() for t in tokens]
    if len(tokens) >= 8:
        unique_ratio = len(set(lower_tokens)) / max(len(lower_tokens), 1)
        if unique_ratio < 0.35:
            return False, "low_diversity"
        counts = {}
        for tok in lower_tokens:
            counts[tok] = counts.get(tok, 0) + 1
        if (max(counts.values()) / len(lower_tokens)) > 0.45:
            return False, "repetition"

    non_alnum_ratio = sum(
        1 for c in text if (not c.isalnum() and not c.isspace())
    ) / max(len(text), 1)
    if non_alnum_ratio > 0.35:
        return False, "symbol_heavy"

    if _normalize_for_compare(text) == _normalize_for_compare(original_text):
        return False, "no_edit"

    return True, ""


def create_input_format(
    target_group: str,
    visual_evidence: str,
    implicit_meaning: str,
    meme_text: str
) -> str:
    """
    Create prefixed BART encoder input (full conditioning format):

        [T: <target_group>] [V: <visual_evidence>] [M: <implicit_meaning>] | <meme_text>

    Null fields are rendered as the literal string "null".
    This matches MemeRewriter.format_input(condition="full") in models/rewriter.py.

    Args:
        target_group:     e.g. "race_ethnicity" or "null"
        visual_evidence:  short visual cue from explainer or "null"
        implicit_meaning: one-sentence implicit meaning from LLaVA, or ""
        meme_text:        original meme text

    Returns:
        Formatted input string ready for BART tokenisation
    """
    tg = target_group or "null"
    ve = visual_evidence or "null"
    im = implicit_meaning or "null"
    return f"[T: {tg}] [V: {ve}] [M: {im}] | {meme_text}"


def build_training_data(examples: List[Dict]) -> List[Dict]:
    """
    Convert quality-filtered Stage 1 rows into BART training records.

    Each record carries: id, image_path, input_text (full-condition format),
    target_text, raw conditioning fields (original_text, target_group,
    visual_evidence, implicit_meaning), Stage 1 quality scores, and
    a source_similarity score for downstream copy-risk filtering.
    """
    training_data = []
    rejected_quality = 0
    rejected_reasons: Dict[str, int] = {}

    for example in examples:
        original_text = example.get("original_text", "")
        target_group = example.get("target_group") or "null"
        visual_evidence = example.get("visual_evidence") or "null"
        implicit_meaning = example.get("explanation", "") or "null"
        pseudo_rewrite = sanitize_rewrite_text(example.get("pseudo_rewrite", ""))
        dataset = example.get("dataset", "unknown")
        example_id = example.get("id", "")

        if not original_text or not pseudo_rewrite:
            logger.warning(f"Skipping example {example_id}: missing text or rewrite")
            continue

        is_valid, reason = is_valid_rewrite_text(pseudo_rewrite, original_text)
        if not is_valid:
            rejected_quality += 1
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
            continue

        input_text = create_input_format(target_group, visual_evidence, implicit_meaning, original_text)
        condition = target_group

        training_record = {
            "id": example_id,
            "image_path": example.get("image_path", ""),
            # Pre-formatted full-condition encoder input; used as-is when
            # condition="full" in train_stage2.py.  Raw fields below let
            # train_stage2.py reformat the input for ablation conditions
            # (target_only, visual_only, none).
            "input_text": input_text,
            "target_text": pseudo_rewrite,
            "original_text": original_text,
            "target_group": target_group,
            "visual_evidence": visual_evidence,
            "implicit_meaning": implicit_meaning,
            "stage1_passed_filters": bool(example.get("passed_stage1_filters", False)),
            "stage1_sta_score": example.get("sta_score"),
            "stage1_bertscore": example.get("bertscore"),
            "stage1_toxicity_drop": example.get("toxicity_drop"),
            "source_similarity": round(normalized_char_similarity(original_text, pseudo_rewrite), 4),
            "condition": condition,
            "dataset": dataset
        }

        training_data.append(training_record)

    if rejected_quality > 0:
        logger.info(
            f"Dropped {rejected_quality} examples due to rewrite text quality constraints: "
            f"{rejected_reasons}"
        )
    logger.info(f"Built {len(training_data)} training records")
    return training_data


def split_train_val(
    examples: List[Dict],
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Randomly partition examples into (train, val) at the given ratio.

    This is the legacy fallback used when no dedicated val JSONL is provided;
    note that it draws val from the same distribution as train, which is
    unsuitable for final model selection.
    """
    random.seed(seed)
    random.shuffle(examples)

    split_idx = int(len(examples) * train_ratio)
    train = examples[:split_idx]
    val = examples[split_idx:]

    logger.info(f"Split into train ({len(train)}) and val ({len(val)})")
    return train, val


def write_jsonl(examples: List[Dict], output_path: str):
    """Serialise a list of dicts to a newline-delimited JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    logger.info(f"Wrote {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Stage 2 dataset from merged Stage 1 pseudo-rewrites"
    )
    parser.add_argument(
        "--stage1_dir",
        type=str,
        required=True,
        help="Directory containing train_pseudo_rewrites_merged.jsonl"
    )
    parser.add_argument(
        "--rewrites_path",
        type=str,
        default="",
        help="Path to merged rewrites JSONL (default: <stage1_dir>/train_pseudo_rewrites_merged.jsonl)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for train.jsonl and val.jsonl"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: use make_debug_dataset(), skip parse/toxicity filtering"
    )
    parser.add_argument(
        "--hf_cache",
        type=str,
        default=None,
        help="Deprecated/ignored. Kept for backward compatibility with existing job scripts."
    )
    parser.add_argument(
        "--min_toxicity_drop",
        type=float,
        default=0.0,
        help="Minimum required toxicity drop; rows must satisfy toxicity_drop > this value"
    )
    parser.add_argument(
        "--allow_failed_stage1_filters",
        action="store_true",
        help=(
            "Include rows whose best rewrite did not pass Stage 1 filters. "
            "Default is false because failed rows add weak detox supervision."
        )
    )
    parser.add_argument(
        "--max_source_similarity",
        type=float,
        default=1.0,
        help=(
            "Optional normalized source/rewrite similarity ceiling after sanitization. "
            "Set below 1.0 to drop near-copy pseudo rewrites, e.g. 0.92."
        )
    )
    parser.add_argument(
        "--val_rewrites_path",
        type=str,
        default="",
        help=(
            "Path to val_pseudo_rewrites_merged.jsonl (Stage 1 output on unified_val.csv). "
            "When provided, val.jsonl is built from this file using the same quality filters "
            "as train, and the internal train_ratio random split is skipped."
        )
    )
    parser.add_argument(
        "--test_rewrites_path",
        type=str,
        default="",
        help=(
            "Path to test_pseudo_rewrites_merged.jsonl (Stage 1 output on unified_test.csv). "
            "When provided, test.jsonl is built from this file using the same quality filters."
        )
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help=(
            "Train split ratio used only when --val_rewrites_path is NOT provided "
            "(legacy fallback: internal random split of train data)."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # set_seeds imports torch; fall back gracefully in CPU-only preprocessing
    # environments where torch may not be installed.
    try:
        set_seeds(args.seed)
    except ModuleNotFoundError as e:
        logger.warning("set_seeds fallback (missing dependency): %s", e)
        random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Build Stage 2 Dataset")
    print(f"  Stage 1 dir: {args.stage1_dir}")
    default_merged = str(Path(args.stage1_dir) / "train_pseudo_rewrites_merged.jsonl")
    rewrites_path = args.rewrites_path.strip() or default_merged
    val_rewrites_path = args.val_rewrites_path.strip()
    test_rewrites_path = args.test_rewrites_path.strip()
    print(f"  Train rewrites: {rewrites_path}")
    print(f"  Val rewrites:   {val_rewrites_path or '(internal split from train)'}")
    print(f"  Test rewrites:  {test_rewrites_path or '(none)'}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Require Stage 1 pass: {not args.allow_failed_stage1_filters}")
    print(f"  Min tox drop > {args.min_toxicity_drop}")
    print(f"  Max source similarity: {args.max_source_similarity}")
    if not val_rewrites_path:
        print(f"  Train ratio: {args.train_ratio} (legacy internal split)")
    print(f"  Seed:        {args.seed}")
    print(f"  Debug:       {args.debug}")
    print(f"{'='*60}\n")

    if args.hf_cache:
        logger.info("--hf_cache is ignored by build_stage2_dataset.py (kept for compatibility).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        logger.warning("DEBUG MODE ENABLED: Using small debug dataset, skipping parse/toxicity filtering")
        examples = make_debug_dataset()
        total_loaded = len(examples)
        dropped_counts = {
            "failed_stage1_filters": 0,
            "parse_error": 0,
            "missing_toxicity_drop": 0,
            "non_positive_toxicity_drop": 0,
        }
    else:
        examples = load_merged_rewrites(rewrites_path)
        if not examples:
            logger.error("No examples loaded from merged rewrites file")
            return 1

        total_loaded = len(examples)
        examples, dropped_counts = filter_rows_for_stage2(
            examples,
            min_toxicity_drop=args.min_toxicity_drop,
            require_passed_stage1_filters=not args.allow_failed_stage1_filters,
        )
        if not examples:
            logger.error("All examples filtered out by parse_error/toxicity_drop conditions")
            return 1

    training_data = build_training_data(examples)
    if not training_data:
        logger.error("Failed to build training data")
        return 1

    if args.max_source_similarity < 1.0:
        before = len(training_data)
        training_data = [
            ex for ex in training_data
            if _safe_float(ex.get("source_similarity")) is not None
            and float(ex["source_similarity"]) <= args.max_source_similarity
        ]
        logger.info(
            "Dropped %d records with source_similarity > %.3f",
            before - len(training_data),
            args.max_source_similarity,
        )
        if not training_data:
            logger.error("All examples filtered out by --max_source_similarity")
            return 1

    val_data: List[Dict] = []
    if val_rewrites_path:
        val_raw = load_merged_rewrites(val_rewrites_path)
        if not val_raw:
            logger.error("No examples loaded from val rewrites file: %s", val_rewrites_path)
            return 1
        val_raw, val_dropped = filter_rows_for_stage2(
            val_raw,
            min_toxicity_drop=args.min_toxicity_drop,
            require_passed_stage1_filters=not args.allow_failed_stage1_filters,
        )
        val_data = build_training_data(val_raw)
        if args.max_source_similarity < 1.0:
            val_data = [
                ex for ex in val_data
                if _safe_float(ex.get("source_similarity")) is not None
                and float(ex["source_similarity"]) <= args.max_source_similarity
            ]
        train_data = training_data
        logger.info("Using dedicated val set: %d examples", len(val_data))
    else:
        # Legacy fallback: random internal split of train data
        train_data, val_data = split_train_val(
            training_data,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        logger.warning(
            "No --val_rewrites_path provided; falling back to internal %g/%g random split. "
            "This means val data is a subset of train — use --val_rewrites_path for proper evaluation.",
            args.train_ratio,
            1 - args.train_ratio,
        )

    test_data: List[Dict] = []
    if test_rewrites_path:
        test_raw = load_merged_rewrites(test_rewrites_path)
        if not test_raw:
            logger.error("No examples loaded from test rewrites file: %s", test_rewrites_path)
            return 1
        test_raw, _ = filter_rows_for_stage2(
            test_raw,
            min_toxicity_drop=args.min_toxicity_drop,
            require_passed_stage1_filters=not args.allow_failed_stage1_filters,
        )
        test_data = build_training_data(test_raw)
        if args.max_source_similarity < 1.0:
            test_data = [
                ex for ex in test_data
                if _safe_float(ex.get("source_similarity")) is not None
                and float(ex["source_similarity"]) <= args.max_source_similarity
            ]
        logger.info("Built test set: %d examples", len(test_data))

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    write_jsonl(train_data, str(train_path))
    write_jsonl(val_data, str(val_path))

    if test_data:
        test_path = output_dir / "test.jsonl"
        write_jsonl(test_data, str(test_path))

    from collections import Counter
    dataset_counts     = Counter(e.get("dataset",      "unknown") for e in training_data)
    target_group_counts = Counter(e.get("target_group", "null")   for e in training_data)
    visual_evidence_presence = Counter(
        "present" if (e.get("visual_evidence") and e.get("visual_evidence") != "null") else "null"
        for e in training_data
    )

    # Persist full build configuration and data distribution for reproducibility
    # and reporting; does not affect model training.
    stats = {
        "build_config": {
            "stage1_dir": args.stage1_dir,
            "rewrites_path": rewrites_path,
            "require_parse_error_false": True,
            "require_passed_stage1_filters": not args.allow_failed_stage1_filters,
            "min_toxicity_drop_exclusive": args.min_toxicity_drop,
            "max_source_similarity_inclusive": args.max_source_similarity,
            "train_val_ratio": args.train_ratio,
            "seed": args.seed,
            "debug": args.debug,
        },
        "counts": {
            "total_loaded_from_merged_rewrites": total_loaded,
            "dropped_failed_stage1_filters": dropped_counts["failed_stage1_filters"],
            "dropped_parse_error": dropped_counts["parse_error"],
            "dropped_missing_toxicity_drop": dropped_counts["missing_toxicity_drop"],
            "dropped_non_positive_toxicity_drop": dropped_counts["non_positive_toxicity_drop"],
            "after_parse_and_toxicity_filter": len(examples),
            "after_rewrite_text_quality_filter": len(training_data),
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "val_source": val_rewrites_path if val_rewrites_path else "internal_split",
            "test_source": test_rewrites_path if test_rewrites_path else "none",
        },
        "dataset_source_distribution": dict(dataset_counts.most_common()),
        "target_group_distribution": dict(target_group_counts.most_common()),
        "visual_evidence_presence": dict(visual_evidence_presence.most_common()),
    }
    stats_path = output_dir / "dataset_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Dataset statistics saved to {stats_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("STAGE 2 DATASET BUILD SUMMARY")
    print("=" * 80)
    print(f"Total examples loaded from merged rewrites:  {total_loaded}")
    if not args.debug:
        print(f"Dropped failed Stage 1 filter rows:          {dropped_counts['failed_stage1_filters']}")
        print(f"Dropped parse_error rows:                    {dropped_counts['parse_error']}")
        print(f"Dropped rows with missing toxicity_drop:     {dropped_counts['missing_toxicity_drop']}")
        print(f"Dropped rows with toxicity_drop <= {args.min_toxicity_drop}: {dropped_counts['non_positive_toxicity_drop']}")
        print(f"After parse/toxicity filter:                 {len(examples)}")
    print(f"After rewrite text quality filter:           {len(training_data)}")
    print(f"  Training examples:                {len(train_data)}")
    print(f"  Validation examples:              {len(val_data)}  (source: {'dedicated val set' if val_rewrites_path else 'internal split'})")
    if test_data:
        print(f"  Test examples:                    {len(test_data)}")
    print(f"\nBreakdown by source dataset:")
    for ds, count in sorted(dataset_counts.items()):
        print(f"  {ds:<20} {count:>6} examples")
    print(f"\nBreakdown by target group:")
    for tg, count in target_group_counts.most_common():
        print(f"  {tg:<25} {count:>6} examples")
    print(f"\nVisual evidence field:")
    for label, count in visual_evidence_presence.most_common():
        print(f"  {label:<25} {count:>6} examples")
    print(f"\nOutput directory: {output_dir}")
    print(f"  train.jsonl:            {len(train_data)} examples")
    print(f"  val.jsonl:              {len(val_data)} examples")
    if test_data:
        print(f"  test.jsonl:             {len(test_data)} examples")
    print(f"  dataset_statistics.json saved")
    if training_data:
        ex = training_data[0]
        print(f"\nExample input format:")
        print(f"  INPUT:  {ex['input_text'][:80]}...")
        print(f"  TARGET: {ex['target_text'][:80]}...")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
