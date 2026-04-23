"""
Stage 1: LLaVA explanations + pseudo-rewrites with quality filtering.

Generates explanations for hateful meme text and creates pseudo-rewrites
using pattern-based methods. Applies quality filters based on toxicity
and semantic similarity.
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from codecarbon import EmissionsTracker
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def load_sta_model(model_name: str = "s-nlp/roberta_toxicity_classifier", device: str = "cuda"):
    """Load STA (toxicity) model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def compute_sta_score(texts: List[str], model: Any, tokenizer: Any, device: str = "cuda") -> List[float]:
    """
    Compute toxicity scores using STA model.
    Returns the toxicity probability (class 1).
    """
    if not texts:
        return []

    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        toxicity_scores = probs[:, 1].cpu().numpy()

    return toxicity_scores.tolist()


def load_existing_ids(jsonl_path: str) -> set:
    """Load IDs of already-processed examples from JSONL file."""
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
    """Append batch of examples to JSONL file."""
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


def sanitize_generated_rewrite(text: str) -> str:
    """
    Deterministically sanitize LLaVA rewrite output into plain sentence text.

    This strips wrappers and removes metadata artifacts that should never
    be learned as rewrite targets.
    """
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


def has_invalid_rewrite_format(rewrite: str, original_text: str) -> tuple[bool, str]:
    """
    Reject rewrites with URLs/artifacts, extreme repetition, or no real edit.
    """
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

    return False, ""


def ensure_hateful_explanation_non_null(explanation: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """Ensure hateful explanations always have non-null conditioning fields."""
    normalized = dict(explanation) if isinstance(explanation, dict) else {}
    changed = False

    target_group = normalized.get("target_group")
    if not isinstance(target_group, str) or not target_group.strip() or target_group.strip().lower() in {
        "null", "none", "n/a", "na", "unknown"
    }:
        normalized["target_group"] = "other"
        changed = True

    attack_type = normalized.get("attack_type")
    if not isinstance(attack_type, str) or not attack_type.strip() or attack_type.strip().lower() in {
        "null", "none", "n/a", "na", "unknown"
    }:
        normalized["attack_type"] = "contempt"
        changed = True

    implicit_meaning = normalized.get("implicit_meaning")
    if (
        not isinstance(implicit_meaning, str)
        or not implicit_meaning.strip()
        or implicit_meaning.strip().lower() in {"null", "none", "n/a", "na", "unknown"}
    ):
        normalized["implicit_meaning"] = (
            "The meme communicates a hateful or derogatory framing toward a target group."
        )
        changed = True

    return normalized, changed


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Generate explanations and pseudo-rewrites")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'training')")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest CSV from Stage 0 (output of filter_meme_images.py)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSONL files")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load LLaVA in 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--hateful_only", action="store_true", help="Only process examples where hateful=1 (skip non-hateful memes)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process max 16 examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "stage1.log")),
            logging.StreamHandler()
        ]
    )

    print(f"\n{'='*60}")
    print(f"  Stage 1: LLaVA Explanations + Pseudo-rewrites")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Images:     {args.images_dir}")
    print(f"  Manifest:   {args.manifest_path}")
    print(f"  Output:     {args.output_dir}")
    print(f"  HF cache:   {args.hf_cache}")
    print(f"  4-bit quant:{args.load_in_4bit}")
    print(f"  Debug:      {args.debug}")
    print(f"{'='*60}\n")
    logger.info(f"Starting Stage 1 with dataset={args.dataset}, debug={args.debug}")
    logger.info(f"Arguments: {vars(args)}")

    # Load manifest
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
    logger.info(f"Processing {len(manifest_df)} examples")

    # Initialize models
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

    sta_model, sta_tokenizer = load_sta_model(device=device)

    # Load BERTScorer once — reusing it per example avoids reloading the model every call
    bertscore_scorer = create_bertscore_scorer(device=device)

    # Prepare output paths
    explanations_path = os.path.join(args.output_dir, f"{args.dataset}_explanations.jsonl")
    pseudo_rewrites_path = os.path.join(args.output_dir, f"{args.dataset}_pseudo_rewrites.jsonl")

    # Load already-processed IDs for resume
    processed_explanation_ids = load_existing_ids(explanations_path)
    processed_rewrite_ids = load_existing_ids(pseudo_rewrites_path)

    logger.info(f"Already processed explanations: {len(processed_explanation_ids)}")
    logger.info(f"Already processed rewrites: {len(processed_rewrite_ids)}")

    # Process examples
    explanations_batch = []
    rewrites_batch = []
    json_parse_failures = 0
    forced_non_null_explanations = 0
    total_examples = 0
    kept_rewrites = 0
    total_pseudo_rewrites = 0
    invalid_rewrite_format = 0
    invalid_rewrite_reason_counts: Dict[str, int] = {}
    rewrite_generation_failures = 0

    tracker = EmissionsTracker(log_level="warning", output_dir=args.output_dir, output_file="emissions.csv")
    tracker.start()

    try:
        with tqdm.tqdm(total=len(manifest_df), desc="Processing examples") as pbar:
            for idx, row in manifest_df.iterrows():
                example_id = row.get("id")
                # unified manifest has absolute image_path; fallback joins images_dir
                raw_img = str(row.get("image_path", ""))
                if os.path.isabs(raw_img) and os.path.exists(raw_img):
                    image_path = raw_img
                else:
                    image_path = os.path.join(args.images_dir, raw_img)
                # prefer annotated text; fall back to OCR text from Stage 0
                original_text = str(row.get("text", "") or row.get("ocr_text", ""))
                is_hateful = bool(row.get("hateful", False))

                # Check if already processed
                if example_id in processed_explanation_ids:
                    pbar.update(1)
                    continue

                total_examples += 1

                # Generate explanation
                try:
                    explanation = explainer.explain(
                        image_path,
                        original_text,
                        force_hateful=is_hateful,
                        max_retries=2,
                    )
                except Exception as e:
                    logger.warning(f"Explanation generation failed for {example_id}: {e}")
                    if is_hateful:
                        explanation = {
                            "target_group": "other",
                            "attack_type": "contempt",
                            "implicit_meaning": (
                                "The meme communicates a hateful or derogatory framing toward a target group."
                            ),
                            "error": str(e),
                        }
                    else:
                        explanation = {
                            "target_group": None,
                            "attack_type": None,
                            "implicit_meaning": None,
                            "error": str(e),
                        }

                if explanation.get("parse_error"):
                    json_parse_failures += 1

                if is_hateful:
                    explanation, was_forced = ensure_hateful_explanation_non_null(explanation)
                    if was_forced:
                        forced_non_null_explanations += 1

                explanation_record = {
                    "id": example_id,
                    "image_path": row.get("image_path"),
                    "original_text": original_text,
                    "explanation": explanation,
                    "is_hateful": is_hateful
                }
                explanations_batch.append(explanation_record)
                processed_explanation_ids.add(example_id)

                # Generate pseudo-rewrite via LLaVA if the meme is hateful
                if is_hateful and example_id not in processed_rewrite_ids:
                    total_pseudo_rewrites += 1
                    cleaned_rewrite = None
                    max_rewrite_attempts = 3

                    for attempt_idx in range(max_rewrite_attempts):
                        try:
                            raw_rewrite = explainer.generate_rewrite(
                                image_path, original_text, explanation
                            )
                        except Exception as e:
                            rewrite_generation_failures += 1
                            logger.warning(
                                f"Rewrite generation failed for {example_id} "
                                f"(attempt {attempt_idx + 1}/{max_rewrite_attempts}): {e}"
                            )
                            continue

                        candidate = sanitize_generated_rewrite(raw_rewrite)
                        is_invalid, reason = has_invalid_rewrite_format(candidate, original_text)
                        if is_invalid:
                            invalid_rewrite_format += 1
                            invalid_rewrite_reason_counts[reason] = (
                                invalid_rewrite_reason_counts.get(reason, 0) + 1
                            )
                            logger.info(
                                f"Rejected rewrite for {example_id} "
                                f"(attempt {attempt_idx + 1}/{max_rewrite_attempts}, reason={reason})"
                            )
                            continue

                        cleaned_rewrite = candidate
                        break

                    pseudo_rewrites = [cleaned_rewrite] if cleaned_rewrite else []

                    # Filter by quality
                    for rewrite in pseudo_rewrites:
                        # Compute STA score (1 = toxic, 0 = non-toxic)
                        sta_scores = compute_sta_score([rewrite], sta_model, sta_tokenizer, device)
                        sta_score = 1.0 - sta_scores[0]  # Invert: 1 = non-toxic, 0 = toxic

                        # Compute BERTScore (scorer pre-loaded — no model reload per call)
                        bertscore = compute_bertscore_batch([original_text], [rewrite], scorer=bertscore_scorer)[0]

                        # Apply quality filter
                        if sta_score > 0.6 and bertscore > 0.4:
                            kept_rewrites += 1
                            rewrite_record = {
                                "id": example_id,
                                "image_path": row.get("image_path"),
                                "original_text": original_text,
                                "explanation": explanation,
                                "pseudo_rewrite": rewrite,
                                "sta_score": float(sta_score),
                                "bertscore": float(bertscore)
                            }
                            rewrites_batch.append(rewrite_record)
                            processed_rewrite_ids.add(example_id)

                pbar.update(1)

                # Print running stats every 50 examples
                if total_examples > 0 and total_examples % 50 == 0:
                    keep_rate = 100 * kept_rewrites / max(total_pseudo_rewrites, 1)
                    logger.info(
                        f"[{total_examples}/{len(manifest_df)}] "
                        f"explanations={total_examples} | "
                        f"rewrites_kept={kept_rewrites}/{total_pseudo_rewrites} ({keep_rate:.1f}%) | "
                        f"json_failures={json_parse_failures} | "
                        f"forced_non_null={forced_non_null_explanations} | "
                        f"rewrite_invalid={invalid_rewrite_format} | "
                        f"rewrite_failures={rewrite_generation_failures}"
                    )

                # Write batches every 100 examples
                if (total_examples % 100) == 0 and explanations_batch:
                    write_jsonl_batch(explanations_batch, explanations_path)
                    explanations_batch = []
                    logger.info(f"Wrote explanations batch at example {total_examples}")

                if rewrites_batch and (total_pseudo_rewrites % 100) == 0:
                    write_jsonl_batch(rewrites_batch, pseudo_rewrites_path)
                    rewrites_batch = []
                    logger.info(f"Wrote rewrites batch at example {total_pseudo_rewrites}")

        # Write final batches
        if explanations_batch:
            write_jsonl_batch(explanations_batch, explanations_path)
            logger.info(f"Wrote final explanations batch ({len(explanations_batch)} items)")

        if rewrites_batch:
            write_jsonl_batch(rewrites_batch, pseudo_rewrites_path)
            logger.info(f"Wrote final rewrites batch ({len(rewrites_batch)} items)")

        # Compute metrics
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
        logger.info(f"Rewrite generation failures: {rewrite_generation_failures}")
        logger.info(f"Explanations JSONL: {explanations_path}")
        logger.info(f"Pseudo-rewrites JSONL: {pseudo_rewrites_path}")

    finally:
        emissions = tracker.stop()
        if emissions is not None:
            logger.info(f"Carbon emissions: {emissions:.6f} kg CO2")
            logger.info(f"Emissions saved to: {os.path.join(args.output_dir, 'emissions.csv')}")
        else:
            logger.warning("Carbon emissions could not be measured (CodeCarbon tracking failed)")


if __name__ == "__main__":
    main()
