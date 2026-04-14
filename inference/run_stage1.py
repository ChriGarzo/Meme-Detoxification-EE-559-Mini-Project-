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
from utils.bertscore_utils import compute_bertscore_batch

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


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Generate explanations and pseudo-rewrites")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'training')")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest CSV from Stage 0 (output of filter_meme_images.py)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSONL files")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load LLaVA in 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
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
    total_examples = 0
    kept_rewrites = 0
    total_pseudo_rewrites = 0

    tracker = EmissionsTracker(log_level="warning")
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
                    explanation = explainer.explain(image_path, original_text)
                except json.JSONDecodeError:
                    json_parse_failures += 1
                    logger.warning(f"JSON parse failure for example {example_id}")
                    explanation = {
                        "visual_content": "",
                        "hateful_elements": "",
                        "target_group": ""
                    }

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
                    try:
                        raw_rewrite = explainer.generate_rewrite(
                            image_path, original_text, explanation
                        )
                    except Exception as e:
                        logger.warning(f"Rewrite generation failed for {example_id}: {e}")
                        raw_rewrite = None

                    pseudo_rewrites = [raw_rewrite] if raw_rewrite else []

                    # Filter by quality
                    for rewrite in pseudo_rewrites:
                        # Compute STA score (1 = toxic, 0 = non-toxic)
                        sta_scores = compute_sta_score([rewrite], sta_model, sta_tokenizer, device)
                        sta_score = 1.0 - sta_scores[0]  # Invert: 1 = non-toxic, 0 = toxic

                        # Compute BERTScore
                        bertscore = compute_bertscore_batch([original_text], [rewrite])[0]

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
                        f"json_failures={json_parse_failures}"
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
        logger.info(f"Total pseudo-rewrites generated: {total_pseudo_rewrites}")
        logger.info(f"Pseudo-rewrites kept (passed filters): {kept_rewrites}")
        logger.info(f"Keep rate: {keep_rate:.2f}%")
        logger.info(f"Explanations JSONL: {explanations_path}")
        logger.info(f"Pseudo-rewrites JSONL: {pseudo_rewrites_path}")

    finally:
        emissions = tracker.stop()
        logger.info(f"Carbon emissions: {emissions:.6f} kg CO2")


if __name__ == "__main__":
    main()
