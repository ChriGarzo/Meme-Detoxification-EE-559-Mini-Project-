"""
Stage 2: BART-based rewriting with conditional control.

Generates final rewrites for all examples using a fine-tuned BART model
with support for multiple conditioning strategies.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from codecarbon import EmissionsTracker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.rewriter import MemeRewriter

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_explanation_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load examples from Stage 1 explanation JSONL file."""
    examples = []
    if not os.path.exists(jsonl_path):
        logger.error(f"File not found: {jsonl_path}")
        return examples

    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(data)
    except Exception as e:
        logger.error(f"Error loading JSONL: {e}")

    logger.info(f"Loaded {len(examples)} examples from {jsonl_path}")
    return examples


def write_jsonl_batch(data: List[Dict], output_path: str) -> None:
    """Append batch of examples to JSONL file."""
    with open(output_path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def build_condition_prompt(
    original_text: str,
    explanation: Dict[str, str],
    condition: str
) -> str:
    """
    Build BART encoder input string based on ablation condition.

    Format mirrors MemeRewriter.format_input (models/rewriter.py):
      full:        [T: <target_group>] [V: <visual_evidence>] [M: <implicit_meaning>] | {text}
      target_only: [T: <target_group>] [V: null] [M: null] | {text}
      visual_only: [T: null] [V: <visual_evidence>] [M: null] | {text}
      none:        [T: null] [V: null] [M: null] | {text}

    Null fields are rendered as the literal string "null" (not Python None).
    """
    explanation_str = explanation or {}
    tg = explanation_str.get("target_group") or "null"
    ve = explanation_str.get("visual_evidence") or explanation_str.get("attack_type") or "null"
    im = explanation_str.get("implicit_meaning") or "null"

    if condition == "full":
        prefix = f"[T: {tg}] [V: {ve}] [M: {im}]"
    elif condition == "target_only":
        prefix = f"[T: {tg}] [V: null] [M: null]"
    elif condition in {"visual_only", "attack_only"}:
        prefix = f"[T: null] [V: {ve}] [M: null]"
    else:  # 'none'
        prefix = "[T: null] [V: null] [M: null]"

    return f"{prefix} | {original_text}"


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Generate BART-based rewrites")
    parser.add_argument(
        "--stage1_output_dir",
        type=str,
        required=True,
        help="Directory containing per-dataset Stage 1 JSONL outputs (e.g. /scratch/hmr_stage1_output)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory of the fine-tuned BART checkpoint (e.g. /scratch/hmr_stage2_phase2_full_checkpoint)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["full", "target_only", "visual_only", "none"],
        default="full",
        help="Ablation conditioning strategy (full | target_only | visual_only | none)"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSONL")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
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
            logging.FileHandler(os.path.join(args.output_dir, "stage2.log")),
            logging.StreamHandler()
        ]
    )

    logger.info(f"Starting Stage 2 with condition={args.condition}, debug={args.debug}")
    logger.info(f"Arguments: {vars(args)}")

    # Load Stage 1 outputs — collect all per-dataset explanation JSONL files
    stage1_dir = Path(args.stage1_output_dir)
    all_jsonl = sorted(stage1_dir.rglob("*_explanations.jsonl"))
    examples = []
    for jsonl_file in all_jsonl:
        examples.extend(load_explanation_jsonl(str(jsonl_file)))
    logger.info(f"Loaded {len(examples)} total examples from {len(all_jsonl)} datasets")
    if args.debug:
        examples = examples[:16]
    logger.info(f"Processing {len(examples)} examples")

    # Initialize BART model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    if args.debug:
        # Use bart-base for debug mode
        model_name = "facebook/bart-base"
        logger.info("Using bart-base for debug mode")
    else:
        model_name = args.checkpoint_dir

    try:
        rewriter = MemeRewriter(
            model_name=model_name,
            cache_dir=args.hf_cache,
            device=device,
            num_beams=args.num_beams,
            debug=args.debug,
        )
        rewriter.load_model()
    except Exception as e:
        logger.error(f"Failed to load BART model from {model_name}: {e}")
        logger.info("Attempting fallback to facebook/bart-large")
        rewriter = MemeRewriter(
            model_name="facebook/bart-large",
            cache_dir=args.hf_cache,
            device=device,
            num_beams=args.num_beams,
            debug=args.debug,
        )
        rewriter.load_model()

    # Prepare output path
    output_path = os.path.join(args.output_dir, f"stage2_rewrites_{args.condition}.jsonl")

    # Process examples
    batch_texts = []
    batch_prompts = []
    batch_original = []
    batch_explanations = []
    batch_records = []
    total_processed = 0

    tracker = EmissionsTracker(log_level="warning", output_dir=args.output_dir, output_file="emissions.csv")
    tracker.start()

    try:
        with tqdm.tqdm(total=len(examples), desc="Generating rewrites") as pbar:
            for idx, example in enumerate(examples):
                example_id = example.get("id")
                image_path = example.get("image_path")
                original_text = example.get("original_text", "")
                explanation = example.get("explanation", {})

                # Build conditioning prompt
                prompt = build_condition_prompt(original_text, explanation, args.condition)
                batch_texts.append(prompt)
                batch_prompts.append(prompt)
                batch_original.append(original_text)
                batch_explanations.append(explanation)

                batch_records.append({
                    "id": example_id,
                    "image_path": image_path,
                    "original_text": original_text,
                    "explanation": explanation,
                    "condition": args.condition
                })

                # Process batch
                if len(batch_texts) >= args.batch_size or (idx == len(examples) - 1 and batch_texts):
                    try:
                        # prompts are already fully formatted by build_condition_prompt;
                        # use generate_from_formatted to avoid double-prefixing
                        rewrites = rewriter.generate_from_formatted(batch_prompts, max_length=128)

                        for i, rewrite in enumerate(rewrites):
                            batch_records[len(batch_records) - len(batch_texts) + i]["rewrite"] = rewrite

                    except Exception as e:
                        logger.error(f"Error generating rewrites: {e}")
                        for i in range(len(batch_texts)):
                            batch_records[len(batch_records) - len(batch_texts) + i]["rewrite"] = ""

                    # Write batch
                    write_jsonl_batch(batch_records, output_path)
                    total_processed += len(batch_records)
                    logger.info(f"Processed batch of {len(batch_records)} examples")

                    batch_texts = []
                    batch_prompts = []
                    batch_original = []
                    batch_explanations = []
                    batch_records = []

                pbar.update(1)

        logger.info(f"\n=== Stage 2 Summary ===")
        logger.info(f"Total examples processed: {total_processed}")
        logger.info(f"Condition: {args.condition}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Num beams: {args.num_beams}")
        logger.info(f"Output JSONL: {output_path}")

    finally:
        emissions = tracker.stop()
        if emissions is not None:
            logger.info(f"Carbon emissions: {emissions:.6f} kg CO2")
        else:
            logger.warning("CO2 emissions could not be measured")


if __name__ == "__main__":
    main()
