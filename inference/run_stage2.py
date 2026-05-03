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


def load_stage2_eval_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load Stage 2 train/val JSONL rows and convert them to inference examples."""
    examples = []
    if not os.path.exists(jsonl_path):
        logger.error(f"Stage 2 eval JSONL not found: {jsonl_path}")
        return examples

    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON in %s line %d: %s", jsonl_path, line_num, exc)
                continue
            explanation = row.get("explanation")
            if not isinstance(explanation, dict):
                explanation = {
                    "target_group": row.get("target_group"),
                    "visual_evidence": row.get("visual_evidence"),
                    "implicit_meaning": row.get("implicit_meaning"),
                }
            examples.append({
                "id": row.get("id"),
                "image_path": row.get("image_path", ""),
                "original_text": row.get("original_text") or row.get("text") or "",
                "target_text": row.get("target_text", ""),
                "explanation": explanation,
                "dataset": row.get("dataset"),
            })

    logger.info(f"Loaded {len(examples)} Stage 2 eval examples from {jsonl_path}")
    return examples


def write_jsonl_batch(data: List[Dict], output_path: str) -> None:
    """Append batch of examples to JSONL file."""
    with open(output_path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def discover_explanation_files(stage1_dir: Path) -> List[Path]:
    """
    Discover Stage 1 explanation JSONL files.

    Preference order avoids duplicate loading:
      1) merged files: *_explanations_merged.jsonl
      2) direct files: *_explanations.jsonl
      3) sharded files: *_explanations_shard*.jsonl
    """
    merged = sorted(stage1_dir.rglob("*_explanations_merged.jsonl"))
    if merged:
        return merged

    direct = sorted(stage1_dir.rglob("*_explanations.jsonl"))
    if direct:
        return direct

    sharded = sorted(stage1_dir.rglob("*_explanations_shard*.jsonl"))
    if sharded:
        return sharded

    # Backward/forward-compat fallback if naming changes again.
    return sorted(stage1_dir.rglob("*explanations*.jsonl"))


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
    ve = explanation_str.get("visual_evidence") or "null"
    im = explanation_str.get("implicit_meaning") or "null"

    if condition == "full":
        prefix = f"[T: {tg}] [V: {ve}] [M: {im}]"
    elif condition == "target_only":
        prefix = f"[T: {tg}] [V: null] [M: null]"
    elif condition == "visual_only":
        prefix = f"[T: null] [V: {ve}] [M: null]"
    else:  # 'none'
        prefix = "[T: null] [V: null] [M: null]"
    return f"{prefix} | {original_text}"


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Generate BART-based rewrites")
    parser.add_argument(
        "--stage1_output_dir",
        type=str,
        default=None,
        help="Directory containing per-dataset Stage 1 JSONL outputs (e.g. /scratch/hmr_stage1_output)"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
        help="Optional Stage 2 val/test JSONL. If set, inference runs only on this file.",
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
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum generated sequence length")
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="Prevent repeated n-grams inside generated rewrites",
    )
    parser.add_argument(
        "--encoder_no_repeat_ngram_size",
        type=int,
        default=3,
        help="Prevent copying n-grams from the input prompt/original text",
    )
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

    # Load evaluation examples. Prefer an explicit Stage 2 val/test JSONL so
    # model comparisons are made on a held-out split rather than all Stage 1 rows.
    if args.input_jsonl:
        examples = load_stage2_eval_jsonl(args.input_jsonl)
    else:
        if not args.stage1_output_dir:
            logger.error("Either --input_jsonl or --stage1_output_dir is required")
            sys.exit(1)
        stage1_dir = Path(args.stage1_output_dir)
        all_jsonl = discover_explanation_files(stage1_dir)
        logger.info(f"Discovered {len(all_jsonl)} explanation JSONL files under {stage1_dir}")
        if all_jsonl:
            preview = ", ".join(str(p.name) for p in all_jsonl[:5])
            if len(all_jsonl) > 5:
                preview += ", ..."
            logger.info(f"Example input files: {preview}")

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
    if os.path.exists(output_path):
        logger.info(f"Removing existing rewrite output before regeneration: {output_path}")
        os.remove(output_path)

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
                    "target_text": example.get("target_text", ""),
                    "explanation": explanation,
                    "condition": args.condition
                })

                # Process batch
                if len(batch_texts) >= args.batch_size or (idx == len(examples) - 1 and batch_texts):
                    try:
                        # prompts are already fully formatted by build_condition_prompt;
                        # use generate_from_formatted to avoid double-prefixing
                        rewrites = rewriter.generate_from_formatted(
                            batch_prompts,
                            max_length=args.max_length,
                            num_beams=args.num_beams,
                            no_repeat_ngram_size=args.no_repeat_ngram_size,
                            encoder_no_repeat_ngram_size=args.encoder_no_repeat_ngram_size,
                        )

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
