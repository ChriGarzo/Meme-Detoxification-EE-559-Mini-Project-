"""
Build BART training dataset from Stage 1 LLaVA outputs.

This script:
1. Reads Stage 1 JSONL outputs (explanations + pseudo_rewrites)
2. Combines all datasets into training/validation splits (90/10)
3. Filters by BERTScore > 0.4
4. Creates prefixed input format: [T: ...] [A: ...] [M: ...] </s> {text}
5. Outputs train.jsonl and val.jsonl with standardized fields
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.debug import is_debug_mode, set_seeds


logger = logging.getLogger(__name__)


def make_debug_dataset() -> List[Dict]:
    """Create a small debug dataset for testing."""
    debug_examples = [
        {
            "id": "debug_001",
            "original_text": "that's disgusting",
            "target_group": "race_ethnicity",
            "attack_type": "contempt",
            "explanation": "This uses contempt language targeting a racial group",
            "pseudo_rewrite": "I disagree with that viewpoint",
            "bert_score": 0.85
        },
        {
            "id": "debug_002",
            "original_text": "stupid people",
            "target_group": "disability",
            "attack_type": "mocking",
            "explanation": "This mocks disabled individuals",
            "pseudo_rewrite": "people with different perspectives",
            "bert_score": 0.72
        }
    ]
    return debug_examples


def load_stage1_outputs(stage1_dir: str) -> List[Dict]:
    """
    Load all Stage 1 pseudo-rewrite JSONL outputs from a tree of per-dataset
    sub-directories.

    run_stage1.py writes files named:
        {stage1_dir}/{dataset}/{dataset}_pseudo_rewrites.jsonl

    Each line contains:
    {
        "id": "...",
        "image_path": "...",
        "original_text": "...",
        "explanation": {
            "target_group": "...",
            "attack_type": "...",
            "implicit_meaning": "..."
        },
        "pseudo_rewrite": "...",
        "sta_score": 0.xx,
        "bertscore": 0.xx       ← note: "bertscore" not "bert_score"
    }

    This function normalises the field names so downstream code uses
    a consistent schema.
    """
    stage1_dir = Path(stage1_dir)
    if not stage1_dir.is_dir():
        logger.error(f"Stage 1 directory not found: {stage1_dir}")
        return []

    examples = []
    # Collect *_pseudo_rewrites.jsonl from any depth
    jsonl_files = sorted(stage1_dir.rglob("*_pseudo_rewrites.jsonl"))

    if not jsonl_files:
        logger.warning(
            f"No *_pseudo_rewrites.jsonl files found under {stage1_dir}. "
            "Make sure Stage 1 has completed for all datasets."
        )
        return []

    for jsonl_path in tqdm(jsonl_files, desc="Loading Stage 1 pseudo-rewrites"):
        logger.info(f"Loading {jsonl_path}")
        # Derive dataset name from the file stem: "{dataset}_pseudo_rewrites"
        dataset_name = jsonl_path.stem.replace("_pseudo_rewrites", "")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse line {line_num} in {jsonl_path.name}: {e}"
                    )
                    continue

                # Normalise schema: flatten explanation sub-fields to top level
                expl = raw.get("explanation") or {}
                example = {
                    "id": raw.get("id", ""),
                    "image_path": raw.get("image_path", ""),
                    "original_text": raw.get("original_text", ""),
                    "target_group": expl.get("target_group") or "null",
                    "attack_type": expl.get("attack_type") or "null",
                    "explanation": expl.get("implicit_meaning") or "",
                    "pseudo_rewrite": raw.get("pseudo_rewrite", ""),
                    # Accept both "bertscore" (stage1 output) and "bert_score"
                    "bert_score": raw.get("bertscore", raw.get("bert_score", 0.0)),
                    "dataset": raw.get("dataset", dataset_name),
                }
                examples.append(example)

    logger.info(f"Loaded {len(examples)} pseudo-rewrite examples from Stage 1")
    return examples


def filter_by_bert_score(examples: List[Dict], min_score: float = 0.4) -> List[Dict]:
    """Filter examples by minimum BERTScore."""
    filtered = [
        ex for ex in examples
        if ex.get("bert_score", 0) >= min_score
    ]
    removed = len(examples) - len(filtered)
    logger.info(f"Filtered by BERTScore > {min_score}: removed {removed} examples, kept {len(filtered)}")
    return filtered


def create_input_format(
    target_group: str,
    attack_type: str,
    implicit_meaning: str,
    meme_text: str
) -> str:
    """
    Create prefixed BART encoder input (full conditioning format):

        [T: <target_group>] [A: <attack_type>] [M: <implicit_meaning>] </s> <meme_text>

    Null fields are rendered as the literal string "null".
    This matches MemeRewriter.format_input(condition="full") in models/rewriter.py.

    Args:
        target_group:     e.g. "race_ethnicity" or "null"
        attack_type:      e.g. "contempt" or "null"
        implicit_meaning: one-sentence implicit meaning from LLaVA, or ""
        meme_text:        original meme text

    Returns:
        Formatted input string ready for BART tokenisation
    """
    tg = target_group or "null"
    at = attack_type or "null"
    im = implicit_meaning or "null"
    return f"[T: {tg}] [A: {at}] [M: {im}] </s> {meme_text}"


def build_training_data(examples: List[Dict]) -> List[Dict]:
    """
    Build training data from loaded examples.

    Creates records with fields: id, input_text, target_text, condition, dataset
    """
    training_data = []

    for example in examples:
        original_text = example.get("original_text", "")
        target_group = example.get("target_group") or "null"
        attack_type = example.get("attack_type") or "null"
        implicit_meaning = example.get("explanation", "") or "null"
        pseudo_rewrite = example.get("pseudo_rewrite", "")
        dataset = example.get("dataset", "unknown")
        example_id = example.get("id", "")

        if not original_text or not pseudo_rewrite:
            logger.warning(f"Skipping example {example_id}: missing text or rewrite")
            continue
        input_text = create_input_format(target_group, attack_type, implicit_meaning, original_text)
        condition = f"{target_group}_{attack_type}"

        training_record = {
            "id": example_id,
            # Pre-formatted full-condition input (used as-is for condition=full)
            "input_text": input_text,
            "target_text": pseudo_rewrite,
            # Raw fields stored separately so train_stage2_phase2.py can
            # reformat the input for conditions other than 'full'
            "original_text": original_text,
            "target_group": target_group,
            "attack_type": attack_type,
            "implicit_meaning": implicit_meaning,
            "condition": condition,
            "dataset": dataset
        }

        training_data.append(training_record)

    logger.info(f"Built {len(training_data)} training records")
    return training_data


def split_train_val(
    examples: List[Dict],
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split examples into training and validation sets."""
    random.seed(seed)
    random.shuffle(examples)

    split_idx = int(len(examples) * train_ratio)
    train = examples[:split_idx]
    val = examples[split_idx:]

    logger.info(f"Split into train ({len(train)}) and val ({len(val)})")
    return train, val


def write_jsonl(examples: List[Dict], output_path: str):
    """Write examples to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    logger.info(f"Wrote {len(examples)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build BART training dataset from Stage 1 LLaVA outputs"
    )
    parser.add_argument(
        "--stage1_dir",
        type=str,
        required=True,
        help="Directory containing Stage 1 JSONL outputs"
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
        help="Debug mode: use make_debug_dataset(), skip BERTScore filtering"
    )
    parser.add_argument(
        "--hf_cache",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set seeds for reproducibility
    set_seeds(42)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.debug:
        logger.warning("DEBUG MODE ENABLED: Using small debug dataset, skipping BERTScore filtering")
        examples = make_debug_dataset()
    else:
        examples = load_stage1_outputs(args.stage1_dir)
        if not examples:
            logger.error("No examples loaded from Stage 1")
            return 1

        # Filter by BERTScore
        examples = filter_by_bert_score(examples, min_score=0.4)
        if not examples:
            logger.error("All examples filtered out by BERTScore threshold")
            return 1

    # Build training data
    training_data = build_training_data(examples)
    if not training_data:
        logger.error("Failed to build training data")
        return 1

    # Split into train/val
    train_data, val_data = split_train_val(training_data, train_ratio=0.9)

    # Write outputs
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    write_jsonl(train_data, str(train_path))
    write_jsonl(val_data, str(val_path))

    # Print summary
    print("\n" + "=" * 80)
    print("STAGE 2 DATASET BUILD SUMMARY")
    print("=" * 80)
    print(f"Total examples loaded: {len(examples)}")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Output directory: {output_dir}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
