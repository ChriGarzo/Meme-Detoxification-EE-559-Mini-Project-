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
    Load all Stage 1 JSONL outputs from directory.

    Expects files matching: stage1_harmeme.jsonl, stage1_mami.jsonl, stage1_mmhs150k.jsonl

    Each line should contain:
    {
        "id": "...",
        "original_text": "...",
        "target_group": "...",
        "attack_type": "...",
        "explanation": "...",
        "pseudo_rewrite": "...",
        "bert_score": 0.xx
    }
    """
    stage1_dir = Path(stage1_dir)
    if not stage1_dir.is_dir():
        logger.error(f"Stage 1 directory not found: {stage1_dir}")
        return []

    examples = []
    jsonl_files = list(stage1_dir.glob("stage1_*.jsonl"))

    if not jsonl_files:
        logger.warning(f"No stage1_*.jsonl files found in {stage1_dir}")
        return []

    for jsonl_path in tqdm(jsonl_files, desc="Loading Stage 1 outputs"):
        logger.info(f"Loading {jsonl_path.name}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {jsonl_path.name}: {e}")
                    continue

    logger.info(f"Loaded {len(examples)} examples from Stage 1")
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


def create_input_format(target_group: str, attack_type: str, meme_text: str) -> str:
    """
    Create prefixed input format: [T: ...] [A: ...] [M: ...] </s> {text}

    Args:
        target_group: Target group label (e.g., "race_ethnicity")
        attack_type: Attack type label (e.g., "contempt")
        meme_text: Original meme text

    Returns:
        Formatted input string
    """
    return f"[T: {target_group}] [A: {attack_type}] [M: {meme_text}] </s>"


def build_training_data(examples: List[Dict]) -> List[Dict]:
    """
    Build training data from loaded examples.

    Creates records with fields: id, input_text, target_text, condition, dataset
    """
    training_data = []

    for example in examples:
        original_text = example.get("original_text", "")
        target_group = example.get("target_group", "unknown")
        attack_type = example.get("attack_type", "unknown")
        pseudo_rewrite = example.get("pseudo_rewrite", "")
        dataset = example.get("dataset", "unknown")
        example_id = example.get("id", "")

        if not original_text or not pseudo_rewrite:
            logger.warning(f"Skipping example {example_id}: missing text or rewrite")
            continue

        input_text = create_input_format(target_group, attack_type, original_text)
        condition = f"{target_group}_{attack_type}"

        training_record = {
            "id": example_id,
            "input_text": input_text,
            "target_text": pseudo_rewrite,
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
