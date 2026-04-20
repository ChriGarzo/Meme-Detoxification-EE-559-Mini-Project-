"""
LLaVA baseline: end-to-end rewriting and structured prompt rewriting.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal
import random

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from codecarbon import EmissionsTracker

from models.explainer import MemeExplainer
from utils.debug import is_debug_mode, setup_debug_mode, DEBUG_CONFIG

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


class LLaVABaseline:
    """LLaVA baseline for hateful meme text rewriting."""

    def __init__(self, hf_cache: str = None, load_in_4bit: bool = False):
        """
        Initialize LLaVA baseline.

        Args:
            hf_cache: Hugging Face cache directory
            load_in_4bit: Load model in 4-bit quantization
        """
        self.hf_cache = hf_cache
        self.load_in_4bit = load_in_4bit

        if hf_cache:
            import os
            os.environ["HF_HOME"] = hf_cache

        # Initialize explainer (which uses LLaVA internally)
        self.explainer = MemeExplainer(
            hf_cache=hf_cache,
            load_in_4bit=load_in_4bit
        )

    def rewrite_end_to_end(self, image_path: str, text: str) -> str:
        """
        End-to-end rewriting: give LLaVA the image+text and ask it to rewrite directly.

        Args:
            image_path: Path to meme image
            text: Original meme text

        Returns:
            Rewritten text
        """
        prompt = f"""[INST] <image>
The text in this meme is: '{text}'
This meme may contain hateful content. Rewrite only the meme text to be non-hateful while:
- Preserving the approximate length and informal register of the original
- Keeping the same topic but removing any hateful framing
- Producing natural language that could plausibly appear on a meme

Respond with ONLY the rewritten text. No quotes, no explanation, no preamble.
[/INST]"""

        try:
            image = Image.open(image_path).convert("RGB")
            # Use the explainer's LLaVA model to generate response
            response = self.explainer.processor.decode(
                self.explainer.model.generate(
                    self.explainer.processor(prompt, image, return_tensors="pt").to(self.explainer.device),
                    max_new_tokens=100
                )[0],
                skip_special_tokens=True
            )
            # Extract just the rewritten text part
            if "[/INST]" in response:
                rewrite = response.split("[/INST]")[-1].strip()
            else:
                rewrite = response.strip()
            return rewrite
        except Exception as e:
            logger.error(f"Error in end-to-end rewriting: {e}")
            return text

    def rewrite_structured_prompt(
        self,
        image_path: str,
        text: str
    ) -> str:
        """
        Structured prompt rewriting: use Stage 1 explanation flow, then ask LLaVA to rewrite.

        Args:
            image_path: Path to meme image
            text: Original meme text

        Returns:
            Rewritten text
        """
        try:
            # Get Stage 1 explanation
            image = Image.open(image_path).convert("RGB")
            explanation = self.explainer.explain(image, text)

            # Build context from explanation
            context_parts = []
            if explanation.get("description"):
                context_parts.append(f"Description: {explanation['description']}")
            if explanation.get("attack_type"):
                context_parts.append(f"Attack Type: {explanation['attack_type']}")
            if explanation.get("offensive_keywords"):
                context_parts.append(f"Offensive Keywords: {', '.join(explanation['offensive_keywords'])}")
            if explanation.get("rationale"):
                context_parts.append(f"Rationale: {explanation['rationale']}")

            context = "\n".join(context_parts)

            # Now ask LLaVA to rewrite with this context
            prompt = f"""[INST] <image>
The text in this meme is: '{text}'

Context about why this is hateful:
{context}

Based on this understanding, rewrite only the meme text to be non-hateful while:
- Preserving the approximate length and informal register of the original
- Keeping the same topic but removing any hateful framing
- Producing natural language that could plausibly appear on a meme

Respond with ONLY the rewritten text. No quotes, no explanation, no preamble.
[/INST]"""

            response = self.explainer.processor.decode(
                self.explainer.model.generate(
                    self.explainer.processor(prompt, image, return_tensors="pt").to(self.explainer.device),
                    max_new_tokens=100
                )[0],
                skip_special_tokens=True
            )

            if "[/INST]" in response:
                rewrite = response.split("[/INST]")[-1].strip()
            else:
                rewrite = response.strip()

            return rewrite
        except Exception as e:
            logger.error(f"Error in structured prompt rewriting: {e}")
            return text

    def process_batch(
        self,
        image_paths: List[str],
        texts: List[str],
        mode: Literal["end_to_end", "structured_prompt"],
        batch_size: int = 1
    ) -> List[Dict]:
        """
        Process a batch of examples.

        Args:
            image_paths: List of image paths
            texts: List of original texts
            mode: Rewriting mode
            batch_size: Batch size (for consistency)

        Returns:
            List of results with keys: idx, original_text, rewrite
        """
        results = []
        rewrite_fn = (
            self.rewrite_end_to_end if mode == "end_to_end"
            else self.rewrite_structured_prompt
        )

        for idx, (img_path, text) in enumerate(tqdm(zip(image_paths, texts), total=len(texts))):
            rewrite = rewrite_fn(img_path, text)
            results.append({
                "idx": idx,
                "original_text": text,
                "rewrite": rewrite
            })

        return results


def load_stage1_outputs(stage1_file: Path) -> List[Dict]:
    """Load Stage 1 outputs from JSONL."""
    outputs = []
    if not stage1_file.exists():
        logger.warning(f"Stage 1 file not found: {stage1_file}")
        return outputs

    with open(stage1_file) as f:
        for line in f:
            outputs.append(json.loads(line))

    return outputs


def main():
    parser = argparse.ArgumentParser(description="LLaVA baseline for hateful meme text rewriting")
    parser.add_argument(
        "--mode",
        choices=["end_to_end", "structured_prompt"],
        required=True,
        help="Rewriting mode"
    )
    parser.add_argument("--stage1_outputs", type=Path, required=True, help="Stage 1 outputs JSONL")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory with meme images")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--hf_cache", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        setup_debug_mode()

    setup_logging(debug=args.debug)
    logger.info(f"Starting LLaVA baseline (mode={args.mode})")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # Load Stage 1 outputs
    stage1_outputs = load_stage1_outputs(args.stage1_outputs)
    logger.info(f"Loaded {len(stage1_outputs)} Stage 1 outputs")

    if args.debug:
        stage1_outputs = stage1_outputs[:DEBUG_CONFIG["max_examples"]]
        logger.info(f"DEBUG mode: processing only {len(stage1_outputs)} examples")

    # Extract image paths and texts
    image_paths = []
    texts = []
    for item in stage1_outputs:
        idx = item["idx"]
        img_path = args.images_dir / f"{idx}.jpg"
        if not img_path.exists():
            img_path = args.images_dir / f"{idx}.png"
        if img_path.exists():
            image_paths.append(str(img_path))
            texts.append(item["text"])

    logger.info(f"Processing {len(image_paths)} examples")

    # Initialize baseline
    baseline = LLaVABaseline(
        hf_cache=args.hf_cache,
        load_in_4bit=args.load_in_4bit
    )

    # Process batch with CO2 tracking
    def run_rewriting():
        return baseline.process_batch(
            image_paths,
            texts,
            mode=args.mode,
            batch_size=args.batch_size
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(log_level="warning", output_dir=str(args.output_dir), output_file="emissions.csv")
    tracker.start()
    results = run_rewriting()
    co2_emissions = tracker.stop()
    if co2_emissions is not None:
        logger.info(f"CO2 emissions: {co2_emissions:.4f}g")
    else:
        logger.warning("CO2 emissions could not be measured")

    system_name = f"llava_{args.mode}"
    output_file = args.output_dir / f"{system_name}.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Total: {len(results)} examples processed")


if __name__ == "__main__":
    main()
