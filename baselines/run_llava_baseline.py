"""
LLaVA baseline for hateful meme detoxification.

NOTE: This script is unmaintained and not part of the evaluation pipeline.
The evaluation pipeline (scripts/run_evaluate_all_job.sh) does not invoke this
file; the LLaVA teacher row in the results table is derived from the pre-computed
target_text labels in the Stage 2 JSONL, not from re-running this script.

This file has known API incompatibilities with the current MemeExplainer
(models/explainer.py): the processor calling convention and the explanation
field schema (target_group/visual_evidence/implicit_meaning) differ from what
is hard-coded here. Do not run this script without first updating it to match
the current MemeExplainer API.

Two rewriting modes (concept only):
  end_to_end        — LLaVA receives (image, original text) and is prompted to
                      produce a non-hateful rewrite directly, without explicit
                      Stage 1 structured analysis.
  structured_prompt — Stage 1 MemeExplainer produces a structured explanation
                      which is appended as context before asking LLaVA to rewrite.
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
from utils.debug import is_debug_mode, DEBUG_CONFIG

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure root logger level; DEBUG when debug=True."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


class LLaVABaseline:
    """LLaVA-based meme text rewriter; wraps MemeExplainer for model access."""

    def __init__(self, hf_cache: str = None, load_in_4bit: bool = False):
        """
        Initialise LLaVA baseline.

        Args:
            hf_cache: Hugging Face model cache directory.
            load_in_4bit: Enable bitsandbytes 4-bit quantisation to reduce GPU memory.
        """
        self.hf_cache = hf_cache
        self.load_in_4bit = load_in_4bit

        if hf_cache:
            import os
            os.environ["HF_HOME"] = hf_cache

        # MemeExplainer holds the loaded LLaVA model; we reuse it directly
        # to avoid loading the weights twice across both rewriting modes.
        self.explainer = MemeExplainer(
            cache_dir=hf_cache,
            load_in_4bit=load_in_4bit
        )

    def rewrite_end_to_end(self, image_path: str, text: str) -> str:
        """
        Prompt LLaVA directly with (image, text) to produce a non-hateful rewrite.

        No structured Stage 1 analysis is performed; the model must identify
        hateful content and neutralise it in a single generation step.
        Falls back to the original text on generation error.
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
            # Llama-style models echo the [/INST] tag; extract only the generated suffix.
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
        Run Stage 1 MemeExplainer then prompt LLaVA to rewrite with the structured context.

        Falls back to the original text on any error.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            explanation = self.explainer.explain(image, text)

            # Assemble available explanation fields into a freeform context string.
            context_parts = []
            if explanation.get("description"):
                context_parts.append(f"Description: {explanation['description']}")
            if explanation.get("visual_evidence"):
                context_parts.append(f"Visual Evidence: {explanation['visual_evidence']}")
            if explanation.get("offensive_keywords"):
                context_parts.append(f"Offensive Keywords: {', '.join(explanation['offensive_keywords'])}")
            if explanation.get("rationale"):
                context_parts.append(f"Rationale: {explanation['rationale']}")

            context = "\n".join(context_parts)

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
        Rewrite a list of (image_path, text) pairs; return records with idx, original_text, rewrite.

        batch_size is accepted for interface consistency but generation is
        sequential (LLaVA inference is effectively batch_size=1).
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
    """Parse Stage 1 output JSONL; return a list of record dicts."""
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

    setup_logging(debug=args.debug)
    logger.info(f"Starting LLaVA baseline (mode={args.mode})")

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    stage1_outputs = load_stage1_outputs(args.stage1_outputs)
    logger.info(f"Loaded {len(stage1_outputs)} Stage 1 outputs")

    if args.debug:
        stage1_outputs = stage1_outputs[:DEBUG_CONFIG["max_samples"]]
        logger.info(f"DEBUG mode: processing only {len(stage1_outputs)} examples")

    # Resolve image files; skip entries whose image cannot be found on disk.
    image_paths = []
    texts = []
    for item in stage1_outputs:
        idx = item["id"]
        img_path = args.images_dir / f"{idx}.jpg"
        if not img_path.exists():
            img_path = args.images_dir / f"{idx}.png"
        if img_path.exists():
            image_paths.append(str(img_path))
            texts.append(item["text"])

    logger.info(f"Processing {len(image_paths)} examples")

    baseline = LLaVABaseline(
        hf_cache=args.hf_cache,
        load_in_4bit=args.load_in_4bit
    )

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
