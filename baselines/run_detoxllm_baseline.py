"""
DetoxLLM baseline: text-only detoxification without images.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker

from utils.debug import is_debug_mode, setup_debug_mode, DEBUG_CONFIG

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


class DetoxLLMBaseline:
    """DetoxLLM baseline for hateful meme text rewriting (text-only)."""

    def __init__(
        self,
        model_name: str = "UBC-NLP/DetoxLLM-7B",
        hf_cache: str = None,
        load_in_4bit: bool = False,
        debug: bool = False
    ):
        """
        Initialize DetoxLLM baseline.

        Args:
            model_name: Model identifier
            hf_cache: Hugging Face cache directory
            load_in_4bit: Load model in 4-bit quantization
            debug: Debug mode (skip actual model loading)
        """
        self.model_name = model_name
        self.hf_cache = hf_cache
        self.load_in_4bit = load_in_4bit
        self.debug = debug

        if hf_cache:
            import os
            os.environ["HF_HOME"] = hf_cache

        # Only load model if not in debug mode
        if not debug:
            logger.info(f"Loading {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            device_map = None
            quantization_config = None

            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    cache_dir=hf_cache,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=hf_cache,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                if torch.cuda.is_available():
                    self.model = self.model.cuda()

            self.model.eval()
            self.device = next(self.model.parameters()).device
            logger.info(f"Model loaded on {self.device}")
        else:
            self.tokenizer = None
            self.model = None
            self.device = "cpu"
            logger.info("DEBUG mode: skipping model loading")

    def detoxify(self, text: str) -> str:
        """
        Detoxify a single text using DetoxLLM.

        Args:
            text: Original meme text

        Returns:
            Detoxified text
        """
        if self.debug:
            # Debug mode: return dummy rewrite
            return f"safe: {text}"

        try:
            # Prepare input prompt
            prompt = f"detoxify: {text}\noutput: "

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract generated portion (after "output: ")
            if "output:" in generated_text:
                rewrite = generated_text.split("output:")[-1].strip()
            else:
                rewrite = generated_text.replace(prompt, "").strip()

            # Clean up common artifacts
            if not rewrite or len(rewrite.split()) < 2:
                rewrite = text  # Fallback to original if generation failed

            return rewrite

        except Exception as e:
            logger.error(f"Error in detoxification: {e}")
            return text

    def process_batch(self, texts: List[str], batch_size: int = 4) -> List[Dict]:
        """
        Process a batch of texts.

        Args:
            texts: List of original texts
            batch_size: Batch size (for consistency)

        Returns:
            List of results with keys: idx, original_text, rewrite
        """
        results = []

        for idx, text in enumerate(tqdm(texts, desc="Detoxifying")):
            rewrite = self.detoxify(text)
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
    parser = argparse.ArgumentParser(description="DetoxLLM baseline for hateful meme text rewriting")
    parser.add_argument("--stage1_outputs", type=Path, required=True, help="Stage 1 outputs JSONL")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--hf_cache", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        setup_debug_mode()

    setup_logging(debug=args.debug)
    logger.info("Starting DetoxLLM baseline")

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

    # Extract texts
    texts = [item["text"] for item in stage1_outputs]
    logger.info(f"Processing {len(texts)} texts")

    # Initialize baseline
    baseline = DetoxLLMBaseline(
        hf_cache=args.hf_cache,
        load_in_4bit=args.load_in_4bit,
        debug=args.debug
    )

    # Process batch with CO2 tracking
    def run_detoxification():
        return baseline.process_batch(texts, batch_size=args.batch_size)

    tracker = EmissionsTracker(log_level="warning")
    tracker.start()
    results = run_detoxification()
    co2_emissions = tracker.stop()
    logger.info(f"CO2 emissions: {co2_emissions:.4f}g")

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_file = args.output_dir / "detoxllm_text_only.jsonl"

    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Total: {len(results)} texts processed")


if __name__ == "__main__":
    main()
