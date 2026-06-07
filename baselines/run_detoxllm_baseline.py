"""
DetoxLLM text-only baseline for hateful meme detoxification.

Uses UBC-NLP/DetoxLLM-7B, a 7B causal language model instruction-tuned for
text detoxification, to rewrite meme text without any visual context.  This
baseline isolates the contribution of multimodal grounding in the full pipeline:
if DetoxLLM performs comparably to our BART system, the image-based conditioning
is not adding significant signal.

Reads the Stage 2 validation JSONL produced by build_stage2_dataset.py so
evaluation is performed on the same held-out examples as all other systems.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.debug import is_debug_mode, DEBUG_CONFIG

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Configure root logger level; DEBUG when debug=True."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


class DetoxLLMBaseline:
    """Text-only detoxification baseline using UBC-NLP/DetoxLLM-7B."""

    def __init__(
        self,
        model_name: str = "UBC-NLP/DetoxLLM-7B",
        hf_cache: str = None,
        load_in_4bit: bool = False,
        debug: bool = False
    ):
        """
        Load DetoxLLM-7B in float16 (or 4-bit via bitsandbytes if load_in_4bit=True).

        Args:
            model_name: HuggingFace model identifier.
            hf_cache: HuggingFace model cache directory.
            load_in_4bit: Enable bitsandbytes NF4 quantisation for reduced GPU memory.
            debug: Skip model loading and return dummy rewrites (for CI/smoke tests).
        """
        self.model_name = model_name
        self.hf_cache = hf_cache
        self.load_in_4bit = load_in_4bit
        self.debug = debug

        if hf_cache:
            import os
            os.environ["HF_HOME"] = hf_cache

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
        Run DetoxLLM on a single text; return detoxified output.

        The prompt format "detoxify: <text>\\noutput: " follows the instruction
        template from the DetoxLLM paper. Generation falls back to the original
        text if the output is shorter than 2 tokens or an exception occurs.
        """
        if self.debug:
            return f"safe: {text}"

        try:
            prompt = f"detoxify: {text}\noutput: "

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=input_len + 100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the generated suffix after the "output:" sentinel.
            if "output:" in generated_text:
                rewrite = generated_text.split("output:")[-1].strip()
            else:
                rewrite = generated_text.replace(prompt, "").strip()

            # Fall back to the original when generation produces fewer than 2 tokens.
            if not rewrite or len(rewrite.split()) < 2:
                rewrite = text

            return rewrite

        except Exception as e:
            logger.error(f"Error in detoxification: {e}")
            return text

    def process_records(self, records: List[Dict]) -> List[Dict]:
        """
        Detoxify a list of validation records; return output records with
        id, system, original_text, rewrite, image_path.
        """
        results = []
        for rec in tqdm(records, desc="Detoxifying"):
            original = rec.get("original_text") or rec.get("text") or ""
            rewrite = self.detoxify(original) if original else original
            results.append({
                "id": rec.get("id"),
                "system": "detoxllm",
                "original_text": original,
                "rewrite": rewrite,
                "image_path": rec.get("image_path") or "",
            })
        return results


def load_validation_jsonl(path: Path) -> List[Dict]:
    """Parse the Stage 2 validation JSONL; return a list of record dicts."""
    records: List[Dict] = []
    if not path.exists():
        logger.warning(f"Validation JSONL not found: {path}")
        return records
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping invalid JSON line: {exc}")
    return records


def main():
    parser = argparse.ArgumentParser(description="DetoxLLM baseline for hateful meme text rewriting")
    parser.add_argument(
        "--validation_jsonl", type=Path, required=True,
        help="Stage 2 validation JSONL (hmr_stage2_dataset/val.jsonl)"
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--hf_cache", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        os.environ["DEBUG"] = "1"

    setup_logging(debug=args.debug)
    logger.info("Starting DetoxLLM baseline")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = load_validation_jsonl(args.validation_jsonl)
    logger.info(f"Loaded {len(records)} validation records")

    if args.debug:
        records = records[:DEBUG_CONFIG["max_samples"]]
        logger.info(f"DEBUG mode: processing only {len(records)} examples")

    baseline = DetoxLLMBaseline(
        hf_cache=args.hf_cache,
        load_in_4bit=args.load_in_4bit,
        debug=args.debug
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(log_level="warning", output_dir=str(args.output_dir), output_file="emissions.csv")
    tracker.start()
    results = baseline.process_records(records)
    co2_emissions = tracker.stop()
    if co2_emissions is not None:
        logger.info(f"CO2 emissions: {co2_emissions:.4f}g")
    else:
        logger.warning("CO2 emissions could not be measured")

    output_file = args.output_dir / "detoxllm_rewrites.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Total: {len(results)} texts processed")


if __name__ == "__main__":
    main()
