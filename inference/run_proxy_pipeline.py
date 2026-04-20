"""
Proxy Pipeline: VLM-free inference using CLIP + ExplanationProxy + BART.

Provides efficient end-to-end inference without LLaVA by using pre-computed
or learned visual representations through a proxy network.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from codecarbon import EmissionsTracker
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoModelForSeq2SeqLM, AutoTokenizer

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


class ExplanationProxy(torch.nn.Module):
    """
    Proxy network that predicts full explanation embedding from CLIP features.

    Maps CLIP visual features to BART encoder hidden states for explanation.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 768):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_features: (batch_size, input_dim) CLIP visual features

        Returns:
            explanation_embedding: (batch_size, output_dim) predicted explanation embedding
        """
        x = self.fc1(clip_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CLIPFeatureExtractor:
    """Extract visual features from images using CLIP."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", cache_dir: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, image_path: str) -> torch.Tensor:
        """
        Extract CLIP visual features from an image.

        Args:
            image_path: Path to image file

        Returns:
            Visual features (512,) for ViT-B/32
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return zero tensor if image loading fails
            return torch.zeros(512, dtype=torch.float32, device=self.device)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.squeeze(0)

    def extract_batch(self, image_paths: List[str]) -> torch.Tensor:
        """
        Extract CLIP features for a batch of images.

        Args:
            image_paths: List of paths to images

        Returns:
            Visual features (batch_size, 512)
        """
        features_list = []
        for path in image_paths:
            features = self.extract(path)
            features_list.append(features)

        return torch.stack(features_list)


class ProxyPipeline:
    """End-to-end inference pipeline using proxy network."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        proxy_checkpoint: Optional[str] = None,
        bart_model_name: str = "facebook/bart-base",
        proxy_hidden_dim: int = 256,
        cache_dir: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = device
        self.cache_dir = cache_dir

        # Load CLIP
        logger.info(f"Loading CLIP from {clip_model_name}")
        self.clip_extractor = CLIPFeatureExtractor(clip_model_name, cache_dir)

        # Load proxy network
        logger.info(f"Loading ExplanationProxy (hidden_dim={proxy_hidden_dim})")
        self.proxy = ExplanationProxy(input_dim=512, hidden_dim=proxy_hidden_dim, output_dim=768)

        if proxy_checkpoint and os.path.exists(proxy_checkpoint):
            logger.info(f"Loading proxy weights from {proxy_checkpoint}")
            state = torch.load(proxy_checkpoint, map_location=device)
            self.proxy.load_state_dict(state)
        else:
            logger.warning(f"Proxy checkpoint not found: {proxy_checkpoint}, using random initialization")

        self.proxy.to(device)
        self.proxy.eval()

        # Load BART (decoder only, using encoder from proxy)
        logger.info(f"Loading BART from {bart_model_name}")
        self.bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_model_name, cache_dir=cache_dir)
        self.bart_tokenizer = AutoTokenizer.from_pretrained(bart_model_name, cache_dir=cache_dir)
        self.bart_model.to(device)
        self.bart_model.eval()

    def rewrite(
        self,
        image_path: str,
        original_text: str,
        max_length: int = 128,
        num_beams: int = 5
    ) -> str:
        """
        Generate rewrite for a single example.

        Args:
            image_path: Path to meme image
            original_text: Original hateful text
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search

        Returns:
            Rewritten text
        """
        # Extract CLIP features
        image_features = self.clip_extractor.extract(image_path)  # (512,)

        # Predict encoder hidden state via proxy
        with torch.no_grad():
            encoder_hidden = self.proxy(image_features.unsqueeze(0))  # (1, 768)

            # Tokenize input text
            inputs = self.bart_tokenizer(
                original_text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)

            # Generate with proxy-predicted context
            outputs = self.bart_model.generate(
                input_ids,
                encoder_outputs=(encoder_hidden,),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=2.0
            )

            rewrite = self.bart_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return rewrite

    def rewrite_batch(
        self,
        image_paths: List[str],
        texts: List[str],
        max_length: int = 128,
        num_beams: int = 5
    ) -> List[str]:
        """
        Generate rewrites for a batch of examples.

        Args:
            image_paths: List of paths to meme images
            texts: List of original texts
            max_length: Maximum length of generated text
            num_beams: Number of beams for beam search

        Returns:
            List of rewritten texts
        """
        assert len(image_paths) == len(texts), "Mismatch between images and texts"

        batch_size = len(image_paths)
        rewrites = []

        # Extract CLIP features for batch
        image_features = self.clip_extractor.extract_batch(image_paths)  # (batch_size, 512)

        # Predict encoder hidden states via proxy
        with torch.no_grad():
            encoder_hidden = self.proxy(image_features)  # (batch_size, 768)

            # Tokenize all texts
            inputs = self.bart_tokenizer(
                texts,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Generate for batch
            outputs = self.bart_model.generate(
                input_ids,
                attention_mask=attention_mask,
                encoder_outputs=(encoder_hidden,),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=2.0
            )

            for output in outputs:
                rewrite = self.bart_tokenizer.decode(output, skip_special_tokens=True)
                rewrites.append(rewrite)

        return rewrites


def load_manifest(manifest_path: str, max_examples: Optional[int] = None) -> pd.DataFrame:
    """Load manifest CSV with image data."""
    df = pd.read_csv(manifest_path)
    if max_examples:
        df = df.head(max_examples)
    logger.info(f"Loaded {len(df)} examples from manifest")
    return df


def write_jsonl_batch(data: List[Dict], output_path: str) -> None:
    """Append batch of examples to JSONL file."""
    with open(output_path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Proxy Pipeline: VLM-free meme rewriting")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'training')")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--bart_checkpoint", type=str, default="facebook/bart-base", help="BART model or checkpoint")
    parser.add_argument("--proxy_checkpoint", type=str, default=None, help="Path to trained proxy checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSONL")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--proxy_hidden_dim", type=int, default=256, help="Proxy network hidden dimension")
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
            logging.FileHandler(os.path.join(args.output_dir, "proxy_pipeline.log")),
            logging.StreamHandler()
        ]
    )

    logger.info(f"Starting Proxy Pipeline with dataset={args.dataset}, debug={args.debug}")
    logger.info(f"Arguments: {vars(args)}")

    # Load manifest
    max_examples = 16 if args.debug else None
    manifest_df = load_manifest(args.manifest, max_examples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Determine BART model for debug mode
    bart_model = "facebook/bart-base" if args.debug else args.bart_checkpoint

    # Initialize pipeline
    try:
        pipeline = ProxyPipeline(
            clip_model_name="openai/clip-vit-base-patch32",
            proxy_checkpoint=args.proxy_checkpoint,
            bart_model_name=bart_model,
            proxy_hidden_dim=args.proxy_hidden_dim,
            cache_dir=args.hf_cache,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Prepare output path
    output_path = os.path.join(args.output_dir, f"{args.dataset}_proxy_rewrites.jsonl")

    # Process examples
    batch_image_paths = []
    batch_texts = []
    batch_records = []
    total_processed = 0

    tracker = EmissionsTracker(log_level="warning", output_dir=args.output_dir, output_file="emissions.csv")
    tracker.start()

    try:
        with tqdm.tqdm(total=len(manifest_df), desc="Generating rewrites") as pbar:
            for idx, row in manifest_df.iterrows():
                example_id = row.get("id")
                image_path = os.path.join(args.images_dir, row.get("image_path"))
                original_text = row.get("text", "")

                batch_image_paths.append(image_path)
                batch_texts.append(original_text)

                batch_records.append({
                    "id": example_id,
                    "image_path": row.get("image_path"),
                    "original_text": original_text,
                    "explanation": {},
                    "condition": "proxy",
                    "rewrite": ""
                })

                # Process batch
                if len(batch_texts) >= args.batch_size or (idx == len(manifest_df) - 1 and batch_texts):
                    try:
                        rewrites = pipeline.rewrite_batch(
                            batch_image_paths,
                            batch_texts,
                            max_length=128,
                            num_beams=args.num_beams
                        )

                        for i, rewrite in enumerate(rewrites):
                            batch_records[len(batch_records) - len(batch_texts) + i]["rewrite"] = rewrite

                    except Exception as e:
                        logger.error(f"Error generating batch: {e}")
                        for i in range(len(batch_texts)):
                            batch_records[len(batch_records) - len(batch_texts) + i]["rewrite"] = ""

                    # Write batch
                    write_jsonl_batch(batch_records, output_path)
                    total_processed += len(batch_records)
                    logger.info(f"Processed batch of {len(batch_records)} examples")

                    batch_image_paths = []
                    batch_texts = []
                    batch_records = []

                pbar.update(1)

        logger.info(f"\n=== Proxy Pipeline Summary ===")
        logger.info(f"Total examples processed: {total_processed}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Num beams: {args.num_beams}")
        logger.info(f"Proxy hidden dim: {args.proxy_hidden_dim}")
        logger.info(f"Output JSONL: {output_path}")

    finally:
        emissions = tracker.stop()
        if emissions is not None:
            logger.info(f"Carbon emissions: {emissions:.6f} kg CO2")
        else:
            logger.warning("CO2 emissions could not be measured")


if __name__ == "__main__":
    main()
