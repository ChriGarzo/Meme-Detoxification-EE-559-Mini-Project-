"""
Stage 0: Filter meme images using OCR and CLIP.

This script filters meme images by:
1. Extracting text via EasyOCR (keep 10-300 characters)
2. Using CLIP to distinguish memes from screenshots

Output: CSV manifest with OCR/CLIP scores and filtering decisions.
"""

import argparse
import csv
import logging
import os
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import easyocr
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.debug import is_debug_mode, set_seeds


logger = logging.getLogger(__name__)


class MemeImageFilter:
    """Filter meme images using OCR and CLIP."""

    def __init__(self, hf_cache: Optional[str] = None, debug: bool = False):
        """
        Initialize OCR and CLIP models.

        Args:
            hf_cache: HuggingFace cache directory
            debug: If True, skip filtering and return all images
        """
        self.debug = debug

        if not debug:
            logger.info("Loading EasyOCR model...")
            ocr_cache = os.path.join(hf_cache, "easyocr") if hf_cache else None
            if ocr_cache:
                os.makedirs(ocr_cache, exist_ok=True)
            self.ocr = easyocr.Reader(
                ['en'],
                gpu=torch.cuda.is_available(),
                model_storage_directory=ocr_cache,
            )

            logger.info("Loading CLIP model (openai/clip-vit-large-patch14)...")
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=hf_cache
            )
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=hf_cache
            )

            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
            self.clip_model.eval()

        self.prompt_meme = "a meme with a photo, illustration, or face with text overlaid"
        self.prompt_screenshot = "a screenshot of a text message, tweet, or text conversation"

    def extract_text_ocr(self, image_path: str) -> Tuple[str, int]:
        """
        Extract text from image using EasyOCR.

        Returns:
            (ocr_text, char_count)
        """
        try:
            result = self.ocr.readtext(image_path, detail=0)
            text = " ".join(result)
            return text, len(text)
        except Exception as e:
            logger.warning(f"OCR failed for {image_path}: {e}")
            return "", 0

    def compute_clip_scores(self, image_path: str) -> Tuple[float, float]:
        """
        Compute CLIP similarity scores for meme and screenshot prompts.

        Returns:
            (meme_score, screenshot_score)
        """
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.clip_processor(
                text=[self.prompt_meme, self.prompt_screenshot],
                images=image,
                return_tensors="pt",
                padding=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            meme_score = float(probs[0, 0].cpu())
            screenshot_score = float(probs[0, 1].cpu())

            return meme_score, screenshot_score
        except Exception as e:
            logger.warning(f"CLIP scoring failed for {image_path}: {e}")
            return 0.0, 0.0

    def filter_image(
        self,
        image_path: str,
        dataset: str,
        original_label: Optional[str] = None
    ) -> dict:
        """
        Filter a single image and return metadata.

        Returns:
            dict with keys: image_path, dataset, original_label, ocr_text, ocr_char_count,
                           clip_meme_score, clip_screenshot_score, kept
        """
        result = {
            "image_path": image_path,
            "dataset": dataset,
            "original_label": original_label or "",
            "ocr_text": "",
            "ocr_char_count": 0,
            "clip_meme_score": 0.0,
            "clip_screenshot_score": 0.0,
            "kept": False
        }

        if self.debug:
            result["kept"] = True
            return result

        # Check if image file exists
        if not os.path.isfile(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return result

        # Step 1: OCR filtering
        ocr_text, char_count = self.extract_text_ocr(image_path)
        result["ocr_text"] = ocr_text
        result["ocr_char_count"] = char_count

        if char_count < 10 or char_count > 300:
            return result

        # Step 2: CLIP filtering
        meme_score, screenshot_score = self.compute_clip_scores(image_path)
        result["clip_meme_score"] = round(meme_score, 4)
        result["clip_screenshot_score"] = round(screenshot_score, 4)

        if meme_score > screenshot_score:
            result["kept"] = True

        return result

    def filter_dataset(
        self,
        images_dir: str,
        dataset: str,
        labels_dict: Optional[dict] = None
    ) -> Tuple[list, dict]:
        """
        Filter all images in a directory.

        Args:
            images_dir: Directory containing images
            dataset: Dataset name (harmeme|mami|mmhs150k)
            labels_dict: Optional dict mapping image_path to original_label

        Returns:
            (filtered_results, summary_stats)
        """
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            logger.error(f"Images directory not found: {images_dir}")
            return [], {}

        # Find all image files
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + \
                      list(images_dir.glob("*.png")) + list(images_dir.glob("*.JPG")) + \
                      list(images_dir.glob("*.PNG"))

        if not image_files:
            logger.warning(f"No image files found in {images_dir}")
            return [], {}

        logger.info(f"Found {len(image_files)} images in {images_dir}")

        results = []
        failed_ocr_low = 0
        failed_ocr_high = 0
        failed_clip = 0

        print(f"\n{'='*60}")
        print(f"  STAGE 0 — Filtering {dataset.upper()} ({len(image_files)} images)")
        print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
        print(f"{'='*60}\n")

        for i, image_path in enumerate(tqdm(image_files, desc=f"Filtering {dataset}")):
            original_label = labels_dict.get(str(image_path)) if labels_dict else None
            result = self.filter_image(str(image_path), dataset, original_label)
            results.append(result)

            if not result["kept"]:
                char_count = result["ocr_char_count"]
                if char_count == 0:
                    failed_ocr_low += 1
                elif char_count < 10:
                    failed_ocr_low += 1
                elif char_count > 300:
                    failed_ocr_high += 1
                else:
                    failed_clip += 1

            # Print running stats every 100 images
            if (i + 1) % 100 == 0:
                num_kept_so_far = sum(1 for r in results if r["kept"])
                keep_rate = 100 * num_kept_so_far / len(results)
                logger.info(
                    f"[{i+1}/{len(image_files)}] kept={num_kept_so_far} "
                    f"({keep_rate:.1f}%) | ocr_low={failed_ocr_low} "
                    f"ocr_high={failed_ocr_high} clip={failed_clip}"
                )

        num_kept = sum(1 for r in results if r["kept"])

        stats = {
            "total": len(results),
            "failed_ocr_low": failed_ocr_low,
            "failed_ocr_high": failed_ocr_high,
            "failed_clip": failed_clip,
            "kept": num_kept
        }

        return results, stats


def save_example_images(
    results: list,
    output_dir: str,
    n_examples: int = 10,
):
    """
    Save up to n_examples individual images into two subfolders:
        <output_dir>/kept/       ← images that passed the filter
        <output_dir>/discarded/  ← images that failed the filter

    Skips a subfolder if it already exists (so re-running is safe).
    """
    output_dir = Path(output_dir)

    kept      = [r for r in results if str(r.get("kept", "")).lower() in ("true", "1")]
    discarded = [r for r in results if str(r.get("kept", "")).lower() not in ("true", "1")]

    random.seed(0)

    def _copy_sample(records: list, dest: Path):
        if dest.exists():
            logger.info(f"  {dest.name}/ already exists — skipping")
            return
        dest.mkdir(parents=True)
        sample = random.sample(records, min(n_examples, len(records)))
        if not sample:
            logger.info(f"  No images to save in {dest.name}/")
            return
        for record in sample:
            src = Path(record["image_path"])
            if src.exists():
                shutil.copy2(src, dest / src.name)
        logger.info(f"  Saved {len(sample)} images → {dest}")

    _copy_sample(kept,      output_dir / "kept")
    _copy_sample(discarded, output_dir / "discarded")


def print_summary_table(all_stats: dict):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<30} {'Count':>10} {'Percentage':>15}")
    print("-" * 80)

    total_images = sum(s["total"] for s in all_stats.values())
    total_kept = sum(s["kept"] for s in all_stats.values())
    total_failed_ocr_low = sum(s["failed_ocr_low"] for s in all_stats.values())
    total_failed_ocr_high = sum(s["failed_ocr_high"] for s in all_stats.values())
    total_failed_clip = sum(s["failed_clip"] for s in all_stats.values())

    print(f"{'Total Images':<30} {total_images:>10} {100.0:>14.1f}%")
    print(f"{'Failed OCR (< 10 chars)':<30} {total_failed_ocr_low:>10} {100*total_failed_ocr_low/max(total_images,1):>14.1f}%")
    print(f"{'Failed OCR (> 300 chars)':<30} {total_failed_ocr_high:>10} {100*total_failed_ocr_high/max(total_images,1):>14.1f}%")
    print(f"{'Failed CLIP':<30} {total_failed_clip:>10} {100*total_failed_clip/max(total_images,1):>14.1f}%")
    print(f"{'Final Kept':<30} {total_kept:>10} {100*total_kept/max(total_images,1):>14.1f}%")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Filter meme images using OCR and CLIP"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["harmeme", "mami", "mmhs150k"],
        help="Dataset name"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images to filter"
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        required=True,
        help="Output CSV manifest path"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: skip filtering, return all images as kept"
    )
    parser.add_argument(
        "--hf_cache",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--save_examples",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "If provided, save two image grids (kept / discarded) to this directory "
            "after filtering, so you can visually verify the filter is working correctly. "
            "Example: --save_examples /scratch/hmr_data/harmeme/filter_examples"
        )
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=10,
        help="Number of example images to copy per folder (default: 10)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("easyocr").setLevel(logging.WARNING)

    # Suppress PyTorch pin_memory warning (fires on every EasyOCR batch on CPU)
    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*no accelerator.*",
        category=UserWarning,
    )

    # Set seeds for reproducibility
    set_seeds(42)

    print(f"\n{'='*60}")
    print(f"  Stage 0: OCR + CLIP Meme Filter")
    print(f"  Dataset:  {args.dataset}")
    print(f"  Images:   {args.images_dir}")
    print(f"  Manifest: {args.output_manifest}")
    print(f"  HF cache: {args.hf_cache}")
    print(f"  Debug:    {args.debug}")
    print(f"{'='*60}\n")

    output_path = Path(args.output_manifest)

    # ------------------------------------------------------------------
    # Resume: if manifest already exists, skip filtering and just create
    # the example folders from the existing results.
    # ------------------------------------------------------------------
    if output_path.exists():
        logger.info(f"Manifest already exists at {output_path} — skipping filtering.")

        if args.save_examples:
            logger.info("Loading existing manifest to create example folders...")
            results = []
            with open(output_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["kept"] = row["kept"].strip().lower() in ("true", "1")
                    results.append(row)

            kept_count = sum(1 for r in results if r["kept"])
            logger.info(f"Loaded {len(results)} rows ({kept_count} kept) from manifest.")

            logger.info(f"Saving example images to {args.save_examples} ...")
            save_example_images(results, args.save_examples, n_examples=args.n_examples)
            print(f"\nExample folders saved to: {args.save_examples}/")
            print(f"  kept/       — up to {args.n_examples} images that PASSED the filter")
            print(f"  discarded/  — up to {args.n_examples} images that FAILED the filter")
        else:
            logger.info("No --save_examples path given, nothing to do.")

        return 0

    # ------------------------------------------------------------------
    # Full run: filter images, write manifest, then create example folders
    # ------------------------------------------------------------------
    if args.debug:
        logger.warning("DEBUG MODE ENABLED: Skipping all filters, returning all images as kept")

    filter_obj = MemeImageFilter(hf_cache=args.hf_cache, debug=args.debug)

    results, stats = filter_obj.filter_dataset(args.images_dir, args.dataset)

    if not results:
        logger.error("No images were processed")
        return 1

    # Write manifest CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path", "dataset", "original_label", "ocr_text", "ocr_char_count",
        "clip_meme_score", "clip_screenshot_score", "kept"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Manifest written to {output_path}")

    # Print summary
    print_summary_table({args.dataset: stats})

    # Save example images
    if args.save_examples:
        logger.info(f"Saving example images to {args.save_examples} ...")
        save_example_images(results, args.save_examples, n_examples=args.n_examples)
        print(f"\nExample folders saved to: {args.save_examples}/")
        print(f"  kept/       — up to {args.n_examples} images that PASSED the filter")
        print(f"  discarded/  — up to {args.n_examples} images that FAILED the filter")

    return 0


if __name__ == "__main__":
    sys.exit(main())
