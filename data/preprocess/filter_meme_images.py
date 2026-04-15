"""
Stage 0: Filter meme images using OCR and CLIP.

This script filters meme images by:
1. Extracting text via EasyOCR (keep 10-300 characters)
2. Using CLIP to distinguish memes from screenshots

CLIP logic is dataset-specific:
  - harmeme / mami : original binary check (2 prompts, keep if meme_score > screenshot_score)
  - mmhs150k       : stricter multi-class check (5 prompts, keep only if the meme prompt
                     has the highest softmax probability AND exceeds a minimum threshold).
                     MMHS150K originates from Twitter and contains many non-meme images
                     (plain photos, social-media video thumbnails, phone UI screenshots)
                     that slip through the simpler binary filter.

Output: CSV manifest with OCR/CLIP scores and filtering decisions.

To visually inspect results after filtering, use the companion script:
    data/preprocess/sample_filter_examples.py
"""

import argparse
import csv
import logging
import os
import random
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

# ---------------------------------------------------------------------------
# MMHS150K-specific CLIP threshold.
# The meme prompt must reach at least this softmax probability (among the 5
# MMHS150K prompts) for the image to be kept.
# ---------------------------------------------------------------------------
MMHS150K_CLIP_THRESHOLD: float = 0.45

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.debug import is_debug_mode, set_seeds


logger = logging.getLogger(__name__)


class MemeImageFilter:
    """Filter meme images using OCR and CLIP."""

    def __init__(self, hf_cache: Optional[str] = None, debug: bool = False,
                 mmhs150k_clip_threshold: Optional[float] = None):
        """
        Initialize OCR and CLIP models.

        Args:
            hf_cache: HuggingFace cache directory
            debug: If True, skip filtering and return all images
            mmhs150k_clip_threshold: Override the CLIP threshold used for
                MMHS150K (default: MMHS150K_CLIP_THRESHOLD = 0.45).
                Has no effect on harmeme or mami processing.
        """
        self.debug = debug
        self.mmhs150k_threshold = (
            mmhs150k_clip_threshold
            if mmhs150k_clip_threshold is not None
            else MMHS150K_CLIP_THRESHOLD
        )

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

        # ------------------------------------------------------------------
        # Original binary prompts — used for harmeme and mami (unchanged).
        # ------------------------------------------------------------------
        self.prompt_meme = "a meme with a photo, illustration, or face with text overlaid"
        self.prompt_screenshot = "a screenshot of a text message, tweet, or text conversation"

        # ------------------------------------------------------------------
        # MMHS150K multi-class prompts (1 positive + 4 targeted negatives).
        # The four negatives cover the main failure modes seen in Twitter
        # images: social-media video thumbnails, plain photos of people,
        # and phone/app UI screenshots.
        # ------------------------------------------------------------------
        self.mmhs150k_prompts = [
            # positive (index 0)
            "a meme with a photo, illustration, or face with overlaid text",
            # negatives (indices 1-4)
            "a screenshot of a text message, tweet, or text conversation",
            "a screenshot of a social media video post or video thumbnail",
            "a plain photograph of a person or scene without any overlaid text",
            "a screenshot of a mobile phone or social media app interface",
        ]

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
        Used for harmeme and mami (original binary logic, unchanged).

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

    def compute_clip_scores_mmhs(self, image_path: str) -> Tuple[float, float, float, str]:
        """
        Compute CLIP similarity scores over the MMHS150K multi-class prompts.
        Used exclusively for mmhs150k.

        Returns:
            (meme_score, screenshot_score, best_negative_score, best_negative_label)
            where:
              meme_score          – softmax probability of the meme prompt (index 0)
              screenshot_score    – softmax probability of the text-screenshot
                                    prompt (index 1); kept for CSV column compat
              best_negative_score – highest probability among ALL 4 negative prompts
              best_negative_label – short name of the winning negative class
        """
        neg_labels = ["text_screenshot", "video_screenshot", "plain_photo", "phone_ui"]
        try:
            image = Image.open(image_path).convert("RGB")

            inputs = self.clip_processor(
                text=self.mmhs150k_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)

            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0].cpu().tolist()

            meme_score       = probs[0]
            screenshot_score = probs[1]        # text-screenshot (legacy column)
            neg_probs        = probs[1:]       # all four negatives
            best_neg_idx     = int(np.argmax(neg_probs))
            best_neg_score   = neg_probs[best_neg_idx]
            best_neg_label   = neg_labels[best_neg_idx]

            return meme_score, screenshot_score, best_neg_score, best_neg_label
        except Exception as e:
            logger.warning(f"CLIP scoring failed for {image_path}: {e}")
            return 0.0, 0.0, 0.0, "unknown"

    def filter_image(
        self,
        image_path: str,
        dataset: str,
        original_label: Optional[str] = None
    ) -> dict:
        """
        Filter a single image and return metadata.

        For harmeme / mami (original logic, unchanged):
          1. OCR: discard if extracted text is < 10 or > 300 characters
          2. CLIP binary: keep if meme_score > screenshot_score

        For mmhs150k (stricter logic):
          1. OCR: same as above
          2. CLIP multi-class (5 prompts): keep only if the meme prompt has the
             highest softmax probability among all 5 prompts AND that probability
             is >= self.mmhs150k_threshold.

        Returns:
            dict — base keys (all datasets):
                image_path, dataset, original_label, ocr_text, ocr_char_count,
                clip_meme_score, clip_screenshot_score, kept
            extra keys (mmhs150k only):
                clip_best_negative, clip_threshold_used
        """
        is_mmhs = dataset.lower() == "mmhs150k"

        result = {
            "image_path":            image_path,
            "dataset":               dataset,
            "original_label":        original_label or "",
            "ocr_text":              "",
            "ocr_char_count":        0,
            "clip_meme_score":       0.0,
            "clip_screenshot_score": 0.0,
            "kept":                  False,
        }
        if is_mmhs:
            result["clip_best_negative"]  = ""
            result["clip_threshold_used"] = self.mmhs150k_threshold

        if self.debug:
            result["kept"] = True
            return result

        # Check if image file exists
        if not os.path.isfile(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return result

        # Step 1: OCR filtering (identical for all datasets)
        ocr_text, char_count = self.extract_text_ocr(image_path)
        result["ocr_text"]       = ocr_text
        result["ocr_char_count"] = char_count

        if char_count < 10 or char_count > 300:
            return result

        # Step 2: CLIP filtering — dataset-specific
        if is_mmhs:
            # ── MMHS150K: multi-class, strict threshold ──────────────────────
            meme_score, screenshot_score, best_neg_score, best_neg_label = \
                self.compute_clip_scores_mmhs(image_path)

            result["clip_meme_score"]       = round(meme_score, 4)
            result["clip_screenshot_score"] = round(screenshot_score, 4)
            result["clip_best_negative"]    = best_neg_label

            if meme_score > best_neg_score and meme_score >= self.mmhs150k_threshold:
                result["kept"] = True
        else:
            # ── harmeme / mami: original binary check ────────────────────────
            meme_score, screenshot_score = self.compute_clip_scores(image_path)
            result["clip_meme_score"]       = round(meme_score, 4)
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

        is_mmhs = dataset.lower() == "mmhs150k"

        results = []
        failed_ocr_low  = 0
        failed_ocr_high = 0
        failed_clip     = 0
        # MMHS150K-only sub-counters
        failed_clip_threshold = 0
        failed_clip_not_top   = 0

        print(f"\n{'='*60}")
        print(f"  STAGE 0 — Filtering {dataset.upper()} ({len(image_files)} images)")
        if is_mmhs:
            print(f"  CLIP mode      : multi-class (5 prompts, threshold={self.mmhs150k_threshold})")
        else:
            print(f"  CLIP mode      : binary (2 prompts, keep if meme > screenshot)")
        print(f"  Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
        print(f"{'='*60}\n")

        for i, image_path in enumerate(tqdm(image_files, desc=f"Filtering {dataset}")):
            original_label = labels_dict.get(str(image_path)) if labels_dict else None
            result = self.filter_image(str(image_path), dataset, original_label)
            results.append(result)

            if not result["kept"]:
                char_count = result["ocr_char_count"]
                if char_count == 0 or char_count < 10:
                    failed_ocr_low += 1
                elif char_count > 300:
                    failed_ocr_high += 1
                else:
                    failed_clip += 1
                    if is_mmhs:
                        ms = result["clip_meme_score"]
                        if ms < self.mmhs150k_threshold:
                            failed_clip_threshold += 1
                        else:
                            failed_clip_not_top += 1

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
            "total":           len(results),
            "failed_ocr_low":  failed_ocr_low,
            "failed_ocr_high": failed_ocr_high,
            "failed_clip":     failed_clip,
            "kept":            num_kept,
        }
        if is_mmhs:
            stats["failed_clip_threshold"] = failed_clip_threshold
            stats["failed_clip_not_top"]   = failed_clip_not_top
            stats["clip_threshold"]        = self.mmhs150k_threshold

        return results, stats


def print_summary_table(all_stats: dict):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<38} {'Count':>10} {'Percentage':>12}")
    print("-" * 80)

    total_images          = sum(s["total"]           for s in all_stats.values())
    total_kept            = sum(s["kept"]             for s in all_stats.values())
    total_failed_ocr_low  = sum(s["failed_ocr_low"]  for s in all_stats.values())
    total_failed_ocr_high = sum(s["failed_ocr_high"] for s in all_stats.values())
    total_failed_clip     = sum(s["failed_clip"]      for s in all_stats.values())

    def _row(label, count):
        pct = 100 * count / max(total_images, 1)
        print(f"  {label:<36} {count:>10} {pct:>11.1f}%")

    _row("Total Images",             total_images)
    _row("Failed OCR (< 10 chars)",  total_failed_ocr_low)
    _row("Failed OCR (> 300 chars)", total_failed_ocr_high)
    _row("Failed CLIP",              total_failed_clip)
    _row("Final Kept",               total_kept)

    # Extra breakdown for MMHS150K
    for ds, s in all_stats.items():
        if "failed_clip_threshold" in s:
            print(f"\n  MMHS150K CLIP breakdown (threshold={s.get('clip_threshold', '?')}):")
            _row("  ↳ meme score below threshold",   s["failed_clip_threshold"])
            _row("  ↳ negative class scored higher", s["failed_clip_not_top"])

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
        "--mmhs150k_clip_threshold",
        type=float,
        default=None,
        help=(
            f"Override the CLIP meme-probability threshold used for MMHS150K "
            f"(default: {MMHS150K_CLIP_THRESHOLD}). "
            "Has no effect when --dataset is harmeme or mami. "
            "Raise this value to keep fewer but cleaner MMHS150K images."
        )
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
    print(f"  Dataset:     {args.dataset}")
    print(f"  Images:      {args.images_dir}")
    print(f"  Manifest:    {args.output_manifest}")
    print(f"  HF cache:    {args.hf_cache}")
    print(f"  Debug:    {args.debug}")
    print(f"{'='*60}\n")

    output_path = Path(args.output_manifest)

    # ------------------------------------------------------------------
    # Resume check: skip if manifest already exists, unless --force_rerun.
    # ------------------------------------------------------------------
    if output_path.exists():
        logger.info(f"Manifest already exists at {output_path} — skipping filtering.")
        return 0

    # ------------------------------------------------------------------
    # Full run: filter images and write manifest.
    # ------------------------------------------------------------------
    if args.debug:
        logger.warning("DEBUG MODE ENABLED: Skipping all filters, returning all images as kept")

    filter_obj = MemeImageFilter(
        hf_cache=args.hf_cache,
        debug=args.debug,
        mmhs150k_clip_threshold=args.mmhs150k_clip_threshold,
    )

    results, stats = filter_obj.filter_dataset(args.images_dir, args.dataset)

    if not results:
        logger.error("No images were processed")
        return 1

    # Write manifest CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Base fieldnames (harmeme, mami) — identical to the original script.
    # MMHS150K gets two extra diagnostic columns.
    base_fieldnames = [
        "image_path", "dataset", "original_label", "ocr_text", "ocr_char_count",
        "clip_meme_score", "clip_screenshot_score", "kept"
    ]
    mmhs_extra = ["clip_best_negative", "clip_threshold_used"]

    if args.dataset == "mmhs150k":
        fieldnames = (
            base_fieldnames[:-1]   # everything except "kept"
            + mmhs_extra
            + ["kept"]             # "kept" stays last
        )
    else:
        fieldnames = base_fieldnames

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Manifest written to {output_path}")

    # Print summary
    print_summary_table({args.dataset: stats})

    return 0


if __name__ == "__main__":
    sys.exit(main())
