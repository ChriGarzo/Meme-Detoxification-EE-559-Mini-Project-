"""
Stage 1 (explanations-only, sharded): generate LLaVA explanations only.

This script intentionally does NOT generate pseudo-rewrites.
Use it to populate/refresh explanation JSONL shards.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import tqdm
from codecarbon import EmissionsTracker
from transformers.utils import logging as hf_logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.explainer import MemeExplainer

# Reuse shared helpers from the combined Stage 1 script.
from run_stage1_multimodal_sharded import (
    ensure_explanation_non_null,
    load_existing_ids,
    set_seed,
    write_jsonl_batch,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 (explanations-only + sharded): Generate explanations"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'train')")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest CSV from Stage 0")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSONL files")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load LLaVA in 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_shards", type=int, default=8, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id in [0, num_shards-1]")
    parser.add_argument("--hateful_only", action="store_true", help="Only process examples where hateful=1")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process max 16 examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--explain_max_retries",
        type=int,
        default=0,
        help="Additional retries for explanation generation (0 => single attempt)",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards-1]")
    if args.explain_max_retries < 0:
        raise ValueError("--explain_max_retries must be >= 0")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    shard_tag = f"shard{args.shard_id:02d}of{args.num_shards:02d}"
    stage1_log_path = os.path.join(args.output_dir, f"stage1_explain_only_{shard_tag}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(stage1_log_path),
            logging.StreamHandler(),
        ],
    )

    for noisy_logger in ["httpx", "huggingface_hub", "urllib3", "matplotlib", "PIL"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    print(f"\n{'='*60}")
    print("  Stage 1: Explanations Only")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Images:     {args.images_dir}")
    print(f"  Manifest:   {args.manifest_path}")
    print(f"  Output:     {args.output_dir}")
    print(f"  HF cache:   {args.hf_cache}")
    print(f"  4-bit quant:{args.load_in_4bit}")
    print(f"  Shard:      {args.shard_id + 1}/{args.num_shards} ({shard_tag})")
    print(f"  Debug:      {args.debug}")
    print(f"{'='*60}\n")

    logger.info("Starting explanations-only Stage 1")
    logger.info(f"Arguments: {vars(args)}")

    manifest_df = pd.read_csv(args.manifest_path)
    total_in_manifest = len(manifest_df)
    kept_in_manifest = int(manifest_df["kept"].sum()) if "kept" in manifest_df.columns else total_in_manifest
    logger.info(f"Manifest loaded: {total_in_manifest} total rows, {kept_in_manifest} kept by Stage 0")

    manifest_df = manifest_df[manifest_df["kept"] == True] if "kept" in manifest_df.columns else manifest_df
    if args.hateful_only and "hateful" in manifest_df.columns:
        before = len(manifest_df)
        manifest_df = manifest_df[manifest_df["hateful"] == 1]
        logger.info(f"--hateful_only: kept {len(manifest_df)}/{before} hateful examples")
    if args.debug:
        manifest_df = manifest_df.head(16)

    logger.info(f"Manifest rows after filters (before sharding): {len(manifest_df)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        logger.info("No GPU found — running on CPU (will be slow)")

    explainer = MemeExplainer(
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.hf_cache,
        device=device,
        debug=args.debug,
    )

    explanations_path = os.path.join(args.output_dir, f"{args.dataset}_explanations_{shard_tag}.jsonl")
    processed_explanation_ids = load_existing_ids(explanations_path)
    logger.info(f"Already processed explanations: {len(processed_explanation_ids)}")

    explanations_batch: List[Dict] = []
    total_examples = 0
    json_parse_failures = 0
    forced_non_null_explanations = 0

    emissions_file = f"emissions_explain_only_{shard_tag}.csv"
    tracker = EmissionsTracker(log_level="warning", output_dir=args.output_dir, output_file=emissions_file)
    tracker.start()

    try:
        all_records = manifest_df.to_dict("records")
        records = [row for idx, row in enumerate(all_records) if idx % args.num_shards == args.shard_id]
        logger.info(
            "Shard %d/%d selected %d/%d rows",
            args.shard_id,
            args.num_shards,
            len(records),
            len(all_records),
        )

        next_stats_log = 50
        next_flush = 100

        with tqdm.tqdm(total=len(records), desc="Processing explanations") as pbar:
            for start_idx in range(0, len(records), args.batch_size):
                raw_batch_rows = records[start_idx:start_idx + args.batch_size]
                pending_rows = []

                for row in raw_batch_rows:
                    example_id = row.get("id")
                    if example_id in processed_explanation_ids:
                        pbar.update(1)
                        continue
                    pending_rows.append(row)

                if not pending_rows:
                    continue

                batch_ids: List[str] = []
                batch_image_paths: List[str] = []
                batch_original_texts: List[str] = []
                batch_hateful_flags: List[bool] = []

                for row in pending_rows:
                    example_id = row.get("id")
                    raw_img = str(row.get("image_path", ""))
                    if os.path.isabs(raw_img) and os.path.exists(raw_img):
                        image_path = raw_img
                    else:
                        image_path = os.path.join(args.images_dir, raw_img)
                    original_text = str(row.get("text", "") or row.get("ocr_text", ""))
                    is_hateful = bool(row.get("hateful", False))

                    batch_ids.append(example_id)
                    batch_image_paths.append(image_path)
                    batch_original_texts.append(original_text)
                    batch_hateful_flags.append(is_hateful)

                total_examples += len(pending_rows)

                try:
                    batch_explanations = explainer.batch_explain(
                        batch_image_paths,
                        batch_original_texts,
                        max_retries=args.explain_max_retries,
                    )
                except Exception as e:
                    logger.warning(
                        "Batch explanation generation failed for rows %d-%d: %s. Falling back to serial explain.",
                        start_idx,
                        start_idx + len(pending_rows) - 1,
                        e,
                    )
                    batch_explanations = []
                    for image_path, original_text, is_hateful in zip(
                        batch_image_paths,
                        batch_original_texts,
                        batch_hateful_flags,
                    ):
                        try:
                            explanation = explainer.explain(
                                image_path,
                                original_text,
                                max_retries=args.explain_max_retries,
                            )
                        except Exception as inner_e:
                            if is_hateful:
                                explanation = {
                                    "target_group": "other",
                                    "visual_evidence": "A visual cue in the meme is used to frame the target group negatively.",
                                    "implicit_meaning": "The meme uses both text and visual context to communicate a hateful or derogatory framing toward a target group.",
                                    "error": str(inner_e),
                                }
                            else:
                                explanation = {
                                    "target_group": None,
                                    "visual_evidence": None,
                                    "implicit_meaning": None,
                                    "error": str(inner_e),
                                }
                        batch_explanations.append(explanation)

                for i, row in enumerate(pending_rows):
                    example_id = batch_ids[i]
                    explanation = batch_explanations[i]
                    is_hateful = batch_hateful_flags[i]
                    original_text = batch_original_texts[i]

                    if explanation.get("parse_error"):
                        json_parse_failures += 1

                    if is_hateful:
                        explanation, was_forced = ensure_explanation_non_null(explanation)
                        if was_forced:
                            forced_non_null_explanations += 1

                    explanation_record = {
                        "id": example_id,
                        "image_path": row.get("image_path"),
                        "original_text": original_text,
                        "explanation": explanation,
                        "is_hateful": is_hateful,
                    }
                    explanations_batch.append(explanation_record)
                    processed_explanation_ids.add(example_id)

                pbar.update(len(pending_rows))

                if total_examples >= next_stats_log:
                    logger.info(
                        "[%d/%d] explanations=%d | json_failures=%d | forced_non_null=%d",
                        total_examples,
                        len(records),
                        total_examples,
                        json_parse_failures,
                        forced_non_null_explanations,
                    )
                    next_stats_log = ((total_examples // 50) + 1) * 50

                if explanations_batch and total_examples >= next_flush:
                    write_jsonl_batch(explanations_batch, explanations_path)
                    explanations_batch = []
                    logger.info(f"Wrote explanations batch at example {total_examples}")
                    next_flush = ((total_examples // 100) + 1) * 100

        if explanations_batch:
            write_jsonl_batch(explanations_batch, explanations_path)
            logger.info(f"Wrote final explanations batch ({len(explanations_batch)} items)")

        json_parse_rate = (json_parse_failures / max(total_examples, 1)) * 100

        logger.info("\n=== Explanations-Only Summary ===")
        logger.info(f"Total examples processed: {total_examples}")
        logger.info(f"JSON parse failures: {json_parse_failures} ({json_parse_rate:.2f}%)")
        logger.info(f"Hateful explanations forced to non-null: {forced_non_null_explanations}")
        logger.info(f"Explanations JSONL: {explanations_path}")

    finally:
        emissions = tracker.stop()
        if emissions is not None:
            logger.info(f"Carbon emissions: {emissions:.6f} kg CO2")
            logger.info(f"Emissions saved to: {os.path.join(args.output_dir, emissions_file)}")
        else:
            logger.warning("Carbon emissions could not be measured (CodeCarbon tracking failed)")


if __name__ == "__main__":
    main()
