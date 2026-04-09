"""
Entry point for evaluation: computes all metrics for all systems and outputs results table + JSON.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch

from metrics import compute_sta, compute_sim, compute_clipscore, compute_rewrite_precision, compute_co2, compute_aggregate_J
from utils.debug import is_debug_mode, setup_debug_mode, DEBUG_CONFIG

logger = logging.getLogger(__name__)


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def load_rewrites(rewrites_dir: Path, system_name: str) -> Tuple[List[str], List[str]]:
    """
    Load original texts and rewrites from JSONL file.

    Args:
        rewrites_dir: Directory containing rewrite outputs
        system_name: Name of the system

    Returns:
        Tuple of (originals, rewrites)
    """
    filepath = rewrites_dir / f"{system_name}.jsonl"

    if not filepath.exists():
        logger.warning(f"Rewrite file not found: {filepath}")
        return [], []

    originals = []
    rewrites = []

    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            originals.append(item["original_text"])
            rewrites.append(item["rewrite"])

            if is_debug_mode() and len(rewrites) >= DEBUG_CONFIG["max_examples"]:
                break

    logger.info(f"Loaded {len(rewrites)} examples from {system_name}")
    return originals, rewrites


def load_images(images_dir: Path, rewrite_ids: List[str]) -> List[Path]:
    """Load image paths."""
    images = []
    for idx in rewrite_ids:
        img_path = images_dir / f"{idx}.jpg"
        if not img_path.exists():
            img_path = images_dir / f"{idx}.png"
        if img_path.exists():
            images.append(img_path)
    return images


def load_stage1_outputs(stage1_dir: Path) -> Dict:
    """Load Stage 1 explanations."""
    explanations = {}
    if not stage1_dir.exists():
        logger.warning(f"Stage 1 directory not found: {stage1_dir}")
        return explanations

    for json_file in stage1_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            explanations[json_file.stem] = data

    return explanations


def evaluate_system(
    system_name: str,
    originals: List[str],
    rewrites: List[str],
    images: List,
    original_explanations: List[Dict],
    images_dir: Path,
    explainer=None,
    hf_cache: str = None
) -> Dict:
    """
    Evaluate a single system across all metrics.

    Args:
        system_name: Name of the system
        originals: Original texts
        rewrites: Rewritten texts
        images: Image paths or PIL Images
        original_explanations: Original Stage 1 explanations
        images_dir: Directory of images
        explainer: MemeExplainer instance (for Rewrite Precision)
        hf_cache: Hugging Face cache directory

    Returns:
        Dict with all metric results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {system_name}")
    logger.info(f"{'='*60}")

    if len(rewrites) == 0:
        logger.warning(f"No rewrites for {system_name}, skipping")
        return None

    results = {
        "system": system_name,
        "num_examples": len(rewrites)
    }

    # Set cache directory
    if hf_cache:
        import os
        os.environ["HF_HOME"] = hf_cache

    # STA: Semantic Toxicity Attenuation
    logger.info("Computing STA...")
    sta_result = compute_sta(rewrites)
    results["sta"] = sta_result

    # SIM: Semantic Similarity (BERTScore)
    logger.info("Computing SIM...")
    sim_result = compute_sim(originals, rewrites)
    results["sim"] = sim_result

    # CLIPScore
    if len(images) > 0:
        logger.info("Computing CLIPScore...")
        try:
            clip_result = compute_clipscore(images, rewrites)
            results["clip"] = clip_result
        except Exception as e:
            logger.error(f"CLIPScore computation failed: {e}")
            results["clip"] = None
    else:
        logger.warning("No images provided, skipping CLIPScore")
        results["clip"] = None

    # Rewrite Precision
    if explainer is not None and len(original_explanations) > 0:
        logger.info("Computing Rewrite Precision...")
        try:
            rp_result = compute_rewrite_precision(images, rewrites, original_explanations, explainer)
            results["rewrite_precision"] = rp_result
        except Exception as e:
            logger.error(f"Rewrite Precision computation failed: {e}")
            results["rewrite_precision"] = None
    else:
        logger.warning("Explainer or original explanations not available, skipping Rewrite Precision")
        results["rewrite_precision"] = None

    # Aggregate J
    if results["clip"] is not None and results["rewrite_precision"] is not None:
        J = compute_aggregate_J(sta_result, sim_result, results["clip"], results["rewrite_precision"])
        results["aggregate_j"] = J
    else:
        logger.warning("Cannot compute aggregate J without CLIPScore and Rewrite Precision")
        results["aggregate_j"] = None

    # Estimate params (hardcoded for known models)
    params_millions = estimate_params(system_name)
    results["params_millions"] = params_millions

    return results


def estimate_params(system_name: str) -> float:
    """Estimate number of parameters (in millions) for a system."""
    params_map = {
        "llava_end_to_end": 34.0,  # LLaVA 1.5 (34B)
        "llava_structured": 34.0,
        "detoxllm_text_only": 7.0,  # DetoxLLM 7B
        "bart_none": 0.4,  # BART base
        "bart_target_only": 0.4,
        "bart_attack_only": 0.4,
        "bart_full": 0.4,
        "clip_proxy_bart": 0.4 + 0.428,  # BART + CLIP ViT
    }
    return params_map.get(system_name, 0.0)


def format_results_table(all_results: List[Dict]) -> str:
    """Format results as a nice table."""
    lines = []
    lines.append("=" * 120)
    lines.append(f"{'System':<25} {'STA':<10} {'SIM':<10} {'CLIP':<10} {'RP':<10} {'J':<10} {'Params(M)':<12} {'CO2(g)':<10}")
    lines.append("-" * 120)

    for result in all_results:
        if result is None:
            continue

        system = result["system"]
        sta = f"{result['sta']['mean']:.4f}" if result["sta"] else "N/A"
        sim = f"{result['sim']['mean']:.4f}" if result["sim"] else "N/A"
        clip = f"{result['clip']['mean']:.4f}" if result["clip"] else "N/A"
        rp = f"{result['rewrite_precision']['mean']:.4f}" if result["rewrite_precision"] else "N/A"
        j = f"{result['aggregate_j']:.6f}" if result["aggregate_j"] else "N/A"
        params = f"{result['params_millions']:.1f}" if result["params_millions"] else "0.0"
        co2 = result.get("co2", "N/A")
        if isinstance(co2, (int, float)):
            co2 = f"{co2:.4f}"

        lines.append(f"{system:<25} {sta:<10} {sim:<10} {clip:<10} {rp:<10} {j:<10} {params:<12} {co2:<10}")

    lines.append("=" * 120)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate hateful meme detoxification systems")
    parser.add_argument("--rewrites_dir", type=Path, required=True, help="Directory with rewrite JSONL files")
    parser.add_argument("--images_dir", type=Path, required=True, help="Directory with meme images")
    parser.add_argument("--stage1_outputs_dir", type=Path, default=None, help="Directory with Stage 1 outputs")
    parser.add_argument("--output_path", type=Path, required=True, help="Output JSON file path")
    parser.add_argument("--hf_cache", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        setup_debug_mode()

    setup_logging(debug=args.debug)
    logger.info("Starting evaluation")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # List of systems to evaluate
    systems = [
        "llava_end_to_end",
        "llava_structured",
        "detoxllm_text_only",
        "bart_none",
        "bart_target_only",
        "bart_attack_only",
        "bart_full",
        "clip_proxy_bart",
    ]

    all_results = []

    # Try to load explainer if stage1 outputs available
    explainer = None
    if args.stage1_outputs_dir:
        try:
            from models.explainer import MemeExplainer
            explainer = MemeExplainer(hf_cache=args.hf_cache)
        except Exception as e:
            logger.warning(f"Could not load explainer: {e}")

    # Evaluate each system
    for system in systems:
        originals, rewrites = load_rewrites(args.rewrites_dir, system)

        if not rewrites:
            logger.warning(f"Skipping {system}: no rewrites found")
            continue

        # Load images (match by index)
        images = []
        if args.images_dir.exists():
            # Assuming images are named by index: 0.jpg, 1.jpg, etc.
            for i in range(len(rewrites)):
                img_path = args.images_dir / f"{i}.jpg"
                if not img_path.exists():
                    img_path = args.images_dir / f"{i}.png"
                if img_path.exists():
                    images.append(img_path)

        # Load original explanations
        original_explanations = []
        if args.stage1_outputs_dir:
            stage1_data = load_stage1_outputs(args.stage1_outputs_dir)
            original_explanations = [stage1_data.get(i, {}) for i in range(len(rewrites))]

        result = evaluate_system(
            system,
            originals,
            rewrites,
            images,
            original_explanations,
            args.images_dir,
            explainer=explainer,
            hf_cache=args.hf_cache
        )

        if result:
            all_results.append(result)

    # Print table
    table = format_results_table(all_results)
    print("\n" + table)

    # Save JSON
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
