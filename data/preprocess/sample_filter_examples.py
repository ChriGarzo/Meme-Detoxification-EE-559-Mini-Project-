"""
Sample filter examples from Stage 0 manifests.

Reads the three per-dataset manifests produced by filter_meme_images.py and
copies a random sample of kept and discarded images into an output directory
so you can visually inspect the quality of the filter.

Output layout:
    <output_dir>/                 (default: /scratch/hmr_data/filtering_results)
        kept/
            harmeme/
            mami/
            mmhs150k/
        discarded/
            harmeme/
            mami/
            mmhs150k/

Usage example:
    python data/preprocess/sample_filter_examples.py \\
        --harmeme_manifest  /scratch/hmr_data/harmeme/manifest.csv \\
        --mami_manifest     /scratch/hmr_data/mami/manifest.csv \\
        --mmhs150k_manifest /scratch/hmr_data/mmhs150k/manifest.csv \\
        --output_dir        /scratch/hmr_data/filtering_results \\
        --n_examples        50
"""

import argparse
import csv
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str, dataset_name: str) -> List[dict]:
    """
    Load a Stage 0 manifest CSV and return a list of row dicts.
    The 'kept' field is normalised to a Python bool.
    """
    path = Path(manifest_path)
    if not path.exists():
        logger.error(f"Manifest not found for {dataset_name}: {path}")
        return []

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["kept"] = row["kept"].strip().lower() in ("true", "1")
            rows.append(row)

    kept  = sum(1 for r in rows if r["kept"])
    total = len(rows)
    logger.info(
        f"{dataset_name}: loaded {total} rows from manifest "
        f"({kept} kept, {total - kept} discarded)"
    )
    return rows


# ---------------------------------------------------------------------------
# Sampling and copying
# ---------------------------------------------------------------------------

def sample_and_copy(
    rows: List[dict],
    dataset_name: str,
    output_dir: Path,
    n_examples: int,
    seed: int,
) -> None:
    """
    Sample up to n_examples kept and n_examples discarded images from `rows`
    and copy them into:
        output_dir / dataset_name / kept /
        output_dir / dataset_name / discarded /
    """
    rng = random.Random(seed)

    kept_rows      = [r for r in rows if r["kept"]]
    discarded_rows = [r for r in rows if not r["kept"]]

    kept_sample      = rng.sample(kept_rows,      min(n_examples, len(kept_rows)))
    discarded_sample = rng.sample(discarded_rows, min(n_examples, len(discarded_rows)))

    for subset_name, sample in [("kept", kept_sample), ("discarded", discarded_sample)]:
        dest = output_dir / subset_name / dataset_name
        dest.mkdir(parents=True, exist_ok=True)

        copied  = 0
        missing = 0
        for row in sample:
            src = Path(row["image_path"])
            if not src.exists():
                missing += 1
                continue
            shutil.copy2(src, dest / src.name)
            copied += 1

        logger.info(
            f"  {dataset_name}/{subset_name}: "
            f"copied {copied}/{len(sample)} images"
            + (f" ({missing} source files missing)" if missing else "")
        )

    # Print a compact per-dataset summary
    print(
        f"  {dataset_name:<12}  "
        f"kept={len(kept_rows):>6}  discarded={len(discarded_rows):>6}  "
        f"sampled {min(n_examples, len(kept_rows))} kept "
        f"+ {min(n_examples, len(discarded_rows))} discarded"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample kept / discarded images from Stage 0 filter manifests "
            "for visual inspection."
        )
    )
    parser.add_argument(
        "--harmeme_manifest",
        type=str,
        default="/scratch/hmr_data/harmeme/manifest.csv",
        help="Path to the HarMeme Stage 0 manifest CSV",
    )
    parser.add_argument(
        "--mami_manifest",
        type=str,
        default="/scratch/hmr_data/mami/manifest.csv",
        help="Path to the MAMI Stage 0 manifest CSV",
    )
    parser.add_argument(
        "--mmhs150k_manifest",
        type=str,
        default="/scratch/hmr_data/mmhs150k/manifest.csv",
        help="Path to the MMHS150K Stage 0 manifest CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/hmr_data/filtering_results",
        help="Root directory where sampled images will be written",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=50,
        help="Number of kept and discarded images to sample per dataset (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling (default: 0)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["harmeme", "mami", "mmhs150k"],
        default=["harmeme", "mami", "mmhs150k"],
        help=(
            "Which datasets to sample. Defaults to all three. "
            "Example: --datasets mmhs150k"
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_map = {
        "harmeme":  args.harmeme_manifest,
        "mami":     args.mami_manifest,
        "mmhs150k": args.mmhs150k_manifest,
    }

    print(f"\n{'='*70}")
    print(f"  Sample Filter Examples")
    print(f"  N per split : {args.n_examples} kept + {args.n_examples} discarded")
    print(f"  Output      : {output_dir}")
    print(f"  Datasets    : {', '.join(args.datasets)}")
    print(f"{'='*70}\n")

    any_loaded = False
    for dataset in args.datasets:
        manifest_path = manifest_map[dataset]
        rows = load_manifest(manifest_path, dataset)
        if not rows:
            print(f"  {dataset}: skipped (manifest not found or empty)\n")
            continue
        any_loaded = True
        sample_and_copy(rows, dataset, output_dir, args.n_examples, args.seed)

    if not any_loaded:
        logger.error("No manifests were loaded. Check the manifest paths.")
        return 1

    print(f"\nDone. Open {output_dir} to inspect the samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
