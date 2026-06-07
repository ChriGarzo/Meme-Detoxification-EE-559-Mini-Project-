"""
Post-Stage-0 pipeline step: build unified 80/10/10 stratified splits.

Data sources  : Stage 0 manifests (per-dataset `kept` flags) + raw annotation
                files for HarMeme (di-dimitrov/mmf), MAMI (SemEval-2022 Task 5),
                and MMHS150K (Gomez et al. 2020).
Output        : /scratch/hmr_data/unified_splits/
                    unified_train.csv, unified_val.csv, unified_test.csv,
                    split_stats.json
                Each row: id, image_path, dataset, text, hateful,
                          original_label, split.

Design choices
--------------
- Stage 0 manifests restrict the labeled pool to images that passed the OCR +
  CLIP filter; unannotated or visually non-meme images are excluded before
  splitting, ensuring training and evaluation data contain genuine meme images.
- Stratification is performed jointly over (dataset, hateful) groups so every
  split maintains proportional representation of hateful and non-hateful memes
  from all three source datasets.
- Split ratios: 80 / 10 / 10 (train / val / test), seeded at 42 for
  reproducibility.
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ── Stage-0 manifest helpers ───────────────────────────────────────────────

def load_kept_paths(manifest_path: Optional[str]) -> Optional[Set[str]]:
    """
    Return the set of absolute image paths that passed Stage 0 filtering.

    Returns None when no manifest path is given, signalling that the
    downstream filter should be a no-op. Exits if a path is given but the
    file does not exist.
    """
    if not manifest_path:
        return None

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        logger.error(f"Stage 0 manifest not found: {manifest_path}")
        logger.error("  → Run Stage 0 (filter_meme_images.py) first, then re-run this script.")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    if "kept" not in df.columns:
        logger.error(f"Manifest at {manifest_path} has no 'kept' column.")
        sys.exit(1)

    kept = set(df.loc[df["kept"] == True, "image_path"].astype(str).tolist())
    total = len(df)
    logger.info(f"Manifest {manifest_path.name}: {len(kept)}/{total} images kept by Stage 0")
    return kept


def apply_manifest_filter(examples: List[Dict], kept_paths: Optional[Set[str]],
                           dataset_name: str) -> List[Dict]:
    """
    Retain only examples whose image_path appears in the Stage 0 kept set.

    When kept_paths is None (no manifest provided), all examples pass through
    and a warning is emitted so the absence of filtering is visible in logs.
    """
    if kept_paths is None:
        logger.warning(
            f"{dataset_name}: no Stage 0 manifest provided — "
            "using ALL annotated examples (no OCR/CLIP filtering applied)."
        )
        return examples

    before = len(examples)
    filtered = [e for e in examples if e["image_path"] in kept_paths]
    after = len(filtered)
    removed = before - after
    logger.info(
        f"{dataset_name}: Stage 0 filter → {after}/{before} kept "
        f"({removed} removed, {100*after/max(before,1):.1f}% pass rate)"
    )
    return filtered


# ── Dataset loaders: raw annotations → list of example dicts ──────────────

def load_harmeme(data_dir: str) -> List[Dict]:
    """Load and unify HarMeme annotations from di-dimitrov/mmf JSONL splits."""
    data_dir = Path(data_dir)
    ann_dir  = data_dir / "annotations"
    img_dir  = data_dir / "images"

    if not ann_dir.is_dir():
        logger.error(f"HarMeme annotations not found at {ann_dir}")
        return []

    examples = []
    for split_file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = ann_dir / split_file
        if not path.exists():
            logger.warning(f"HarMeme: {path} not found, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                img_file = row.get("image", "")
                img_path = str(img_dir / img_file)
                label_str = (
                    row.get("labels", ["not harmful"])[0]
                    if isinstance(row.get("labels"), list)
                    else str(row.get("labels", "not harmful"))
                )
                hateful = label_str.lower() != "not harmful"
                examples.append({
                    "id":             row.get("id", Path(img_file).stem),
                    "image_path":     img_path,
                    "dataset":        "harmeme",
                    "text":           row.get("text", ""),
                    "hateful":        int(hateful),
                    "original_label": label_str,
                })

    logger.info(f"HarMeme: loaded {len(examples)} annotated examples")
    return examples


def load_mami(data_dir: str) -> List[Dict]:
    """
    Load MAMI annotations from the training split only.

    The MAMI test split ships without gold labels, so only the training
    annotation file (CSV or XLSX, containing file_name, misogynous,
    Text Transcription) is loaded; examples with no misogynous label are
    dropped.
    """
    data_dir   = Path(data_dir)
    img_dir    = data_dir / "images"
    ann_dir    = data_dir / "annotations"

    if not ann_dir.is_dir():
        logger.error(f"MAMI annotations not found at {ann_dir}")
        return []

    # Prefer any filename containing "train"; fall back to the first CSV/XLSX found.
    ann_file = None
    for candidate in sorted(ann_dir.iterdir()):
        if candidate.suffix in {".csv", ".xlsx", ".xls"}:
            if "train" in candidate.name.lower():
                ann_file = candidate
                break
    if ann_file is None:
        for candidate in sorted(ann_dir.iterdir()):
            if candidate.suffix in {".csv", ".xlsx", ".xls"}:
                ann_file = candidate
                break

    if ann_file is None:
        logger.error(f"No annotation file found in {ann_dir}")
        return []

    logger.info(f"MAMI: reading annotations from {ann_file}")
    if ann_file.suffix == ".csv":
        try:
            df = pd.read_csv(ann_file, sep=None, engine="python", quoting=3)  # QUOTE_NONE
        except Exception:
            df = pd.read_csv(ann_file, sep="\t", quoting=3, on_bad_lines="skip")
    else:
        df = pd.read_excel(ann_file)

    # Normalise to snake_case so column lookup is uniform across CSV/XLSX variants.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info(f"MAMI annotation columns: {list(df.columns)}")

    fname_col = next((c for c in df.columns if "file" in c), None)
    miso_col  = next((c for c in df.columns if "misogyn" in c), None)
    text_col  = next((c for c in df.columns if "text" in c or "transcri" in c), None)

    if fname_col is None:
        logger.error("MAMI: cannot find file_name column")
        return []

    examples = []
    for _, row in df.iterrows():
        img_file = str(row[fname_col]).strip()
        img_path = str(img_dir / img_file)

        if miso_col is not None:
            try:
                hateful = int(row[miso_col]) == 1
            except (ValueError, TypeError):
                hateful = False
        else:
            hateful = None

        if hateful is None:
            continue

        text      = str(row[text_col]).strip() if text_col else ""
        label_str = str(int(row[miso_col])) if miso_col is not None else "unknown"

        examples.append({
            "id":             Path(img_file).stem,
            "image_path":     img_path,
            "dataset":        "mami",
            "text":           text,
            "hateful":        int(hateful),
            "original_label": f"misogynous={label_str}",
        })

    logger.info(f"MAMI: loaded {len(examples)} labeled examples")
    return examples


def load_mmhs150k(data_dir: str) -> List[Dict]:
    """
    Load MMHS150K from MMHS150K_GT.json and resolve hateful labels by majority vote.

    An image is marked hateful if at least 2 of 3 annotators assigned a
    non-zero label; the predominant non-zero label determines the category string.
    Images whose file is absent on disk are silently skipped.
    """
    data_dir = Path(data_dir)
    img_dir  = data_dir / "images"
    gt_path  = data_dir / "annotations" / "MMHS150K_GT.json"

    if not gt_path.exists():
        logger.error(f"MMHS150K GT not found at {gt_path}")
        return []

    logger.info(f"MMHS150K: loading {gt_path} ...")
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    LABEL_MAP = {0: "NotHate", 1: "Racist", 2: "Sexist",
                 3: "Homophobe", 4: "Religion", 5: "OtherHate"}

    examples = []
    missing  = 0
    for tweet_id, entry in gt.items():
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_dir / f"{tweet_id}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break
        if img_path is None:
            missing += 1
            continue

        labels     = entry.get("labels", [0, 0, 0])
        hate_votes = sum(1 for l in labels if l != 0)
        hateful    = hate_votes >= 2

        non_zero = [l for l in labels if l != 0]
        if non_zero:
            from collections import Counter
            most_common_label = Counter(non_zero).most_common(1)[0][0]
            label_str = LABEL_MAP.get(most_common_label, str(most_common_label))
        else:
            label_str = "NotHate"

        examples.append({
            "id":             tweet_id,
            "image_path":     img_path,
            "dataset":        "mmhs150k",
            "text":           entry.get("tweet_text", ""),
            "hateful":        int(hateful),
            "original_label": label_str,
        })

    if missing:
        logger.warning(f"MMHS150K: {missing} entries skipped (image file not found)")
    logger.info(f"MMHS150K: loaded {len(examples)} examples (with images)")
    return examples


# ── Stratified split ──────────────────────────────────────────────────────

def stratified_split(
    examples: List[Dict],
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Partition examples into train/val/test while preserving (dataset, hateful)
    proportions; ensures every split has balanced source-dataset and
    hate-label representation.
    """
    random.seed(seed)

    groups: Dict[tuple, List[Dict]] = defaultdict(list)
    for ex in examples:
        key = (ex["dataset"], ex["hateful"])
        groups[key].append(ex)

    train, val, test = [], [], []
    for key, group in sorted(groups.items()):
        random.shuffle(group)
        n       = len(group)
        n_val   = max(1, round(n * val_ratio))
        n_test  = max(1, round(n * (1 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test

        train += group[:n_train]
        val   += group[n_train:n_train + n_val]
        test  += group[n_train + n_val:]

        logger.info(
            f"  {key[0]:<12} hateful={key[1]}  "
            f"total={n}  train={n_train}  val={n_val}  test={n_test}"
        )

    return train, val, test


# ── Statistics helpers ────────────────────────────────────────────────────

def compute_stats(examples: List[Dict], split_name: str) -> Dict:
    stats = {"split": split_name, "total": len(examples), "by_dataset": {}}
    by_ds = defaultdict(lambda: {"total": 0, "hateful": 0})
    for ex in examples:
        by_ds[ex["dataset"]]["total"]   += 1
        by_ds[ex["dataset"]]["hateful"] += ex["hateful"]
    stats["by_dataset"] = {
        ds: {**counts, "hateful_pct": round(100 * counts["hateful"] / max(counts["total"], 1), 1)}
        for ds, counts in by_ds.items()
    }
    return stats


def print_stats(train, val, test):
    print("\n" + "=" * 70)
    print("  UNIFIED SPLIT SUMMARY  (after Stage 0 filtering)")
    print("=" * 70)
    for split_name, examples in [("train", train), ("val", val), ("test", test)]:
        total   = len(examples)
        hateful = sum(e["hateful"] for e in examples)
        print(f"\n  {split_name.upper()} — {total} examples ({100*hateful//max(total,1)}% hateful)")
        by_ds = defaultdict(lambda: {"total": 0, "hateful": 0})
        for ex in examples:
            by_ds[ex["dataset"]]["total"]   += 1
            by_ds[ex["dataset"]]["hateful"] += ex["hateful"]
        for ds, c in sorted(by_ds.items()):
            pct_h = 100 * c["hateful"] // max(c["total"], 1)
            print(f"    {ds:<12}  {c['total']:>6} imgs  {pct_h:>3}% hateful")
    print("=" * 70 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build unified 80/10/10 stratified splits (run AFTER Stage 0)"
    )
    parser.add_argument("--harmeme_dir",  type=str, default="/scratch/hmr_data/harmeme")
    parser.add_argument("--mami_dir",     type=str, default="/scratch/hmr_data/mami")
    parser.add_argument("--mmhs150k_dir", type=str, default="/scratch/hmr_data/mmhs150k")

    parser.add_argument("--harmeme_manifest",  type=str,
                        default="/scratch/hmr_data/harmeme/manifest.csv",
                        help="Stage 0 manifest for HarMeme")
    parser.add_argument("--mami_manifest",     type=str,
                        default="/scratch/hmr_data/mami/manifest.csv",
                        help="Stage 0 manifest for MAMI")
    parser.add_argument("--mmhs150k_manifest", type=str,
                        default="/scratch/hmr_data/mmhs150k/manifest.csv",
                        help="Stage 0 manifest for MMHS150K")
    parser.add_argument("--skip_manifest_filter", action="store_true",
                        help="Ignore Stage 0 manifests and use all annotated examples (not recommended)")

    # Split params
    parser.add_argument("--output_dir",   type=str, default="/scratch/hmr_data/unified_splits")
    parser.add_argument("--train_ratio",  type=float, default=0.80)
    parser.add_argument("--val_ratio",    type=float, default=0.10)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--debug",        action="store_true",
                        help="Use tiny subset of each dataset (for testing)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    print("\n" + "=" * 70)
    print("  Build Unified Splits  (post-Stage-0)")
    print(f"  Split:  {int(args.train_ratio*100)} / {int(args.val_ratio*100)} / "
          f"{int((1-args.train_ratio-args.val_ratio)*100)}  (train/val/test)")
    print(f"  Output: {args.output_dir}")
    if args.skip_manifest_filter:
        print("  Stage 0 filter: SKIPPED (--skip_manifest_filter)")
    else:
        print(f"  HarMeme manifest:  {args.harmeme_manifest}")
        print(f"  MAMI manifest:     {args.mami_manifest}")
        print(f"  MMHS150K manifest: {args.mmhs150k_manifest}")
    print("=" * 70 + "\n")

    if args.skip_manifest_filter:
        harmeme_kept = mami_kept = mmhs_kept = None
    else:
        logger.info("Loading Stage 0 manifests...")
        harmeme_kept = load_kept_paths(args.harmeme_manifest)
        mami_kept    = load_kept_paths(args.mami_manifest)
        mmhs_kept    = load_kept_paths(args.mmhs150k_manifest)

    all_examples = []

    harmeme = load_harmeme(args.harmeme_dir)
    if not harmeme:
        logger.warning("HarMeme returned no examples — check the path.")
    else:
        harmeme = apply_manifest_filter(harmeme, harmeme_kept, "HarMeme")
    all_examples += harmeme

    mami = load_mami(args.mami_dir)
    if not mami:
        logger.warning("MAMI returned no examples — check the path.")
    else:
        mami = apply_manifest_filter(mami, mami_kept, "MAMI")
    all_examples += mami

    mmhs = load_mmhs150k(args.mmhs150k_dir)
    if not mmhs:
        logger.warning("MMHS150K returned no examples — check the path.")
    else:
        mmhs = apply_manifest_filter(mmhs, mmhs_kept, "MMHS150K")
    all_examples += mmhs

    if not all_examples:
        logger.error("No examples loaded from any dataset. Exiting.")
        sys.exit(1)

    logger.info(f"\nTotal examples after Stage 0 filtering: {len(all_examples)}")
    by_ds: Dict[str, int] = defaultdict(int)
    for e in all_examples:
        by_ds[e["dataset"]] += 1
    for ds, n in sorted(by_ds.items()):
        logger.info(f"  {ds:<12} {n} examples")

    if args.debug:
        random.seed(args.seed)
        random.shuffle(all_examples)
        all_examples = all_examples[:300]
        logger.warning(f"DEBUG: truncated to {len(all_examples)} examples")

    logger.info("\nCreating stratified splits:")
    train, val, test = stratified_split(
        all_examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        for ex in split_data:
            ex["split"] = split_name
        df = pd.DataFrame(split_data, columns=["id", "image_path", "dataset", "text",
                                                "hateful", "original_label", "split"])
        out_path = output_dir / f"unified_{split_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Written {len(df)} examples → {out_path}")

    print_stats(train, val, test)

    stats = {
        "train": compute_stats(train, "train"),
        "val":   compute_stats(val,   "val"),
        "test":  compute_stats(test,  "test"),
    }
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats written to {stats_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
