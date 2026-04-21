"""
Extract dataset statistics from existing train.jsonl / val.jsonl files.

Use this to generate dataset_statistics.json WITHOUT rerunning the full
build_stage2_dataset.py pipeline. Just point it at the directory that already
has train.jsonl and val.jsonl on scratch.

Usage:
    python analysis/extract_dataset_stats.py \\
        --dataset_dir /scratch/hmr_stage2_dataset \\
        --output_dir  /scratch/hmr_stage2_dataset      # (same dir is fine)
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def compute_stats(examples, split_name: str) -> dict:
    target_group_counts = Counter(e.get("target_group", "null") for e in examples)
    attack_type_counts  = Counter(e.get("attack_type",  "null") for e in examples)
    dataset_counts      = Counter(e.get("dataset",      "unknown") for e in examples)
    return {
        "split": split_name,
        "total": len(examples),
        "target_group_distribution": dict(target_group_counts.most_common()),
        "attack_type_distribution":  dict(attack_type_counts.most_common()),
        "dataset_source_distribution": dict(dataset_counts.most_common()),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract dataset statistics from existing JSONL splits")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing train.jsonl and val.jsonl")
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Where to write dataset_statistics.json (default: same as dataset_dir)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.jsonl"
    val_path   = dataset_dir / "val.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found.")
        return 1

    print(f"Loading {train_path}...")
    train_data = load_jsonl(train_path)
    val_data   = load_jsonl(val_path) if val_path.exists() else []
    all_data   = train_data + val_data

    train_stats = compute_stats(train_data, "train")
    val_stats   = compute_stats(val_data,   "val")
    combined    = compute_stats(all_data,   "combined")

    stats = {
        "note": "Extracted post-hoc from existing train.jsonl / val.jsonl",
        "counts": {
            "train_samples":    len(train_data),
            "val_samples":      len(val_data),
            "total":            len(all_data),
        },
        "combined":   combined,
        "train_split": train_stats,
        "val_split":   val_stats,
    }

    out_path = output_dir / "dataset_statistics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ── pretty print ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"  Train: {len(train_data):,}  |  Val: {len(val_data):,}  |  Total: {len(all_data):,}")

    print("\n  Target group distribution (combined):")
    for tg, n in combined["target_group_distribution"].items():
        pct = 100 * n / max(1, len(all_data))
        print(f"    {tg:<28} {n:>5}  ({pct:.1f}%)")

    print("\n  Attack type distribution (combined):")
    for at, n in combined["attack_type_distribution"].items():
        pct = 100 * n / max(1, len(all_data))
        print(f"    {at:<28} {n:>5}  ({pct:.1f}%)")

    print("\n  Dataset source distribution (combined):")
    for ds, n in combined["dataset_source_distribution"].items():
        pct = 100 * n / max(1, len(all_data))
        print(f"    {ds:<28} {n:>5}  ({pct:.1f}%)")

    print(f"\n  Saved: {out_path}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
