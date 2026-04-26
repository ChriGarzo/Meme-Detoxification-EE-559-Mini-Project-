"""
Merge Stage 1 sharded pseudo-rewrite JSONL files into a single deduplicated file.

Example:
  python inference/merge_stage1_rewrites_shards.py \
    --dataset train \
    --input_dir /scratch/hmr_stage1_output
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.split()).strip()


def _normalize_image_key(path: str) -> str:
    if not isinstance(path, str):
        return ""
    return os.path.normpath(path).replace("\\", "/")


def _record_key(record: Dict) -> str:
    rid = str(record.get("id", "")).strip()
    if rid:
        return f"id::{rid}"

    image_key = _normalize_image_key(str(record.get("image_path", "")))
    orig = _normalize_text(str(record.get("original_text", "")))
    if image_key or orig:
        return f"imgtxt::{image_key}::{orig}"

    # Last-resort key from rewrite content.
    rewrite = _normalize_text(str(record.get("pseudo_rewrite", "")))
    return f"rw::{rewrite}" if rewrite else ""


def merge_shards(
    dataset: str,
    input_dir: str,
    output_path: str,
    num_shards: int,
) -> Tuple[int, int, int, int, List[str]]:
    pattern = os.path.join(
        input_dir,
        f"{dataset}_pseudo_rewrites_shard*of{num_shards:02d}.jsonl",
    )
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No shard files found with pattern: {pattern}")

    total_lines = 0
    invalid_lines = 0
    kept_rows = 0
    duplicate_rows = 0

    by_key: Dict[str, Dict] = {}

    for fp in files:
        with open(fp, "r", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                total_lines += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    invalid_lines += 1
                    continue

                if not isinstance(obj, dict):
                    invalid_lines += 1
                    continue

                key = _record_key(obj)
                if not key:
                    invalid_lines += 1
                    continue

                if key in by_key:
                    duplicate_rows += 1
                    # Keep first occurrence for deterministic behavior.
                    continue

                by_key[key] = obj
                kept_rows += 1

    merged_rows = [by_key[k] for k in sorted(by_key.keys())]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as out:
        for row in merged_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    return total_lines, invalid_lines, kept_rows, duplicate_rows, files


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Stage 1 pseudo-rewrite shard files")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset prefix (e.g., train)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing shard JSONLs")
    parser.add_argument("--num_shards", type=int, default=8, help="Expected shard count")
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Output JSONL path (default: <input_dir>/<dataset>_pseudo_rewrites_merged.jsonl)",
    )

    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")

    output_path = args.output_path.strip() or os.path.join(
        args.input_dir,
        f"{args.dataset}_pseudo_rewrites_merged.jsonl",
    )

    total_lines, invalid_lines, kept_rows, duplicate_rows, files = merge_shards(
        dataset=args.dataset,
        input_dir=args.input_dir,
        output_path=output_path,
        num_shards=args.num_shards,
    )

    print("Merged pseudo-rewrite shards")
    print(f"Input files: {len(files)}")
    for fp in files:
        print(f"  - {fp}")
    print(f"Read JSONL lines: {total_lines}")
    print(f"Invalid lines skipped: {invalid_lines}")
    print(f"Duplicate rows skipped: {duplicate_rows}")
    print(f"Rows written: {kept_rows}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
