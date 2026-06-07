"""
Stage 4 trainer: ExplanationProxy MLP — CLIP features to BART soft encoder tokens.

Pipeline position: AFTER train_stage2.py (full condition) has completed.

Inputs : Stage 1 pseudo-rewrite JSONL (image_path lookup) +
         Stage 2 train/val JSONL (original_text, target_group,
         visual_evidence, implicit_meaning).
Outputs: <output_dir>/best_proxy.pt   — best-val-MSE proxy checkpoint
         <output_dir>/proxy_config.json
         <output_dir>/training_history.json
         <output_dir>/eval_results.json

Design decisions:
- The proxy predicts `num_soft_tokens` BART encoder hidden states from a
  1536-dim CLIP feature vector (image_embed ‖ text_embed).  The training
  target is the h_full BART encoder output for the full-condition prompt,
  minimising MSE.  This distills structured explanation grounding into a
  compact representation that can replace the LLaVA stage at inference.
- BART is loaded from the full-condition Stage 2 checkpoint and frozen;
  it is only used to produce soft-token targets, never updated.

Usage (cluster):
    python training/train_proxy.py \\
        --stage1_output_dir    /scratch/hmr_stage1_output \\
        --stage2_dataset_dir   /scratch/hmr_stage2_dataset \\
        --bart_checkpoint_dir  /scratch/hmr_stage2_full_checkpoint \\
        --output_dir           /scratch/hmr_proxy_checkpoint \\
        --hf_cache             /scratch/hf_cache \\
        --num_train_epochs 20 \\
        --batch_size 64 \\
        --learning_rate 1e-3 \\
        --input_format legacy \\
        --seed 42
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.debug import DEBUG_CONFIG, is_debug_mode, set_seeds, make_debug_dataset

logger = logging.getLogger(__name__)


# ── Data loading helpers ──────────────────────────────────────────────────────

def _build_stage1_image_index(stage1_output_dir: str) -> Dict[str, str]:
    """Build {dataset::id: image_path} lookup from Stage 1 pseudo-rewrite JSONL files."""
    root = Path(stage1_output_dir)
    if not root.exists():
        logger.warning(f"Stage 1 output dir not found: {root}")
        return {}

    image_index: Dict[str, str] = {}
    files = sorted(root.rglob("*_pseudo_rewrites.jsonl"))
    if not files:
        logger.warning(f"No *_pseudo_rewrites.jsonl files found under {root}")
        return {}

    for path in files:
        dataset_name = path.stem.replace("_pseudo_rewrites", "")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ex_id = rec.get("id")
                img = rec.get("image_path")
                if not ex_id or not img:
                    continue

                # Dataset-scoped key prevents collisions across datasets.
                image_index[f"{dataset_name}::{ex_id}"] = img
                # Bare-ID fallback for older records that lack a dataset field.
                image_index.setdefault(str(ex_id), img)

    logger.info(f"Indexed {len(image_index)} stage1 image references")
    return image_index

def load_stage2_dataset(stage1_output_dir: str, dataset_dir: str, debug: bool) -> tuple:
    """Load Stage 2 train/val JSONL and attach image paths from the Stage 1 index."""
    if debug:
        raw = make_debug_dataset(n=DEBUG_CONFIG["max_samples"])
        examples = [
            {
                "image_path":      e["image_path"],
                "original_text":   e["text"],
                "target_group":    e.get("target_group"),
                "visual_evidence": e.get("visual_evidence"),
                "implicit_meaning": (e.get("explanation") or {}).get("implicit_meaning"),
            }
            for e in raw if e.get("label") == 1    # proxy is trained only on hateful examples (have pseudo-rewrites)
        ]
        split = max(1, len(examples) - 2)
        return examples[:split], examples[split:]

    d = Path(dataset_dir)
    train_path, val_path = d / "train.jsonl", d / "val.jsonl"

    def _load(p):
        lines = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines

    train = _load(train_path)
    val   = _load(val_path) if val_path.exists() else train[:max(1, len(train) // 10)]

    stage1_index = _build_stage1_image_index(stage1_output_dir)

    def _attach_image_path(examples: List[Dict], split_name: str) -> List[Dict]:
        enriched = []
        missing = 0
        for ex in examples:
            if ex.get("image_path"):
                enriched.append(ex)
                continue

            ex_id = ex.get("id")
            dataset = ex.get("dataset")
            scoped_key = f"{dataset}::{ex_id}" if dataset and ex_id else None

            image_path = None
            if scoped_key:
                image_path = stage1_index.get(scoped_key)
            if image_path is None and ex_id is not None:
                image_path = stage1_index.get(str(ex_id))

            if image_path:
                ex["image_path"] = image_path
                enriched.append(ex)
            else:
                missing += 1

        if missing:
            logger.warning(
                f"{split_name}: dropped {missing} examples without image_path after Stage 1 lookup"
            )
        return enriched

    train = _attach_image_path(train, "train")
    val = _attach_image_path(val, "val")

    if not train:
        logger.error("No train examples with valid image_path were found for proxy training")
        sys.exit(1)
    if not val:
        logger.error("No val examples with valid image_path were found for proxy training")
        sys.exit(1)

    logger.info(f"Loaded {len(train)} train, {len(val)} val examples for proxy training")
    return train, val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Train ExplanationProxy network")
    parser.add_argument("--stage1_output_dir",   type=str, required=True,
                        help="Root dir with per-dataset Stage 1 JSONL outputs (for image_path lookup)")
    parser.add_argument("--stage2_dataset_dir",  type=str, required=True,
                        help="Dir with train.jsonl/val.jsonl from build_stage2_dataset.py")
    parser.add_argument("--bart_checkpoint_dir", type=str, required=True,
                        help="Phase 2 (full condition) BART checkpoint (for BART hidden state targets)")
    parser.add_argument("--output_dir",          type=str, required=True)
    parser.add_argument("--hf_cache",            type=str, default=None)
    parser.add_argument("--num_train_epochs",    type=int,   default=20)
    parser.add_argument("--batch_size",          type=int,   default=64)
    parser.add_argument("--learning_rate",       type=float, default=1e-3)
    parser.add_argument("--num_soft_tokens",     type=int,   default=16,
                        help="Number of proxy-predicted BART encoder soft tokens")
    parser.add_argument("--input_format",
                        type=str,
                        default="legacy",
                        choices=["legacy", "explicit_detox"],
                        help="BART full-condition prompt format used as proxy target")
    parser.add_argument("--task_prefix", type=str, default="",
                        help="Optional prefix prepended to proxy target prompts")
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    debug = is_debug_mode(args)
    set_seeds(args.seed)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Stage 4: Proxy Network Training")
    print(f"  Stage 1 dir:  {args.stage1_output_dir}")
    print(f"  Stage 2 dir:  {args.stage2_dataset_dir}")
    print(f"  BART ckpt:    {args.bart_checkpoint_dir}")
    print(f"  Output:       {args.output_dir}")
    print(f"  Epochs:       {args.num_train_epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  LR:           {args.learning_rate}")
    print(f"  Soft tokens:  {args.num_soft_tokens}")
    print(f"  Input fmt:    {args.input_format}")
    if args.task_prefix.strip():
        print(f"  Task prefix:  {args.task_prefix.strip()}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  Device:       CPU (no GPU found)")
    print(f"{'='*60}\n")
    logger.info(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_examples, val_examples = load_stage2_dataset(
        args.stage1_output_dir,
        args.stage2_dataset_dir,
        debug,
    )

    if debug:
        num_epochs  = DEBUG_CONFIG["proxy_epochs"]
        batch_size  = DEBUG_CONFIG["proxy_batch_size"]
        bart_hidden = DEBUG_CONFIG["bart_hidden_size"]
        bart_model  = DEBUG_CONFIG["stage2_model"]    # bart-base in debug mode
    else:
        num_epochs  = args.num_train_epochs
        batch_size  = args.batch_size
        bart_hidden = 1024           # BART-large hidden size
        bart_model  = args.bart_checkpoint_dir

    logger.info(f"epochs={num_epochs}, batch={batch_size}, bart_hidden={bart_hidden}")

    # ── Load BART (frozen) ────────────────────────────────────────────────────
    # BART is the full-condition Stage 2 checkpoint; it is used only to extract
    # h_full encoder hidden states that serve as the MSE regression targets for
    # the proxy.  No gradients flow through BART during proxy training.
    from models.rewriter import MemeRewriter
    rewriter = MemeRewriter(
        model_name=bart_model,
        cache_dir=args.hf_cache,
        device=device,
    )
    rewriter.load_model()
    for p in rewriter.model.parameters():
        p.requires_grad = False
    logger.info("BART model loaded and frozen")

    # ── ExplanationProxyTrainer (also loads CLIP) ─────────────────────────────
    from models.proxy import ExplanationProxyTrainer
    trainer = ExplanationProxyTrainer(
        rewriter=rewriter,
        clip_model_name="openai/clip-vit-large-patch14",
        cache_dir=args.hf_cache,
        device=device,
        num_soft_tokens=args.num_soft_tokens,
        input_format=args.input_format,
        task_prefix=args.task_prefix,
    )

    # ── Unpack examples into parallel lists for the trainer API ───────────────
    def _unpack(examples):
        images   = [e["image_path"]      for e in examples]
        texts    = [e["original_text"]   for e in examples]
        tgs      = [e.get("target_group")      for e in examples]
        ves      = [e.get("visual_evidence")   for e in examples]
        ims      = [e.get("implicit_meaning")  for e in examples]
        return images, texts, tgs, ves, ims

    tr_images, tr_texts, tr_tgs, tr_ves, tr_ims = _unpack(train_examples)
    va_images, va_texts, va_tgs, va_ves, va_ims = _unpack(val_examples)

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    history = trainer.train(
        images=tr_images,
        texts=tr_texts,
        target_groups=tr_tgs,
        visual_evidences=tr_ves,
        implicit_meanings=tr_ims,
        val_images=va_images,
        val_texts=va_texts,
        val_target_groups=va_tgs,
        val_visual_evidences=va_ves,
        val_implicit_meanings=va_ims,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.output_dir,
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    proxy_config = {
        "architecture": "clip_to_bart_soft_encoder_tokens",
        "num_soft_tokens": args.num_soft_tokens,
        "bart_hidden_size": rewriter.hidden_size,
        "clip_model": "openai/clip-vit-large-patch14",
        "bart_checkpoint_dir": args.bart_checkpoint_dir,
        "input_format": args.input_format,
        "task_prefix": args.task_prefix,
    }
    config_path = Path(args.output_dir) / "proxy_config.json"
    with open(config_path, "w") as f:
        json.dump(proxy_config, f, indent=2)

    logger.info(f"Proxy training complete. Checkpoint saved to {args.output_dir}")
    print(f"\n{'='*60}")
    print(f"  Proxy Network Training COMPLETE")
    print(f"  Checkpoint: {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Final validation evaluation ───────────────────────────────────────────
    logger.info("Running final proxy evaluation on validation set...")
    eval_results = trainer.evaluate(
        images=va_images,
        texts=va_texts,
        target_groups=va_tgs,
        visual_evidences=va_ves,
        implicit_meanings=va_ims,
    )
    logger.info(f"Proxy eval MSE: {eval_results['mse_loss']:.6f}")

    results_path = Path(args.output_dir) / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()
