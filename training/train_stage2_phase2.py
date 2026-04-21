"""
Stage 2: BART meme conditioning fine-tune (with optional ParaDetox mixing).

Fine-tunes facebook/bart-large directly on pseudo-labelled meme pairs generated
by Stage 1 LLaVA.  A fraction of clean ParaDetox (s-nlp/paradetox) pairs is
mixed into the training set to give the model a detoxification prior without a
separate warm-up phase that can cause autoregressive generation collapse.

Run once per conditioning condition (full / target_only / attack_only / none).

Pipeline position: AFTER build_stage2_dataset.

Usage (cluster):
    python training/train_stage2_phase2.py \
        --condition full \
        --dataset_dir           /scratch/hmr_stage2_dataset \
        --output_dir            /scratch/hmr_stage2_phase2_full_checkpoint \
        --hf_cache              /scratch/hf_cache \
        --num_train_epochs 5 \
        --per_device_train_batch_size 8 \
        --learning_rate 2e-5 \
        --warmup_steps 50 \
        --weight_decay 0.01 \
        --paradetox_mix_ratio 0.2 \
        --seed 42
"""

import argparse
import json
import logging
import os
import random
import sys
import inspect
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.debug import DEBUG_CONFIG, is_debug_mode, set_seeds, make_debug_dataset

logger = logging.getLogger(__name__)

CONDITION_TYPE = Literal["full", "target_only", "attack_only", "none"]


# ---------------------------------------------------------------------------
# Input formatting  (mirrors MemeRewriter.format_input)
# ---------------------------------------------------------------------------

def format_input(
    original_text: str,
    target_group: str,
    attack_type: str,
    implicit_meaning: str,
    condition: str,
) -> str:
    """Build BART encoder input string for the given condition."""
    tg = target_group   or "null"
    at = attack_type    or "null"
    im = implicit_meaning or "null"

    if condition == "full":
        prefix = f"[T: {tg}] [A: {at}] [M: {im}]"
    elif condition == "target_only":
        prefix = f"[T: {tg}] [A: null] [M: null]"
    elif condition == "attack_only":
        prefix = f"[T: null] [A: {at}] [M: null]"
    else:  # "none"
        prefix = "[T: null] [A: null] [M: null]"

    return f"{prefix} | {original_text}"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MemeRewriteDataset(Dataset):
    """Meme pseudo-rewrite pairs with condition-specific BART encoder inputs."""

    def __init__(self, examples: List[Dict], tokenizer, condition: str,
                 max_input_length: int = 128, max_target_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.condition = condition
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        try:
            tokenizer_call_params = inspect.signature(self.tokenizer.__call__).parameters
        except (TypeError, ValueError):
            tokenizer_call_params = {}
        self._supports_text_target = "text_target" in tokenizer_call_params

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        input_text = format_input(
            original_text=ex.get("original_text", ""),
            target_group=ex.get("target_group"),
            attack_type=ex.get("attack_type"),
            implicit_meaning=ex.get("implicit_meaning"),
            condition=self.condition,
        )
        target_text = ex.get("target_text", "")

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        if self._supports_text_target:
            labels = self.tokenizer(
                text_target=target_text,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        elif hasattr(self.tokenizer, "as_target_tokenizer"):
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    target_text,
                    max_length=self.max_target_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
        else:
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_paradetox_mix(
    hf_cache: Optional[str],
    mix_ratio: float,
    n_meme_train: int,
    seed: int,
    debug: bool,
) -> List[Dict]:
    """Load a fixed, seeded sample of ParaDetox pairs for data mixing.

    Returns a list of dicts compatible with MemeRewriteDataset.  All
    conditioning fields are None so format_input() will produce
    "[T: null] [A: null] [M: null] | <toxic_text>", regardless of which
    condition is being trained.  This teaches the model that the null-
    conditioned format always means "detoxify", acting as a clean prior
    without the overfitting risk of a separate warm-up phase.

    Args:
        hf_cache:     HuggingFace cache directory (may be None).
        mix_ratio:    Target fraction of ParaDetox in the combined training
                      set.  E.g. 0.2 → 20 % ParaDetox, 80 % meme.
        n_meme_train: Number of meme training examples (before mixing).
        seed:         Random seed for reproducible sampling.
        debug:        If True, return a tiny subset.

    Returns:
        List of example dicts ready to append to train_examples.
    """
    if mix_ratio <= 0.0:
        return []

    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' package is required for ParaDetox mixing. "
            "Install it with: pip install datasets"
        )
        sys.exit(1)

    logger.info(
        f"Loading s-nlp/paradetox for {mix_ratio:.0%} data mixing "
        f"(target: {int(n_meme_train * mix_ratio / (1.0 - mix_ratio))} examples)..."
    )
    ds = hf_load_dataset("s-nlp/paradetox", cache_dir=hf_cache)

    examples = []
    for row in ds["train"]:
        toxic   = (row.get("en_toxic_comment")   or "").strip()
        neutral = (row.get("en_neutral_comment")  or "").strip()

        # --- Quality filters ---
        if not toxic or not neutral:
            continue
        if toxic == neutral:                       # not actually detoxified
            continue
        if not (10 <= len(toxic) <= 200):          # stay in meme-like length range
            continue
        if not (5 <= len(neutral) <= 200):
            continue

        examples.append({
            "original_text":    toxic,
            "target_text":      neutral,
            "target_group":     None,              # → "null" in format_input
            "attack_type":      None,              # → "null" in format_input
            "implicit_meaning": None,              # → "null" in format_input
            "source":           "paradetox",
        })

    logger.info(f"ParaDetox: {len(ds['train'])} raw rows → {len(examples)} after quality filtering")

    if debug:
        return examples[:10]

    # Compute how many examples give mix_ratio of the combined set:
    #   n_para / (n_meme + n_para) = mix_ratio
    #   ⟹  n_para = n_meme * mix_ratio / (1 - mix_ratio)
    n_target = int(n_meme_train * mix_ratio / (1.0 - mix_ratio))
    n_target = min(n_target, len(examples))

    rng = random.Random(seed)
    sampled = rng.sample(examples, n_target)

    actual_ratio = n_target / (n_meme_train + n_target)
    logger.info(
        f"ParaDetox mix: {n_target} examples sampled "
        f"({actual_ratio:.1%} of combined train set — "
        f"meme: {n_meme_train}, paradetox: {n_target}, "
        f"total: {n_meme_train + n_target})"
    )
    return sampled


def load_dataset(dataset_dir: str, debug: bool) -> tuple:
    """Load train.jsonl and val.jsonl from the Stage 2 dataset directory."""
    if debug:
        raw = make_debug_dataset(n=DEBUG_CONFIG["max_samples"])
        # Convert debug format to training format
        examples = [
            {
                "original_text":   e["text"],
                "target_text":     e.get("explanation", {}).get("implicit_meaning", e["text"]),
                "target_group":    e.get("target_group"),
                "attack_type":     e.get("attack_type"),
                "implicit_meaning": (e.get("explanation") or {}).get("implicit_meaning"),
                "dataset":         "debug",
            }
            for e in raw
        ]
        split = max(1, len(examples) - 2)
        return examples[:split], examples[split:]

    d = Path(dataset_dir)
    train_path = d / "train.jsonl"
    val_path   = d / "val.jsonl"

    if not train_path.exists():
        logger.error(f"train.jsonl not found at {train_path}. Run build_stage2_dataset.py first.")
        sys.exit(1)

    train_examples = load_jsonl(train_path)
    val_examples   = load_jsonl(val_path) if val_path.exists() else train_examples[:max(1, len(train_examples) // 10)]

    logger.info(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")
    return train_examples, val_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Phase 2: BART meme conditioning fine-tune")
    parser.add_argument("--condition",
                        type=str, required=True,
                        choices=["full", "target_only", "attack_only", "none"],
                        help="Conditioning strategy for the ablation study")
    parser.add_argument("--phase1_checkpoint_dir", type=str, default=None,
                        help="Directory of the Phase 1 checkpoint. If not set, starts from --base_model directly.")
    parser.add_argument("--base_model", type=str, default="facebook/bart-large",
                        help="Base model to use when --phase1_checkpoint_dir is not provided.")
    parser.add_argument("--dataset_dir",           type=str, required=True,
                        help="Directory with train.jsonl/val.jsonl (output of build_stage2_dataset.py)")
    parser.add_argument("--output_dir",            type=str, required=True)
    parser.add_argument("--hf_cache",              type=str, default=None)
    parser.add_argument("--num_train_epochs",            type=int,   default=5)
    parser.add_argument("--per_device_train_batch_size", type=int,   default=8)
    parser.add_argument("--learning_rate",               type=float, default=2e-5)
    parser.add_argument("--warmup_steps",                type=int,   default=50)
    parser.add_argument("--weight_decay",                type=float, default=0.01)
    parser.add_argument("--seed",                        type=int,   default=42)
    parser.add_argument(
        "--paradetox_mix_ratio", type=float, default=0.2,
        help=(
            "Fraction of the combined training set that should come from "
            "s-nlp/paradetox.  0.2 = 20%% ParaDetox / 80%% meme. "
            "Set to 0.0 to disable mixing entirely."
        ),
    )
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

    # -----------------------------------------------------------------------
    # Model / hyperparameters
    # -----------------------------------------------------------------------
    # Phase 2 starts from Phase 1 checkpoint if provided, else directly from base model
    checkpoint = args.phase1_checkpoint_dir if args.phase1_checkpoint_dir else args.base_model
    logger.info(f"Starting from: {checkpoint}")
    num_epochs  = DEBUG_CONFIG["num_train_epochs"] if debug else args.num_train_epochs
    train_batch = DEBUG_CONFIG["per_device_train_batch_size"] if debug else args.per_device_train_batch_size
    max_steps   = DEBUG_CONFIG["max_steps"] if debug else -1
    save_steps  = DEBUG_CONFIG["save_steps"] if debug else 200
    eval_steps  = DEBUG_CONFIG["eval_steps"] if debug else 200
    use_fp16    = False
    use_bf16    = False
    if (not debug) and torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            use_fp16 = True
    precision_mode = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")

    print(f"\n{'='*60}")
    print(f"  Stage 2 Phase 2: BART Meme Fine-tuning")
    print(f"  Condition:  {args.condition}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Epochs:     {num_epochs}")
    print(f"  Batch size: {train_batch}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Precision:  {precision_mode}")
    print(f"  ParaDetox:  {args.paradetox_mix_ratio:.0%} mix ratio")
    print(f"  Output:     {args.output_dir}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  Device:     CPU (no GPU found)")
    print(f"{'='*60}\n")
    logger.info(
        f"Condition: {args.condition} | checkpoint: {checkpoint} | precision: {precision_mode}"
    )

    try:
        from transformers import (
            BartForConditionalGeneration,
            BartTokenizer,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            DataCollatorForSeq2Seq,
        )
        import evaluate as hf_evaluate
    except ImportError as e:
        logger.error(f"Missing package: {e}. Install: pip install transformers evaluate")
        sys.exit(1)

    tokenizer = BartTokenizer.from_pretrained(checkpoint, cache_dir=args.hf_cache)
    model     = BartForConditionalGeneration.from_pretrained(checkpoint, cache_dir=args.hf_cache)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_examples, val_examples = load_dataset(args.dataset_dir, debug)

    # --- ParaDetox mixing ---
    # Load a seeded sample of clean ParaDetox pairs and append to train.
    # ParaDetox examples have no conditioning fields (all None → "null"),
    # so format_input always produces "[T: null] [A: null] [M: null] | text"
    # for them regardless of which condition is being trained.
    # The val set is kept pure (meme only) — evaluation is on the target task.
    paradetox_examples = load_paradetox_mix(
        hf_cache=args.hf_cache,
        mix_ratio=args.paradetox_mix_ratio,
        n_meme_train=len(train_examples),
        seed=args.seed,
        debug=debug,
    )
    n_meme_train = len(train_examples)
    n_paradetox_train = len(paradetox_examples)
    if paradetox_examples:
        combined = train_examples + paradetox_examples
        random.seed(args.seed)
        random.shuffle(combined)
        train_examples = combined
        logger.info(
            f"Combined training set: {n_meme_train} meme + {n_paradetox_train} ParaDetox "
            f"= {len(train_examples)} total"
        )

    train_dataset = MemeRewriteDataset(train_examples, tokenizer, args.condition)
    val_dataset   = MemeRewriteDataset(val_examples,   tokenizer, args.condition)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    rouge = hf_evaluate.load("rouge")

    # Load toxicity classifier once — reused at every eval step
    logger.info("Loading toxicity classifier (s-nlp/roberta_toxicity_classifier)...")
    from transformers import pipeline as hf_pipeline
    toxicity_pipe = hf_pipeline(
        "text-classification",
        model="s-nlp/roberta_toxicity_classifier",
        device=0 if torch.cuda.is_available() else -1,
        cache_dir=args.hf_cache,
        batch_size=32,
        truncation=True,
        max_length=128,
    )
    logger.info("Toxicity classifier loaded.")

    def compute_sta_score(texts):
        """Proportion of texts classified as non-toxic (neutral)."""
        non_toxic = 0
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            try:
                results = toxicity_pipe(batch)
                non_toxic += sum(1 for r in results if r["label"].lower() != "toxic")
            except Exception as e:
                logger.warning(f"STA batch failed: {e}")
        return round(non_toxic / max(1, len(texts)), 4)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        metrics = {k: round(v, 4) for k, v in result.items()}
        # STA — is the model actually generating non-toxic text?
        metrics["sta"] = compute_sta_score(decoded_preds)
        logger.info(f"  eval STA: {metrics['sta']:.4f}  (rougeL: {metrics.get('rougeL', 0):.4f})")
        # Sample outputs — helps diagnose generation quality
        for i in range(min(2, len(decoded_preds))):
            logger.info(f"  [sample {i+1}] INPUT REF: {decoded_labels[i][:80]}")
            logger.info(f"  [sample {i+1}] GENERATED: {decoded_preds[i][:80]}")
        return metrics

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    seq2seq_args_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_strategy_key = "evaluation_strategy" if "evaluation_strategy" in seq2seq_args_params else "eval_strategy"

    training_kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": num_epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": train_batch,
        "per_device_eval_batch_size": train_batch,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "predict_with_generate": True,
        "generation_max_length": 128,
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": DEBUG_CONFIG["logging_steps"] if debug else 25,
        "seed": args.seed,
        "report_to": "none",
        "save_total_limit": 2,
    }
    if "fp16" in seq2seq_args_params:
        training_kwargs["fp16"] = use_fp16
    if "bf16" in seq2seq_args_params:
        training_kwargs["bf16"] = use_bf16
    training_kwargs[eval_strategy_key] = "steps"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    steps_per_epoch = len(train_dataset) // train_batch
    total_steps = steps_per_epoch * num_epochs
    logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    logger.info(f"Steps:   {steps_per_epoch} per epoch × {num_epochs} epochs = {total_steps} total")

    logger.info(f"Starting Phase 2 training (condition={args.condition})...")
    t0 = time.time()
    trainer.train()
    training_duration = time.time() - t0

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Phase 2 ({args.condition}) checkpoint saved to {args.output_dir}")

    # ------------------------------------------------------------------
    # Save training history for the final report
    # trainer.state.log_history contains every logged event:
    #   - training steps: {"loss": ..., "learning_rate": ..., "epoch": ..., "step": ...}
    #   - eval steps:     {"eval_loss": ..., "eval_rouge1": ..., "eval_rougeL": ..., ...}
    # ------------------------------------------------------------------
    history_data = {
        "phase": "phase2_meme_finetune",
        "condition": args.condition,
        "run_config": {
            "phase1_checkpoint": args.phase1_checkpoint_dir,
            "base_model": checkpoint,
            "condition": args.condition,
            "num_epochs": num_epochs,
            "batch_size": train_batch,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "precision": precision_mode,
            "seed": args.seed,
            "paradetox_mix_ratio": args.paradetox_mix_ratio,
            "meme_train_samples": n_meme_train,
            "paradetox_train_samples": n_paradetox_train,
            "total_train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "debug": debug,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                       if torch.cuda.is_available() else None,
        },
        "results": {
            "training_duration_seconds": round(training_duration, 1),
            "total_steps": trainer.state.global_step,
            "best_eval_loss": trainer.state.best_metric,
            "best_model_checkpoint": str(trainer.state.best_model_checkpoint)
                                     if trainer.state.best_model_checkpoint else None,
        },
        "log_history": trainer.state.log_history,
    }
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    print(f"\n{'='*60}")
    print(f"  Phase 2 [{args.condition}] COMPLETE — checkpoint saved to:")
    print(f"  {args.output_dir}")
    print(f"  Training time: {training_duration/60:.1f} min  |  Steps: {trainer.state.global_step}")
    print(f"  Best eval_loss: {trainer.state.best_metric}")
    print(f"  History:        {history_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
