"""
Stage 2 Phase 1: BART warm-up on ParaDetox.

Pre-trains facebook/bart-large on the s-nlp/paradetox dataset (text detoxification)
before meme-specific conditioning fine-tune in Phase 2.

Pipeline position: AFTER build_stage2_dataset, BEFORE train_stage2_phase2.

Usage (cluster):
    python training/train_stage2_phase1.py \
        --output_dir /scratch/hmr_stage2_phase1_checkpoint \
        --hf_cache  /scratch/hf_cache \
        --num_train_epochs 2 \
        --per_device_train_batch_size 16 \
        --learning_rate 5e-5 \
        --warmup_steps 500 \
        --weight_decay 0.01 \
        --seed 42
"""

import argparse
import inspect
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.debug import DEBUG_CONFIG, is_debug_mode, set_seeds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParaDetoxDataset(Dataset):
    """Wraps ParaDetox (toxic → neutral) pairs for Seq2SeqTrainer."""

    def __init__(self, examples: List[Dict], tokenizer, max_input_length: int = 128,
                 max_target_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
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
        # Phase 1 uses null conditioning prefix: BART learns plain style-transfer
        input_text = f"[T: null] [A: null] [M: null] </s> {ex['toxic']}"
        target_text = ex["neutral"]

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

        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        label_ids = labels["input_ids"].squeeze()
        # Replace pad token id in labels with -100 (ignored by loss)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paradetox(hf_cache: Optional[str], debug: bool) -> tuple:
    """Download and return ParaDetox train/validation splits."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package missing. Install: pip install datasets")
        sys.exit(1)

    logger.info("Loading s-nlp/paradetox from HuggingFace...")
    os.environ["HF_HOME"] = hf_cache or os.environ.get("HF_HOME", "~/.cache/huggingface")
    ds = load_dataset("s-nlp/paradetox", cache_dir=hf_cache)

    train_examples = [{"toxic": r["en_toxic_comment"], "neutral": r["en_neutral_comment"]}
                      for r in ds["train"]]
    # ParaDetox has no official val split — carve 10 %
    random.shuffle(train_examples)
    split = int(0.9 * len(train_examples))
    val_examples = train_examples[split:]
    train_examples = train_examples[:split]

    if debug:
        train_examples = train_examples[: DEBUG_CONFIG["max_samples"]]
        val_examples = val_examples[: DEBUG_CONFIG["max_samples"] // 2]
        logger.warning(f"DEBUG: truncated to {len(train_examples)} train, {len(val_examples)} val")
        return train_examples, val_examples

    # Production: filter pairs where BERTScore F1 < 0.5
    logger.info(f"Loaded {len(train_examples)} train pairs. Filtering by BERTScore ≥ 0.5 ...")
    from utils.bertscore_utils import compute_bertscore_batch

    toxics   = [e["toxic"]   for e in train_examples]
    neutrals = [e["neutral"] for e in train_examples]
    scores   = compute_bertscore_batch(toxics, neutrals)

    train_examples = [e for e, s in zip(train_examples, scores) if s >= 0.5]
    logger.info(f"After BERTScore filter: {len(train_examples)} train pairs remain")

    return train_examples, val_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Phase 1: BART ParaDetox warm-up")
    parser.add_argument("--output_dir",                  type=str, required=True)
    parser.add_argument("--hf_cache",                    type=str, default=None)
    parser.add_argument("--num_train_epochs",            type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate",               type=float, default=3e-5)
    parser.add_argument("--warmup_steps",                type=int, default=150)
    parser.add_argument("--weight_decay",                type=float, default=0.01)
    parser.add_argument("--seed",                        type=int, default=42)
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: tiny dataset, bart-base, 1 epoch")
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
    model_name = DEBUG_CONFIG["stage2_model"] if debug else "facebook/bart-large"
    num_epochs = DEBUG_CONFIG["num_train_epochs"] if debug else args.num_train_epochs
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
    print(f"  Stage 2 Phase 1: BART ParaDetox Warm-up")
    print(f"  Model:      {model_name}")
    print(f"  Epochs:     {num_epochs}")
    print(f"  Batch size: {train_batch}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Precision:  {precision_mode}")
    print(f"  Output:     {args.output_dir}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  Device:     CPU (no GPU found)")
    print(f"{'='*60}\n")
    logger.info(
        f"Model: {model_name} | epochs: {num_epochs} | batch: {train_batch} | precision: {precision_mode}"
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

    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=args.hf_cache)
    model     = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=args.hf_cache)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_examples, val_examples = load_paradetox(args.hf_cache, debug)

    train_dataset = ParaDetoxDataset(train_examples, tokenizer)
    val_dataset   = ParaDetoxDataset(val_examples,   tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    rouge = hf_evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}

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
        "metric_for_best_model": "rougeL",
        "greater_is_better": True,
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
    logger.info(f"Eval every {eval_steps} steps | Save every {save_steps} steps")

    logger.info("Starting Phase 1 training...")
    t0 = time.time()
    trainer.train()
    training_duration = time.time() - t0

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Phase 1 checkpoint saved to {args.output_dir}")

    # ------------------------------------------------------------------
    # Save training history for the final report
    # trainer.state.log_history contains every logged event:
    #   - training steps: {"loss": ..., "learning_rate": ..., "epoch": ..., "step": ...}
    #   - eval steps:     {"eval_loss": ..., "eval_rouge1": ..., "eval_rougeL": ..., ...}
    # ------------------------------------------------------------------
    history_data = {
        "phase": "phase1_paradetox",
        "run_config": {
            "model": model_name,
            "num_epochs": num_epochs,
            "batch_size": train_batch,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "precision": precision_mode,
            "seed": args.seed,
            "train_samples": len(train_dataset),
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
            "best_metric_rougeL": trainer.state.best_metric,
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
    print(f"  Phase 1 COMPLETE — checkpoint saved to:")
    print(f"  {args.output_dir}")
    print(f"  Training time: {training_duration/60:.1f} min  |  Steps: {trainer.state.global_step}")
    print(f"  Best rougeL:   {trainer.state.best_metric}")
    print(f"  History:       {history_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
