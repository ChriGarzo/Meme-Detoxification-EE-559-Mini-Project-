"""
Stage 2 Phase 1: BART warm-up on ParaDetox.

Pre-trains facebook/bart-large on the s-nlp/paradetox dataset (text
detoxification) before meme-specific conditioning fine-tune in Phase 2.

Key design decisions
--------------------
1.  Input format is ``[T: null] [V: null] [M: null] | {toxic_text}`` --
    IDENTICAL to the Phase 2 format so the model learns the conditioning
    prefix during Phase 1.  The original code used ``</s>`` as the
    separator, which is processed differently by BART's tokeniser and
    created a distribution shift between phases.

2.  Generation config is explicitly reset before training to avoid
    inheriting any summarisation-era BART settings (min_length,
    length_penalty, no_repeat_ngram_size, etc.) that cause beam search
    to collapse to degenerate output (e.g. ",") on OOD inputs.

3.  Dynamic padding via DataCollatorForSeq2Seq -- no per-item
    ``padding="max_length"`` in __getitem__.  Pre-padding every item to
    the same length makes the attention mask all-ones for every token,
    wastes compute, and can confuse beam search when the encoder
    receives only padding tokens.

4.  Quality filter applied consistently to BOTH train and val:
      - Remove pairs where toxic == neutral (zero learning signal).
      - Remove pairs with BERTScore F1 < --bertscore_min (default 0.5)
        to discard examples where the "neutral" text is semantically
        unrelated to the toxic input (bad annotations).
    Applying the same filter to val ensures the rougeL metric used for
    best-model selection is computed on genuinely good reference pairs.

5.  label_smoothing_factor=0.1 prevents the model from becoming
    overconfident on ParaDetox's specific lexical patterns, which
    improves generalisation to Phase 2 meme data.

Pipeline position: BEFORE train_stage2_phase2.py.
Pass this checkpoint to Phase 2 via --phase1_checkpoint_dir.

Usage (cluster):
    python training/train_stage2_phase1.py \\
        --output_dir            /scratch/hmr_stage2_phase1_checkpoint \\
        --hf_cache              /scratch/hf_cache \\
        --num_train_epochs      3 \\
        --per_device_train_batch_size 16 \\
        --learning_rate         3e-5 \\
        --warmup_steps          500 \\
        --weight_decay          0.01 \\
        --bertscore_min         0.5 \\
        --seed                  42
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

# Null-conditioning prefix used for ALL Phase 1 examples.
# Must match the format produced by format_input(..., condition="none")
# in train_stage2_phase2.py so the model sees a consistent input format
# across both phases.
NULL_PREFIX = "[T: null] [V: null] [M: null]"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParaDetoxDataset(Dataset):
    """Wraps ParaDetox (toxic -> neutral) pairs for Seq2SeqTrainer."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_input_length: int = 128,
        max_target_length: int = 128,
    ):
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

        # Use the SAME format as Phase 2 (pipe separator, null fields).
        # Phase 2 extends this by filling in [T:], [V:], [M:] for meme
        # examples; Phase 1 teaches the model that the null-conditioned
        # format always means "detoxify the text after the |".
        input_text  = f"{NULL_PREFIX} | {ex['toxic']}"
        target_text = ex["neutral"]

        # No padding here -- DataCollatorForSeq2Seq handles dynamic per-batch
        # padding.  Pre-padding to max_length makes the attention mask all-ones
        # and wastes encoder capacity on pure padding tokens.
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
        )
        if self._supports_text_target:
            labels = self.tokenizer(
                text_target=target_text,
                max_length=self.max_target_length,
                truncation=True,
            )
        elif hasattr(self.tokenizer, "as_target_tokenizer"):
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    target_text,
                    max_length=self.max_target_length,
                    truncation=True,
                )
        else:
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                truncation=True,
            )

        # DataCollatorForSeq2Seq replaces pad_token_id with -100 in labels.
        return {
            "input_ids":      model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels":         labels["input_ids"],
        }


# ---------------------------------------------------------------------------
# Data loading and quality filtering
# ---------------------------------------------------------------------------

def _basic_filter(examples: List[Dict]) -> List[Dict]:
    """Remove pairs where toxic == neutral (zero learning signal)."""
    before = len(examples)
    filtered = [
        e for e in examples
        if e["toxic"].strip() and e["neutral"].strip()
        and e["toxic"].strip().lower() != e["neutral"].strip().lower()
    ]
    removed = before - len(filtered)
    if removed:
        logger.info(f"Basic filter: removed {removed} identical/empty pairs")
    return filtered


def _bertscore_filter(
    examples: List[Dict],
    min_score: float,
    hf_cache: Optional[str],
) -> List[Dict]:
    """
    Remove pairs whose BERTScore F1 (rescaled) is below min_score.

    BERTScore measures semantic similarity between neutral and toxic texts.
    Pairs below the threshold are likely bad annotations where the neutral
    text is unrelated to the original rather than a genuine detoxification.
    Uses rescale_with_baseline=True so 0 = baseline similarity, 1 = identical.
    """
    logger.info(
        f"Computing BERTScore on {len(examples)} examples "
        f"(threshold: F1 >= {min_score}) ..."
    )
    try:
        from utils.bertscore_utils import compute_bertscore_batch
    except ImportError:
        logger.warning("bertscore_utils not found -- skipping BERTScore filter.")
        return examples

    toxics   = [e["toxic"]   for e in examples]
    neutrals = [e["neutral"] for e in examples]
    scores   = compute_bertscore_batch(toxics, neutrals, device=None)

    filtered = [e for e, s in zip(examples, scores) if s >= min_score]
    logger.info(
        f"BERTScore filter: kept {len(filtered)} / {len(examples)} pairs "
        f"(removed {len(examples) - len(filtered)})"
    )
    return filtered


def load_paradetox(
    hf_cache: Optional[str],
    debug: bool,
    bertscore_min: float,
    seed: int,
) -> tuple:
    """
    Download ParaDetox, apply quality filters, and return train/val splits.

    Quality filtering is applied to the FULL dataset BEFORE splitting so that
    both train and val contain only genuine detoxification pairs.  This makes
    the rougeL metric used for best-model selection reliable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets package missing. Install: pip install datasets")
        sys.exit(1)

    logger.info("Loading s-nlp/paradetox from HuggingFace...")
    if hf_cache:
        os.environ["HF_HOME"] = hf_cache

    ds = load_dataset("s-nlp/paradetox", cache_dir=hf_cache)

    examples = [
        {"toxic": r["en_toxic_comment"], "neutral": r["en_neutral_comment"]}
        for r in ds["train"]
        if r.get("en_toxic_comment") and r.get("en_neutral_comment")
    ]
    logger.info(f"Loaded {len(examples)} raw ParaDetox pairs")

    if debug:
        examples = examples[: DEBUG_CONFIG["max_samples"]]
        logger.warning(f"DEBUG: truncated to {len(examples)} examples (skipping filters)")
    else:
        # Step 1: remove trivially bad pairs (identical or empty)
        examples = _basic_filter(examples)

        # Step 2: BERTScore quality filter applied before train/val split so
        # both sets share the same quality standard.
        if bertscore_min > 0.0:
            examples = _bertscore_filter(examples, bertscore_min, hf_cache)

    # Reproducible shuffle using an explicit seeded RNG -- not affected by
    # any prior calls to the global random state.
    rng = random.Random(seed)
    rng.shuffle(examples)

    split          = int(0.9 * len(examples))
    train_examples = examples[:split]
    val_examples   = examples[split:]

    logger.info(
        f"Split: {len(train_examples)} train / {len(val_examples)} val "
        f"(both sets use the same quality filter)"
    )
    return train_examples, val_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 Phase 1: BART ParaDetox warm-up"
    )
    parser.add_argument("--output_dir",                  type=str,   required=True)
    parser.add_argument("--hf_cache",                    type=str,   default=None)
    parser.add_argument("--num_train_epochs",            type=int,   default=3)
    parser.add_argument("--per_device_train_batch_size", type=int,   default=16)
    parser.add_argument("--learning_rate",               type=float, default=3e-5)
    parser.add_argument("--warmup_steps",                type=int,   default=500)
    parser.add_argument("--weight_decay",                type=float, default=0.01)
    parser.add_argument(
        "--bertscore_min", type=float, default=0.5,
        help=(
            "Minimum rescaled BERTScore F1 for a pair to be kept. "
            "Applied to both train and val before splitting. "
            "Set to 0.0 to disable. Default: 0.5."
        ),
    )
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: tiny dataset, bart-base, few steps")
    args = parser.parse_args()

    debug = is_debug_mode(args)
    set_seeds(args.seed)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # -----------------------------------------------------------------------
    # Hyperparameters
    # -----------------------------------------------------------------------
    model_name  = DEBUG_CONFIG["stage2_model"] if debug else "facebook/bart-large"
    num_epochs  = DEBUG_CONFIG["num_train_epochs"]            if debug else args.num_train_epochs
    train_batch = DEBUG_CONFIG["per_device_train_batch_size"] if debug else args.per_device_train_batch_size
    max_steps   = DEBUG_CONFIG["max_steps"]  if debug else -1
    save_steps  = DEBUG_CONFIG["save_steps"] if debug else 200
    eval_steps  = DEBUG_CONFIG["eval_steps"] if debug else 200

    use_fp16 = False
    use_bf16 = False
    if (not debug) and torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            use_fp16 = True
    precision_mode = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")

    print(f"\n{'='*60}")
    print(f"  Stage 2 Phase 1: BART ParaDetox Warm-up")
    print(f"  Model:          {model_name}")
    print(f"  Epochs:         {num_epochs}")
    print(f"  Batch size:     {train_batch}")
    print(f"  LR:             {args.learning_rate}")
    print(f"  Warmup steps:   {args.warmup_steps}")
    print(f"  Precision:      {precision_mode}")
    print(f"  BERTScore min:  {args.bertscore_min}")
    print(f"  Input format:   {NULL_PREFIX} | {{toxic_text}}")
    print(f"  Output:         {args.output_dir}")
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  Device:         CPU (no GPU found)")
    print(f"{'='*60}\n")

    try:
        from transformers import (
            BartForConditionalGeneration,
            BartTokenizer,
            GenerationConfig,
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
    # Reset generation config to clean BART defaults.
    #
    # facebook/bart-large ships with a generation_config.json inherited from
    # its summarisation fine-tuning: min_length=56, length_penalty=2.0,
    # no_repeat_ngram_size=3.  These settings cause beam search to degenerate
    # to trivial outputs (e.g. ",") when the encoder receives the OOD
    # conditioning prefix before the model has been trained on it.
    # -----------------------------------------------------------------------
    model.generation_config = GenerationConfig(
        decoder_start_token_id=model.config.decoder_start_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        forced_bos_token_id=None,
        forced_eos_token_id=tokenizer.eos_token_id,
    )
    logger.info(
        f"Generation config reset: max_new_tokens=64, num_beams=4, "
        f"decoder_start_token_id={model.config.decoder_start_token_id}"
    )

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_examples, val_examples = load_paradetox(
        hf_cache=args.hf_cache,
        debug=debug,
        bertscore_min=args.bertscore_min,
        seed=args.seed,
    )

    train_dataset = ParaDetoxDataset(train_examples, tokenizer)
    val_dataset   = ParaDetoxDataset(val_examples,   tokenizer)

    # Dynamic padding -- DataCollatorForSeq2Seq pads each batch to its longest
    # sequence and replaces padding positions in labels with -100.
    # pad_to_multiple_of=8 aligns tensor sizes to hardware-friendly boundaries
    # when running in fp16/bf16.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if (use_fp16 or use_bf16) else None,
    )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    rouge = hf_evaluate.load("rouge")

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

    def compute_sta_score(texts: List[str]) -> float:
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

        # Sanitize preds: replace any out-of-range or negative values (pad
        # placeholders inserted by the Seq2SeqTrainer) with pad_token_id so
        # the tokenizer C-extension doesn't overflow on int conversion.
        preds  = np.where((preds  >= 0) & (preds  < tokenizer.vocab_size), preds,  tokenizer.pad_token_id)
        labels = np.where((labels >= 0) & (labels < tokenizer.vocab_size), labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result  = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        metrics = {k: round(v, 4) for k, v in result.items()}
        metrics["sta"] = compute_sta_score(decoded_preds)

        logger.info(
            f"  eval STA: {metrics['sta']:.4f}  "
            f"(rougeL: {metrics.get('rougeL', 0):.4f})"
        )
        # Sample outputs -- mirrors Phase 2 logging for easy cross-phase comparison
        for i in range(min(2, len(decoded_preds))):
            logger.info(f"  [sample {i+1}] REF:       {decoded_labels[i][:80]}")
            logger.info(f"  [sample {i+1}] GENERATED: {decoded_preds[i][:80]}")

        return metrics

    # -----------------------------------------------------------------------
    # Training arguments
    # -----------------------------------------------------------------------
    seq2seq_args_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    eval_strategy_key   = (
        "evaluation_strategy" if "evaluation_strategy" in seq2seq_args_params
        else "eval_strategy"
    )

    training_kwargs = {
        "output_dir":                   args.output_dir,
        "num_train_epochs":             num_epochs,
        "max_steps":                    max_steps,
        "per_device_train_batch_size":  train_batch,
        "per_device_eval_batch_size":   train_batch,
        "learning_rate":                args.learning_rate,
        "warmup_steps":                 args.warmup_steps,
        "weight_decay":                 args.weight_decay,
        # label_smoothing_factor=0.1 prevents the model from becoming
        # overconfident on ParaDetox's specific lexical patterns.  This is
        # especially important for Phase 1 because the model must later
        # generalise to meme-style rewrites in Phase 2.
        "label_smoothing_factor":       0.1,
        "predict_with_generate":        True,
        "generation_max_length":        64,
        "generation_num_beams":         4,
        eval_strategy_key:              "steps",
        "eval_steps":                   eval_steps,
        "save_strategy":                "steps",
        "save_steps":                   save_steps,
        "load_best_model_at_end":       True,
        "metric_for_best_model":        "rougeL",
        "greater_is_better":            True,
        "logging_steps":                DEBUG_CONFIG["logging_steps"] if debug else 25,
        "seed":                         args.seed,
        "report_to":                    "none",
        "save_total_limit":             2,
    }
    if "fp16" in seq2seq_args_params:
        training_kwargs["fp16"] = use_fp16
    if "bf16" in seq2seq_args_params:
        training_kwargs["bf16"] = use_bf16

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    trainer_kwargs = {
        "model":            model,
        "args":             training_args,
        "train_dataset":    train_dataset,
        "eval_dataset":     val_dataset,
        "data_collator":    data_collator,
        "compute_metrics":  compute_metrics,
    }
    trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    steps_per_epoch = len(train_dataset) // train_batch
    total_steps     = steps_per_epoch * num_epochs
    logger.info(f"Dataset: {len(train_dataset)} train / {len(val_dataset)} val")
    logger.info(f"Steps:   {steps_per_epoch} per epoch x {num_epochs} = {total_steps} total")
    logger.info(f"Eval every {eval_steps} steps | Save every {save_steps} steps")

    # -----------------------------------------------------------------------
    # Generation sanity check -- verify generation works BEFORE any training.
    # If this outputs "," or an empty string, the generation config is still
    # wrong and you should inspect model.generation_config directly.
    # -----------------------------------------------------------------------
    logger.info("Running pre-training generation sanity check...")
    _ex     = val_examples[0]
    _input  = f"{NULL_PREFIX} | {_ex['toxic']}"
    _enc    = tokenizer(_input, return_tensors="pt", truncation=True, max_length=128)
    _device = next(model.parameters()).device
    _enc    = {k: v.to(_device) for k, v in _enc.items()}
    with torch.no_grad():
        _gen = model.generate(**_enc, max_new_tokens=32, num_beams=4, early_stopping=True)
    _decoded = tokenizer.decode(_gen[0], skip_special_tokens=True)
    logger.info(f"  [sanity] toxic  : {_ex['toxic'][:80]}")
    logger.info(f"  [sanity] neutral: {_ex['neutral'][:80]}")
    logger.info(f"  [sanity] output : {_decoded[:80]}")
    if len(_decoded.strip()) <= 2:
        logger.warning(
            "Sanity check FAILED: model generates only 1-2 characters before "
            "training. Check model.generation_config."
        )
    else:
        logger.info("Sanity check passed -- generation looks normal, starting training.")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    logger.info("Starting Phase 1 training...")
    t0 = time.time()
    trainer.train()
    training_duration = time.time() - t0

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Persist the clean generation config so Phase 2 inherits sane defaults
    # when loading this checkpoint (Phase 2 also resets it, but saving here
    # makes standalone inference from the Phase 1 checkpoint work correctly).
    model.generation_config.save_pretrained(args.output_dir)
    logger.info(f"Phase 1 checkpoint + generation_config saved to {args.output_dir}")

    # -----------------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------------
    history_data = {
        "phase": "phase1_paradetox",
        "run_config": {
            "model":             model_name,
            "input_format":      f"{NULL_PREFIX} | {{toxic_text}}",
            "num_epochs":        num_epochs,
            "batch_size":        train_batch,
            "learning_rate":     args.learning_rate,
            "warmup_steps":      args.warmup_steps,
            "weight_decay":      args.weight_decay,
            "label_smoothing":   0.1,
            "bertscore_min":     args.bertscore_min,
            "precision":         precision_mode,
            "seed":              args.seed,
            "train_samples":     len(train_dataset),
            "val_samples":       len(val_dataset),
            "eval_steps":        eval_steps,
            "save_steps":        save_steps,
            "debug":             debug,
        },
        "hardware": {
            "gpu":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                       if torch.cuda.is_available() else None,
        },
        "results": {
            "training_duration_seconds": round(training_duration, 1),
            "total_steps":               trainer.state.global_step,
            "best_rougeL":               trainer.state.best_metric,
            "best_model_checkpoint":     str(trainer.state.best_model_checkpoint)
                                         if trainer.state.best_model_checkpoint else None,
        },
        "log_history": trainer.state.log_history,
    }
    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    print(f"\n{'='*60}")
    print(f"  Phase 1 COMPLETE -- checkpoint saved to:")
    print(f"  {args.output_dir}")
    print(f"  Training time: {training_duration/60:.1f} min  |  Steps: {trainer.state.global_step}")
    print(f"  Best rougeL:   {trainer.state.best_metric}")
    print(f"  History:       {history_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
