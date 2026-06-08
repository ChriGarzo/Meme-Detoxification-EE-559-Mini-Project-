"""
Stage 2 trainer: BART-large LoRA fine-tuning for conditioned meme detoxification.

Pipeline position: AFTER build_stage2_dataset.py.

Inputs : train.jsonl / val.jsonl produced by build_stage2_dataset.py.
         Each row contains (original_text, target_text, target_group,
         visual_evidence, implicit_meaning, image_path).
Outputs: <output_dir>/              — merged BART model (pipeline-compatible)
         <output_dir>/lora_adapter/ — best-checkpoint LoRA adapter weights
         <output_dir>/lora_adapter_final/ — final-step adapter (fallback)
         <output_dir>/training_history.json

Design decisions:
- LoRA targets all attention projections (q/k/v/out_proj) and FFN layers
  (fc1, fc2) in both encoder and decoder, yielding ~17M trainable / 400M total.
- Four conditioning ablations: full | target_only | visual_only | none.
  Run this script once per condition with a separate --output_dir.
- load_best_model_at_end=False is required for PEFT compatibility; the best
  adapter is selected manually after training via eval_detox_quality.

Usage (cluster):
    python training/train_stage2.py \\
        --condition full \\
        --dataset_dir           /scratch/hmr_stage2_dataset \\
        --output_dir            /scratch/hmr_stage2_full_checkpoint \\
        --hf_cache              /scratch/hf_cache \\
        --stage1_output_dir     /scratch/hmr_stage1_output \\
        --num_train_epochs 5 \\
        --per_device_train_batch_size 8 \\
        --learning_rate 1e-4 \\
        --warmup_steps 50 \\
        --weight_decay 0.01 \\
        --lora_r 32 \\
        --lora_alpha 64 \\
        --lora_dropout 0.05 \\
        --seed 42
"""

import argparse
import difflib
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

from codecarbon import EmissionsTracker

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.debug import DEBUG_CONFIG, is_debug_mode, make_debug_dataset, set_seeds

logger = logging.getLogger(__name__)

CONDITION_TYPE = Literal["full", "target_only", "visual_only", "none"]


# ---------------------------------------------------------------------------
# Input formatting  (mirrors MemeRewriter.format_input)
# ---------------------------------------------------------------------------

def format_input(
    original_text: str,
    target_group: str,
    visual_evidence: str,
    implicit_meaning: str,
    condition: str,
    input_format: str = "legacy",
    task_prefix: str = "",
) -> str:
    """Build BART encoder input string for the given condition."""
    raw_tg = target_group or "null"
    raw_ve = visual_evidence or "null"
    raw_im = implicit_meaning or "null"

    if condition == "full":
        tg = raw_tg
        ve = raw_ve
        im = raw_im
        prefix = f"[T: {tg}] [V: {ve}] [M: {im}]"
    elif condition == "target_only":
        tg = raw_tg
        ve = "null"
        im = "null"
        prefix = f"[T: {tg}] [V: null] [M: null]"
    elif condition == "visual_only":
        tg = "null"
        ve = raw_ve
        im = "null"
        prefix = f"[T: null] [V: {ve}] [M: null]"
    else:  # "none"
        tg = "null"
        ve = "null"
        im = "null"
        prefix = "[T: null] [V: null] [M: null]"

    if input_format == "explicit_detox":
        formatted = (
            "Task: rewrite the original meme text to be non-toxic while preserving "
            "the meme topic and intended meaning. "
            f"Context: target group = {tg}; visual evidence = {ve}; "
            f"implicit harmful meaning = {im}. "
            f"Original meme text to detoxify: {original_text}"
        )
    else:
        formatted = f"{prefix} | {original_text}"

    task_prefix = (task_prefix or "").strip()
    if task_prefix:
        return f"{task_prefix} {formatted}"
    return formatted


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def normalized_char_similarity(a: str, b: str) -> float:
    left = _normalize_for_compare(a)
    right = _normalize_for_compare(b)
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MemeRewriteDataset(Dataset):
    """Meme pseudo-rewrite pairs formatted as condition-specific BART encoder inputs."""

    def __init__(self, examples: List[Dict], tokenizer, condition: str,
                 input_format: str = "legacy", task_prefix: str = "",
                 max_input_length: int = 128, max_target_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.condition = condition
        self.input_format = input_format
        self.task_prefix = task_prefix
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        try:
            tokenizer_call_params = inspect.signature(self.tokenizer.__call__).parameters
        except (TypeError, ValueError):
            tokenizer_call_params = {}
        # text_target kwarg is the canonical seq2seq API; older tokenizers expose
        # as_target_tokenizer() context manager instead.
        self._supports_text_target = "text_target" in tokenizer_call_params

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        input_text = format_input(
            original_text=ex.get("original_text", ""),
            target_group=ex.get("target_group"),
            visual_evidence=ex.get("visual_evidence"),
            implicit_meaning=ex.get("implicit_meaning"),
            condition=self.condition,
            input_format=self.input_format,
            task_prefix=self.task_prefix,
        )
        target_text = ex.get("target_text", "")

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

        return {
            "input_ids":      model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels":         labels["input_ids"],
        }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    """Read a newline-delimited JSON file into a list of dicts."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_dataset(dataset_dir: str, debug: bool) -> tuple:
    """Load train.jsonl and val.jsonl from the Stage 2 dataset directory."""
    if debug:
        raw = make_debug_dataset(n=DEBUG_CONFIG["max_samples"])
        examples = [
            {
                "original_text":    e["text"],
                "target_text":      e.get("explanation", {}).get("implicit_meaning", e["text"]),
                "target_group":     e.get("target_group"),
                "visual_evidence":  e.get("visual_evidence"),
                "implicit_meaning": (e.get("explanation") or {}).get("implicit_meaning"),
                "image_path":       e.get("image_path", ""),
                "dataset":          "debug",
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
    parser = argparse.ArgumentParser(description="Stage 2: BART LoRA meme fine-tune")
    parser.add_argument("--condition",
                        type=str, required=True,
                        choices=["full", "target_only", "visual_only", "none"],
                        help="Conditioning strategy for the ablation study")
    parser.add_argument("--base_model", type=str, default="facebook/bart-large",
                        help="Base model to start from.")
    parser.add_argument("--dataset_dir",     type=str, required=True,
                        help="Directory with train.jsonl/val.jsonl (output of build_stage2_dataset.py)")
    parser.add_argument("--output_dir",      type=str, required=True)
    parser.add_argument("--hf_cache",        type=str, default=None)
    parser.add_argument("--stage1_output_dir", type=str, default=None,
                        help="Stage 1 output dir for image_path lookup (fallback if dataset "
                             "was built before image_path was added to build_stage2_dataset.py)")
    parser.add_argument("--input_format",
                        type=str,
                        default="legacy",
                        choices=["legacy", "explicit_detox"],
                        help=(
                            "Encoder input format. legacy keeps the original bracket prompt; "
                            "explicit_detox uses labeled context plus an explicit original text "
                            "detoxification instruction."
                        ))
    parser.add_argument("--task_prefix", type=str, default="",
                        help="Optional prefix prepended to the encoder input after formatting.")
    # Training hyperparameters
    parser.add_argument("--num_train_epochs",            type=int,   default=5)
    parser.add_argument("--per_device_train_batch_size", type=int,   default=8)
    parser.add_argument("--learning_rate",               type=float, default=1e-4)
    parser.add_argument("--warmup_steps",                type=int,   default=50)
    parser.add_argument("--weight_decay",                type=float, default=0.01)
    parser.add_argument("--seed",                        type=int,   default=42)
    # LoRA hyperparameters
    parser.add_argument("--lora_r",       type=int,   default=32,
                        help="LoRA rank. Higher = more capacity. Default 32.")
    parser.add_argument("--lora_alpha",   type=int,   default=64,
                        help="LoRA scaling factor (lora_alpha/r). Default 64.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout applied inside LoRA adapters. Default 0.05.")
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
    checkpoint = args.base_model
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Stage 2 Phase 2: BART LoRA Meme Fine-tuning")
    print(f"  Condition:  {args.condition}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Input fmt:  {args.input_format}")
    if args.task_prefix.strip():
        print(f"  Task prefix:{args.task_prefix.strip()}")
    print(f"  Epochs:     {num_epochs}")
    print(f"  Batch size: {train_batch}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Precision:  {precision_mode}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  Output:     {args.output_dir}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"  Device:     CPU (no GPU found)")
    print(f"{'='*60}\n")

    try:
        from transformers import (
            BartForConditionalGeneration,
            BartTokenizer,
            DataCollatorForSeq2Seq,
            GenerationConfig,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            pipeline as hf_pipeline,
        )
        import evaluate as hf_evaluate
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as e:
        logger.error(f"Missing package: {e}. Install: pip install transformers evaluate peft")
        sys.exit(1)

    tokenizer = BartTokenizer.from_pretrained(checkpoint, cache_dir=args.hf_cache)
    model     = BartForConditionalGeneration.from_pretrained(checkpoint, cache_dir=args.hf_cache)

    # ── Generation config ─────────────────────────────────────────────────────
    # Explicitly overwrite the generation config stored inside the BART checkpoint
    # before wrapping with LoRA.  BART-large ships a generation_config.json that
    # sets forced_bos_token_id and other defaults that interfere with beam search
    # and the Seq2SeqTrainer's predict_with_generate path.  Resetting here ensures
    # the training-time and inference-time configs are identical.
    seq2seq_args_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    generation_kwargs = {
        "decoder_start_token_id": model.config.decoder_start_token_id,
        "eos_token_id":           tokenizer.eos_token_id,
        "pad_token_id":           tokenizer.pad_token_id,
        "bos_token_id":           tokenizer.bos_token_id,
        "num_beams":              4,
        "early_stopping":         True,
        "no_repeat_ngram_size":   3,
        "encoder_no_repeat_ngram_size": 3,
        "forced_bos_token_id":    None,
        "forced_eos_token_id":    tokenizer.eos_token_id,
        "max_length":             64,
        "min_length":             8,
    }
    if "min_new_tokens" in inspect.signature(GenerationConfig.__init__).parameters:
        generation_kwargs["min_new_tokens"] = 8

    stored_gen_config = GenerationConfig(**generation_kwargs)
    model.generation_config = stored_gen_config
    logger.info(
        "Generation config reset: max_length=%s, min_length=%s, num_beams=%s",
        stored_gen_config.max_length,
        stored_gen_config.min_length,
        stored_gen_config.num_beams,
    )

    # ── LoRA adapter ──────────────────────────────────────────────────────────
    # Targets all attention projections (encoder self-attn, decoder self-attn,
    # decoder cross-attn) plus FFN layers, covering every weight-intensive block
    # while leaving embedding tables and layer norms frozen.
    # bias="none" prevents adapter bias terms from leaking into the frozen base.
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied — trainable: %s / %s (%.2f%%)",
        f"{trainable_params:,}", f"{total_params:,}",
        100.0 * trainable_params / total_params,
    )
    model.print_trainable_parameters()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_examples, val_examples = load_dataset(args.dataset_dir, debug)

    train_dataset = MemeRewriteDataset(
        train_examples,
        tokenizer,
        args.condition,
        input_format=args.input_format,
        task_prefix=args.task_prefix,
    )
    val_dataset = MemeRewriteDataset(
        val_examples,
        tokenizer,
        args.condition,
        input_format=args.input_format,
        task_prefix=args.task_prefix,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    val_original_texts: List[str] = [ex.get("original_text", "") for ex in val_examples]
    logger.info(f"Val set: {len(val_examples)} examples")

    # ── Text-only STA classifier ──────────────────────────────────────────────
    # s-nlp/roberta_toxicity_classifier; text path only, no images.
    logger.info("Loading text toxicity classifier (s-nlp/roberta_toxicity_classifier)...")
    toxicity_pipe = hf_pipeline(
        "text-classification",
        model="s-nlp/roberta_toxicity_classifier",
        device=0 if torch.cuda.is_available() else -1,
        cache_dir=args.hf_cache,
        batch_size=32,
        truncation=True,
        max_length=128,
    )
    logger.info("Text toxicity classifier loaded.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    rouge = hf_evaluate.load("rouge")

    def compute_sta_score(texts: List[str]) -> float:
        """Compute Style Transfer Accuracy: fraction of texts classified non-toxic."""
        tox_probs = compute_text_toxicity_probs(texts)
        return round(
            sum(1 for p in tox_probs if p < 0.5) / max(1, len(tox_probs)),
            4,
        )

    def compute_text_toxicity_probs(texts: List[str]) -> List[float]:
        """Estimate P(toxic) per text using the RoBERTa toxicity classifier."""
        tox_probs: List[float] = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            try:
                results = toxicity_pipe(batch)
                for r in results:
                    label = str(r.get("label", "")).lower()
                    score = float(r.get("score", 0.0))
                    tox_probs.append(score if label == "toxic" else 1.0 - score)
            except Exception as e:
                logger.warning(f"Text STA batch failed: {e}")
                tox_probs.extend([0.5] * len(batch))
        return tox_probs

    logger.info("Precomputing original validation text toxicity...")
    val_original_toxicity_probs = compute_text_toxicity_probs(val_original_texts)
    val_original_toxicity_mean = (
        sum(val_original_toxicity_probs) / max(1, len(val_original_toxicity_probs))
        if val_original_toxicity_probs else 0.0
    )
    val_original_text_sta = (
        sum(1 for p in val_original_toxicity_probs if p < 0.5)
        / max(1, len(val_original_toxicity_probs))
        if val_original_toxicity_probs else 0.0
    )

    def _is_collapsed_output(text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True
        tokens = stripped.split()
        if len(tokens) <= 1:
            return True
        norm_tokens = [tok.lower() for tok in tokens]
        unique_ratio = len(set(norm_tokens)) / max(len(norm_tokens), 1)
        if len(tokens) >= 6 and unique_ratio < 0.35:
            return True
        if len(set(stripped)) <= 2 and len(stripped) >= 4:
            return True
        return False

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        metrics = {k: round(v, 4) for k, v in result.items()}

        # Degenerate-output guard: if more than half the batch collapses to empty
        # strings, repetitions, or single characters, ROUGE scores are meaningless
        # and would mislead the best-checkpoint selection logic.
        collapsed    = sum(1 for p in decoded_preds if _is_collapsed_output(p))
        collapse_rate = collapsed / max(1, len(decoded_preds))
        metrics["collapse_rate"] = round(collapse_rate, 4)
        if collapse_rate >= 0.5:
            logger.warning(
                "Collapse guard triggered: %.1f%% outputs look degenerate. "
                "Forcing ROUGE metrics to 0 for this eval step.",
                100 * collapse_rate,
            )
            for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                metrics[key] = 0.0

        # Text-only STA
        pred_toxicity_probs = compute_text_toxicity_probs(decoded_preds)
        metrics["sta"] = round(
            sum(1 for p in pred_toxicity_probs if p < 0.5) / max(1, len(pred_toxicity_probs)),
            4,
        )
        if val_original_toxicity_probs:
            paired = list(zip(val_original_toxicity_probs, pred_toxicity_probs))
            metrics["text_toxicity_drop"] = round(
                sum(orig - pred for orig, pred in paired) / max(1, len(paired)),
                4,
            )
            metrics["pred_toxicity_mean"] = round(
                sum(pred_toxicity_probs) / max(1, len(pred_toxicity_probs)),
                4,
            )
            metrics["original_toxicity_mean"] = round(val_original_toxicity_mean, 4)
            metrics["text_sta_delta"] = round(metrics["sta"] - val_original_text_sta, 4)

        source_sims = [
            normalized_char_similarity(orig, pred)
            for orig, pred in zip(val_original_texts, decoded_preds)
        ]
        exact_copies = [
            _normalize_for_compare(orig) == _normalize_for_compare(pred)
            for orig, pred in zip(val_original_texts, decoded_preds)
        ]
        metrics["source_similarity_mean"] = round(
            sum(source_sims) / max(1, len(source_sims)),
            4,
        )
        metrics["copy_rate_exact"] = round(
            sum(1 for x in exact_copies if x) / max(1, len(exact_copies)),
            4,
        )
        metrics["copy_rate_high"] = round(
            sum(1 for x in source_sims if x >= 0.85) / max(1, len(source_sims)),
            4,
        )
        # Composite metric used for best-checkpoint selection:
        # rewards toxicity reduction and fluency (ROUGE-L), penalises near-copies.
        metrics["detox_quality"] = round(
            metrics.get("text_toxicity_drop", 0.0)
            + 0.10 * metrics.get("rougeL", 0.0)
            - 0.25 * metrics["copy_rate_high"],
            4,
        )

        logger.info(
            "  eval text_sta: %.4f  |  tox_drop: %.4f  |  copy_high: %.4f  |  "
            "detox_quality: %.4f  |  rougeL: %.4f  |  collapse_rate: %.4f",
            metrics["sta"],
            metrics.get("text_toxicity_drop", 0.0),
            metrics["copy_rate_high"],
            metrics["detox_quality"],
            metrics.get("rougeL", 0.0),
            metrics["collapse_rate"],
        )

        # Five (original → generated → reference) triples for qualitative inspection.
        logger.info("  --- qualitative samples ---")
        for i in range(min(5, len(decoded_preds))):
            orig = val_original_texts[i] if i < len(val_original_texts) else "N/A"
            logger.info("  [ex %d] ORIGINAL : %s", i + 1, orig[:100])
            logger.info("  [ex %d] GENERATED: %s", i + 1, decoded_preds[i][:100])
            logger.info("  [ex %d] REFERENCE: %s", i + 1, decoded_labels[i][:100])

        return metrics

    # ── Training arguments ────────────────────────────────────────────────────
    # load_best_model_at_end=False: required for PEFT — HuggingFace Trainer's
    # model restore path does not handle PeftModel objects; the best adapter is
    # selected manually after training by scanning eval_detox_quality in log_history.
    eval_strategy_key = (
        "evaluation_strategy"
        if "evaluation_strategy" in seq2seq_args_params
        else "eval_strategy"
    )

    training_kwargs = {
        "output_dir":                     args.output_dir,
        "num_train_epochs":               num_epochs,
        "max_steps":                      max_steps,
        "per_device_train_batch_size":    train_batch,
        "per_device_eval_batch_size":     train_batch,
        "learning_rate":                  args.learning_rate,
        "max_grad_norm":                  1.0,
        "warmup_steps":                   args.warmup_steps,
        "weight_decay":                   args.weight_decay,
        "predict_with_generate":          True,
        "generation_max_length":          64,
        "generation_num_beams":           4,
        "eval_steps":                     eval_steps,
        "save_strategy":                  "steps",
        "save_steps":                     save_steps,
        "load_best_model_at_end":         False,
        "metric_for_best_model":          "eval_detox_quality",
        "greater_is_better":              True,
        "logging_steps":                  DEBUG_CONFIG["logging_steps"] if debug else 25,
        "seed":                           args.seed,
        "report_to":                      "none",
        "save_total_limit":               5,
    }
    if "overwrite_output_dir" in seq2seq_args_params:
        training_kwargs["overwrite_output_dir"] = True
    if "generation_min_length" in seq2seq_args_params:
        training_kwargs["generation_min_length"] = 8
    if "fp16" in seq2seq_args_params:
        training_kwargs["fp16"] = use_fp16
    if "bf16" in seq2seq_args_params:
        training_kwargs["bf16"] = use_bf16
    training_kwargs[eval_strategy_key] = "steps"
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model":           model,
        "args":            training_args,
        "train_dataset":   train_dataset,
        "eval_dataset":    val_dataset,
        "data_collator":   data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    grad_accum = max(1, int(getattr(training_args, "gradient_accumulation_steps", 1)))
    train_batches_per_epoch = max(1, math.ceil(len(train_dataset) / train_batch))
    steps_per_epoch = max(1, math.ceil(train_batches_per_epoch / grad_accum))
    total_steps = steps_per_epoch * num_epochs
    logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    logger.info(f"Steps:   {steps_per_epoch} per epoch × {num_epochs} epochs = {total_steps} total")

    # ── Pre-training generation sanity check ──────────────────────────────────
    # Verifies that the generation config reset above produces non-trivial output
    # before committing hours of GPU time.
    logger.info("Running pre-training generation sanity check...")
    _sample = val_examples[0]
    _input_str = format_input(
        original_text=_sample.get("original_text", "test input"),
        target_group=_sample.get("target_group"),
        visual_evidence=_sample.get("visual_evidence"),
        implicit_meaning=_sample.get("implicit_meaning"),
        condition=args.condition,
        input_format=args.input_format,
        task_prefix=args.task_prefix,
    )
    _enc = tokenizer(_input_str, return_tensors="pt", truncation=True, max_length=128)
    _dev = next(model.parameters()).device
    _enc = {k: v.to(_dev) for k, v in _enc.items()}
    sanity_gen_kwargs = {
        "max_new_tokens": 32,
        "num_beams": 4,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
        "encoder_no_repeat_ngram_size": 3,
    }
    if "min_new_tokens" in inspect.signature(model.generate).parameters:
        sanity_gen_kwargs["min_new_tokens"] = 8
    with torch.no_grad():
        _gen = model.generate(**_enc, **sanity_gen_kwargs)
    _decoded = tokenizer.decode(_gen[0], skip_special_tokens=True)
    logger.info(f"  [sanity] input : {_input_str[:80]}")
    logger.info(f"  [sanity] output: {_decoded[:80]}")
    if len(_decoded.strip()) <= 2:
        logger.warning(
            "Sanity check: model generates only 1-2 characters. "
            "Check model.generation_config before proceeding."
        )
    else:
        logger.info("Sanity check passed — proceeding to training.")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Starting Phase 2 LoRA training (condition={args.condition})...")
    os.makedirs(args.output_dir, exist_ok=True)
    _co2_tracker = EmissionsTracker(
        log_level="warning",
        output_dir=str(args.output_dir),
        output_file="emissions.csv",
    )
    _co2_tracker.start()
    t0 = time.time()
    trainer.train()
    training_duration = time.time() - t0
    _training_co2_kg = _co2_tracker.stop() or 0.0
    if _training_co2_kg > 0:
        logger.info(f"Training CO2 emissions: {_training_co2_kg*1e3:.4f} g CO2 ({_training_co2_kg:.6f} kg)")

    # ── Save: adapter weights + merged model ──────────────────────────────────
    # Two artefacts are saved:
    #   1. lora_adapter/       — best-eval adapter (PeftModel-compatible).
    #   2. <output_dir>/       — merged BART weights for pipeline-compatible loading
    #      via BartForConditionalGeneration.from_pretrained().
    # lora_adapter_final/ preserves the final training-step adapter as a fallback.
    os.makedirs(args.output_dir, exist_ok=True)

    eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
    min_eval_loss = min((e.get("eval_loss") for e in eval_entries), default=None)
    best_eval_entry = None
    if eval_entries:
        best_eval_entry = max(
            eval_entries,
            key=lambda e: (
                e.get("eval_detox_quality")
                if e.get("eval_detox_quality") is not None
                else e.get("eval_rougeL", float("-inf"))
            ),
        )
    best_step = int(best_eval_entry["step"]) if best_eval_entry and best_eval_entry.get("step") else None
    best_checkpoint_dir = (
        Path(args.output_dir) / f"checkpoint-{best_step}"
        if best_step is not None
        else None
    )
    use_best_checkpoint = bool(best_checkpoint_dir and best_checkpoint_dir.exists())

    # 1. LoRA adapter: prefer best eval_detox_quality checkpoint over final step.
    lora_adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(lora_adapter_dir, exist_ok=True)

    if use_best_checkpoint:
        logger.info(
            "Best eval checkpoint selected for export: %s (detox_quality=%s, rougeL=%s, copy_rate_high=%s)",
            best_checkpoint_dir,
            best_eval_entry.get("eval_detox_quality"),
            best_eval_entry.get("eval_rougeL"),
            best_eval_entry.get("eval_copy_rate_high"),
        )
        best_base_model = BartForConditionalGeneration.from_pretrained(
            checkpoint,
            cache_dir=args.hf_cache,
        )
        best_peft_model = PeftModel.from_pretrained(
            best_base_model,
            str(best_checkpoint_dir),
        )
        best_peft_model.save_pretrained(lora_adapter_dir)
        tokenizer.save_pretrained(lora_adapter_dir)
        logger.info(f"Best LoRA adapter saved to {lora_adapter_dir}")
        export_model = best_peft_model
        exported_checkpoint = str(best_checkpoint_dir)
        exported_is_best = True
    else:
        logger.warning(
            "No best checkpoint directory found; exporting final training step instead."
        )
        trainer.model.save_pretrained(lora_adapter_dir)
        tokenizer.save_pretrained(lora_adapter_dir)
        logger.info(f"Final LoRA adapter saved to {lora_adapter_dir}")
        export_model = trainer.model
        exported_checkpoint = "final_training_step"
        exported_is_best = False

    final_adapter_dir = os.path.join(args.output_dir, "lora_adapter_final")
    os.makedirs(final_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    logger.info(f"Final-step LoRA adapter saved to {final_adapter_dir}")

    # 2. Merge adapter into base model so downstream scripts can load it with
    #    BartForConditionalGeneration.from_pretrained() without the PEFT library.
    logger.info("Merging exported LoRA weights into base model...")
    merged_model = export_model.merge_and_unload()
    merged_model.generation_config = stored_gen_config
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Merged model saved to {args.output_dir}")

    # ── Training history ──────────────────────────────────────────────────────
    history_data = {
        "phase": "stage2_lora_meme_finetune",
        "condition": args.condition,
        "run_config": {
            "base_model":           checkpoint,
            "condition":            args.condition,
            "input_format":          args.input_format,
            "task_prefix":           args.task_prefix,
            "num_epochs":           num_epochs,
            "batch_size":           train_batch,
            "learning_rate":        args.learning_rate,
            "warmup_steps":         args.warmup_steps,
            "weight_decay":         args.weight_decay,
            "precision":            precision_mode,
            "seed":                 args.seed,
            "stage1_output_dir":    args.stage1_output_dir,
            "train_samples":        len(train_dataset),
            "val_samples":          len(val_dataset),
            "eval_steps":           eval_steps,
            "save_steps":           save_steps,
            "debug":                debug,
        },
        "lora_config": {
            "r":               args.lora_r,
            "alpha":           args.lora_alpha,
            "dropout":         args.lora_dropout,
            "target_modules":  lora_target_modules,
            "bias":            "none",
            "trainable_params": trainable_params,
            "total_params":     total_params,
            "trainable_pct":    round(100.0 * trainable_params / total_params, 2),
        },
        "hardware": {
            "gpu":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                       if torch.cuda.is_available() else None,
        },
        "results": {
            "training_duration_seconds":  round(training_duration, 1),
            "training_co2_kg":            round(_training_co2_kg, 8),
            "total_steps":                trainer.state.global_step,
            "best_metric_name":           "eval_detox_quality",
            "best_metric_value":          best_eval_entry.get("eval_detox_quality") if best_eval_entry else None,
            "best_eval_rougeL":           best_eval_entry.get("eval_rougeL") if best_eval_entry else None,
            "best_eval_copy_rate_high":   best_eval_entry.get("eval_copy_rate_high") if best_eval_entry else None,
            "best_eval_text_toxicity_drop": best_eval_entry.get("eval_text_toxicity_drop") if best_eval_entry else None,
            "best_model_checkpoint":      exported_checkpoint,
            "exported_is_best_checkpoint": exported_is_best,
            "min_eval_loss":              min_eval_loss,
            "note": (
                "Merged model is exported from the best eval_detox_quality checkpoint "
                "when that checkpoint directory exists; lora_adapter_final preserves "
                "the final training-step adapter."
            ),
        },
        "log_history": trainer.state.log_history,
    }

    history_path = Path(args.output_dir) / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    print(f"\n{'='*60}")
    print(f"  Phase 2 LoRA [{args.condition}] COMPLETE")
    print(f"  Merged model:  {args.output_dir}")
    print(f"  LoRA adapter:  {lora_adapter_dir}")
    print(f"  Final adapter: {final_adapter_dir}")
    print(f"  History:       {history_path}")
    print(f"  Training time: {training_duration/60:.1f} min  |  Steps: {trainer.state.global_step}")
    print(f"  Exported checkpoint: {exported_checkpoint}")
    print(f"  Best eval_detox_quality: {best_eval_entry.get('eval_detox_quality') if best_eval_entry else None}")
    print(f"  LoRA — trainable params: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
