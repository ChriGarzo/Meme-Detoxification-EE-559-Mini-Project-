"""
Stage 2: BART meme conditioning fine-tune with LoRA.

LoRA fine-tunes a frozen BART-large base with low-rank adapters on all attention
projection matrices (q_proj, k_proj, v_proj, out_proj) and feed-forward layers
(fc1, fc2), giving ~17M trainable parameters out of 400M total (~4.3%).

Evaluation tracks at every checkpoint:
  - ROUGE-1/2/L       (text quality vs LLaVA pseudo-rewrites)
  - Collapse rate     (degenerate output detection)
  - Text STA          (s-nlp/roberta_toxicity_classifier, text-only)
  - Multimodal STA    (chiragmittal92/visualbert-hateful-memes-finetuned-model,
                       image + generated text — EVALUATION ONLY, no gradient
                       signal; images never influence training)

  Five (original, generated, reference) triples are logged at every eval step
  for qualitative inspection.

Run once per conditioning condition (full / target_only / visual_only / none).

Pipeline position: AFTER build_stage2_dataset.

Usage (cluster):
    python training/train_stage2_phase2.py \\
        --condition full \\
        --dataset_dir           /scratch/hmr_stage2_dataset \\
        --output_dir            /scratch/hmr_stage2_phase2_full_checkpoint \\
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
import inspect
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.debug import DEBUG_CONFIG, is_debug_mode, make_debug_dataset, set_seeds

logger = logging.getLogger(__name__)

CONDITION_TYPE = Literal["full", "target_only", "visual_only", "none"]


# ---------------------------------------------------------------------------
# Stage 1 image index (fallback when dataset was built without image_path)
# ---------------------------------------------------------------------------

def _build_stage1_image_index(stage1_output_dir: str) -> Dict[str, str]:
    """Build {dataset::id: image_path} index from Stage 1 pseudo-rewrite JSONL files."""
    root = Path(stage1_output_dir)
    if not root.exists():
        logger.warning(f"Stage 1 output dir not found: {root}")
        return {}

    index: Dict[str, str] = {}
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
                index[f"{dataset_name}::{ex_id}"] = img
                index.setdefault(str(ex_id), img)

    logger.info(f"Stage 1 image index: {len(index)} entries")
    return index


# ---------------------------------------------------------------------------
# Multimodal toxicity metric (VisualBERT, eval-only, no gradient signal)
# ---------------------------------------------------------------------------

def _compute_multimodal_sta(
    texts: List[str],
    image_paths: List[str],
    vb_model,
    vb_tokenizer,
    clip_model,
    clip_processor,
    device: str,
    batch_size: int = 16,
) -> Optional[float]:
    """
    Proportion of (image, generated_text) pairs classified as non-hateful
    by chiragmittal92/visualbert-hateful-memes-finetuned-model.

    Visual features: CLIP-ViT-Large/14 image embeddings (768-dim) zero-padded
    to 2048-dim and treated as a single visual token.  Images are never used
    for gradient computation — this is evaluation only.

    Returns None if no valid image paths are found.
    """
    valid_pairs = [
        (t, p) for t, p in zip(texts, image_paths)
        if p and Path(p).exists()
    ]
    if not valid_pairs:
        logger.warning("No valid image paths — multimodal STA skipped")
        return None

    non_hateful = 0
    total = 0

    for i in range(0, len(valid_pairs), batch_size):
        batch = valid_pairs[i : i + batch_size]
        batch_texts = [x[0] for x in batch]
        batch_paths = [x[1] for x in batch]

        # Load images and extract CLIP visual features
        try:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
        except Exception as e:
            logger.warning(f"Image load error in multimodal STA batch {i}: {e}")
            continue

        clip_inputs = clip_processor(images=images, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].to(device)

        with torch.no_grad():
            # [B, 768] — CLIP global image embedding via vision_model + visual_projection
            vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
            img_features = clip_model.visual_projection(vision_outputs.pooler_output).float()

        # Project 768 → 2048 by zero-padding (single visual token per image)
        B = img_features.shape[0]
        visual_embeds = torch.zeros(B, 1, 2048, dtype=torch.float32, device=device)
        visual_embeds[:, 0, :768] = img_features

        # Tokenize generated texts with BERT tokenizer (max 64 as per model config)
        text_inputs = vb_tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding="max_length",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            # VisualBERT logits: [B, 2] — 0=not hateful, 1=hateful
            logits = vb_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                visual_embeds=visual_embeds,
            )

        preds = logits.argmax(dim=-1)  # 0=not hateful, 1=hateful
        non_hateful += (preds == 0).sum().item()
        total += B

    if total == 0:
        return None
    return round(non_hateful / total, 4)


# ---------------------------------------------------------------------------
# Input formatting  (mirrors MemeRewriter.format_input)
# ---------------------------------------------------------------------------

def format_input(
    original_text: str,
    target_group: str,
    visual_evidence: str,
    implicit_meaning: str,
    condition: str,
) -> str:
    """Build BART encoder input string for the given condition."""
    tg = target_group    or "null"
    ve = visual_evidence or "null"
    im = implicit_meaning or "null"

    if condition == "full":
        prefix = f"[T: {tg}] [V: {ve}] [M: {im}]"
    elif condition == "target_only":
        prefix = f"[T: {tg}] [V: null] [M: null]"
    elif condition in {"visual_only", "attack_only"}:
        prefix = f"[T: null] [V: {ve}] [M: null]"
    else:  # "none"
        prefix = "[T: null] [V: null] [M: null]"

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
            visual_evidence=ex.get("visual_evidence", ex.get("attack_type")),
            implicit_meaning=ex.get("implicit_meaning"),
            condition=self.condition,
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
                "visual_evidence":  e.get("visual_evidence", e.get("attack_type")),
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
    parser = argparse.ArgumentParser(description="Stage 2 Phase 2: BART LoRA meme fine-tune")
    parser.add_argument("--condition",
                        type=str, required=True,
                        choices=["full", "target_only", "visual_only", "none"],
                        help="Conditioning strategy for the ablation study")
    parser.add_argument("--phase1_checkpoint_dir", type=str, default=None,
                        help="Phase 1 checkpoint dir. If not set, starts from --base_model directly.")
    parser.add_argument("--base_model", type=str, default="facebook/bart-large",
                        help="Base model when --phase1_checkpoint_dir is not provided.")
    parser.add_argument("--dataset_dir",     type=str, required=True,
                        help="Directory with train.jsonl/val.jsonl (output of build_stage2_dataset.py)")
    parser.add_argument("--output_dir",      type=str, required=True)
    parser.add_argument("--hf_cache",        type=str, default=None)
    parser.add_argument("--stage1_output_dir", type=str, default=None,
                        help="Stage 1 output dir for image_path lookup (fallback if dataset "
                             "was built before image_path was added to build_stage2_dataset.py)")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Stage 2 Phase 2: BART LoRA Meme Fine-tuning")
    print(f"  Condition:  {args.condition}")
    print(f"  Checkpoint: {checkpoint}")
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
            BertTokenizer,
            CLIPModel,
            CLIPProcessor,
            DataCollatorForSeq2Seq,
            GenerationConfig,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            pipeline as hf_pipeline,
        )
        import evaluate as hf_evaluate
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        logger.error(f"Missing package: {e}. Install: pip install transformers evaluate peft")
        sys.exit(1)

    tokenizer = BartTokenizer.from_pretrained(checkpoint, cache_dir=args.hf_cache)
    model     = BartForConditionalGeneration.from_pretrained(checkpoint, cache_dir=args.hf_cache)

    # -----------------------------------------------------------------------
    # Reset generation config to clean BART defaults (must happen before LoRA).
    # -----------------------------------------------------------------------
    seq2seq_args_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    generation_kwargs = {
        "decoder_start_token_id": model.config.decoder_start_token_id,
        "eos_token_id":           tokenizer.eos_token_id,
        "pad_token_id":           tokenizer.pad_token_id,
        "bos_token_id":           tokenizer.bos_token_id,
        "num_beams":              4,
        "early_stopping":         True,
        "no_repeat_ngram_size":   3,
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

    # -----------------------------------------------------------------------
    # LoRA
    # Target: all attention projections (encoder self-attn, decoder self-attn,
    # decoder cross-attn) + FFN layers in both encoder and decoder.
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_examples, val_examples = load_dataset(args.dataset_dir, debug)

    train_dataset = MemeRewriteDataset(train_examples, tokenizer, args.condition)
    val_dataset   = MemeRewriteDataset(val_examples,   tokenizer, args.condition)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # -----------------------------------------------------------------------
    # Build val image-path list for the multimodal STA metric.
    # Priority: image_path field in val.jsonl (set when dataset is rebuilt
    # after the build_stage2_dataset.py update). Fallback: Stage 1 index.
    # Images are used ONLY inside compute_metrics — never for gradients.
    # -----------------------------------------------------------------------
    stage1_index: Dict[str, str] = {}
    if args.stage1_output_dir:
        logger.info(f"Building Stage 1 image index from {args.stage1_output_dir}...")
        stage1_index = _build_stage1_image_index(args.stage1_output_dir)

    val_image_paths: List[str] = []
    for ex in val_examples:
        path = ex.get("image_path", "")
        if not path and stage1_index:
            ex_id   = ex.get("id")
            dataset = ex.get("dataset")
            scoped  = f"{dataset}::{ex_id}" if dataset and ex_id else None
            if scoped:
                path = stage1_index.get(scoped, "")
            if not path and ex_id:
                path = stage1_index.get(str(ex_id), "")
        val_image_paths.append(path or "")

    val_original_texts: List[str] = [ex.get("original_text", "") for ex in val_examples]

    n_with_images = sum(1 for p in val_image_paths if p and Path(p).exists())
    logger.info(
        f"Val set: {len(val_examples)} examples, {n_with_images} with valid image paths"
    )

    # -----------------------------------------------------------------------
    # Load VisualBERT + CLIP for multimodal STA (eval-only models, frozen).
    # We load them now so they are available inside compute_metrics via closure.
    # -----------------------------------------------------------------------
    vb_model        = None
    vb_tokenizer    = None
    clip_eval_model = None
    clip_eval_proc  = None
    mm_model_id     = "chiragmittal92/visualbert-hateful-memes-finetuned-model"

    if n_with_images > 0:
        try:
            logger.info(f"Loading VisualBERT from {mm_model_id}...")
            from transformers import VisualBertModel, VisualBertConfig
            import torch.nn as nn
            from huggingface_hub import hf_hub_download

            class _VBClassifier(nn.Module):
                """Wrapper matching chiragmittal92/visualbert-hateful-memes-finetuned-model.
                That checkpoint was saved with self.visualbert = VisualBertModel(...)
                and self.classifier = Linear(hidden_size, 2), so its state-dict keys
                carry the 'visualbert.' prefix that bare VisualBertModel does not expect.
                """
                def __init__(self, config):
                    super().__init__()
                    self.visualbert = VisualBertModel(config)
                    self.classifier = nn.Linear(config.hidden_size, 2)

                def forward(self, input_ids, attention_mask, visual_embeds, **kwargs):
                    out = self.visualbert(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        visual_embeds=visual_embeds,
                        **kwargs,
                    )
                    return self.classifier(out.pooler_output)  # [B, 2] logits

            vb_config = VisualBertConfig.from_pretrained(mm_model_id, cache_dir=args.hf_cache)
            vb_model = _VBClassifier(vb_config)

            # Load raw weights from Hub — the checkpoint uses pytorch_model.bin
            ckpt_path = hf_hub_download(
                mm_model_id, "pytorch_model.bin", cache_dir=args.hf_cache
            )
            state_dict = torch.load(ckpt_path, map_location="cpu")
            missing, unexpected = vb_model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"VisualBERT: {len(missing)} missing keys after load")
            if unexpected:
                logger.warning(f"VisualBERT: {len(unexpected)} unexpected keys after load")
            else:
                logger.info("VisualBERT weights loaded cleanly.")

            vb_model = vb_model.to(device).float().eval()
            for p in vb_model.parameters():
                p.requires_grad = False

            vb_tokenizer = BertTokenizer.from_pretrained(
                mm_model_id,
                cache_dir=args.hf_cache,
            )
            logger.info("VisualBERT loaded and frozen.")

            logger.info("Loading CLIP for visual feature extraction...")
            clip_eval_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=args.hf_cache,
            ).to(device).eval()
            for p in clip_eval_model.parameters():
                p.requires_grad = False
            clip_eval_proc = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=args.hf_cache,
            )
            logger.info("CLIP loaded and frozen.")
        except Exception as e:
            logger.warning(f"Could not load multimodal models: {e}. Multimodal STA disabled.")
            vb_model = vb_tokenizer = clip_eval_model = clip_eval_proc = None

    # -----------------------------------------------------------------------
    # Text-only STA classifier (s-nlp/roberta_toxicity_classifier)
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    rouge = hf_evaluate.load("rouge")

    def compute_sta_score(texts: List[str]) -> float:
        """Proportion of texts classified as non-toxic (text-only)."""
        non_toxic = 0
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            try:
                results = toxicity_pipe(batch)
                non_toxic += sum(1 for r in results if r["label"].lower() != "toxic")
            except Exception as e:
                logger.warning(f"Text STA batch failed: {e}")
        return round(non_toxic / max(1, len(texts)), 4)

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

        # Collapse guard
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
        metrics["sta"] = compute_sta_score(decoded_preds)

        # Multimodal STA (VisualBERT — image + generated text, no gradient)
        if vb_model is not None and vb_tokenizer is not None:
            mm_sta = _compute_multimodal_sta(
                texts=decoded_preds,
                image_paths=val_image_paths,
                vb_model=vb_model,
                vb_tokenizer=vb_tokenizer,
                clip_model=clip_eval_model,
                clip_processor=clip_eval_proc,
                device=device,
            )
            if mm_sta is not None:
                metrics["multimodal_sta"] = mm_sta
                logger.info(
                    "  eval multimodal_sta: %.4f  |  text_sta: %.4f  |  "
                    "rougeL: %.4f  |  collapse_rate: %.4f",
                    mm_sta, metrics["sta"],
                    metrics.get("rougeL", 0.0), metrics["collapse_rate"],
                )
            else:
                logger.info(
                    "  eval text_sta: %.4f  |  rougeL: %.4f  |  collapse_rate: %.4f",
                    metrics["sta"], metrics.get("rougeL", 0.0), metrics["collapse_rate"],
                )
        else:
            logger.info(
                "  eval text_sta: %.4f  |  rougeL: %.4f  |  collapse_rate: %.4f",
                metrics["sta"], metrics.get("rougeL", 0.0), metrics["collapse_rate"],
            )

        # 5 qualitative examples: original → generated → reference
        logger.info("  --- qualitative samples ---")
        for i in range(min(5, len(decoded_preds))):
            orig = val_original_texts[i] if i < len(val_original_texts) else "N/A"
            logger.info("  [ex %d] ORIGINAL : %s", i + 1, orig[:100])
            logger.info("  [ex %d] GENERATED: %s", i + 1, decoded_preds[i][:100])
            logger.info("  [ex %d] REFERENCE: %s", i + 1, decoded_labels[i][:100])

        return metrics

    # -----------------------------------------------------------------------
    # Training arguments
    # NOTE: load_best_model_at_end=False is required for PEFT compatibility.
    # The best checkpoint adapter is available in output_dir/checkpoint-*/
    # and can be loaded manually with PeftModel.from_pretrained() if needed.
    # -----------------------------------------------------------------------
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
        "metric_for_best_model":          "eval_rougeL",
        "greater_is_better":              True,
        "logging_steps":                  DEBUG_CONFIG["logging_steps"] if debug else 25,
        "seed":                           args.seed,
        "report_to":                      "none",
        "save_total_limit":               5,
    }
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

    steps_per_epoch = max(1, len(train_dataset) // train_batch)
    total_steps = steps_per_epoch * num_epochs
    logger.info(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    logger.info(f"Steps:   {steps_per_epoch} per epoch × {num_epochs} epochs = {total_steps} total")

    # -----------------------------------------------------------------------
    # Generation sanity check (before training)
    # -----------------------------------------------------------------------
    logger.info("Running pre-training generation sanity check...")
    _sample = val_examples[0]
    _input_str = format_input(
        original_text=_sample.get("original_text", "test input"),
        target_group=_sample.get("target_group"),
        visual_evidence=_sample.get("visual_evidence", _sample.get("attack_type")),
        implicit_meaning=_sample.get("implicit_meaning"),
        condition=args.condition,
    )
    _enc = tokenizer(_input_str, return_tensors="pt", truncation=True, max_length=128)
    _dev = next(model.parameters()).device
    _enc = {k: v.to(_dev) for k, v in _enc.items()}
    sanity_gen_kwargs = {"max_new_tokens": 32, "num_beams": 4, "early_stopping": True}
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

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    logger.info(f"Starting Phase 2 LoRA training (condition={args.condition})...")
    t0 = time.time()
    trainer.train()
    training_duration = time.time() - t0

    # -----------------------------------------------------------------------
    # Save: adapter weights (reproducibility) + merged model (pipeline compat)
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Save LoRA adapter weights for reference / later re-use
    lora_adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(lora_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_adapter_dir)
    tokenizer.save_pretrained(lora_adapter_dir)
    logger.info(f"LoRA adapter saved to {lora_adapter_dir}")

    # 2. Merge adapter into base model — makes the checkpoint compatible with
    #    all downstream scripts (run_stage2.py, train_proxy.py, evaluate.py)
    #    that load with BartForConditionalGeneration.from_pretrained().
    logger.info("Merging LoRA weights into base model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.generation_config = stored_gen_config
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Merged model saved to {args.output_dir}")

    # -----------------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------------
    eval_entries  = [e for e in trainer.state.log_history if "eval_loss" in e]
    min_eval_loss = min((e.get("eval_loss") for e in eval_entries), default=None)

    history_data = {
        "phase": "phase2_lora_meme_finetune",
        "condition": args.condition,
        "run_config": {
            "phase1_checkpoint":    args.phase1_checkpoint_dir,
            "base_model":           checkpoint,
            "condition":            args.condition,
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
            "val_samples_with_img": n_with_images,
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
            "total_steps":                trainer.state.global_step,
            "best_metric_name":           "eval_rougeL",
            "best_metric_value":          trainer.state.best_metric,
            "best_model_checkpoint":      str(trainer.state.best_model_checkpoint)
                                          if trainer.state.best_model_checkpoint else None,
            "min_eval_loss":              min_eval_loss,
            "note": (
                "load_best_model_at_end=False (PEFT compatibility). "
                "Saved model is from the last training step. "
                "Best adapter at best_model_checkpoint can be loaded with "
                "PeftModel.from_pretrained(base_model, best_model_checkpoint)."
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
    print(f"  History:       {history_path}")
    print(f"  Training time: {training_duration/60:.1f} min  |  Steps: {trainer.state.global_step}")
    print(f"  Best eval_rougeL: {trainer.state.best_metric}")
    print(f"  LoRA — trainable params: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
