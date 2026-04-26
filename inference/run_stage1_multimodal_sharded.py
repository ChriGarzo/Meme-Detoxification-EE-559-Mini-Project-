"""
Stage 1 (multimodal + sharded): LLaVA explanations + pseudo-rewrites with quality filtering.

Generates explanations for hateful meme text and creates pseudo-rewrites
using pattern-based methods. Applies quality filters based on multimodal
hatefulness (image + text) and semantic similarity.
"""

import argparse
from collections import Counter
import json
import logging
import os
import random
import re
import inspect
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from codecarbon import EmissionsTracker
from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import logging as hf_logging

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.explainer import MemeExplainer
from utils.bertscore_utils import compute_bertscore_batch, create_bertscore_scorer

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VisualBertMultimodalScorer:
    """Compute hateful probabilities from image + text with a VisualBERT-style model."""

    def __init__(self, model_name: str, device: str, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        self.processor = None
        self.tokenizer = None
        self.image_processor = None

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("AutoProcessor load failed for %s: %s", model_name, e)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("AutoTokenizer load failed for %s: %s", model_name, e)

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("AutoImageProcessor load failed for %s: %s", model_name, e)

        self.model = self._load_model()
        self.model = self.model.to(device)
        self.model.eval()
        self._forward_param_names = set(inspect.signature(self.model.forward).parameters.keys())
        self._resnet_encoder = None
        self._resnet_transform = None
        self._try_init_resnet_visual_encoder()
        self.positive_index = self._infer_positive_index()

        logger.info(
            "Loaded multimodal hate scorer: %s (positive label index=%d)",
            model_name,
            self.positive_index,
        )

    def _load_model(self):
        """Load a multimodal classification model with fallback for custom heads."""
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(
                "AutoModelForSequenceClassification load failed for %s: %s. "
                "Trying custom architecture fallback.",
                self.model_name,
                e,
            )
            primary_error = e

        # Fallback path for repos exposing a custom class (e.g. VisualBertForHatefulMemes).
        try:
            config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception:
            raise primary_error

        architectures = list(getattr(config, "architectures", []) or [])
        auto_map = getattr(config, "auto_map", {}) or {}
        class_refs: List[str] = []

        # Explicit model references from config auto_map.
        for key, value in auto_map.items():
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            if not isinstance(value, str):
                continue
            if "AutoModel" in key:
                class_refs.append(value)

        class_names: List[str] = []
        for arch in architectures:
            class_names.append(arch.split(".")[-1])

        # Fallback candidates commonly used by custom VisualBERT hateful meme checkpoints.
        fallback_names = [
            "VisualBertForHatefulMemes",
            "VisualBertForClassification",
            "VisualBertForSequenceClassification",
        ]
        class_names.extend(fallback_names)

        # If architecture already includes module path, try it first.
        for arch in architectures:
            if "." in arch:
                class_refs.append(arch)

        # Probe modeling*.py files in the model repo for a matching class.
        try:
            repo_files = list_repo_files(self.model_name, repo_type="model")
        except Exception:
            repo_files = []
        module_names = [
            Path(f).stem
            for f in repo_files
            if f.endswith(".py") and Path(f).name != "__init__.py"
        ]
        class_refs.extend(
            [f"{m}.{class_name}" for m in module_names for class_name in class_names]
        )

        # Deduplicate while preserving order.
        deduped_refs = []
        seen = set()
        for ref in class_refs:
            if ref not in seen:
                seen.add(ref)
                deduped_refs.append(ref)

        for class_ref in deduped_refs:
            try:
                cls = get_class_from_dynamic_module(
                    class_ref,
                    self.model_name,
                    cache_dir=self.cache_dir,
                )
                model = cls.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                )
                logger.info(
                    "Loaded custom multimodal scorer via dynamic class: %s",
                    class_ref,
                )
                return model
            except Exception as e:
                logger.warning(
                    "Dynamic class load failed for %s: %s",
                    class_ref,
                    e,
                )
                # Some custom classes fail in from_pretrained due meta init flows.
                # Fallback: explicit instantiate + manual checkpoint load.
                try:
                    config = AutoConfig.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        trust_remote_code=True,
                    )
                    model = cls(config)
                    state_dict = self._load_checkpoint_state_dict()
                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    logger.info(
                        "Loaded custom multimodal scorer via manual state_dict for %s "
                        "(missing=%d, unexpected=%d)",
                        class_ref,
                        len(missing),
                        len(unexpected),
                    )
                    return model
                except Exception as e2:
                    logger.warning(
                        "Manual state_dict load failed for %s: %s",
                        class_ref,
                        e2,
                    )

        raise primary_error

    def _load_checkpoint_state_dict(self) -> Dict[str, torch.Tensor]:
        """Load model checkpoint weights from HF repo (safetensors preferred)."""
        tried: List[str] = []
        for filename in ("model.safetensors", "pytorch_model.bin"):
            try:
                ckpt_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename=filename,
                    cache_dir=self.cache_dir,
                )
                if filename.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    return load_file(ckpt_path)
                return torch.load(ckpt_path, map_location="cpu")
            except Exception as e:
                tried.append(f"{filename}: {e}")
                continue
        raise RuntimeError(
            f"Could not load checkpoint for {self.model_name}. Tried: {' | '.join(tried)}"
        )

    def _infer_positive_index(self) -> int:
        id2label = getattr(self.model.config, "id2label", {}) or {}
        if not id2label:
            return 1

        # Normalize potential str/int keys.
        norm = {}
        for k, v in id2label.items():
            try:
                idx = int(k)
            except Exception:
                idx = k
            norm[idx] = str(v).lower()

        # Prefer explicit hateful labels that are not negated.
        for idx, label in norm.items():
            has_hate = ("hate" in label) or ("hateful" in label) or ("toxic" in label) or ("offensive" in label)
            negated = ("not" in label) or ("non" in label) or ("benign" in label) or ("neutral" in label)
            if has_hate and not negated and isinstance(idx, int):
                return idx

        if isinstance(max(norm.keys(), default=1), int):
            return 1 if 1 in norm else int(max(norm.keys()))
        return 1

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _try_init_resnet_visual_encoder(self) -> None:
        """Optional image feature extractor for VisualBERT-style visual_embeds."""
        try:
            from torchvision import models

            try:
                weights = models.ResNet50_Weights.DEFAULT
                encoder = models.resnet50(weights=weights)
                transform = weights.transforms()
            except Exception:
                encoder = models.resnet50(pretrained=True)
                from torchvision import transforms as T

                transform = T.Compose(
                    [
                        T.Resize((224, 224)),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )

            encoder.fc = torch.nn.Identity()
            encoder.eval()
            encoder.to(self.device)
            self._resnet_encoder = encoder
            self._resnet_transform = transform
            logger.info("Initialized ResNet50 visual encoder for visual_embeds")
        except Exception as e:
            logger.warning(
                "Could not initialize ResNet visual encoder (%s). "
                "Falling back to lightweight image embeddings.",
                e,
            )
            self._resnet_encoder = None
            self._resnet_transform = None

    def _compute_visual_embeds(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Build VisualBERT-style visual_embeds tensor [batch, num_visual_tokens, 2048].
        """
        if not images:
            return torch.zeros((0, 1, 2048), dtype=torch.float32, device=self.device)

        if self._resnet_encoder is not None and self._resnet_transform is not None:
            tensors = [self._resnet_transform(im) for im in images]
            batch = torch.stack(tensors, dim=0).to(self.device)
            with torch.no_grad():
                feats = self._resnet_encoder(batch)
            if feats.ndim > 2:
                feats = feats.flatten(1)
            return feats.float().unsqueeze(1)

        # Lightweight fallback without torchvision: downsample raw pixels into 2048 dims.
        vecs: List[torch.Tensor] = []
        for im in images:
            arr = np.asarray(im.resize((32, 32)), dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).reshape(1, 1, -1)
            x = F.adaptive_avg_pool1d(x, 2048).reshape(-1)
            vecs.append(x)
        out = torch.stack(vecs, dim=0).to(self.device)
        return out.float().unsqueeze(1)

    def _filter_model_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only arguments accepted by model.forward unless it has **kwargs.
        """
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(self.model.forward).parameters.values()):
            return inputs
        return {k: v for k, v in inputs.items() if k in self._forward_param_names}

    def _build_inputs(self, images: List[Image.Image], texts: List[str]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None

        if self.processor is not None:
            call_variants = [
                {"images": images, "text": texts},
                {"image": images, "text": texts},
                {"images": images, "texts": texts},
            ]
            for kwargs in call_variants:
                try:
                    built = self.processor(
                        **kwargs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    )
                    built = self._filter_model_inputs(dict(built))
                    if "visual_embeds" in self._forward_param_names and "visual_embeds" not in built:
                        built["visual_embeds"] = self._compute_visual_embeds(images)
                    return built
                except Exception as e:
                    last_err = e

        if self.tokenizer is not None and self.image_processor is not None:
            try:
                text_inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                image_inputs = self.image_processor(
                    images=images,
                    return_tensors="pt",
                )
                merged = dict(text_inputs)
                merged.update(image_inputs)
                merged = self._filter_model_inputs(merged)
                if "visual_embeds" in self._forward_param_names and "visual_embeds" not in merged:
                    merged["visual_embeds"] = self._compute_visual_embeds(images)
                return merged
            except Exception as e:
                last_err = e

        if self.tokenizer is not None:
            try:
                text_inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                merged = self._filter_model_inputs(dict(text_inputs))
                if "visual_embeds" in self._forward_param_names:
                    merged["visual_embeds"] = self._compute_visual_embeds(images)
                return merged
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"Could not build multimodal inputs for model {self.model_name}: {last_err}"
        )

    @staticmethod
    def _extract_score_tensor(outputs: Any) -> torch.Tensor:
        """Extract logits/scores tensor from different model output types."""
        if torch.is_tensor(outputs):
            return outputs

        if hasattr(outputs, "logits") and torch.is_tensor(outputs.logits):
            return outputs.logits

        if isinstance(outputs, dict):
            for key in ("logits", "scores", "probs", "probabilities"):
                val = outputs.get(key)
                if torch.is_tensor(val):
                    return val

        if isinstance(outputs, (tuple, list)) and outputs:
            first = outputs[0]
            if torch.is_tensor(first):
                return first

        raise RuntimeError("Could not extract logits/scores tensor from model outputs")

    def score(self, image_paths: List[str], texts: List[str], batch_size: int = 8) -> List[float]:
        """Return hateful probability per example (higher means more hateful)."""
        if len(image_paths) != len(texts):
            raise ValueError(
                f"image_paths and texts must have same length, got {len(image_paths)} vs {len(texts)}"
            )
        if not texts:
            return []

        probs_out: List[float] = []
        batch_size = max(1, int(batch_size))

        for start in range(0, len(texts), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_texts = texts[start:start + batch_size]

            images = []
            for p in batch_paths:
                try:
                    images.append(self._load_image(p))
                except Exception as e:
                    logger.warning("Failed to load image for multimodal scoring (%s): %s", p, e)
                    images.append(Image.new("RGB", (336, 336), color=(0, 0, 0)))

            inputs = self._build_inputs(images, batch_texts)
            inputs = {
                k: (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = self._extract_score_tensor(outputs)

            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

            if logits.shape[-1] == 1:
                column = logits[:, 0]
                if torch.all((column >= 0.0) & (column <= 1.0)):
                    hateful_probs = column
                else:
                    hateful_probs = torch.sigmoid(column)
            else:
                # Handle both raw logits and already-normalized probabilities.
                if (
                    torch.all((logits >= 0.0) & (logits <= 1.0))
                    and torch.allclose(
                        logits.sum(dim=-1),
                        torch.ones(logits.shape[0], device=logits.device),
                        atol=1e-3,
                    )
                ):
                    soft = logits
                else:
                    soft = torch.softmax(logits, dim=-1)
                idx = min(max(self.positive_index, 0), soft.shape[-1] - 1)
                hateful_probs = soft[:, idx]

            probs_out.extend(hateful_probs.detach().cpu().tolist())

        return [float(x) for x in probs_out]


def compute_multimodal_hatefulness(
    image_paths: List[str],
    texts: List[str],
    scorer: VisualBertMultimodalScorer,
    batch_size: int,
) -> List[float]:
    """Wrapper for image+text hateful probabilities."""
    return scorer.score(image_paths=image_paths, texts=texts, batch_size=batch_size)


def load_existing_ids(jsonl_path: str) -> set:
    """Load IDs of already-processed examples from JSONL file."""
    processed = set()
    if os.path.exists(jsonl_path):
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed.add(data.get("id"))
        except Exception as e:
            logger.warning(f"Could not load existing IDs from {jsonl_path}: {e}")
    return processed


def write_jsonl_batch(data: List[Dict], output_path: str) -> None:
    """Append batch of examples to JSONL file."""
    with open(output_path, "a") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")
DOMAIN_RE = re.compile(
    r"(?i)\b[a-z0-9][a-z0-9-]{1,62}\.(?:com|org|net|co|io|ai|edu|gov|uk|us|ru|de|fr|it|me|ly|info|biz)(?:/\S*)?\b"
)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
LEADING_LABEL_RE = re.compile(
    r"(?i)^\s*(?:rewrite|rewritten text|rewritten_text|output|answer|response)\s*:\s*"
)


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def _token_change_ratio(original_text: str, rewrite_text: str) -> float:
    """
    Compute how much token content changed between original and rewrite.

    Returns value in [0,1], where 0 means "identical bag of tokens"
    and 1 means "no overlap".
    """
    orig_tokens = _normalize_for_compare(original_text).split()
    rew_tokens = _normalize_for_compare(rewrite_text).split()
    if not orig_tokens and not rew_tokens:
        return 0.0
    if not orig_tokens or not rew_tokens:
        return 1.0

    overlap = sum((Counter(orig_tokens) & Counter(rew_tokens)).values())
    denom = max(len(orig_tokens), len(rew_tokens), 1)
    return 1.0 - (overlap / denom)


def sanitize_generated_rewrite(text: str) -> str:
    """
    Deterministically sanitize LLaVA rewrite output into plain sentence text.

    This strips wrappers and removes metadata artifacts that should never
    be learned as rewrite targets.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    if "[/INST]" in cleaned:
        cleaned = cleaned.split("[/INST]")[-1].strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = " ".join(lines).strip()

    cleaned = LEADING_LABEL_RE.sub("", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = MENTION_RE.sub(" ", cleaned)
    cleaned = HASHTAG_RE.sub(" ", cleaned)
    cleaned = URL_RE.sub(" ", cleaned)
    cleaned = DOMAIN_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\u2022", " ").replace("\ufffd", " ")
    cleaned = re.sub(r"([!?.,;:])\1{2,}", r"\1", cleaned)
    cleaned = re.sub(r"\s+([!?.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("\"'` ").strip()
    return cleaned


def has_invalid_rewrite_format(
    rewrite: str,
    original_text: str,
    min_lexical_change: float = 0.0,
    max_char_similarity: float = 1.0,
) -> tuple[bool, str]:
    """
    Reject rewrites with URLs/artifacts, extreme repetition, or no real edit.
    """
    text = (rewrite or "").strip()
    if not text:
        return True, "empty"

    if URL_RE.search(text) or DOMAIN_RE.search(text):
        return True, "url"
    if MENTION_RE.search(text):
        return True, "mention"
    if HASHTAG_RE.search(text):
        return True, "hashtag"

    tokens = text.split()
    if len(tokens) < 2:
        return True, "too_short"

    if len(text) > 280:
        return True, "too_long"

    lower_tokens = [t.lower() for t in tokens]
    if len(tokens) >= 8:
        unique_ratio = len(set(lower_tokens)) / max(len(lower_tokens), 1)
        if unique_ratio < 0.35:
            return True, "low_diversity"
        counts = {}
        for tok in lower_tokens:
            counts[tok] = counts.get(tok, 0) + 1
        if (max(counts.values()) / len(lower_tokens)) > 0.45:
            return True, "repetition"

    non_alnum_ratio = sum(
        1 for c in text if (not c.isalnum() and not c.isspace())
    ) / max(len(text), 1)
    if non_alnum_ratio > 0.35:
        return True, "symbol_heavy"

    if _normalize_for_compare(text) == _normalize_for_compare(original_text):
        return True, "no_edit"

    # Optional similarity guards (can be relaxed via CLI thresholds).
    if min_lexical_change > 0.0:
        token_change = _token_change_ratio(original_text, text)
        if token_change < min_lexical_change:
            return True, "too_similar"

    if max_char_similarity < 1.0:
        original_norm = _normalize_for_compare(original_text)
        rewrite_norm = _normalize_for_compare(text)
        if len(original_norm) >= 24:
            char_similarity = SequenceMatcher(None, original_norm, rewrite_norm).ratio()
            if char_similarity > max_char_similarity:
                return True, "too_similar"

    return False, ""


def ensure_explanation_non_null(explanation: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """Ensure explanations always have non-null conditioning fields."""
    normalized = dict(explanation) if isinstance(explanation, dict) else {}
    changed = False

    target_group = normalized.get("target_group")
    if not isinstance(target_group, str) or not target_group.strip() or target_group.strip().lower() in {
        "null", "none", "n/a", "na", "unknown"
    }:
        normalized["target_group"] = "other"
        changed = True

    visual_evidence = normalized.get("visual_evidence")
    if (
        not isinstance(visual_evidence, str)
        or not visual_evidence.strip()
        or visual_evidence.strip().lower() in {"null", "none", "n/a", "na", "unknown"}
    ):
        normalized["visual_evidence"] = (
            "A visual cue in the meme is used to frame the target group negatively."
        )
        changed = True

    implicit_meaning = normalized.get("implicit_meaning")
    if (
        not isinstance(implicit_meaning, str)
        or not implicit_meaning.strip()
        or implicit_meaning.strip().lower() in {"null", "none", "n/a", "na", "unknown"}
    ):
        normalized["implicit_meaning"] = (
            "The meme uses both text and visual context to communicate a hateful or derogatory framing toward a target group."
        )
        changed = True

    return normalized, changed


def main():
    parser = argparse.ArgumentParser(description="Stage 1 (multimodal + sharded): Generate explanations and pseudo-rewrites")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., 'training')")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest CSV from Stage 0 (output of filter_meme_images.py)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSONL files")
    parser.add_argument("--hf_cache", type=str, default="./hf_cache", help="Hugging Face cache directory")
    parser.add_argument(
        "--multimodal_model_name",
        type=str,
        default="chiragmittal92/visualbert-hateful-memes-finetuned-model",
        help="HuggingFace model for multimodal (image+text) hatefulness scoring",
    )
    parser.add_argument("--load_in_4bit", action="store_true", help="Load LLaVA in 4-bit quantization")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--score_batch_size", type=int, default=8, help="Batch size for multimodal hatefulness scorer")
    parser.add_argument("--num_shards", type=int, default=8, help="Total number of shards for parallel processing")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id to process in [0, num_shards-1]")
    parser.add_argument("--hateful_only", action="store_true", help="Only process examples where hateful=1 (skip non-hateful memes)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process max 16 examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sta_threshold", type=float, default=0.45, help="Minimum non-hateful score (1 - hatefulness) for keeping rewrites")
    parser.add_argument("--bertscore_min", type=float, default=0.25, help="Minimum BERTScore similarity for keeping rewrites")
    parser.add_argument("--bertscore_max", type=float, default=1.0, help="Maximum BERTScore similarity to avoid near-copy rewrites (set 1.0 to disable)")
    parser.add_argument("--min_lexical_change", type=float, default=0.0, help="Minimum token-level change ratio required between original and rewrite (set 0.0 to disable)")
    parser.add_argument("--max_char_similarity", type=float, default=1.0, help="Maximum normalized char-level similarity allowed between original and rewrite (set 1.0 to disable)")
    parser.add_argument("--min_toxicity_drop", type=float, default=0.0, help="Minimum required hatefulness decrease from original to rewrite (set 0.0 to disable)")
    parser.add_argument("--min_source_toxicity_for_drop", type=float, default=0.20, help="Only enforce min_toxicity_drop when original hatefulness is at least this value")
    parser.add_argument("--explain_max_retries", type=int, default=0, help="Additional retries for explanation generation (0 => single attempt)")
    parser.add_argument("--rewrite_max_attempts", type=int, default=2, help="Maximum rewrite attempts per example")

    args = parser.parse_args()
    if not (0.0 <= args.sta_threshold <= 1.0):
        raise ValueError("--sta_threshold must be in [0,1]")
    if not (0.0 <= args.bertscore_min <= 1.0):
        raise ValueError("--bertscore_min must be in [0,1]")
    if not (0.0 <= args.bertscore_max <= 1.0):
        raise ValueError("--bertscore_max must be in [0,1]")
    if args.bertscore_max <= args.bertscore_min:
        raise ValueError("--bertscore_max must be greater than --bertscore_min")
    if not (0.0 <= args.min_lexical_change <= 1.0):
        raise ValueError("--min_lexical_change must be in [0,1]")
    if not (0.0 <= args.max_char_similarity <= 1.0):
        raise ValueError("--max_char_similarity must be in [0,1]")
    if not (0.0 <= args.min_toxicity_drop <= 1.0):
        raise ValueError("--min_toxicity_drop must be in [0,1]")
    if not (0.0 <= args.min_source_toxicity_for_drop <= 1.0):
        raise ValueError("--min_source_toxicity_for_drop must be in [0,1]")
    if args.explain_max_retries < 0:
        raise ValueError("--explain_max_retries must be >= 0")
    if args.rewrite_max_attempts < 1:
        raise ValueError("--rewrite_max_attempts must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.score_batch_size < 1:
        raise ValueError("--score_batch_size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards-1]")

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    shard_tag = f"shard{args.shard_id:02d}of{args.num_shards:02d}"
    stage1_log_path = os.path.join(args.output_dir, f"stage1_{shard_tag}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(stage1_log_path),
            logging.StreamHandler()
        ]
    )

    # Keep console readable: show our pipeline logs + tqdm, suppress noisy deps.
    for noisy_logger in [
        "httpx",
        "huggingface_hub",
        "urllib3",
        "matplotlib",
        "PIL",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    print(f"\n{'='*60}")
    print(f"  Stage 1: LLaVA Explanations + Pseudo-rewrites")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Images:     {args.images_dir}")
    print(f"  Manifest:   {args.manifest_path}")
    print(f"  Output:     {args.output_dir}")
    print(f"  HF cache:   {args.hf_cache}")
    print(f"  4-bit quant:{args.load_in_4bit}")
    print(f"  Scorer:     {args.multimodal_model_name}")
    print(f"  Shard:      {args.shard_id + 1}/{args.num_shards} ({shard_tag})")
    print(f"  Debug:      {args.debug}")
    print(f"{'='*60}\n")
    logger.info(f"Starting Stage 1 with dataset={args.dataset}, debug={args.debug}")
    logger.info(f"Arguments: {vars(args)}")

    # Load manifest
    manifest_df = pd.read_csv(args.manifest_path)
    total_in_manifest = len(manifest_df)
    kept_in_manifest = int(manifest_df["kept"].sum()) if "kept" in manifest_df.columns else total_in_manifest
    logger.info(f"Manifest loaded: {total_in_manifest} total rows, {kept_in_manifest} kept by Stage 0")
    manifest_df = manifest_df[manifest_df["kept"] == True] if "kept" in manifest_df.columns else manifest_df
    if args.hateful_only and "hateful" in manifest_df.columns:
        before = len(manifest_df)
        manifest_df = manifest_df[manifest_df["hateful"] == 1]
        logger.info(f"--hateful_only: kept {len(manifest_df)}/{before} hateful examples")
    if args.debug:
        manifest_df = manifest_df.head(16)
    logger.info(f"Manifest rows after filters (before sharding): {len(manifest_df)}")

    # Initialize models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("No GPU found — running on CPU (will be slow)")

    explainer = MemeExplainer(
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.hf_cache,
        device=device,
        debug=args.debug,
    )

    hate_scorer = VisualBertMultimodalScorer(
        model_name=args.multimodal_model_name,
        device=device,
        cache_dir=args.hf_cache,
    )

    # Load BERTScorer once — reusing it per example avoids reloading the model every call
    bertscore_scorer = create_bertscore_scorer(device=device)

    # Prepare output paths
    explanations_path = os.path.join(args.output_dir, f"{args.dataset}_explanations_{shard_tag}.jsonl")
    pseudo_rewrites_path = os.path.join(args.output_dir, f"{args.dataset}_pseudo_rewrites_{shard_tag}.jsonl")

    # Load already-processed IDs for resume
    processed_explanation_ids = load_existing_ids(explanations_path)
    processed_rewrite_ids = load_existing_ids(pseudo_rewrites_path)

    logger.info(f"Already processed explanations: {len(processed_explanation_ids)}")
    logger.info(f"Already processed rewrites: {len(processed_rewrite_ids)}")

    # Process examples
    explanations_batch = []
    rewrites_batch = []
    json_parse_failures = 0
    forced_non_null_explanations = 0
    total_examples = 0
    kept_rewrites = 0
    total_pseudo_rewrites = 0
    invalid_rewrite_format = 0
    invalid_rewrite_reason_counts: Dict[str, int] = {}
    quality_reject_count = 0
    quality_reject_reason_counts: Dict[str, int] = {}
    rewrite_generation_failures = 0

    emissions_file = f"emissions_{shard_tag}.csv"
    tracker = EmissionsTracker(log_level="warning", output_dir=args.output_dir, output_file=emissions_file)
    tracker.start()

    try:
        all_records = manifest_df.to_dict("records")
        records = [row for idx, row in enumerate(all_records) if idx % args.num_shards == args.shard_id]
        logger.info(
            "Shard %d/%d selected %d/%d rows",
            args.shard_id,
            args.num_shards,
            len(records),
            len(all_records),
        )
        next_stats_log = 50
        next_explanations_flush = 100
        next_rewrites_flush = 100

        with tqdm.tqdm(total=len(records), desc="Processing examples") as pbar:
            for start_idx in range(0, len(records), args.batch_size):
                raw_batch_rows = records[start_idx:start_idx + args.batch_size]
                pending_rows = []

                for row in raw_batch_rows:
                    example_id = row.get("id")
                    if example_id in processed_explanation_ids:
                        pbar.update(1)
                        continue
                    pending_rows.append(row)

                if not pending_rows:
                    continue

                batch_ids: List[str] = []
                batch_image_paths: List[str] = []
                batch_original_texts: List[str] = []
                batch_hateful_flags: List[bool] = []

                for row in pending_rows:
                    example_id = row.get("id")
                    raw_img = str(row.get("image_path", ""))
                    if os.path.isabs(raw_img) and os.path.exists(raw_img):
                        image_path = raw_img
                    else:
                        image_path = os.path.join(args.images_dir, raw_img)
                    original_text = str(row.get("text", "") or row.get("ocr_text", ""))
                    is_hateful = bool(row.get("hateful", False))

                    batch_ids.append(example_id)
                    batch_image_paths.append(image_path)
                    batch_original_texts.append(original_text)
                    batch_hateful_flags.append(is_hateful)

                total_examples += len(pending_rows)

                try:
                    batch_explanations = explainer.batch_explain(
                        batch_image_paths,
                        batch_original_texts,
                        max_retries=args.explain_max_retries,
                    )
                except Exception as e:
                    logger.warning(
                        f"Batch explanation generation failed for rows "
                        f"{start_idx}-{start_idx + len(pending_rows) - 1}: {e}. "
                        f"Falling back to per-example explain."
                    )
                    batch_explanations = []
                    for image_path, original_text, is_hateful in zip(
                        batch_image_paths, batch_original_texts, batch_hateful_flags
                    ):
                        try:
                            explanation = explainer.explain(
                                image_path,
                                original_text,
                                max_retries=args.explain_max_retries,
                            )
                        except Exception as inner_e:
                            if is_hateful:
                                explanation = {
                                    "target_group": "other",
                                    "visual_evidence": (
                                        "A visual cue in the meme is used to frame the target group negatively."
                                    ),
                                    "implicit_meaning": (
                                        "The meme uses both text and visual context to communicate a hateful or derogatory framing toward a target group."
                                    ),
                                    "error": str(inner_e),
                                }
                            else:
                                explanation = {
                                    "target_group": None,
                                    "visual_evidence": None,
                                    "implicit_meaning": None,
                                    "error": str(inner_e),
                                }
                        batch_explanations.append(explanation)

                for i, row in enumerate(pending_rows):
                    example_id = batch_ids[i]
                    explanation = batch_explanations[i]
                    is_hateful = batch_hateful_flags[i]
                    original_text = batch_original_texts[i]

                    if explanation.get("parse_error"):
                        json_parse_failures += 1

                    if is_hateful:
                        explanation, was_forced = ensure_explanation_non_null(explanation)
                        if was_forced:
                            forced_non_null_explanations += 1

                    batch_explanations[i] = explanation

                    explanation_record = {
                        "id": example_id,
                        "image_path": row.get("image_path"),
                        "original_text": original_text,
                        "explanation": explanation,
                        "is_hateful": is_hateful,
                    }
                    explanations_batch.append(explanation_record)
                    processed_explanation_ids.add(example_id)

                rewrite_positions = [
                    i for i, example_id in enumerate(batch_ids)
                    if batch_hateful_flags[i] and example_id not in processed_rewrite_ids
                ]

                if rewrite_positions:
                    total_pseudo_rewrites += len(rewrite_positions)
                    rw_ids = [batch_ids[i] for i in rewrite_positions]
                    rw_image_paths = [batch_image_paths[i] for i in rewrite_positions]
                    rw_original_texts = [batch_original_texts[i] for i in rewrite_positions]
                    rw_explanations = [batch_explanations[i] for i in rewrite_positions]
                    rw_original_toxicities = compute_multimodal_hatefulness(
                        rw_image_paths,
                        rw_original_texts,
                        hate_scorer,
                        batch_size=args.score_batch_size,
                    )

                    cleaned_rewrites: List[Optional[str]] = [None] * len(rewrite_positions)
                    active_indices = list(range(len(rewrite_positions)))
                    max_rewrite_attempts = args.rewrite_max_attempts

                    for attempt_idx in range(max_rewrite_attempts):
                        if not active_indices:
                            break

                        active_image_paths = [rw_image_paths[i] for i in active_indices]
                        active_original_texts = [rw_original_texts[i] for i in active_indices]
                        active_explanations = [rw_explanations[i] for i in active_indices]
                        active_ids = [rw_ids[i] for i in active_indices]

                        try:
                            raw_rewrites = explainer.batch_rewrite(
                                active_image_paths,
                                active_original_texts,
                                active_explanations,
                            )
                        except Exception as e:
                            rewrite_generation_failures += len(active_indices)
                            logger.warning(
                                "Batch rewrite generation failed at attempt %d/%d for %d examples: %s",
                                attempt_idx + 1,
                                max_rewrite_attempts,
                                len(active_indices),
                                e,
                            )
                            break

                        unresolved: List[int] = []
                        for pos, active_slot in enumerate(active_indices):
                            example_id = active_ids[pos]
                            raw_rewrite = raw_rewrites[pos] if pos < len(raw_rewrites) else ""

                            if isinstance(raw_rewrite, str) and raw_rewrite.startswith("[REWRITE ERROR:"):
                                rewrite_generation_failures += 1
                                logger.warning(
                                    f"Rewrite generation failed for {example_id} "
                                    f"(attempt {attempt_idx + 1}/{max_rewrite_attempts}): {raw_rewrite}"
                                )
                                unresolved.append(active_slot)
                                continue

                            candidate = sanitize_generated_rewrite(raw_rewrite)
                            is_invalid, reason = has_invalid_rewrite_format(
                                candidate,
                                rw_original_texts[active_slot],
                                min_lexical_change=args.min_lexical_change,
                                max_char_similarity=args.max_char_similarity,
                            )
                            if is_invalid:
                                invalid_rewrite_format += 1
                                invalid_rewrite_reason_counts[reason] = (
                                    invalid_rewrite_reason_counts.get(reason, 0) + 1
                                )
                                logger.info(
                                    f"Rejected rewrite for {example_id} "
                                    f"(attempt {attempt_idx + 1}/{max_rewrite_attempts}, reason={reason})"
                                )
                                unresolved.append(active_slot)
                                continue

                            cleaned_rewrites[active_slot] = candidate

                        active_indices = unresolved

                    kept_slots = [i for i, text in enumerate(cleaned_rewrites) if text]
                    if kept_slots:
                        kept_rewrite_texts = [cleaned_rewrites[i] for i in kept_slots]
                        kept_original_texts = [rw_original_texts[i] for i in kept_slots]
                        kept_image_paths = [rw_image_paths[i] for i in kept_slots]
                        kept_original_toxicities = [rw_original_toxicities[i] for i in kept_slots]

                        rewrite_toxicities = compute_multimodal_hatefulness(
                            kept_image_paths,
                            kept_rewrite_texts,
                            hate_scorer,
                            batch_size=args.score_batch_size,
                        )
                        sta_scores = [1.0 - tox for tox in rewrite_toxicities]
                        bertscores = compute_bertscore_batch(
                            kept_original_texts,
                            kept_rewrite_texts,
                            scorer=bertscore_scorer,
                            batch_size=max(32, args.batch_size * 8),
                        )

                        for slot_idx, rewrite, sta_score, rewrite_toxicity, bertscore, original_toxicity in zip(
                            kept_slots,
                            kept_rewrite_texts,
                            sta_scores,
                            rewrite_toxicities,
                            bertscores,
                            kept_original_toxicities,
                        ):
                            example_id = rw_ids[slot_idx]
                            row_idx = rewrite_positions[slot_idx]
                            source_row = pending_rows[row_idx]
                            explanation = rw_explanations[slot_idx]
                            original_text = rw_original_texts[slot_idx]

                            toxicity_drop = original_toxicity - rewrite_toxicity
                            required_toxicity_drop = (
                                args.min_toxicity_drop
                                if original_toxicity >= args.min_source_toxicity_for_drop
                                else 0.0
                            )

                            passes_sta = sta_score > args.sta_threshold
                            passes_bertscore = bertscore > args.bertscore_min
                            if args.bertscore_max < 1.0:
                                passes_bertscore = passes_bertscore and (bertscore < args.bertscore_max)
                            passes_toxicity_delta = toxicity_drop >= required_toxicity_drop

                            if passes_sta and passes_bertscore and passes_toxicity_delta:
                                kept_rewrites += 1
                                rewrite_record = {
                                    "id": example_id,
                                    "image_path": source_row.get("image_path"),
                                    "original_text": original_text,
                                    "explanation": explanation,
                                    "pseudo_rewrite": rewrite,
                                    "sta_score": float(sta_score),
                                    "bertscore": float(bertscore),
                                    "original_toxicity": float(original_toxicity),
                                    "rewrite_toxicity": float(rewrite_toxicity),
                                    "toxicity_drop": float(toxicity_drop),
                                }
                                rewrites_batch.append(rewrite_record)
                                processed_rewrite_ids.add(example_id)
                            else:
                                quality_reject_count += 1
                                if not passes_sta:
                                    q_reason = "low_sta"
                                elif bertscore <= args.bertscore_min:
                                    q_reason = "low_bertscore"
                                elif bertscore >= args.bertscore_max:
                                    q_reason = "high_bertscore"
                                else:
                                    q_reason = "low_toxicity_drop"
                                quality_reject_reason_counts[q_reason] = (
                                    quality_reject_reason_counts.get(q_reason, 0) + 1
                                )

                pbar.update(len(pending_rows))

                if total_examples >= next_stats_log:
                    keep_rate = 100 * kept_rewrites / max(total_pseudo_rewrites, 1)
                    logger.info(
                        f"[{total_examples}/{len(records)}] "
                        f"explanations={total_examples} | "
                        f"rewrites_kept={kept_rewrites}/{total_pseudo_rewrites} ({keep_rate:.1f}%) | "
                        f"json_failures={json_parse_failures} | "
                        f"forced_non_null={forced_non_null_explanations} | "
                        f"rewrite_invalid={invalid_rewrite_format} | "
                        f"quality_rejected={quality_reject_count} | "
                        f"rewrite_failures={rewrite_generation_failures}"
                    )
                    next_stats_log = ((total_examples // 50) + 1) * 50

                if explanations_batch and total_examples >= next_explanations_flush:
                    write_jsonl_batch(explanations_batch, explanations_path)
                    explanations_batch = []
                    logger.info(f"Wrote explanations batch at example {total_examples}")
                    next_explanations_flush = ((total_examples // 100) + 1) * 100

                if rewrites_batch and total_pseudo_rewrites >= next_rewrites_flush:
                    write_jsonl_batch(rewrites_batch, pseudo_rewrites_path)
                    rewrites_batch = []
                    logger.info(f"Wrote rewrites batch at example {total_pseudo_rewrites}")
                    next_rewrites_flush = ((total_pseudo_rewrites // 100) + 1) * 100

        # Write final batches
        if explanations_batch:
            write_jsonl_batch(explanations_batch, explanations_path)
            logger.info(f"Wrote final explanations batch ({len(explanations_batch)} items)")

        if rewrites_batch:
            write_jsonl_batch(rewrites_batch, pseudo_rewrites_path)
            logger.info(f"Wrote final rewrites batch ({len(rewrites_batch)} items)")

        # Compute metrics
        json_parse_rate = (json_parse_failures / max(total_examples, 1)) * 100
        keep_rate = (kept_rewrites / max(total_pseudo_rewrites, 1)) * 100

        logger.info(f"\n=== Stage 1 Summary ===")
        logger.info(f"Total examples processed: {total_examples}")
        logger.info(f"JSON parse failures: {json_parse_failures} ({json_parse_rate:.2f}%)")
        logger.info(f"Hateful explanations forced to non-null: {forced_non_null_explanations}")
        logger.info(f"Total pseudo-rewrites generated: {total_pseudo_rewrites}")
        logger.info(f"Pseudo-rewrites kept (passed filters): {kept_rewrites}")
        logger.info(f"Keep rate: {keep_rate:.2f}%")
        logger.info(f"Rejected rewrites due to invalid format: {invalid_rewrite_format}")
        if invalid_rewrite_reason_counts:
            logger.info(f"Invalid rewrite reasons: {invalid_rewrite_reason_counts}")
        logger.info(f"Rejected rewrites due to quality filter: {quality_reject_count}")
        if quality_reject_reason_counts:
            logger.info(f"Quality reject reasons: {quality_reject_reason_counts}")
        logger.info(f"Rewrite generation failures: {rewrite_generation_failures}")
        logger.info(f"Explanations JSONL: {explanations_path}")
        logger.info(f"Pseudo-rewrites JSONL: {pseudo_rewrites_path}")

    finally:
        emissions = tracker.stop()
        if emissions is not None:
            logger.info(f"Carbon emissions: {emissions:.6f} kg CO2")
            logger.info(f"Emissions saved to: {os.path.join(args.output_dir, emissions_file)}")
        else:
            logger.warning("Carbon emissions could not be measured (CodeCarbon tracking failed)")


if __name__ == "__main__":
    main()
