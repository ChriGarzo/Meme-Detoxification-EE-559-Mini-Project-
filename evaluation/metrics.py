"""
Metric computation functions for hateful meme text detoxification.
"""
import logging
from typing import List, Dict, Tuple, Union, Optional, Callable
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer
from bert_score import score as bert_score
from codecarbon import EmissionsTracker
from huggingface_hub import hf_hub_download

from utils.debug import is_debug_mode, DEBUG_CONFIG

logger = logging.getLogger(__name__)

_TEXT_STA_CACHE = {}
_VISUALBERT_CACHE = {}


def _load_visualbert_multimodal_models(
    hf_cache: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """
    Load VisualBERT hate classifier + CLIP visual encoder used for multimodal hate scoring.
    """
    from transformers import CLIPProcessor, CLIPModel, VisualBertConfig, VisualBertModel

    mm_model_id = "chiragmittal92/visualbert-hateful-memes-finetuned-model"
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (mm_model_id, str(hf_cache), str(device))
    if cache_key in _VISUALBERT_CACHE:
        return _VISUALBERT_CACHE[cache_key]

    class _VBClassifier(nn.Module):
        """
        Wrapper compatible with chiragmittal92/visualbert-hateful-memes-finetuned-model
        checkpoint layout.
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
            return self.classifier(out.pooler_output)

    logger.info("Loading VisualBERT multimodal hate model...")
    vb_config = VisualBertConfig.from_pretrained(mm_model_id, cache_dir=hf_cache)
    vb_model = _VBClassifier(vb_config)
    state_dict = None
    load_errors = []
    for weights_name in ("model.safetensors", "pytorch_model.bin"):
        try:
            ckpt_path = hf_hub_download(mm_model_id, weights_name, cache_dir=hf_cache)
            if weights_name.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load_file
                state_dict = safe_load_file(ckpt_path, device="cpu")
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
            logger.info("VisualBERT weights loaded from %s", weights_name)
            break
        except Exception as exc:
            load_errors.append(f"{weights_name}: {exc}")
    if state_dict is None:
        raise RuntimeError(
            "Could not download VisualBERT weights. Tried: "
            + " | ".join(load_errors)
        )
    vb_model.load_state_dict(state_dict, strict=False)
    vb_model = vb_model.to(device).float().eval()
    for p in vb_model.parameters():
        p.requires_grad = False

    vb_tokenizer = BertTokenizer.from_pretrained(mm_model_id, cache_dir=hf_cache)

    logger.info("Loading CLIP visual encoder for VisualBERT features...")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=hf_cache
    ).to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", cache_dir=hf_cache
    )

    loaded = (vb_model, vb_tokenizer, clip_model, clip_processor, device)
    _VISUALBERT_CACHE[cache_key] = loaded
    return loaded


def compute_sta(texts: List[str]) -> Dict:
    """
    Compute STA (Safety through Toxicity Attenuation) score.
    Uses s-nlp/roberta_toxicity_classifier to measure proportion of rewrites classified as non-toxic.

    Args:
        texts: List of rewritten meme texts

    Returns:
        Dict with keys: mean, std, per_example
    """
    logger.info(f"Computing STA for {len(texts)} texts")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "s-nlp/roberta_toxicity_classifier"
    cache_key = (model_name, str(device))
    if cache_key in _TEXT_STA_CACHE:
        tokenizer, model = _TEXT_STA_CACHE[cache_key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        _TEXT_STA_CACHE[cache_key] = (tokenizer, model)

    scores = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            # Class 0 is non-toxic
            non_toxic_prob = probs[0, 0].item()
            scores.append(non_toxic_prob)

    scores = np.array(scores)
    result = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "per_example": scores.tolist()
    }
    logger.info(f"STA mean: {result['mean']:.4f}, std: {result['std']:.4f}")
    return result


def compute_multimodal_hateness(
    images: List[Union[str, Path, Image.Image]],
    rewrites: List[str],
    hf_cache: Optional[str] = None,
    batch_size: int = 16,
) -> Dict:
    """
    Compute multimodal hate scores using VisualBERT + CLIP visual features.

    Returns:
        Dict with:
          - hate_prob_mean: mean P(hateful)
          - non_hate_prob_mean: mean P(non-hateful)
          - hate_pred_rate: fraction of argmax predictions == hateful
          - non_hate_pred_rate: fraction of argmax predictions == non-hateful (multimodal STA analogue)
          - n_valid: number of valid image-text pairs scored
    """
    logger.info("Computing multimodal hateness for %d image-text pairs", len(rewrites))
    if len(images) == 0 or len(rewrites) == 0:
        return {
            "hate_prob_mean": None,
            "non_hate_prob_mean": None,
            "hate_pred_rate": None,
            "non_hate_pred_rate": None,
            "n_valid": 0,
        }

    vb_model, vb_tokenizer, clip_model, clip_processor, device = _load_visualbert_multimodal_models(
        hf_cache=hf_cache
    )

    hate_probs: List[float] = []
    non_hate_probs: List[float] = []
    hate_preds = 0
    total = 0

    pairs = list(zip(images, rewrites))
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]

        batch_images: List[Image.Image] = []
        batch_texts: List[str] = []

        for image, text in batch:
            try:
                if isinstance(image, (str, Path)):
                    img = Image.open(image).convert("RGB")
                else:
                    img = image.convert("RGB")
                batch_images.append(img)
                batch_texts.append(text if isinstance(text, str) else "")
            except Exception as e:
                logger.warning("Skipping invalid image in multimodal metric: %s", e)

        if not batch_images:
            continue

        clip_inputs = clip_processor(images=batch_images, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].to(device)

        with torch.no_grad():
            vision_outputs = clip_model.vision_model(pixel_values=pixel_values)
            img_features = clip_model.visual_projection(vision_outputs.pooler_output).float()  # [B, 768]

        bsz = img_features.shape[0]
        visual_embeds = torch.zeros(bsz, 1, 2048, dtype=torch.float32, device=device)
        visual_embeds[:, 0, :768] = img_features

        text_inputs = vb_tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding="max_length",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            logits = vb_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                visual_embeds=visual_embeds,
            )
            probs = torch.softmax(logits, dim=-1)  # [B,2] 0=non-hateful,1=hateful

        non_h = probs[:, 0].detach().cpu().numpy().tolist()
        hate = probs[:, 1].detach().cpu().numpy().tolist()
        preds = probs.argmax(dim=-1)

        non_hate_probs.extend(float(x) for x in non_h)
        hate_probs.extend(float(x) for x in hate)
        hate_preds += int((preds == 1).sum().item())
        total += bsz

    if total == 0:
        return {
            "hate_prob_mean": None,
            "non_hate_prob_mean": None,
            "hate_pred_rate": None,
            "non_hate_pred_rate": None,
            "n_valid": 0,
        }

    hate_prob_mean = float(np.mean(hate_probs))
    non_hate_prob_mean = float(np.mean(non_hate_probs))
    hate_pred_rate = float(hate_preds / total)
    non_hate_pred_rate = float(1.0 - hate_pred_rate)

    result = {
        "hate_prob_mean": hate_prob_mean,
        "non_hate_prob_mean": non_hate_prob_mean,
        "hate_pred_rate": hate_pred_rate,
        "non_hate_pred_rate": non_hate_pred_rate,
        "n_valid": int(total),
    }
    logger.info(
        "Multimodal hate: hate_prob=%.4f | hate_pred_rate=%.4f | non_hate_pred_rate=%.4f | n=%d",
        hate_prob_mean,
        hate_pred_rate,
        non_hate_pred_rate,
        total,
    )
    return result


def compute_sim(originals: List[str], rewrites: List[str]) -> Dict:
    """
    Compute SIM (Semantic Similarity) using BERTScore.

    Args:
        originals: List of original meme texts
        rewrites: List of rewritten meme texts

    Returns:
        Dict with keys: mean, std, per_example (F1 scores)
    """
    logger.info(f"Computing SIM for {len(originals)} text pairs")

    if is_debug_mode():
        logger.warning("DEBUG mode: skipping BERTScore, returning dummy 0.5")
        dummy_scores = [0.5] * len(originals)
        return {
            "mean": 0.5,
            "std": 0.0,
            "per_example": dummy_scores
        }

    P, R, F1 = bert_score(
        rewrites,
        originals,
        model_type="roberta-large",
        rescale_with_baseline=True,
        lang="en",
        batch_size=32
    )

    f1_scores = F1.cpu().numpy()
    result = {
        "mean": float(np.mean(f1_scores)),
        "std": float(np.std(f1_scores)),
        "per_example": f1_scores.tolist()
    }
    logger.info(f"SIM mean: {result['mean']:.4f}, std: {result['std']:.4f}")
    return result


def compute_clipscore(
    images: List[Union[str, Path, Image.Image]],
    rewrites: List[str]
) -> Dict:
    """
    Compute CLIP-based similarity between images and rewritten text.

    Args:
        images: List of image paths or PIL Image objects
        rewrites: List of rewritten meme texts

    Returns:
        Dict with keys: mean, std, per_example (normalized cosine similarities)
    """
    logger.info(f"Computing CLIPScore for {len(images)} image-text pairs")

    from transformers import CLIPProcessor, CLIPModel

    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for image, text in zip(images, rewrites):
            # Load image if it's a path
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")

            # CLIP's text tower has a hard 77-token context window. Keep the
            # saved/generated rewrite unchanged; truncate only the text view
            # used internally by CLIPScore.
            inputs = processor(
                text=text if isinstance(text, str) else "",
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get embeddings
            outputs = model(**inputs)
            image_emb = outputs.image_embeds
            text_emb = outputs.text_embeds

            # Compute cosine similarity
            cosine_sim = torch.nn.functional.cosine_similarity(image_emb, text_emb).item()
            # Normalize to [0, 1]
            normalized_sim = (cosine_sim + 1) / 2
            scores.append(normalized_sim)

    scores = np.array(scores)
    result = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "per_example": scores.tolist()
    }
    logger.info(f"CLIPScore mean: {result['mean']:.4f}, std: {result['std']:.4f}")
    return result


def compute_rewrite_precision(
    images: List[Union[str, Path, Image.Image]],
    rewrites: List[str],
    original_explanations: List[Dict],
    explainer: "MemeExplainer"
) -> Dict:
    """
    Compute Rewrite Precision: proportion of rewrites where the visual evidence cue changes or disappears.
    Re-runs Stage 1 (LLaVA explain) on (image, rewrite) pairs.

    Args:
        images: List of image paths or PIL Image objects
        rewrites: List of rewritten meme texts
        original_explanations: List of original Stage 1 explanations (dicts with 'visual_evidence' key)
        explainer: MemeExplainer instance

    Returns:
        Dict with keys: mean, per_example
    """
    logger.info(f"Computing Rewrite Precision for {len(images)} rewrites")

    successes = []
    for i, (image, rewrite, orig_expl) in enumerate(zip(images, rewrites, original_explanations)):
        # Get Stage 1 explanation for rewrite
        new_expl = explainer.explain(image, rewrite)

        orig_visual = orig_expl.get("visual_evidence")
        new_visual = new_expl.get("visual_evidence")

        # Success: predicted null OR different visual evidence cue
        is_success = (new_visual is None) or (new_visual != orig_visual)
        successes.append(1.0 if is_success else 0.0)

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(images)} examples")

    successes = np.array(successes)
    result = {
        "mean": float(np.mean(successes)),
        "per_example": successes.tolist()
    }
    logger.info(f"Rewrite Precision: {result['mean']:.4f}")
    return result


def compute_co2(func: Callable, *args, **kwargs) -> float:
    """
    Measure CO2 emissions of a function call using codecarbon.

    Args:
        func: Callable to measure
        *args: Positional arguments to func
        **kwargs: Keyword arguments to func

    Returns:
        CO2 emissions in grams
    """
    import tempfile
    logger.info(f"Measuring CO2 emissions for {func.__name__}")

    output_dir = kwargs.pop("_emissions_output_dir", tempfile.mkdtemp())
    tracker = EmissionsTracker(log_level="warning", output_dir=output_dir, output_file="emissions.csv")
    tracker.start()

    try:
        result = func(*args, **kwargs)
    finally:
        emissions_grams = tracker.stop()

    if emissions_grams is not None:
        logger.info(f"CO2 emissions: {emissions_grams:.4f}g")
    else:
        logger.warning("CO2 emissions could not be measured")
    return emissions_grams


def compute_aggregate_J(
    sta: Dict,
    sim: Dict,
    clip: Dict,
    rp: Dict
) -> float:
    """
    Compute aggregate J metric as product of mean scores.

    J = STA × SIM × CLIPScore × Rewrite_Precision

    Args:
        sta: STA metric dict
        sim: SIM metric dict
        clip: CLIPScore metric dict
        rp: Rewrite Precision metric dict

    Returns:
        Aggregate J score
    """
    J = sta["mean"] * sim["mean"] * clip["mean"] * rp["mean"]
    logger.info(f"Aggregate J: {J:.6f}")
    return float(J)
