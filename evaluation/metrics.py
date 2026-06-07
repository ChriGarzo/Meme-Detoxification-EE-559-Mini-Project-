"""
Metric computation for hateful meme detoxification evaluation.

Provides three primary metrics:
  compute_sta       — text-only Safety through Toxicity Attenuation score
                      (mean P(non-toxic) from s-nlp/roberta_toxicity_classifier).
  compute_sim       — BERTScore F1 (roberta-large, baseline-rescaled) between
                      original and rewritten meme text.
  compute_clipscore — normalised image-text cosine similarity via
                      openai/clip-vit-large-patch14.

Module-level cache (_TEXT_STA_CACHE) prevents redundant model loading across calls.
"""
import logging
from typing import List, Dict, Tuple, Union, Optional, Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bert_score import score as bert_score
from codecarbon import EmissionsTracker

from utils.debug import is_debug_mode, DEBUG_CONFIG

logger = logging.getLogger(__name__)

_TEXT_STA_CACHE = {}


def compute_sta(texts: List[str]) -> Dict:
    """
    Compute text STA (Safety through Toxicity Attenuation) scores.

    Returns mean P(non-toxic) across all input texts using
    s-nlp/roberta_toxicity_classifier (class 0 = non-toxic).
    Result dict keys: mean, std, per_example.
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
            non_toxic_prob = probs[0, 0].item()  # class 0 = non-toxic
            scores.append(non_toxic_prob)

    scores = np.array(scores)
    result = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "per_example": scores.tolist()
    }
    logger.info(f"STA mean: {result['mean']:.4f}, std: {result['std']:.4f}")
    return result


def compute_sim(originals: List[str], rewrites: List[str]) -> Dict:
    """
    Compute SIM (semantic similarity) using BERTScore F1 (roberta-large, rescaled).

    Rescaling with the baseline corrects for the high absolute values typical
    of BERTScore; the resulting F1 is more interpretable as a similarity metric.
    Result dict keys: mean, std, per_example.
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
    Compute CLIPScore: normalised image-text cosine similarity via clip-vit-large-patch14.

    Cosine similarities are linearly mapped from [−1, 1] to [0, 1] for
    interpretability. Text inputs are truncated to CLIP's 77-token context limit.
    Result dict keys: mean, std, per_example.
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
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")

            # CLIP's text encoder has a hard 77-token context window;
            # truncation applies only to the internal embedding, not to the
            # stored rewrite text.
            inputs = processor(
                text=text if isinstance(text, str) else "",
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            image_emb = outputs.image_embeds
            text_emb = outputs.text_embeds

            cosine_sim = torch.nn.functional.cosine_similarity(image_emb, text_emb).item()
            # Map cosine similarity from [−1, 1] to [0, 1].
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
    Compute Rewrite Precision: fraction of rewrites where Stage 1 visual evidence changes or disappears.

    Re-runs LLaVA (MemeExplainer) on each (image, rewrite) pair; a rewrite is
    deemed successful if the predicted visual_evidence is None or differs from
    the original, indicating that the hateful visual cue is no longer grounded.
    Result dict keys: mean, per_example.
    """
    logger.info(f"Computing Rewrite Precision for {len(images)} rewrites")

    successes = []
    for i, (image, rewrite, orig_expl) in enumerate(zip(images, rewrites, original_explanations)):
        # Get Stage 1 explanation for rewrite
        new_expl = explainer.explain(image, rewrite)

        orig_visual = orig_expl.get("visual_evidence")
        new_visual = new_expl.get("visual_evidence")

        # Null or changed visual evidence signals the hateful cue is decoupled from the rewrite.
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
    Measure CO2 emissions of a single function call using codecarbon.

    Pass `_emissions_output_dir` as a keyword argument to control the
    directory where emissions.csv is written; defaults to a temporary directory.
    Returns emissions in grams, or None if measurement fails.
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
    Compute the aggregate J score: J = STA × SIM × CLIPScore × Rewrite_Precision.

    All four metric dicts must contain a 'mean' key. Returns a scalar in [0, 1].
    """
    J = sta["mean"] * sim["mean"] * clip["mean"] * rp["mean"]
    logger.info(f"Aggregate J: {J:.6f}")
    return float(J)
