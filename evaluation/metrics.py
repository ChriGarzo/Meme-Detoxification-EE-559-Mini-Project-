"""
Metric computation functions for hateful meme text detoxification.
"""
import logging
from typing import List, Dict, Tuple, Union, Optional, Callable
from pathlib import Path
import warnings

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bert_score import score as bert_score
from codecarbon import EmissionsTracker

from utils.debug import is_debug_mode, DEBUG_CONFIG

logger = logging.getLogger(__name__)


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

    model_name = "s-nlp/roberta_toxicity_classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

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

            # Process inputs
            inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
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
    Compute Rewrite Precision: proportion of rewrites where the attack type is removed.
    Re-runs Stage 1 (LLaVA explain) on (image, rewrite) pairs.

    Args:
        images: List of image paths or PIL Image objects
        rewrites: List of rewritten meme texts
        original_explanations: List of original Stage 1 explanations (dicts with 'attack_type' key)
        explainer: MemeExplainer instance

    Returns:
        Dict with keys: mean, per_example
    """
    logger.info(f"Computing Rewrite Precision for {len(images)} rewrites")

    successes = []
    for i, (image, rewrite, orig_expl) in enumerate(zip(images, rewrites, original_explanations)):
        # Get Stage 1 explanation for rewrite
        new_expl = explainer.explain(image, rewrite)

        orig_attack = orig_expl.get("attack_type")
        new_attack = new_expl.get("attack_type")

        # Success: predicted null OR different attack type
        is_success = (new_attack is None) or (new_attack != orig_attack)
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
    logger.info(f"Measuring CO2 emissions for {func.__name__}")

    tracker = EmissionsTracker(log_level="warning")
    tracker.start()

    try:
        result = func(*args, **kwargs)
    finally:
        emissions_grams = tracker.stop()

    logger.info(f"CO2 emissions: {emissions_grams:.4f}g")
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
