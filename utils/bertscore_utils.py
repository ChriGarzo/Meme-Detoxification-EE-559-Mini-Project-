"""
BERTScore batch computation helpers for the hateful meme rewriting pipeline.

The main usage pattern is: call create_bertscore_scorer() once to load model
weights, then pass the returned scorer to compute_bertscore_batch() in a loop.
This avoids per-call model reloading and is used by run_stage1.py to quality-
filter pseudo-rewrites (threshold: F1 > 0.4).

Returns rescaled roberta-large F1 by default; scores are approximately in [0, 1]
after baseline rescaling but can be negative on very poor candidates.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def create_bertscore_scorer(
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = True,
    device: Optional[str] = None,
):
    """Load and return a BERTScorer; returns None if bert_score is not installed."""
    try:
        from bert_score import BERTScorer
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BERTScorer ({model_type}) on {device} ...")
        return BERTScorer(
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )
    except ImportError:
        logger.warning("bert_score package not installed. Returning None scorer.")
        return None


def compute_bertscore_batch(
    references: List[str],
    candidates: List[str],
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = True,
    batch_size: int = 64,
    device: Optional[str] = None,
    scorer=None,
) -> List[float]:
    """Return BERTScore F1 for each (reference, candidate) pair.

    Pass a pre-loaded scorer from create_bertscore_scorer() to avoid reloading
    model weights on every call. Returns 0.5 on ImportError and 0.0 on runtime
    failure so upstream quality filters degrade gracefully.

    scorer: pre-loaded BERTScorer; if None a fresh scorer is constructed.
    """
    if len(references) != len(candidates):
        raise ValueError(
            f"references and candidates must have the same length, "
            f"got {len(references)} vs {len(candidates)}"
        )

    if not references:
        return []

    try:
        if scorer is not None:
            _, _, F1 = scorer.score(
                cands=candidates,
                refs=references,
                batch_size=batch_size,
                verbose=False,
            )
            return F1.tolist()

        from bert_score import score as bert_score_fn
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        _, _, F1 = bert_score_fn(
            cands=candidates,
            refs=references,
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            batch_size=batch_size,
            device=device,
            verbose=False,
        )
        return F1.tolist()

    except ImportError:
        logger.warning("bert_score not installed; returning 0.5 fallback (pip install bert-score)")
        return [0.5] * len(references)
    except Exception as e:
        logger.error("BERTScore computation failed: %s", e)
        return [0.0] * len(references)
