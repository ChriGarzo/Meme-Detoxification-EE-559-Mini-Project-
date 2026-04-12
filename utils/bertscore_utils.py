"""
BERTScore utility wrapper for the hateful meme rewriting pipeline.

Used in:
  - run_stage1.py: quality filter for pseudo-rewrites (BERTScore > 0.4)
  - train_stage2_phase1.py: optional filter for ParaDetox pairs (BERTScore > 0.5)
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def compute_bertscore_batch(
    references: List[str],
    candidates: List[str],
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = True,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> List[float]:
    """
    Compute BERTScore F1 for a list of (reference, candidate) pairs.

    Args:
        references:              List of reference strings (originals)
        candidates:              List of candidate strings (rewrites)
        model_type:              HuggingFace model used for BERTScore (default: roberta-large)
        lang:                    Language code (default: 'en')
        rescale_with_baseline:   Whether to rescale scores with a baseline (recommended)
        batch_size:              Batch size for BERTScore computation
        device:                  Device string ('cuda' / 'cpu'). Auto-detected if None.

    Returns:
        List of F1 BERTScore values (one per pair), floats in roughly [0, 1].
        Returns a list of 0.0 values if bert_score is unavailable or fails.
    """
    if len(references) != len(candidates):
        raise ValueError(
            f"references and candidates must have the same length, "
            f"got {len(references)} vs {len(candidates)}"
        )

    if not references:
        return []

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.warning(
            "bert_score package not installed. Returning dummy scores of 0.5. "
            "Install with: pip install bert-score"
        )
        return [0.5] * len(references)

    try:
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

    except Exception as e:
        logger.error(f"BERTScore computation failed: {e}")
        return [0.0] * len(references)
