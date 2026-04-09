"""
Central debug configuration for the hateful meme rewriting pipeline.
All scripts import DEBUG settings from here to ensure consistent behavior.
"""

import os
import random
from typing import List, Dict, Any

DEBUG_CONFIG = {
    "max_samples": 16,
    # Stage 1: same model but batch_size=1 to reduce memory
    "stage1_model": "llava-hf/llava-v1.6-mistral-7b-hf",
    "stage1_batch_size": 1,
    # Stage 2: replace bart-large with bart-base
    "stage2_model": "facebook/bart-base",
    "num_train_epochs": 1,
    "max_steps": 4,
    "save_steps": 2,
    "eval_steps": 2,
    "logging_steps": 1,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "bf16": False,
    "fp16": False,
    "inference_batch_size": 1,
    "skip_bertscore": True,
    "dummy_bertscore": 0.5,
    "output_dir_suffix": "_debug",
    # Proxy network: reduced architecture for CPU
    "proxy_hidden_dim": 256,
    "proxy_epochs": 2,
    "proxy_batch_size": 4,
    "proxy_max_steps": 4,
    # bart-base hidden size is 768 (not 1024); proxy output dim must match
    "bart_hidden_size": 768,
}


def is_debug_mode(args=None) -> bool:
    """Check if debug mode is active from args or environment variable."""
    env_debug = bool(int(os.environ.get("DEBUG", "0")))
    if args is not None and hasattr(args, "debug"):
        return args.debug or env_debug
    return env_debug


def make_debug_dataset(n: int = 16) -> List[Dict[str, Any]]:
    """
    Returns synthetic in-memory examples mimicking real dataset structure.
    Fields: id, image_path, text, label, target_group, attack_type, explanation
    image_path values do NOT need to exist on disk.
    """
    examples = []

    hateful_texts = [
        "These people are ruining our country",
        "They don't belong here go back",
        "All of them are criminals and thugs",
        "Women can't do anything right",
        "They are like animals not humans",
        "Keep them out of our neighborhood",
        "These illegals need to be deported",
        "They are all terrorists watch out",
    ]

    safe_texts = [
        "What a beautiful day outside",
        "I love spending time with family",
        "This recipe turned out amazing",
        "Great game last night",
        "Learning something new today",
        "The sunset is so pretty",
        "Having a good time at the park",
        "My dog is the best companion",
    ]

    target_groups = ["race_ethnicity", "nationality", "religion", "gender",
                     "sexual_orientation", "disability", "other"]
    attack_types = ["contempt", "mocking", "inferiority", "slurs",
                    "exclusion", "dehumanizing", "inciting_violence"]

    for i in range(n):
        if i < n // 2:
            text = hateful_texts[i % len(hateful_texts)]
            label = 1
            tg = target_groups[i % len(target_groups)]
            at = attack_types[i % len(attack_types)]
            explanation = {
                "target_group": tg,
                "attack_type": at,
                "implicit_meaning": f"implies {tg} are inferior through {at}"
            }
        else:
            text = safe_texts[i % len(safe_texts)]
            label = 0
            tg = None
            at = None
            explanation = {
                "target_group": None,
                "attack_type": None,
                "implicit_meaning": None
            }

        examples.append({
            "id": f"debug_{i:04d}",
            "image_path": f"/tmp/debug_images/debug_{i:04d}.jpg",
            "text": text,
            "label": label,
            "dataset": "debug",
            "target_group": tg,
            "attack_type": at,
            "explanation": explanation,
        })

    return examples


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass
