"""
Screencast demo: CLIP Proxy + BART-large FT Full inference on 20 test images.

Runs the full deployment pipeline (no LLaVA at inference time):
  image + text  ->  CLIP embeddings  ->  Proxy MLP soft tokens
                ->  BART-large FT Full decoder  ->  rewrite

Computes four evaluation metrics per example:
  STA        : non-toxic probability (s-nlp/roberta_toxicity_classifier)
  Delta STA  : STA(rewrite) - STA(original text)
  SIM        : BERTScore F1 (roberta-large, semantic similarity)
  CLIP Score : cosine similarity between image and rewrite text embeddings

Designed for clean, professional log output suitable for screen recording.
"""

# ---------------------------------------------------------------------------
# Warning / noise suppression — must happen before any library imports
# ---------------------------------------------------------------------------
import os
import threading
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # suppress tqdm bars

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Silence the safetensors background conversion thread that sometimes prints a
# spurious OSError to stderr when a model has no pytorch_model.bin/safetensors.
_orig_thread_excepthook = threading.excepthook

def _quiet_thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if args.thread is not None and "auto_conversion" in args.thread.name:
        return  # suppress safetensors conversion thread noise
    _orig_thread_excepthook(args)

threading.excepthook = _quiet_thread_excepthook

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    CLIPModel,
    CLIPProcessor,
)
from transformers.modeling_outputs import BaseModelOutput

import transformers
transformers.logging.set_verbosity_error()

# Silence remaining noisy loggers (httpx = HuggingFace HTTP client)
for _noisy in [
    "filelock", "urllib3", "huggingface_hub", "huggingface_hub.file_download",
    "bert_score", "absl", "matplotlib", "PIL", "torch", "codecarbon",
    "codecarbon.output_methods", "codecarbon.core",
    "httpx", "httpcore", "hpack",              # suppress HTTP HEAD requests
]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.proxy import ExplanationProxy

# ---------------------------------------------------------------------------
# Logger and box-drawing constants
# ---------------------------------------------------------------------------
logger = logging.getLogger("screencast_demo")

_SEP_HEAVY = "═" * 66
_SEP_LIGHT = "─" * 66

# Metrics box: inner width = 52 chars → total line = "  │" (3) + 52 + "│" (1) = 56
_BOX_W = 52
_BOX_TOP    = "  ┌" + "─" * _BOX_W + "┐"
_BOX_BOTTOM = "  └" + "─" * _BOX_W + "┘"


def _box_line(text: str) -> str:
    """Return a box-framed line whose content is padded to exactly _BOX_W chars."""
    return "  │" + text[:_BOX_W].ljust(_BOX_W) + "│"


def _metric_row(label: str, value: str) -> str:
    """Build one metrics-box row: 2 indent + label (26 wide) + 2 gap + value."""
    content = f"  {label:<26}  {value}"
    return _box_line(content)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "screencast_demo.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    # Re-apply suppression after basicConfig resets levels
    for _noisy in [
        "filelock", "urllib3", "huggingface_hub", "huggingface_hub.file_download",
        "bert_score", "absl", "matplotlib", "PIL", "torch", "codecarbon",
        "codecarbon.output_methods", "codecarbon.core",
        "httpx", "httpcore", "hpack",
    ]:
        logging.getLogger(_noisy).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_examples(jsonl_path: Path) -> List[Dict[str, Any]]:
    examples = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("original_text") or row.get("text") or ""
            image_path = row.get("image_path") or ""
            if not text or not image_path:
                continue
            # Infer source dataset from image path
            for ds in ("harmeme", "mami", "mmhs150k"):
                if ds in image_path.lower():
                    source = ds
                    break
            else:
                source = row.get("dataset", "unknown")
            examples.append({
                "id": str(row.get("id", "")),
                "image_path": image_path,
                "original_text": text,
                "target_group": row.get("target_group") or "—",
                "source": source,
            })
    return examples


# ---------------------------------------------------------------------------
# STA metric  (s-nlp/roberta_toxicity_classifier)
# ---------------------------------------------------------------------------

def load_sta_model(hf_cache: str, device: torch.device) -> Tuple:
    model_id = "s-nlp/roberta_toxicity_classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=hf_cache)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, cache_dir=hf_cache
    ).to(device).eval()
    return tokenizer, model


def compute_sta_single(text: str, tokenizer, model, device: torch.device) -> float:
    """Return P(non-toxic) for a single text string."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        prob = torch.softmax(logits, dim=-1)[0, 0].item()  # class 0 = non-toxic
    return prob


# ---------------------------------------------------------------------------
# CLIP  (shared for proxy pipeline features and CLIPScore metric)
# ---------------------------------------------------------------------------

def load_clip(hf_cache: str, device: torch.device) -> Tuple:
    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=hf_cache)
    model = CLIPModel.from_pretrained(model_id, cache_dir=hf_cache).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return processor, model


def compute_clip_score(
    image_path: str,
    rewrite: str,
    clip_processor,
    clip_model,
    device: torch.device,
) -> float:
    """Normalised cosine similarity between CLIP image and rewrite text embeddings."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return float("nan")
    inputs = clip_processor(
        images=img,
        text=rewrite,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        out = clip_model(**inputs)
        cos = torch.nn.functional.cosine_similarity(
            out.image_embeds, out.text_embeds
        ).item()
    return (cos + 1.0) / 2.0   # normalise to [0, 1]


# ---------------------------------------------------------------------------
# BERTScore / SIM metric
# Using BERTScorer class so the model is loaded once and reused every call,
# avoiding repeated HTTP HEAD requests between examples.
# ---------------------------------------------------------------------------

def load_bertscore(hf_cache: str):
    """Load roberta-large for BERTScore once; return a reusable BERTScorer."""
    from bert_score import BERTScorer
    if hf_cache:
        os.environ.setdefault("BERT_SCORE_CACHE", hf_cache)
    scorer = BERTScorer(
        model_type="roberta-large",
        rescale_with_baseline=True,
        lang="en",
    )
    return scorer


def compute_sim_single(original: str, rewrite: str, scorer) -> float:
    """Return BERTScore F1 between original and rewrite."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, F1 = scorer.score([rewrite], [original], verbose=False)
    return float(F1[0])


# ---------------------------------------------------------------------------
# BART + Proxy inference
# ---------------------------------------------------------------------------

def load_bart(checkpoint_dir: str, hf_cache: str, device: torch.device) -> Tuple:
    tokenizer = BartTokenizer.from_pretrained(checkpoint_dir, cache_dir=hf_cache)
    model = BartForConditionalGeneration.from_pretrained(
        checkpoint_dir, cache_dir=hf_cache
    ).to(device).eval()
    return tokenizer, model


def load_proxy(proxy_checkpoint: str, device: torch.device) -> ExplanationProxy:
    proxy_config_path = Path(proxy_checkpoint).parent / "proxy_config.json"
    num_soft_tokens = 16
    bart_hidden_size = 1024
    if proxy_config_path.exists():
        with proxy_config_path.open("r") as f:
            cfg = json.load(f)
            num_soft_tokens = int(cfg.get("num_soft_tokens", 16))
            bart_hidden_size = int(cfg.get("bart_hidden_size", 1024))
    proxy = ExplanationProxy(
        bart_hidden_size=bart_hidden_size,
        num_soft_tokens=num_soft_tokens,
    ).to(device)
    state_dict = torch.load(proxy_checkpoint, map_location=device)
    proxy.load_state_dict(state_dict)
    proxy.eval()
    return proxy


def _build_none_explicit_detox_prompt(text: str) -> str:
    return (
        "Task: rewrite the original meme text to be non-toxic while preserving "
        "the meme topic and intended meaning. "
        "Context: target group = null; visual evidence = null; "
        "implicit harmful meaning = null. "
        f"Original meme text to detoxify: {text}"
    )


def generate_rewrite(
    image_path: str,
    original_text: str,
    clip_processor,
    clip_model,
    bart_tokenizer,
    bart_model,
    proxy: ExplanationProxy,
    device: torch.device,
    max_length: int = 64,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
) -> str:
    """Run the proxy+BART pipeline for a single example."""
    # 1. CLIP features: image + original text -> concatenated 1536-dim vector
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))

    clip_inputs = clip_processor(
        images=img,
        text=original_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    clip_inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in clip_inputs.items()
    }
    with torch.no_grad():
        clip_out = clip_model(**clip_inputs)
        features = torch.cat(
            [clip_out.image_embeds, clip_out.text_embeds], dim=1
        ).float()  # [1, 1536]

    # 2. Proxy MLP: CLIP features -> soft encoder tokens
    with torch.no_grad():
        proxy_hidden = proxy(features)  # [1, num_soft_tokens, hidden_size]

    # 3. BART none-prompt encoder states
    prompt = _build_none_explicit_detox_prompt(original_text)
    enc_inputs = bart_tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        enc_out = bart_model.get_encoder()(
            input_ids=enc_inputs["input_ids"],
            attention_mask=enc_inputs["attention_mask"],
            return_dict=True,
        )
        text_hidden = enc_out.last_hidden_state  # [1, T, hidden_size]

    # 4. Concatenate proxy soft tokens + BART text encoder states
    dtype = next(bart_model.parameters()).dtype
    proxy_hidden = proxy_hidden.to(dtype=dtype)
    text_hidden = text_hidden.to(dtype=dtype)
    hidden = torch.cat([proxy_hidden, text_hidden], dim=1)
    proxy_mask = torch.ones(
        proxy_hidden.shape[:2], dtype=enc_inputs["attention_mask"].dtype, device=device
    )
    attention_mask = torch.cat([proxy_mask, enc_inputs["attention_mask"]], dim=1)

    # 5. Beam-search decoding
    with torch.no_grad():
        output_ids = bart_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
    return bart_tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Clean logging helpers
# ---------------------------------------------------------------------------

def log_section(title: str) -> None:
    logger.info("")
    logger.info(_SEP_LIGHT)
    logger.info("  %s", title)
    logger.info(_SEP_LIGHT)


def log_example_result(
    idx: int,
    n_total: int,
    example: Dict[str, Any],
    rewrite: str,
    sta_orig: float,
    sta_rewrite: float,
    sim: float,
    clip_score: float,
) -> None:
    delta = sta_rewrite - sta_orig
    delta_sign = "+" if delta >= 0 else ""

    # --- Header (two compact lines that always fit within _SEP_HEAVY width) ---
    logger.info("")
    logger.info(_SEP_HEAVY)
    logger.info(
        "  Example %2d / %d   |   Source: %s",
        idx, n_total, example["source"],
    )
    logger.info(
        "  ID: %-20s  Target group: %s",
        example["id"], example["target_group"],
    )
    logger.info(_SEP_HEAVY)

    # --- Texts ---
    logger.info("  Original : %s", example["original_text"])
    logger.info("  Rewrite  : %s", rewrite)
    logger.info("")

    # --- Metrics box (all lines are exactly _BOX_W + 4 chars wide) ---
    logger.info(_BOX_TOP)
    logger.info(_box_line("  Metrics"))
    logger.info(_box_line(""))
    logger.info(_metric_row(
        "STA (original -> rewrite)",
        f"{sta_orig:.4f}  ->  {sta_rewrite:.4f}",
    ))
    logger.info(_metric_row(
        "Delta Toxicity  (^ better)",
        f"{delta_sign}{delta:.4f}",
    ))
    logger.info(_metric_row("SIM / BERTScore F1", f"{sim:.4f}"))
    logger.info(_metric_row("CLIP Score", f"{clip_score:.4f}"))
    logger.info(_BOX_BOTTOM)


def log_aggregate_summary(
    n: int,
    sta_scores: List[float],
    delta_scores: List[float],
    sim_scores: List[float],
    clip_scores: List[float],
) -> None:
    # Column widths: label=24, mean=10, std=12 (inner) → total inner = 50
    top    = "  ╔" + "═" * 26 + "╦" + "═" * 12 + "╦" + "═" * 12 + "╗"
    divhdr = "  ╠" + "═" * 26 + "╬" + "═" * 12 + "╬" + "═" * 12 + "╣"
    bot    = "  ╚" + "═" * 26 + "╩" + "═" * 12 + "╩" + "═" * 12 + "╝"

    def hdr_row(a: str, b: str, c: str) -> str:
        return f"  ║  {a:<24}║  {b:<10}║  {c:<10}║"

    def data_row(label: str, vals: List[float]) -> str:
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        mean_str = f"{mean:+.4f}"   # always 7 chars: "+0.1234" or "-0.1234"
        std_str = f"±{std:.4f}"     # always 7 chars: "±0.1234"
        return f"  ║  {label:<24}║  {mean_str:<10}║  {std_str:<10}║"

    logger.info("")
    logger.info(top)
    logger.info(hdr_row(f"AGGREGATE RESULTS (n={n})", "", ""))
    logger.info(divhdr)
    logger.info(hdr_row("Metric", "Mean", "Std Dev"))
    logger.info(divhdr)
    logger.info(data_row("STA (rewrites)", sta_scores))
    logger.info(data_row("Delta Toxicity", delta_scores))
    logger.info(data_row("SIM (BERTScore F1)", sim_scores))
    logger.info(data_row("CLIP Score", clip_scores))
    logger.info(bot)
    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Screencast demo: CLIP Proxy + BART-large FT Full on 20 test images"
    )
    p.add_argument(
        "--input_jsonl",
        type=Path,
        default=Path("/scratch/stages/hmr_stage2_dataset/test.jsonl"),
    )
    p.add_argument(
        "--bart_checkpoint",
        type=str,
        default="/scratch/stages/hmr_stage2_phase2_full_explicit_detox_checkpoint",
    )
    p.add_argument(
        "--proxy_checkpoint",
        type=str,
        default="/scratch/stages/hmr_proxy_checkpoint_explicit_detox/best_proxy.pt",
    )
    p.add_argument("--hf_cache", type=str, default="/scratch/hf_cache")
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/scratch/eval_results/screencast_demo"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_samples", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("")
    logger.info(_SEP_HEAVY)
    logger.info("  Hateful Meme Rewriting  --  Proxy + BART Screencast Demo")
    logger.info(_SEP_HEAVY)
    logger.info("  Pipeline  : CLIP Proxy  +  BART-large FT Full")
    logger.info("  Device    : %s", device)
    logger.info("  Seed      : %d", args.seed)
    logger.info("  Samples   : %d", args.n_samples)
    logger.info("  Test set  : %s", args.input_jsonl)
    logger.info("")

    # -----------------------------------------------------------------------
    # Step 1: Load and sample test examples
    # -----------------------------------------------------------------------
    log_section("Step 1 -- Loading test set and sampling examples")

    all_examples = load_test_examples(args.input_jsonl)
    logger.info("  Total examples in test set : %d", len(all_examples))

    if len(all_examples) < args.n_samples:
        raise ValueError(
            f"Test set has only {len(all_examples)} examples, "
            f"cannot sample {args.n_samples}."
        )
    sampled = random.sample(all_examples, args.n_samples)
    logger.info("  Sampled (seed=%d)           : %d examples", args.seed, len(sampled))
    logger.info("  IDs: %s", ", ".join(ex["id"] for ex in sampled))

    # -----------------------------------------------------------------------
    # Step 2: Load evaluation metric models
    # -----------------------------------------------------------------------
    log_section("Step 2 -- Loading evaluation metric models")

    logger.info("  [2-A]  STA classifier  :  s-nlp/roberta_toxicity_classifier")
    t0 = time.time()
    sta_tokenizer, sta_model = load_sta_model(args.hf_cache, device)
    logger.info("         Loaded in %.1f s", time.time() - t0)

    logger.info("  [2-B]  CLIP model      :  openai/clip-vit-large-patch14")
    t0 = time.time()
    clip_processor, clip_model = load_clip(args.hf_cache, device)
    logger.info("         Loaded in %.1f s", time.time() - t0)

    logger.info("  [2-C]  BERTScore       :  roberta-large")
    t0 = time.time()
    bert_scorer = load_bertscore(args.hf_cache)
    logger.info("         Loaded in %.1f s", time.time() - t0)

    # -----------------------------------------------------------------------
    # Step 3: Load proxy + BART pipeline
    # -----------------------------------------------------------------------
    log_section("Step 3 -- Loading CLIP Proxy + BART-large FT Full pipeline")

    logger.info("  [3-A]  BART checkpoint :  %s", args.bart_checkpoint)
    t0 = time.time()
    bart_tokenizer, bart_model = load_bart(args.bart_checkpoint, args.hf_cache, device)
    logger.info("         Loaded in %.1f s", time.time() - t0)

    logger.info("  [3-B]  Proxy network   :  %s", args.proxy_checkpoint)
    t0 = time.time()
    proxy = load_proxy(args.proxy_checkpoint, device)
    logger.info("         Loaded in %.1f s", time.time() - t0)

    logger.info("")
    logger.info("  All models loaded. Starting inference.")

    # -----------------------------------------------------------------------
    # Step 4: Inference loop
    # -----------------------------------------------------------------------
    log_section("Step 4 -- Generating rewrites and computing metrics")

    results = []
    sta_scores, delta_scores, sim_scores, clip_scores = [], [], [], []

    for idx, example in enumerate(sampled, start=1):
        image_path = example["image_path"]
        original_text = example["original_text"]

        # Generate rewrite via proxy + BART
        rewrite = generate_rewrite(
            image_path=image_path,
            original_text=original_text,
            clip_processor=clip_processor,
            clip_model=clip_model,
            bart_tokenizer=bart_tokenizer,
            bart_model=bart_model,
            proxy=proxy,
            device=device,
        )

        # STA for original and rewrite
        sta_orig    = compute_sta_single(original_text, sta_tokenizer, sta_model, device)
        sta_rewrite = compute_sta_single(rewrite,       sta_tokenizer, sta_model, device)
        delta       = sta_rewrite - sta_orig

        # Semantic similarity (BERTScore F1)
        sim = compute_sim_single(original_text, rewrite, bert_scorer)

        # CLIP image-text alignment for the rewrite
        clip_score = compute_clip_score(
            image_path, rewrite, clip_processor, clip_model, device
        )

        # Accumulate
        sta_scores.append(sta_rewrite)
        delta_scores.append(delta)
        sim_scores.append(sim)
        clip_scores.append(clip_score)

        log_example_result(
            idx=idx,
            n_total=args.n_samples,
            example=example,
            rewrite=rewrite,
            sta_orig=sta_orig,
            sta_rewrite=sta_rewrite,
            sim=sim,
            clip_score=clip_score,
        )

        results.append({
            "id": example["id"],
            "source": example["source"],
            "image_path": image_path,
            "original_text": original_text,
            "rewrite": rewrite,
            "sta_original": sta_orig,
            "sta_rewrite": sta_rewrite,
            "delta_toxicity": delta,
            "sim": sim,
            "clip_score": clip_score,
        })

    # -----------------------------------------------------------------------
    # Step 5: Aggregate summary
    # -----------------------------------------------------------------------
    log_section("Step 5 -- Aggregate results")
    log_aggregate_summary(
        n=args.n_samples,
        sta_scores=sta_scores,
        delta_scores=delta_scores,
        sim_scores=sim_scores,
        clip_scores=clip_scores,
    )

    # Save results
    out_path = args.output_dir / "screencast_results.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("  Results saved to : %s", out_path)
    logger.info("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
