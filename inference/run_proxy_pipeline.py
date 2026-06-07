"""
Proxy pipeline inference: VLM-free meme rewriting via CLIP + ExplanationProxy + BART.

Pipeline position: AFTER train_proxy.py; used as the deployment-mode counterpart
to run_stage2.py (full condition).

Inputs : --input_jsonl  Stage 2 val/test JSONL with image_path and original_text.
Outputs: <output_dir>/stage2_rewrites_clip_proxy_bart_full.jsonl
         (format compatible with evaluation/evaluate.py)

Inference sequence per batch:
  1. CLIP: image + original_text  ->  [B, 1536] joint embedding
  2. ExplanationProxy MLP:         ->  [B, num_soft_tokens, hidden_size]
  3. BART text encoder (none-condition prompt):  ->  [B, T, hidden_size]
  4. Concatenate proxy tokens + text encoder states along sequence dimension.
  5. BART decoder beam search -> rewrite strings.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from codecarbon import EmissionsTracker
from PIL import Image
from transformers import BartForConditionalGeneration, BartTokenizer, CLIPModel, CLIPProcessor
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.proxy import ExplanationProxy

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path, debug: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "proxy_pipeline.log"),
            logging.StreamHandler(),
        ],
    )


def load_stage2_jsonl(path: Path, max_examples: Optional[int]) -> List[Dict[str, Any]]:
    """Load Stage 2 val/test JSONL; skip rows without both original_text and image_path."""
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON in %s line %d: %s", path, line_num, exc)
                continue
            original_text = row.get("original_text") or row.get("text") or ""
            image_path = row.get("image_path") or ""
            if not original_text or not image_path:
                continue
            examples.append({
                "id": row.get("id"),
                "image_path": image_path,
                "original_text": original_text,
                "target_text": row.get("target_text", ""),
                "target_group": row.get("target_group"),
                "visual_evidence": row.get("visual_evidence"),
                "implicit_meaning": row.get("implicit_meaning"),
                "dataset": row.get("dataset"),
            })
            if max_examples and len(examples) >= max_examples:
                break
    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


def write_jsonl_batch(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Append a batch of output records to a JSONL file."""
    with output_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ProxyBartPipeline:
    """
    Batched inference wrapper: CLIP -> ExplanationProxy soft tokens + BART encoder -> decoder.

    text_prompt_format selects which none-condition prompt is prepended to the
    BART text encoder path (none_legacy: bracket format; none_explicit_detox:
    natural-language format).  Both were trained alongside the proxy; the choice
    must match the format used during proxy training.
    """

    def __init__(
        self,
        bart_checkpoint: str,
        proxy_checkpoint: str,
        clip_model_name: str,
        hf_cache: Optional[str],
        device: str,
        num_soft_tokens: Optional[int] = None,
        text_prompt_format: str = "none_legacy",
    ) -> None:
        self.device = device
        self.text_prompt_format = text_prompt_format

        logger.info("Loading CLIP model: %s", clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name, cache_dir=hf_cache)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, cache_dir=hf_cache).to(device)
        self.clip_model.eval()

        logger.info("Loading BART checkpoint: %s", bart_checkpoint)
        self.tokenizer = BartTokenizer.from_pretrained(bart_checkpoint, cache_dir=hf_cache)
        self.bart = BartForConditionalGeneration.from_pretrained(bart_checkpoint, cache_dir=hf_cache).to(device)
        self.bart.eval()

        hidden_size = int(self.bart.config.d_model)
        # Read num_soft_tokens from proxy_config.json if not overridden via CLI.
        proxy_config_path = Path(proxy_checkpoint).parent / "proxy_config.json"
        if num_soft_tokens is None and proxy_config_path.exists():
            try:
                with proxy_config_path.open("r", encoding="utf-8") as f:
                    num_soft_tokens = int(json.load(f).get("num_soft_tokens", 16))
            except Exception as exc:
                logger.warning("Could not read %s: %s", proxy_config_path, exc)
        num_soft_tokens = num_soft_tokens or 16
        logger.info("Loading proxy checkpoint: %s", proxy_checkpoint)
        logger.info("Proxy soft tokens: %d", num_soft_tokens)
        self.proxy = ExplanationProxy(
            bart_hidden_size=hidden_size,
            num_soft_tokens=num_soft_tokens,
        ).to(device)
        state_dict = torch.load(proxy_checkpoint, map_location=device)
        self.proxy.load_state_dict(state_dict)
        self.proxy.eval()

    def _build_text_prompts(self, texts: List[str]) -> List[str]:
        """Construct the none-condition BART encoder prompt for each original text."""
        if self.text_prompt_format == "none_explicit_detox":
            return [
                "Task: rewrite the original meme text to be non-toxic while preserving "
                "the meme topic and intended meaning. "
                "Context: target group = null; visual evidence = null; "
                "implicit harmful meaning = null. "
                f"Original meme text to detoxify: {text}"
                for text in texts
            ]
        return [f"[T: null] [V: null] [M: null] | {text}" for text in texts]

    def _encode_text_prompts(self, texts: List[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode none-condition prompts through the BART encoder; return (hidden_state, mask)."""
        prompts = self._build_text_prompts(texts)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(self.device)
        encoder = self.bart.get_encoder()
        with torch.no_grad():
            encoder_outputs = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
        return encoder_outputs.last_hidden_state, inputs["attention_mask"]

    def _clip_features(self, image_paths: List[str], texts: List[str]) -> torch.Tensor:
        """Extract CLIP joint embedding [B, 1536] = image_embed ‖ text_embed."""
        images = []
        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB"))
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", image_path, exc)
                images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))

        inputs = self.clip_processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Concatenate along feature dimension: [B, 768] ‖ [B, 768] -> [B, 1536].
            return torch.cat([outputs.image_embeds, outputs.text_embeds], dim=1).float()

    def rewrite_batch(
        self,
        image_paths: List[str],
        texts: List[str],
        max_length: int,
        num_beams: int,
        no_repeat_ngram_size: int,
    ) -> List[str]:
        features = self._clip_features(image_paths, texts)
        with torch.no_grad():
            proxy_hidden = self.proxy(features)  # [B, num_soft_tokens, hidden_size]
            text_hidden, text_attention_mask = self._encode_text_prompts(texts)
            # Cast to BART's parameter dtype (fp32 / bf16 depending on hardware).
            dtype = next(self.bart.parameters()).dtype
            proxy_hidden = proxy_hidden.to(dtype=dtype)
            text_hidden = text_hidden.to(dtype=dtype)
            # Prepend proxy soft tokens to the text encoder sequence.
            hidden = torch.cat([proxy_hidden, text_hidden], dim=1)
            proxy_attention_mask = torch.ones(
                proxy_hidden.shape[:2],
                dtype=text_attention_mask.dtype,
                device=self.device,
            )
            attention_mask = torch.cat([proxy_attention_mask, text_attention_mask], dim=1)
            # Wrap in BaseModelOutput so BART's generate() accepts pre-computed states.
            encoder_outputs = BaseModelOutput(last_hidden_state=hidden)
            output_ids = self.bart.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run proxy -> BART-full VLM-free rewriting")
    parser.add_argument("--input_jsonl", type=Path, required=True, help="Stage 2 val/test JSONL")
    parser.add_argument("--bart_checkpoint", type=str, required=True, help="Full-condition BART checkpoint")
    parser.add_argument("--proxy_checkpoint", type=str, required=True, help="Trained proxy .pt checkpoint")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hf_cache", type=str, default=None)
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--num_soft_tokens", type=int, default=None)
    parser.add_argument(
        "--text_prompt_format",
        type=str,
        choices=["none_legacy", "none_explicit_detox"],
        default="none_legacy",
        help="Text prompt encoded after proxy tokens.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
    set_seed(args.seed)
    setup_logging(args.output_dir, debug=args.debug)

    max_examples = args.max_examples
    if args.debug and max_examples is None:
        max_examples = 16

    examples = load_stage2_jsonl(args.input_jsonl, max_examples=max_examples)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    logger.info("Arguments: %s", vars(args))

    pipeline = ProxyBartPipeline(
        bart_checkpoint=args.bart_checkpoint,
        proxy_checkpoint=args.proxy_checkpoint,
        clip_model_name=args.clip_model_name,
        hf_cache=args.hf_cache,
        device=device,
        num_soft_tokens=args.num_soft_tokens,
        text_prompt_format=args.text_prompt_format,
    )

    output_path = args.output_dir / "stage2_rewrites_clip_proxy_bart_full.jsonl"
    if output_path.exists():
        logger.info("Removing existing output before regeneration: %s", output_path)
        output_path.unlink()

    tracker = EmissionsTracker(log_level="warning", output_dir=str(args.output_dir), output_file="emissions.csv")
    tracker.start()

    total = 0
    try:
        for start in tqdm.tqdm(range(0, len(examples), args.batch_size), desc="Proxy+BART rewrites"):
            batch = examples[start:start + args.batch_size]
            image_paths = [ex["image_path"] for ex in batch]
            texts = [ex["original_text"] for ex in batch]
            try:
                rewrites = pipeline.rewrite_batch(
                    image_paths=image_paths,
                    texts=texts,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
            except Exception as exc:
                logger.exception("Proxy+BART generation failed for batch starting at %d: %s", start, exc)
                rewrites = [""] * len(batch)

            records = []
            system_name = (
                "clip_proxy_bart_full_explicit_detox"
                if args.text_prompt_format == "none_explicit_detox"
                else "clip_proxy_bart_full"
            )
            for ex, rewrite in zip(batch, rewrites):
                records.append({
                    "id": ex.get("id"),
                    "image_path": ex.get("image_path"),
                    "original_text": ex.get("original_text"),
                    "target_text": ex.get("target_text", ""),
                    "rewrite": rewrite,
                    "condition": system_name,
                    "system": system_name,
                    "explanation": {
                        "target_group": None,
                        "visual_evidence": "proxy_predicted_from_clip_image_text",
                        "implicit_meaning": f"proxy_soft_tokens_plus_bart_{args.text_prompt_format}_encoder_states",
                    },
                    "dataset": ex.get("dataset"),
                })
            write_jsonl_batch(records, output_path)
            total += len(records)
            logger.info("Processed %d/%d examples", total, len(examples))
    finally:
        emissions = tracker.stop()
        if emissions is not None:
            logger.info("Carbon emissions: %.6f kg CO2", emissions)

    logger.info("Proxy+BART output JSONL: %s", output_path)
    logger.info("Total examples processed: %d", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
