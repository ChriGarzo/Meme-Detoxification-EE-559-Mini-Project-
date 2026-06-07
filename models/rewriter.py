"""
rewriter.py — Student model (BART-LoRA) for the hateful meme detoxification pipeline.

MemeRewriter wraps facebook/bart-large (or bart-base) for explanation-conditioned
sequence-to-sequence text detoxification. It serves two roles:

  1. Inference: format_input() prepends structured explanation tokens to the raw meme
     text, and generate() / batch_rewrite() produce non-hateful rewrites.
  2. Feature extraction: get_encoder_hidden_state() exposes the mean-pooled BART
     encoder representation, which is used as the regression target for training the
     CLIP-based ExplanationProxy (proxy.py).

Input format (legacy): "[T: {target_group}] [V: {visual_evidence}] [M: {implicit_meaning}] | {text}"
Input format (explicit_detox): natural-language task description prepended to context and text.

Design decisions:
  - Lazy model loading (load_model on first use) allows the object to be constructed
    before GPU memory is available.
  - generate_from_formatted() bypasses format_input() for callers that pre-build the
    encoder string (e.g., the RL/RLHF training loop).
  - decode_from_hidden_state() accepts a synthetic encoder state, enabling the proxy
    to drive BART decoding at inference without a text input.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
)


logger = logging.getLogger(__name__)


class MemeRewriter:
    """BART student model: explanation-conditioned seq2seq detoxification of meme text."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def __init__(
        self,
        model_name: str = "facebook/bart-large",
        checkpoint_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        num_beams: int = 4,
        debug: bool = False,
    ):
        """
        Defer model loading; store configuration for lazy initialisation via load_model().

        debug substitutes bart-base for bart-large to reduce memory during development.
        """
        if debug:
            model_name = "facebook/bart-base"
            logger.info("Debug mode: using facebook/bart-base")

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_beams = num_beams
        self.tokenizer = None
        self.model = None
        self.hidden_size = None
        logger.info(f"MemeRewriter initialized with device: {self.device}")

    def load_model(self) -> None:
        """Load the BART tokenizer and model weights; apply checkpoint if checkpoint_path is set."""
        logger.info(f"Loading model {self.model_name}...")

        self.tokenizer = BartTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        ).to(self.device)

        # BartConfig exposes d_model; generic HF configs use hidden_size.
        if isinstance(self.model.config, BartConfig):
            self.hidden_size = self.model.config.d_model
        else:
            self.hidden_size = self.model.config.hidden_size

        logger.info(f"Model hidden size: {self.hidden_size}")

        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)

        logger.info("Model loaded successfully")

    # ── Internal utilities ────────────────────────────────────────────────────

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a state dict from checkpoint_path into the BART model; no-op with warning if path absent."""
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    # ── Public API ────────────────────────────────────────────────────────────

    def format_input(
        self,
        text: str,
        target_group: Optional[str] = None,
        visual_evidence: Optional[str] = None,
        implicit_meaning: Optional[str] = None,
        mode: Literal["full", "target_only", "visual_only", "none"] = "full",
        input_format: Literal["legacy", "explicit_detox"] = "legacy",
        task_prefix: str = "",
    ) -> str:
        """
        Assemble the BART encoder input string from meme text and explanation fields.

        mode controls which explanation fields are populated vs. replaced with "null",
        supporting ablation experiments. input_format selects between the compact
        bracket notation (legacy) and a natural-language task description (explicit_detox).
        task_prefix is prepended verbatim when set (e.g. for task-specific fine-tuning signals).
        """
        if mode == "full":
            tg = target_group or "null"
            ve = visual_evidence or "null"
            im = implicit_meaning or "null"
        elif mode == "target_only":
            tg = target_group or "null"
            ve = "null"
            im = "null"
        elif mode == "visual_only":
            tg = "null"
            ve = visual_evidence or "null"
            im = "null"
        else:  # mode == "none"
            tg = "null"
            ve = "null"
            im = "null"

        if input_format == "explicit_detox":
            formatted = (
                "Task: rewrite the original meme text to be non-toxic while preserving "
                "the meme topic and intended meaning. "
                f"Context: target group = {tg}; visual evidence = {ve}; "
                f"implicit harmful meaning = {im}. "
                f"Original meme text to detoxify: {text}"
            )
        else:
            formatted = f"[T: {tg}] [V: {ve}] [M: {im}] | {text}"

        task_prefix = (task_prefix or "").strip()
        if task_prefix:
            return f"{task_prefix} {formatted}"
        return formatted

    def rewrite(
        self,
        text: str,
        target_group: Optional[str] = None,
        visual_evidence: Optional[str] = None,
        implicit_meaning: Optional[str] = None,
        mode: Literal["full", "target_only", "visual_only", "none"] = "full",
        input_format: Literal["legacy", "explicit_detox"] = "legacy",
        task_prefix: str = "",
        max_length: int = 150,
    ) -> str:
        """Generate a single detoxified rewrite using greedy beam search (do_sample=False)."""
        if self.model is None:
            self.load_model()

        formatted_input = self.format_input(
            text,
            target_group=target_group,
            visual_evidence=visual_evidence,
            implicit_meaning=implicit_meaning,
            mode=mode,
            input_format=input_format,
            task_prefix=task_prefix,
        )

        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
            )

        rewritten = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return rewritten

    def batch_rewrite(
        self,
        texts: List[str],
        target_groups: Optional[List[Optional[str]]] = None,
        visual_evidences: Optional[List[Optional[str]]] = None,
        implicit_meanings: Optional[List[Optional[str]]] = None,
        mode: Literal["full", "target_only", "visual_only", "none"] = "full",
        input_format: Literal["legacy", "explicit_detox"] = "legacy",
        task_prefix: str = "",
        max_length: int = 150,
    ) -> List[str]:
        """Generate detoxified rewrites for a list of texts; errors per example are replaced with a sentinel string."""
        if self.model is None:
            self.load_model()

        # Default None lists to per-example None, preserving the single-example code path.
        target_groups = target_groups or [None] * len(texts)
        visual_evidences = visual_evidences or [None] * len(texts)
        implicit_meanings = implicit_meanings or [None] * len(texts)

        results = []
        for text, tg, ve, im in zip(
            texts, target_groups, visual_evidences, implicit_meanings
        ):
            try:
                rewritten = self.rewrite(
                    text,
                    target_group=tg,
                    visual_evidence=ve,
                    implicit_meaning=im,
                    mode=mode,
                    input_format=input_format,
                    task_prefix=task_prefix,
                    max_length=max_length,
                )
                results.append(rewritten)
            except Exception as e:
                logger.error(f"Error rewriting text '{text}': {e}")
                results.append(f"[REWRITE ERROR: {str(e)}]")

        return results

    def get_encoder_hidden_state(
        self,
        text: str,
        target_group: Optional[str] = None,
        visual_evidence: Optional[str] = None,
        implicit_meaning: Optional[str] = None,
        mode: Literal["full", "target_only", "visual_only", "none"] = "full",
        input_format: Literal["legacy", "explicit_detox"] = "legacy",
        task_prefix: str = "",
    ) -> torch.Tensor:
        """
        Return the mean-pooled BART encoder representation for the formatted input.

        Output shape: [1, hidden_size]. Used by ExplanationProxyTrainer.extract_bart_targets()
        to build regression targets for the proxy network.
        """
        if self.model is None:
            self.load_model()

        formatted_input = self.format_input(
            text,
            target_group=target_group,
            visual_evidence=visual_evidence,
            implicit_meaning=implicit_meaning,
            mode=mode,
            input_format=input_format,
            task_prefix=task_prefix,
        )

        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            # Encoder access varies across Transformers versions and model wrappers.
            if hasattr(self.model, "get_encoder"):
                encoder = self.model.get_encoder()
            elif hasattr(self.model, "encoder"):
                encoder = self.model.encoder
            elif hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
                encoder = self.model.model.encoder
            else:
                raise AttributeError(
                    f"Could not find encoder on model type {type(self.model).__name__}"
                )

            encoder_outputs = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            last_hidden_state = encoder_outputs.last_hidden_state  # [B, seq_len, H]
            # Mean pooling collapses variable-length sequences to a fixed-size vector
            # suitable for MSE regression against the proxy output.
            hidden_state = last_hidden_state.mean(dim=1)  # [1, hidden_size]

        return hidden_state

    def generate_from_formatted(
        self,
        formatted_inputs: List[str],
        max_length: int = 64,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: int = 3,
        encoder_no_repeat_ngram_size: int = 3,
    ) -> List[str]:
        """
        Generate rewrites from pre-formatted input strings.

        Use this when the caller has already built the full BART encoder string
        (e.g. '[T: ...] [V: ...] [M: ...] | {text}') and does NOT want
        format_input() to be applied again.

        Args:
            formatted_inputs: List of already-formatted strings
            max_length: Maximum generation length
            num_beams: Beam count. Defaults to self.num_beams.
            no_repeat_ngram_size: Prevent repeated n-grams inside the output.
            encoder_no_repeat_ngram_size: Prevent copying n-grams from the input.

        Returns:
            List of generated strings
        """
        if self.model is None:
            self.load_model()

        results = []
        for text in formatted_inputs:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    num_beams=num_beams or self.num_beams,
                    early_stopping=True,
                    do_sample=False,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                )

            decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append(decoded)

        return results

    def decode_from_hidden_state(
        self,
        hidden_state: torch.Tensor,
        max_length: int = 150,
    ) -> str:
        """
        Decode text by injecting a synthetic encoder state, bypassing the text encoder entirely.

        hidden_state is expected to be [1, hidden_size] or [batch_size, hidden_size] as
        produced by ExplanationProxy.forward(); it is expanded to [B, 1, H] to match the
        shape expected by BART's cross-attention.
        """
        if self.model is None:
            self.load_model()

        # Wrap the raw tensor in a minimal namespace that satisfies BART's decoder
        # cross-attention interface without instantiating a full BaseModelOutput.
        encoder_outputs = type(
            "EncoderOutputs",
            (),
            {
                "last_hidden_state": hidden_state.to(self.device),
                "hidden_states": None,
                "attentions": None,
            },
        )()

        with torch.no_grad():
            output_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
            )

        decoded = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return decoded
