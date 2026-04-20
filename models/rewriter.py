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
    """BART wrapper for text detoxification with explanation-aware formatting."""

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
        Initialize the MemeRewriter.

        Args:
            model_name: HuggingFace model identifier ('facebook/bart-large' or 'facebook/bart-base')
            checkpoint_path: Path to a saved checkpoint to load from
            device: Device to run on ('cuda', 'cpu'). Auto-detected if None.
            debug: Debug mode (uses bart-base instead of bart-large)
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
        """Load the BART model and tokenizer."""
        logger.info(f"Loading model {self.model_name}...")

        self.tokenizer = BartTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        ).to(self.device)

        # Determine hidden size
        if isinstance(self.model.config, BartConfig):
            self.hidden_size = self.model.config.d_model
        else:
            self.hidden_size = self.model.config.hidden_size

        logger.info(f"Model hidden size: {self.hidden_size}")

        # Load checkpoint if provided
        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)

        logger.info("Model loaded successfully")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def format_input(
        self,
        text: str,
        target_group: Optional[str] = None,
        attack_type: Optional[str] = None,
        implicit_meaning: Optional[str] = None,
        mode: Literal["full", "target_only", "attack_only", "none"] = "full",
    ) -> str:
        """
        Format input text with explanation prefix tokens.

        Args:
            text: Original meme text
            target_group: Target group (or None)
            attack_type: Attack type (or None)
            implicit_meaning: Implicit meaning (or None)
            mode: Formatting mode:
                - 'full': include all fields
                - 'target_only': include only target_group
                - 'attack_only': include only attack_type
                - 'none': all fields as null

        Returns:
            Formatted input string for BART
        """
        if mode == "full":
            tg = target_group or "null"
            at = attack_type or "null"
            im = implicit_meaning or "null"
        elif mode == "target_only":
            tg = target_group or "null"
            at = "null"
            im = "null"
        elif mode == "attack_only":
            tg = "null"
            at = attack_type or "null"
            im = "null"
        else:  # mode == "none"
            tg = "null"
            at = "null"
            im = "null"

        return f"[T: {tg}] [A: {at}] [M: {im}] </s> {text}"

    def rewrite(
        self,
        text: str,
        target_group: Optional[str] = None,
        attack_type: Optional[str] = None,
        implicit_meaning: Optional[str] = None,
        mode: Literal["full", "target_only", "attack_only", "none"] = "full",
        max_length: int = 150,
    ) -> str:
        """
        Generate a detoxified rewrite of input text.

        Args:
            text: Original text
            target_group: Target group from explanation
            attack_type: Attack type from explanation
            implicit_meaning: Implicit meaning from explanation
            mode: Explanation prefix mode
            max_length: Maximum length of generated text

        Returns:
            Rewritten text
        """
        if self.model is None:
            self.load_model()

        formatted_input = self.format_input(
            text,
            target_group=target_group,
            attack_type=attack_type,
            implicit_meaning=implicit_meaning,
            mode=mode,
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
        attack_types: Optional[List[Optional[str]]] = None,
        implicit_meanings: Optional[List[Optional[str]]] = None,
        mode: Literal["full", "target_only", "attack_only", "none"] = "full",
        max_length: int = 150,
    ) -> List[str]:
        """
        Generate rewrites for a batch of texts.

        Args:
            texts: List of original texts
            target_groups: List of target groups (or None for each)
            attack_types: List of attack types (or None for each)
            implicit_meanings: List of implicit meanings (or None for each)
            mode: Explanation prefix mode
            max_length: Maximum length of generated texts

        Returns:
            List of rewritten texts
        """
        if self.model is None:
            self.load_model()

        # Handle None inputs
        target_groups = target_groups or [None] * len(texts)
        attack_types = attack_types or [None] * len(texts)
        implicit_meanings = implicit_meanings or [None] * len(texts)

        results = []
        for text, tg, at, im in zip(
            texts, target_groups, attack_types, implicit_meanings
        ):
            try:
                rewritten = self.rewrite(
                    text,
                    target_group=tg,
                    attack_type=at,
                    implicit_meaning=im,
                    mode=mode,
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
        attack_type: Optional[str] = None,
        implicit_meaning: Optional[str] = None,
        mode: Literal["full", "target_only", "attack_only", "none"] = "full",
    ) -> torch.Tensor:
        """
        Extract mean-pooled encoder hidden state from BART.

        Args:
            text: Input text
            target_group: Target group from explanation
            attack_type: Attack type from explanation
            implicit_meaning: Implicit meaning from explanation
            mode: Explanation prefix mode

        Returns:
            Tensor of shape [1, hidden_size] with mean-pooled encoder output
        """
        if self.model is None:
            self.load_model()

        formatted_input = self.format_input(
            text,
            target_group=target_group,
            attack_type=attack_type,
            implicit_meaning=implicit_meaning,
            mode=mode,
        )

        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            # Transformers versions expose the encoder through different attributes.
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
            # encoder_outputs[0] has shape [batch_size, seq_len, hidden_size]
            last_hidden_state = encoder_outputs.last_hidden_state

            # Mean pooling over sequence dimension, keeping batch dim
            hidden_state = last_hidden_state.mean(dim=1)  # [1, hidden_size]

        return hidden_state

    def generate_from_formatted(
        self,
        formatted_inputs: List[str],
        max_length: int = 128,
    ) -> List[str]:
        """
        Generate rewrites from pre-formatted input strings.

        Use this when the caller has already built the full BART encoder string
        (e.g. '[T: ...] [A: ...] [M: ...] </s> {text}') and does NOT want
        format_input() to be applied again.

        Args:
            formatted_inputs: List of already-formatted strings
            max_length: Maximum generation length

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
                    num_beams=self.num_beams,
                    early_stopping=True,
                    do_sample=False,
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
        Decode text from an injected encoder hidden state.

        Args:
            hidden_state: Tensor of shape [1, hidden_size] or [batch_size, hidden_size]
            max_length: Maximum length of generated text

        Returns:
            Decoded text string
        """
        if self.model is None:
            self.load_model()

        # Create encoder outputs object
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
