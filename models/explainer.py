import json
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
from PIL import Image


logger = logging.getLogger(__name__)


class MemeExplainer:
    """LLaVA wrapper for generating structured hate speech explanations."""

    EXPLAIN_PROMPT = """[INST] <image>
You are analyzing a meme for hate speech. The text in the meme is: '{text}'.
If this meme contains hate speech, respond ONLY with a valid JSON object in this exact format:
{{"target_group": "<group>", "attack_type": "<type>", "implicit_meaning": "<one sentence>"}}

Valid attack_type values: contempt, mocking, inferiority, slurs, exclusion, dehumanizing, inciting_violence
Valid target_group values: race_ethnicity, nationality, religion, gender, sexual_orientation, disability, other

If the meme does NOT contain hate speech, respond ONLY with:
{{"target_group": null, "attack_type": null, "implicit_meaning": null}}

Respond with JSON only. No preamble, no explanation.
[/INST]"""

    REWRITE_PROMPT = """[INST] <image>
The text in this meme is: '{text}'
Analysis: this meme targets {target_group} using {attack_type} — {implicit_meaning}.
Rewrite only the meme text to be non-hateful while:
- Preserving the approximate length and informal register of the original
- Keeping the same topic but removing the hateful framing
- Producing natural language that could plausibly appear on a meme

Respond with ONLY the rewritten text. No quotes, no explanation, no preamble.
[/INST]"""

    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        load_in_4bit: bool = False,
        device: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Initialize the MemeExplainer.

        Args:
            model_name: HuggingFace model identifier
            load_in_4bit: Whether to load model in 4-bit quantization
            device: Device to run on ('cuda', 'cpu'). Auto-detected if None.
            debug: Debug mode for missing images (return zeros instead of error)
        """
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.debug = debug
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        logger.info(f"MemeExplainer initialized with device: {self.device}")

    def load_model(self) -> None:
        """Load the LLaVA model and processor."""
        logger.info(f"Loading model {self.model_name}...")

        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Loading model with 4-bit quantization")

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=self.device,
            torch_dtype=torch.float16 if self.load_in_4bit else torch.float32,
        )

        logger.info("Model loaded successfully")

    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load image from path with debug fallback.

        Args:
            image_path: Path to image file

        Returns:
            Tensor or PIL Image object

        Raises:
            FileNotFoundError: In production mode if file doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            if self.debug:
                logger.warning(
                    f"Image file not found (debug mode): {image_path}. "
                    "Returning zero tensor."
                )
                return torch.zeros(1, 3, 336, 336)
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(path).convert("RGB")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from model with fallback.

        Args:
            response: Raw model output string

        Returns:
            Parsed dictionary or null-dict with parse_error=True
        """
        try:
            parsed = json.loads(response.strip())
            return parsed
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parse error: {e}. Raw output:\n{response}"
            )
            return {
                "target_group": None,
                "attack_type": None,
                "implicit_meaning": None,
                "parse_error": True,
                "raw_output": response,
            }

    def explain(self, image_path: str, text: str) -> Dict[str, Any]:
        """
        Generate structured hate speech explanation for a meme.

        Args:
            image_path: Path to meme image
            text: Text content of the meme

        Returns:
            Dictionary with target_group, attack_type, implicit_meaning,
            and parse_error flag if JSON parsing failed
        """
        if self.model is None:
            self.load_model()

        image = self._load_image(image_path)
        prompt = self.EXPLAIN_PROMPT.format(text=text)

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        response = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return self._parse_json_response(response)

    def generate_rewrite(
        self,
        image_path: str,
        text: str,
        explanation: Dict[str, Any],
    ) -> str:
        """
        Generate a non-hateful rewrite of meme text using explanation.

        Args:
            image_path: Path to meme image
            text: Original meme text
            explanation: Dictionary with target_group, attack_type, implicit_meaning

        Returns:
            Rewritten text string
        """
        if self.model is None:
            self.load_model()

        image = self._load_image(image_path)

        target_group = explanation.get("target_group", "null")
        attack_type = explanation.get("attack_type", "null")
        implicit_meaning = explanation.get("implicit_meaning", "null")

        prompt = self.REWRITE_PROMPT.format(
            text=text,
            target_group=target_group,
            attack_type=attack_type,
            implicit_meaning=implicit_meaning,
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
            )

        rewrite = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return rewrite.strip()

    def batch_explain(
        self,
        image_paths: List[str],
        texts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of memes.

        Args:
            image_paths: List of paths to meme images
            texts: List of meme texts

        Returns:
            List of explanation dictionaries
        """
        results = []
        for image_path, text in zip(image_paths, texts):
            try:
                explanation = self.explain(image_path, text)
                results.append(explanation)
            except Exception as e:
                logger.error(
                    f"Error explaining meme ({image_path}, {text}): {e}"
                )
                results.append({
                    "target_group": None,
                    "attack_type": None,
                    "implicit_meaning": None,
                    "parse_error": True,
                    "error": str(e),
                })

        return results

    def batch_rewrite(
        self,
        image_paths: List[str],
        texts: List[str],
        explanations: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate rewrites for a batch of memes.

        Args:
            image_paths: List of paths to meme images
            texts: List of original meme texts
            explanations: List of explanation dictionaries

        Returns:
            List of rewritten texts
        """
        results = []
        for image_path, text, explanation in zip(
            image_paths, texts, explanations
        ):
            try:
                rewrite = self.generate_rewrite(
                    image_path, text, explanation
                )
                results.append(rewrite)
            except Exception as e:
                logger.error(
                    f"Error rewriting meme ({image_path}, {text}): {e}"
                )
                results.append(f"[REWRITE ERROR: {str(e)}]")

        return results
