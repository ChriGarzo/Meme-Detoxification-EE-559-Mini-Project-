import json
import logging
import torch
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

    VALID_TARGET_GROUPS = {
        "race_ethnicity",
        "nationality",
        "religion",
        "gender",
        "sexual_orientation",
        "disability",
        "other",
    }

    EXPLAIN_PROMPT = """[INST] <image>
You are analyzing a meme for hate speech using BOTH image content and text.
The text in the meme is: '{text}'.
Respond ONLY with a valid JSON object in this exact format:
{{"target_group": "<group>", "visual_evidence": "<short visual cue>", "implicit_meaning": "<one sentence about text+image hate meaning>"}}

Valid target_group values: race_ethnicity, nationality, religion, gender, sexual_orientation, disability, other

Rules:
- visual_evidence must describe at least one concrete visual element from the image (person, symbol, gesture, object, scene, edit style, juxtaposition).
- implicit_meaning must explain why the COMBINATION of text and image is hateful (not text-only paraphrase).

Respond with JSON only. No preamble, no explanation.
[/INST]"""

    REWRITE_PROMPT = """[INST] <image>
The text in this meme is: '{text}'
Analysis:
- Targeted group: {target_group}
- Visual evidence: {visual_evidence}
- Why text + image are harmful: {implicit_meaning}
Rewrite the meme text so it stays on topic but becomes clearly non-hateful.
Guidelines:
- Remove slurs, insults, demeaning stereotypes, and group-blaming language.
- Keep the same core situation/topic and meme-like tone (short, punchy, informal).
- Replace hateful framing with neutral or inclusive wording (focus on behavior/situation, not identity).
- Prefer a fluent paraphrase over deleting words.
- If the original is already mild, still produce a cleaner paraphrase rather than copying it.
- Make at least one substantial wording change; do not lightly copy the original phrasing.
- Keep it as one short sentence, usually under 18 words.

Output rules (strict):
- Return one plain rewritten text only
- No URL, no @mention, no hashtag, no watermark/site string
- No quotes around the full answer, no markdown, no labels, no explanation
- No newline characters

Respond with ONLY the rewritten text.
[/INST]"""

    REWRITE_RETRY_HINTS = {
        "no_edit": "Retry instruction: rewrite more substantially. Use noticeably different wording from the original.",
        "too_similar": "Retry instruction: keep the same topic, but paraphrase more aggressively and avoid reusing the original phrasing.",
        "too_long": "Retry instruction: make the rewrite shorter and punchier. Stay under 18 words.",
        "low_sta": "Retry instruction: make the rewrite clearly safer. Remove identity-based blame, insults, stereotypes, and contempt.",
        "low_toxicity_drop": "Retry instruction: reduce the hateful framing more clearly while keeping the topic.",
        "low_bertscore": "Retry instruction: stay closer to the original situation and meme topic while removing the hate.",
        "high_bertscore": "Retry instruction: keep the meaning, but change the wording more so it is not a near copy.",
        "empty": "Retry instruction: return exactly one short rewritten sentence and nothing else.",
        "too_short": "Retry instruction: write one complete short sentence, not just a fragment.",
        "url": "Retry instruction: do not include links, site names, or external references.",
        "mention": "Retry instruction: do not include usernames or @mentions.",
        "hashtag": "Retry instruction: do not include hashtags.",
        "low_diversity": "Retry instruction: write a fluent sentence without repeating the same words.",
        "repetition": "Retry instruction: avoid repetition and produce one clean sentence.",
        "symbol_heavy": "Retry instruction: use normal sentence text only, not symbols or formatting artifacts.",
    }

    def __init__(
        self,
        model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        load_in_4bit: bool = False,
        cache_dir: Optional[str] = None,
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
        self.cache_dir = cache_dir
        self.debug = debug
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        logger.info(f"MemeExplainer initialized with device: {self.device}")

    def load_model(self) -> None:
        """Load the LLaVA model and processor."""
        logger.info(f"Loading model {self.model_name}...")

        self.processor = LlavaNextProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

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
            cache_dir=self.cache_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if self.load_in_4bit else torch.float32,
        )

        # Avoid repetitive generation warning spam.
        eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            self.model.generation_config.pad_token_id = eos_id

        logger.info("Model loaded successfully")

    def _load_image(self, image_path: str) -> Any:
        """
        Load image from path with debug fallback.

        Args:
            image_path: Path to image file

        Returns:
            PIL Image object

        Raises:
            FileNotFoundError: In production mode if file doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            if self.debug:
                logger.warning(
                    f"Image file not found (debug mode): {image_path}. "
                    "Returning black placeholder image."
                )
                return Image.new("RGB", (336, 336), color=(0, 0, 0))
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(path).convert("RGB")

    def _generate_batch_responses(
        self,
        prompts: List[str],
        images: List[Any],
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        """Run one batched LLaVA generation call and decode per-example responses."""
        if not prompts:
            return []

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output_ids = self.model.generate(**generation_kwargs)

        responses: List[str] = []
        # Use padded input width as boundary for all rows (same as single-example
        # path). attention_mask-based slicing can include prompt echoes with
        # left-padding decoder-only models.
        prompt_len = int(inputs["input_ids"].shape[1])
        for i in range(output_ids.shape[0]):
            generated_ids = output_ids[i][prompt_len:]
            responses.append(
                self.processor.decode(generated_ids, skip_special_tokens=True)
            )
        return responses

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response from model with fallback.

        Args:
            response: Raw model output string

        Returns:
            Parsed dictionary or null-dict with parse_error=True
        """
        raw = (response or "").strip()

        # LLaVA can still wrap JSON in markdown fences despite prompt constraints.
        # Strip common fence wrappers and keep only the JSON object payload.
        candidate = raw
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()

        if candidate.lower().startswith("json\n"):
            candidate = candidate[5:].strip()

        if "{" in candidate and "}" in candidate:
            # Keep only the last balanced JSON object. This handles cases where
            # the model echoes prompt text containing example JSON patterns.
            end = candidate.rfind("}")
            start = -1
            depth = 0
            for i in range(end, -1, -1):
                ch = candidate[i]
                if ch == "}":
                    depth += 1
                elif ch == "{":
                    depth -= 1
                    if depth == 0:
                        start = i
                        break
            if start != -1 and end != -1 and end >= start:
                candidate = candidate[start:end + 1].strip()

        try:
            parsed = json.loads(candidate)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parse error: {e}. Raw output:\n{response}"
            )
            return {
                "target_group": None,
                "visual_evidence": None,
                "implicit_meaning": None,
                "parse_error": True,
                "raw_output": response,
            }

    @staticmethod
    def _is_null_like(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {
                "",
                "null",
                "none",
                "n/a",
                "na",
                "unknown",
                "unspecified",
            }
        return False

    def _normalize_target_group(self, value: Any) -> Optional[str]:
        if self._is_null_like(value) or not isinstance(value, str):
            return None

        cleaned = value.strip().lower().replace("-", "_").replace(" ", "_")
        if cleaned in self.VALID_TARGET_GROUPS:
            return cleaned

        if any(k in cleaned for k in ["race", "ethnic", "asian", "black", "white"]):
            return "race_ethnicity"
        if any(k in cleaned for k in ["nation", "country", "immigrant"]):
            return "nationality"
        if any(k in cleaned for k in ["relig", "muslim", "christian", "jew", "hindu"]):
            return "religion"
        if any(k in cleaned for k in ["gender", "woman", "women", "man", "men", "female", "male"]):
            return "gender"
        if any(k in cleaned for k in ["gay", "lesbian", "lgbt", "sexual"]):
            return "sexual_orientation"
        if any(k in cleaned for k in ["disab", "autis", "handicap"]):
            return "disability"
        if "other" in cleaned:
            return "other"
        return None

    def _normalize_implicit_meaning(self, value: Any) -> Optional[str]:
        if self._is_null_like(value) or not isinstance(value, str):
            return None
        cleaned = " ".join(value.strip().split())
        return cleaned if cleaned else None

    def _normalize_visual_evidence(self, value: Any) -> Optional[str]:
        if self._is_null_like(value) or not isinstance(value, str):
            return None
        cleaned = " ".join(value.strip().split())
        return cleaned if cleaned else None

    @staticmethod
    def _is_complete_explanation(explanation: Dict[str, Any]) -> bool:
        return bool(
            explanation.get("target_group")
            and explanation.get("visual_evidence")
            and explanation.get("implicit_meaning")
        )

    def _normalize_explanation(
        self,
        explanation: Dict[str, Any],
    ) -> Dict[str, Any]:
        src = explanation if isinstance(explanation, dict) else {}
        normalized = {
            "target_group": self._normalize_target_group(src.get("target_group")),
            "visual_evidence": self._normalize_visual_evidence(src.get("visual_evidence")),
            "implicit_meaning": self._normalize_implicit_meaning(src.get("implicit_meaning")),
        }

        if src.get("parse_error"):
            normalized["parse_error"] = True
        if "raw_output" in src:
            normalized["raw_output"] = src["raw_output"]
        return normalized

    def explain(
        self,
        image_path: str,
        text: str,
        max_retries: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate structured hate speech explanation for a meme.

        Args:
            image_path: Path to meme image
            text: Text content of the meme

        Returns:
            Dictionary with target_group, visual_evidence, implicit_meaning,
            and parse_error flag if JSON parsing failed
        """
        if self.model is None:
            self.load_model()

        image = self._load_image(image_path)
        prompt = self.EXPLAIN_PROMPT.format(text=text)
        attempts = 1 + max(0, max_retries)
        last_parsed: Dict[str, Any] = {
            "target_group": None,
            "visual_evidence": None,
            "implicit_meaning": None,
        }

        for attempt_idx in range(attempts):
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

            parsed = self._parse_json_response(response)
            normalized = self._normalize_explanation(parsed)
            if self._is_complete_explanation(normalized):
                return normalized

            last_parsed = parsed
            logger.warning(
                "Explanation incomplete (attempt %d/%d), retrying",
                attempt_idx + 1,
                attempts,
            )

        return self._normalize_explanation(last_parsed)

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
            explanation: Dictionary with target_group, visual_evidence, implicit_meaning

        Returns:
            Rewritten text string
        """
        if self.model is None:
            self.load_model()

        image = self._load_image(image_path)

        target_group = explanation.get("target_group") or "other"
        visual_evidence = explanation.get("visual_evidence") or (
            "visual details in the meme"
        )
        implicit_meaning = explanation.get("implicit_meaning") or (
            "The meme uses text and visual context to frame a target group in a hateful way."
        )

        prompt = self._build_rewrite_prompt(
            text=text,
            target_group=target_group,
            visual_evidence=visual_evidence,
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
        max_retries: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of memes.

        Args:
            image_paths: List of paths to meme images
            texts: List of meme texts

        Returns:
            List of explanation dictionaries
        """
        if len(image_paths) != len(texts):
            raise ValueError(
                f"image_paths and texts must have the same length, got {len(image_paths)} vs {len(texts)}"
            )

        if self.model is None:
            self.load_model()

        n = len(texts)
        results: List[Optional[Dict[str, Any]]] = [None] * n
        prompts: List[Optional[str]] = [None] * n
        images: List[Optional[Any]] = [None] * n
        last_parsed: Dict[int, Dict[str, Any]] = {}

        for i, (image_path, text) in enumerate(zip(image_paths, texts)):
            try:
                images[i] = self._load_image(image_path)
                prompts[i] = self.EXPLAIN_PROMPT.format(text=text)
                last_parsed[i] = {
                    "target_group": None,
                    "visual_evidence": None,
                    "implicit_meaning": None,
                }
            except Exception as e:
                logger.error(f"Error preparing meme ({image_path}, {text}): {e}")
                fallback = {
                    "target_group": None,
                    "visual_evidence": None,
                    "implicit_meaning": None,
                    "parse_error": True,
                    "error": str(e),
                }
                results[i] = self._normalize_explanation(fallback)

        active = [i for i in range(n) if results[i] is None]
        attempts = 1 + max(0, max_retries)

        for attempt_idx in range(attempts):
            if not active:
                break

            batch_prompts = [prompts[i] for i in active]
            batch_images = [images[i] for i in active]
            responses: List[Optional[str]] = []
            try:
                responses = self._generate_batch_responses(
                    prompts=batch_prompts,
                    images=batch_images,
                    max_new_tokens=200,
                )
            except Exception as e:
                logger.warning(
                    "Batch explanation generation failed (%s). Falling back to serial for %d examples.",
                    e,
                    len(active),
                )
                for i in active:
                    try:
                        response = self._generate_batch_responses(
                            prompts=[prompts[i]],
                            images=[images[i]],
                            max_new_tokens=200,
                        )[0]
                    except Exception as inner_e:
                        logger.error(
                            "Error explaining meme (%s, %s): %s",
                            image_paths[i],
                            texts[i],
                            inner_e,
                        )
                        response = None
                    responses.append(response)

            next_active: List[int] = []
            for pos, idx in enumerate(active):
                response = responses[pos] if pos < len(responses) else None
                if response is None:
                    fallback = {
                        "target_group": None,
                        "visual_evidence": None,
                        "implicit_meaning": None,
                        "parse_error": True,
                        "error": "Model generation failed",
                    }
                    results[idx] = self._normalize_explanation(fallback)
                    continue

                parsed = self._parse_json_response(response)
                normalized = self._normalize_explanation(parsed)

                if self._is_complete_explanation(normalized):
                    results[idx] = normalized
                    continue

                last_parsed[idx] = parsed
                if attempt_idx + 1 < attempts:
                    logger.warning(
                        "Explanation incomplete (attempt %d/%d), retrying",
                        attempt_idx + 1,
                        attempts,
                    )
                    next_active.append(idx)
                else:
                    results[idx] = self._normalize_explanation(last_parsed[idx])

            active = next_active

        for i in range(n):
            if results[i] is None:
                results[i] = self._normalize_explanation(last_parsed.get(i, {}))

        return [
            r if r is not None else {
                "target_group": None,
                "visual_evidence": None,
                "implicit_meaning": None,
                "parse_error": True,
            }
            for r in results
        ]

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
        if not (len(image_paths) == len(texts) == len(explanations)):
            raise ValueError(
                "image_paths, texts, and explanations must have the same length"
            )

        if self.model is None:
            self.load_model()

        n = len(texts)
        results: List[str] = ["" for _ in range(n)]
        valid_indices: List[int] = []
        prompts: List[str] = []
        images: List[Any] = []

        for i, (image_path, text, explanation) in enumerate(
            zip(image_paths, texts, explanations)
        ):
            try:
                image = self._load_image(image_path)
            except Exception as e:
                logger.error(f"Error rewriting meme ({image_path}, {text}): {e}")
                results[i] = f"[REWRITE ERROR: {str(e)}]"
                continue

            target_group = explanation.get("target_group") or "other"
            visual_evidence = explanation.get("visual_evidence") or (
                "visual details in the meme"
            )
            implicit_meaning = explanation.get("implicit_meaning") or (
                "The meme uses text and visual context to frame a target group in a hateful way."
            )
            prompt = self._build_rewrite_prompt(
                text=text,
                target_group=target_group,
                visual_evidence=visual_evidence,
                implicit_meaning=implicit_meaning,
            )

            valid_indices.append(i)
            prompts.append(prompt)
            images.append(image)

        if not valid_indices:
            return results

        responses: List[Optional[str]] = []
        try:
            responses = self._generate_batch_responses(
                prompts=prompts,
                images=images,
                max_new_tokens=150,
            )
        except Exception as e:
            logger.warning(
                "Batch rewrite generation failed (%s). Falling back to serial for %d examples.",
                e,
                len(valid_indices),
            )
            for prompt, image in zip(prompts, images):
                try:
                    response = self._generate_batch_responses(
                        prompts=[prompt],
                        images=[image],
                        max_new_tokens=150,
                    )[0]
                except Exception as inner_e:
                    response = f"[REWRITE ERROR: {str(inner_e)}]"
                responses.append(response)

        for pos, idx in enumerate(valid_indices):
            response = responses[pos] if pos < len(responses) else ""
            if response is None:
                response = ""
            results[idx] = response.strip()

        return results

    def _build_rewrite_prompt(
        self,
        text: str,
        target_group: str,
        visual_evidence: str,
        implicit_meaning: str,
        feedback_reason: Optional[str] = None,
    ) -> str:
        prompt = self.REWRITE_PROMPT.format(
            text=text,
            target_group=target_group,
            visual_evidence=visual_evidence,
            implicit_meaning=implicit_meaning,
        )
        extra_hint = self.REWRITE_RETRY_HINTS.get((feedback_reason or "").strip(), "")
        if extra_hint:
            prompt = prompt.replace(
                "Respond with ONLY the rewritten text.",
                f"{extra_hint}\nRespond with ONLY the rewritten text.",
            )
        return prompt

    def batch_rewrite_candidates(
        self,
        image_paths: List[str],
        texts: List[str],
        explanations: List[Dict[str, Any]],
        feedback_reasons: Optional[List[Optional[str]]] = None,
        candidates_per_example: int = 3,
        do_sample: bool = True,
        temperature: float = 0.75,
        top_p: float = 0.92,
    ) -> List[List[str]]:
        """
        Generate multiple rewrite candidates per example.

        The generation call expands the batch by repeating each prompt/image pair,
        which keeps the output mapping simple while still allowing sampling to
        produce diverse candidates.
        """
        if not (len(image_paths) == len(texts) == len(explanations)):
            raise ValueError(
                "image_paths, texts, and explanations must have the same length"
            )
        if feedback_reasons is not None and len(feedback_reasons) != len(texts):
            raise ValueError("feedback_reasons must match the number of examples")
        if candidates_per_example < 1:
            raise ValueError("candidates_per_example must be >= 1")

        if self.model is None:
            self.load_model()

        n = len(texts)
        grouped_results: List[List[str]] = [[] for _ in range(n)]
        valid_indices: List[int] = []
        prompts: List[str] = []
        images: List[Any] = []

        for i, (image_path, text, explanation) in enumerate(
            zip(image_paths, texts, explanations)
        ):
            try:
                image = self._load_image(image_path)
            except Exception as e:
                logger.error(f"Error rewriting meme ({image_path}, {text}): {e}")
                grouped_results[i] = [f"[REWRITE ERROR: {str(e)}]"]
                continue

            target_group = explanation.get("target_group") or "other"
            visual_evidence = explanation.get("visual_evidence") or (
                "visual details in the meme"
            )
            implicit_meaning = explanation.get("implicit_meaning") or (
                "The meme uses text and visual context to frame a target group in a hateful way."
            )
            feedback_reason = (
                feedback_reasons[i] if feedback_reasons is not None else None
            )
            prompt = self._build_rewrite_prompt(
                text=text,
                target_group=target_group,
                visual_evidence=visual_evidence,
                implicit_meaning=implicit_meaning,
                feedback_reason=feedback_reason,
            )

            valid_indices.append(i)
            for _ in range(candidates_per_example):
                prompts.append(prompt)
                images.append(image)

        if not prompts:
            return grouped_results

        responses: List[Optional[str]] = []
        try:
            responses = self._generate_batch_responses(
                prompts=prompts,
                images=images,
                max_new_tokens=150,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            logger.warning(
                "Batch rewrite candidate generation failed (%s). Falling back to serial for %d examples.",
                e,
                len(valid_indices),
            )
            for idx in valid_indices:
                image = self._load_image(image_paths[idx])
                target_group = explanations[idx].get("target_group") or "other"
                visual_evidence = explanations[idx].get("visual_evidence") or (
                    "visual details in the meme"
                )
                implicit_meaning = explanations[idx].get("implicit_meaning") or (
                    "The meme uses text and visual context to frame a target group in a hateful way."
                )
                feedback_reason = (
                    feedback_reasons[idx] if feedback_reasons is not None else None
                )
                prompt = self._build_rewrite_prompt(
                    text=texts[idx],
                    target_group=target_group,
                    visual_evidence=visual_evidence,
                    implicit_meaning=implicit_meaning,
                    feedback_reason=feedback_reason,
                )
                for _ in range(candidates_per_example):
                    try:
                        response = self._generate_batch_responses(
                            prompts=[prompt],
                            images=[image],
                            max_new_tokens=150,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                        )[0]
                    except Exception as inner_e:
                        response = f"[REWRITE ERROR: {str(inner_e)}]"
                    responses.append(response)

        cursor = 0
        for idx in valid_indices:
            grouped: List[str] = []
            for _ in range(candidates_per_example):
                response = responses[cursor] if cursor < len(responses) else ""
                if response is None:
                    response = ""
                grouped.append(response.strip())
                cursor += 1
            grouped_results[idx] = grouped

        return grouped_results
