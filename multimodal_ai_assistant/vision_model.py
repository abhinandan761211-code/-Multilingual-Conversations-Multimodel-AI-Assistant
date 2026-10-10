"""
vision_model.py
===============
Thin, defensive wrapper around the Ollama client for the vision-language model.

Design notes
------------
* The router decides *what* to do; this module only knows *how to talk to a
  model*. It exposes small, purpose-specific calls (describe, extract, answer)
  rather than one generic "chat" — that keeps prompts close to their intent.
* Every call returns a structured `ModelResponse`, never a bare string, so the
  validator downstream has something to check.
* Model availability is probed explicitly (`health_check`) so the UI can show a
  clear message instead of a stack trace.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field

import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


class ModelResponse(BaseModel):
    """Everything we know about one model generation."""

    text: str = ""
    model: str = ""
    success: bool = False
    error: Optional[str] = None
    had_image: bool = False
    latency_s: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class ModelHealth(BaseModel):
    server_up: bool = False
    vision_model_available: bool = False
    text_model_available: bool = False
    installed_models: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    # Set when the configured text model was missing and we auto-swapped it.
    text_model_fallback: Optional[str] = None

    def summary(self) -> str:
        if not self.server_up:
            return f"Ollama server unreachable at {config.OLLAMA_HOST}."
        if not self.vision_model_available:
            return (
                f"Ollama is running but '{config.VISION_MODEL}' is not pulled. "
                f"Run: ollama pull {config.VISION_MODEL}"
            )
        if self.text_model_fallback:
            return (
                "Ollama ready (vision available; text model auto-switched to "
                f"'{self.text_model_fallback}')."
            )
        return "Ollama ready (vision + text models available)."


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

# Ollama's python client reads OLLAMA_HOST from env; we also pass it explicitly
# where the API allows, to make behaviour obvious in the source.

SYSTEM_PROMPT = (
    "You are a careful multimodal research assistant. "
    "You answer ONLY from what you can actually see in the provided image and "
    "from the reference context you are given. "
    "If something is unclear, blurred, or not visible, say so explicitly and "
    "explain what is uncertain — never invent details, numbers, or labels. "
    "When the user asks about counts or quantities, state what you can clearly "
    "identify and flag anything obscured. "
    "Prefer short, structured answers with concrete visual evidence."
)


def _make_client():
    """Import the ollama client lazily so tests can run without it installed."""
    try:
        import ollama  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "The 'ollama' python package is not installed. "
            "Run: pip install ollama"
        ) from exc
    return ollama


class VisionModel:
    """Wrapper over one vision model + one text model on a local Ollama host."""

    def __init__(
        self,
        vision_model: str = config.VISION_MODEL,
        text_model: str = config.TEXT_MODEL,
        host: str = config.OLLAMA_HOST,
    ) -> None:
        self.vision_model = vision_model
        self.text_model = text_model
        self.host = host

    # ── infrastructure ───────────────────────────────────────────────────────

    def resolve_available_models(self) -> ModelHealth:
        """
        Health check that also *repairs* a missing text model.

        The vision model is special — there is no sensible generic fallback, so
        if it is absent we report it. A text model is swappable: if the
        configured one is not pulled, any other local text model is better than
        failing every text-only question.
        """
        health = self.health_check()
        if health.server_up and not health.text_model_available:
            fallback = self._pick_text_fallback(health.installed_models)
            if fallback:
                log.warning(
                    "Text model '%s' is not pulled; falling back to '%s'",
                    self.text_model, fallback,
                )
                self.text_model = fallback
                health.text_model_available = True
                health.text_model_fallback = fallback
        return health

    @staticmethod
    def _pick_text_fallback(installed: Sequence[str]) -> Optional[str]:
        """Prefer a small general model; avoid vision models for text-only work."""
        candidates = [n for n in installed if "vision" not in n.lower() and "vl" not in n.lower()]
        if not candidates:
            candidates = list(installed)
        if not candidates:
            return None
        # Prefer names that look like general instruct models, smallest first.
        def rank(name: str) -> tuple[int, int]:
            lowered = name.lower()
            preferred = 0 if any(k in lowered for k in ("llama3", "qwen", "mistral", "gemma")) else 1
            return (preferred, len(name))
        return sorted(candidates, key=rank)[0]

    def health_check(self) -> ModelHealth:
        health = ModelHealth()
        try:
            client = _make_client()
            models = client.list()
            names = []
            for entry in models.get("models", []):
                name = entry.get("name") or entry.get("model") or ""
                names.append(name)
            health.server_up = True
            health.installed_models = names
            health.vision_model_available = self._matches(self.vision_model, names)
            health.text_model_available = self._matches(self.text_model, names)
        except Exception as exc:  # noqa: BLE001
            health.error = str(exc)
        return health

    @staticmethod
    def _matches(wanted: str, installed: Sequence[str]) -> bool:
        """Ollama tags carry ':latest' etc., so compare loosely."""
        base = wanted.split(":")[0]
        return any(n == wanted or n.split(":")[0] == base for n in installed)

    # ── core generation ──────────────────────────────────────────────────────

    def _chat(
        self,
        model: str,
        user_text: str,
        images: Optional[list[str]] = None,
        system: str = SYSTEM_PROMPT,
        temperature: float = config.VISION_TEMPERATURE,
        json_mode: bool = False,
    ) -> ModelResponse:
        import time

        started = time.time()
        try:
            client = _make_client()
        except RuntimeError as exc:
            return ModelResponse(model=model, error=str(exc))

        message: dict[str, Any] = {"role": "user", "content": user_text}
        if images:
            message["images"] = images

        options = {"temperature": temperature}
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["format"] = "json"

        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    message,
                ],
                options=options,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResponse(
                model=model,
                error=f"{type(exc).__name__}: {exc}",
                had_image=bool(images),
                latency_s=round(time.time() - started, 2),
            )

        text = ""
        try:
            text = response["message"]["content"] if isinstance(response, dict) else response.message.content  # type: ignore[index]
        except Exception:  # noqa: BLE001
            try:
                text = str(response)
            except Exception:  # noqa: BLE001
                text = ""

        return ModelResponse(
            text=text.strip(),
            model=model,
            success=bool(text.strip()),
            had_image=bool(images),
            latency_s=round(time.time() - started, 2),
            raw={"done": getattr(response, "done", None)},
        )

    # ── purpose-specific vision calls ────────────────────────────────────────

    def describe_image(
        self,
        image_b64: str,
        user_question: str = "",
        context: str = "",
    ) -> ModelResponse:
        """
        Free-form description. Used when the user uploaded an image with no
        specific question, or asks "what do you see?".
        """
        prompt = (
            "Describe this image factually and in order of visual importance.\n"
            "Cover: (1) the type of image (photo / chart / diagram / document / "
            "screenshot), (2) the main objects or regions, (3) any visible text "
            "you can actually read, (4) any numbers, axes, or labels.\n"
            "If any part is unclear or unreadable, say so instead of guessing."
        )
        if context:
            prompt += f"\n\nConversation context:\n{context}"
        if user_question:
            prompt += f"\n\nUser's request: {user_question}"
        return self._chat(
            self.vision_model, prompt, images=[image_b64]
        )

    def extract_text_from_image(self, image_b64: str, ocr_hint: str = "") -> ModelResponse:
        """
        Ask the VLM to transcribe visible text. Used *in addition to* OCR — the
        two transcripts are cross-checked by the validator.
        """
        prompt = (
            "Transcribe ALL readable text in this image, preserving line breaks. "
            "Include titles, axis labels, legends and handwritten text. "
            "Do NOT translate, correct, summarise or explain. "
            "If a word is illegible, write [illegible] in its place."
        )
        if ocr_hint:
            prompt += (
                "\n\nA separate OCR pass produced the following candidate text. "
                "Use it only as a hint; report what you can truly see:\n"
                f"---\n{ocr_hint}\n---"
            )
        return self._chat(
            self.vision_model,
            prompt,
            images=[image_b64],
            temperature=0.0,
        )

    def analyse_chart(self, image_b64: str, user_question: str, context: str = "") -> ModelResponse:
        """Structured chart reading — trends, axes, specific values."""
        prompt = (
            "This image appears to contain a chart, graph or plot. Analyse it:\n"
            "- Identify chart type, title, x-axis and y-axis (units if visible).\n"
            "- List the series/legend entries.\n"
            "- Describe the overall trend of each series.\n"
            "- Report specific values ONLY where you can clearly read them, and "
            "quote the axis labels that support them.\n"
            "- If a value is unclear, say 'unclear' rather than estimating.\n"
        )
        if context:
            prompt += f"\nConversation so far:\n{context}\n"
        prompt += f"\nUser's question: {user_question}"
        return self._chat(self.vision_model, prompt, images=[image_b64])

    def answer_with_image(
        self,
        image_b64: str,
        user_question: str,
        context: str = "",
        reference_context: str = "",
    ) -> ModelResponse:
        """
        The main path: answer the user's actual question using the image, plus
        conversation context and retrieved reference material as supporting
        evidence.
        """
        parts = [
            "Answer the user's question using the image as primary evidence.",
            "Ground every factual claim in something visible in the image; if "
            "the image does not support a claim, say so.",
        ]
        if context:
            parts.append(f"\nConversation context:\n{context}")
        if reference_context:
            parts.append(
                "\nReference material (may be relevant, may not be — use it only "
                f"if it genuinely helps):\n{reference_context}"
            )
        parts.append(f"\nUser's question: {user_question}")
        return self._chat(self.vision_model, "\n".join(parts), images=[image_b64])

    def answer_without_image(
        self,
        user_question: str,
        context: str = "",
        reference_context: str = "",
    ) -> ModelResponse:
        """Text-only path (and the path used for follow-ups about a stored image)."""
        parts = [
            "Answer the question in your own words.",
            "There is NO image in this turn: do not claim to see, look at, or be "
            "shown an image. If the conversation context above describes an "
            "image that was shared earlier, you may refer to that description, "
            "but describe it as what was previously established rather than as "
            "something you are currently viewing.",
        ]
        if context:
            parts.append(f"Conversation context:\n{context}")
        if reference_context:
            parts.append(f"Reference material:\n{reference_context}")
        parts.append(f"User's question: {user_question}")
        if reference_context or context:
            parts.append(
                "\nBase your answer on the context and reference material above. "
                "If they do not contain the answer, say what is missing instead "
                "of guessing."
            )
        return self._chat(
            self.text_model,
            "\n\n".join(parts),
            temperature=config.TEXT_TEMPERATURE,
        )

    # ── structured helper for the router ─────────────────────────────────────

    def classify_request(self, user_text: str, has_image: bool) -> dict:
        """
        Ask the small text model to judge intent and ambiguity.

        Returns a dict; on any failure returns a safe default with
        `fallback: True` so the router can continue with heuristics.
        """
        default = {
            "intent": "unknown",
            "specificity": "unknown",
            "needs_clarification": False,
            "clarifying_question": "",
            "reasoning": "",
            "fallback": True,
        }
        prompt = (
            "You are routing a request in a multimodal assistant.\n"
            f"has_image: {has_image}\n"
            f"user_message: {json.dumps(user_text)}\n\n"
            "Return ONLY JSON with these keys:\n"
            '  "intent": one of [describe_image, read_text, analyse_chart, '
            'answer_about_image, follow_up, general_question, small_talk, '
            'unclear]\n'
            '  "specificity": one of [high, medium, low]\n'
            '  "needs_clarification": boolean\n'
            '  "clarifying_question": string (empty if not needed)\n'
            '  "reasoning": short string\n\n'
            "Rules: an image with a vague message like 'check this' needs "
            "clarification. A specific question like 'what is the accuracy at "
            "epoch 10?' does not. Follow-ups that reference a previous image "
            "should be 'follow_up'."
        )
        response = self._chat(
            self.text_model,
            prompt,
            temperature=0.0,
            json_mode=True,
        )
        if not response.success:
            default["reasoning"] = response.error or "classifier unavailable"
            return default

        parsed = _safe_json(response.text)
        if not parsed:
            default["reasoning"] = "classifier returned unparsable JSON"
            return default

        for key, value in default.items():
            parsed.setdefault(key, value)
        parsed["fallback"] = False
        return parsed

    def check_entailment(self, claim: str, evidence: str) -> dict:
        """
        Lightweight self-check: does `evidence` actually support `claim`?

        Used by response_validator for the claims that matter (counts, numbers,
        named entities) rather than every sentence.
        """
        default = {"supported": None, "confidence": 0.0, "reason": "check failed"}
        if not claim.strip():
            default["reason"] = "empty claim"
            return default
        prompt = (
            "Decide whether the EVIDENCE supports the CLAIM.\n"
            f"CLAIM: {json.dumps(claim)}\n"
            f"EVIDENCE: {json.dumps(evidence[:3000])}\n\n"
            'Return ONLY JSON: {"supported": true|false, "confidence": 0-1, '
            '"reason": "one sentence"}'
        )
        response = self._chat(
            self.text_model, prompt, temperature=0.0, json_mode=True
        )
        if not response.success:
            default["reason"] = response.error or "model unavailable"
            return default
        parsed = _safe_json(response.text)
        if not parsed:
            default["reason"] = "unparsable check output"
            return default
        parsed.setdefault("supported", None)
        parsed.setdefault("confidence", 0.0)
        parsed.setdefault("reason", "")
        return parsed


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_json(text: str) -> Optional[dict]:
    """Parse JSON that may be wrapped in prose or code fences."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json ... ``` fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    # Last resort: grab the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
