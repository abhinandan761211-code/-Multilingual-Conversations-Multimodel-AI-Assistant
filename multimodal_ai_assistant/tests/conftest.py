"""Shared test fixtures: synthetic images and a fake vision model.

The tests must run without Ollama, without a GPU, and without network access —
that is the whole point of injecting collaborators.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

# Make the package importable when running `pytest` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from conversation_memory import ConversationMemory  # noqa: E402
from image_processor import ImageDiagnostics, OCRResult  # noqa: E402
from reasoning_router import ReasoningRouter  # noqa: E402
from response_validator import ResponseValidator  # noqa: E402
from vision_model import ModelResponse  # noqa: E402


def _png_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sharp_text_image() -> bytes:
    """A crisp image of black text on white — OCR-friendly, high sharpness."""
    img = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), "Training Loss 0.42", fill="black")
    draw.text((30, 90), "Validation Loss 0.71", fill="black")
    draw.text((30, 140), "Epoch 10", fill="black")
    for y in range(200, 260, 8):
        draw.line([(30, y), (860, y)], fill="black", width=2)
    return _png_bytes(img)


@pytest.fixture
def blurry_image() -> bytes:
    """Same content, heavily blurred → should be flagged as low quality."""
    img = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), "Training Loss 0.42", fill="black")
    draw.text((30, 90), "Validation Loss 0.71", fill="black")
    img = img.filter(ImageFilter.GaussianBlur(radius=12))
    return _png_bytes(img)


@pytest.fixture
def tiny_image() -> bytes:
    return _png_bytes(Image.new("RGB", (24, 24), "grey"))


@pytest.fixture
def corrupt_bytes() -> bytes:
    return b"this is definitely not an image file"


@pytest.fixture
def memory() -> ConversationMemory:
    return ConversationMemory()


@pytest.fixture
def router() -> ReasoningRouter:
    """Router with no LLM classifier — fully deterministic."""
    return ReasoningRouter(classifier=None)


class FakeVisionModel:
    """
    Drop-in replacement for VisionModel that returns canned responses and
    records the calls it received. Lets pipeline tests assert routing without
    a real model.
    """

    def __init__(self, replies: dict | None = None) -> None:
        self.replies = replies or {}
        self.calls: list[str] = []
        self.text_model = "fake-text"
        self.vision_model = "fake-vision"

    def _reply(self, key: str, default: str) -> ModelResponse:
        self.calls.append(key)
        return ModelResponse(
            text=self.replies.get(key, default),
            model="fake",
            success=True,
        )

    def describe_image(self, image_b64, user_question="", context=""):
        return self._reply(
            "describe_image",
            "This is a line chart titled 'Model Loss' with epochs on the x-axis "
            "and loss on the y-axis. Two series are visible: Training and "
            "Validation. Validation loss rises after epoch 6.",
        )

    def extract_text_from_image(self, image_b64, ocr_hint=""):
        return self._reply(
            "extract_text_from_image", "Training Loss 0.42\nValidation Loss 0.71"
        )

    def analyse_chart(self, image_b64, user_question, context=""):
        return self._reply(
            "analyse_chart",
            "Chart type: line plot. X-axis: Epoch (0-10). Y-axis: Loss (0.0-1.0). "
            "Training loss falls steadily. Validation loss falls to 0.31 at epoch 6 "
            "then rises to 0.71 by epoch 10, indicating overfitting.",
        )

    def answer_with_image(self, image_b64, user_question, context="", reference_context=""):
        return self._reply(
            "answer_with_image",
            "The validation loss begins to increase after epoch 6 (value 0.31). "
            "From that point the training loss keeps falling while validation loss "
            "rises to 0.71 — the classic signature of overfitting.",
        )

    def answer_without_image(self, user_question, context="", reference_context=""):
        return self._reply(
            "answer_without_image",
            "Based on the earlier image, the middle (hidden) layer transforms the "
            "input representation into features the output layer can separate.",
        )

    def classify_request(self, user_text: str, has_image: bool) -> dict:
        return {
            "intent": "unknown",
            "specificity": "unknown",
            "needs_clarification": False,
            "clarifying_question": "",
            "reasoning": "fake",
            "fallback": True,   # forces the router to stay deterministic
        }

    def check_entailment(self, claim: str, evidence: str) -> dict:
        return {"supported": True, "confidence": 0.9, "reason": "fake check"}

    def health_check(self):
        from vision_model import ModelHealth

        return ModelHealth(server_up=True, vision_model_available=True)


@pytest.fixture
def fake_vision() -> FakeVisionModel:
    return FakeVisionModel()


@pytest.fixture
def validator() -> ResponseValidator:
    """Validator without an embedder → lexical checks, no downloads."""
    return ResponseValidator(embedder=None, entailment_checker=None)


@pytest.fixture
def good_diagnostics() -> ImageDiagnostics:
    return ImageDiagnostics(
        valid=True,
        width=900,
        height=400,
        format="PNG",
        blur_score=500.0,
        looks_like_chart_or_diagram=True,
    )


@pytest.fixture
def blurry_diagnostics() -> ImageDiagnostics:
    return ImageDiagnostics(
        valid=True,
        width=900,
        height=400,
        format="PNG",
        blur_score=12.0,
        is_blurry=True,
        looks_like_chart_or_diagram=True,
    )


@pytest.fixture
def ok_ocr() -> OCRResult:
    return OCRResult(
        text="Training Loss 0.42\nValidation Loss 0.71",
        confidence=0.88,
        backend="tesseract",
        attempted=True,
        succeeded=True,
    )
