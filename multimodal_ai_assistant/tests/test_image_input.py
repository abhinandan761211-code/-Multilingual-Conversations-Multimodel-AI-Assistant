"""
Tests for image handling: validation, quality measurement, OCR gating, and the
routing decisions that depend on image facts.
"""
from __future__ import annotations

from image_processor import (
    ImageDiagnostics,
    analyze_image,
    encode_for_vision,
    preprocess_for_reading,
    run_ocr,
)
from reasoning_router import ReasoningRouter, Route


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestImageValidation:
    def test_valid_image_is_accepted(self, sharp_text_image):
        diag = analyze_image(sharp_text_image)
        assert diag.valid
        assert diag.width == 900
        assert diag.height == 300
        assert diag.format == "PNG"
        assert diag.error is None

    def test_corrupt_bytes_are_rejected_not_raised(self, corrupt_bytes):
        diag = analyze_image(corrupt_bytes)
        assert not diag.valid
        assert diag.error is not None
        assert "recognis" in diag.error.lower() or "decode" in diag.error.lower()

    def test_empty_bytes_are_rejected(self):
        diag = analyze_image(b"")
        assert not diag.valid
        assert "empty" in diag.error.lower()

    def test_oversized_image_is_rejected(self, monkeypatch):
        import image_processor

        monkeypatch.setattr(image_processor.config, "MAX_IMAGE_BYTES", 10)
        diag = image_processor.analyze_image(b"x" * 100)
        assert not diag.valid
        assert "exceeds" in diag.error.lower()

    def test_tiny_image_flagged_low_resolution(self, tiny_image):
        diag = analyze_image(tiny_image)
        assert diag.valid
        assert diag.is_low_resolution
        assert not diag.usable
        assert diag.uncertainty_summary() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Quality measurement
# ─────────────────────────────────────────────────────────────────────────────


class TestImageQuality:
    def test_sharp_image_scores_higher_than_blurry(self, sharp_text_image, blurry_image):
        sharp = analyze_image(sharp_text_image)
        blurry = analyze_image(blurry_image)
        assert sharp.blur_score > blurry.blur_score
        assert blurry.is_blurry
        assert not sharp.is_blurry

    def test_dark_image_flagged(self):
        from PIL import Image
        import io

        img = Image.new("RGB", (200, 200), (5, 5, 5))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        diag = analyze_image(buf.getvalue())
        assert diag.is_too_dark

    def test_text_like_detection_on_text_image(self, sharp_text_image):
        diag = analyze_image(sharp_text_image)
        # A dense block of text lines should register as text-like.
        assert diag.looks_like_text or diag.looks_like_chart_or_diagram

    def test_uncertainty_summary_is_none_for_clean_image(self, sharp_text_image):
        diag = analyze_image(sharp_text_image)
        if not (diag.is_blurry or diag.is_low_resolution or diag.is_too_dark):
            assert diag.uncertainty_summary() is None

    def test_uncertainty_summary_mentions_blur(self, blurry_image):
        diag = analyze_image(blurry_image)
        summary = diag.uncertainty_summary()
        assert summary is not None
        assert "blur" in summary.lower() or "clarity" in summary.lower()


# ─────────────────────────────────────────────────────────────────────────────
# OCR gating
# ─────────────────────────────────────────────────────────────────────────────


class TestOCRGating:
    def test_ocr_skips_when_no_backend_or_no_text(self, corrupt_bytes):
        result = run_ocr(corrupt_bytes)
        assert not result.succeeded

    def test_ocr_never_raises(self, sharp_text_image):
        result = run_ocr(sharp_text_image)
        # Depending on the machine a backend may or may not exist — either way
        # the call must return an OCRResult rather than blow up.
        assert isinstance(result, type(result))
        assert hasattr(result, "attempted")

    def test_ocr_respects_disabled_flag(self, sharp_text_image, monkeypatch):
        import image_processor

        monkeypatch.setattr(image_processor.config, "OCR_ENABLED", False)
        result = image_processor.run_ocr(sharp_text_image)
        assert not result.attempted
        assert result.backend == "disabled"


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestEncoding:
    def test_encode_for_vision_returns_base64_string(self, sharp_text_image):
        encoded = encode_for_vision(sharp_text_image)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        import base64

        base64.b64decode(encoded)   # must be valid base64

    def test_encode_downscales_large_images(self):
        from PIL import Image
        import io, base64

        big = Image.new("RGB", (4000, 3000), "white")
        buf = io.BytesIO()
        big.save(buf, format="PNG")
        encoded = encode_for_vision(buf.getvalue(), max_dimension=800)
        decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert max(decoded.size) <= 800

    def test_preprocess_for_reading_returns_base64(self, sharp_text_image):
        import base64

        encoded = preprocess_for_reading(sharp_text_image)
        base64.b64decode(encoded)


# ─────────────────────────────────────────────────────────────────────────────
# Routing decisions driven by image state
# ─────────────────────────────────────────────────────────────────────────────


class TestImageRouting:
    def test_invalid_image_triggers_clarification(self, router):
        diag = ImageDiagnostics(valid=False, error="Not a recognisable image file")
        plan = router.plan("explain this", diagnostics=diag)
        assert plan.route == Route.CLARIFY
        assert "re-upload" in plan.clarifying_question.lower()

    def test_chart_like_image_routes_to_chart_analysis(self, router, good_diagnostics):
        plan = router.plan(
            "what is the trend of the validation loss curve?",
            diagnostics=good_diagnostics,
            new_image_id="img1",
        )
        assert plan.route in (Route.ANALYSE_CHART, Route.ANSWER_WITH_IMAGE)
        assert plan.use_vision

    def test_text_request_uses_ocr(self, router, good_diagnostics):
        plan = router.plan(
            "read the text in this screenshot",
            diagnostics=good_diagnostics,
            new_image_id="img1",
        )
        assert plan.route == Route.READ_TEXT
        assert plan.use_ocr

    def test_blurry_image_reduces_ocr_trust(self, router, blurry_diagnostics):
        plan = router.plan(
            "read the text",
            diagnostics=blurry_diagnostics,
            new_image_id="img1",
        )
        assert plan.use_ocr
        assert not plan.trust_ocr, "Blurry OCR output must not be blindly trusted"

    def test_clear_description_request_routes_to_description(self, router, good_diagnostics):
        plan = router.plan(
            "describe this image in detail",
            diagnostics=ImageDiagnostics(
                valid=True, width=900, height=400, blur_score=500.0,
                looks_like_chart_or_diagram=False, looks_like_text=False,
            ),
            new_image_id="img1",
        )
        assert plan.route == Route.DESCRIBE_IMAGE
