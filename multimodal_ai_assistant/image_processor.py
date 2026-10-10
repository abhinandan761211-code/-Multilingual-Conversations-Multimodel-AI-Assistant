"""
image_processor.py
==================
Image validation + optional OCR.

This module is deliberately *pure* with respect to the model: it never calls a
VLM. Its job is to give the router hard evidence about an image so the router
can decide whether OCR is even worth trying, and so the validator can later
report uncertainty that is grounded in measurable facts (size, blur, DPI),
not vibes.

Public surface
--------------
- ImageDiagnostics        : pydantic model describing what we measured
- analyze_image(bytes)    : validate + measure (never raises on bad images)
- run_ocr(bytes, diag)    : OCR only when the image plausibly contains text
- OCRResult               : text + confidence + backend used
- encode_for_vision(bytes): bytes -> base64 that Ollama accepts
- detect_text_likelihood() : heuristic "does this image contain text?"
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError
from pydantic import BaseModel, Field

import config
from config import (
    BLUR_THRESHOLD,
    MIN_IMAGE_DIMENSION,
    OCR_MIN_TEXT_CHARS,
    SUPPORTED_FORMATS,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


class ImageDiagnostics(BaseModel):
    """Measured, verifiable facts about an image. No interpretation here."""

    valid: bool = False
    error: Optional[str] = None

    width: int = 0
    height: int = 0
    format: Optional[str] = None
    mode: Optional[str] = None
    size_bytes: int = 0
    aspect_ratio: float = 0.0

    # Quality signals used for uncertainty reporting
    blur_score: float = 0.0          # variance of Laplacian; higher = sharper
    is_blurry: bool = False
    is_low_resolution: bool = False
    mean_brightness: float = 0.0
    is_too_dark: bool = False
    is_too_bright: bool = False

    # Heuristic flags (NOT OCR — computed from pixel statistics)
    looks_like_text: bool = False
    looks_like_chart_or_diagram: bool = False
    dominant_colors: int = 0

    quality_notes: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Enough signal to attempt a vision read?"""
        return self.valid and not self.is_low_resolution

    def uncertainty_summary(self) -> Optional[str]:
        """Human-readable uncertainty line, or None if the image is clean."""
        if not self.valid:
            return f"Image could not be read ({self.error})."
        problems = []
        if self.is_low_resolution:
            problems.append(
                f"resolution is low ({self.width}x{self.height})"
            )
        if self.is_blurry:
            problems.append("the image appears blurry or out of focus")
        if self.is_too_dark:
            problems.append("the image is quite dark")
        if self.is_too_bright:
            problems.append("the image is over-exposed")
        if not problems:
            return None
        return "Visual clarity is limited: " + "; ".join(problems) + "."


class OCRResult(BaseModel):
    text: str = ""
    confidence: float = 0.0
    backend: str = "none"
    attempted: bool = False
    succeeded: bool = False
    error: Optional[str] = None

    @property
    def has_usable_text(self) -> bool:
        return len(self.text.strip()) >= OCR_MIN_TEXT_CHARS


# ─────────────────────────────────────────────────────────────────────────────
# Validation & measurement
# ─────────────────────────────────────────────────────────────────────────────


def _laplacian_variance(gray: np.ndarray) -> float:
    """
    Variance of the Laplacian — the classic cheap blur metric.
    Implemented with a 3x3 kernel via convolution so we don't need OpenCV.
    """
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Pad manually and slide — avoids adding scipy as a dependency.
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    response = np.einsum("ijkl,kl->ij", windows, kernel)
    return float(response.var())


def _edge_density(gray_small: np.ndarray) -> float:
    """
    Fraction of pixels that are strong edges. Text and line-charts produce
    a lot of high-frequency edges; photographs generally do not.
    """
    gy, gx = np.gradient(gray_small)
    magnitude = np.hypot(gx, gy)
    if magnitude.max() <= 0:
        return 0.0
    threshold = magnitude.mean() + magnitude.std()
    return float((magnitude > threshold).mean())


def _estimate_text_likelihood(img: Image.Image) -> tuple[bool, bool]:
    """
    Heuristic (no ML) guess at: (looks_like_text, looks_like_chart_or_diagram).

    Text  → dense, uniformly distributed, high-contrast edges.
    Chart → sparse straight lines with large empty background regions.
    """
    gray = img.convert("L")
    small = gray.resize((160, 160))
    arr = np.asarray(small, dtype=np.float32)

    # Binarise around the mean to isolate ink from background.
    binary = (arr < arr.mean() - 0.5 * arr.std()).astype(np.float32)
    ink_ratio = float(binary.mean())
    edge_density = _edge_density(arr)

    # Row-wise ink profile: text has *several horizontal bands* with ink
    # (text lines), separated by near-blank gaps. A solid photo fills rows
    # uniformly; a chart has few, tall inked regions.
    row_ink = binary.mean(axis=1)
    rows_with_ink = float((row_ink > 0.02).mean())
    # Count transitions blank→ink, i.e. roughly how many text lines exist.
    inked = row_ink > 0.02
    transitions = int(np.sum(np.diff(inked.astype(int)) == 1)) if inked.size else 0

    # Text: moderate ink coverage, many distinct inked rows arranged as bands.
    looks_like_text = (
        0.01 < ink_ratio < 0.55
        and edge_density > 0.04
        and rows_with_ink > 0.15
        and transitions >= 2
    )

    # Charts/graphs: sparse ink with lots of blank area and some edges. This is
    # a superset of "text" (a text document also has blank space), so we flag
    # chart only for sparse, low-band-count images — which catches plots.
    blank_ratio = float((arr > arr.mean() + arr.std()).mean())
    looks_like_chart = (
        ink_ratio < 0.30
        and blank_ratio > 0.15
        and edge_density > 0.02
        and not looks_like_text
    )
    return looks_like_text, looks_like_chart


def analyze_image(data: bytes) -> ImageDiagnostics:
    """
    Validate and measure an uploaded image.

    Never raises: a broken upload returns ImageDiagnostics(valid=False, error=...).
    That keeps the router logic simple and makes failure a first-class case.
    """
    diag = ImageDiagnostics(size_bytes=len(data))

    if not data:
        diag.error = "Empty file"
        return diag

    if len(data) > config.MAX_IMAGE_BYTES:
        diag.error = (
            f"File is {len(data) / 1e6:.1f} MB which exceeds the "
            f"{config.MAX_IMAGE_BYTES / 1e6:.0f} MB limit"
        )
        return diag

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()                       # cheap structural check
        img = Image.open(io.BytesIO(data))  # verify() closes the fp → reopen
    except UnidentifiedImageError:
        diag.error = "Not a recognisable image file"
        return diag
    except Exception as exc:               # noqa: BLE001 - we want everything
        diag.error = f"Could not decode image ({exc})"
        return diag

    diag.valid = True
    diag.format = img.format
    diag.mode = img.mode
    diag.width, diag.height = img.size
    diag.aspect_ratio = round(img.width / img.height, 3) if img.height else 0.0

    if img.format and img.format.upper() not in SUPPORTED_FORMATS:
        diag.quality_notes.append(f"Unusual format: {img.format}")

    # ── Quality metrics ──────────────────────────────────────────────────────
    rgb = img.convert("RGB")
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    diag.mean_brightness = float(gray.mean())

    # Downscale before the blur metric so huge images don't dominate runtime.
    if max(img.size) > 1024:
        scale = 1024 / max(img.size)
        blur_img = img.convert("L").resize(
            (max(3, int(img.width * scale)), max(3, int(img.height * scale)))
        )
    else:
        blur_img = img.convert("L")
    diag.blur_score = _laplacian_variance(np.asarray(blur_img, dtype=np.float32))

    diag.is_low_resolution = min(img.width, img.height) < MIN_IMAGE_DIMENSION
    diag.is_blurry = diag.blur_score < BLUR_THRESHOLD
    diag.is_too_dark = diag.mean_brightness < 35

    # Colour richness — cheap proxy for "is this a screenshot/diagram or photo"
    small_rgb = np.asarray(rgb.resize((64, 64)), dtype=np.uint8).reshape(-1, 3)
    quantised = (small_rgb // 32)
    diag.dominant_colors = int(len(np.unique(quantised, axis=0)))

    diag.looks_like_text, diag.looks_like_chart_or_diagram = _estimate_text_likelihood(rgb)

    # Over-exposure is only meaningful when it destroys detail. A white-background
    # document or a clean chart naturally has a high mean brightness, so require
    # both a very bright mean AND low local contrast (i.e. washed-out detail).
    contrast = float(gray.std())
    diag.is_too_bright = diag.mean_brightness > 245 and contrast < 12.0

    # ── Human-readable notes ─────────────────────────────────────────────────
    if diag.is_low_resolution:
        diag.quality_notes.append(
            f"Very small image ({diag.width}x{diag.height}); fine detail may be lost"
        )
    if diag.is_blurry:
        diag.quality_notes.append(
            f"Low sharpness score ({diag.blur_score:.1f}); text may be unreadable"
        )
    if diag.is_too_dark:
        diag.quality_notes.append("Image is very dark; contrast may be poor")
    if diag.is_too_bright:
        diag.quality_notes.append("Image is over-exposed; details may be washed out")

    return diag


# ─────────────────────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_reader = None          # lazily initialised; loading is slow (~seconds)
_easyocr_failed = False


def _try_easyocr(img: Image.Image) -> Optional[OCRResult]:
    global _easyocr_reader, _easyocr_failed
    if _easyocr_failed:
        return None
    try:
        import easyocr  # type: ignore
    except Exception:  # noqa: BLE001
        _easyocr_failed = True
        return None

    try:
        if _easyocr_reader is None:
            # English only keeps the model download small for the demo.
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = _easyocr_reader.readtext(np.asarray(img))
    except Exception as exc:  # noqa: BLE001
        log.warning("EasyOCR failed: %s", exc)
        _easyocr_failed = True
        return None

    if not results:
        return OCRResult(attempted=True, succeeded=True, backend="easyocr")

    texts = [r[1] for r in results]
    confidences = [float(r[2]) for r in results]
    return OCRResult(
        text="\n".join(texts),
        confidence=round(sum(confidences) / len(confidences), 3),
        backend="easyocr",
        attempted=True,
        succeeded=True,
    )


def _try_tesseract(img: Image.Image) -> Optional[OCRResult]:
    try:
        import pytesseract  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        # Upscaling small text helps Tesseract a lot.
        if min(img.size) < 1000:
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Tesseract failed: %s", exc)
        return None

    words, confs = [], []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        if word and word.strip() and conf not in ("-1", -1):
            words.append(word.strip())
            confs.append(float(conf) / 100.0)

    return OCRResult(
        text=" ".join(words),
        confidence=round(sum(confs) / len(confs), 3) if confs else 0.0,
        backend="tesseract",
        attempted=True,
        succeeded=True,
    )


def run_ocr(data: bytes, diag: Optional[ImageDiagnostics] = None) -> OCRResult:
    """
    Extract text from an image. Returns attempted=False when skipped.

    Skipping is itself an intelligent decision: we don't burn seconds of OCR on
    an image that our pixel heuristics say is a photograph.
    """
    if not config.OCR_ENABLED:
        return OCRResult(backend="disabled")

    diag = diag or analyze_image(data)
    if not diag.valid:
        return OCRResult(backend="none", error=diag.error)

    backend_pref = config.OCR_BACKEND

    # Smart skip: don't OCR photos with no text-like structure.
    if not diag.looks_like_text and not diag.looks_like_chart_or_diagram:
        if diag.blur_score > 400 and diag.dominant_colors > 40:
            return OCRResult(backend="skipped", error="no text-like structure detected")

    try:
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return OCRResult(backend="none", error=str(exc))

    if backend_pref == "auto":
        # Tesseract is fast and near-universally installed; try it first,
        # fall back to EasyOCR when it isn't available.
        order = [_try_tesseract, _try_easyocr]
    elif backend_pref == "easyocr":
        order = [_try_easyocr, _try_tesseract]
    elif backend_pref == "tesseract":
        order = [_try_tesseract]
    else:
        order = []

    if not order:
        return OCRResult(backend="disabled")

    for backend in order:
        result = backend(img)
        if result is not None:
            return result

    return OCRResult(
        backend="none",
        error="No OCR backend available (install pytesseract + tesseract, or easyocr)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vision payload helper
# ─────────────────────────────────────────────────────────────────────────────


def encode_for_vision(data: bytes, max_dimension: int = 1280) -> str:
    """
    Return base64 PNG/JPEG suitable for the Ollama vision API.

    Large images get downscaled: the VLM gains nothing from a 6000px photo but
    pays a lot of latency for it.
    """
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    if max(img.size) > max_dimension:
        scale = max_dimension / max(img.size)
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
        )

    buffer = io.BytesIO()
    fmt = "PNG" if img.mode == "L" else "JPEG"
    save_kwargs = {"format": fmt}
    if fmt == "JPEG":
        save_kwargs.update(quality=88, optimize=True)
    img.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def preprocess_for_reading(data: bytes) -> str:
    """
    Aggressive enhancement for text-heavy images before OCR / VLM reading.
    Grayscale + autocontrast + mild sharpening.
    """
    img = Image.open(io.BytesIO(data)).convert("L")
    stat = ImageStat.Stat(img)
    mean = stat.mean[0]
    # Autocontrast pulls faint handwriting out of a grey background.
    from PIL import ImageOps

    img = ImageOps.autocontrast(img, cutoff=1)
    if mean > 200:                       # washed-out scan
        img = ImageOps.invert(img)
    img = img.filter(ImageFilter.SHARPEN)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
