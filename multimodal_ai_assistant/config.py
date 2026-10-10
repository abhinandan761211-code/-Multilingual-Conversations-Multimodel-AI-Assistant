"""
Central configuration for the Multimodal AI Assistant.

Everything tunable lives here so the rest of the code stays declarative.
Values can be overridden with environment variables (useful for the demo /
teacher evaluation where a different model may already be pulled).
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma_store"
UPLOAD_CACHE_DIR = DATA_DIR / "upload_cache"

for _d in (DATA_DIR, DOCUMENTS_DIR, CHROMA_DIR, UPLOAD_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Ollama / models ───────────────────────────────────────────────────────────
# NOTE ON THE VISION MODEL
# The brief suggests llama3.2-vision. Recent Ollama builds bundle a llama.cpp
# that no longer loads its 'mllama' architecture, which fails with
# "unknown model architecture: 'mllama'" before any inference happens (see
# ollama issues #16490, #16547, #18486). qwen2.5vl is a modern drop-in that
# supports the same image+text interface and is stronger at OCR and chart
# reading, so it is the default here.
#
# To use llama3.2-vision anyway (on an older Ollama that still supports mllama):
#     VISION_MODEL=llama3.2-vision:11b streamlit run app.py
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
TEXT_MODEL = os.getenv("TEXT_MODEL", "llama3.2:3b")

# Generation parameters
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.2"))
TEXT_TEMPERATURE = float(os.getenv("TEXT_TEMPERATURE", "0.3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))  # seconds

# ── Image handling ────────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 20 * 1024 * 1024          # 20 MB upload cap
MIN_IMAGE_DIMENSION = 64                    # below this → "too small / blurry"
SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "WEBP", "BMP", "GIF", "TIFF"}
# Laplacian-variance thresholds for a rough sharpness estimate
BLUR_THRESHOLD = 60.0

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") == "1"
OCR_BACKEND = os.getenv("OCR_BACKEND", "auto")   # auto | easyocr | tesseract | none
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "12"))

# ── Retrieval / evidence ──────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.25"))

# ── Conversation memory ───────────────────────────────────────────────────────
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))          # keep last N turns
MAX_IMAGE_BYTES_IN_MEMORY = 8 * 1024 * 1024            # don't hoard huge blobs

# ── Chat persistence ──────────────────────────────────────────────────────────
# Conversations are written to data/chats/ so history survives a refresh.
PERSIST_CHATS = os.getenv("PERSIST_CHATS", "1") == "1"
MAX_STORED_CONVERSATIONS = int(os.getenv("MAX_STORED_CONVERSATIONS", "100"))
# Cap the stored image so a 20 MB upload doesn't bloat every save.
MAX_STORED_IMAGE_BYTES = int(os.getenv("MAX_STORED_IMAGE_BYTES", str(5 * 1024 * 1024)))

# ── Router behaviour ──────────────────────────────────────────────────────────
# Confidence below this on the "understandable image" check triggers clarification.
AMBIGUITY_CONFIDENCE_THRESHOLD = float(os.getenv("AMBIGUITY_CONFIDENCE_THRESHOLD", "0.45"))
# Answers whose grounding score is below this get flagged / downgraded.
GROUNDING_MIN_SCORE = float(os.getenv("GROUNDING_MIN_SCORE", "0.30"))
