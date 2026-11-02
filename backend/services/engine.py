"""
engine.py
=========
Process-wide ownership of the expensive, stateful collaborators.

Why this exists
---------------
`AssistantPipeline` is cheap to construct but internally owns conversation
memory. The pieces *inside* it — the vision model, the knowledge-base retriever
and the sentence-transformer used for validation — are expensive: loading the
embedder used to add ~90 s to the first page render.

So there is exactly one `Engine` per process:

* one `VisionModel`  (all Ollama traffic goes through it)
* one `EvidenceRetriever` (the vector index is built once)
* one `ResponseValidator` (shared, thread-safe checks)
* one `AssistantPipeline` **per conversation**, because memory is per chat

The old Streamlit app re-created a pipeline on every client refresh for the same
reason; here the lifetime is keyed by conversation id, and the API can serve
many clients without each one rebuilding the world.
"""
from __future__ import annotations

import logging
import threading

from backend.core import config
from backend.core.conversation_memory import ConversationMemory
from backend.core.evidence_retriever import EvidenceRetriever
from backend.core.pipeline import AssistantPipeline
from backend.core.reasoning_router import ReasoningRouter
from backend.core.response_validator import ResponseValidator
from backend.core.vision_model import VisionModel

log = logging.getLogger(__name__)

# Keep memory bounded on a long-running server. Oldest conversations are evicted
# first; the transcript itself is always safe in the database.
MAX_CACHED_PIPELINES = 32


class _LazyEmbedder:
    """
    Duck-types the subset of the SentenceTransformer API the validator uses
    (`encode`), loading the real model on first call.

    `ResponseValidator` already treats a failing embedder as "fall back to
    lexical checks", so a load failure degrades gracefully instead of breaking
    validation.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._failed = False
        self._lock = threading.Lock()

    def encode(self, texts, show_progress_bar: bool = False):  # noqa: D401
        if self._failed:
            raise RuntimeError("embedder unavailable")
        if self._model is None:
            with self._lock:
                if self._model is None and not self._failed:
                    try:
                        from sentence_transformers import (
                            SentenceTransformer,  # type: ignore
                        )

                        self._model = SentenceTransformer(self._model_name)
                    except Exception as exc:  # noqa: BLE001
                        self._failed = True
                        log.warning(
                            "Embedder unavailable (%s); lexical checks only", exc
                        )
        if self._model is None:
            raise RuntimeError("embedder unavailable")
        return self._model.encode(texts, show_progress_bar=show_progress_bar)


class Engine:
    """Lazily-built singletons plus per-conversation pipelines."""

    def __init__(self) -> None:
        self._vision: VisionModel | None = None
        self._retriever: EvidenceRetriever | None = None
        self._validator: ResponseValidator | None = None
        self._index_backend: str | None = None
        self._pipelines: dict[str, AssistantPipeline] = {}
        self._lock = threading.RLock()

    # ── shared singletons ───────────────────────────────────────────────────

    @property
    def vision(self) -> VisionModel:
        with self._lock:
            if self._vision is None:
                model = VisionModel()
                # A missing text model would otherwise fail every text-only
                # turn. Swap in an available local model once, at startup.
                model.resolve_available_models()
                self._vision = model
            return self._vision

    @property
    def retriever(self) -> EvidenceRetriever:
        with self._lock:
            if self._retriever is None:
                self._retriever = EvidenceRetriever()
            return self._retriever

    @property
    def validator(self) -> ResponseValidator:
        with self._lock:
            if self._validator is None:
                self._validator = ResponseValidator(
                    embedder=_LazyEmbedder(),
                    entailment_checker=self.vision.check_entailment,
                )
            return self._validator

    # ── knowledge base ──────────────────────────────────────────────────────

    def ensure_index(self, force: bool = False) -> str:
        with self._lock:
            if self._index_backend is None or force:
                self._index_backend = self.retriever.build_index(force=force)
                log.info("Evidence index ready (backend=%s)", self._index_backend)
            return self._index_backend

    # ── pipelines ───────────────────────────────────────────────────────────

    def pipeline(self, conversation_id: str) -> AssistantPipeline:
        """The pipeline for one conversation (created on first use)."""
        with self._lock:
            pipeline = self._pipelines.get(conversation_id)
            if pipeline is not None:
                return pipeline

            if len(self._pipelines) >= MAX_CACHED_PIPELINES:
                oldest = next(iter(self._pipelines))
                self._pipelines.pop(oldest, None)

            pipeline = AssistantPipeline(
                vision_model=self.vision,
                retriever=self.retriever,
                memory=ConversationMemory(),
                router=ReasoningRouter(classifier=self.vision.classify_request),
                validator=self.validator,
            )
            self._pipelines[conversation_id] = pipeline
            return pipeline

    def reset_pipeline(self, conversation_id: str) -> None:
        with self._lock:
            self._pipelines.pop(conversation_id, None)

    def seed_image(
        self, conversation_id: str, image_bytes: bytes, filename: str, summary: str = ""
    ) -> None:
        """Re-attach a stored image so follow-up turns work after a restart."""
        pipeline = self.pipeline(conversation_id)
        record = pipeline.memory.register_image(image_bytes, filename=filename)
        if summary:
            pipeline.memory.annotate_image(record.image_id, summary=summary[:600])

    # ── health / stats ──────────────────────────────────────────────────────

    def health(self):
        return self.vision.health_check()

    def stats(self) -> dict:
        try:
            stats = self.retriever.stats()
        except Exception as exc:  # noqa: BLE001
            stats = {"backend": "unavailable", "chunks": 0, "error": str(exc)}
        return {
            "vision_model": self.vision.vision_model,
            "text_model": self.vision.text_model,
            "retrieval": {
                "backend": stats.get("backend", "unknown"),
                "chunks": stats.get("chunks", 0),
                "documents": stats.get("documents", []),
            },
            "cached_conversations": len(self._pipelines),
        }


# Module-level singleton, imported by routes.
engine = Engine()
