"""
Operational endpoints: liveness, model health, configuration and the knowledge
base. The front-end's sidebar status panel is driven entirely from here.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.core import config
from backend.services.engine import engine

log = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    """Fast liveness probe — does not touch Ollama."""
    return {"status": "ok", "version": "2.0.0"}


@router.get("/system/status")
def system_status() -> dict:
    """Full readiness picture: Ollama, models, knowledge base, routing config."""
    health = engine.health()
    try:
        retrieval = engine.stats()["retrieval"]
    except Exception:  # noqa: BLE001
        retrieval = {"backend": "unavailable", "chunks": 0, "documents": []}

    return {
        "ollama": {
            "host": config.OLLAMA_HOST,
            "server_up": health.server_up,
            "vision_model": config.VISION_MODEL,
            "vision_ready": health.vision_model_available,
            "text_model": engine.vision.text_model,
            "text_ready": bool(
                health.text_model_available or health.text_model_fallback
            ),
            "text_fallback": health.text_model_fallback,
            "installed_models": health.installed_models,
        },
        "knowledge_base": {
            "backend": retrieval.get("backend"),
            "chunks": retrieval.get("chunks", 0),
            "documents": retrieval.get("documents", []),
        },
        "config": {
            "ocr_enabled": config.OCR_ENABLED,
            "ocr_backend": config.OCR_BACKEND,
            "default_language": config.DEFAULT_LANGUAGE,
            "max_turns": config.MAX_TURNS,
            "top_k": config.RETRIEVAL_TOP_K,
        },
        "ready": bool(health.server_up and health.vision_model_available),
    }


@router.get("/system/models")
def models() -> dict:
    health = engine.health()
    return {
        "server_up": health.server_up,
        "installed": health.installed_models,
        "vision": {"name": config.VISION_MODEL, "ready": health.vision_model_available},
        "text": {
            "name": engine.vision.text_model,
            "ready": bool(health.text_model_available or health.text_model_fallback),
            "fallback_of": health.text_model_fallback,
        },
    }


@router.get("/documents")
def list_documents() -> dict:
    """The Markdown/text files the retriever indexes."""
    root = Path(config.DOCUMENTS_DIR)
    if not root.exists():
        return {"documents": []}
    documents = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            documents.append(
                {
                    "name": path.name,
                    "path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {"documents": documents}


@router.post("/documents/reindex")
def reindex() -> dict:
    """Rebuild the vector index after files were added or changed."""
    try:
        backend = engine.ensure_index(force=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("Reindex failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"backend": backend, "status": "reindexed"}
