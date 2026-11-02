"""
app.py
======
FastAPI application factory.

Run it with:
    uvicorn backend.api.app:app --reload --port 8000
The interactive docs are then at http://localhost:8000/docs.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import chat, system, widgets
from backend.core import config
from backend.db.database import init_db
from backend.services.engine import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create tables on startup; build the knowledge base in the background."""
    init_db()
    log.info("Database ready at %s", config.DATABASE_URL)
    # Warm the retriever so the first chat request is not the slow one. A failure
    # here (no embeddings installed) is not fatal: retrieval degrades to keyword.
    try:
        backend = engine.ensure_index()
        log.info("Knowledge base indexed (backend=%s)", backend)
    except Exception as exc:  # noqa: BLE001
        log.warning("Knowledge base indexing skipped: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multimodal AI Assistant API",
        version="2.0.0",
        description=(
            "Local, image-aware assistant: routing, OCR, retrieval, validation "
            "and multilingual answering, exposed over HTTP."
        ),
        lifespan=lifespan,
    )

    # The Streamlit front-end calls this from its own server process, and the
    # API may also be driven from a notebook. Local demo scope only.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/api")
    app.include_router(system.router, prefix="/api")
    app.include_router(widgets.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "service": "Multimodal AI Assistant API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()
