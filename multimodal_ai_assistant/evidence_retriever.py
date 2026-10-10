"""
evidence_retriever.py
=====================
Retrieval over a small, trusted corpus of documents so the assistant can cite
supporting evidence instead of free-associating.

Two backends:
  * ChromaDB + Sentence-Transformers  (preferred — semantic search)
  * keyword fallback                  (TF-style scoring, zero dependencies)

The fallback matters: a demo machine may not have the embedding model cached or
may be offline. Degrading to keyword search keeps the assistant honest about
*what it can support*, which is the whole point of this module.

Everything returned is an `Evidence` with a source and a score, so the
validator can reason about grounding strength.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import config

log = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "by", "for", "with", "about", "as", "into", "and",
    "or", "but", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "i", "you", "we", "they", "he", "she", "what", "which", "who",
    "how", "why", "when", "where", "do", "does", "did", "can", "could",
    "would", "should", "will", "explain", "please", "tell", "me", "my",
}


@dataclass
class Evidence:
    text: str
    source: str
    score: float = 0.0
    chunk_id: str = ""
    backend: str = ""

    def citation(self) -> str:
        return f"{self.source}" + (f"#{self.chunk_id}" if self.chunk_id else "")


class EvidenceRetriever:
    """
    Indexes `data/documents/*.md|txt` and retrieves the most relevant chunks.

    Usage:
        retriever = EvidenceRetriever()
        retriever.build_index()          # idempotent-ish
        hits = retriever.retrieve("what is overfitting", top_k=3)
    """

    def __init__(
        self,
        documents_dir: Path = config.DOCUMENTS_DIR,
        persist_dir: Path = config.CHROMA_DIR,
        embedding_model: str = config.EMBEDDING_MODEL,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.persist_dir = Path(persist_dir)
        self.embedding_model_name = embedding_model

        self._chunks: list[dict] = []
        self._embedder = None
        self._collection = None
        self.backend = "not_built"
        self._chunk_vectors: list[list[float]] = []

    # ── indexing ────────────────────────────────────────────────────────────

    def _load_documents(self) -> list[tuple[str, str]]:
        docs: list[tuple[str, str]] = []
        if not self.documents_dir.exists():
            return docs
        for path in sorted(self.documents_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                try:
                    docs.append((path.name, path.read_text(encoding="utf-8")))
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not read %s: %s", path, exc)
        return docs

    @staticmethod
    def _chunk(text: str, source: str, max_chars: int = 900) -> list[dict]:
        """
        Split on blank lines so chunks respect the document's own structure,
        then merge small paragraphs up to `max_chars`.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[dict] = []
        buffer = ""
        index = 0
        for para in paragraphs:
            if buffer and len(buffer) + len(para) + 2 > max_chars:
                chunks.append(
                    {
                        "id": f"{source}-{index}",
                        "text": buffer.strip(),
                        "source": source,
                    }
                )
                index += 1
                buffer = para
            else:
                buffer = f"{buffer}\n\n{para}" if buffer else para
        if buffer.strip():
            chunks.append(
                {"id": f"{source}-{index}", "text": buffer.strip(), "source": source}
            )
        return chunks

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer(self.embedding_model_name)
        except Exception as exc:  # noqa: BLE001
            log.warning("Embedding model unavailable (%s); using keyword search", exc)
            self._embedder = False
        return self._embedder

    def build_index(self, force: bool = False) -> str:
        """Build (or rebuild) the index. Returns the backend actually used."""
        if self._chunks and not force:
            return self.backend

        documents = self._load_documents()
        self._chunks = []
        for source, text in documents:
            self._chunks.extend(self._chunk(text, source))

        if not self._chunks:
            self.backend = "empty"
            return self.backend

        embedder = self._get_embedder()
        if embedder:
            try:
                import chromadb  # type: ignore
                from chromadb.config import Settings  # type: ignore

                client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                name = "evidence"
                try:
                    client.delete_collection(name)
                except Exception:  # noqa: BLE001
                    pass
                collection = client.create_collection(name)

                texts = [c["text"] for c in self._chunks]
                embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
                collection.add(
                    ids=[c["id"] for c in self._chunks],
                    documents=texts,
                    metadatas=[{"source": c["source"]} for c in self._chunks],
                    embeddings=embeddings,
                )
                self._collection = collection
                self._chunk_vectors = embeddings
                self.backend = "chromadb"
                return self.backend
            except Exception as exc:  # noqa: BLE001
                log.warning("ChromaDB unavailable (%s); using numpy vectors", exc)
                texts = [c["text"] for c in self._chunks]
                self._chunk_vectors = embedder.encode(
                    texts, show_progress_bar=False
                ).tolist()
                self.backend = "numpy+embeddings"
                return self.backend

        self.backend = "keyword"
        return self.backend

    # ── retrieval ───────────────────────────────────────────────────────────

    @staticmethod
    def _keywords(text: str) -> set[str]:
        return {
            w for w in re.findall(r"[a-z0-9']+", text.lower())
            if w not in _STOPWORDS and len(w) > 2
        }

    def _keyword_search(self, query: str, top_k: int) -> list[Evidence]:
        query_terms = self._keywords(query)
        if not query_terms:
            return []
        scored: list[Evidence] = []
        for chunk in self._chunks:
            chunk_terms = self._keywords(chunk["text"])
            if not chunk_terms:
                continue
            overlap = query_terms & chunk_terms
            if not overlap:
                continue
            # Cosine-ish score with an IDF flavour for rare terms.
            idf = sum(
                1.0 / (1.0 + math.log(1 + sum(
                    1 for c in self._chunks if term in self._keywords(c["text"])
                )))
                for term in overlap
            )
            score = idf / math.sqrt(len(query_terms) * len(chunk_terms))
            scored.append(
                Evidence(
                    text=chunk["text"],
                    source=chunk["source"],
                    score=round(min(score, 1.0), 4),
                    chunk_id=chunk["id"].split("-")[-1],
                    backend="keyword",
                )
            )
        scored.sort(key=lambda e: e.score, reverse=True)
        return scored[:top_k]

    def _vector_search(self, query: str, top_k: int) -> list[Evidence]:
        embedder = self._get_embedder()
        if not embedder:
            return []
        query_vector = embedder.encode([query], show_progress_bar=False)[0]

        if self._collection is not None:
            try:
                result = self._collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=min(top_k, len(self._chunks)),
                )
                out: list[Evidence] = []
                docs = result.get("documents", [[]])[0]
                metas = result.get("metadatas", [[]])[0]
                distances = result.get("distances", [[]])[0]
                for doc, meta, dist in zip(docs, metas, distances):
                    # Chroma returns squared L2 by default → convert to a
                    # bounded similarity so thresholds stay meaningful.
                    score = 1.0 / (1.0 + float(dist))
                    out.append(
                        Evidence(
                            text=doc,
                            source=meta.get("source", "unknown"),
                            score=round(score, 4),
                            backend="chromadb",
                        )
                    )
                return out
            except Exception as exc:  # noqa: BLE001
                log.warning("Chroma query failed (%s); falling back", exc)

        # numpy cosine
        import numpy as np

        matrix = np.array(self._chunk_vectors, dtype=float)
        q = np.array(query_vector, dtype=float)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
        norms[norms == 0] = 1e-9
        sims = (matrix @ q) / norms
        order = np.argsort(sims)[::-1][:top_k]
        return [
            Evidence(
                text=self._chunks[i]["text"],
                source=self._chunks[i]["source"],
                score=round(float(sims[i]), 4),
                chunk_id=self._chunks[i]["id"].split("-")[-1],
                backend="numpy+embeddings",
            )
            for i in order
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = config.RETRIEVAL_TOP_K,
        min_score: float = config.RETRIEVAL_MIN_SCORE,
    ) -> list[Evidence]:
        """Return evidence above `min_score`, best first. Never raises."""
        if not query or not query.strip():
            return []
        if self.backend == "not_built":
            self.build_index()
        if not self._chunks:
            return []

        results: list[Evidence] = []
        if self.backend in {"chromadb", "numpy+embeddings"}:
            results = self._vector_search(query, top_k)
            if not results:
                results = self._keyword_search(query, top_k)
        else:
            results = self._keyword_search(query, top_k)

        return [e for e in results if e.score >= min_score]

    def format_for_prompt(self, evidence: Iterable[Evidence]) -> str:
        blocks = []
        for i, item in enumerate(evidence, 1):
            blocks.append(f"[{i}] ({item.source}) {item.text}")
        return "\n\n".join(blocks)

    # ── convenience ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        sources = sorted({c["source"] for c in self._chunks})
        return {
            "backend": self.backend,
            "chunks": len(self._chunks),
            "documents": sources,
        }
