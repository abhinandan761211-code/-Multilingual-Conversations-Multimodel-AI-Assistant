# Multimodal AI Assistant

A local, image-aware assistant for asking questions about images and reference
documents. A reasoning pipeline decides between image analysis, OCR, document
retrieval, conversation memory and a clarification question, then validates the
answer before it is shown.

Everything runs locally: model inference through [Ollama](https://ollama.com/),
storage in SQLite, and no data leaves the machine.

## Architecture

The project is split into three independent layers, each with its own
dependencies, entry point and test coverage.

```text
┌─────────────────────┐      HTTP/JSON      ┌──────────────────────┐
│  frontend/          │ ──────────────────► │  backend/api         │
│  Streamlit UI       │                     │  FastAPI             │
│  (presentation)     │ ◄────────────────── │  (HTTP contract)     │
└─────────────────────┘                     └──────────┬───────────┘
                                                       │
                                            ┌──────────▼───────────┐
                                            │  backend/services    │
                                            │  use cases, engine   │
                                            └──────┬────────┬──────┘
                                                   │        │
                              ┌────────────────────▼──┐  ┌──▼─────────────────┐
                              │  backend/core         │  │  backend/db        │
                              │  routing · vision     │  │  SQLAlchemy models │
                              │  OCR · retrieval      │  │  repositories      │
                              │  validation · i18n    │  │  SQLite/Postgres   │
                              └───────────────────────┘  └────────────────────┘
```

| Layer | Folder | Responsibility | Depends on |
|---|---|---|---|
| **Frontend** | `frontend/` | Streamlit pages, rendering, session state | the API only |
| **API** | `backend/api/` | Routes, request/response schemas, validation | services |
| **Services** | `backend/services/` | Use cases; owns model lifetimes | core, db |
| **Core engine** | `backend/core/` | Routing, inference, OCR, retrieval, validation | libraries only |
| **Database** | `backend/db/` | Schema, sessions, repositories | SQLAlchemy |
| **Data** | `data/` | SQLite database, documents, vector index | — |

The front-end never imports engine code and the engine never imports FastAPI —
so either side can be replaced, and the API can serve other clients.

## Features

- Ask about an image, extract its text, inspect a chart, or ask text-only.
- Attach one image per conversation and keep asking follow-up questions.
- Route ambiguous image requests to clarification instead of guessing.
- Measure image quality and surface uncertainty for blurry or small inputs.
- OCR through EasyOCR or Tesseract, both optional.
- Retrieve passages from the local Markdown corpus with a keyword fallback.
- Validate relevance, grounding, numeric claims, uncertainty, format and
  possible fabrication before showing the answer.
- Persist conversations in a real database, with a developer view showing the
  routing trace, diagnostics, evidence and validation checks.
- **Multilingual**: automatic English / Hindi / Odia / Bengali detection
  (including Roman-script Hinglish), context that survives a language switch,
  and answers in the language you used. See [docs/MULTILINGUAL.md](docs/MULTILINGUAL.md).

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) running locally
- The models pulled into Ollama:

```bash
ollama pull qwen2.5vl:7b     # vision model
ollama pull qwen3:4b         # multilingual text model (en / hi / or / bn)
ollama serve
```

Tesseract/EasyOCR are optional OCR backends; sentence-transformers and ChromaDB
are optional (the app falls back to keyword retrieval and lexical checks).

## Quick start

```bash
git clone https://github.com/abhinandan761211-code/Multimodal-AI-Assistant.git
cd Multimodal-AI-Assistant

python -m venv .venv && source .venv/bin/activate
make install          # backend + frontend dependencies
```

Run the two applications in two terminals:

```bash
make api              # http://localhost:8000  (docs at /docs)
make ui               # http://localhost:8501
```

Or both at once:

```bash
make dev
```

The first request is slow: the embedding model is downloaded once and the
vision model loads into Ollama. Subsequent turns are fast.

### Using the app

1. Upload an image with **Add an image**, or just type a question.
2. Ask something specific: “When does validation loss start increasing?”,
   “Extract the text from this screenshot.”
3. Keep the image attached and ask follow-ups; **Remove image** detaches it.
4. Enable **Developer mode** in the sidebar to see how an answer was produced.

## API

Interactive documentation is served at <http://localhost:8000/docs>.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Ask a question (text and/or base64 image) |
| `GET` | `/api/conversations` | Sidebar list, newest first |
| `POST` | `/api/conversations` | Create an empty conversation |
| `GET` | `/api/conversations/{id}` | Full transcript |
| `PATCH` | `/api/conversations/{id}` | Rename |
| `DELETE` | `/api/conversations/{id}` | Delete (messages cascade) |
| `GET`/`DELETE` | `/api/conversations/{id}/image` | Fetch / detach the pinned image |
| `GET` | `/api/conversations/{id}/export` | JSON export for reports |
| `GET` | `/api/system/status` | Ollama, model, knowledge-base health |
| `GET` | `/api/documents` | Indexed reference files |
| `POST` | `/api/documents/reindex` | Rebuild the vector index |
| `GET` | `/api/health` | Liveness probe |

Example:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is overfitting and how do you prevent it?"}'
```

```json
{
  "answer": "Overfitting occurs when a model learns the training data too well…",
  "route": "general_question",
  "conversation_id": "91cad8965eb7",
  "sources": [
    {"citation": "machine_learning_basics.md", "score": 0.633}
  ],
  "validation": {"status": "qualified", "confidence": 0.94},
  "language": {"code": "en", "name": "English", "confidence": 0.88}
}
```

Send `"developer_mode": true` to also receive the routing plan, trace, OCR
output, evidence and validation checks under `dev`.

## Database

Conversations are stored with SQLAlchemy in `data/assistant.db` (SQLite by
default). The schema is two tables:

```text
conversations ──1:N── messages
  conversation_id (unique)      role, content
  title, preview                image_name
  message_count, has_image      dev_json (turn telemetry)
  image_bytes (one pinned image)
```

- Deleting a conversation deletes its messages (`ON DELETE CASCADE`).
- Sidebar queries read the `conversations` row only — no message loading.
- Point at Postgres without changing code: `DATABASE_URL=postgresql+psycopg://…`.

```bash
make db-shell     # sqlite3 data/assistant.db
make db-reset     # delete chat history
```

## Configuration

Settings are environment variables (see [.env.example](.env.example)). The full
list lives in [`backend/core/config.py`](backend/core/config.py).

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `VISION_MODEL` | `qwen2.5vl:7b` | Image-capable model |
| `TEXT_MODEL` | `qwen3:4b` | Multilingual text model |
| `DATABASE_URL` | `sqlite:///data/assistant.db` | Database connection |
| `DATA_DIR` | `./data` | Runtime data root |
| `API_BASE_URL` | `http://127.0.0.1:8000/api` | Frontend → API |
| `OCR_ENABLED` / `OCR_BACKEND` | `1` / `auto` | OCR behaviour |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Retrieval embeddings |
| `RETRIEVAL_TOP_K` | `4` | Maximum retrieved passages |
| `GROUNDING_MIN_SCORE` | `0.30` | Minimum grounding score |
| `MAX_TURNS` | `12` | Turns kept in conversation memory |
| `LANGUAGE_BACKEND` | `auto` | `auto` (FastText/lingua) or `heuristic` |
| `LANGUAGE_CONFIDENCE_THRESHOLD` | `0.55` | Below this, ask instead of assuming |
| `DEFAULT_LANGUAGE` | `en` | Fallback for unclassifiable input |
| `ENFORCE_RESPONSE_LANGUAGE` | `1` | Corrective re-expression pass |

## Project structure

```text
.
├── backend/                    # All server-side code
│   ├── api/                    # FastAPI application
│   │   ├── app.py              # App factory + lifespan
│   │   ├── deps.py             # Shared dependencies
│   │   ├── routes/             # chat · system · widgets
│   │   └── schemas/            # Pydantic request/response models
│   ├── core/                   # Reasoning engine (framework-free)
│   │   ├── config.py           # Settings and paths
│   │   ├── pipeline.py         # Coordinates a turn end to end
│   │   ├── reasoning_router.py # Chooses the strategy
│   │   ├── vision_model.py     # Ollama client
│   │   ├── image_processor.py  # Validation, OCR, encoding
│   │   ├── evidence_retriever.py
│   │   ├── response_validator.py
│   │   ├── conversation_memory.py
│   │   ├── language_detector.py
│   │   ├── intent_manager.py
│   │   └── multilingual_model.py
│   ├── db/                     # Persistence
│   │   ├── database.py         # Engine, session, init
│   │   ├── models.py           # ORM schema
│   │   └── repositories/       # SQL lives here
│   ├── services/               # Use cases
│   │   ├── engine.py           # Shared models, per-chat pipelines
│   │   ├── chat_service.py     # The ask-a-question use case
│   │   └── serializers.py      # Dataclasses → JSON
│   └── requirements.txt
├── frontend/                   # Streamlit client
│   ├── app.py                  # Page + session bootstrap
│   ├── api_client.py           # The only HTTP caller
│   ├── theme.py                # CSS and presentation helpers
│   ├── components/             # sidebar · chat_view · composer · turn
│   └── requirements.txt
├── data/                       # Runtime state (not committed)
│   ├── documents/              # Markdown/text knowledge base
│   ├── assistant.db            # SQLite database
│   └── chroma_store/           # Vector index
├── tests/                      # Offline test suite
├── scripts/                    # Demo image and scenario utilities
├── docs/                       # ARCHITECTURE · MULTILINGUAL · DEMO_OUTPUT
├── Makefile                    # make api | ui | dev | test | reindex
└── pyproject.toml              # Tooling config (pytest, ruff, black)
```

## Development

```bash
make test                     # 180 offline tests, no Ollama needed
make lint                     # ruff
make format                   # black
```

The unit tests inject a fake vision model and synthetic images, so they run
without a GPU, a network or a model server. `tests/test_api.py` exercises the
HTTP contract with a stubbed pipeline.

End-to-end demo scenarios (requires Ollama and the vision model):

```bash
make demodata                 # generate demo images
make demo                     # run all scenarios
python scripts/demo_scenarios.py 2 4    # selected scenarios
```

## Knowledge base and data

Add `.md` or `.txt` files to [`data/documents/`](data/documents/) and run
`make reindex` (or restart the API) to index them.

`data/` holds user content — chat history, uploads and the vector index are
excluded from git. Do not commit personal conversations or private uploads.

For design details see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/MULTILINGUAL.md](docs/MULTILINGUAL.md).
