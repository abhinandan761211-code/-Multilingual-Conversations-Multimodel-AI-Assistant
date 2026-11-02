# ── Multimodal AI Assistant — developer entry points ─────────────────────────
# Everything is run from the repository root.

PY ?= python3
PORT_API ?= 8000
PORT_UI ?= 8501

.DEFAULT_GOAL := help
.PHONY: help install install-dev api ui dev test lint format reindex db-shell db-reset clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install backend + frontend dependencies
	$(PY) -m pip install -r backend/requirements.txt -r frontend/requirements.txt

install-dev: ## Install everything, including test/lint tooling
	$(PY) -m pip install -r requirements-dev.txt

api: ## Run the FastAPI backend (docs at /docs)
	$(PY) -m uvicorn backend.api.app:app --reload --host 0.0.0.0 --port $(PORT_API)

ui: ## Run the Streamlit frontend (needs the API running)
	$(PY) -m streamlit run frontend/app.py --server.port $(PORT_UI)

dev: ## Run the API and the UI together
	@echo "API  -> http://localhost:$(PORT_API)/docs"
	@echo "UI   -> http://localhost:$(PORT_UI)"
	$(PY) -m uvicorn backend.api.app:app --host 0.0.0.0 --port $(PORT_API) & \
	$(PY) -m streamlit run frontend/app.py --server.port $(PORT_UI)

test: ## Run the offline test suite
	$(PY) -m pytest tests/ -q

lint: ## Check style and obvious bugs
	$(PY) -m ruff check backend frontend tests

format: ## Format the code
	$(PY) -m black backend frontend tests

reindex: ## Rebuild the knowledge-base vector index
	curl -s -X POST http://127.0.0.1:$(PORT_API)/api/documents/reindex

db-shell: ## Open a SQLite shell on the app database
	sqlite3 data/assistant.db

db-reset: ## Delete the database (destroys chat history)
	rm -f data/assistant.db data/assistant.db-wal data/assistant.db-shm

demodata: ## Generate the demo images used by scripts/demo_scenarios.py
	$(PY) scripts/make_demo_images.py

demo: ## Run the scripted evaluation scenarios (needs Ollama running)
	$(PY) scripts/demo_scenarios.py

clean: ## Remove caches and generated artefacts
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + ; \
	rm -rf .pytest_cache .ruff_cache .mypy_cache
