"""
HTTP-level tests for the API.

These run without Ollama: the chat route is exercised with the pipeline's
`handle_turn` replaced by a stub, so what is under test is the API contract —
validation, persistence, response shape and error codes — not the model.
"""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.core.pipeline import AssistantTurn
from backend.db.database import Base, get_session


@pytest.fixture
def client(monkeypatch):
    # One in-memory database shared by every connection in this test.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    def _override():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    from backend.api.app import create_app
    from backend.services import chat_service
    from backend.services import engine as engine_module

    app = create_app()
    app.dependency_overrides[get_session] = _override

    # Skip index building and model loading.
    monkeypatch.setattr(engine_module.engine, "ensure_index", lambda force=False: "stub")

    def fake_pipeline(_cid):
        class _Memory:
            images: dict = {}

        class _Pipeline:
            memory = _Memory()

            def handle_turn(self, user_message, image_bytes=None, image_filename="image"):
                turn = AssistantTurn()
                turn.answer = f"echo: {user_message}"
                turn.route = "general_question"
                turn.trace = ["routed", "answered"]
                return turn

        return _Pipeline()

    monkeypatch.setattr(engine_module.engine, "pipeline", fake_pipeline)
    monkeypatch.setattr(chat_service.config, "PERSIST_CHATS", True)

    with TestClient(app) as test_client:
        yield test_client

    engine.dispose()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_creates_conversation_and_persists_turn(client):
    response = client.post("/api/chat", json={"message": "What is a layer?"})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "echo: What is a layer?"
    assert body["conversation_id"]
    assert body["conversation_title"] == "What is a layer?"

    detail = client.get(f"/api/conversations/{body['conversation_id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["message_count"] == 2


def test_follow_up_appends_to_same_conversation(client):
    first = client.post("/api/chat", json={"message": "First"}).json()
    cid = first["conversation_id"]

    second = client.post(
        "/api/chat", json={"message": "Second", "conversation_id": cid}
    ).json()
    assert second["conversation_id"] == cid

    detail = client.get(f"/api/conversations/{cid}").json()
    # Two turns, and each message stored exactly once.
    assert len(detail["messages"]) == 4
    assert detail["message_count"] == 4


def test_empty_message_is_rejected(client):
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422


def test_image_only_turn_is_accepted(client):
    data = base64.b64encode(b"\x89PNG\r\n\x1a\n fake").decode()
    response = client.post(
        "/api/chat", json={"message": "", "image_b64": data, "image_name": "x.png"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["used_image"] is True
    # The stored text is labelled as an attachment, never the internal prompt.
    assert body["display_text"] == "[Attached x.png]"
    # ...and it must not title the conversation.
    assert body["conversation_title"] == "New conversation"


def test_invalid_base64_is_rejected(client):
    response = client.post("/api/chat", json={"message": "hi", "image_b64": "!!!"})
    assert response.status_code == 422


def test_conversation_crud(client):
    cid = client.post("/api/chat", json={"message": "Hello there"}).json()[
        "conversation_id"
    ]

    listing = client.get("/api/conversations").json()
    assert listing["total"] == 1
    assert listing["items"][0]["conversation_id"] == cid

    renamed = client.patch(f"/api/conversations/{cid}", json={"title": "Renamed"})
    assert renamed.json()["title"] == "Renamed"

    assert client.delete(f"/api/conversations/{cid}").status_code == 204
    assert client.get(f"/api/conversations/{cid}").status_code == 404


def test_unknown_conversation_is_404(client):
    assert client.get("/api/conversations/nope").status_code == 404
    assert client.delete("/api/conversations/nope").status_code == 404


def test_developer_mode_returns_trace(client):
    body = client.post(
        "/api/chat", json={"message": "Hi", "developer_mode": True}
    ).json()
    assert body["dev"]["route"] == "general_question"
    assert body["dev"]["trace"] == ["routed", "answered"]

    plain = client.post("/api/chat", json={"message": "Hi"}).json()
    assert "dev" not in plain
