"""
Tests for conversation persistence.

These cover the same guarantees the old JSON store had — auto-titles, the
one-image-per-conversation rule, reliable reloads — plus the ones the database
adds: messages are deleted with their conversation, and the sidebar summary is
kept in sync inside the same transaction.

Each test gets its own in-memory SQLite database, so nothing touches
``data/assistant.db``.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base
from backend.db.models import Message
from backend.db.repositories.conversations import (
    ConversationRepository,
    preview_of,
    title_from,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture
def repo(session) -> ConversationRepository:
    return ConversationRepository(session)


# ─────────────────────────────────────────────────────────────────────────────
# Titles
# ─────────────────────────────────────────────────────────────────────────────


class TestTitle:
    def test_title_from_question(self):
        assert title_from("How many people are in this image?") == (
            "How many people are in this image?"
        )

    def test_title_strips_stacked_pleasantries(self):
        # Pleasantries stack, so this needs more than one pass to strip.
        assert title_from("Hey, could you please tell me what a layer is?") == (
            "What a layer is?"
        )

    def test_title_does_not_eat_meaningful_opening(self):
        assert title_from("Explain overfitting") == "Explain overfitting"

    def test_empty_input_falls_back(self):
        assert title_from("") == "New conversation"

    def test_preview_truncates(self):
        text = "a" * 200
        preview = preview_of(text)
        assert len(preview) <= 71 and preview.endswith("…")


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────


class TestCrud:
    def test_create_and_get_round_trip(self, repo):
        created = repo.create()
        assert created.conversation_id
        assert repo.get(created.conversation_id).id == created.id

    def test_get_missing_returns_none(self, repo):
        assert repo.get("does-not-exist") is None

    def test_list_is_newest_first(self, repo):
        first = repo.create()
        repo.add_message(first, "user", "first")
        second = repo.create()
        repo.add_message(second, "user", "second")

        ids = [c.conversation_id for c in repo.list()]
        assert ids[0] == second.conversation_id

    def test_multiple_turns_all_persisted(self, repo):
        conversation = repo.create()
        for i in range(3):
            repo.add_message(conversation, "user", f"question {i}")
            repo.add_message(conversation, "assistant", f"answer {i}")

        loaded = repo.get(conversation.conversation_id)
        assert len(loaded.messages) == 6
        assert loaded.message_count == 6

    def test_rename(self, repo):
        conversation = repo.create()
        repo.rename(conversation, "Renamed")
        assert repo.get(conversation.conversation_id).title == "Renamed"

    def test_delete_cascades_to_messages(self, repo, session):
        conversation = repo.create()
        repo.add_message(conversation, "user", "hello")
        repo.add_message(conversation, "assistant", "hi")
        session.commit()

        repo.delete(conversation)
        session.commit()

        remaining = session.scalars(select(Message)).all()
        assert remaining == []

    def test_clear_removes_everything(self, repo):
        for _ in range(3):
            repo.add_message(repo.create(), "user", "x")
        assert repo.clear() == 3
        assert repo.list() == []


# ─────────────────────────────────────────────────────────────────────────────
# Auto-titles and summaries
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoTitle:
    def test_title_derived_from_first_user_message(self, repo):
        conversation = repo.create()
        repo.add_message(conversation, "user", "How does retrieval work?")
        assert repo.get(conversation.conversation_id).title == "How does retrieval work?"

    def test_synthetic_attachment_turn_does_not_become_the_title(self, repo):
        """A chat that starts with a bare upload stays untitled until the user types."""
        conversation = repo.create()
        repo.add_message(
            conversation, "user", "[Attached chart.png]", synthetic_user=True
        )
        assert repo.get(conversation.conversation_id).title == "New conversation"

        repo.add_message(conversation, "user", "What does this chart show?")
        assert (
            repo.get(conversation.conversation_id).title == "What does this chart show?"
        )

    def test_later_messages_do_not_overwrite_title(self, repo):
        conversation = repo.create()
        repo.add_message(conversation, "user", "First question")
        repo.add_message(conversation, "user", "Second question")
        assert repo.get(conversation.conversation_id).title == "First question"

    def test_summary_exposes_only_sidebar_fields(self, repo):
        conversation = repo.create()
        repo.add_message(conversation, "user", "Hello there")
        summary = repo.summary(conversation)
        assert set(summary) == {
            "conversation_id",
            "title",
            "updated_at",
            "message_count",
            "preview",
            "has_image",
        }

    def test_preview_uses_first_user_message(self, repo):
        conversation = repo.create()
        repo.add_message(conversation, "user", "The very first question")
        repo.add_message(conversation, "user", "Another")
        assert repo.summary(conversation)["preview"] == "The very first question"


# ─────────────────────────────────────────────────────────────────────────────
# The one-image-per-conversation rule
# ─────────────────────────────────────────────────────────────────────────────


class TestConversationImage:
    def test_attach_and_retrieve_bytes(self, repo):
        conversation = repo.create()
        repo.attach_image(conversation, "chart.png", b"\x89PNG data", "image/png")
        loaded = repo.get(conversation.conversation_id)
        assert loaded.image_bytes == b"\x89PNG data"
        assert loaded.image_name == "chart.png"
        assert loaded.has_image is True

    def test_image_stored_once_not_per_message(self, repo, session):
        """A long Q&A over one image must not grow per turn."""
        conversation = repo.create()
        repo.add_message(conversation, "user", "[Attached chart.png]", image_name="chart.png")
        repo.attach_image(conversation, "chart.png", b"payload")

        for _ in range(5):
            repo.add_message(conversation, "user", "follow up")
            repo.add_message(conversation, "assistant", "answer")

        assert len(conversation.messages) == 11
        # Exactly one message carries the label; the blob lives on the header row.
        assert sum(1 for m in conversation.messages if m.image_name) == 1

    def test_detach_clears_image(self, repo):
        conversation = repo.create()
        repo.attach_image(conversation, "a.png", b"data")
        repo.detach_image(conversation)
        loaded = repo.get(conversation.conversation_id)
        assert loaded.image_bytes is None
        assert loaded.has_image is False

    def test_dev_payload_round_trips_as_json(self, repo):
        conversation = repo.create()
        dev = {"route": "image_question", "trace": ["step 1"], "validation": {"status": "pass"}}
        repo.add_message(conversation, "assistant", "answer", dev=dev)
        loaded = repo.get(conversation.conversation_id)
        serialised = repo.message_to_dict(loaded.messages[0])
        assert serialised["dev"]["route"] == "image_question"
        assert serialised["dev"]["trace"] == ["step 1"]
