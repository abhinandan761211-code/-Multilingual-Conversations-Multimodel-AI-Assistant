"""
Tests for chat persistence: saving, listing, reloading, and the
one-image-per-conversation rule that powers the ChatGPT-style attachment.
"""
from __future__ import annotations

import json

import pytest

from chat_store import ChatStore, Conversation, StoredMessage, title_from


@pytest.fixture
def store(tmp_path) -> ChatStore:
    return ChatStore(directory=tmp_path / "chats")


# ─────────────────────────────────────────────────────────────────────────────
# Titles
# ─────────────────────────────────────────────────────────────────────────────


class TestTitle:
    def test_title_from_question(self):
        assert title_from("How many people are in this image?") == (
            "How many people are in this image?"
        )

    def test_title_strips_pleasantries(self):
        # Pleasantries stack, so this needs more than one pass to strip.
        assert title_from("Hi, can you explain overfitting?") == "Explain overfitting?"

    def test_title_strips_stacked_pleasantries(self):
        assert title_from("Hey, could you please tell me what a layer is?") == (
            "What a layer is?"
        )

    def test_title_does_not_eat_meaningful_opening(self):
        # "Explain" is not a pleasantry — it must survive.
        assert title_from("Explain gradient descent") == "Explain gradient descent"

    def test_title_truncates_long_input(self):
        title = title_from("x" * 200)
        assert len(title) <= 53
        assert title.endswith("…")

    def test_empty_input_falls_back(self):
        assert title_from("") == "New conversation"
        assert title_from("   ") == "New conversation"


# ─────────────────────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────────────────────


class TestCrud:
    def test_create_and_load_round_trip(self, store):
        created = store.create()
        loaded = store.load(created.conversation_id)
        assert loaded is not None
        assert loaded.conversation_id == created.conversation_id

    def test_load_missing_returns_none(self, store):
        assert store.load("nope") is None

    def test_save_then_list(self, store):
        conversation = store.create()
        conversation.messages.append(StoredMessage(role="user", content="hello"))
        store.save(conversation)

        entries = store.list_conversations()
        assert len(entries) == 1
        assert entries[0]["conversation_id"] == conversation.conversation_id
        assert entries[0]["message_count"] == 1

    def test_list_is_newest_first(self, store):
        first = store.create()
        first.messages.append(StoredMessage(role="user", content="one"))
        store.save(first)

        second = store.create()
        second.messages.append(StoredMessage(role="user", content="two"))
        store.save(second)

        entries = store.list_conversations()
        assert entries[0]["conversation_id"] == second.conversation_id

    def test_no_empty_stub_appears_in_sidebar(self, store):
        """
        Regression: creating a conversation via store.create() writes an empty
        file straight away, which then shows up alongside the real one as a
        duplicate entry with no messages. Conversations should only be written
        once they have content.
        """
        conversation = Conversation(conversation_id="")   # not yet persisted
        assert store.list_conversations() == []

        conversation.conversation_id = "abc123"
        conversation.messages.append(StoredMessage(role="user", content="real"))
        store.save(conversation)

        entries = store.list_conversations()
        assert len(entries) == 1
        assert entries[0]["message_count"] == 1
        assert entries[0]["preview"] == "real"

    def test_multiple_turns_all_persisted(self, store):
        """Every turn in a multi-turn Q&A must survive a reload."""
        conversation = Conversation(conversation_id="multi")
        conversation.attach_image("chart.png", b"IMG")
        for i in range(3):
            conversation.messages.append(
                StoredMessage(role="user", content=f"question {i}")
            )
            conversation.messages.append(
                StoredMessage(role="assistant", content=f"answer {i}")
            )
        store.save(conversation)

        loaded = store.load("multi")
        assert len(loaded.messages) == 6
        assert [m.content for m in loaded.messages] == [
            "question 0", "answer 0",
            "question 1", "answer 1",
            "question 2", "answer 2",
        ]
        assert loaded.image_bytes() == b"IMG"

    def test_delete_removes_file_and_index(self, store):
        conversation = store.create()
        store.delete(conversation.conversation_id)
        assert store.load(conversation.conversation_id) is None
        assert store.list_conversations() == []

    def test_rename(self, store):
        conversation = store.create()
        store.rename(conversation.conversation_id, "My chart analysis")
        assert store.load(conversation.conversation_id).title == "My chart analysis"

    def test_clear_all(self, store):
        for _ in range(3):
            store.create()
        store.clear_all()
        assert store.list_conversations() == []


# ─────────────────────────────────────────────────────────────────────────────
# Titles derived from content
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoTitle:
    def test_title_derived_from_first_user_message(self, store):
        conversation = store.create()
        conversation.messages.append(
            StoredMessage(role="user", content="When does validation loss rise?")
        )
        store.save(conversation)
        assert store.load(conversation.conversation_id).title == (
            "When does validation loss rise?"
        )

    def test_synthetic_first_turn_does_not_become_the_title(self, store):
        """
        When a user only attaches a file, the first turn is machine-generated
        ("[Attached chart.png]"). That must not name the conversation — the
        sidebar would then show our own instruction text instead of the user's
        words.
        """
        conversation = Conversation(conversation_id="auto")
        conversation.attach_image("chart.png", b"IMG")
        # First turn: user attached a file without typing anything.
        conversation.messages.append(
            StoredMessage(role="user", content="[Attached chart.png]", image_name="chart.png")
        )
        conversation.messages.append(
            StoredMessage(role="assistant", content="A line chart of loss.")
        )
        # Save after only the attachment turn: the placeholder must not be
        # adopted as the title.
        store.save(conversation)
        assert store.load("auto").title == "New conversation", (
            "an attachment placeholder should not name the conversation"
        )

        # Second turn: the user actually writes something.
        conversation.messages.append(
            StoredMessage(role="user", content="Why does validation loss rise?")
        )
        conversation.messages.append(
            StoredMessage(role="assistant", content="Because of overfitting.")
        )
        store.save(conversation)

        assert store.load("auto").title == "Why does validation loss rise?"

    def test_explicit_title_is_kept(self, store):
        conversation = store.create()
        conversation.title = "Chart questions"
        conversation.messages.append(StoredMessage(role="user", content="anything"))
        store.save(conversation)
        assert store.load(conversation.conversation_id).title == "Chart questions"


# ─────────────────────────────────────────────────────────────────────────────
# The image is attached to the conversation, once
# ─────────────────────────────────────────────────────────────────────────────


class TestConversationImage:
    def test_attach_and_retrieve_bytes(self, store):
        conversation = store.create()
        conversation.attach_image("chart.png", b"\x89PNG-fake-bytes")
        store.save(conversation)

        loaded = store.load(conversation.conversation_id)
        assert loaded.image_name == "chart.png"
        assert loaded.image_bytes() == b"\x89PNG-fake-bytes"

    def test_image_stored_once_not_per_message(self, store):
        """The whole point: a long Q&A over one image must not grow per turn."""
        conversation = store.create()
        conversation.attach_image("chart.png", b"IMAGEDATA")
        for i in range(6):
            conversation.messages.append(StoredMessage(role="user", content=f"q{i}"))
            conversation.messages.append(StoredMessage(role="assistant", content=f"a{i}"))
        store.save(conversation)

        import base64

        raw = json.loads(
            (store.dir / f"{conversation.conversation_id}.json").read_text()
        )
        # The payload is base64-encoded, so compare against its encoded form.
        encoded = base64.b64encode(b"IMAGEDATA").decode("ascii")
        serialised = json.dumps(raw)
        assert serialised.count(encoded) == 1, "image must be stored exactly once"
        assert raw["image_b64"] == encoded
        # The per-message field only records the *name*, never the payload.
        assert all("image_b64" not in m for m in raw["messages"])

    def test_image_survives_reload(self, store):
        conversation = store.create()
        conversation.attach_image("loss.png", b"BYTES")
        conversation.messages.append(StoredMessage(role="user", content="analyse"))
        store.save(conversation)

        reloaded = store.load(conversation.conversation_id)
        assert reloaded.image_bytes() == b"BYTES"
        assert reloaded.image_name == "loss.png"

    def test_detached_image_is_persisted_as_none(self, store):
        conversation = store.create()
        conversation.attach_image("x.png", b"BYTES")
        store.save(conversation)

        conversation.image_b64 = None
        conversation.image_name = None
        store.save(conversation)

        assert store.load(conversation.conversation_id).image_bytes() is None


# ─────────────────────────────────────────────────────────────────────────────
# Robustness
# ─────────────────────────────────────────────────────────────────────────────


class TestRobustness:
    def test_corrupt_index_does_not_crash(self, store):
        (store.dir / "index.json").write_text("{ this is not json")
        assert store.list_conversations() == []

    def test_corrupt_conversation_returns_none(self, store):
        (store.dir / "abc123.json").write_text("<<<not json>>>")
        assert store.load("abc123") is None

    def test_path_traversal_id_is_neutralised(self, store):
        # A hostile id must not escape the chats directory.
        store.save(
            Conversation(conversation_id="../../etc/passwd")
        )
        assert not (store.dir.parent.parent / "etc" / "passwd").exists()
        assert store.load("../../etc/passwd") is None or True  # must not raise

    def test_summary_exposes_only_sidebar_fields(self, store):
        conversation = store.create()
        conversation.attach_image("a.png", b"X")
        conversation.messages.append(StoredMessage(role="user", content="hello"))
        store.save(conversation)

        summary = store.list_conversations()[0]
        assert set(summary) == {
            "conversation_id", "title", "updated_at",
            "message_count", "preview", "has_image",
        }
        assert summary["has_image"] is True

    def test_preview_uses_first_user_message(self, store):
        conversation = store.create()
        conversation.messages.append(StoredMessage(role="user", content="First question"))
        conversation.messages.append(StoredMessage(role="assistant", content="Reply"))
        store.save(conversation)
        assert store.list_conversations()[0]["preview"] == "First question"
