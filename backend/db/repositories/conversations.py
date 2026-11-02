"""
conversations.py
================
Every conversation/message query lives here.

Callers (services, routes, scripts) never write SQL or touch the ORM directly —
that keeps the persistence strategy swappable and makes the API behaviour easy
to reason about from one file.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.db.models import Conversation, Message

DEFAULT_TITLE = "New conversation"
# First-turn messages the user did not type (they only attached a file). They are
# stored so the transcript is complete, but they never name the conversation.
SYNTHETIC_PREFIX = "[Attached "


def title_from(text: str, fallback: str = DEFAULT_TITLE) -> str:
    """Derive a ChatGPT-style short title from the first user message."""
    text = (text or "").strip().replace("\n", " ")
    if not text:
        return fallback

    # Strip leading pleasantries so the title carries signal. Applied in a loop
    # because they stack: "Hi, can you explain overfitting?" has two.
    pleasantry = re.compile(
        r"^(?:hi|hey|hello|yo|please|thanks|thank you|"
        r"can you|could you|would you|will you|"
        r"i want to|i need to|i'd like to|i would like to|"
        r"help me|tell me|show me|give me|"
        r"what about|how about)"
        r"\b[,:]?\s*",
        re.IGNORECASE,
    )
    previous = None
    while previous != text:
        previous = text
        text = pleasantry.sub("", text).strip()

    text = text[:52].strip()
    if not text:
        return fallback
    text = text[0].upper() + text[1:]
    return text + ("…" if len(text) >= 52 else "")


def preview_of(text: str, width: int = 70) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:width] + ("…" if len(text) > width else "")


class ConversationRepository:
    """Data access for conversations and their messages."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ── queries ─────────────────────────────────────────────────────────────

    def list(self, limit: int = 100, offset: int = 0) -> list[Conversation]:
        stmt = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.scalars(stmt).unique())

    def count(self) -> int:
        return int(self.session.scalar(select(func.count(Conversation.id))) or 0)

    def get(self, conversation_id: str) -> Conversation | None:
        stmt = select(Conversation).where(
            Conversation.conversation_id == conversation_id
        )
        return self.session.scalars(stmt).first()

    # ── writes ──────────────────────────────────────────────────────────────

    def create(self, title: str = DEFAULT_TITLE) -> Conversation:
        conversation = Conversation(title=title or DEFAULT_TITLE)
        self.session.add(conversation)
        self.session.flush()
        return conversation

    def ensure(self, conversation_id: str | None) -> Conversation:
        """Return the conversation for this id, creating it when the id is new.

        A client may hold an id that was deleted from another tab; in that case
        a fresh conversation is created rather than raising, which keeps the
        chat usable instead of erroring on save.
        """
        if conversation_id:
            existing = self.get(conversation_id)
            if existing is not None:
                return existing
        return self.create()

    def add_message(
        self,
        conversation: Conversation,
        role: str,
        content: str,
        *,
        image_name: str | None = None,
        dev: dict[str, Any] | None = None,
        synthetic_user: bool = False,
    ) -> Message:
        """Append a message and refresh the denormalised conversation summary."""
        # Assign through the relationship, not through `conversation_id` too:
        # doing both makes SQLAlchemy treat the object as a second, distinct
        # pending instance and insert it twice.
        message = Message(
            role=role,
            content=content or "",
            image_name=image_name,
            dev_json=json.dumps(dev or {}, ensure_ascii=False, default=str),
        )
        conversation.messages.append(message)
        conversation.updated_at = time.time()

        # Sidebar summary columns.
        if role == "user" and not conversation.preview:
            conversation.preview = preview_of(content)
        conversation.message_count = len(conversation.messages)

        # Only a message the user actually wrote can name the conversation.
        if (
            role == "user"
            and not synthetic_user
            and content.strip()
            and not content.strip().startswith(SYNTHETIC_PREFIX)
            and conversation.title in ("", DEFAULT_TITLE)
        ):
            conversation.title = title_from(content)

        self.session.flush()
        return message

    def attach_image(
        self, conversation: Conversation, name: str, data: bytes, mime: str = "image/*"
    ) -> None:
        conversation.image_name = name
        conversation.image_bytes = data
        conversation.image_mime = mime
        conversation.has_image = bool(data)
        conversation.updated_at = time.time()
        self.session.flush()

    def detach_image(self, conversation: Conversation) -> None:
        conversation.image_name = None
        conversation.image_bytes = None
        conversation.image_mime = None
        conversation.has_image = False
        conversation.updated_at = time.time()
        self.session.flush()

    def rename(self, conversation: Conversation, title: str) -> Conversation:
        conversation.title = title.strip() or conversation.title
        conversation.updated_at = time.time()
        self.session.flush()
        return conversation

    def delete(self, conversation: Conversation) -> None:
        self.session.delete(conversation)
        self.session.flush()

    def clear(self) -> int:
        conversations = self.list(limit=10_000)
        for conversation in conversations:
            self.session.delete(conversation)
        self.session.flush()
        return len(conversations)

    # ── serialisation ───────────────────────────────────────────────────────

    @staticmethod
    def message_to_dict(message: Message) -> dict[str, Any]:
        try:
            dev = json.loads(message.dev_json or "{}")
        except json.JSONDecodeError:
            dev = {}
        return {
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "image_name": message.image_name,
            "dev": dev,
            "created_at": message.created_at,
        }

    def to_dict(self, conversation: Conversation, include_image: bool = False) -> dict:
        payload = {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "message_count": conversation.message_count,
            "preview": conversation.preview,
            "has_image": conversation.has_image,
            "image_name": conversation.image_name,
            "messages": [self.message_to_dict(m) for m in conversation.messages],
        }
        if include_image and conversation.image_bytes:
            import base64

            payload["image_b64"] = base64.b64encode(conversation.image_bytes).decode(
                "ascii"
            )
        return payload

    def summary(self, conversation: Conversation) -> dict:
        """Exactly what a sidebar needs — no messages, no blobs."""
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "updated_at": conversation.updated_at,
            "message_count": conversation.message_count,
            "preview": conversation.preview or "No messages yet",
            "has_image": conversation.has_image,
        }
