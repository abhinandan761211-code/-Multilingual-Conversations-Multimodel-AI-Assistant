"""
models.py
=========
Relational schema for conversations.

Why tables instead of JSON files
--------------------------------
The previous version wrote one JSON file per conversation plus an ``index.json``
for the sidebar. That works for one user and one process, but it cannot be
queried, paginated, filtered, or safely written by two clients at once. The
tables below give us:

* atomic, transactional saves (a half-written chat is impossible),
* a cheap sidebar query (index columns on ``conversations``, no file reads),
* ``ON DELETE CASCADE`` so deleting a conversation removes its messages,
* one image per conversation stored as a BLOB, not re-embedded per message.

Schema
------
conversations  1 ── N  messages
The image belongs to the conversation because the UI's rule is "attach once,
keep asking" — not "re-embed on every turn".
"""
from __future__ import annotations

import time
import uuid

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> float:
    return time.time()


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Short, URL-safe identifier exposed to clients. Kept separate from the
    # integer primary key so the id in a URL can never be used to enumerate
    # rows by counting.
    conversation_id: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, default=_uuid
    )
    title: Mapped[str] = mapped_column(
        String(200), default="New conversation", nullable=False
    )

    created_at: Mapped[float] = mapped_column(Float, default=_now, nullable=False)
    updated_at: Mapped[float] = mapped_column(
        Float, default=_now, onupdate=_now, nullable=False, index=True
    )

    # ── pinned attachment (one per conversation) ─────────────────────────────
    image_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_mime: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # Anything the UI needs to re-derive a sidebar summary cheaply.
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    preview: Mapped[str] = mapped_column(String(160), default="", nullable=False)
    has_image: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.id",
        lazy="selectin",
    )

    __table_args__ = (Index("ix_conversations_updated_at_desc", "updated_at"),)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # "user" | "assistant"
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Only set on the message that introduced the image (the UI shows it once).
    image_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Developer-mode telemetry for this turn, serialised as JSON text. Storing
    # it as JSON keeps the schema stable while the trace format evolves.
    dev_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    created_at: Mapped[float] = mapped_column(Float, default=_now, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
