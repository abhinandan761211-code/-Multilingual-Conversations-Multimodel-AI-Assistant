"""
chat_store.py
=============
Persistent chat history, saved to disk as JSON so conversations survive a
browser refresh, a Streamlit restart, and a machine reboot.

Layout on disk::

    data/chats/
        index.json                  # lightweight list for the sidebar
        <conversation_id>.json      # full transcript of one conversation

Why a separate module
---------------------
Streamlit's `session_state` dies with the websocket. A history sidebar that
forgets everything on refresh is worse than no sidebar, so persistence has to
live outside the session. Keeping it here — rather than in app.py — means it can
be unit-tested without a browser.

Design notes
------------
* The **index** is stored separately from the transcripts so the sidebar can
  render instantly without reading every conversation file.
* Writes are atomic (temp file + `os.replace`) so a crash mid-save cannot leave
  a truncated JSON file behind.
* Uploaded images are stored per-conversation as a single base64 blob, not per
  message. That mirrors the UI rule: one image is attached to a conversation and
  reused, rather than re-embedded on every turn.
"""
from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import config

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

CHATS_DIR: Path = config.DATA_DIR / "chats"
CHATS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH: Path = CHATS_DIR / "index.json"


# ─────────────────────────────────────────────────────────────────────────────
# Small JSON helpers
# ─────────────────────────────────────────────────────────────────────────────


def _atomic_write(path: Path, payload: Any) -> None:
    """Write JSON atomically so an interrupted save can't corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        # Clean up the temp file if anything went wrong.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_json(path: Path, default: Any) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StoredMessage:
    role: str                       # "user" | "assistant"
    content: str = ""
    image_name: Optional[str] = None    # only set on the message that introduced it
    # Turn telemetry for Developer mode. Kept minimal: anything the UI needs to
    # re-render the developer panel after a reload.
    dev: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Conversation:
    conversation_id: str
    title: str = "New conversation"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    messages: list[StoredMessage] = field(default_factory=list)
    # One attached image per conversation, reused across turns.
    image_name: Optional[str] = None
    image_b64: Optional[str] = None

    # ── derived ──────────────────────────────────────────────────────────────

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def preview(self) -> str:
        for message in self.messages:
            if message.role == "user" and message.content.strip():
                text = message.content.strip().replace("\n", " ")
                return text[:70] + ("…" if len(text) > 70 else "")
        return "No messages yet"

    def summary(self) -> dict:
        """Exactly what the sidebar needs — nothing more."""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "preview": self.preview,
            "has_image": bool(self.image_b64),
        }

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [m.to_dict() for m in self.messages],
            "image_name": self.image_name,
            "image_b64": self.image_b64,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        return cls(
            conversation_id=data.get("conversation_id") or uuid.uuid4().hex[:12],
            title=data.get("title") or "New conversation",
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            messages=[
                StoredMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    image_name=m.get("image_name"),
                    dev=m.get("dev") or {},
                )
                for m in data.get("messages", [])
            ],
            image_name=data.get("image_name"),
            image_b64=data.get("image_b64"),
        )

    # ── image helpers ────────────────────────────────────────────────────────

    def attach_image(self, name: str, data: bytes) -> None:
        self.image_name = name
        self.image_b64 = base64.b64encode(data).decode("ascii")

    def image_bytes(self) -> Optional[bytes]:
        if not self.image_b64:
            return None
        try:
            return base64.b64decode(self.image_b64)
        except Exception:  # noqa: BLE001
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Store
# ─────────────────────────────────────────────────────────────────────────────


def title_from(text: str, fallback: str = "New conversation") -> str:
    """Derive a ChatGPT-style short title from the first user message."""
    text = (text or "").strip().replace("\n", " ")
    if not text:
        return fallback

    # Strip leading pleasantries so the title carries signal. Applied in a loop
    # because they stack: "Hi, can you explain overfitting?" has two of them, and
    # a single pass would leave "can you explain overfitting?".
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


class ChatStore:
    """File-backed collection of conversations."""

    def __init__(self, directory: Path = CHATS_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"

    # ── index ───────────────────────────────────────────────────────────────

    def _read_index(self) -> list[dict]:
        data = _read_json(self.index_path, default=[])
        return data if isinstance(data, list) else []

    def _write_index(self, entries: list[dict]) -> None:
        _atomic_write(self.index_path, entries)

    def _upsert_index(self, conversation: Conversation) -> None:
        entries = [e for e in self._read_index()
                   if e.get("conversation_id") != conversation.conversation_id]
        entries.append(conversation.summary())
        entries.sort(key=lambda e: e.get("updated_at", 0), reverse=True)
        self._write_index(entries)

    def _remove_from_index(self, conversation_id: str) -> None:
        entries = [e for e in self._read_index()
                   if e.get("conversation_id") != conversation_id]
        self._write_index(entries)

    # ── paths ───────────────────────────────────────────────────────────────

    def _path(self, conversation_id: str) -> Path:
        # Guard against a crafted id escaping the chats directory.
        safe = re.sub(r"[^A-Za-z0-9_-]", "", conversation_id) or "unknown"
        return self.dir / f"{safe}.json"

    # ── CRUD ────────────────────────────────────────────────────────────────

    def list_conversations(self) -> list[dict]:
        """Sidebar entries, newest first. Cheap — reads only the index."""
        entries = self._read_index()
        entries.sort(key=lambda e: e.get("updated_at", 0), reverse=True)
        return entries

    def create(self, title: str = "New conversation") -> Conversation:
        conversation = Conversation(
            conversation_id=uuid.uuid4().hex[:12], title=title
        )
        self.save(conversation)
        return conversation

    def load(self, conversation_id: str) -> Optional[Conversation]:
        data = _read_json(self._path(conversation_id), default=None)
        if not isinstance(data, dict):
            return None
        return Conversation.from_dict(data)

    # A first-turn message the user didn't type (they only attached a file).
    # It is stored so the transcript is complete, but it must not name the chat.
    _SYNTHETIC_PREFIX = "[Attached "

    @classmethod
    def _is_synthetic(cls, content: str) -> bool:
        return content.strip().startswith(cls._SYNTHETIC_PREFIX)

    def save(self, conversation: Conversation) -> None:
        conversation.updated_at = time.time()
        if not conversation.title or conversation.title == "New conversation":
            # Title from the first user message the user actually wrote,
            # ChatGPT-style. Attachment placeholders are skipped, so a chat that
            # starts with a bare upload stays untitled until the user says
            # something — then takes that wording.
            for message in conversation.messages:
                if (
                    message.role == "user"
                    and message.content.strip()
                    and not self._is_synthetic(message.content)
                ):
                    conversation.title = title_from(message.content)
                    break
        _atomic_write(self._path(conversation.conversation_id), conversation.to_dict())
        self._upsert_index(conversation)

    def delete(self, conversation_id: str) -> None:
        path = self._path(conversation_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self._remove_from_index(conversation_id)

    def rename(self, conversation_id: str, title: str) -> Optional[Conversation]:
        conversation = self.load(conversation_id)
        if conversation is None:
            return None
        conversation.title = title.strip() or conversation.title
        self.save(conversation)
        return conversation

    def clear_all(self) -> None:
        for entry in self.list_conversations():
            self.delete(entry["conversation_id"])
        self._write_index([])
