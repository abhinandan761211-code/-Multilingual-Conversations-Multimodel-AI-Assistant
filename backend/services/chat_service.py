"""
chat_service.py
===============
The one use case the product is built on: *ask a question, get a checked answer*.

Responsibilities, in order:
  1. resolve or create the conversation,
  2. adopt a newly uploaded image (one pinned image per conversation),
  3. run the reasoning pipeline,
  4. persist both messages in one transaction,
  5. return a JSON-safe response.

The service takes a `Session` rather than opening its own, so a route can wrap
several service calls in one transaction and tests can roll everything back.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from backend.core import config
from backend.db.models import Conversation
from backend.db.repositories.conversations import ConversationRepository
from backend.services.engine import engine
from backend.services.serializers import turn_to_dev, turn_to_response

log = logging.getLogger(__name__)

# Sent to the model when the user only drops an image and types nothing. It is
# deliberately never stored or shown as if the user had written it.
DEFAULT_IMAGE_PROMPT = "Describe this image and point out anything important."
SYNTHETIC_PREFIX = "[Attached "


def detect_mime(data: bytes, filename: str = "") -> str:
    """Cheap sniff so a stored image can be re-served with the right header."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:2] == b"BM":
        return "image/bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "application/octet-stream"


class ChatService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.conversations = ConversationRepository(session)

    # ── conversation lookup ─────────────────────────────────────────────────

    def conversation(self, conversation_id: str) -> Conversation | None:
        return self.conversations.get(conversation_id)

    # ── the main use case ───────────────────────────────────────────────────

    def handle_turn(
        self,
        *,
        conversation_id: str | None,
        message: str,
        image: tuple[str, bytes] | None = None,
        developer_mode: bool = False,
        persist: bool | None = None,
    ) -> dict:
        """
        Ask one question and return the answer plus conversation id.

        Parameters
        ----------
        conversation_id
            Existing conversation, or ``None`` to start a new one. An unknown id
            also starts a new conversation (a deleted chat in another tab should
            not produce an error).
        message
            What the user typed. May be empty **only** when an image is supplied.
        image
            ``(filename, bytes)`` of a newly uploaded image. Omitted → the
            conversation's pinned image is reused.
        """
        persist = config.PERSIST_CHATS if persist is None else persist
        repo = self.conversations

        conversation = (
            repo.get(conversation_id) if conversation_id else None
        ) or repo.create()
        cid = conversation.conversation_id

        typed = (message or "").strip()
        just_attached = image is not None

        # ── 1. image handling ───────────────────────────────────────────────
        if image is not None:
            name, data = image
            repo.attach_image(conversation, name, data, detect_mime(data, name))
            image_bytes: bytes | None = data
            filename = name
            # The user typed nothing: still analyse the image, but keep the
            # display text honest about the fact they didn't ask anything.
            internal_prompt = typed or DEFAULT_IMAGE_PROMPT
            display_text = typed or f"{SYNTHETIC_PREFIX}{name}]"
            synthetic = not typed
        else:
            if not typed:
                raise ValueError("A message or an image is required.")
            internal_prompt = typed
            display_text = typed
            synthetic = False
            image_bytes = conversation.image_bytes
            filename = conversation.image_name or "image"

        # ── 2. pipeline ─────────────────────────────────────────────────────
        engine.ensure_index()
        pipeline = engine.pipeline(cid)

        # Re-seed memory when the pinned image exists but the pipeline was
        # rebuilt (server restart, eviction) — otherwise follow-ups lose it.
        if image_bytes and not just_attached and not pipeline.memory.images:
            previous_answer = next(
                (m.content for m in conversation.messages if m.role == "assistant"),
                "",
            )
            engine.seed_image(cid, image_bytes, filename, previous_answer[:600])

        turn = pipeline.handle_turn(
            user_message=internal_prompt,
            image_bytes=image_bytes,
            image_filename=filename,
        )

        # ── 3. persist ──────────────────────────────────────────────────────
        if persist:
            # Only the turn that introduced the image carries its label; the UI
            # shows the picture once instead of on every follow-up.
            shown_name = filename if just_attached else None
            repo.add_message(
                conversation,
                "user",
                display_text,
                image_name=shown_name,
                synthetic_user=synthetic,
            )
            repo.add_message(
                conversation,
                "assistant",
                turn.answer,
                dev=turn_to_dev(turn),
            )
            self.session.commit()

        payload = turn_to_response(turn, developer_mode=developer_mode)
        payload["conversation_id"] = cid
        payload["conversation_title"] = conversation.title
        payload["display_text"] = display_text
        payload["used_image"] = bool(image_bytes)
        return payload

    # ── CRUD passthroughs used by the routes ────────────────────────────────

    def delete(self, conversation: Conversation) -> None:
        engine.reset_pipeline(conversation.conversation_id)
        self.conversations.delete(conversation)
        self.session.commit()

    def clear(self) -> int:
        removed = self.conversations.clear()
        self.session.commit()
        return removed
