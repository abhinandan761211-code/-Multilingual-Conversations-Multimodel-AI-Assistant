"""
conversation_memory.py
======================
Maintains conversational context across turns, including *which image* each
turn referred to.

The hard part of "maintain context" is not storing messages — it is resolving
referring expressions like "it", "that layer", "the middle one" to a concrete
artefact (an image, a previous answer). This module does that with an explicit,
inspectable model:

    Turn   = one user message + one assistant reply, plus the image it used
    Memory = ordered list of Turns + a set of Images the session has seen

Resolving a follow-up then becomes a lookup: find the most recent image, or the
image matching a mentioned label, and attach it to the new turn.

It is storage-agnostic on purpose. Streamlit drives `st.session_state`, but the
core is a plain object so the unit tests can run headless.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import config


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ImageRecord:
    """One image the user has shown, with the summary the model gave it."""

    image_id: str
    filename: str
    data: bytes
    width: int = 0
    height: int = 0
    summary: str = ""
    extracted_text: str = ""
    labels: list[str] = field(default_factory=list)
    turn_index: int = 0

    @property
    def is_small(self) -> bool:
        return len(self.data) <= config.MAX_IMAGE_BYTES_IN_MEMORY


@dataclass
class Turn:
    """One exchange."""

    index: int
    user_message: str
    assistant_message: str = ""
    image_id: Optional[str] = None
    route: str = ""
    validation_status: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Words that signal the user is pointing at something rather than introducing
# a new topic. Deliberately broad — the router does the fine-grained work.
_REFERENTIAL_CUES = {
    "it", "its", "this", "that", "these", "those", "them", "they",
    "the image", "the picture", "the photo", "the diagram", "the graph",
    "the chart", "above", "previous", "earlier", "same", "there", "here",
}

# Phrases like "the middle layer", "the second line" imply a *part* of the
# most recent image.
_PART_CUES = {
    "middle", "center", "centre", "top", "bottom", "left", "right",
    "upper", "lower", "inner", "outer", "first", "second", "third", "last",
    "next", "final", "input", "output", "hidden",
}


class ConversationMemory:
    """Ordered history + image registry for a single chat session."""

    def __init__(self, max_turns: int = config.MAX_TURNS) -> None:
        self.turns: list[Turn] = []
        self.images: dict[str, ImageRecord] = {}
        self.max_turns = max_turns
        self._counter = 0
        self._active_image_id: Optional[str] = None

    # ── writing ─────────────────────────────────────────────────────────────

    @staticmethod
    def hash_image(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:16]

    def register_image(
        self,
        data: bytes,
        filename: str = "image",
        width: int = 0,
        height: int = 0,
    ) -> ImageRecord:
        """Store (or reuse) an image and make it the active one."""
        image_id = self.hash_image(data)
        if image_id in self.images:
            record = self.images[image_id]
            record.filename = filename or record.filename
            self._active_image_id = image_id
            return record

        record = ImageRecord(
            image_id=image_id,
            filename=filename,
            data=data if len(data) <= config.MAX_IMAGE_BYTES_IN_MEMORY else data,
            width=width,
            height=height,
            turn_index=self._counter,
        )
        self.images[image_id] = record
        self._active_image_id = image_id
        return record

    def annotate_image(
        self,
        image_id: str,
        summary: str = "",
        extracted_text: str = "",
        labels: Optional[Iterable[str]] = None,
    ) -> None:
        """Store what we learned about an image so later turns can reuse it
        without paying for another vision call."""
        record = self.images.get(image_id)
        if not record:
            return
        if summary:
            record.summary = summary
        if extracted_text:
            record.extracted_text = extracted_text
        if labels:
            record.labels = list(dict.fromkeys(labels))

    def add_turn(
        self,
        user_message: str,
        assistant_message: str = "",
        image_id: Optional[str] = None,
        route: str = "",
        validation_status: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Turn:
        self._counter += 1
        if image_id:
            self._active_image_id = image_id
        turn = Turn(
            index=self._counter,
            user_message=user_message,
            assistant_message=assistant_message,
            image_id=image_id,
            route=route,
            validation_status=validation_status,
            metadata=metadata or {},
        )
        self.turns.append(turn)
        self._prune()
        return turn

    def _prune(self) -> None:
        """
        Keep the last N turns in full. Images are kept even if their turn is
        pruned, because a follow-up may still point at them — that is exactly
        the context the assignment asks us to preserve.
        """
        if len(self.turns) <= self.max_turns:
            return
        self.turns = self.turns[-self.max_turns:]

    # ── reading / resolving ─────────────────────────────────────────────────

    @property
    def active_image(self) -> Optional[ImageRecord]:
        if self._active_image_id is None:
            return None
        return self.images.get(self._active_image_id)

    def last_turn(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None

    def get_image(self, image_id: Optional[str]) -> Optional[ImageRecord]:
        if not image_id:
            return None
        return self.images.get(image_id)

    def is_follow_up(self, message: str) -> bool:
        """
        Does this message appear to refer back to earlier context?

        Two signals:
          1. An explicit referential cue ("it", "that graph", "previous").
          2. A part-of-image cue ("the middle layer") *and* we already have an
             image in the session.
          3. The message is very short and not an imperative question on its own
             ("and the loss?").
        """
        lowered = message.lower().strip()
        if not lowered:
            return False

        tokens = set(re.findall(r"[a-z']+", lowered))
        if tokens & _REFERENTIAL_CUES:
            return True

        if self.images and (tokens & _PART_CUES):
            return True

        # A deictic determiner ("the blue one", "that layer") points at
        # something already on screen — only meaningful once we hold an image.
        if self.images:
            # "the/that + <adjective> one" or "the <noun>" with no new-image cue
            if re.search(
                r"\b(?:the|that|those|these|another|other)\b[^?]*\b"
                r"(?:one|ones|line|curve|bar|layer|part|section|bit|side|"
                r"axis|label|value|point|region|node|color|colour)\b",
                lowered,
            ):
                return True

        # Short fragments that don't stand alone usually continue the topic.
        if len(tokens) <= 3 and self.turns and not lowered.endswith("?"):
            if tokens & {"and", "but", "so", "then", "why", "how", "what"}:
                return True

        return False

    def resolve_image_for(self, message: str) -> Optional[ImageRecord]:
        """
        Decide which image a new message is about.

        Rules, most specific first:
          1. If the user names a file we've seen, use that.
          2. If the message looks like a follow-up, reuse the active image.
          3. Otherwise return None — the caller may attach a freshly uploaded
             image instead.
        """
        lowered = message.lower()
        for record in self.images.values():
            stem = record.filename.lower().rsplit(".", 1)[0]
            if stem and stem in lowered:
                return record
        if self.is_follow_up(message):
            return self.active_image
        return None

    # ── prompt-friendly rendering ───────────────────────────────────────────

    def context_for_prompt(
        self,
        max_turns: Optional[int] = None,
        exclude_images: bool = False,
    ) -> str:
        """
        A compact transcript for the model. Includes image summaries rather than
        raw pixels — cheaper and more reliable than replaying every image.

        `exclude_images` drops every turn that involved an image, and removes
        the "[attached image: …]" annotations. Use it when answering a
        self-contained question: leaving the earlier chart description in
        context makes the model answer *about the chart* instead of the
        question (e.g. replying "the image does not provide information about
        the capital of France" rather than naming the capital).
        """
        if not self.turns:
            return ""

        turns = self.turns[-(max_turns or self.max_turns):]
        if exclude_images:
            turns = [t for t in turns if not t.image_id]

        lines: list[str] = []
        for turn in turns:
            image_note = ""
            if turn.image_id and not exclude_images:
                record = self.images.get(turn.image_id)
                label = record.filename if record else turn.image_id
                image_note = f" [attached image: {label}]"
            lines.append(f"User{image_note}: {turn.user_message}")
            if turn.assistant_message:
                reply = turn.assistant_message.strip().replace("\n", " ")
                if len(reply) > 400:
                    reply = reply[:400] + " …"
                lines.append(f"Assistant: {reply}")
        return "\n".join(lines)

    def image_context(self, record: Optional[ImageRecord] = None) -> str:
        """What we already know about an image, for reuse in follow-ups."""
        record = record or self.active_image
        if not record:
            return ""
        parts = [f"Previously shown image '{record.filename}'."]
        if record.summary:
            parts.append(f"Earlier description: {record.summary}")
        if record.extracted_text:
            snippet = record.extracted_text.strip()
            if len(snippet) > 800:
                snippet = snippet[:800] + " …"
            parts.append(f"Text previously read from it: {snippet}")
        return "\n".join(parts)

    # ── serialisation for Streamlit ─────────────────────────────────────────

    def to_state(self) -> dict[str, Any]:
        """Plain dict safe to stash in st.session_state across reruns."""
        return {
            "turns": [t.__dict__ for t in self.turns],
            "images": {k: v.__dict__ for k, v in self.images.items()},
            "counter": self._counter,
            "active_image_id": self._active_image_id,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], max_turns: int = config.MAX_TURNS) -> "ConversationMemory":
        memory = cls(max_turns=max_turns)
        memory._counter = state.get("counter", 0)
        memory._active_image_id = state.get("active_image_id")
        for data in state.get("images", {}).values():
            record = ImageRecord(**data)
            memory.images[record.image_id] = record
        for data in state.get("turns", []):
            memory.turns.append(Turn(**data))
        return memory

    def reset(self) -> None:
        self.turns.clear()
        self.images.clear()
        self._active_image_id = None
        self._counter = 0
