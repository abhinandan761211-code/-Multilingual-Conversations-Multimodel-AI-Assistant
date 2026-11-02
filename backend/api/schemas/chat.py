"""
Request/response models for the chat and conversation endpoints.

FastAPI uses these for validation *and* to generate the OpenAPI docs, so they
double as the API contract the front-end is written against.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── requests ──────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(
        default="",
        description="What the user typed. May be empty only when an image is sent.",
        max_length=8000,
    )
    conversation_id: str | None = Field(
        default=None,
        description="Continue an existing conversation. Omit to start a new one.",
    )
    developer_mode: bool = Field(
        default=False, description="Include the routing/validation trace in `dev`."
    )
    image_b64: str | None = Field(
        default=None, description="Base64 (or data URI) of an image to attach."
    )
    image_name: str | None = Field(default=None, description="Filename of the image.")

    def image_bytes(self) -> tuple[str, bytes] | None:
        """Decode the optional inline image into ``(name, bytes)``."""
        if not self.image_b64:
            return None
        import base64
        import binascii

        raw = self.image_b64
        name = self.image_name or "image"
        # Accept both a bare base64 string and a `data:image/png;base64,...` URI.
        if raw.startswith("data:") and "," in raw:
            header, _, raw = raw.partition(",")
            name = self.image_name or header.split(";")[0].split("/")[-1] or "image"
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image_b64 is not valid base64: {exc}") from exc
        return name, data


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


# ── responses ─────────────────────────────────────────────────────────────────


class SourceChip(BaseModel):
    citation: str = ""
    score: float = 0.0
    source: str = ""


class LanguageInfo(BaseModel):
    code: str = "en"
    name: str = ""
    confidence: float = 0.0
    script: str = ""
    backend: str = ""
    reasoning: str = ""
    mixed: bool = False
    mixed_languages: list[str] = Field(default_factory=list)
    supported: bool = True
    switched: bool = False
    response_language: str = "en"
    corrected: bool = False


class ChatResponse(BaseModel):
    answer: str
    route: str = ""
    error: str | None = None
    conversation_id: str
    conversation_title: str = ""
    display_text: str = ""
    used_image: bool = False
    sources: list[SourceChip] = Field(default_factory=list)
    language: LanguageInfo | None = None
    validation: dict[str, Any] | None = None
    # Omitted entirely unless developer_mode was requested (the route drops
    # None fields), so a normal client never sees a null trace.
    dev: dict[str, Any] | None = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    image_name: str | None = None
    dev: dict[str, Any] = Field(default_factory=dict)
    created_at: float


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    updated_at: float
    message_count: int
    preview: str
    has_image: bool


class ConversationList(BaseModel):
    total: int
    items: list[ConversationSummary]


class ConversationDetail(ConversationSummary):
    created_at: float
    image_name: str | None = None
    messages: list[MessageOut] = Field(default_factory=list)
