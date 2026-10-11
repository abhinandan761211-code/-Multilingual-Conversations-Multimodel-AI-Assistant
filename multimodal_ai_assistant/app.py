"""
app.py — Streamlit front-end for the Multimodal AI Assistant.

Presentation philosophy
-----------------------
The engineering surface (route names, plan strings, backend identifiers,
similarity scores, OCR confidence) is useful while building and noise while
using. This module therefore has two deliberately separate surfaces:

  * **User view (default)** — an answer, a plain-language trust indicator, and
    which source documents informed it. Nothing else.
  * **Developer view (opt-in)** — the full reasoning trace, image diagnostics,
    OCR output, retrieval scores and validation checks, for the evaluation
    demo and for debugging.

The toggle lives in the sidebar and defaults to OFF, so the first thing a user
sees is a clean product rather than a debug console.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from typing import Any, Optional

import streamlit as st
from PIL import Image

import config
import ui_theme
from chat_store import ChatStore, Conversation, StoredMessage, title_from
from conversation_memory import ConversationMemory
from evidence_retriever import EvidenceRetriever
from pipeline import AssistantPipeline
from reasoning_router import ReasoningRouter
from response_validator import ResponseValidator
from vision_model import VisionModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui_theme.inject()


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────


def image_to_data_uri(data: bytes, max_px: int = 320) -> str:
    """Thumbnail as a data URI, for the small pinned-attachment preview."""
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max(img.size) > max_px:
        scale = max_px / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    fmt = "PNG" if img.mode == "L" else "JPEG"
    img.save(buffer, format=fmt, **({"quality": 82} if fmt == "JPEG" else {}))
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_near_memory_limit(data: bytes) -> bool:
    """True when the upload is large enough that storing it is a trade-off."""
    return len(data) > config.MAX_STORED_IMAGE_BYTES

# ─────────────────────────────────────────────────────────────────────────────
# Cached / session singletons
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def get_vision_model() -> VisionModel:
    model = VisionModel()
    # A missing text model would otherwise fail every text-only turn. Swap in
    # an available local model once, at startup, rather than at request time.
    model.resolve_available_models()
    return model


@st.cache_resource(show_spinner=False)
def get_retriever() -> EvidenceRetriever:
    retriever = EvidenceRetriever()
    retriever.build_index()
    return retriever


@st.cache_resource(show_spinner=False)
def get_shared_validator() -> ResponseValidator:
    """
    Validator shared across sessions.

    The sentence-transformer is loaded lazily (see `_LazyEmbedder`) rather than
    at startup. Loading it eagerly added ~90s to the first page render, during
    which the websocket kept dropping and the app looked broken. The validator
    falls back to lexical checks until the embedder is ready, so deferring it
    changes nothing except how fast the UI appears.
    """
    vision = get_vision_model()
    return ResponseValidator(
        embedder=_LazyEmbedder(), entailment_checker=vision.check_entailment
    )


class _LazyEmbedder:
    """
    Duck-types the subset of the SentenceTransformer API the validator uses
    (`encode`), loading the real model on first call.

    `ResponseValidator` already treats a failing embedder as "fall back to
    lexical checks", so a load failure here degrades gracefully instead of
    breaking validation.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._model = None
        self._failed = False

    def encode(self, texts, show_progress_bar: bool = False):  # noqa: D401
        if self._failed:
            raise RuntimeError("embedder unavailable")
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:  # noqa: BLE001
                self._failed = True
                log.warning("Embedder unavailable (%s); lexical checks only", exc)
                raise RuntimeError("embedder unavailable") from exc
        return self._model.encode(texts, show_progress_bar=show_progress_bar)


@st.cache_resource(show_spinner=False)
def get_chat_store() -> ChatStore:
    return ChatStore()


def fresh_pipeline() -> AssistantPipeline:
    """A pipeline with empty memory — used whenever a conversation is loaded."""
    vision = get_vision_model()
    return AssistantPipeline(
        vision_model=vision,
        retriever=get_retriever(),
        memory=ConversationMemory(),
        router=ReasoningRouter(classifier=vision.classify_request),
        validator=get_shared_validator(),
    )


def get_pipeline() -> AssistantPipeline:
    """One pipeline per browser session, kept alive across Streamlit reruns."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = fresh_pipeline()
    return st.session_state.pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Session <-> persisted conversation
# ─────────────────────────────────────────────────────────────────────────────


def new_conversation() -> None:
    """Start an empty conversation and reset in-memory context."""
    st.session_state.conversation = Conversation(conversation_id="")
    st.session_state.messages = []
    st.session_state.pipeline = fresh_pipeline()


def load_conversation(conversation_id: str) -> None:
    """Restore a saved conversation, including its attached image."""
    conversation = get_chat_store().load(conversation_id)
    if conversation is None:
        st.warning("That conversation could no longer be found.")
        return

    st.session_state.conversation = conversation
    st.session_state.pipeline = fresh_pipeline()

    # Rebuild the chat transcript from storage. The thumbnail is encoded once
    # and reused across messages — encoding per message re-processed the same
    # image for every turn in a long conversation.
    stored_image = conversation.image_bytes()
    image_uri = image_to_data_uri(stored_image) if stored_image else None

    messages = []
    for stored in conversation.messages:
        messages.append(
            {
                "role": stored.role,
                "content": stored.content,
                # Only the message that introduced the image shows it.
                "image_uri": image_uri if stored.image_name else None,
                "turn": DevTurnView(stored.dev) if stored.dev else None,
            }
        )
    st.session_state.messages = messages

    # Re-seed the pipeline's memory so follow-ups keep working after a reload.
    if stored_image:
        record = st.session_state.pipeline.memory.register_image(
            stored_image, filename=conversation.image_name or "image"
        )
        summary = next(
            (m.content for m in conversation.messages if m.role == "assistant"),
            "",
        )
        if summary:
            st.session_state.pipeline.memory.annotate_image(
                record.image_id, summary=summary[:600]
            )


def persist_turn(
    user_text: str,
    answer: str,
    attach: Optional[tuple[str, bytes]] = None,
    dev: Optional[dict] = None,
    synthetic_user: bool = False,
) -> None:
    """
    Append the turn to the stored conversation and save it.

    `synthetic_user` marks a first turn the user triggered but did not phrase
    themselves (they only attached a file). The message is still stored — the
    transcript should show that a turn happened — but the sidebar title is left
    for the first message the user actually wrote.
    """
    if not config.PERSIST_CHATS:
        return
    conversation: Conversation = st.session_state.get("conversation")
    if conversation is None:
        return

    # Assign the id locally rather than calling store.create(), which would
    # write an empty conversation file that then shows up as a duplicate in the
    # sidebar. The conversation is only written when it has real content.
    if not conversation.conversation_id:
        conversation.conversation_id = uuid.uuid4().hex[:12]

    # A conversation holds ONE image, and later turns are answered against it.
    #
    # The message label must therefore describe the image actually stored, not
    # the file the user just picked. Previously the label came straight from
    # `attach[0]` while the bytes were kept only if none existed yet — so
    # attaching a second image left a message labelled with the new filename
    # rendering the OLD picture. The label and the bytes must never disagree.
    stored_name: Optional[str] = None
    if attach is not None:
        name, data = attach
        if conversation.image_b64 is None:
            # First image for this conversation: adopt it.
            conversation.attach_image(name, data)
        # Only claim a label when the stored image really is this file.
        if conversation.image_name == name:
            stored_name = name
    elif conversation.image_b64 is not None:
        # A later turn reusing the pinned image does not repeat it in the
        # transcript, so it carries no label.
        stored_name = None

    conversation.messages.append(
        StoredMessage(role="user", content=user_text, image_name=stored_name)
    )
    conversation.messages.append(
        StoredMessage(role="assistant", content=answer, dev=dev or {})
    )

    # Only let a user-written message set the title. Otherwise the first
    # auto-generated turn would be adopted as the conversation's name.
    if not synthetic_user and (
        not conversation.title or conversation.title == "New conversation"
    ):
        conversation.title = title_from(user_text)

    get_chat_store().save(conversation)


class DevTurnView:
    """
    A rehydrated stand-in for the pipeline's `AssistantTurn`, reconstructed from
    the small dict we persist. It exposes only the attributes the developer
    panel reads, so a reloaded conversation can still show its reasoning.
    """

    def __init__(self, data: Optional[dict] = None) -> None:
        data = data or {}
        self.route = data.get("route", "")
        self.trace = data.get("trace", []) or []
        self.plan = _PlanView(data.get("plan") or {})
        self.diagnostics = _DiagView(data.get("diagnostics")) if data.get("diagnostics") else None
        self.ocr = _Simple(data.get("ocr")) if data.get("ocr") else None
        self.evidence = []
        self.validation = _ValidationView(data.get("validation")) if data.get("validation") else None
        lang = data.get("language")
        self.language = _LanguageView(lang) if lang else None
        self.intent = _Simple(data.get("intent")) if data.get("intent") else None
        self.response_language = (lang or {}).get("response_language", "en")
        self.language_switched = bool((lang or {}).get("switched", False))
        self.response_language_corrected = bool((lang or {}).get("corrected", False))
        self.answer = ""
        self.error = None


class _Simple:
    """
    A read-only attribute bag for data rehydrated from disk.

    Attributes are copied from a dict at runtime, so they are declared here as
    `Any` to document the shape for readers and satisfy the type checker.
    """

    value: Any = None
    name: str = ""
    passed: bool = False
    detail: str = ""
    backend: str = ""
    confidence: float = 0.0
    text: str = ""
    attempted: bool = False
    succeeded: bool = False

    def __init__(self, data: Optional[dict] = None) -> None:
        for key, value in (data or {}).items():
            setattr(self, key, value)


class _PlanView(_Simple):
    def __init__(self, data: Optional[dict] = None) -> None:
        super().__init__(data or {})
        self.rationale = (data or {}).get("rationale", []) or []
        self.confidence = float((data or {}).get("confidence", 0.0) or 0.0)
        self.use_vision = bool((data or {}).get("use_vision", False))
        self.use_ocr = bool((data or {}).get("use_ocr", False))
        self.use_retrieval = bool((data or {}).get("use_retrieval", False))
        self.trust_ocr = bool((data or {}).get("trust_ocr", True))

    def explain(self) -> str:
        return f"route={getattr(self, 'route', '?')} | confidence={self.confidence:.2f}"


class _DiagView(_Simple):
    def __init__(self, data: Optional[dict] = None) -> None:
        super().__init__(data or {})
        self.valid = bool((data or {}).get("valid", True))
        self.error = (data or {}).get("error")
        self.quality_notes = (data or {}).get("quality_notes", []) or []


class _LanguageView(_Simple):
    """Rehydrated `LanguageReport` for the developer panel after a reload."""

    def __init__(self, data: Optional[dict] = None) -> None:
        super().__init__(data or {})
        self.code = (data or {}).get("code", "en")
        self.name = (data or {}).get("name", "")
        self.mixed = bool((data or {}).get("mixed", False))
        self.mixed_languages = list((data or {}).get("mixed_languages", []) or [])
        self.switched = bool((data or {}).get("switched", False))
        self.corrected = bool((data or {}).get("corrected", False))
        self.response_language = (data or {}).get("response_language", "en")

    def is_supported(self) -> bool:
        return bool(getattr(self, "supported", True))


class _ValidationView(_Simple):
    def __init__(self, data: Optional[dict] = None) -> None:
        super().__init__(data or {})
        self.status = _Simple({"value": (data or {}).get("status", "pass")})
        self.confidence = float((data or {}).get("confidence", 0.0) or 0.0)
        self.checks = [_Simple(c) for c in (data or {}).get("checks", [])]
        self.warnings = (data or {}).get("warnings", []) or []

    def as_badge(self) -> str:
        return f"{self.status.value}"


def turn_to_dev(turn) -> dict:
    """The minimum needed to re-render the developer panel after a reload."""
    if turn is None:
        return {}
    plan = getattr(turn, "plan", None)
    diag = getattr(turn, "diagnostics", None)
    report = getattr(turn, "validation", None)
    ocr = getattr(turn, "ocr", None)

    dev: dict = {
        "route": getattr(turn, "route", ""),
        "trace": (getattr(turn, "trace", []) or [])[-14:],
    }
    # Multilingual telemetry, persisted so the developer panel still explains the
    # language handling after a page reload.
    language = getattr(turn, "language", None)
    intent = getattr(turn, "intent", None)
    if language is not None:
        dev["language"] = {
            "code": getattr(language, "code", "en"),
            "name": getattr(language, "name", ""),
            "confidence": round(float(getattr(language, "confidence", 0.0) or 0.0), 2),
            "mixed": bool(getattr(language, "mixed", False)),
            "mixed_languages": list(getattr(language, "mixed_languages", []) or []),
            "script": getattr(language, "script", ""),
            "backend": getattr(language, "backend", ""),
            "reasoning": getattr(language, "reasoning", ""),
            "supported": bool(getattr(language, "is_supported", lambda: True)()),
            "switched": bool(getattr(turn, "language_switched", False)),
            "response_language": getattr(turn, "response_language", "en"),
            "corrected": bool(getattr(turn, "response_language_corrected", False)),
        }
    if intent is not None:
        dev["intent"] = {
            "intent": getattr(intent, "intent", ""),
            "entities": list(getattr(intent, "entities", []) or []),
            "needs_clarification": bool(getattr(intent, "needs_clarification", False)),
            "response_language": getattr(intent, "response_language", ""),
            "confidence": round(float(getattr(intent, "confidence", 0.0) or 0.0), 2),
            "source": getattr(intent, "source_of_resolution", "heuristic"),
        }
    if plan is not None:
        dev["plan"] = {
            "confidence": getattr(plan, "confidence", 0.0),
            "rationale": getattr(plan, "rationale", []) or [],
            "use_vision": getattr(plan, "use_vision", False),
            "use_ocr": getattr(plan, "use_ocr", False),
            "use_retrieval": getattr(plan, "use_retrieval", False),
            "trust_ocr": getattr(plan, "trust_ocr", True),
        }
    if diag is not None and getattr(diag, "valid", False):
        dev["diagnostics"] = {
            "valid": True,
            "width": diag.width,
            "height": diag.height,
            "blur_score": round(float(diag.blur_score), 1),
            "mean_brightness": round(float(diag.mean_brightness), 1),
            "looks_like_text": bool(diag.looks_like_text),
            "quality_notes": getattr(diag, "quality_notes", []) or [],
        }
    if ocr is not None and getattr(ocr, "attempted", False) and getattr(ocr, "succeeded", False):
        dev["ocr"] = {
            "backend": getattr(ocr, "backend", ""),
            "confidence": round(float(getattr(ocr, "confidence", 0.0) or 0.0), 2),
            "text": (getattr(ocr, "text", "") or "")[:1200],
            "attempted": True,
            "succeeded": True,
        }
    if report is not None:
        dev["validation"] = {
            "status": getattr(getattr(report, "status", None), "value", "pass"),
            "confidence": round(float(getattr(report, "confidence", 0.0) or 0.0), 2),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                }
                for c in getattr(report, "checks", [])
            ],
            "warnings": getattr(report, "warnings", []) or [],
        }
    return dev


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────


def _status_row(label: str, ok: bool, warn: bool = False) -> str:
    cls = "dot-ok" if ok else ("dot-warn" if warn else "dot-off")
    return (
        f'<div class="status-row"><span class="status-dot {cls}"></span>'
        f"{label}</div>"
    )


def render_history() -> None:
    """Sidebar chat list, newest first, with the active one highlighted."""
    store = get_chat_store()
    entries = store.list_conversations()
    active_id = getattr(st.session_state.get("conversation"), "conversation_id", "")

    if not entries:
        st.caption("No saved conversations yet.")
        return

    for entry in entries:
        cid = entry["conversation_id"]
        is_active = cid == active_id
        label = entry.get("title") or "Untitled"

        # A tight split: title takes the space, the delete control sits beside it.
        #
        # No `help=` on either button: Streamlit renders the tooltip target as its
        # own element, which appeared as a detached box floating outside the row
        # and pushed the title past the sidebar edge. The title is already a
        # readable preview, and the × glyph is self-explanatory.
        col_title, col_del = st.columns([0.84, 0.16], vertical_alignment="center")

        with col_title:
            if st.button(
                label,
                key=f"open_{cid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    load_conversation(cid)
                    st.rerun()

        with col_del:
            if st.button(
                "×",
                key=f"del_{cid}",
                use_container_width=True,
            ):
                store.delete(cid)
                if is_active:
                    new_conversation()
                st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        if st.button("＋  New chat", use_container_width=True, type="primary"):
            new_conversation()
            st.rerun()

        # ── history ─────────────────────────────────────────────────────────
        st.markdown(
            '<div class="side-label">Recent</div>', unsafe_allow_html=True
        )
        render_history()

        # ── system status (compact) ─────────────────────────────────────────
        st.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)
        health = get_vision_model().health_check()
        ready = health.vision_model_available

        docs = 0
        try:
            docs = len(get_retriever().stats().get("documents", []))
        except Exception:  # noqa: BLE001
            pass

        rows = [
            _status_row(
                "Vision model ready" if ready else "Vision model unavailable", ready
            ),
            _status_row(
                "Language model ready",
                bool(health.text_model_available or health.text_model_fallback),
            ),
            _status_row(
                f"Knowledge base · {docs} documents", docs > 0, warn=docs == 0
            ),
        ]
        st.markdown("".join(rows), unsafe_allow_html=True)

        # Setup guidance only when something is actually wrong.
        if not health.server_up:
            st.warning("The local model service isn't running.", icon="⚠️")
            st.code("ollama serve", language="bash")
        elif not ready:
            st.warning("The vision model hasn't been downloaded yet.", icon="⚠️")
            st.code(f"ollama pull {config.VISION_MODEL}", language="bash")

        # ── options ─────────────────────────────────────────────────────────
        st.markdown('<div class="side-label">Options</div>', unsafe_allow_html=True)
        # NOTE: deliberately NOT st.toggle(). Streamlit renders that widget's
        # <label> as a filled, bordered container, which drew a grey pill behind
        # the text. The fill is applied by Streamlit's own single-class utility
        # rules injected after this stylesheet, so no amount of `!important` on
        # our side reliably removed it.
        #
        # st.checkbox has the plainer structure we want, and we style it
        # ourselves from scratch below.
        st.session_state["dev_mode"] = st.checkbox(
            "Developer mode",
            value=st.session_state.get("dev_mode", False),
        )
        st.caption("Show how each answer was decided.")

        if st.session_state["dev_mode"]:
            vision = get_vision_model()
            with st.expander("Model configuration"):
                st.caption("Vision model")
                st.code(vision.vision_model, language=None)
                st.caption("Language model")
                st.code(vision.text_model, language=None)
                if health.text_model_fallback:
                    st.caption(
                        f"Auto-selected because `{config.TEXT_MODEL}` isn't installed."
                    )
                try:
                    stats = get_retriever().stats()
                    st.caption("Retrieval")
                    st.code(
                        f"{stats['backend']} · {stats['chunks']} chunks", language=None
                    )
                except Exception:  # noqa: BLE001
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────


def render_user_message(message: dict) -> None:
    st.markdown(message["content"])
    # The image appears once, where it was introduced — not on every follow-up.
    if message.get("image_uri"):
        st.markdown(
            f'<img class="chat-image" src="{message["image_uri"]}" alt="attachment">',
            unsafe_allow_html=True,
        )


def render_assistant_message(message: dict) -> None:
    """The clean answer surface: text, one trust pill, and source chips."""
    turn = message.get("turn")

    st.markdown(ui_theme.clean_answer(message["content"]))

    if turn is None:
        return

    # A visible caution when the input itself was hard to read.
    notice = ui_theme.quality_notice(getattr(turn, "diagnostics", None))
    if notice:
        st.caption(f"ℹ️ {notice}")

    pill = ui_theme.confidence_pill(getattr(turn, "validation", None))
    if pill:
        st.markdown(
            f'<div style="margin-top:0.7rem">{pill}</div>',
            unsafe_allow_html=True,
        )

    chips = ui_theme.source_chips(getattr(turn, "evidence", []))
    if chips:
        st.markdown(
            f'<div style="margin-top:0.35rem;font-size:0.72rem;'
            f'color:{ui_theme.COLORS["text_muted"]}">Based on</div>{chips}',
            unsafe_allow_html=True,
        )

    # Which language the answer was written in — and whether a switch/mix was
    # handled. Shown always, because it is user-facing state, not telemetry.
    badge = ui_theme.language_badge(turn)
    if badge:
        st.markdown(badge, unsafe_allow_html=True)

    # Everything technical is opt-in and collapsed.
    if st.session_state.get("dev_mode", False):
        render_developer_panel(turn)


def _render_language_section(turn) -> None:
    """The language-handling trace: detection, code-mixing, and the answer's language."""
    report = getattr(turn, "language", None)
    if report is None:
        return

    st.markdown("**Language**")
    lines = [
        f"- Detected: `{report.name} ({report.code})` "
        f"confidence `{report.confidence:.2f}`, method `{report.backend}`, "
        f"script `{report.script}`",
    ]
    if getattr(report, "mixed", False):
        lines.append(
            f"- Code-mixing detected: `{', '.join(report.mixed_languages)}`"
        )
    if report.reasoning:
        lines.append(f"- Reasoning: {report.reasoning}")
    response_lang = getattr(turn, "response_language", None)
    if response_lang:
        lines.append(f"- Answered in: `{response_lang}`")
    if getattr(turn, "language_switched", False):
        lines.append("- Language switch handled; earlier context retained")
    if getattr(turn, "response_language_corrected", False):
        lines.append("- Answer re-expressed to match the requested language")
    st.markdown("\n".join(lines))


def render_developer_panel(turn) -> None:
    """The engineering surface: routing, diagnostics, retrieval, validation."""
    activity = ui_theme.activity_label(turn.route)
    icon = ui_theme.activity_icon(turn.route)

    with st.expander(f"{icon} How this was handled — {activity}", expanded=False):
        _render_language_section(turn)

        plan = turn.plan
        if plan is not None:
            st.markdown("**Decision**")
            st.markdown(
                f"- Route: `{turn.route}`  \n"
                f"- Confidence: `{plan.confidence:.2f}`  \n"
                f"- Vision: `{plan.use_vision}` · OCR: `{plan.use_ocr}` "
                f"(trusted: `{plan.trust_ocr}`) · Retrieval: `{plan.use_retrieval}`"
            )
            if plan.rationale:
                st.markdown("**Rationale**")
                for reason in plan.rationale:
                    st.markdown(f"- {reason}")

        if turn.trace:
            st.markdown("**Trace**")
            for step in turn.trace:
                st.markdown(f"- {step}")

    if turn.diagnostics is not None:
        with st.expander("Image diagnostics", expanded=False):
            d = turn.diagnostics
            if not d.valid:
                st.error(f"Invalid image: {d.error}")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Size", f"{d.width}×{d.height}")
                c2.metric("Sharpness", f"{d.blur_score:.0f}")
                c3.metric("Brightness", f"{d.mean_brightness:.0f}")
                c4.metric("Text-like", "yes" if d.looks_like_text else "no")
                if d.quality_notes:
                    st.caption(" · ".join(d.quality_notes))

                if turn.ocr is not None:
                    ocr = turn.ocr
                    if ocr.attempted and ocr.succeeded:
                        st.markdown(
                            f"**OCR** — `{ocr.backend}`, confidence `{ocr.confidence:.2f}`"
                        )
                        st.code(ocr.text or "(no text found)", language=None)
                    else:
                        st.caption(f"OCR not used: {ocr.error or 'n/a'}")

    if getattr(turn, "evidence", None):
        with st.expander(f"Retrieved evidence ({len(turn.evidence)})", expanded=False):
            for e in turn.evidence:
                st.markdown(f"`{e.citation()}` — score `{e.score}`")
                st.caption(e.text[:220] + ("…" if len(e.text) > 220 else ""))

    report = getattr(turn, "validation", None)
    if report is not None:
        with st.expander("Validation checks", expanded=False):
            st.markdown(
                f"Status: `{report.status.value}` · confidence `{report.confidence:.2f}`"
            )
            for check in report.checks:
                mark = "✅" if check.passed else "❌"
                st.markdown(f"{mark} **{check.name}** — {check.detail}")
            for warning in report.warnings:
                st.warning(warning)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def render_header() -> None:
    health = get_vision_model().health_check()
    ready = health.server_up and health.vision_model_available
    badge = (
        '<span class="badge">● Ready</span>'
        if ready
        else '<span class="badge" style="color:#E0A33E;background:#33280F;'
             'border-color:#E0A33E55">● Setup needed</span>'
    )
    st.markdown(
        f"""
        <div class="app-header">
            <div class="mark">◆</div>
            <div>
                <div class="title">Multimodal AI Assistant</div>
                <div class="subtitle">Ask about an image, a document, or just a question.</div>
            </div>
            {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    """A purposeful landing state instead of a bare chat box."""
    st.markdown(
        f'<div style="margin:1.4rem 0 0.9rem 0;font-size:0.86rem;'
        f'color:{ui_theme.COLORS["text_muted"]}">'
        "Upload an image, or start with a question."
        "</div>",
        unsafe_allow_html=True,
    )

    # Icons are inline SVG, not emoji: the font stack on this machine renders
    # emoji as broken/boxed glyphs inside these cards.
    _chart_icon = (
        '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" '
        'stroke="#4F7DF3" stroke-width="2" stroke-linecap="round">'
        '<path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/></svg>'
    )
    _text_icon = (
        '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" '
        'stroke="#4F7DF3" stroke-width="2" stroke-linecap="round">'
        '<path d="M4 7V5h16v2"/><path d="M12 5v14"/><path d="M9 19h6"/></svg>'
    )
    _link_icon = (
        '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" '
        'stroke="#4F7DF3" stroke-width="2" stroke-linecap="round">'
        '<path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1"/>'
        '<path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1"/></svg>'
    )
    _doc_icon = (
        '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" '
        'stroke="#4F7DF3" stroke-width="2" stroke-linecap="round">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h5"/></svg>'
    )

    examples = [
        (_chart_icon, "Analyse a chart", "When does validation loss start increasing?"),
        (_text_icon, "Read text", "Extract the text from this screenshot."),
        (_link_icon, "Follow up", "What does the middle layer do?"),
        (_doc_icon, "Ask a question", "What is overfitting and how do you prevent it?"),
    ]

    # A one-line, honest statement of the multilingual capability, so it is
    # discoverable without reading the source. The languages are named in their
    # own scripts.
    st.markdown(
        f'<div style="margin:0.2rem 0 0.9rem 0;font-size:0.8rem;'
        f'color:{ui_theme.COLORS["text_muted"]}">'
        "Ask in <b>English</b>, <b>हिन्दी</b>, <b>ଓଡ଼ିଆ</b> or <b>বাংলা</b> — or mix "
        "them. The assistant detects the language automatically, keeps the "
        "context when you switch, and answers in the language you used."
        "</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(examples))
    for col, (icon, title, body) in zip(cols, examples):
        with col:
            st.markdown(
                f'<div style="background:{ui_theme.COLORS["surface"]};'
                f'border:1px solid {ui_theme.COLORS["border"]};border-radius:10px;'
                f'padding:0.85rem 0.9rem;min-height:110px;'
                f'display:flex;flex-direction:column;gap:0.3rem">'
                f'<div style="font-size:1rem;line-height:1.1">{icon}</div>'
                f'<div style="font-weight:600;font-size:0.84rem;line-height:1.3;'
                f'color:{ui_theme.COLORS["text"]}">{title}</div>'
                f'<div style="font-size:0.76rem;color:{ui_theme.COLORS["text_muted"]};'
                f'line-height:1.5">{body}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


def render_attachment_bar(conversation: Conversation) -> None:
    """
    The pinned-image strip above the composer.

    This is the ChatGPT-style affordance: the image belongs to the conversation,
    not to one message, so the user uploads once and keeps asking. The bar makes
    that state visible without repeating the picture in every reply.
    """
    data = conversation.image_bytes()
    if not data:
        return

    thumb = image_to_data_uri(data, max_px=96)
    name = conversation.image_name or "image"
    st.markdown(
        '<div class="attach-bar">'
        f'<img src="{thumb}" alt="attached image">'
        '<div class="meta">'
        f'<div class="name">{name}</div>'
        '<div class="hint">Attached to this chat — keep asking about it.</div>'
        "</div>"
        '<span class="tag">Active</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def restore_session() -> None:
    """
    Initialise session state for a fresh browser session.

    Resumes the most recent saved conversation rather than always starting a
    blank one. Without this, every page load created another empty conversation
    and the history list filled up with duplicates of the same chat.
    """
    if "conversation" in st.session_state:
        return

    store = get_chat_store()
    entries = store.list_conversations()
    if entries:
        load_conversation(entries[0]["conversation_id"])
    else:
        new_conversation()

    st.session_state.setdefault("messages", [])


def main() -> None:
    restore_session()

    render_sidebar()
    render_header()

    conversation: Conversation = st.session_state.conversation

    if not st.session_state.messages:
        render_empty_state()
        # The empty state is short, so the docked composer floats up and crowds
        # the example cards. A modest spacer pushes it back down without leaving
        # a large empty band on taller viewports.
        st.markdown('<div style="height:2.5rem"></div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=None):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                render_user_message(message)

    # The composer is a fixed overlay: it occupies no layout space, so without
    # a real spacer the final message sits underneath it (measured:
    # msgClippedByDock = true with 9rem). The dock is ~150px tall including its
    # padding, so reserve clearly more than that.
    if st.session_state.messages:
        st.markdown('<div style="height:13rem"></div>', unsafe_allow_html=True)

    # ── attachment + composer ───────────────────────────────────────────────
    # Streamlit pins st.chat_input to a bottom dock, so no widget can share a row
    # with it. Whatever we add here lands in the transcript, above the dock.
    #
    # Left-aligned (not centred) so it lines up with the content column and sits
    # directly above the composer, reading as part of it rather than as a stray
    # button adrift in the middle of the page.
    uploaded = None

    with st.container():
        if conversation.image_b64:
            # Compact bar: thumbnail, name, and a small detach action beside it.
            render_attachment_bar(conversation)
            detach_col, _ = st.columns([0.18, 0.82], vertical_alignment="center")
            with detach_col:
                if st.button(
                    "Remove image",
                    key="detach_image",
                    use_container_width=True,
                ):
                    conversation.image_b64 = None
                    conversation.image_name = None
                    st.session_state.pipeline = fresh_pipeline()
                    if config.PERSIST_CHATS and conversation.conversation_id:
                        get_chat_store().save(conversation)
                    st.rerun()
        else:
            # No image yet: a compact attach control, left-aligned with the
            # transcript column rather than centred over empty space.
            up_col, _ = st.columns([0.18, 0.82], vertical_alignment="center")
            with up_col:
                uploaded = st.file_uploader(
                    "Add an image",
                    type=["png", "jpg", "jpeg", "webp", "bmp", "gif"],
                    label_visibility="collapsed",
                    key="image_uploader",
                )

    prompt = st.chat_input("Message the assistant…")

    # An upload takes priority: adopt the image, then ask about it.
    #
    # The composer above was already built for the *previous* state (no image),
    # so we re-run afterwards to swap the uploader for the attachment bar. Without
    # this rerun the uploader stays on screen and "Detach image" never appears.
    if uploaded is not None:
        handle_user_turn(
            prompt=None,
            staged_upload=(uploaded.name, uploaded.getvalue()),
        )
        st.rerun()

    if prompt:
        handle_user_turn(prompt=prompt)


def handle_user_turn(
    prompt: Optional[str],
    staged_upload: Optional[tuple[str, bytes]] = None,
) -> None:
    """
    Process one turn.

    Two ways in:
      * `staged_upload` — the user just attached an image. Adopt it into the
        conversation, then immediately analyse it so the first reply is useful
        rather than another question.
      * `prompt` — a normal message, answered against the pinned image (if any)
        or purely from text and reference material.
    """
    pipeline = get_pipeline()
    conversation: Conversation = st.session_state.conversation

    just_attached = staged_upload is not None

    if just_attached:
        name, data = staged_upload
        conversation.attach_image(name, data)

        # What the user actually said (may be empty — they only dropped a file).
        typed = (prompt or "").strip()
        # What we send to the model. With no typed question we still want a real
        # analysis rather than a request for clarification.
        internal_prompt = typed or (
            "Describe this image and point out anything important."
        )
        image_bytes, filename = data, name
        # Displayed/persisted text: never the machine-generated instruction, or
        # the conversation gets titled with words the user never wrote.
        display_text = typed or f"[Attached {name}]"
        is_synthetic = not typed
    else:
        internal_prompt = (prompt or "").strip()
        if not internal_prompt:
            return
        display_text = internal_prompt
        is_synthetic = False
        # Reuse the pinned image — this is the whole point of attaching it once.
        image_bytes = conversation.image_bytes()
        filename = conversation.image_name or "image"

    # Show the picture only once per conversation — on the turn that introduced
    # it. Later turns are answered from the pinned attachment, so repeating the
    # thumbnail on every message is noise.
    #
    # Uses the *persisted* conversation (which survives reruns) rather than the
    # transient session list, so a rerun can't re-render the image as if it were
    # a new attachment.
    already_shown = any(m.image_name for m in conversation.messages)
    show_image = just_attached or not already_shown
    image_uri = image_to_data_uri(image_bytes) if (image_bytes and show_image) else None

    st.session_state.messages.append(
        {"role": "user", "content": display_text, "image_uri": image_uri}
    )
    with st.chat_message("user", avatar=None):
        st.markdown(display_text)
        if image_uri:
            st.markdown(
                f'<img class="chat-image" src="{image_uri}" alt="attachment">',
                unsafe_allow_html=True,
            )

    with st.chat_message("assistant", avatar=None):
        placeholder = st.empty()
        # A short, honest status line — vision inference can take a while.
        placeholder.caption("Reading your image…" if image_bytes else "Thinking…")

        turn = pipeline.handle_turn(
            user_message=internal_prompt,
            image_bytes=image_bytes,
            image_filename=filename,
        )
        placeholder.empty()

        st.session_state.messages.append(
            {"role": "assistant", "content": turn.answer, "turn": turn}
        )
        render_assistant_message(
            {"role": "assistant", "content": turn.answer, "turn": turn}
        )

    # ── persist ─────────────────────────────────────────────────────────────
    # The image is stored once on the conversation, so follow-up turns save only
    # text — a long Q&A over one image stays small on disk.
    #
    # `display_text` is stored rather than the internal prompt, so the sidebar
    # title reflects what the user wrote (or "[Attached chart.png]" when they
    # only dropped a file) instead of our own instruction text.
    persist_turn(
        user_text=display_text,
        answer=turn.answer,
        attach=staged_upload if just_attached else None,
        dev=turn_to_dev(turn),
        synthetic_user=is_synthetic,
    )


if __name__ == "__main__":
    main()
