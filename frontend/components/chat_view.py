"""
chat_view.py
============
Rendering of the transcript: user turns, assistant turns, the trust signal and
the opt-in developer panel.

The widgets in the theme module expect attribute access
(`turn.validation.confidence`). The API speaks JSON (dicts), so the tiny view
objects below adapt one to the other. They are the *only* place in the front-end
that knows about the API's turn shape.
"""
from __future__ import annotations

import base64
import io
from typing import Any

import streamlit as st
from PIL import Image

from frontend import theme

# ─────────────────────────────────────────────────────────────────────────────
# View adapters (dict -> attribute bag)
# ─────────────────────────────────────────────────────────────────────────────


class _Bag:
    """Attribute access over a dict, reading through to getattr-based widgets."""

    def __init__(self, data: dict | None = None) -> None:
        for key, value in (data or {}).items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:  # only called when missing
        return None


class ValidationView(_Bag):
    def __init__(self, data: dict | None) -> None:
        data = data or {}
        super().__init__(data)
        self.status = _Bag({"value": str(data.get("status", "pass"))})
        self.confidence = float(data.get("confidence", 0.0) or 0.0)
        self.checks = [_Bag(c) for c in data.get("checks", []) or []]
        self.warnings = data.get("warnings", []) or []

    def as_badge(self) -> str:
        return self.status.value


class DiagnosticsView(_Bag):
    def __init__(self, data: dict | None) -> None:
        data = data or {}
        super().__init__(data)
        self.valid = bool(data.get("valid", False))
        self.error = data.get("error")
        self.quality_notes = data.get("quality_notes", []) or []
        self.is_blurry = False
        self.is_low_resolution = False
        self.is_too_dark = False
        self.is_too_bright = False
        for note in self.quality_notes:
            low = note.lower()
            if "blur" in low:
                self.is_blurry = True
            if "small" in low or "low-res" in low or "resolution" in low:
                self.is_low_resolution = True
            if "dark" in low:
                self.is_too_dark = True
            if "bright" in low or "wash" in low:
                self.is_too_bright = True


class LanguageView(_Bag):
    def __init__(self, data: dict | None) -> None:
        data = data or {}
        super().__init__(data)
        self.code = data.get("code", "en")
        self.name = data.get("name", "")
        self.confidence = float(data.get("confidence", 0.0) or 0.0)
        self.script = data.get("script", "")
        self.backend = data.get("backend", "")
        self.reasoning = data.get("reasoning", "")
        self.mixed = bool(data.get("mixed", False))
        self.mixed_languages = data.get("mixed_languages", []) or []

    def is_supported(self) -> bool:
        return bool(self.supported if self.supported is not None else True)


class PlanView(_Bag):
    def __init__(self, data: dict | None) -> None:
        data = data or {}
        super().__init__(data)
        self.confidence = float(data.get("confidence", 0.0) or 0.0)
        self.rationale = data.get("rationale", []) or []
        self.use_vision = bool(data.get("use_vision", False))
        self.use_ocr = bool(data.get("use_ocr", False))
        self.use_retrieval = bool(data.get("use_retrieval", False))
        self.trust_ocr = bool(data.get("trust_ocr", True))

    def explain(self) -> str:
        return f"route={self.route or '?'} | confidence={self.confidence:.2f}"


class TurnView(_Bag):
    """A completed turn, rehydrated from the `dev` payload stored in the DB."""

    def __init__(self, dev: dict | None, answer: str = "") -> None:
        dev = dev or {}
        self.answer = answer or dev.get("answer", "")
        self.route = dev.get("route", "")
        self.trace = dev.get("trace", []) or []
        self.error = dev.get("error")
        self.plan = PlanView(dev["plan"]) if dev.get("plan") else None
        self.diagnostics = (
            DiagnosticsView(dev["diagnostics"]) if dev.get("diagnostics") else None
        )
        self.ocr = _Bag(dev["ocr"]) if dev.get("ocr") else None
        self.validation = (
            ValidationView(dev["validation"]) if dev.get("validation") else None
        )
        self.evidence = [_Bag(e) for e in dev.get("evidence", []) or []]
        self.intent = _Bag(dev["intent"]) if dev.get("intent") else None

        lang = dev.get("language") or {}
        self.language = LanguageView(lang) if lang else None
        self.response_language = lang.get("response_language", "en")
        self.language_switched = bool(lang.get("switched", False))
        self.response_language_corrected = bool(lang.get("corrected", False))


class ResponseTurnView(TurnView):
    """A live response from POST /api/chat (public fields + optional `dev`)."""

    def __init__(self, response: dict) -> None:
        dev = dict(response.get("dev") or {})
        # The public response always carries these even without developer mode.
        dev.setdefault("route", response.get("route", ""))
        if response.get("validation") is not None:
            dev.setdefault("validation", response["validation"])
        if response.get("language") is not None:
            dev.setdefault("language", response["language"])
        dev.setdefault("evidence", response.get("sources", []) or [])
        super().__init__(dev, answer=response.get("answer", ""))


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────


def image_to_data_uri(data: bytes, max_px: int = 320) -> str:
    """Thumbnail as a data URI for the inline attachment preview."""
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


def attachment_uri() -> str | None:
    attachment = st.session_state.get("attachment")
    if not attachment:
        return None
    try:
        data = base64.b64decode(attachment["b64"])
        return image_to_data_uri(data)
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Message rendering
# ─────────────────────────────────────────────────────────────────────────────


def render_user_message(message: dict) -> None:
    st.markdown(message.get("content", ""))
    if message.get("image_uri"):
        st.markdown(
            f'<img class="chat-image" src="{message["image_uri"]}" alt="attachment">',
            unsafe_allow_html=True,
        )


def _render_language_section(turn) -> None:
    report = turn.language
    if report is None:
        return
    st.markdown("**Language**")
    lines = [
        f"- Detected: `{report.name} ({report.code})` "
        f"confidence `{report.confidence:.2f}`, method `{report.backend}`, "
        f"script `{report.script}`"
    ]
    if report.mixed:
        lines.append(f"- Code-mixing detected: `{', '.join(report.mixed_languages)}`")
    if report.reasoning:
        lines.append(f"- Reasoning: {report.reasoning}")
    if turn.response_language:
        lines.append(f"- Answered in: `{turn.response_language}`")
    if turn.language_switched:
        lines.append("- Language switch handled; earlier context retained")
    if turn.response_language_corrected:
        lines.append("- Answer re-expressed to match the requested language")
    st.markdown("\n".join(lines))


def render_developer_panel(turn) -> None:
    """The engineering surface: routing, diagnostics, retrieval, validation."""
    if turn.route:
        activity = theme.activity_label(turn.route)
        icon = theme.activity_icon(turn.route)
        title = f"{icon} How this was handled — {activity}"
    else:
        title = "How this was handled"

    with st.expander(title, expanded=False):
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

        intent = turn.intent
        if intent is not None:
            st.markdown("**Intent**")
            st.markdown(
                f"- `{getattr(intent, 'intent', '')}` "
                f"(confidence `{float(getattr(intent, 'confidence', 0) or 0):.2f}`, "
                f"source `{getattr(intent, 'source', '')}`)"
            )

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
                        st.caption(f"OCR not used: {getattr(ocr, 'error', None) or 'n/a'}")

    if turn.evidence:
        with st.expander(f"Retrieved evidence ({len(turn.evidence)})", expanded=False):
            for e in turn.evidence:
                citation = getattr(e, "citation", "") or getattr(e, "source", "")
                score = getattr(e, "score", 0.0) or 0.0
                st.markdown(f"`{citation}` — score `{score}`")
                text = getattr(e, "text", "") or ""
                if text:
                    st.caption(text[:220] + ("…" if len(text) > 220 else ""))

    report = turn.validation
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


def render_assistant_message(message: dict) -> None:
    """The clean answer surface: text, one trust pill, and source chips."""
    st.markdown(theme.clean_answer(message.get("content", "")))

    dev = message.get("dev") or {}
    if not dev:
        return
    turn = TurnView(dev, answer=message.get("content", ""))

    notice = theme.quality_notice(turn.diagnostics)
    if notice:
        st.caption(f"ℹ️ {notice}")

    pill = theme.confidence_pill(turn.validation)
    if pill:
        st.markdown(f'<div style="margin-top:0.7rem">{pill}</div>', unsafe_allow_html=True)

    chips = theme.source_chips(turn.evidence)
    if chips:
        st.markdown(
            f'<div style="margin-top:0.35rem;font-size:0.72rem;'
            f'color:{theme.COLORS["text_muted"]}">Based on</div>{chips}',
            unsafe_allow_html=True,
        )

    badge = theme.language_badge(turn)
    if badge:
        st.markdown(badge, unsafe_allow_html=True)

    if st.session_state.get("dev_mode", False):
        render_developer_panel(turn)
