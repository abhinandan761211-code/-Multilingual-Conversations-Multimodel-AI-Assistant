"""
composer.py
===========
The pinned-attachment bar plus the message input.

Streamlit pins `st.chat_input` to a bottom dock, so no widget can share a row
with it. The attachment control therefore lives in the transcript area directly
above the composer, reading as part of it.
"""
from __future__ import annotations

import streamlit as st

from frontend import api_client, theme
from frontend.components import chat_view


def render_attachment_bar(name: str, uri: str) -> None:
    """ChatGPT-style: the image belongs to the conversation, not one message."""
    st.markdown(
        '<div class="attach-bar">'
        f'<img src="{uri}" alt="attached image">'
        '<div class="meta">'
        f'<div class="name">{name}</div>'
        '<div class="hint">Attached to this chat — keep asking about it.</div>'
        "</div>"
        '<span class="tag">Active</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    try:
        status = api_client.system_status()
        ready = bool(status.get("ready"))
    except api_client.ApiError:
        ready = False

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
        f'color:{theme.COLORS["text_muted"]}">'
        "Upload an image, or start with a question."
        "</div>",
        unsafe_allow_html=True,
    )

    # Inline SVG, not emoji: the font stack on this machine renders emoji as
    # broken/boxed glyphs inside these cards.
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

    st.markdown(
        f'<div style="margin:0.2rem 0 0.9rem 0;font-size:0.8rem;'
        f'color:{theme.COLORS["text_muted"]}">'
        "Ask in <b>English</b>, <b>हिन्दी</b>, <b>ଓଡ଼ିଆ</b> or <b>বাংলা</b> — or mix "
        "them. The assistant detects the language automatically, keeps the "
        "context when you switch, and answers in the language you used."
        "</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(examples))
    for col, (icon, title, body) in zip(cols, examples, strict=False):
        with col:
            st.markdown(
                f'<div style="background:{theme.COLORS["surface"]};'
                f'border:1px solid {theme.COLORS["border"]};border-radius:10px;'
                f'padding:0.85rem 0.9rem;min-height:110px;'
                f'display:flex;flex-direction:column;gap:0.3rem">'
                f'<div style="font-size:1rem;line-height:1.1">{icon}</div>'
                f'<div style="font-weight:600;font-size:0.84rem;line-height:1.3;'
                f'color:{theme.COLORS["text"]}">{title}</div>'
                f'<div style="font-size:0.76rem;color:{theme.COLORS["text_muted"]};'
                f'line-height:1.5">{body}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


def render_composer() -> None:
    """Attachment control, then the chat input. Both feed the same turn handler."""
    conversation_id = st.session_state.get("conversation_id")
    attachment = st.session_state.get("attachment")
    uploaded = None

    with st.container():
        if attachment:
            uri = chat_view.attachment_uri()
            if uri:
                render_attachment_bar(attachment.get("name") or "image", uri)
            detach_col, _ = st.columns([0.18, 0.82], vertical_alignment="center")
            with detach_col:
                if st.button("Remove image", key="detach_image", use_container_width=True):
                    if conversation_id:
                        try:
                            api_client.detach_image(conversation_id)
                        except api_client.ApiError as exc:
                            st.warning(str(exc))
                    st.session_state.attachment = None
                    st.session_state.previous_validations = {}
                    st.rerun()
        else:
            up_col, _ = st.columns([0.18, 0.82], vertical_alignment="center")
            with up_col:
                uploaded = st.file_uploader(
                    "Add an image",
                    type=["png", "jpg", "jpeg", "webp", "bmp", "gif"],
                    label_visibility="collapsed",
                    key="image_uploader",
                )

    prompt = st.chat_input("Message the assistant…")

    # An upload takes priority: adopt it, then ask about it.
    if uploaded is not None:
        from frontend.components.turn import handle_turn

        handle_turn(
            prompt=prompt,
            upload=(uploaded.name, uploaded.getvalue()),
        )
        st.rerun()

    if prompt:
        from frontend.components.turn import handle_turn

        handle_turn(prompt=prompt)
