"""
sidebar.py
==========
New chat, history list, system status and the developer-mode switch.

Every value shown here comes from the API (`/system/status`,
`/conversations`) — the sidebar holds no business state of its own.
"""
from __future__ import annotations

import streamlit as st

from frontend import api_client


def _status_row(label: str, ok: bool, warn: bool = False) -> str:
    cls = "dot-ok" if ok else ("dot-warn" if warn else "dot-off")
    return (
        f'<div class="status-row"><span class="status-dot {cls}"></span>'
        f"{label}</div>"
    )


@st.cache_data(ttl=15, show_spinner=False)
def _system_status() -> dict | None:
    """Cached briefly: the sidebar renders on every rerun, Ollama does not."""
    try:
        return api_client.system_status()
    except api_client.ApiError:
        return None


def _history() -> None:
    st.markdown('<div class="side-label">Recent</div>', unsafe_allow_html=True)
    try:
        listing = api_client.list_conversations()
    except api_client.ApiError as exc:
        st.caption(str(exc))
        return

    entries = listing.get("items", [])
    if not entries:
        st.caption("No saved conversations yet.")
        return

    active_id = st.session_state.get("conversation_id")

    for entry in entries:
        cid = entry["conversation_id"]
        is_active = cid == active_id
        label = entry.get("title") or "Untitled"

        # Title takes the space, the delete control sits beside it. No `help=`
        # on either button: Streamlit renders the tooltip target as its own
        # element, which detaches the box from the row.
        col_title, col_del = st.columns([0.84, 0.16], vertical_alignment="center")
        with col_title:
            if st.button(
                label,
                key=f"open_{cid}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                if not is_active:
                    _open(cid)
        with col_del:
            if st.button("×", key=f"del_{cid}", use_container_width=True):
                try:
                    api_client.delete_conversation(cid)
                except api_client.ApiError as exc:
                    st.warning(str(exc))
                if is_active:
                    start_new_conversation()
                st.rerun()


def _open(conversation_id: str) -> None:
    """Load a stored conversation into the session."""
    try:
        conversation = api_client.get_conversation(conversation_id)
    except api_client.ApiError as exc:
        st.warning(str(exc))
        return

    image = api_client.get_image_b64(conversation_id)
    st.session_state.conversation_id = conversation_id
    st.session_state.conversation_title = conversation.get("title", "")
    st.session_state.messages = conversation.get("messages", [])
    st.session_state.attachment = (
        {"name": image.get("name") or "image", "b64": image.get("image_b64")}
        if image
        else None
    )
    st.session_state.pending_prompt = None


def start_new_conversation() -> None:
    """Reset to an empty chat. The row is only written once a turn is sent."""
    st.session_state.conversation_id = None
    st.session_state.conversation_title = ""
    st.session_state.messages = []
    st.session_state.attachment = None
    st.session_state.pending_prompt = None
    st.session_state.previous_validations = {}


def _status_panel(status: dict | None) -> None:
    st.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)

    if status is None:
        st.markdown(
            _status_row("Assistant service unreachable", False), unsafe_allow_html=True
        )
        st.code("make api", language="bash")
        return

    ollama = status.get("ollama", {})
    kb = status.get("knowledge_base", {})
    docs = len(kb.get("documents", []) or [])

    rows = [
        _status_row(
            "Vision model ready" if ollama.get("vision_ready") else "Vision model unavailable",
            bool(ollama.get("vision_ready")),
        ),
        _status_row("Language model ready", bool(ollama.get("text_ready"))),
        _status_row(
            f"Knowledge base · {docs} documents", docs > 0, warn=docs == 0
        ),
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)

    if not ollama.get("server_up"):
        st.warning("The local model service isn't running.", icon="⚠️")
        st.code("ollama serve", language="bash")
    elif not ollama.get("vision_ready"):
        st.warning("The vision model hasn't been downloaded yet.", icon="⚠️")
        st.code(f"ollama pull {ollama.get('vision_model')}", language="bash")


def _developer_options(status: dict | None) -> None:
    st.markdown('<div class="side-label">Options</div>', unsafe_allow_html=True)
    # Deliberately st.checkbox, not st.toggle: Streamlit renders a toggle's
    # <label> as a filled bordered container, which drew a grey pill behind the
    # text. The stylesheet restyles st.checkbox from scratch instead.
    st.session_state["dev_mode"] = st.checkbox(
        "Developer mode", value=st.session_state.get("dev_mode", False)
    )
    st.caption("Show how each answer was decided.")

    if not st.session_state["dev_mode"]:
        return

    with st.expander("Configuration"):
        if status is None:
            st.caption("Backend unreachable.")
            return
        ollama = status.get("ollama", {})
        kb = status.get("knowledge_base", {})
        st.caption("Vision model")
        st.code(ollama.get("vision_model", "—"), language=None)
        st.caption("Language model")
        st.code(ollama.get("text_model", "—"), language=None)
        st.caption("Retrieval")
        st.code(
            f"{kb.get('backend', '—')} · {kb.get('chunks', 0)} chunks", language=None
        )
        st.caption("Database")
        st.code("SQLite · data/assistant.db", language=None)
        if st.button("Reindex documents", use_container_width=True):
            try:
                result = api_client.reindex()
                _system_status.clear()
                st.success(f"Reindexed ({result.get('backend')}).")
            except api_client.ApiError as exc:
                st.error(str(exc))


def render() -> None:
    with st.sidebar:
        if st.button("＋  New chat", use_container_width=True, type="primary"):
            start_new_conversation()
            st.rerun()

        _history()
        status = _system_status()
        _status_panel(status)
        _developer_options(status)
