"""
app.py — Streamlit front-end for the Multimodal AI Assistant.

Presentation philosophy
-----------------------
The engineering surface (route names, plan strings, backend identifiers,
similarity scores, OCR confidence) is useful while building and noise while
using. There are therefore two deliberately separate surfaces:

  * **User view (default)** — the answer, a plain-language trust indicator, and
    which source documents informed it.
  * **Developer view (opt-in)** — the full reasoning trace, image diagnostics,
    OCR output, retrieval scores and validation checks.

Architecture
------------
This file is *only* the page: session bootstrap and layout. It holds no model
state and no business logic. Every answer comes from the FastAPI backend
(`make api`), so the UI can be restarted without reloading models, and other
clients can use the same API.

Run (from the repository root):
    streamlit run frontend/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run frontend/app.py` from the repo root: make the repo root
# importable so `frontend.*` and `backend.*` resolve from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend import api_client, theme  # noqa: E402
from frontend.components import chat_view, composer, sidebar  # noqa: E402

st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject()


def _restore_session() -> None:
    """
    Initialise session state for a fresh browser session.

    Resumes the most recent saved conversation instead of always starting a
    blank one — otherwise every page load created another empty chat and the
    history list filled with duplicates.
    """
    if "messages" in st.session_state:
        return

    st.session_state.messages = []
    st.session_state.conversation_id = None
    st.session_state.attachment = None
    st.session_state.previous_validations = {}

    try:
        entries = api_client.list_conversations(limit=1).get("items", [])
    except api_client.ApiError:
        entries = []

    if entries:
        sidebar._open(entries[0]["conversation_id"])


def main() -> None:
    _restore_session()

    sidebar.render()
    composer.render_header()

    messages = st.session_state.messages

    if not messages:
        composer.render_empty_state()
        # The empty state is short, so the docked composer floats up and crowds
        # the example cards. A modest spacer pushes it back down.
        st.markdown('<div style="height:2.5rem"></div>', unsafe_allow_html=True)

    for message in messages:
        with st.chat_message(message["role"], avatar=None):
            if message["role"] == "assistant":
                chat_view.render_assistant_message(message)
            else:
                chat_view.render_user_message(message)

    # The composer is a fixed overlay that occupies no layout space, so the last
    # message needs clearance or it sits underneath the dock (~150px tall).
    if messages:
        st.markdown('<div style="height:13rem"></div>', unsafe_allow_html=True)

    composer.render_composer()


if __name__ == "__main__":
    main()
