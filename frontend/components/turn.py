"""
turn.py
=======
Handle one user turn against the API.

The front-end used to call the pipeline directly, which meant the UI process
owned model state. Now the API owns it: this module only decides *what* to send
and how to show the result.
"""
from __future__ import annotations

import streamlit as st

from frontend import api_client
from frontend.components import chat_view


def _append(message: dict) -> None:
    st.session_state.messages.append(message)


def handle_turn(
    prompt: str | None,
    upload: tuple[str, bytes] | None = None,
) -> None:
    """
    Send one turn.

    Two ways in:
      * `upload` — the user attached an image. Adopt it, then analyse it (with
        their text if they typed any) so the first reply is useful.
      * `prompt` — a normal message, answered against the pinned image (if any)
        or from text and the reference documents.
    """
    typed = (prompt or "").strip()
    if not typed and upload is None:
        return

    # Optimistic UI: the user's message appears before the model answers.
    display_text = typed or (f"[Attached {upload[0]}]" if upload else "")
    just_attached = upload is not None
    already_shown = any(m.get("image_uri") for m in st.session_state.messages)
    show_image = just_attached or not already_shown

    image_uri = None
    if just_attached:
        st.session_state.attachment = {
            "name": upload[0],
            "b64": __import__("base64").b64encode(upload[1]).decode("ascii"),
        }
    if show_image:
        image_uri = chat_view.attachment_uri() if st.session_state.attachment else None

    user_message = {"role": "user", "content": display_text}
    if image_uri:
        user_message["image_uri"] = image_uri
    _append(user_message)

    with st.chat_message("user", avatar=None):
        chat_view.render_user_message(user_message)

    with st.chat_message("assistant", avatar=None):
        placeholder = st.empty()
        placeholder.caption("Reading your image…" if upload or st.session_state.attachment else "Thinking…")

        try:
            response = api_client.send_message(
                message=typed,
                conversation_id=st.session_state.get("conversation_id"),
                image=upload,
                developer_mode=bool(st.session_state.get("dev_mode", False)),
            )
        except api_client.ApiError as exc:
            placeholder.empty()
            st.error(str(exc))
            return

        placeholder.empty()

    st.session_state.conversation_id = response.get("conversation_id")
    st.session_state.conversation_title = response.get("conversation_title", "")

    assistant_message = {
        "role": "assistant",
        "content": response.get("answer", ""),
        # Always populated: the trust pill, source chips and language badge are
        # user-facing, not developer telemetry.
        "dev": _dev_for_storage(response),
    }
    _append(assistant_message)
    chat_view.render_assistant_message(assistant_message)


def _dev_for_storage(response: dict) -> dict:
    """The subset persisted with a message so a reload can re-render the turn."""
    dev = dict(response.get("dev") or {})
    dev.setdefault("route", response.get("route", ""))
    dev.setdefault("answer", response.get("answer", ""))
    if response.get("validation") is not None:
        dev.setdefault("validation", response["validation"])
    if response.get("language") is not None:
        dev.setdefault("language", response["language"])
    if response.get("sources"):
        dev.setdefault("evidence", response["sources"])
    return dev
