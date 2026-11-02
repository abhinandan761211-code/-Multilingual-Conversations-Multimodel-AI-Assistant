"""
api_client.py
=============
The front-end's only way to reach the backend.

Keeping every HTTP call in this one file means the Streamlit pages never build
URLs or parse error bodies, and swapping the UI (or the API) later is a change
in one place rather than twenty.
"""
from __future__ import annotations

import base64
import os
from typing import Any

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api").rstrip("/")

# Vision inference on a laptop can take a minute or two on the first call
# (model load), so the chat timeout is generous by design.
CHAT_TIMEOUT = int(os.getenv("API_CHAT_TIMEOUT", "300"))
DEFAULT_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))


class ApiError(RuntimeError):
    """A failed backend call, with a message fit to show a user."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _request(method: str, path: str, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> Any:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.ConnectionError as exc:
        raise ApiError(
            f"Can't reach the assistant service at {API_BASE_URL}. "
            "Is the backend running?  (make api)"
        ) from exc
    except requests.Timeout as exc:
        raise ApiError(
            "The assistant took too long to respond. The local model may still "
            "be loading — try again."
        ) from exc

    if response.status_code >= 400:
        detail = ""
        try:
            detail = response.json().get("detail", "")
        except ValueError:
            detail = response.text[:200]
        raise ApiError(
            detail or f"Request failed ({response.status_code}).",
            status_code=response.status_code,
        )

    if response.status_code == 204 or not response.content:
        return None
    return response.json()


# ── system ────────────────────────────────────────────────────────────────────


def health() -> dict:
    return _request("GET", "/health", timeout=5)


def system_status() -> dict:
    return _request("GET", "/system/status")


def list_documents() -> list[dict]:
    return _request("GET", "/documents").get("documents", [])


def reindex() -> dict:
    return _request("POST", "/documents/reindex", timeout=180)


# ── conversations ─────────────────────────────────────────────────────────────


def list_conversations(limit: int = 100) -> dict:
    return _request("GET", "/conversations", params={"limit": limit})


def get_conversation(conversation_id: str) -> dict:
    return _request("GET", f"/conversations/{conversation_id}")


def rename_conversation(conversation_id: str, title: str) -> dict:
    return _request(
        "PATCH", f"/conversations/{conversation_id}", json={"title": title}
    )


def delete_conversation(conversation_id: str) -> None:
    _request("DELETE", f"/conversations/{conversation_id}")


def get_image_b64(conversation_id: str) -> dict | None:
    """The pinned attachment as base64, or None when there isn't one."""
    try:
        return _request("GET", f"/conversations/{conversation_id}/image/base64")
    except ApiError as exc:
        if exc.status_code == 404:
            return None
        raise


def detach_image(conversation_id: str) -> None:
    _request("DELETE", f"/conversations/{conversation_id}/image")


# ── chat ──────────────────────────────────────────────────────────────────────


def send_message(
    message: str,
    conversation_id: str | None = None,
    image: tuple[str, bytes] | None = None,
    developer_mode: bool = False,
) -> dict:
    payload: dict[str, Any] = {
        "message": message,
        "developer_mode": developer_mode,
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if image is not None:
        name, data = image
        payload["image_name"] = name
        payload["image_b64"] = base64.b64encode(data).decode("ascii")

    return _request("POST", "/chat", timeout=CHAT_TIMEOUT, json=payload)
