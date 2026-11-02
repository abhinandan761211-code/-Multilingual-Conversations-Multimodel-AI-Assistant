"""
Presentation helpers used by the front-end.

The trust pill, source chips and language badge are produced from the same
validation/language objects the API already returns, but rendering them
server-side keeps the HTML in one place and the client thin.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["widgets"])


class WidgetRequest(BaseModel):
    validation: dict | None = None
    language: dict | None = None
    sources: list[dict] = Field(default_factory=list)


@router.post("/widgets/trust")
def trust_widget(payload: WidgetRequest) -> dict:
    """Human sentence describing how much to trust an answer."""
    report = payload.validation or {}
    status = (report.get("status") or "pass").lower()
    confidence = float(report.get("confidence") or 0.0)

    if status == "pass":
        label = "Well supported"
        tone = "ok"
    elif status == "warn":
        label = "Mostly supported"
        tone = "warn"
    else:
        label = "Low confidence"
        tone = "danger"

    warnings = report.get("warnings") or []
    detail = warnings[0] if warnings else ""
    return {"label": label, "tone": tone, "confidence": confidence, "detail": detail}
