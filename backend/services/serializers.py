"""
serializers.py
==============
Turn the engine's dataclasses into JSON-safe dicts.

The API returns one shape for a completed turn; the detail level is chosen by
``developer_mode``. Keeping this in one place means the front-end never has to
know about `AssistantTurn`, `RoutePlan` or `ValidationReport` — which is also
why the UI can be replaced (or another client written) without touching the
engine.
"""
from __future__ import annotations

from typing import Any


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def evidence_to_dict(item: Any) -> dict:
    """One retrieved passage, with a citation string for the UI chips."""
    try:
        citation = item.citation()
    except Exception:  # noqa: BLE001
        citation = _get(item, "source", "")
    return {
        "text": _get(item, "text", "")[:600],
        "score": round(float(_get(item, "score", 0.0) or 0.0), 3),
        "citation": citation,
        "source": _get(item, "source", ""),
    }


def validation_to_dict(report: Any) -> dict | None:
    if report is None:
        return None
    return {
        "status": _enum_value(_get(report, "status", "pass")),
        "confidence": round(float(_get(report, "confidence", 0.0) or 0.0), 2),
        "checks": [
            {
                "name": _get(c, "name", ""),
                "passed": bool(_get(c, "passed", False)),
                "detail": _get(c, "detail", ""),
            }
            for c in _get(report, "checks", []) or []
        ],
        "warnings": list(_get(report, "warnings", []) or []),
    }


def diagnostics_to_dict(diag: Any) -> dict | None:
    if diag is None:
        return None
    return {
        "valid": bool(_get(diag, "valid", False)),
        "error": _get(diag, "error"),
        "width": _get(diag, "width", 0),
        "height": _get(diag, "height", 0),
        "blur_score": round(float(_get(diag, "blur_score", 0.0) or 0.0), 1),
        "mean_brightness": round(
            float(_get(diag, "mean_brightness", 0.0) or 0.0), 1
        ),
        "looks_like_text": bool(_get(diag, "looks_like_text", False)),
        "quality_notes": list(_get(diag, "quality_notes", []) or []),
    }


def language_to_dict(turn: Any) -> dict | None:
    report = _get(turn, "language")
    if report is None:
        return None
    return {
        "code": _get(report, "code", "en"),
        "name": _get(report, "name", ""),
        "confidence": round(float(_get(report, "confidence", 0.0) or 0.0), 2),
        "script": _get(report, "script", ""),
        "backend": _get(report, "backend", ""),
        "reasoning": _get(report, "reasoning", ""),
        "mixed": bool(_get(report, "mixed", False)),
        "mixed_languages": list(_get(report, "mixed_languages", []) or []),
        "supported": bool(
            _get(report, "is_supported", lambda: True)()
            if report is not None
            else True
        ),
        "switched": bool(_get(turn, "language_switched", False)),
        "response_language": _get(turn, "response_language", "en"),
        "corrected": bool(_get(turn, "response_language_corrected", False)),
    }


def turn_to_dev(turn: Any) -> dict:
    """
    The developer-mode payload: routing, plan, diagnostics, OCR, evidence and
    validation. This is exactly what the UI persists per message, so a reloaded
    conversation can still explain how each answer was produced.
    """
    if turn is None:
        return {}

    dev: dict[str, Any] = {
        "answer": _get(turn, "answer", ""),
        "route": _get(turn, "route", ""),
        "trace": list(_get(turn, "trace", []) or [])[-14:],
        "error": _get(turn, "error"),
    }

    plan = _get(turn, "plan")
    if plan is not None:
        dev["plan"] = {
            "route": _enum_value(_get(plan, "route", "")),
            "confidence": round(float(_get(plan, "confidence", 0.0) or 0.0), 2),
            "rationale": list(_get(plan, "rationale", []) or []),
            "use_vision": bool(_get(plan, "use_vision", False)),
            "use_ocr": bool(_get(plan, "use_ocr", False)),
            "use_retrieval": bool(_get(plan, "use_retrieval", False)),
            "trust_ocr": bool(_get(plan, "trust_ocr", True)),
        }

    diag = diagnostics_to_dict(_get(turn, "diagnostics"))
    if diag is not None and diag.get("valid"):
        dev["diagnostics"] = diag

    ocr = _get(turn, "ocr")
    if ocr is not None and _get(ocr, "attempted", False) and _get(
        ocr, "succeeded", False
    ):
        dev["ocr"] = {
            "backend": _get(ocr, "backend", ""),
            "confidence": round(float(_get(ocr, "confidence", 0.0) or 0.0), 2),
            "text": (_get(ocr, "text", "") or "")[:1200],
            "attempted": True,
            "succeeded": True,
        }

    intent = _get(turn, "intent")
    if intent is not None:
        dev["intent"] = {
            "intent": _get(intent, "intent", ""),
            "entities": list(_get(intent, "entities", []) or []),
            "needs_clarification": bool(_get(intent, "needs_clarification", False)),
            "response_language": _get(intent, "response_language", ""),
            "confidence": round(float(_get(intent, "confidence", 0.0) or 0.0), 2),
            "source": _get(intent, "source_of_resolution", "heuristic"),
        }

    language = language_to_dict(turn)
    if language is not None:
        dev["language"] = language

    dev["evidence"] = [
        evidence_to_dict(e) for e in (_get(turn, "evidence", []) or [])
    ]

    validation = validation_to_dict(_get(turn, "validation"))
    if validation is not None:
        dev["validation"] = validation

    return dev


def turn_to_response(turn: Any, developer_mode: bool = False) -> dict:
    """The public per-turn payload returned by POST /api/chat."""
    payload: dict[str, Any] = {
        "answer": _get(turn, "answer", ""),
        "route": _get(turn, "route", ""),
        "error": _get(turn, "error"),
        # Always returned: the user-facing trust signal and source chips.
        "validation": validation_to_dict(_get(turn, "validation")),
        "sources": [
            {
                "citation": e["citation"],
                "score": e["score"],
                "source": e["source"],
            }
            for e in (evidence_to_dict(x) for x in (_get(turn, "evidence", []) or []))
        ],
        "language": language_to_dict(turn),
    }
    if developer_mode:
        payload["dev"] = turn_to_dev(turn)
    return payload
