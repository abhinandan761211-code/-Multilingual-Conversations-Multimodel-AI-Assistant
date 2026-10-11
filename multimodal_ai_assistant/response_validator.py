"""
response_validator.py
=====================
Post-generation checks. The assistant does not ship a model answer verbatim —
it inspects the answer against the evidence it actually had, and either passes
it, qualifies it, or downgrades it to an honest "I'm not sure".

Checks implemented (mirroring the assignment's validation checklist):

  V1  Answers the question      — does the reply address what was asked?
  V2  Grounded in evidence      — claims must trace to the image / documents.
  V3  Numeric fidelity          — numbers in the answer appear in the evidence.
  V4  Uncertainty when unclear  — blurry/ambiguous inputs must be flagged.
  V5  Format & safety           — length, structure, no obvious refusal-hacks.
  V6  No fabrication            — detects invented proper nouns / values.
  V7  Response language         — the answer is in the language the user needs.

Design notes
------------
* Wholly deterministic checks (V3, V4, V6) always run — no model needed.
* Semantic checks (V1, V2) use embeddings when available and fall back to
  lexical overlap, so validation never silently disappears.
* The output is a `ValidationReport` with a status the UI can surface:
  PASS / QUALIFIED / NEEDS_CLARIFICATION / FAILED.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from pydantic import BaseModel, Field

import config
from image_processor import ImageDiagnostics
from reasoning_router import Route, RoutePlan


# ─────────────────────────────────────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────────────────────────────────────


class ValidationStatus(str, Enum):
    PASS = "pass"
    QUALIFIED = "qualified"                 # usable, but with caveats
    NEEDS_CLARIFICATION = "needs_clarification"
    FAILED = "failed"


class CheckResult(BaseModel):
    name: str
    passed: bool
    score: float = 0.0
    detail: str = ""


class ValidationReport(BaseModel):
    status: ValidationStatus = ValidationStatus.PASS
    checks: list[CheckResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    final_answer: str = ""
    uncertainty_note: Optional[str] = None

    def add(self, name: str, passed: bool, score: float, detail: str = "") -> None:
        self.checks.append(
            CheckResult(name=name, passed=passed, score=round(score, 3), detail=detail)
        )

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def as_badge(self) -> str:
        icon = {
            ValidationStatus.PASS: "✅",
            ValidationStatus.QUALIFIED: "⚠️",
            ValidationStatus.NEEDS_CLARIFICATION: "❓",
            ValidationStatus.FAILED: "⛔",
        }[self.status]
        return f"{icon} {self.status.value}"


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "to",
    "of", "in", "on", "at", "by", "for", "with", "about", "as", "into", "and",
    "or", "but", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "i", "you", "we", "they", "he", "she", "what", "which", "who",
    "how", "why", "when", "where", "do", "does", "did", "can", "could",
    "would", "should", "will", "may", "might", "from", "have", "has", "had",
}

# Stems, matched as substrings so morphology is handled: "blurr" covers
# blurry / blurriness / blurred; "indistinct" covers indistinctly, etc.
_HEDGE_STEMS = (
    "approximat", "roughly", "appear", "seem", "likely", "possibl",
    "unclear", "uncertain", "might", "estimate", "around", "about",
    "cannot", "can't", "unable", "not sure", "blurr", "illegible",
    "indistinct", "obscur", "unreadable", "hard to tell", "difficult to",
    "not possible", "definitively", "insufficient", "not enough information",
    "no clear", "ambiguous", "unsure", "tentative", "lower bound",
    "at least", "approximately", "give or take", "best guess",
)

# Retained for readability / backwards compatibility in tests.
_HEDGE_WORDS = set(_HEDGE_STEMS)

_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b")

# Words that are fine to start a sentence with and shouldn't count as "invented
# entity" evidence.
_SENTENCE_STARTERS = {
    "The", "This", "That", "These", "Those", "There", "It", "In", "On", "At",
    "From", "Based", "However", "According", "Note", "Yes", "No", "If", "When",
    "While", "Both", "Each", "First", "Second", "Third", "Finally", "Also",
    "Here", "I", "You", "We", "They", "He", "She", "As", "For", "To", "Of",
    "With", "Without", "Because", "Since", "But", "And", "Or", "So", "Your",
    "My", "Our", "Their", "Its", "A", "An", "User", "Assistant", "Image",
    "Visual", "Overall", "Looking", "Given", "Using", "Regarding", "Once",
}

# A *capability* refusal — the assistant declining because it believes it
# fundamentally cannot do the task. This is different from legitimate hedging
# ("I cannot confirm the exact value from this blurry image, but..."), which we
# actively want. So these are matched as phrases that indicate a blanket
# refusal, and a hedge elsewhere in the sentence clears the flag.
_REFUSAL_MARKERS = (
    "as an ai language model", "as an ai, i", "i don't have access to",
    "i do not have access to", "i'm not able to help", "i am not able to help",
    "i cannot help with that", "i can't help with that",
    "i cannot assist with that", "i'm unable to assist",
)

# If the sentence containing a refusal marker also contains one of these, it is
# a scoped hedge rather than a blanket refusal.
_SCOPED_HEDGE_MARKERS = (
    "but", "however", "from this", "in this image", "i can see",
    "based on", "clearly", "identify", "appears",
)


def _tokens(text: str) -> set[str]:
    """
    Content-bearing word tokens, script-agnostic.

    The original pattern was ASCII-only (`[a-z0-9']+`), which meant a Hindi,
    Odia or Bengali answer produced *zero* tokens — so its relevance score was
    always 0 and every correct multilingual turn looked irrelevant. Using a
    Unicode word class makes the lexical checks work in every supported script.
    """
    return {
        w for w in re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        if w not in _STOPWORDS and len(w) > 2
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _coverage(question_terms: set[str], answer: set[str]) -> float:
    """
    Fraction of the *question's* meaningful terms that appear in the answer.

    This is the right measure for "does a long answer address a short question".
    Jaccard punishes length (a 200-word answer about a 6-word question scores
    near zero even when it covers every term), so we use containment instead.
    """
    if not question_terms:
        return 0.0
    return len(question_terms & answer) / len(question_terms)


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ValidationContext:
    """Everything the validator is allowed to treat as evidence."""

    user_message: str = ""
    plan: Optional[RoutePlan] = None
    diagnostics: Optional[ImageDiagnostics] = None
    image_summary: str = ""                 # VLM's own description of the image
    ocr_text: str = ""                      # OCR transcript
    vlm_text: str = ""                      # VLM text-extraction transcript
    evidence: list = field(default_factory=list)   # evidence_retriever.Evidence
    conversation_context: str = ""
    ocr_was_untrusted: bool = False
    # The language the answer is *required* to be in. Set by the pipeline from
    # the normalised intent; used by the V7 language check.
    response_language: str = "en"

    def evidence_blob(self) -> str:
        parts = [self.image_summary, self.ocr_text, self.vlm_text,
                 self.conversation_context]
        parts += [getattr(e, "text", str(e)) for e in self.evidence]
        return "\n".join(p for p in parts if p)


class ResponseValidator:
    """
    Runs the V1–V6 checks and produces a `ValidationReport`.

    `embedder` is optional: pass a sentence-transformers model to upgrade the
    semantic checks; otherwise lexical overlap is used.
    """

    def __init__(self, embedder=None, entailment_checker=None) -> None:
        # entailment_checker: callable(claim, evidence) -> {"supported": bool, ...}
        # normally VisionModel.check_entailment. Optional.
        self._embedder = embedder
        self._entailment = entailment_checker

    # ── public API ──────────────────────────────────────────────────────────

    def validate(self, answer: str, context: ValidationContext) -> ValidationReport:
        report = ValidationReport(final_answer=answer.strip())

        if not answer or not answer.strip():
            report.status = ValidationStatus.FAILED
            report.add("V0-nonempty", False, 0.0, "Model returned an empty response")
            report.warnings.append("The model produced no answer.")
            report.confidence = 0.0
            return report

        answer = answer.strip()

        # ── V5 format & safety (cheap, first) ───────────────────────────────
        self._check_format(answer, report)

        # ── V1 answers the question ─────────────────────────────────────────
        self._check_relevance(answer, context, report)

        # ── V2 grounded in evidence ─────────────────────────────────────────
        self._check_grounding(answer, context, report)

        # ── V3 numeric fidelity ─────────────────────────────────────────────
        self._check_numbers(answer, context, report)

        # ── V6 no fabrication of entities ───────────────────────────────────
        self._check_fabrication(answer, context, report)

        # ── V4 uncertainty for unclear inputs ───────────────────────────────
        self._check_uncertainty(answer, context, report)

        # ── optional model-based entailment spot-check ──────────────────────
        self._check_entailment_spotcheck(answer, context, report)

        # ── V7 answered in the required language ────────────────────────────
        self._check_language(answer, context, report)

        self._decide_status(report, context)
        return report

    # ── individual checks ───────────────────────────────────────────────────

    def _check_format(self, answer: str, report: ValidationReport) -> None:
        words = len(answer.split())
        if words < 3:
            report.add("V5-format", False, 0.2, f"Answer is only {words} words")
            report.warnings.append("Answer is suspiciously short.")
        elif words > 900:
            report.add("V5-format", False, 0.6, f"Answer is very long ({words} words)")
            report.warnings.append("Answer is unusually long; may be padded.")
        else:
            report.add("V5-format", True, 1.0, f"{words} words")

        lowered = answer.lower()
        refusal_found = None
        for marker in _REFUSAL_MARKERS:
            if marker not in lowered:
                continue
            # Look at the sentence(s) around the marker: if the model goes on to
            # actually answer ("...but I can see three people"), treat it as a
            # scoped hedge, not a refusal.
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", lowered) if marker in s]
            scoped = sentences and all(
                any(h in s for h in _SCOPED_HEDGE_MARKERS) for s in sentences
            )
            if not scoped:
                refusal_found = marker
                break

        if refusal_found:
            report.add(
                "V5-safety",
                False,
                0.3,
                f"Contains a capability refusal ('{refusal_found}')",
            )
            report.warnings.append(
                "The reply contains a generic capability refusal; it may not "
                "address the user's actual question."
            )
        else:
            report.add("V5-safety", True, 1.0, "No capability refusals")

    def _semantic_similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        if self._embedder is not None:
            try:
                import numpy as np

                vecs = self._embedder.encode([a, b], show_progress_bar=False)
                denom = (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])) or 1e-9
                return float(np.dot(vecs[0], vecs[1]) / denom)
            except Exception:  # noqa: BLE001
                pass
        return _jaccard(_tokens(a), _tokens(b))

    def _check_relevance(
        self, answer: str, context: ValidationContext, report: ValidationReport
    ) -> None:
        question = context.user_message.strip()
        if not question:
            report.add("V1-relevance", True, 0.5, "No question to compare against")
            return

        # A clarification route is *supposed* to not answer the question yet.
        if context.plan and context.plan.route == Route.CLARIFY:
            report.add(
                "V1-relevance",
                True,
                1.0,
                "Clarification response; relevance measured differently",
            )
            return

        # Cross-script answers cannot be scored lexically.
        #
        # When the user asks in Hinglish (Latin) and the answer is Devanagari
        # Hindi — or vice versa — the two share no Latin tokens by construction,
        # so a lexical coverage score of 0 says nothing about relevance. This is
        # the *translation* case, which the assistant is meant to support, so
        # judging it on token overlap would fail every correct multilingual turn.
        # The semantic check still applies when an embedder is available.
        if not self._same_script(question, answer):
            similarity = self._semantic_similarity(question, answer)
            if similarity > 0.0:
                report.add(
                    "V1-relevance",
                    similarity >= 0.15,
                    similarity,
                    f"Cross-script answer; semantic={similarity:.2f} "
                    "(lexical overlap not applicable)",
                )
            else:
                report.add(
                    "V1-relevance",
                    True,
                    0.6,
                    "Cross-script answer (translation/re-expression); "
                    "lexical relevance not applicable",
                )
            return

        question_terms = _tokens(question)

        similarity = self._semantic_similarity(question, answer)
        # Term coverage: of the question's content words, how many appear in the
        # answer? Robust to the answer being much longer than the question.
        coverage = _coverage(question_terms, _tokens(answer))
        lexical = _jaccard(question_terms, _tokens(answer))

        # A long structured reply (headings, bullets, legend descriptions) can
        # dilute a whole-blob score even when one sentence answers the question
        # directly. Also score the strongest single sentence, and take the best.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        sentence_coverage = max(
            (_coverage(question_terms, _tokens(s)) for s in sentences), default=0.0
        )
        coverage = max(coverage, sentence_coverage)

        # For count questions, the presence of a numeral is strong evidence of
        # relevance even when wording differs.
        count_bonus = 0.0
        if re.search(r"\bhow many|count|number of\b", question, re.IGNORECASE):
            if _NUMBER_RE.search(answer):
                count_bonus = 0.25

        # "when/why/where" answers legitimately reuse few of the question's
        # words (the question says "when does it start" and the answer says
        # "after epoch 6"), so a partial coverage still counts as relevant.
        score = min(1.0, max(similarity, coverage, lexical * 2) + count_bonus)
        passed = score >= 0.15
        report.add(
            "V1-relevance",
            passed,
            score,
            f"semantic={similarity:.2f} coverage={coverage:.2f} "
            f"lexical={lexical:.2f}"
            + (f" count_bonus={count_bonus:.2f}" if count_bonus else ""),
        )
        if not passed:
            report.warnings.append(
                "The answer may not directly address the question asked."
            )

    @staticmethod
    def _same_script(a: str, b: str) -> bool:
        """
        True when two strings predominantly share one writing system.

        Used to detect the translation case (question in one script, answer in
        another) where lexical overlap is meaningless. Falls back to "same" when
        the script cannot be determined, so an import failure never changes the
        validator's default behaviour.
        """
        try:
            from language_detector import script_profile

            profile_a = script_profile(a)
            profile_b = script_profile(b)
        except Exception:  # noqa: BLE001
            return True
        if not profile_a or not profile_b:
            return True
        dominant_a = max(profile_a, key=lambda s: profile_a[s])
        dominant_b = max(profile_b, key=lambda s: profile_b[s])
        return dominant_a == dominant_b

    def _check_grounding(
        self, answer: str, context: ValidationContext, report: ValidationReport
    ) -> None:
        blob = context.evidence_blob()
        if not blob.strip():
            # Nothing to ground against — this is itself a red flag unless the
            # route legitimately needs no evidence.
            route = context.plan.route if context.plan else None
            if route == Route.SMALL_TALK:
                report.add("V2-grounding", True, 1.0, "Small talk needs no evidence")
            elif route == Route.CLARIFY:
                # A clarifying question makes no factual claim, so there is
                # nothing that needs grounding. Flagging it here would be noise.
                report.add(
                    "V2-grounding",
                    True,
                    1.0,
                    "Clarification asks rather than asserts — no claims to ground",
                )
            else:
                report.add(
                    "V2-grounding",
                    False,
                    0.0,
                    "No evidence available to ground the answer",
                )
                report.warnings.append(
                    "The answer was produced without any supporting evidence. "
                    "Treat it as unverified."
                )
            return

        score = self._semantic_similarity(answer, blob)
        answer_terms = _tokens(answer)
        blob_terms = _tokens(blob)
        # Grounding asks: how much of what the *answer* asserts is present in
        # the evidence? That is containment of the answer's terms in the blob,
        # not symmetric overlap — a long well-grounded answer must not be
        # penalised merely for being longer than a terse evidence summary.
        lexical = _coverage(answer_terms, blob_terms)
        combined = max(score, lexical)
        passed = combined >= config.GROUNDING_MIN_SCORE
        report.add(
            "V2-grounding",
            passed,
            combined,
            f"semantic={score:.2f} answer_term_coverage={lexical:.2f} "
            f"(threshold={config.GROUNDING_MIN_SCORE})",
        )
        if not passed:
            report.warnings.append(
                "Low overlap between the answer and the available evidence — "
                "some claims may be unsupported."
            )

    def _check_numbers(
        self, answer: str, context: ValidationContext, report: ValidationReport
    ) -> None:
        answer_numbers = set(_NUMBER_RE.findall(answer))
        if not answer_numbers:
            report.add("V3-numbers", True, 1.0, "No numeric claims to verify")
            return

        # For pure text-only general questions we have no image evidence; the
        # numeric check only applies when an image or documents are in play.
        blob = context.evidence_blob()
        if not blob.strip():
            report.add(
                "V3-numbers",
                False,
                0.3,
                f"Answer states {sorted(answer_numbers)} with no verifiable source",
            )
            report.warnings.append(
                f"Numbers {sorted(answer_numbers)} could not be traced to any "
                "visible evidence."
            )
            return

        blob_numbers = set(_NUMBER_RE.findall(blob))
        unsupported = answer_numbers - blob_numbers
        # Numbers that are trivially present everywhere (0,1,2) or clearly
        # structural (list markers) shouldn't tank the score.
        trivial = {"0", "1", "2", "3", "100"}
        meaningful_unsupported = unsupported - trivial

        if not meaningful_unsupported:
            report.add(
                "V3-numbers",
                True,
                1.0,
                f"All values {sorted(answer_numbers)} traceable to evidence",
            )
        else:
            ratio = 1 - (len(meaningful_unsupported) / len(answer_numbers))
            report.add(
                "V3-numbers",
                False,
                max(0.0, ratio),
                f"Untraceable values: {sorted(meaningful_unsupported)}",
            )
            report.warnings.append(
                "The answer contains numbers that do not appear in the visible "
                f"evidence: {sorted(meaningful_unsupported)}. Verify before trusting."
            )

    def _check_fabrication(
        self, answer: str, context: ValidationContext, report: ValidationReport
    ) -> None:
        blob = context.evidence_blob()
        if not blob.strip():
            report.add("V6-fabrication", True, 0.5, "No evidence to compare entities against")
            return

        blob_lower = blob.lower()
        suspicious: list[str] = []
        for match in _PROPER_NOUN_RE.finditer(answer):
            entity = match.group(1)
            head = entity.split()[0]
            if head in _SENTENCE_STARTERS:
                continue
            if entity.lower() not in blob_lower:
                suspicious.append(entity)

        # Only flag when the entity is a *new* name the evidence never mentions.
        unique_suspicious = sorted(set(suspicious))
        if not unique_suspicious:
            report.add("V6-fabrication", True, 1.0, "No invented entities detected")
        else:
            score = max(0.0, 1 - len(unique_suspicious) / 6)
            report.add(
                "V6-fabrication",
                False,
                score,
                f"Entities not present in evidence: {unique_suspicious[:5]}",
            )
            report.warnings.append(
                "Possibly fabricated details: "
                + ", ".join(unique_suspicious[:5])
                + ". These were not found in the image text or retrieved documents."
            )

    def _check_uncertainty(
        self, answer: str, context: ValidationContext, report: ValidationReport
    ) -> None:
        diag = context.diagnostics
        unclear = bool(
            diag and (diag.is_blurry or diag.is_low_resolution or diag.is_too_dark)
        ) or context.ocr_was_untrusted

        if not unclear:
            report.add("V4-uncertainty", True, 1.0, "Image quality adequate")
            return

        lowered = answer.lower()
        matched = [stem for stem in _HEDGE_STEMS if stem in lowered]
        hedged = bool(matched)
        if hedged:
            report.add(
                "V4-uncertainty",
                True,
                1.0,
                "Answer appropriately flags uncertainty given image quality "
                f"(markers: {', '.join(matched[:4])})",
            )
        else:
            report.add(
                "V4-uncertainty",
                False,
                0.4,
                "Image quality is poor but the answer states no caveats",
            )
            report.warnings.append(
                "The image was low-quality/unclear, yet the answer is stated with "
                "full confidence. Uncertainty should have been reported."
            )

    def _check_entailment_spotcheck(
        self, answer: str, context: ValidationContext, report: ValidationReport
    ) -> None:
        """
        Ask the small text model whether the highest-stakes claim is supported.
        Only run when there is a concrete numeric/count claim and evidence to
        check it against — keeps latency sane.
        """
        if self._entailment is None:
            return
        blob = context.evidence_blob()
        if not blob.strip():
            return
        if not _NUMBER_RE.search(answer):
            return

        # Pick the sentence containing the first number as the claim.
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        claim = next((s for s in sentences if _NUMBER_RE.search(s)), "")
        if not claim:
            return

        try:
            verdict = self._entailment(claim, blob)
        except Exception:  # noqa: BLE001
            return

        supported = verdict.get("supported")
        if supported is None:
            return
        conf = float(verdict.get("confidence", 0.0) or 0.0)
        report.add(
            "V2b-entailment",
            bool(supported),
            conf,
            f"claim={claim[:120]!r} reason={verdict.get('reason', '')[:160]}",
        )
        if not supported and conf >= 0.6:
            report.warnings.append(
                "A model check suggests a key numeric claim is not supported by "
                "the evidence."
            )

    def _check_language(
        self, answer: str, context: ValidationContext, report: ValidationReport
    ) -> None:
        """
        V7 — was the answer written in the language the user needs?

        This is the check the assignment's "consistent responses regardless of
        language" requirement depends on: an answer that is correct but arrives
        in the wrong language is a failure, not a pass. The comparison is on the
        *detected* language of the answer against the requested one, and a
        code-mixed answer counts as correct when the requested language is one of
        the languages present (very common for technical Hindi/Bengali).

        Detection is best-effort: if it is unavailable, the check passes to
        avoid punishing a good answer for a detector failure.
        """
        requested = (context.response_language or "en").lower()

        # A clarification or an image transcription is not language-bound in the
        # same way — a transcription must return the text as it appears.
        route = context.plan.route if context.plan else None
        if route in (Route.READ_TEXT, Route.CLARIFY):
            report.add(
                "V7-language",
                True,
                1.0,
                f"Language pinning not enforced for route '{route.value}'",
            )
            return

        try:
            from language_detector import SUPPORTED_LANGUAGES, detect_language_cached

            detected = detect_language_cached(answer)
        except Exception:  # noqa: BLE001 - detection must never break validation
            report.add(
                "V7-language",
                True,
                0.5,
                "Language detection unavailable; skipping the language check",
            )
            return

        matches = detected.code == requested or (
            detected.mixed and requested in (detected.mixed_languages or [])
        )
        # Short answers ("Yes.", a number, a single term) carry too little signal
        # for reliable detection, so they are not held to the check.
        if len(answer.split()) < 3:
            matches = True

        requested_name = SUPPORTED_LANGUAGES.get(requested, requested)
        detected_name = SUPPORTED_LANGUAGES.get(detected.code, detected.code)
        report.add(
            "V7-language",
            matches,
            1.0 if matches else 0.3,
            f"requested={requested_name} detected={detected_name} "
            f"(confidence={detected.confidence:.2f})",
        )
        if not matches:
            report.warnings.append(
                f"The answer came back in {detected_name} but {requested_name} was "
                "requested."
            )

    # ── status decision ─────────────────────────────────────────────────────

    def _decide_status(
        self, report: ValidationReport, context: ValidationContext
    ) -> None:
        route = context.plan.route if context.plan else None

        if route == Route.CLARIFY:
            report.status = ValidationStatus.NEEDS_CLARIFICATION
            report.confidence = 0.9
            return

        failed = {c.name for c in report.failed_checks() if not c.passed}

        # Hard failures
        if "V0-nonempty" in failed or "V5-format" in failed and len(report.final_answer.split()) < 3:
            report.status = ValidationStatus.FAILED
            report.confidence = 0.1
            return

        # Anything critical wrong → QUALIFIED (still shown, with caveats)
        critical = {"V2-grounding", "V3-numbers", "V6-fabrication", "V2b-entailment"}
        if failed & critical:
            report.status = ValidationStatus.QUALIFIED
            report.confidence = max(
                0.2, sum(c.score for c in report.checks) / max(1, len(report.checks))
            )
            note_parts = []
            if "V2-grounding" in failed:
                note_parts.append("evidence overlap was low")
            if "V3-numbers" in failed:
                note_parts.append("some numbers could not be verified")
            if "V6-fabrication" in failed:
                note_parts.append("some named details were not found in the evidence")
            if "V2b-entailment" in failed:
                note_parts.append("a model check disputed a key claim")
            if "V7-language" in failed:
                note_parts.append(
                    "the answer may not be in the requested language"
                )
            report.uncertainty_note = (
                "Validation caveat: " + "; ".join(note_parts) + "."
            )
            return

        # Soft failures only → PASS with a warning
        if failed:
            report.status = ValidationStatus.QUALIFIED
            report.confidence = 0.7
            if failed == {"V7-language"}:
                report.uncertainty_note = (
                    "The answer may not be in the language you asked for."
                )
            else:
                report.uncertainty_note = (
                    "Minor validation warnings: " + ", ".join(sorted(failed)) + "."
                )
            return

        report.status = ValidationStatus.PASS
        report.confidence = min(
            0.98, sum(c.score for c in report.checks) / max(1, len(report.checks))
        )

    # ── answer rewriting ────────────────────────────────────────────────────

    @staticmethod
    def apply_uncertainty_note(report: ValidationReport, answer: str) -> str:
        """
        Prepend the model-independent quality note (if any) to the answer. Keeps
        the assistant honest even when the model was confident.
        """
        note = report.uncertainty_note
        if not note:
            return answer
        if note.strip().lower() in answer.lower():
            return answer
        return f"_{note}_\n\n{answer}"
