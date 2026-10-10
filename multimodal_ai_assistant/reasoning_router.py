"""
reasoning_router.py
===================
The decision layer. This is what separates the assistant from "send image to a
model and print the answer".

Given (user message, optional image, conversation memory) the router produces a
`RoutePlan` describing *how* the request should be handled:

    - which strategy applies (describe / read_text / analyse_chart / answer /
      follow_up / general / clarify / refuse)
    - whether OCR should run (and whether the result should be trusted)
    - whether evidence retrieval is worthwhile
    - what pre-processing the image needs
    - whether to ask a clarifying question instead of guessing

Decision inputs
---------------
1. Deterministic heuristics  — cheap, explainable, always available.
2. Measured image facts      — from image_processor.ImageDiagnostics.
3. Conversation state        — from ConversationMemory (follow-up detection).
4. An optional LLM classifier — used only when the heuristics are genuinely
   ambiguous. The router never *depends* on the model to make a decision.

Every plan carries a `rationale` list so the UI (and the teacher) can see *why*
a path was chosen.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import config
from conversation_memory import ConversationMemory, ImageRecord
from image_processor import ImageDiagnostics


class Route(str, Enum):
    """The handling strategies the assistant knows about."""

    CLARIFY = "clarify"                    # ambiguous → ask before acting
    DESCRIBE_IMAGE = "describe_image"      # image, no specific question
    READ_TEXT = "read_text"                # OCR-first (document/screenshot)
    ANALYSE_CHART = "analyse_chart"        # graph/plot interpretation
    ANSWER_WITH_IMAGE = "answer_with_image"  # specific question + image
    FOLLOW_UP = "follow_up"                # refers to earlier image/turn
    GENERAL_QUESTION = "general_question"  # text only (may need retrieval)
    SMALL_TALK = "small_talk"              # greetings etc.
    REFUSE_UNSUPPORTED = "refuse_unsupported"  # not enough evidence to answer


# ─────────────────────────────────────────────────────────────────────────────
# Plan object
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RoutePlan:
    route: Route
    rationale: list[str] = field(default_factory=list)

    # image handling
    use_vision: bool = False
    use_ocr: bool = False
    trust_ocr: bool = True
    preprocess_image: bool = False
    image_id: Optional[str] = None
    image_is_new: bool = False

    # evidence handling
    use_retrieval: bool = False
    retrieval_query: str = ""

    # clarification
    clarifying_question: str = ""

    # book-keeping
    confidence: float = 0.5
    classifier_used: bool = False
    detected_intent: str = ""

    @property
    def needs_image_from_memory(self) -> bool:
        return self.route == Route.FOLLOW_UP and self.image_id is not None

    def explain(self) -> str:
        bits = [f"route={self.route.value}", f"confidence={self.confidence:.2f}"]
        if self.use_vision:
            bits.append("vision=yes")
        if self.use_ocr:
            bits.append(f"ocr=yes(trust={self.trust_ocr})")
        if self.use_retrieval:
            bits.append("retrieval=yes")
        if self.classifier_used:
            bits.append("classifier=llm")
        return " | ".join(bits)


# ─────────────────────────────────────────────────────────────────────────────
# Signals extracted from the user message
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IntentSignals:
    wants_description: bool = False
    wants_text_extraction: bool = False
    wants_chart_analysis: bool = False
    wants_count: bool = False
    wants_explanation: bool = False
    wants_translation: bool = False
    asks_about_part: bool = False
    is_greeting: bool = False
    is_too_vague: bool = False
    mentions_uncertainty: bool = False
    explicit_fact_request: bool = False   # likely needs external knowledge
    is_question: bool = False
    word_count: int = 0


_DESCRIBE_PATTERNS = re.compile(
    r"\b(describe|what(?:'s| is) in|what do you see|tell me about (?:this|the) "
    r"(?:image|picture|photo)|explain (?:this|the) image|summari[sz]e)\b",
    re.IGNORECASE,
)
_TEXT_PATTERNS = re.compile(
    r"\b(read|extract|transcribe|ocr|text in|what does it say|what does the "
    r"(?:text|writing|sign|label) say|copy the text)\b",
    re.IGNORECASE,
)
_CHART_PATTERNS = re.compile(
    r"\b(graph|chart|plot|axis|axes|trend|curve|bar|line graph|histogram|"
    r"scatter|legend|x-axis|y-axis|validation loss|training loss|accuracy curve)\b",
    re.IGNORECASE,
)
_COUNT_PATTERNS = re.compile(
    r"\b(how many|count|number of|how much)\b", re.IGNORECASE
)
_EXPLAIN_PATTERNS = re.compile(
    # Covers "what is X", "what does X do", "how do/does X", "why does X",
    # "explain", "the role/purpose/function of X" — the open-ended question
    # forms whose answers benefit from supporting reference material.
    r"\b(explain|why|how (?:do|does|did|are|is)|what (?:is|are|does|do|did)|"
    r"meaning of|interpret|role of|purpose of|function of|difference between|"
    r"responsible for)\b",
    re.IGNORECASE,
)
_TRANSLATE_PATTERNS = re.compile(r"\b(translate|in hindi|in english|convert to)\b", re.IGNORECASE)
_PART_PATTERNS = re.compile(
    r"\b(middle|center|centre|top|bottom|left|right|upper|lower|inner|outer|"
    r"first|second|third|last|the \w+ layer|the \w+ one)\b",
    re.IGNORECASE,
)
_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hey|hello|yo|namaste|good (morning|evening|afternoon)|thanks|"
    r"thank you|ok|okay|cool)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_VAGUE_PATTERNS = re.compile(
    r"^\s*("
    r"check (?:this|it|the (?:image|picture|photo|pic))|"
    r"look at (?:this|it|the (?:image|picture|photo))|"
    r"see (?:this|it)|"
    r"thoughts|any (?:idea|thought)s?|(?:what|how) about (?:this|it)|"
    r"help(?: me)?|analyse?|analyze|"
    r"(?:this|it)|which one"
    r")"
    # optional trailing object words and punctuation, then end of string
    r"(?:\s+(?:image|picture|photo|pic|graph|chart|diagram|screenshot))?"
    r"\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_FACT_PATTERNS = re.compile(
    r"\b(who (?:invented|discovered|wrote)|when (?:was|did|is)|where is|"
    r"which paper|according to|cite|source|reference|define|definition of|"
    r"what year)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_PATTERNS = re.compile(
    r"\b(unclear|blurry|can't read|cannot read|not sure|is it possible|"
    r"if you can)\b",
    re.IGNORECASE,
)

# Vocabulary that refers to the picture itself. Presence of any of these is the
# clearest sign a question is about an image rather than a general topic.
_IMAGE_VOCAB = {
    "image", "picture", "photo", "photograph", "diagram", "chart", "graph",
    "plot", "figure", "screenshot", "scan", "drawing", "sketch", "curve",
    "axis", "axes", "legend", "label", "caption", "annotation", "title",
    "colour", "color", "blue", "orange", "red", "green", "yellow", "grey",
    "gray", "black", "white", "line", "bar", "point", "dot", "slope",
    "pixel", "resolution", "blurry", "visible", "shown", "depicted",
    "displayed", "pictured", "rendered", "y-axis", "x-axis", "trend",
}

# Words carrying little topical signal, used only when judging whether a
# question shares subject matter with the pinned image.
_STOPWORDS_FOR_RELEVANCE = {
    "what", "when", "where", "which", "whose", "how", "why", "does", "did",
    "can", "could", "would", "should", "will", "explain", "please", "tell",
    "about", "with", "from", "that", "this", "these", "those", "there",
    "their", "have", "has", "had", "been", "being", "into", "than", "then",
    "also", "just", "like", "some", "more", "most", "much", "many", "very",
    "your", "yours", "want", "need", "help", "using", "make", "made",
}


def extract_signals(message: str) -> IntentSignals:
    """Cheap, deterministic reading of what the user is asking for."""
    text = (message or "").strip()
    signals = IntentSignals()
    signals.word_count = len(text.split())
    if not text:
        return signals

    signals.wants_description = bool(_DESCRIBE_PATTERNS.search(text))
    signals.wants_text_extraction = bool(_TEXT_PATTERNS.search(text))
    signals.wants_chart_analysis = bool(_CHART_PATTERNS.search(text))
    signals.wants_count = bool(_COUNT_PATTERNS.search(text))
    signals.asks_about_part = bool(_PART_PATTERNS.search(text))
    signals.wants_explanation = bool(_EXPLAIN_PATTERNS.search(text))
    signals.wants_translation = bool(_TRANSLATE_PATTERNS.search(text))
    signals.asks_about_part = bool(_PART_PATTERNS.search(text))
    signals.is_greeting = bool(_GREETING_PATTERNS.match(text))
    signals.mentions_uncertainty = bool(_UNCERTAINTY_PATTERNS.search(text))
    signals.explicit_fact_request = bool(_FACT_PATTERNS.search(text))
    signals.is_question = text.endswith("?") or bool(
        re.match(r"^\s*(what|why|how|when|where|who|which|is|are|can|could|does|do)\b",
                 text, re.IGNORECASE)
    )

    # "Too vague" = matches a vague phrase, or is a very short fragment that
    # doesn't ask for anything specific. A vague phrase always wins over a
    # keyword like "graph" appearing elsewhere in the sentence — "check this
    # graph" still doesn't say *what* to check.
    specific_task = (
        signals.wants_text_extraction
        or signals.wants_description
        or signals.wants_count
        or signals.explicit_fact_request
        or signals.is_question
    )
    vague_match = bool(_VAGUE_PATTERNS.match(text))
    signals.is_too_vague = vague_match or (
        signals.word_count <= 2
        and not specific_task
        and not signals.is_greeting
    )
    # A bare chart keyword with no question is a request to do *something* with
    # a chart, not a specific ask → treat as vague.
    if signals.wants_chart_analysis and not specific_task and signals.word_count <= 4:
        signals.is_too_vague = True
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────


class ReasoningRouter:
    """
    Chooses a `RoutePlan`. Pure decision-making; it does not call the model
    unless the heuristics are ambiguous *and* a classifier is supplied.
    """

    def __init__(self, classifier=None) -> None:
        # `classifier` is an optional callable(user_text, has_image) -> dict,
        # normally VisionModel.classify_request. Injected so tests stay offline.
        self.classifier = classifier

    # ── public entry point ──────────────────────────────────────────────────

    def plan(
        self,
        user_message: str,
        diagnostics: Optional[ImageDiagnostics] = None,
        memory: Optional[ConversationMemory] = None,
        new_image_id: Optional[str] = None,
    ) -> RoutePlan:
        signals = extract_signals(user_message)
        memory = memory or ConversationMemory()
        has_new_image = diagnostics is not None and diagnostics.valid
        has_image = has_new_image or memory.active_image is not None

        # 1. Small talk — no image work needed.
        if signals.is_greeting and not has_new_image:
            return RoutePlan(
                route=Route.SMALL_TALK,
                rationale=["Message is a greeting/social turn"],
                confidence=0.95,
            )

        # 2. A broken/unreadable image is itself a decision point.
        if diagnostics is not None and not diagnostics.valid:
            return RoutePlan(
                route=Route.CLARIFY,
                rationale=[
                    f"Image could not be read: {diagnostics.error}",
                    "Asking the user for a usable image instead of guessing",
                ],
                clarifying_question=(
                    "I couldn't read that image "
                    f"({diagnostics.error}). Could you re-upload it as a PNG or "
                    "JPG, ideally a little larger?"
                ),
                confidence=0.95,
            )

        # 3. No image at all → text-only reasoning (maybe with retrieval).
        if not has_new_image and not has_image:
            return self._plan_text_only(user_message, signals)

        # ── From here on we have an image, either fresh or from memory ──────

        # 4. New image with a vague request → clarify rather than guess.
        if has_new_image and signals.is_too_vague:
            plan = RoutePlan(
                route=Route.CLARIFY,
                rationale=[
                    "A new image was supplied but the request is ambiguous",
                    f"Detected vague phrasing (words={signals.word_count})",
                ],
                image_id=new_image_id,
                image_is_new=True,
                clarifying_question=self._clarifying_question_for(diagnostics),
                confidence=0.9,
            )
            # Offer an LLM second opinion when available; heuristics stay the
            # default because they cannot hallucinate a route.
            if self.classifier is not None:
                verdict = self._safe_classify(user_message, has_image=True)
                if verdict and verdict.get("intent") not in (None, "unclear") and not verdict.get("needs_clarification", False):
                    # Model thinks there IS a concrete task — trust it, but note it.
                    plan.rationale.append(
                        f"LLM classifier disagreed (intent={verdict.get('intent')}); deferring to specifics"
                    )
                    return self._plan_for_image_intent(
                        user_message, verdict, signals, diagnostics,
                        new_image_id, memory, rationale_extra=plan.rationale,
                    )
                plan.classifier_used = bool(verdict)
                if verdict:
                    plan.clarifying_question = (
                        verdict.get("clarifying_question") or plan.clarifying_question
                    )
                    plan.detected_intent = verdict.get("intent", "")
            return plan

        # 5. Follow-up that refers to a stored image.
        if not has_new_image and memory.is_follow_up(user_message):
            record = memory.resolve_image_for(user_message)
            return RoutePlan(
                route=Route.FOLLOW_UP,
                rationale=[
                    "Message contains referring expressions to earlier context",
                    f"Resolved to stored image '{record.filename}'" if record
                    else "No stored image matched; will use conversation context only",
                ],
                image_id=record.image_id if record else None,
                use_vision=bool(record and record.summary),
                use_retrieval=signals.explicit_fact_request or signals.wants_explanation,
                retrieval_query=self._retrieval_query(user_message, record),
                confidence=0.8,
            )

        # 6. A concrete request that ARRIVES WITH a new image is about that image.
        if has_new_image:
            why = ["Fresh image supplied with a concrete request"]
            if diagnostics and diagnostics.looks_like_chart_or_diagram:
                why.append("Pixel structure suggests a chart or diagram")
            if diagnostics and diagnostics.looks_like_text:
                why.append("Pixel structure suggests text content")
            return self._plan_for_image_intent(
                user_message, None, signals, diagnostics, new_image_id, memory,
                rationale_extra=why,
            )

        # 7. No new image, and the message did not look like a follow-up.
        #
        # An image merely EXISTING earlier in the conversation is not a reason to
        # route every later question through the vision model. "What is the
        # capital of France?" asked while a chart is pinned is a text question.
        # Forcing it through the image path makes the assistant answer about the
        # chart instead of the question — the reported failure mode.
        record = memory.active_image
        if self._question_needs_image(user_message, signals, record, memory):
            return RoutePlan(
                route=Route.FOLLOW_UP,
                rationale=[
                    "Question appears to be about the pinned image",
                    "Reusing the stored image context rather than re-uploading",
                ],
                image_id=record.image_id if record else None,
                use_vision=bool(record and record.summary),
                use_retrieval=signals.explicit_fact_request or signals.wants_explanation,
                retrieval_query=self._retrieval_query(user_message, record),
                confidence=0.7,
            )

        # 8. Self-contained question → answer it on its own terms.
        plan = self._plan_text_only(user_message, signals)
        plan.rationale.insert(
            0,
            "An image is in context but this question stands on its own — "
            "answering it directly instead of forcing it through the image",
        )
        return plan

    # ── image relevance ─────────────────────────────────────────────────────

    @staticmethod
    def _question_needs_image(
        user_message: str,
        signals: IntentSignals,
        record: Optional[ImageRecord],
        memory: ConversationMemory,
    ) -> bool:
        """
        Decide whether a question is actually about the pinned image.

        This is the guard that stops the assistant from dragging every question
        back to the last picture. Erring toward "no" is deliberate: a text
        question answered well is better than a relevant-sounding answer about
        the wrong thing.

        Signals that DO indicate the image:
          * explicit visual vocabulary (chart, graph, image, diagram, colour…)
          * a question about a region/part ("the middle layer", "the top line")
          * an anaphoric request to continue with the same subject
            ("and now?", "what about it?")
          * a short prepositional fragment that has nowhere else to attach
            ("in the graph?", "of the curve?")

        Signals that do NOT: naming a different topic, or asking something that
        is complete on its own (general knowledge, coding, arithmetic).
        """
        if record is None:
            return False

        lowered = user_message.lower()
        tokens = set(re.findall(r"[a-z']+", lowered))

        # Explicit imagery vocabulary — the strongest positive signal.
        if tokens & _IMAGE_VOCAB:
            return True

        # Named parts of a picture ("the middle layer", "the top line").
        # `asks_about_part` is computed by extract_signals, so the part
        # vocabulary lives in exactly one place.
        if signals.asks_about_part:
            return True

        # Continuing the same subject: "and now", "what about it", "also?"
        if re.match(r"^\s*(and|also|then|what about|how about)\b", lowered):
            return True

        # A question that opens by pointing back into the picture.
        if re.match(r"^\s*(in|on|from|of)\s+(the|this|that)\b", lowered):
            return True

        # Self-contained topics: if the message names a subject that the image
        # summary does not mention, it is a new question, not a follow-up.
        summary_terms = (
            set(re.findall(r"[a-z']+", (record.summary or "").lower()))
            | set(re.findall(r"[a-z']+", (record.filename or "").lower()))
        )
        content = {
            token for token in tokens
            if token not in _STOPWORDS_FOR_RELEVANCE and len(token) > 3
        }
        if not content:
            # Nothing substantive ("ok?", "why?") — treat as continuing.
            return True

        # Overlap with what the image is known to be about.
        if content & summary_terms:
            return True

        # No shared vocabulary → a different topic. Let it be a text question.
        return False

    # ── text-only path ──────────────────────────────────────────────────────

    def _plan_text_only(self, user_message: str, signals: IntentSignals) -> RoutePlan:
        rationale = ["No image in this turn or in the session"]
        use_retrieval = (
            signals.explicit_fact_request
            or signals.wants_explanation
            or signals.is_question
        )
        if use_retrieval:
            rationale.append("Question may need supporting documents → retrieval enabled")
        return RoutePlan(
            route=Route.GENERAL_QUESTION,
            rationale=rationale,
            use_retrieval=use_retrieval,
            retrieval_query=user_message.strip(),
            confidence=0.75,
        )

    # ── image strategy selection ────────────────────────────────────────────

    def _plan_for_image_intent(
        self,
        user_message: str,
        classifier_verdict: Optional[dict],
        signals: IntentSignals,
        diagnostics: Optional[ImageDiagnostics],
        image_id: Optional[str],
        memory: ConversationMemory,
        rationale_extra: Optional[list[str]] = None,
    ) -> RoutePlan:
        rationale = list(rationale_extra or [])

        # ── decide OCR ──────────────────────────────────────────────────────
        use_ocr = False
        trust_ocr = True
        if diagnostics is not None:
            if signals.wants_text_extraction:
                use_ocr = True
                rationale.append("User explicitly asked for text extraction → OCR")
            elif diagnostics.looks_like_text:
                use_ocr = True
                rationale.append(
                    "Image shows text-like structure → OCR as cross-check"
                )
            elif diagnostics.looks_like_chart_or_diagram:
                use_ocr = True
                rationale.append(
                    "Chart/diagram → OCR to capture axis labels and legend"
                )
            else:
                rationale.append("No text-like structure detected → skipping OCR")
            if diagnostics.is_blurry or diagnostics.is_low_resolution:
                trust_ocr = False
                rationale.append(
                    "OCR output treated as low-confidence due to image quality"
                )

        # ── decide route ────────────────────────────────────────────────────
        intent = (classifier_verdict or {}).get("intent", "")
        route = Route.ANSWER_WITH_IMAGE
        confidence = 0.7

        if signals.wants_text_extraction or intent == "read_text":
            route = Route.READ_TEXT
            confidence = 0.9
            rationale.append("Strategy: read and return the text")
        elif signals.wants_chart_analysis or diagnostics and diagnostics.looks_like_chart_or_diagram:
            route = Route.ANALYSE_CHART
            confidence = 0.85
            rationale.append("Strategy: structured chart analysis")
        elif signals.wants_description and not (
            signals.wants_chart_analysis or signals.wants_text_extraction
        ):
            route = Route.DESCRIBE_IMAGE
            confidence = 0.85
            rationale.append("Strategy: free-form description of the image")
        else:
            route = Route.ANSWER_WITH_IMAGE
            rationale.append("Strategy: answer the specific question using the image")

        # A specific count/explanation question always wins over a generic
        # "describe" only when both are present, answer the question.
        if signals.wants_count or signals.explicit_fact_request:
            route = Route.ANSWER_WITH_IMAGE
            rationale.append("Question asks for specific facts → direct answer path")

        use_retrieval = (
            signals.explicit_fact_request
            or (signals.wants_explanation and not signals.wants_chart_analysis)
        )
        if use_retrieval:
            rationale.append("Enabling document retrieval for supporting evidence")

        return RoutePlan(
            route=route,
            rationale=rationale,
            use_vision=True,          # every image path uses the VLM at least once
            use_ocr=use_ocr,
            trust_ocr=trust_ocr,
            preprocess_image=bool(
                diagnostics and (diagnostics.is_blurry or diagnostics.looks_like_text)
            ),
            image_id=image_id,
            image_is_new=True,
            use_retrieval=use_retrieval,
            retrieval_query=self._retrieval_query(user_message, None),
            confidence=confidence,
            classifier_used=classifier_verdict is not None,
            detected_intent=intent or route.value,
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    def _safe_classify(self, user_message: str, has_image: bool) -> Optional[dict]:
        if self.classifier is None:
            return None
        try:
            verdict = self.classifier(user_message, has_image)
            if isinstance(verdict, dict) and not verdict.get("fallback", False):
                return verdict
        except Exception:  # noqa: BLE001 - classification is best-effort
            return None
        return None

    @staticmethod
    def _clarifying_question_for(diagnostics: Optional[ImageDiagnostics]) -> str:
        base = (
            "I can see you've shared an image, but I'm not sure what you'd like "
            "me to do with it. Would you like me to:\n"
            "1. describe what's in it,\n"
            "2. extract / read the text it contains, or\n"
            "3. analyse a specific detail (tell me which one)?"
        )
        if diagnostics and diagnostics.is_blurry:
            base += (
                "\n\nNote: the image looks a little blurry, so fine text may not "
                "be fully readable."
            )
        return base

    @staticmethod
    def _retrieval_query(user_message: str, record: Optional[ImageRecord]) -> str:
        query = user_message.strip()
        if record and record.summary:
            # Blend the question with what the image is about for better hits.
            query = f"{query} {record.summary[:200]}"
        return query
