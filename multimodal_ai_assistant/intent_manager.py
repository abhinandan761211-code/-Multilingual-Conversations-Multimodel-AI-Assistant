"""
intent_manager.py
=================
Cross-lingual intent normalisation.

The assistant's job is *not* to translate words — it is to preserve **meaning**.
A user can ask the same thing in English ("explain machine learning"), Hindi
("machine learning samjhao"), Hinglish ("isko easy Hindi mein samjhao") or Odia,
and every one of those must resolve to the same internal intent so the rest of
the pipeline behaves identically.

This module produces the *normalised representation* the assignment asks for::

    {
      "detected_language": "hi",
      "mixed_languages": ["en"],
      "intent": "explain_concept",
      "entities": ["training data", "machine learning"],
      "response_language": "hi",
      "needs_clarification": false,
      "confidence": 0.82,
      "source_of_resolution": "heuristic"
    }

Two layers cooperate:

1. **Deterministic heuristics** (always available, no model, no network):
   multilingual keyword tables for intent, technical-term preservation, explicit
   language requests, and pronoun/reference detection for ambiguity.

2. **Optional LLM refinement**: when a small model is supplied it can be asked
   to return the same JSON, which is then validated against the schema below.
   Any parse/validation failure silently falls back to the heuristic result, so
   the system never depends on the model for correctness.

The Pydantic model is the contract: the pipeline, router and validator all read
`NormalizedIntent`, so "intent" means exactly one thing everywhere.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field, field_validator

import config
from language_detector import (
    SUPPORTED_LANGUAGES,
    LanguageReport,
    detect_language_cached,
    detect_requested_language,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intent vocabulary
# ─────────────────────────────────────────────────────────────────────────────


class Intent(str):
    """Convenience alias; intents are plain strings validated against the set."""


# The canonical intents. Every language maps onto one of these — that is the
# whole point of normalisation.
INTENTS = {
    "explain_concept",     # "what is X", "explain X", "X kaise kaam karta hai"
    "define_term",         # "define X", "X ka matlab kya hai"
    "how_to",              # "how do I do X", "X kaise karein"
    "compare",             # "difference between X and Y"
    "example_request",     # "give an example", "ek example do"
    "follow_up",           # refers to earlier turn ("isko samjhao", "why?")
    "translation_request", # "translate this", "Hindi mein bolo"
    "summarise",           # "summarise", "short mein batao"
    "greeting",            # hello / namaste / social
    "describe_image",      # about an uploaded image
    "read_text",           # extract text from an image
    "analyse_chart",       # chart/graph analysis
    "clarify",             # genuinely ambiguous — must ask
    "general_question",    # fallback
}


# ─────────────────────────────────────────────────────────────────────────────
# Multilingual intent keyword tables
# ─────────────────────────────────────────────────────────────────────────────

# Each entry maps an intent to regex fragments across all four languages.
# Roman-Indic (Hinglish) forms are included deliberately, since script
# detection cannot help there.
_INTENT_PATTERNS: dict[str, re.Pattern] = {
    "explain_concept": re.compile(
        r"(explain|what\s+(?:is|are)|how\s+does|why\s+does|meaning\s+of|"
        r"samjha(?:o|ao|do|iye)?|samjhao|batao|bata\s+do|kya\s+(?:hai|hota)|"
        r"kaise\s+kaam|kya\s+matlab|"
        r"ବୁଝାନ୍ତୁ|ବୁଝାଅ|ବ୍ୟାଖ୍ୟା|କ'ଣ|କଣ|"
        r"বুঝ(?:ান|িয়ে)|ব্যাখ্যা|কি|কী)",
        re.IGNORECASE,
    ),
    "define_term": re.compile(
        r"(define|definition\s+of|what\s+does\s+\w+\s+mean|"
        r"matlab\s+kya|matlab\s+batao|arth\s+kya|"
        r"ଅର୍ଥ|ମାନେ|অর্থ|মানে)",
        re.IGNORECASE,
    ),
    "how_to": re.compile(
        r"(how\s+(?:do|can|to|should)\s+i|how\s+to|steps?\s+to|"
        r"kaise\s+(?:kar|kare|karein|bana)|kaise\s+karna|"
        r"କେମିତି|କିପରି|কীভাবে|কিভাবে)",
        re.IGNORECASE,
    ),
    "compare": re.compile(
        r"(difference\s+between|compare|vs\.?|versus|"
        r"antar|farak|antar\s+kya|"
        r"ଫରକ|ଅନ୍ତର|ପାର୍ଥକ୍ୟ|পার্থক্য|তফাৎ)",
        re.IGNORECASE,
    ),
    "example_request": re.compile(
        r"(example|e\.g\.|for\s+instance|illustrat|"
        r"udaharan|udahran|ek\s+example|"
        r"ଉଦାହରଣ|ଦୃଷ୍ଟାନ୍ତ|উদাহরণ)",
        re.IGNORECASE,
    ),
    "translation_request": re.compile(
        r"(translate|in\s+(?:english|hindi|odia|bengali)|"
        r"hindi\s+mein|english\s+mein|odia\s+(?:re|mein|ma)|"
        r"ଓଡ଼ିଆରେ|ଇଂରାଜୀରେ|ହିନ୍ଦୀରେ|বাংলায়|ইংরেজিতে)",
        re.IGNORECASE,
    ),
    "summarise": re.compile(
        r"(summari[sz]e|in\s+brief|short\s+mein|chhota|संक्षेप|"
        r"ସଂକ୍ଷେପ|সংক্ষেপে)",
        re.IGNORECASE,
    ),
    "describe_image": re.compile(
        r"(describe(?:\s+this)?|what(?:'s|\s+is)\s+in|what\s+do\s+you\s+see|"
        r"tell\s+me\s+about\s+(?:this|the)\s+(?:image|picture|photo)|"
        r"is\s+image\s+mein\s+kya|"
        r"ଏହି\s+ଛବି|ଛବିରେ|ଚିତ୍ରରେ|ବର୍ଣ୍ଣନା|"
        r"ছবি|ছবিতে|বর্ণনা)",
        re.IGNORECASE,
    ),
    "read_text": re.compile(
        r"(read|extract|transcribe|ocr|what\s+does\s+it\s+say|"
        r"text\s+in|likha\s+(?:hai|kya)|padho|"
        r"ଲେଖା|ପଢ଼|ପାଠ|পড়|লেখা)",
        re.IGNORECASE,
    ),
    "analyse_chart": re.compile(
        r"(chart|graph|plot|axis|trend|curve|legend|"
        r"ଗ୍ରାଫ୍|ଚାର୍ଟ|ଲେଖଚିତ୍ର|গ্রাফ|চার্ট)",
        re.IGNORECASE,
    ),
    "follow_up": re.compile(
        r"(^\s*(?:and|also|then|what\s+about|and\s+now)\b|"
        r"\b(?:isko|usko|iske|uske|inhe|unhe|yeh|woh|ye|wo|it|this|that|those)\b|"
        r"\b(?:ka\s+role|ka\s+matlab|mein\s+kya|kya\s+role|"
        r"ଏହାର|ଏହା|ତାହା|ଏହି|এর|এটা|সেটা|এটার|ওটার|এর\s+)\b)",
        re.IGNORECASE,
    ),
    "greeting": re.compile(
        r"^\s*(hi|hey|hello|yo|namaste|namaskar|pranam|"
        r"good\s+(?:morning|evening|afternoon)|thanks|thank\s+you|"
        r"ନମସ୍କାର|ନମସ୍ତେ|ଧନ୍ୟବାଦ|নমস্কার|ধন্যবাদ|ok|okay|cool)"
        r"\s*[!.?]*\s*$",
        re.IGNORECASE,
    ),
}


def score_intents(text: str) -> dict[str, int]:
    """Count pattern matches per intent (used for ranking + confidence)."""
    scores: dict[str, int] = {}
    for intent, pattern in _INTENT_PATTERNS.items():
        hits = len(pattern.findall(text))
        if hits:
            scores[intent] = hits
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Technical-term preservation
# ─────────────────────────────────────────────────────────────────────────────

# Terms we must NOT translate: translating "Python" or "API" produces nonsense
# and the assignment explicitly calls this out. Kept as a decent-size general
# list rather than an exhaustive dictionary.
_TECH_TERMS = {
    "api", "python", "javascript", "java", "sql", "html", "css", "json", "xml",
    "http", "https", "url", "rest", "graphql", "sdk", "cli", "gui", "cpu",
    "gpu", "ram", "os", "ide", "git", "github", "docker", "kubernetes",
    "machine learning", "deep learning", "neural network", "model", "dataset",
    "training data", "validation", "overfitting", "underfitting", "regression",
    "classification", "clustering", "gradient descent", "backpropagation",
    "transformer", "embedding", "token", "prompt", "llm", "rag", "vector",
    "database", "server", "client", "frontend", "backend", "framework",
    "library", "function", "variable", "array", "loop", "class", "object",
    "algorithm", "compiler", "runtime", "debug", "error", "exception", "test",
    "deployment", "cache", "memory", "encoding", "pixel", "ocr", "vlm",
}


def extract_entities(text: str) -> list[str]:
    """
    Pull technical terms and likely subject nouns out of the message.

    This is deliberately conservative: it favours known technical terms
    (which must survive translation) and capitalised / quoted spans, rather
    than attempting full NLP noun-phrase chunking.
    """
    found: list[str] = []
    lowered = text.lower()

    # 1. Known technical terms (longest first, so "machine learning" wins over
    #    "learning" if both are present).
    for term in sorted(_TECH_TERMS, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered):
            if term not in found:
                found.append(term)

    # 2. Quoted phrases — the user explicitly marked a subject.
    for quoted in re.findall(r"[\"“'‘]([^\"”'’]{2,40})[\"”'’]", text):
        quoted = quoted.strip()
        if quoted and quoted.lower() not in found:
            found.append(quoted)

    # 3. Acronyms (API, RAG, OCR…) not already captured.
    for acronym in re.findall(r"\b([A-Z]{2,6})\b", text):
        if acronym.lower() not in found:
            found.append(acronym.lower())

    return found[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Schema (the contract)
# ─────────────────────────────────────────────────────────────────────────────


class NormalizedIntent(BaseModel):
    """
    The validated, language-independent representation of one user message.

    Validating with Pydantic means a malformed model response can never inject
    a bogus intent into the router: unknown intents are coerced to
    ``general_question`` and confidences are clamped to 0–1.
    """

    detected_language: str = Field(default=config.DEFAULT_LANGUAGE)
    mixed_languages: list[str] = Field(default_factory=list)
    intent: str = "general_question"
    entities: list[str] = Field(default_factory=list)
    response_language: str = Field(default=config.DEFAULT_LANGUAGE)
    needs_clarification: bool = False
    clarifying_question: str = ""
    is_follow_up: bool = False
    language_confidence: float = 0.5
    confidence: float = 0.5
    source_of_resolution: str = "heuristic"   # heuristic | llm | hybrid

    @field_validator("intent")
    @classmethod
    def _known_intent(cls, value: str) -> str:
        value = (value or "").strip().lower()
        return value if value in INTENTS else "general_question"

    @field_validator("confidence", "language_confidence")
    @classmethod
    def _clamp(cls, value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("entities", "mixed_languages", mode="before")
    @classmethod
    def _coerce_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return list(value)

    def to_dict(self) -> dict:
        return self.model_dump()


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IntentContext:
    """What the manager is allowed to use when normalising a message."""

    previous_language: Optional[str] = None
    has_image: bool = False
    has_prior_turns: bool = False
    recent_subjects: list[str] = field(default_factory=list)
    language_report: Optional[LanguageReport] = None


# A pronoun-led request that asks to "improve / fix" something unnamed. When
# more than one subject is in play, which one is being referred to is genuinely
# unknown, so the assistant must ask rather than pick. The pattern allows the
# "how do I…" tail to follow the pronoun phrase ("isko improve kaise karun").
_AMBIGUOUS_REFERENCE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:isko|usko|ise|use|isme|usme|ye|wo|yeh|woh|"
    r"it|this|that|"
    r"ଏହା|ଏହାକୁ|ଏଟା|ଏହାକୁ|ଏହାର|এটা|এটাকে|এটার)"
    r"[^?.!]{0,60}"
    r"(?:improve|better|fix|thik|theek|behtar|badha|sudhar|"
    r"ଉନ୍ନତ|ଭଲ|ସୁଧାର|ଉପଯୁକ୍ତ|"
    r"উন্নত|ভালো|উন্নয়ন|ঠিক)",
    re.IGNORECASE,
)


class IntentManager:
    """
    Normalises a multilingual message into a `NormalizedIntent`.

    If a `refiner` callable is supplied — normally
    ``MultilingualModel.normalize_intent`` — it is consulted for genuinely
    ambiguous messages. Its JSON output is validated by the schema above; any
    failure leaves the deterministic result in place.
    """

    def __init__(self, refiner=None) -> None:
        self.refiner = refiner

    # ── public API ──────────────────────────────────────────────────────────

    def normalize(
        self,
        text: str,
        context: Optional[IntentContext] = None,
        use_llm: bool = False,
    ) -> NormalizedIntent:
        context = context or IntentContext()
        text = (text or "").strip()

        report = context.language_report or detect_language_cached(
            text, context.previous_language
        )

        intended = self._heuristic_intent(text, context, report)
        entities = extract_entities(text)

        # Decide the answer language: an explicit request wins; otherwise the
        # detected input language; otherwise the previous turn's language.
        requested = detect_requested_language(text)
        response_language = self._resolve_response_language(
            requested, report, context
        )

        result = NormalizedIntent(
            detected_language=report.code,
            mixed_languages=report.mixed_languages if report.mixed else [],
            intent=intended["intent"],
            entities=entities,
            response_language=response_language,
            needs_clarification=intended["needs_clarification"],
            clarifying_question=intended["clarifying_question"],
            is_follow_up=intended["is_follow_up"],
            language_confidence=report.confidence,
            confidence=intended["confidence"],
            source_of_resolution="heuristic",
        )

        # Optional LLM refinement for the hard cases (short, ambiguous, or
        # mixed-language requests where keywords disagree).
        if use_llm and self.refiner is not None and self._worth_refining(result, text):
            refined = self._try_llm(text, result, context)
            if refined is not None:
                result = refined

        return result

    # ── heuristics ──────────────────────────────────────────────────────────

    def _heuristic_intent(
        self,
        text: str,
        context: IntentContext,
        report: LanguageReport,
    ) -> dict:
        scores = score_intents(text)

        # Greeting is a whole-message match; it should dominate.
        if "greeting" in scores and len(text.split()) <= 4:
            return self._result("greeting", 0.95, follow_up=False)

        # Ambiguity is checked BEFORE anything else that could resolve it by
        # guessing. A pronoun-led request with several candidate subjects
        # ("isko improve kaise karun?") must ask, not pick one — the assignment
        # is explicit that clarification beats a confident wrong answer.
        needs_clarification, question = self._ambiguous(text, context, report)
        if needs_clarification:
            return self._result(
                "clarify",
                0.85,
                needs_clarification=True,
                clarifying_question=question,
                follow_up=True,
            )

        # A fresh image with no concrete ask → describe.
        if context.has_image and not scores:
            return self._result("describe_image", 0.7)

        # Reference to earlier context is the follow-up signal. It matters more
        # than a generic "explain" keyword: "isko samjhao" is a follow-up even
        # though "samjhao" matches explain.
        referential = "follow_up" in scores or self._looks_referential(text)
        if referential and context.has_prior_turns:
            return self._result("follow_up", 0.8, follow_up=True)

        if not scores:
            return self._result(
                "general_question", 0.5,
                needs_clarification=False,
            )

        # Pick the highest-scoring non-follow-up intent; ties break by a fixed
        # priority so behaviour is deterministic across languages.
        priority = [
            "read_text", "analyse_chart", "compare", "how_to", "define_term",
            "translation_request", "summarise", "example_request",
            "explain_concept", "describe_image", "general_question",
        ]
        candidates = {k: v for k, v in scores.items() if k != "follow_up"}
        if not candidates:
            return self._result("follow_up", 0.6, follow_up=True)

        best = max(
            candidates.items(),
            key=lambda kv: (kv[1], -priority.index(kv[0]) if kv[0] in priority else -99),
        )[0]
        confidence = min(0.9, 0.6 + 0.1 * candidates[best])

        return self._result(best, confidence)

    # ── ambiguity detection ─────────────────────────────────────────────────

    def _ambiguous(
        self, text: str, context: IntentContext, report: LanguageReport
    ) -> tuple[bool, str]:
        """
        Detect a request that cannot be resolved without asking.

        The canonical case from the brief: "Isko improve kaise karun?" when two
        plausible subjects exist. Guessing is worse than asking, so this returns
        a clarifying question *in the user's language*.
        """
        words = text.split()

        # A pronoun-led imperative with no subject noun is genuinely ambiguous
        # when more than one subject is in play.
        if _AMBIGUOUS_REFERENCE_RE.search(text) and len(context.recent_subjects) > 1:
            subjects = context.recent_subjects[:3]
            pretty = ", ".join(subjects)
            return True, self._clarify_template(report.code, pretty)

        # "it / this / that" with nothing to attach to and no image.
        if (
            report.code in SUPPORTED_LANGUAGES
            and len(words) <= 4
            and re.match(
                r"^\s*(?:it|this|that|isko|usko|ise|ଏହା|ଏଟା|এটা|এটাকে)\b",
                text,
                re.IGNORECASE,
            )
            and not context.has_image
            and context.has_prior_turns
        ):
            return True, self._clarify_template(report.code, "")

        return False, ""

    @staticmethod
    def _clarify_template(code: str, subjects: str) -> str:
        """Localised clarification prompt (English is the safe fallback)."""
        if code == "hi":
            if subjects:
                return f"Aap kis cheez ko improve karna chahte hain — {subjects}?"
            return "Aap kis cheez ke baare mein baat kar rahe hain? Thoda clear kijiye."
        if code == "or":
            if subjects:
                return f"ଆପଣ କାହାକୁ ଉନ୍ନତ କରିବାକୁ ଚାହୁଁଛନ୍ତି — {subjects}?"
            return "ଆପଣ କେଉଁ ବିଷୟ ବିଷୟରେ କହୁଛନ୍ତି? ଟିକେ ସ୍ପଷ୍ଟ କରନ୍ତୁ।"
        if code == "bn":
            if subjects:
                return f"আপনি কোনটি উন্নত করতে চান — {subjects}?"
            return "আপনি কোন বিষয়ে বলছেন? একটু স্পষ্ট করুন।"
        if subjects:
            return f"Which would you like to improve — {subjects}?"
        return "What would you like me to work on? A little more detail would help."

    # ── language resolution ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_response_language(
        requested: Optional[str],
        report: LanguageReport,
        context: IntentContext,
    ) -> str:
        # 1. Explicit request ("explain in English") always wins.
        if requested:
            return requested
        # 2. A supported detected language.
        if report.is_supported():
            return report.code
        # 3. Unsupported language → default fallback, but echo if we know better.
        if context.previous_language in SUPPORTED_LANGUAGES:
            return context.previous_language
        return config.UNSUPPORTED_LANGUAGE_FALLBACK

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _looks_referential(text: str) -> bool:
        return bool(
            re.search(
                r"\b(it|this|that|those|isko|usko|iske|uske|ye|wo|yeh|woh|"
                r"ଏହା|ଏହାର|ତାହା|এটা|এটার|সেটা|এর)\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _result(
        intent: str,
        confidence: float,
        needs_clarification: bool = False,
        clarifying_question: str = "",
        follow_up: bool = False,
    ) -> dict:
        return {
            "intent": intent,
            "confidence": confidence,
            "needs_clarification": needs_clarification,
            "clarifying_question": clarifying_question,
            "is_follow_up": follow_up,
        }

    @staticmethod
    def _worth_refining(result: NormalizedIntent, text: str) -> bool:
        """Only spend a model call when the heuristic result is uncertain."""
        if result.needs_clarification:
            return True
        if result.intent == "general_question" and len(text.split()) <= 6:
            return True
        if result.mixed_languages and result.confidence < 0.7:
            return True
        return False

    def _try_llm(
        self,
        text: str,
        fallback: NormalizedIntent,
        context: IntentContext,
    ) -> Optional[NormalizedIntent]:
        refiner = self.refiner
        if refiner is None:  # guarded here so the call site is provably safe
            return None
        try:
            raw = refiner(
                text,
                context={
                    "has_image": context.has_image,
                    "has_prior_turns": context.has_prior_turns,
                    "previous_language": context.previous_language,
                    "recent_subjects": context.recent_subjects,
                },
                fallback=fallback.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Intent refinement failed: %s", exc)
            return None

        if not raw:
            return None
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if not isinstance(raw, dict):
            return None

        try:
            refined = NormalizedIntent(**{**fallback.to_dict(), **raw})
        except Exception as exc:  # noqa: BLE001
            log.debug("Refined intent failed validation: %s", exc)
            return None

        refined.source_of_resolution = "llm"
        return refined


_refiner_cache: dict[int, IntentManager] = {}


def get_intent_manager(refiner=None) -> IntentManager:
    """Cache one manager per refiner identity (cheap; avoids churn)."""
    key = id(refiner)
    if key not in _refiner_cache:
        _refiner_cache[key] = IntentManager(refiner=refiner)
    return _refiner_cache[key]
