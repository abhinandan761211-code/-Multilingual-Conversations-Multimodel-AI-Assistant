"""
language_detector.py
====================
Automatic language identification for the multilingual assistant.

The assistant must detect the user's language *without* the user selecting it,
across four supported languages:

    en  English
    hi  Hindi      (Devanagari, and Roman-script "Hinglish")
    or  Odia       (Odia script)
    bn  Bengali    (Bengali script)

Design goals
------------
* **Deterministic first.** Unicode script blocks identify Devanagari, Odia and
  Bengali with near-perfect accuracy and zero dependencies. That alone resolves
  three of the four languages reliably, even offline.
* **Fuzzy for Roman-script text.** English vs Roman-Hindi ("Hinglish") cannot be
  told apart by script — "mera model train nahi ho raha" is all Latin. A
  lexicon-based scorer plus an optional statistical backend (FastText `lid.176`
  or `lingua`) decides, with a confidence and an explicit `mixed` flag when both
  look present.
* **Never crash, never block a turn.** Any backend that is missing degrades to
  the heuristic path; detection is *best-effort* and always returns a result
  object rather than raising.

The output of a single analysis is a `LanguageReport`, which the rest of the
system treats as the authoritative record of "what language was this, and did it
mix languages?" — the router, memory, validator and pipeline all key off it.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import config

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Language registry
# ─────────────────────────────────────────────────────────────────────────────

# Canonical codes we support. Everything else is recognised-but-out-of-scope
# and reported as "unsupported" so the assistant can still answer in English.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "or": "Odia",
    "bn": "Bengali",
}

# Human names → code, for parsing explicit user requests ("answer in Odia").
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en": "en", "eng": "en",
    "hindi": "hi", "hi": "hi", "hin": "hi", "हिन्दी": "hi", "हिंदी": "hi",
    "odia": "or", "oriya": "or", "or": "or", "ori": "or", "ଓଡ଼ିଆ": "or",
    "bengali": "bn", "bangla": "bn", "bn": "bn", "ben": "bn", "বাংলা": "bn",
}


# ─────────────────────────────────────────────────────────────────────────────
# Script detection (zero-dependency, exact)
# ─────────────────────────────────────────────────────────────────────────────

def _script_of(char: str) -> str:
    """Map a single character to a coarse script block."""
    code = ord(char)
    # U+0964/U+0965 (danda, double danda) are shared punctuation used by
    # Devanagari, Odia AND Bengali. Counting them as "devanagari" made every
    # Odia/Bengali sentence ending in । look like it contained Hindi, so they
    # are excluded from the script profile entirely.
    if code in (0x0964, 0x0965):
        return "none"
    if 0x0900 <= code <= 0x097F:
        return "devanagari"
    if 0x0B00 <= code <= 0x0B7F:
        return "oriya"
    if 0x0980 <= code <= 0x09FF:
        return "bengali"
    if (
        0x0041 <= code <= 0x005A
        or 0x0061 <= code <= 0x007A
        or 0x00C0 <= code <= 0x024F
    ):
        return "latin"
    # Any other script (Arabic, CJK, Cyrillic, Tamil…) is tracked generically.
    if char.isalpha():
        try:
            name = unicodedata.name(char)
            return name.split()[0].lower()
        except ValueError:
            return "other"
    return "none"


def script_profile(text: str) -> dict[str, int]:
    """Count alphabetic characters per script."""
    counts: dict[str, int] = {}
    for char in text:
        script = _script_of(char)
        if script in ("none", "other"):
            continue
        counts[script] = counts.get(script, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Roman-script heuristics (English vs Hinglish)
# ─────────────────────────────────────────────────────────────────────────────

# High-frequency Hindi words that survive in Roman transliteration. Curated to
# avoid collisions with English (no "par", "us", "is", "so" — those are English
# too and would create false positives).
_ROMAN_HINDI_MARKERS = {
    "kya", "kyu", "kyun", "kyon", "hai", "hain", "hoga", "hogi", "hota", "hoti",
    "nahi", "nahin", "nhi", "kaise", "kaisa", "kaisi", "karo", "kar", "karna",
    "karta", "karti", "karte", "raha", "rahi", "rahe", "mera", "meri", "mere",
    "tera", "teri", "apna", "apni", "tum", "tumhara", "aap", "aapka", "hum",
    "humein", "humko", "mujhe", "tujhe", "isko", "usko", "iske", "uske", "inhe",
    "unhe", "yeh", "ye", "woh", "wo", "aur", "lekin", "magar", "phir", "abhi",
    "ab", "bata", "batao", "samjha", "samjhao", "samjhaao", "dikhao", "chahiye",
    "zaroorat", "matlab", "arth", "kaam", "tarika", "tarika", "banana",
    "banau", "banao", "chala", "chal", "ruk", "dekho", "suno", "theek", "thik",
    "acha", "achha", "accha", "bohot", "bahut", "zyada", "kam", "thoda",
    "koi", "kuch", "sab", "log", "cheez", "baat", "sawal", "jawab", "galti",
    "error", "problem", "seekh", "sikh", "padhai", "kitab",
}

# Common English function words / tech terms — evidence of English.
_ENGLISH_MARKERS = {
    "the", "is", "are", "was", "were", "what", "how", "why", "when", "where",
    "which", "who", "explain", "please", "tell", "about", "with", "from",
    "that", "this", "these", "those", "there", "and", "or", "but", "if",
    "then", "than", "does", "did", "do", "can", "could", "would", "should",
    "will", "not", "no", "yes", "have", "has", "had", "you", "your", "we",
    "our", "they", "their", "give", "show", "define", "difference", "between",
    "example", "code", "error", "python", "model", "data", "training", "api",
    "role", "function", "mean", "means",
}

# Devanagari/Odia/Bengali words that appear *inside* otherwise-Latin text or
# appear as deliberate code-mixes. Used to flag mixed-language input.
_SCRIPT_NAMES = {
    "devanagari": "hi",
    "oriya": "or",
    "bengali": "bn",
    "latin": "en",
}


def roman_script_language(text: str) -> tuple[str, float, dict[str, float]]:
    """
    Decide English vs Hinglish for purely Roman-script text.

    Returns (code, confidence, scores). Confidence is the margin between the
    winning language and the runner-up, squashed into 0.5–0.99. Whitespace /
    empty input yields a low-confidence "en" default.
    """
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    total = len(tokens)
    if total == 0:
        return "en", 0.5, {"en": 0.0, "hi": 0.0}

    hindi_hits = sum(1 for t in tokens if t in _ROMAN_HINDI_MARKERS)
    english_hits = sum(1 for t in tokens if t in _ENGLISH_MARKERS)

    # Normalise by the number of *content* markers actually seen. A message
    # with two Hindi markers and zero English ones is Hinglish even if it is
    # short; a sentence with balanced signals is genuinely mixed and the tie
    # goes to English (the safer default for a technical assistant).
    hindi_score = hindi_hits / total
    english_score = english_hits / total

    if hindi_hits == 0 and english_hits == 0:
        # No lexical evidence at all ("Why?", "Give one real example."). The
        # neutral answer is English, but the confidence is deliberately below the
        # tie-break threshold so the caller's language `hint` — the language of
        # the previous turn — can win. That is what keeps a short follow-up in
        # the conversation's language instead of snapping back to English.
        return "en", 0.4, {"en": english_score, "hi": hindi_score}

    if hindi_hits > english_hits:
        margin = (hindi_hits - english_hits) / total
        confidence = min(0.95, 0.6 + margin)
        return "hi", confidence, {"en": english_score, "hi": hindi_score}

    margin = (english_hits - hindi_hits) / total
    confidence = min(0.95, 0.6 + margin)
    return "en", confidence, {"en": english_score, "hi": hindi_score}


# ─────────────────────────────────────────────────────────────────────────────
# Optional statistical backend (FastText / lingua)
# ─────────────────────────────────────────────────────────────────────────────

_FASTTEXT_MODEL = None
_FASTTEXT_TRIED = False
_LINGUA_DETECTOR = None
_LINGUA_TRIED = False


def _load_fasttext():
    """Lazily load FastText's `lid.176` model if it is available locally."""
    global _FASTTEXT_MODEL, _FASTTEXT_TRIED
    if _FASTTEXT_TRIED:
        return _FASTTEXT_MODEL
    _FASTTEXT_TRIED = True

    if os.getenv("LANGUAGE_BACKEND", "").lower() == "heuristic":
        return None

    path = os.getenv("FASTTEXT_LID_PATH", "").strip()
    if not path:
        # A sensible default location if the user downloaded the model.
        for candidate in (
            config.BASE_DIR / "models" / "lid.176.ftz",
            config.BASE_DIR / "models" / "lid.176.bin",
        ):
            if candidate.exists():
                path = str(candidate)
                break
    if not path:
        return None

    try:
        import fasttext  # type: ignore

        _FASTTEXT_MODEL = fasttext.load_model(path)
        log.info("FastText language ID loaded from %s", path)
    except Exception as exc:  # noqa: BLE001
        log.warning("FastText unavailable (%s); using script heuristics", exc)
        _FASTTEXT_MODEL = None
    return _FASTTEXT_MODEL


def _load_lingua():
    """Lazily build a lingua detector restricted to our supported languages."""
    global _LINGUA_DETECTOR, _LINGUA_TRIED
    if _LINGUA_TRIED:
        return _LINGUA_DETECTOR
    _LINGUA_TRIED = True

    if os.getenv("LANGUAGE_BACKEND", "").lower() == "heuristic":
        return None

    try:
        from lingua import Language, LanguageDetectorBuilder  # type: ignore

        _LINGUA_DETECTOR = (
            LanguageDetectorBuilder.from_languages(
                Language.ENGLISH, Language.HINDI, Language.BENGALI
            )
            .with_preloaded_language_models()
            .build()
        )
        log.info("lingua language detector ready")
    except Exception:  # noqa: BLE001
        _LINGUA_DETECTOR = None
    return _LINGUA_DETECTOR


def _fasttext_predict(text: str) -> Optional[tuple[str, float]]:
    model = _load_fasttext()
    if model is None:
        return None
    try:
        labels, scores = model.predict(text.replace("\n", " "), k=1)
        if not labels:
            return None
        code = labels[0].replace("__label__", "").strip().lower()
        return code, float(scores[0])
    except Exception:  # noqa: BLE001
        return None


def _lingua_predict(text: str) -> Optional[tuple[str, float]]:
    detector = _load_lingua()
    if detector is None:
        return None
    try:
        from lingua import Language  # type: ignore

        mapping = {Language.ENGLISH: "en", Language.HINDI: "hi", Language.BENGALI: "bn"}
        confidence_map = detector.compute_language_confidence_values(text)
        if not confidence_map:
            return None
        best = confidence_map[0]
        code = mapping.get(best.language)
        if code is None:
            return None
        return code, float(best.value)
    except Exception:  # noqa: BLE001
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LanguageReport:
    """
    The authoritative answer to "what language is this message?".

    Attributes
    ----------
    code
        Dominant language code (``en`` / ``hi`` / ``or`` / ``bn``), or the
        detected ISO code when it is outside our supported set.
    name
        Human-readable name of ``code``.
    confidence
        0–1 confidence in the *dominant* language.
    mixed
        True when the message visibly contains more than one language.
    mixed_languages
        The other languages present (excluding the dominant one).
    script
        The dominant Unicode script (``latin`` / ``devanagari`` / …).
    backend
        Which detector produced the decision (``script`` / ``fasttext`` /
        ``lingua`` / ``heuristic``), for the developer panel.
    supported
        Whether ``code`` is one we can answer in natively.
    reasoning
        A short human-readable explanation, surfaced in developer mode.
    """

    code: str = "en"
    confidence: float = 0.5
    mixed: bool = False
    mixed_languages: list[str] = field(default_factory=list)
    script: str = "latin"
    backend: str = "heuristic"
    supported: bool = True
    reasoning: str = ""

    @property
    def name(self) -> str:
        return SUPPORTED_LANGUAGES.get(self.code, self.code.upper())

    def is_supported(self) -> bool:
        return self.code in SUPPORTED_LANGUAGES

    def to_dict(self) -> dict:
        return {
            "detected_language": self.code,
            "language_name": self.name,
            "confidence": round(self.confidence, 3),
            "mixed_languages": list(self.mixed_languages),
            "script": self.script,
            "backend": self.backend,
            "supported": self.is_supported(),
            "reasoning": self.reasoning,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────


class LanguageDetector:
    """
    Detect language + code-mixing for a message.

    The class is intentionally cheap to construct and stateless per call, so it
    can be shared or created per session without caching concerns.
    """

    def __init__(self, min_chars_for_statistical: int = 12) -> None:
        # Very short strings ("Why?") are noise for statistical detectors; the
        # script + lexicon path handles them better.
        self.min_chars_for_statistical = min_chars_for_statistical

    # ── public API ──────────────────────────────────────────────────────────

    def detect(self, text: str, hint: Optional[str] = None) -> LanguageReport:
        """
        Analyse ``text`` and return a `LanguageReport`.

        ``hint`` is an optional previously-known language code (e.g. the
        language of the previous turn). It is used only to break ties — never to
        override clear evidence — which is what lets the assistant follow a user
        who switches language mid-conversation without losing short follow-ups
        like "Why?" to the wrong language.
        """
        text = (text or "").strip()
        if not text:
            return LanguageReport(
                code=hint or "en",
                confidence=0.4,
                backend="empty",
                reasoning="Empty message; inheriting previous language",
            )

        counts = script_profile(text)
        total_alpha = sum(counts.values()) or 1
        dominant_script = max(counts, key=lambda s: counts[s]) if counts else "latin"
        dominant_share = counts.get(dominant_script, 0) / total_alpha

        # ── 1. Supported non-Latin script present → very strong signal ──────
        # Odia and Bengali are distinct blocks; Devanagari is Hindi. These are
        # exact, so we can be confident without any statistical model.
        non_latin = {
            s: n for s, n in counts.items() if s in ("devanagari", "oriya", "bengali")
        }
        if non_latin:
            return self._report_from_scripts(counts, dominant_script, dominant_share)

        # ── 1b. Latin is NOT the dominant script → an unsupported script ────
        # A message written entirely in Tamil, Cyrillic, CJK, etc. must not fall
        # through to the Latin path and masquerade as English. Report it as
        # out-of-scope so the pipeline can answer in the fallback language.
        if dominant_script != "latin" and counts.get("latin", 0) < sum(counts.values()) / 2:
            return LanguageReport(
                code="und",
                confidence=0.6,
                script=dominant_script,
                backend="script",
                supported=False,
                reasoning=(
                    f"Dominant script '{dominant_script}' is outside the supported "
                    "set (en/hi/or/bn)"
                ),
            )

        # ── 2. Mostly Latin script → English vs Hinglish ────────────────────
        return self._report_from_latin(text, hint)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _report_from_scripts(
        self,
        counts: dict[str, int],
        dominant_script: str,
        dominant_share: float,
    ) -> LanguageReport:
        code = _SCRIPT_NAMES.get(dominant_script, "en")
        # Any *other* language-script present alongside the dominant one is
        # code-mixing (e.g. Devanagari sentence with an English phrase).
        others = sorted(
            {
                _SCRIPT_NAMES[s]
                for s, n in counts.items()
                if s in _SCRIPT_NAMES and _SCRIPT_NAMES[s] != code and n > 0
            }
        )
        mixed = bool(others) and dominant_share < 0.9

        reasoning = (
            f"Dominant script '{dominant_script}' "
            f"({dominant_share:.0%} of letters) → {SUPPORTED_LANGUAGES.get(code, code)}"
        )
        if mixed:
            reasoning += f"; code-mixing with {', '.join(others)}"

        return LanguageReport(
            code=code,
            confidence=min(0.99, 0.7 + dominant_share * 0.3),
            mixed=mixed,
            mixed_languages=others,
            script=dominant_script,
            backend="script",
            supported=code in SUPPORTED_LANGUAGES,
            reasoning=reasoning,
        )

    def _report_from_latin(self, text: str, hint: Optional[str]) -> LanguageReport:
        # Try a statistical backend first when the text is long enough to be
        # informative; fall back to the lexical heuristic.
        statistical: Optional[tuple[str, float]] = None
        backend = "heuristic"
        if len(text) >= self.min_chars_for_statistical:
            statistical = _fasttext_predict(text)
            if statistical:
                backend = "fasttext"
            else:
                statistical = _lingua_predict(text)
                if statistical:
                    backend = "lingua"

        code, heuristic_conf, scores = roman_script_language(text)

        # Zero lexical evidence in *any* supported language (a bare numeral, a
        # tick, "OK") — the only sensible signal is the conversation's own
        # language, so inherit it rather than defaulting to English. This is
        # what keeps a one-word follow-up in the conversation's language.
        if (
            not scores.get("en")
            and not scores.get("hi")
            and hint in SUPPORTED_LANGUAGES
        ):
            return LanguageReport(
                code=hint,
                confidence=0.55,
                script="latin",
                backend=f"{backend}+hint",
                reasoning=(
                    "No lexical markers in this short message; inheriting the "
                    f"previous language ({hint})"
                ),
            )

        if statistical is not None:
            stat_code, stat_conf = statistical
            # FastText returns a 176-language code (e.g. "hi", "en", "ne").
            # Only trust it when it names one of our languages; otherwise defer
            # to the lexicon, which is limited to our supported set.
            if stat_code in SUPPORTED_LANGUAGES and stat_conf >= 0.55:
                code = stat_code
                confidence = min(0.97, 0.55 + stat_conf * 0.45)
            else:
                confidence = heuristic_conf
                backend = f"{backend}+heuristic"
        else:
            confidence = heuristic_conf

        # Tie / weak-evidence resolution using the previous language.
        if hint and scores.get("en", 0.0) == scores.get("hi", 0.0):
            if hint in ("en", "hi"):
                code = hint
                confidence = max(confidence, 0.6)
                backend = f"{backend}+hint"

        # Mixed = both English and Hindi markers show real presence.
        mixed = False
        mixed_langs: list[str] = []
        hindi_present = scores.get("hi", 0.0) > 0.08
        english_present = scores.get("en", 0.0) > 0.08
        if code == "hi" and english_present:
            mixed = True
            mixed_langs = ["en"]
        elif code == "en" and hindi_present:
            mixed = True
            mixed_langs = ["hi"]

        reasoning = (
            f"Roman-script text via {backend}; "
            f"english_markers={scores.get('en', 0):.2f} "
            f"hindi_markers={scores.get('hi', 0):.2f}"
        )
        if mixed:
            reasoning += f"; mixed with {', '.join(mixed_langs)}"

        return LanguageReport(
            code=code,
            confidence=confidence,
            mixed=mixed,
            mixed_languages=mixed_langs,
            script="latin",
            backend=backend,
            supported=code in SUPPORTED_LANGUAGES,
            reasoning=reasoning,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Requested-output-language parsing
# ─────────────────────────────────────────────────────────────────────────────

# "answer in English", "reply in Hindi", "Odia me batao", "বাংলায় বলুন"…
_REQUEST_PHRASES = [
    r"\b(?:answer|reply|respond|explain|write|say|tell(?:\s+me)?|speak|"
    r"translate(?:\s+(?:it|this))?(?:\s+to)?|in|into)\s*(?:in\s+)?(english|hindi|"
    r"odia|oriya|bengali|bangla|हिन्दी|हिंदी|ଓଡ଼ିଆ|বাংলা)\b",
    r"\b(english|hindi|odia|oriya|bengali|bangla)\s*(?:mein|me|ma|में|ରେ|য়ে)\b",
    r"(?:in|into)\s+(english|hindi|odia|bengali|bangla)\b",
]
_REQUEST_RE = re.compile("|".join(_REQUEST_PHRASES), re.IGNORECASE)

# Explicit transactional switches: "ab hindi mein samjhao", "now in Odia".
_SWITCH_RE = re.compile(
    r"\b(?:ab|now|phir|next|then|aage)\b[^.!?]{0,20}?\b"
    r"(english|hindi|odia|oriya|bengali|bangla|हिन्दी|हिंदी|ଓଡ଼ିଆ|বাংলা)\b",
    re.IGNORECASE,
)


def _alias_to_code(token: str) -> Optional[str]:
    return _LANGUAGE_ALIASES.get(token.strip().lower())


def detect_requested_language(text: str) -> Optional[str]:
    """
    Parse an explicit request for the *answer* language.

    Returns a language code when the user asks for a specific output language
    (``"explain in Hindi"`` → ``"hi"``), else None. This is distinct from the
    detected *input* language: a Hinglish question can explicitly ask for an
    English answer.
    """
    if not text:
        return None
    for pattern in (_SWITCH_RE, _REQUEST_RE):
        match = pattern.search(text)
        if not match:
            continue
        for group in match.groups():
            if not group:
                continue
            code = _alias_to_code(group)
            if code:
                return code
    return None


_cached_detector: Optional[LanguageDetector] = None


def get_detector() -> LanguageDetector:
    """A process-wide detector (cheap, but avoids re-loading backends)."""
    global _cached_detector
    if _cached_detector is None:
        _cached_detector = LanguageDetector()
    return _cached_detector


@lru_cache(maxsize=256)
def detect_language_cached(text: str, hint: Optional[str] = None) -> LanguageReport:
    """
    Cached convenience wrapper. `lru_cache` needs hashable args, hence the
    primitive signature; the returned report is treated as immutable by callers.
    """
    return get_detector().detect(text, hint=hint)
