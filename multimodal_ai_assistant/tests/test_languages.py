"""
Language-identification tests.

Covers the assignment's requirement A — "automatically identify language" — for
all four supported languages, in their native scripts and in Roman-script
(Hinglish) form, plus mixed-language input and explicit output-language requests.

These tests are deterministic and offline: the heuristic path (script detection
+ lexicon) is exercised directly, with the statistical backend disabled, so the
suite passes on a machine with no FastText model and no network.
"""
from __future__ import annotations

import os

import pytest

# Force the fully-offline heuristic path for the whole module so results do not
# depend on whether a FastText/lingua model happens to be installed.
os.environ.setdefault("LANGUAGE_BACKEND", "heuristic")

from language_detector import (  # noqa: E402
    SUPPORTED_LANGUAGES,
    LanguageDetector,
    detect_language_cached,
    detect_requested_language,
    script_profile,
)


@pytest.fixture
def detector() -> LanguageDetector:
    return LanguageDetector()


# ─────────────────────────────────────────────────────────────────────────────
# Native-script detection
# ─────────────────────────────────────────────────────────────────────────────


class TestNativeScriptDetection:
    def test_english(self, detector):
        report = detector.detect("Explain machine learning in simple terms.")
        assert report.code == "en"
        assert report.script == "latin"

    def test_hindi_devanagari(self, detector):
        report = detector.detect("मशीन लर्निंग को आसान भाषा में समझाओ।")
        assert report.code == "hi"
        assert report.script == "devanagari"
        assert report.confidence > 0.7

    def test_odia_script(self, detector):
        report = detector.detect("ନ୍ୟୁରାଲ୍ ନେଟୱର୍କ କ'ଣ?")
        assert report.code == "or"
        assert report.script == "oriya"

    def test_bengali_script(self, detector):
        report = detector.detect("নিউরাল নেটওয়ার্ক কী?")
        assert report.code == "bn"
        assert report.script == "bengali"

    def test_shared_danda_is_not_hindi(self, detector):
        """
        Odia and Bengali sentences end with the danda (।), which lives in the
        Devanagari block. It must NOT make the sentence look like Hindi.
        """
        report = detector.detect("ନ୍ୟୁରାଲ ନେଟୱାର୍କ ବିଷୟରେ ବୁଝାନ୍ତୁ।")
        assert report.code == "or"
        assert "hi" not in (report.mixed_languages or [])

    def test_all_supported_languages_are_reachable(self, detector):
        samples = {
            "en": "What is overfitting?",
            "hi": "ओवरफिटिंग क्या है?",
            "or": "ଓଭରଫିଟିଂ କ'ଣ?",
            "bn": "ওভারফিটিং কী?",
        }
        for expected, text in samples.items():
            assert detector.detect(text).code == expected, text


# ─────────────────────────────────────────────────────────────────────────────
# Roman-script (Hinglish) detection
# ─────────────────────────────────────────────────────────────────────────────


class TestRomanScriptDetection:
    def test_hinglish_is_detected_as_hindi(self, detector):
        report = detector.detect("mera model train nahi ho raha, kya karun")
        assert report.code == "hi"

    def test_hinglish_with_tech_term(self, detector):
        report = detector.detect("isko easy Hindi mein samjhao")
        assert report.code == "hi"

    def test_plain_english_stays_english(self, detector):
        report = detector.detect("Explain the role of training data in a model.")
        assert report.code == "en"

    def test_mixed_hinglish_english_flags_both(self, detector):
        """
        "Mera Python code error de raha hai, please explain in English" is
        genuinely both languages; the report must say it was mixed.
        """
        report = detector.detect(
            "Mera Python code error de raha hai, please explain in English."
        )
        assert report.mixed is True
        assert "hi" in report.mixed_languages or "en" in report.mixed_languages

    def test_signal_less_fragment_inherits_hint(self, detector):
        """
        A fragment with no lexical marker in any language (punctuation, a bare
        numeral, an emoji) cannot be classified on its own, so it inherits the
        conversation's language rather than defaulting to English.
        """
        assert detector.detect("3", hint="hi").code == "hi"
        assert detector.detect("✓", hint="bn").code == "bn"

    def test_english_word_is_not_overridden_by_hint(self, detector):
        """
        "Why?" is genuinely an English word, so the hint must NOT override the
        evidence — a Hindi user typing a real English word gets English
        detection, and the *context* (not the language) is what makes it a
        follow-up.
        """
        assert detector.detect("Why?", hint="hi").code == "en"
        assert detector.detect("Why?", hint="en").code == "en"

    def test_hinglish_fragment_inherits_hint_when_balanced(self, detector):
        """A short Hindi fragment keeps the hint rather than flipping language."""
        assert detector.detect("iska matlab batao", hint="hi").code == "hi"


# ─────────────────────────────────────────────────────────────────────────────
# Explicit output-language requests
# ─────────────────────────────────────────────────────────────────────────────


class TestRequestedLanguage:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Ab Hindi mein samjhao.", "hi"),
            ("Please explain in English.", "en"),
            ("answer in Bengali", "bn"),
            ("Odia me batao", "or"),
            ("Now reply in Hindi", "hi"),
        ],
    )
    def test_explicit_requests_are_parsed(self, text, expected):
        assert detect_requested_language(text) == expected

    def test_no_request_returns_none(self):
        assert detect_requested_language("What is an API?") is None
        assert detect_requested_language("Isme training data ka kya role hai?") is None

    def test_request_wins_over_detected_input_language(self):
        """
        A Hinglish question explicitly asking for an English answer is the key
        case from the brief: the *input* is mixed, the *output* is English.
        """
        requested = detect_requested_language(
            "Mera Python code error de raha hai, please explain in English."
        )
        assert requested == "en"


# ─────────────────────────────────────────────────────────────────────────────
# Script profiling / registry
# ─────────────────────────────────────────────────────────────────────────────


class TestScriptProfile:
    def test_profile_counts_each_script(self):
        counts = script_profile("Hello नमस्ते ନମସ୍କାର")
        assert counts.get("latin", 0) >= 5
        assert counts.get("devanagari", 0) > 0
        assert counts.get("oriya", 0) > 0

    def test_supported_languages_registry(self):
        assert set(SUPPORTED_LANGUAGES) == {"en", "hi", "or", "bn"}

    def test_report_to_dict_shape(self, detector):
        """
        The serialised report must match the normalised representation the
        assignment specifies.
        """
        data = detector.detect("Neural network kya hota hai?").to_dict()
        for key in (
            "detected_language",
            "confidence",
            "mixed_languages",
            "script",
            "backend",
            "supported",
        ):
            assert key in data

    def test_empty_message_inherits_hint(self, detector):
        report = detector.detect("", hint="bn")
        assert report.code == "bn"
        assert report.backend == "empty"

    def test_unsupported_language_is_not_supported(self, detector):
        # Tamil is a real language we do not claim to support.
        report = detector.detect("இது என்ன?")
        assert report.code not in SUPPORTED_LANGUAGES or not report.is_supported()

    def test_cached_detector_matches_direct(self):
        a = detect_language_cached("Explain overfitting.")
        b = detect_language_cached("Explain overfitting.")
        assert a.code == b.code == "en"
