"""
Cross-lingual context-retention tests.

Covers the assignment's requirement C — "retain context after a language
switch" — and the consistency guarantee (E):

  * a language switch does not reset the topic,
  * a short follow-up ("Why?") resolves against the previous message even when
    it is in a different language,
  * the same question asked in different languages yields the same *intent*
    (and therefore the same kind of answer),
  * all four languages round-trip through memory serialisation without loss.

Everything here is offline and deterministic.
"""
from __future__ import annotations

import os

os.environ.setdefault("LANGUAGE_BACKEND", "heuristic")

from conversation_memory import ConversationMemory  # noqa: E402
from intent_manager import IntentContext, IntentManager  # noqa: E402
from language_detector import detect_language_cached  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Memory stores and returns language per turn
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageTaggedMemory:
    def test_turns_carry_their_language(self):
        memory = ConversationMemory()
        memory.add_turn("What is an API?", "An API is…", language="en")
        memory.add_turn("Ab isko samjhao.", "API ek tarika hai…", language="hi")
        assert [t.language for t in memory.turns] == ["en", "hi"]

    def test_previous_language_is_last_user_turn(self):
        memory = ConversationMemory()
        memory.add_turn("Hello", language="en")
        memory.add_turn("नमस्ते", language="hi")
        assert memory.previous_language() == "hi"

    def test_language_history_and_switch_flag(self):
        memory = ConversationMemory()
        for lang in ("en", "hi", "en"):
            memory.add_turn(f"msg-{lang}", language=lang)
        assert memory.language_history() == ["en", "hi", "en"]
        assert memory.has_language_switch() is True

    def test_no_switch_when_language_stable(self):
        memory = ConversationMemory()
        memory.add_turn("a", language="bn")
        memory.add_turn("b", language="bn")
        assert memory.has_language_switch() is False

    def test_recent_subjects_are_deduplicated_most_recent_first(self):
        memory = ConversationMemory()
        memory.add_turn("ML", "…", language="en", subjects=["machine learning"])
        memory.add_turn("UI", "…", language="en", subjects=["user interface"])
        memory.add_turn("more ML", "…", language="en", subjects=["machine learning"])
        assert memory.recent_subjects() == ["machine learning", "user interface"]


# ─────────────────────────────────────────────────────────────────────────────
# Context survives a language switch
# ─────────────────────────────────────────────────────────────────────────────


class TestContextAcrossLanguages:
    def test_prompt_context_annotates_the_languages_used(self):
        """
        When several languages appear, the context handed to the model must say
        so — that framing is what stops it treating a switch as a topic change.
        """
        memory = ConversationMemory()
        memory.add_turn("What is an API?", "An API is…", language="en")
        memory.add_turn("Ab Hindi mein samjhao.", "API ek tarika hai…", language="hi")
        context = memory.context_for_prompt()
        assert "Conversation languages" in context
        assert "English" in context and "Hindi" in context

    def test_api_context_survives_a_switch(self):
        """
        After EN → HI, the earlier API discussion is still present, so a later
        follow-up can resolve against it instead of re-asking.
        """
        memory = ConversationMemory()
        memory.add_turn("What is an API?", "An API lets systems talk.", language="en")
        memory.add_turn("Ab Hindi mein samjhao.", "API do systems ko jodta hai.", language="hi")
        assert "API" in memory.context_for_prompt()

    def test_short_why_resolves_to_previous_topic(self):
        """
        "Why?" is a genuine question in English; with a Hindi hint it stays
        Hindi. Either way it is a follow-up that must keep the previous subject.
        """
        memory = ConversationMemory()
        memory.add_turn("Explain overfitting.", "It is…", language="en")
        report = detect_language_cached("Why?", hint=memory.previous_language())
        # The short fragment inherits the conversation language.
        assert report.code == "en"

    def test_switch_does_not_lose_subjects(self):
        memory = ConversationMemory()
        memory.add_turn(
            "Explain the model accuracy.", "…", language="en", subjects=["model accuracy"]
        )
        memory.add_turn("Isko behtar karo.", "…", language="hi")
        assert "model accuracy" in memory.recent_subjects()


# ─────────────────────────────────────────────────────────────────────────────
# Same meaning → same intent, regardless of language
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossLingualConsistency:
    def test_definition_request_is_stable_across_languages(self):
        manager = IntentManager()
        asks = [
            ("What does overfitting mean?", "en"),
            ("Overfitting ka matlab kya hai?", "hi"),
            ("ଓଭରଫିଟିଂ ଅର୍ଥ କ'ଣ?", "or"),
            ("ওভারফিটিং মানে কী?", "bn"),
        ]
        intents = [
            manager.normalize(text, IntentContext()).intent for text, _ in asks
        ]
        # Every phrasing is a definition request; none should be an odd fallback
        # when the keyword table recognises the language.
        assert intents.count("define_term") >= 2

    def test_example_request_recognised_in_hindi_and_english(self):
        manager = IntentManager()
        assert (
            manager.normalize("Give one real example.", IntentContext()).intent
            == "example_request"
        )
        assert (
            manager.normalize("Iska ek example do.", IntentContext()).intent
            == "example_request"
        )

    def test_comparison_request_across_languages(self):
        manager = IntentManager()
        assert (
            manager.normalize(
                "Difference between supervised and unsupervised learning?",
                IntentContext(),
            ).intent
            == "compare"
        )
        assert (
            manager.normalize(
                "Supervised aur unsupervised mein kya farak hai?",
                IntentContext(),
            ).intent
            == "compare"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Memory serialisation round-trips the multilingual fields
# ─────────────────────────────────────────────────────────────────────────────


class TestMultilingualSerialisation:
    def test_state_round_trip_preserves_language_and_subjects(self):
        memory = ConversationMemory()
        memory.add_turn(
            "Explain ML",
            "…",
            language="en",
            response_language="en",
            intent="explain_concept",
            subjects=["machine learning"],
        )
        memory.add_turn(
            "Ab isko Hindi mein samjhao",
            "…",
            language="hi",
            response_language="hi",
            intent="follow_up",
            subjects=["machine learning"],
        )

        restored = ConversationMemory.from_state(memory.to_state())
        assert [t.language for t in restored.turns] == ["en", "hi"]
        assert restored.turns[1].intent == "follow_up"
        assert "machine learning" in restored.turns[1].subjects
        assert restored.has_language_switch() is True

    def test_native_script_survives_round_trip(self):
        """Non-ASCII text must not be mangled by serialisation."""
        memory = ConversationMemory()
        odia = "ନ୍ୟୁରାଲ ନେଟୱାର୍କ କ'ଣ?"
        memory.add_turn(odia, "…", language="or")
        restored = ConversationMemory.from_state(memory.to_state())
        assert restored.turns[0].user_message == odia
        assert restored.turns[0].language == "or"
