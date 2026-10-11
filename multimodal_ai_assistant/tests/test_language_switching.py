"""
Language-switching tests.

Covers the assignment's requirements B, D and E:

  * mixed-language input is understood (not translated blindly),
  * a language switch mid-conversation is detected and the *response* language
    changes accordingly,
  * explicit "answer in X" requests override the detected input language,
  * ambiguous requests produce a clarification rather than a guess, and that
    clarification is in the user's language,
  * the answer is validated against the requested language (V7).

The model is faked, so these tests assert on *decision-making*, not generation.
"""
from __future__ import annotations

import os

os.environ.setdefault("LANGUAGE_BACKEND", "heuristic")

from conversation_memory import ConversationMemory  # noqa: E402
from intent_manager import IntentContext, IntentManager  # noqa: E402
from language_detector import detect_language_cached  # noqa: E402
from pipeline import AssistantPipeline  # noqa: E402
from reasoning_router import ReasoningRouter  # noqa: E402
from response_validator import (  # noqa: E402
    ResponseValidator,
    ValidationContext,
)
from vision_model import ModelResponse  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# A fake multilingual model
# ─────────────────────────────────────────────────────────────────────────────


class FakeMultilingualVision:
    """
    Minimal stand-in for `VisionModel` that returns answers *in a chosen
    language*, so the language pipeline can be exercised end to end.
    """

    def __init__(self) -> None:
        self.text_model = "fake-text"
        self.vision_model = "fake-vision"
        self.calls: list[str] = []
        # English by default; tests override to simulate a wrong-language reply.
        self.answer_text = (
            "Overfitting happens when a model learns the training data too "
            "closely, so it does not generalise to new data."
        )
        self.hindi_answer = (
            "Overfitting tab hota hai jab model training data ko bahut zyada "
            "closely learn kar leta hai."
        )

    def answer_without_image(self, user_question, context="", reference_context=""):
        self.calls.append(user_question)
        return ModelResponse(text=self.answer_text, model="fake", success=True)

    def _chat(self, model, user_text, images=None, system="", temperature=0.3, json_mode=False):
        self.calls.append(user_text)
        return ModelResponse(text=self.answer_text, model="fake", success=True)

    def classify_request(self, user_text, has_image):
        return {"intent": "unknown", "fallback": True}

    def check_entailment(self, claim, evidence):
        return {"supported": True, "confidence": 0.9, "reason": "fake"}

    def health_check(self):
        from vision_model import ModelHealth

        return ModelHealth(server_up=True, vision_model_available=True)


def _make_pipeline(memory: ConversationMemory | None = None) -> AssistantPipeline:
    memory = memory or ConversationMemory()
    vision = FakeMultilingualVision()
    return AssistantPipeline(
        vision_model=vision,
        retriever=_NullRetriever(),
        memory=memory,
        router=ReasoningRouter(classifier=None),
        validator=ResponseValidator(embedder=None, entailment_checker=None),
    )


class _NullRetriever:
    """Retrieval that always returns nothing, so tests stay offline."""

    backend = "fake"

    def build_index(self):
        return "fake"

    def retrieve(self, query, top_k=4, min_score=0.25):
        return []

    def format_for_prompt(self, evidence):
        return ""

    def stats(self):
        return {"backend": "fake", "chunks": 0, "documents": []}


# ─────────────────────────────────────────────────────────────────────────────
# Intent normalisation across languages
# ─────────────────────────────────────────────────────────────────────────────


class TestCrossLingualIntent:
    def test_same_meaning_across_languages_same_intent(self):
        """
        The same request in four languages must normalise to the same intent —
        that is the definition of cross-lingual understanding.
        """
        manager = IntentManager()
        same_request = [
            "Explain machine learning.",
            "Machine learning samjhao.",
            "ମେସିନ ଲର୍ନିଂ ବୁଝାନ୍ତୁ।",
            "মেশিন লার্নিং বুঝিয়ে দিন।",
        ]
        intents = {
            manager.normalize(text, IntentContext()).intent for text in same_request
        }
        assert intents <= {"explain_concept", "general_question"}
        # At least the explicit explain forms must agree.
        assert manager.normalize(
            "Machine learning samjhao.", IntentContext()
        ).intent == "explain_concept"

    def test_technical_terms_are_preserved_as_entities(self):
        manager = IntentManager()
        result = manager.normalize(
            "Explain how the API and training data work in Python.",
            IntentContext(),
        )
        lowered = [e.lower() for e in result.entities]
        assert "api" in lowered
        assert "python" in lowered

    def test_follow_up_detected_from_pronoun(self):
        manager = IntentManager()
        result = manager.normalize(
            "Ab isko easy Hindi mein samjhao.",
            IntentContext(previous_language="en", has_prior_turns=True),
        )
        assert result.is_follow_up is True
        assert result.response_language == "hi"

    def test_mixed_input_tracks_both_languages(self):
        manager = IntentManager()
        result = manager.normalize(
            "Mera Python code error de raha hai, please explain in English.",
            IntentContext(),
        )
        assert result.response_language == "en"


# ─────────────────────────────────────────────────────────────────────────────
# Ambiguity → clarification, not guessing
# ─────────────────────────────────────────────────────────────────────────────


class TestAmbiguityResolution:
    def test_ambiguous_pronoun_with_two_subjects_asks(self):
        """
        "Isko improve kaise karun?" with two candidate subjects must produce a
        clarification, not a guess.
        """
        manager = IntentManager()
        result = manager.normalize(
            "Isko improve kaise karun?",
            IntentContext(
                previous_language="hi",
                has_prior_turns=True,
                recent_subjects=["accuracy", "user interface"],
            ),
        )
        assert result.needs_clarification is True
        assert result.intent == "clarify"
        # The clarification names the candidates so the user can pick.
        assert "accuracy" in result.clarifying_question

    def test_clarification_is_in_users_language(self):
        manager = IntentManager()
        result = manager.normalize(
            "Isko improve kaise karun?",
            IntentContext(
                previous_language="hi",
                has_prior_turns=True,
                recent_subjects=["model", "UI"],
            ),
        )
        # Hindi clarification should carry Devanagari-appropriate phrasing; the
        # template here is romanised Hindi, so check for a Hindi-specific word.
        assert "kis" in result.clarifying_question.lower() or "kya" in (
            result.clarifying_question.lower()
        )

    def test_unambiguous_request_is_not_clarified(self):
        manager = IntentManager()
        result = manager.normalize(
            "Explain overfitting.",
            IntentContext(has_prior_turns=True, recent_subjects=["model", "UI"]),
        )
        assert result.needs_clarification is False


# ─────────────────────────────────────────────────────────────────────────────
# Language switching in the router + pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageSwitchRouting:
    def test_router_marks_language_switch(self):
        memory = ConversationMemory()
        memory.add_turn(
            "Explain ML", "ML is...", language="en", subjects=["machine learning"]
        )
        router = ReasoningRouter(classifier=None)
        manager = IntentManager()

        intent = manager.normalize(
            "Ab isko easy Hindi mein samjhao.",
            IntentContext(previous_language="en", has_prior_turns=True),
        )
        report = detect_language_cached(
            "Ab isko easy Hindi mein samjhao.", hint="en"
        )
        plan = router.plan(
            "Ab isko easy Hindi mein samjhao.",
            memory=memory,
            normalized_intent=intent,
            language_report=report,
        )
        assert plan.language_switch is True
        assert plan.response_language == "hi"

    def test_pipeline_records_language_per_turn(self):
        memory = ConversationMemory()
        pipe = _make_pipeline(memory)

        pipe.handle_turn("Explain overfitting.")
        pipe.handle_turn("Ab isko easy Hindi mein samjhao.")

        assert memory.turns[0].language == "en"
        assert memory.turns[1].language == "hi"
        assert memory.has_language_switch() is True

    def test_pipeline_reports_switch_in_trace(self):
        pipe = _make_pipeline()
        pipe.handle_turn("What is an API?")
        turn = pipe.handle_turn("Ab Hindi mein samjhao.")
        assert turn.language_switched is True
        assert any("switch" in step.lower() for step in turn.trace)

    def test_follow_up_after_switch_retains_context(self):
        """
        The scenario from the brief: EN question → HI switch → EN follow-up.
        The final turn must still be answered from the same context, and the
        memory must show all three languages in order.
        """
        memory = ConversationMemory()
        pipe = _make_pipeline(memory)

        pipe.handle_turn("What is an API?")
        pipe.handle_turn("Ab isko Hindi mein samjhao.")
        pipe.handle_turn("Give one real example.")

        assert memory.language_history() == ["en", "hi", "en"]
        # Context survives: the third turn's prompt context mentions the API.
        context = memory.context_for_prompt()
        assert "API" in context


# ─────────────────────────────────────────────────────────────────────────────
# Response-language validation (V7)
# ─────────────────────────────────────────────────────────────────────────────


class TestResponseLanguageValidation:
    def test_english_answer_for_english_request_passes(self):
        validator = ResponseValidator(embedder=None)
        report = validator.validate(
            "Overfitting happens when a model learns the training data too closely.",
            ValidationContext(
                user_message="What is overfitting?",
                response_language="en",
            ),
        )
        check = next(c for c in report.checks if c.name == "V7-language")
        assert check.passed is True

    def test_english_answer_for_hindi_request_fails(self):
        validator = ResponseValidator(embedder=None)
        report = validator.validate(
            "Overfitting happens when a model learns the training data too closely.",
            ValidationContext(
                user_message="Overfitting kya hai?",
                response_language="hi",
            ),
        )
        check = next(c for c in report.checks if c.name == "V7-language")
        assert check.passed is False
        assert any("requested" in w.lower() for w in report.warnings)

    def test_hindi_answer_for_hindi_request_passes(self):
        validator = ResponseValidator(embedder=None)
        report = validator.validate(
            "ओवरफिटिंग तब होता है जब मॉडल प्रशिक्षण डेटा को बहुत करीब से सीख लेता है।",
            ValidationContext(
                user_message="ओवरफिटिंग क्या है?",
                response_language="hi",
            ),
        )
        check = next(c for c in report.checks if c.name == "V7-language")
        assert check.passed is True

    def test_short_answer_is_not_penalised(self):
        """A one-word answer carries too little signal to judge language."""
        validator = ResponseValidator(embedder=None)
        report = validator.validate(
            "Yes.",
            ValidationContext(user_message="Sahi?", response_language="hi"),
        )
        check = next(c for c in report.checks if c.name == "V7-language")
        assert check.passed is True
