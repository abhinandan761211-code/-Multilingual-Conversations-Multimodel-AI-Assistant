"""
multilingual_model.py
=====================
Language-aware wrapper over the local text model (Qwen3 via Ollama).

`vision_model.py` knows how to *talk to a model*. This module knows how to talk
to it **in the right language**. It adds three capabilities the multilingual
layer needs, without touching the vision pipeline:

1. **Language-aware prompting.** Every call injects an instruction to answer in
   the user's language, to preserve technical terms (API, Python, overfitting)
   rather than translating them, and to keep context even when the user switched
   language mid-conversation.

2. **Intent normalisation.** ``normalize_intent`` asks the model for the
   structured JSON representation (intent / entities / clarification), which
   `intent_manager` validates with Pydantic. Failure is safe — the caller keeps
   its heuristic result.

3. **Response-language correction.** ``translate_answer`` re-renders an answer
   that came back in the wrong language, and ``detect_response_language``
   reports which language the model actually produced.

All calls reuse `VisionModel._chat`, so there is exactly one code path that
talks to Ollama and one place where errors are handled.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import config
from language_detector import (
    SUPPORTED_LANGUAGES,
    LanguageReport,
    detect_language_cached,
)
from vision_model import ModelResponse, VisionModel, _safe_json

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Persona / instruction fragments per language
# ─────────────────────────────────────────────────────────────────────────────

# The base system prompt, per language. English is the fallback for anything
# unsupported. Keeping these short matters: they are prepended to every call.
_SYSTEM_BY_LANGUAGE: dict[str, str] = {
    "en": (
        "You are a careful multimodal research assistant. "
        "Answer in ENGLISH, even if the question mixes languages. "
        "Keep technical terms (API, Python, model, overfitting) untranslated. "
        "Use the conversation history to preserve context across language "
        "switches. If something is unclear, say so instead of guessing."
    ),
    "hi": (
        "Aap ek savdhani se kaam karne wale multimodal research assistant hain. "
        "Jawab HINDI (Devanagari) mein dein, chahe sawal do bhashaon ka mishran "
        "ho. Technical shabd (API, Python, model, overfitting) ka anuvad na "
        "karein — unhe waise hi rakhein. Bhasha badalne par bhi pichli baat-"
        "cheet ka sandarbh banaye rakhein. Kuchh saaf na ho to saaf-saaf "
        "bataayein, andaaza na lagayein."
    ),
    "or": (
        "ଆପଣ ଏକ ସତର୍କ ମଲ୍ଟିମୋଡାଲ୍ ରିସର୍ଚ୍ଚ ସହାୟକ। "
        "ଉତ୍ତର ODIA (ଓଡ଼ିଆ) ରେ ଦିଅନ୍ତୁ, ପ୍ରଶ୍ନ ଦୁଇ ଭାଷାର ମିଶ୍ରଣ ହେଲେ ମଧ୍ୟ। "
        "ଟେକ୍ନିକାଲ୍ ଶବ୍ଦ (API, Python, model, overfitting) ଅନୁବାଦ କରନ୍ତୁ ନାହିଁ। "
        "ଭାଷା ବଦଳିଲେ ମଧ୍ୟ ପୂର୍ବ କଥାବାର୍ତ୍ତାର ପ୍ରସଙ୍ଗ ରଖନ୍ତୁ। "
        "କିଛି ଅସ୍ପଷ୍ଟ ହେଲେ ଅନୁମାନ କରିବା ପରିବର୍ତ୍ତେ ସ୍ପଷ୍ଟ କୁହନ୍ତୁ।"
    ),
    "bn": (
        "আপনি একজন সতর্ক মাল্টিমোডাল রিসার্চ সহকারী। "
        "উত্তর BENGALI (বাংলা) এ দিন, প্রশ্ন দুই ভাষার মিশ্রণ হলেও। "
        "প্রযুক্তিগত শব্দ (API, Python, model, overfitting) অনুবাদ করবেন না। "
        "ভাষা বদলালেও আগের কথোপকথনের প্রসঙ্গ ধরে রাখুন। "
        "কিছু স্পষ্ট না হলে অনুমান না করে স্পষ্ট করে বলুন।"
    ),
}


def system_prompt_for(language: str) -> str:
    """The system prompt for one answer language (English fallback)."""
    return _SYSTEM_BY_LANGUAGE.get(language, _SYSTEM_BY_LANGUAGE["en"])


# The explicit output-language instruction appended to each user turn.
_OUTPUT_INSTRUCTION: dict[str, str] = {
    "en": "Answer in English.",
    "hi": "Jawab Hindi mein dein (Devanagari script).",
    "or": "ଉତ୍ତର ଓଡ଼ିଆରେ ଦିଅନ୍ତୁ (Odia script).",
    "bn": "উত্তর বাংলায় দিন (Bengali script).",
}


@dataclass
class LanguageAwareAnswer:
    """One multilingual generation, with what we can say about its language."""

    text: str
    response_language: str = "en"
    requested_language: str = "en"
    language_ok: bool = True
    model: str = ""
    latency_s: float = 0.0
    error: Optional[str] = None
    corrected: bool = False          # True if we re-rendered it in the right language


class MultilingualModel:
    """
    Language-aware reasoning over the local model.

    Every method is defensive: on any failure it returns a result carrying an
    ``error`` rather than raising, because a language-handling bug must never
    take down a turn that the vision pipeline could otherwise complete.
    """

    def __init__(
        self,
        vision_model: Optional[VisionModel] = None,
        translation_model: Optional[str] = None,
    ) -> None:
        self.vision = vision_model or VisionModel()
        self.translation_model = translation_model or config.TRANSLATION_MODEL or ""
        self._validate_only = None  # reserved for future tightening

    # ── model access ────────────────────────────────────────────────────────

    def _text_model(self) -> str:
        return self.translation_model or self.vision.text_model

    def _chat(
        self,
        user_text: str,
        language: str,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        model: Optional[str] = None,
    ) -> ModelResponse:
        """One language-steered call to the text model."""
        temperature_value = (
            config.TEXT_TEMPERATURE if temperature is None else temperature
        )

        # Preferred path: the full system-prompt control that `VisionModel._chat`
        # gives us, which is how the answer language is pinned.
        private_chat = getattr(self.vision, "_chat", None)
        if callable(private_chat):
            return private_chat(
                model or self._text_model(),
                user_text,
                images=None,
                system=system_prompt_for(language),
                temperature=temperature_value,
                json_mode=json_mode,
            )

        # Fallback for test doubles (and any model wrapper) that only expose the
        # public API. Language steering is lost, but the turn still completes —
        # a missing optimisation must never break the pipeline.
        try:
            response = self.vision.answer_without_image(user_text)
        except Exception as exc:  # noqa: BLE001
            return ModelResponse(error=f"{type(exc).__name__}: {exc}")
        return response

    # ── main multilingual answer ────────────────────────────────────────────

    def answer(
        self,
        question: str,
        language: str,
        context: str = "",
        reference_context: str = "",
        extra_instructions: str = "",
        conversation_languages: Optional[list[str]] = None,
    ) -> LanguageAwareAnswer:
        """
        Answer ``question`` in ``language``.

        ``context`` is the conversation transcript (which may span several
        languages on purpose — that continuity is what preserves meaning).
        ``conversation_languages`` lets us tell the model that a language switch
        is expected, so it does not "correct" the user or treat it as an error.
        """
        parts = [
            "Answer the user's question using the conversation context as the "
            "source of continuity. The user may switch languages between turns — "
            "keep following the same topic and refer back to earlier points "
            "rather than starting fresh.",
        ]

        if conversation_languages and len(set(conversation_languages)) > 1:
            seen = ", ".join(
                SUPPORTED_LANGUAGES.get(c, c) for c in dict.fromkeys(conversation_languages)
            )
            parts.append(
                f"The conversation so far has used: {seen}. Match the user's "
                "current language for the answer; do not comment on the switch."
            )

        if context:
            parts.append(f"Conversation so far:\n{context}")
        if reference_context:
            parts.append(
                "Reference material (use it only if it genuinely helps):\n"
                f"{reference_context}"
            )
        parts.append(f"User's question: {question}")
        if extra_instructions:
            parts.append(extra_instructions)
        parts.append(_OUTPUT_INSTRUCTION.get(language, _OUTPUT_INSTRUCTION["en"]))

        response = self._chat("\n\n".join(parts), language=language)
        return self._finish(response, language, correct_if_needed=True)

    # ── greeting (deterministic, no model call) ─────────────────────────────

    # Pre-written greetings so a social turn costs nothing and never fails. A
    # greeting is not worth a model round-trip, and hard-coding it removes the
    # risk of a Hindi "namaste" being answered in English.
    _GREETINGS: dict[str, str] = {
        "en": (
            "Hello! I'm your multimodal assistant. Ask me a question in English, "
            "Hindi, Odia or Bengali — or upload an image (a chart, a diagram, a "
            "screenshot, a handwritten note) and I'll reason over it. You can "
            "switch languages mid-conversation and I'll keep the context."
        ),
        "hi": (
            "Namaste! Main aapka multimodal assistant hoon. Aap English, Hindi, "
            "Odia ya Bengali mein sawal poochh sakte hain — ya ek image "
            "(chart, diagram, screenshot, handwritten note) upload karein aur "
            "main us par vichaar karunga. Aap beech mein bhasha badal sakte "
            "hain; main sandarbh banaye rakhunga."
        ),
        "or": (
            "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କ ମଲ୍ଟିମୋଡାଲ୍ ସହାୟକ। ଆପଣ ଇଂରାଜୀ, ହିନ୍ଦୀ, ଓଡ଼ିଆ କିମ୍ବା "
            "ବଙ୍ଗଳାରେ ପ୍ରଶ୍ନ କରିପାରିବେ — କିମ୍ବା ଏକ ଛବି (ଚାର୍ଟ, ଡାଏଗ୍ରାମ୍, ସ୍କ୍ରିନସଟ୍) "
            "ଅପଲୋଡ୍ କରନ୍ତୁ ଏବଂ ମୁଁ ତାହା ବିଶ୍ଳେଷଣ କରିବି। ଆପଣ ମଝିରେ ଭାଷା ବଦଳାଇ "
            "ପାରିବେ; ମୁଁ ପ୍ରସଙ୍ଗ ରଖିବି।"
        ),
        "bn": (
            "নমস্কার! আমি আপনার মাল্টিমোডাল সহকারী। আপনি ইংরেজি, হিন্দি, ওড়িয়া বা "
            "বাংলায় প্রশ্ন করতে পারেন — অথবা একটি ছবি (চার্ট, ডায়াগ্রাম, স্ক্রিনশট) "
            "আপলোড করুন এবং আমি তা বিশ্লেষণ করব। আপনি মাঝপথে ভাষা বদলাতে পারেন; "
            "আমি প্রসঙ্গ ধরে রাখব।"
        ),
    }

    def greeting(self, language: str) -> str:
        """A localised greeting for a social turn (no model call)."""
        return self._GREETINGS.get(language, self._GREETINGS["en"])

    # ── clarification (kept in the user's language) ─────────────────────────

    def clarifying_question(
        self, question: str, language: str, candidates: Optional[list[str]] = None
    ) -> LanguageAwareAnswer:
        """
        Ask a clarifying question in the user's language.

        Falls back to the deterministic template in `intent_manager` when the
        model is unavailable, so clarification never depends on the model.
        """
        options = ""
        if candidates:
            options = "\nCandidate subjects the user may mean: " + ", ".join(candidates)
        prompt = (
            "The user's request is ambiguous, so you must ask ONE short "
            "clarifying question instead of answering. Do not guess.\n"
            f"User's request: {question}{options}\n"
            "Ask which subject they mean, offering the candidate subjects if any."
        )
        prompt += "\n" + _OUTPUT_INSTRUCTION.get(language, _OUTPUT_INSTRUCTION["en"])
        response = self._chat(prompt, language=language, temperature=0.2)
        return self._finish(response, language, correct_if_needed=False)

    # ── intent normalisation ────────────────────────────────────────────────

    def normalize_intent(
        self,
        text: str,
        context: Optional[dict] = None,
        fallback: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Ask the model for the structured, language-independent intent JSON.

        Returns a dict (or None on failure). The caller validates it with the
        `NormalizedIntent` Pydantic schema, so an off-spec response is discarded
        rather than trusted.
        """
        context = context or {}
        prompt = (
            "Normalise this user message into a language-independent JSON object.\n"
            f"Message: {json.dumps(text)}\n"
            f"Context: {json.dumps(context)}\n"
            f"Heuristic guess (may be wrong): {json.dumps(fallback or {})}\n\n"
            "Return ONLY JSON with these keys:\n"
            '  "intent": one of [explain_concept, define_term, how_to, compare, '
            'example_request, follow_up, translation_request, summarise, greeting, '
            'describe_image, read_text, analyse_chart, clarify, general_question]\n'
            '  "entities": array of technical terms / subject nouns (do NOT '
            'translate technical terms)\n'
            '  "needs_clarification": boolean\n'
            '  "clarifying_question": string in the SAME language as the message\n'
            '  "detected_language": ISO code (en/hi/or/bn)\n'
            '  "response_language": ISO code the answer should use\n'
            '  "confidence": number 0-1\n\n'
            "Rules: a message that points at earlier context with a pronoun "
            "('isko samjhao', 'it', 'why?') is a follow_up. If the request could "
            "reasonably mean more than one thing and you cannot tell which, set "
            "needs_clarification true."
        )
        response = self._chat(prompt, language="en", temperature=0.0, json_mode=True)
        if not response.success:
            return None
        parsed = _safe_json(response.text)
        return parsed if isinstance(parsed, dict) else None

    # ── response-language correction ────────────────────────────────────────

    def translate_answer(
        self, answer: str, target_language: str, source_language: str = ""
    ) -> LanguageAwareAnswer:
        """
        Re-render an answer that came back in the wrong language.

        This is a *re-expression*, not a blind translation: technical terms are
        preserved and the meaning is kept identical, which is exactly the
        "consistent responses regardless of language" requirement.
        """
        target_name = SUPPORTED_LANGUAGES.get(target_language, "English")
        source_note = (
            f"The text is currently in {SUPPORTED_LANGUAGES.get(source_language, source_language)}. "
            if source_language else ""
        )
        prompt = (
            "Rewrite the following answer in "
            f"{target_name}, preserving its meaning exactly and keeping all "
            "technical terms (API, Python, model, overfitting) untranslated. "
            "Do not add or remove information; do not change the structure "
            "beyond what the target language requires.\n"
            f"{source_note}\n"
            f"Answer to rewrite:\n{answer}\n\n"
            f"{_OUTPUT_INSTRUCTION.get(target_language, _OUTPUT_INSTRUCTION['en'])}"
        )
        response = self._chat(prompt, language=target_language, temperature=0.1)
        result = self._finish(response, target_language, correct_if_needed=False)
        result.corrected = True
        return result

    # ── helpers ─────────────────────────────────────────────────────────────

    def detect_response_language(self, text: str) -> LanguageReport:
        return detect_language_cached(text)

    def _finish(
        self,
        response: ModelResponse,
        requested_language: str,
        correct_if_needed: bool,
    ) -> LanguageAwareAnswer:
        text = response.text or ""
        answer = LanguageAwareAnswer(
            text=text,
            requested_language=requested_language,
            model=response.model,
            latency_s=response.latency_s,
            error=response.error,
        )
        if not text.strip():
            answer.language_ok = False
            return answer

        report = self.detect_response_language(text)
        answer.response_language = report.code
        answer.language_ok = self._language_matches(report, requested_language)

        # Correct a wrong-language answer once, when enabled. We do NOT correct
        # short strings like a single word or a code block, where language
        # detection is meaningless and a rewrite could damage the content.
        if (
            correct_if_needed
            and not answer.language_ok
            and config.ENFORCE_RESPONSE_LANGUAGE
            and len(text.split()) >= 4
        ):
            corrected = self.translate_answer(
                text, requested_language, source_language=report.code
            )
            if corrected.text.strip():
                answer.text = corrected.text
                answer.model = corrected.model
                answer.corrected = True
                new_report = self.detect_response_language(corrected.text)
                answer.response_language = new_report.code
                answer.language_ok = self._language_matches(
                    new_report, requested_language
                )
        return answer

    @staticmethod
    def _language_matches(report: LanguageReport, requested: str) -> bool:
        if report.code == requested:
            return True
        # For Latin-script targets, allow a code-mixed answer that is dominated
        # by the requested language (very common for technical Hindi/Bengali).
        if requested in ("en", "hi") and report.mixed and requested in report.mixed_languages:
            return True
        return False


def get_multilingual_model(vision_model: Optional[VisionModel] = None) -> MultilingualModel:
    return MultilingualModel(vision_model=vision_model)
