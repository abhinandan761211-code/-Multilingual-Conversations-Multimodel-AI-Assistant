"""
pipeline.py
===========
Orchestrates the assistant: router → tools → model → validator → answer.

`app.py` (Streamlit) and the tests both drive this class, so the full decision
logic lives in one place and is testable without a browser.

The flow per user turn:

    1. Validate/measure the image              (image_processor)
    2. Resolve context & register the image    (conversation_memory)
    3. Decide the handling strategy            (reasoning_router)
    4. Fetch supporting evidence, if needed    (evidence_retriever)
    5. Run the chosen tool(s):
         - OCR                                  (image_processor)
         - VLM describe / read / analyse / answer
    6. Validate the answer                     (response_validator)
    7. Attach an uncertainty note when warranted
    8. Record the turn                         (conversation_memory)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import config
from conversation_memory import ConversationMemory
from evidence_retriever import Evidence, EvidenceRetriever
from image_processor import (
    ImageDiagnostics,
    analyze_image,
    encode_for_vision,
    preprocess_for_reading,
    run_ocr,
)
from intent_manager import (
    IntentContext,
    IntentManager,
    NormalizedIntent,
    get_intent_manager,
)
from language_detector import (
    SUPPORTED_LANGUAGES,
    LanguageReport,
    detect_language_cached,
)
from multilingual_model import MultilingualModel, system_prompt_for
from reasoning_router import ReasoningRouter, Route, RoutePlan
from response_validator import (
    ResponseValidator,
    ValidationContext,
    ValidationReport,
    ValidationStatus,
)
from vision_model import VisionModel

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Turn result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AssistantTurn:
    """Everything one turn produced — the UI renders this, tests assert on it."""

    answer: str = ""
    route: str = ""
    plan: Optional[RoutePlan] = None
    diagnostics: Optional[ImageDiagnostics] = None
    ocr: Optional[Any] = None
    evidence: list[Evidence] = field(default_factory=list)
    validation: Optional[ValidationReport] = None
    trace: list[str] = field(default_factory=list)
    error: Optional[str] = None

    # Multilingual telemetry — everything the UI/dev panel needs to *show* that
    # language, intent and context were handled, rather than merely claiming it.
    language: Optional[LanguageReport] = None
    intent: Optional[NormalizedIntent] = None
    response_language: str = "en"
    language_switched: bool = False
    response_language_corrected: bool = False

    def trace_text(self) -> str:
        return "\n".join(f"{i}. {step}" for i, step in enumerate(self.trace, 1))

    @property
    def detected_language(self) -> str:
        return self.language.code if self.language else "en"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────


class AssistantPipeline:
    """
    Wires the components together.

    All collaborators are injectable so tests can supply fakes and so the app
    can share expensive objects (embedder, retriever) across reruns.
    """

    def __init__(
        self,
        vision_model: Optional[VisionModel] = None,
        retriever: Optional[EvidenceRetriever] = None,
        memory: Optional[ConversationMemory] = None,
        router: Optional[ReasoningRouter] = None,
        validator: Optional[ResponseValidator] = None,
        multilingual: Optional[MultilingualModel] = None,
        intent_manager: Optional[IntentManager] = None,
    ) -> None:
        self.vision = vision_model or VisionModel()
        self.retriever = retriever or EvidenceRetriever()
        self.memory = memory or ConversationMemory()
        self.router = router or ReasoningRouter(classifier=self.vision.classify_request)
        self.validator = validator or ResponseValidator(
            entailment_checker=self.vision.check_entailment
        )
        # The multilingual layer. It reuses `self.vision` for transport, so there
        # is still exactly one path that talks to Ollama.
        self.multilingual = multilingual or MultilingualModel(vision_model=self.vision)
        self.intent_manager = intent_manager or IntentManager(
            refiner=self.multilingual.normalize_intent
        )
        self._retriever_ready = False

    # ── setup ───────────────────────────────────────────────────────────────

    def ensure_knowledge_base(self) -> str:
        if not self._retriever_ready:
            backend = self.retriever.build_index()
            self._retriever_ready = True
            log.info("Evidence index ready (backend=%s)", backend)
        return self.retriever.backend

    # ── main entry point ────────────────────────────────────────────────────

    def handle_turn(
        self,
        user_message: str,
        image_bytes: Optional[bytes] = None,
        image_filename: str = "image",
    ) -> AssistantTurn:
        turn = AssistantTurn()
        user_message = (user_message or "").strip()

        # ── 1. image validation / measurement ───────────────────────────────
        diagnostics: Optional[ImageDiagnostics] = None
        image_id: Optional[str] = None
        image_b64: Optional[str] = None

        if image_bytes:
            diagnostics = analyze_image(image_bytes)
            turn.diagnostics = diagnostics
            if diagnostics.valid:
                turn.trace.append(
                    f"Image accepted: {diagnostics.width}x{diagnostics.height} "
                    f"{diagnostics.format}, sharpness={diagnostics.blur_score:.1f}"
                )
                if diagnostics.quality_notes:
                    turn.trace.append(
                        "Quality notes: " + "; ".join(diagnostics.quality_notes)
                    )
                record = self.memory.register_image(
                    image_bytes,
                    filename=image_filename,
                    width=diagnostics.width,
                    height=diagnostics.height,
                )
                image_id = record.image_id
                try:
                    image_b64 = encode_for_vision(image_bytes)
                except Exception as exc:  # noqa: BLE001
                    turn.trace.append(f"Could not encode image for the model: {exc}")
            else:
                turn.trace.append(f"Image rejected: {diagnostics.error}")

        # ── 2. language identification + intent normalisation ───────────────
        #
        # This runs BEFORE routing. Two reasons:
        #   * the router's heuristics are English-only, so the normalised intent
        #     (which the multilingual keyword tables understand) must be able to
        #     influence the route;
        #   * every later step — clarification, generation, validation — needs to
        #     agree on one language decision, and this is where it is made.
        previous_language = self.memory.previous_language()
        language = detect_language_cached(user_message, previous_language)
        turn.language = language
        turn.language_switched = bool(
            previous_language and language.code != previous_language
        )

        intent = self.intent_manager.normalize(
            user_message,
            IntentContext(
                previous_language=previous_language,
                has_image=diagnostics is not None and diagnostics.valid,
                has_prior_turns=bool(self.memory.turns),
                recent_subjects=self.memory.recent_subjects(),
                language_report=language,
            ),
            use_llm=False,
        )
        turn.intent = intent
        turn.response_language = intent.response_language

        turn.trace.append(
            f"Language: {language.name} ({language.code}, "
            f"confidence={language.confidence:.2f}, via {language.backend})"
            + (f" + mixed {language.mixed_languages}" if language.mixed else "")
        )
        if turn.language_switched:
            turn.trace.append(
                f"  ↳ Language switch detected: "
                f"{previous_language} → {language.code}; context retained"
            )
        turn.trace.append(
            f"Intent (normalised): {intent.intent} "
            f"(confidence={intent.confidence:.2f}, via {intent.source_of_resolution})"
        )
        if intent.entities:
            turn.trace.append(f"  ↳ Entities: {', '.join(intent.entities)}")
        if intent.response_language != language.code:
            turn.trace.append(
                f"  ↳ Explicit response-language request: "
                f"{intent.response_language}"
            )

        # ── 3. route ────────────────────────────────────────────────────────
        plan = self.router.plan(
            user_message=user_message,
            diagnostics=diagnostics,
            memory=self.memory,
            new_image_id=image_id,
            normalized_intent=intent,
            language_report=language,
        )
        turn.plan = plan
        turn.route = plan.route.value
        turn.trace.append(f"Router chose '{plan.route.value}' — {plan.explain()}")
        for reason in plan.rationale:
            turn.trace.append(f"  ↳ {reason}")

        # ── early exits: clarify / small talk need no model call ────────────
        if plan.route == Route.CLARIFY:
            # The clarifying question is already produced in the user's
            # language by the intent manager (or the router's template). We do
            # not translate it afterwards: a clarification that changes language
            # is exactly the kind of inconsistency this layer exists to prevent.
            turn.answer = plan.clarifying_question
            turn.validation = self._validate(
                turn.answer, plan, diagnostics, user_message,
                image_summary="", ocr=None, evidence=[],
                response_language=intent.response_language,
            )
            turn.trace.append("Returned a clarifying question instead of guessing")
            self.memory.add_turn(
                user_message, turn.answer, image_id=image_id,
                route=turn.route, validation_status=turn.validation.status.value,
                language=language.code,
                response_language=intent.response_language,
                intent=intent.intent,
                subjects=intent.entities,
            )
            return turn

        if plan.route == Route.SMALL_TALK:
            # Greet in the user's language — a multilingual assistant that
            # answers a Hindi greeting in English has not understood the user.
            turn.answer = self.multilingual.greeting(intent.response_language)
            turn.validation = self._validate(
                turn.answer, plan, diagnostics, user_message,
                image_summary="", ocr=None, evidence=[],
                response_language=intent.response_language,
            )
            turn.trace.append(
                f"Handled as a social turn in {intent.response_language}"
            )
            self.memory.add_turn(
                user_message, turn.answer, route=turn.route,
                validation_status=turn.validation.status.value,
                language=language.code,
                response_language=intent.response_language,
                intent=intent.intent,
                subjects=intent.entities,
            )
            return turn

        # ── 4. evidence retrieval ───────────────────────────────────────────
        if plan.use_retrieval and plan.retrieval_query:
            try:
                self.ensure_knowledge_base()
                turn.evidence = self.retriever.retrieve(plan.retrieval_query)
                turn.trace.append(
                    f"Retrieved {len(turn.evidence)} evidence chunk(s) "
                    f"(backend={self.retriever.backend})"
                )
                for e in turn.evidence:
                    turn.trace.append(f"  ↳ {e.citation()} (score={e.score})")
            except Exception as exc:  # noqa: BLE001
                turn.trace.append(f"Retrieval failed (continuing without): {exc}")
        reference_context = self.retriever.format_for_prompt(turn.evidence)

        # ── 5. tools: OCR then the vision model ─────────────────────────────
        ocr_result = None
        ocr_text = ""
        if plan.use_ocr and image_bytes:
            ocr_result = run_ocr(image_bytes, diagnostics)
            turn.ocr = ocr_result
            if ocr_result.attempted and ocr_result.succeeded:
                ocr_text = ocr_result.text
                turn.trace.append(
                    f"OCR ({ocr_result.backend}) extracted "
                    f"{len(ocr_text)} chars, confidence={ocr_result.confidence}"
                )
            else:
                turn.trace.append(
                    f"OCR not used: {ocr_result.error or 'no backend'}"
                )

        context = self.memory.context_for_prompt()
        stored_image_record = self.memory.get_image(plan.image_id)

        # The language pin that every generation path shares. For the vision
        # paths we append it to the user turn; the text paths get it via the
        # multilingual wrapper's system prompt.
        lang_instruction = (
            f"\n\nIMPORTANT: answer in "
            f"{intent.response_language} using its native script. "
            "Keep technical terms (API, Python, model, overfitting) untranslated."
        )

        answer = ""
        model_error = None
        image_summary = ""
        vlm_text = ""

        try:
            if plan.route == Route.READ_TEXT and image_b64:
                response = self.vision.extract_text_from_image(image_b64, ocr_hint=ocr_text)
                answer = response.text
                model_error = response.error
                vlm_text = answer
                turn.trace.append(
                    f"VLM text transcription ({response.model}, {response.latency_s}s)"
                )

            elif plan.route == Route.ANALYSE_CHART and image_b64:
                response = self.vision.analyse_chart(
                    image_b64, user_message + lang_instruction, context
                )
                answer = response.text
                model_error = response.error
                image_summary = answer
                turn.trace.append(
                    f"VLM chart analysis ({response.model}, {response.latency_s}s)"
                )

            elif plan.route == Route.DESCRIBE_IMAGE and image_b64:
                response = self.vision.describe_image(
                    image_b64, user_message + lang_instruction, context
                )
                answer = response.text
                model_error = response.error
                image_summary = answer
                turn.trace.append(
                    f"VLM description ({response.model}, {response.latency_s}s)"
                )

            elif plan.route in (Route.ANSWER_WITH_IMAGE,) and image_b64:
                response = self.vision.answer_with_image(
                    image_b64, user_message + lang_instruction, context,
                    reference_context,
                )
                answer = response.text
                model_error = response.error
                image_summary = answer
                turn.trace.append(
                    f"VLM answer ({response.model}, {response.latency_s}s)"
                )

            elif plan.route == Route.FOLLOW_UP:
                # Reuse what we already know about the image instead of
                # re-encoding pixels — that is the memory payoff.
                image_context = self.memory.image_context(stored_image_record)
                merged_context = "\n\n".join(
                    p for p in (context, image_context) if p
                )
                if image_context:
                    turn.trace.append(
                        "Reused stored image context (no re-upload needed)"
                    )
                role = (
                    "primary evidence" if image_context else "conversation only"
                )
                turn.trace.append(f"Follow-up answered from {role}")
                answer, model_error = self._answer_text(
                    user_message, intent.response_language, merged_context,
                    reference_context, turn
                )

            else:
                # GENERAL_QUESTION, or an image route where the image failed to
                # encode — both degrade to text-only reasoning.
                #
                # A self-contained question must be answered WITHOUT the image
                # context. Sending the earlier chart description alongside
                # "What is the capital of France?" made the model reply "the
                # image does not provide information about the capital of
                # France" — useful image discussion crowding out the answer.
                standalone = plan.route == Route.GENERAL_QUESTION
                if standalone:
                    # Keep only the turns that are not about a picture, so the
                    # model answers the question on its own terms.
                    question_context = self.memory.context_for_prompt(
                        exclude_images=True
                    )
                    turn.trace.append(
                        "Answered as a standalone question — image context withheld "
                        "so it can't displace the answer"
                    )
                else:
                    question_context = context
                    if ocr_text:
                        reference_context = (
                            f"{reference_context}\n\nText read from the image:\n{ocr_text}"
                        )

                answer, model_error = self._answer_text(
                    user_message, intent.response_language, question_context,
                    reference_context, turn
                )

        except Exception as exc:  # noqa: BLE001 - never crash the UI
            model_error = f"{type(exc).__name__}: {exc}"
            log.exception("Model call failed")

        if model_error and not answer:
            turn.error = model_error
            answer = self._model_error_message(model_error, plan)
            turn.trace.append(f"Model call failed: {model_error}")

        # Best-effort: remember what the image was about for future follow-ups.
        if image_id and image_summary:
            self.memory.annotate_image(
                image_id, summary=image_summary, extracted_text=ocr_text or vlm_text
            )

        # ── 6/7. validation + uncertainty note ──────────────────────────────
        validation = self._validate(
            answer, plan, diagnostics, user_message,
            image_summary=image_summary or (stored_image_record.summary if stored_image_record else ""),
            ocr=ocr_result,
            evidence=turn.evidence,
            vlm_text=vlm_text,
            response_language=intent.response_language,
        )
        turn.validation = validation
        final_answer = ResponseValidator.apply_uncertainty_note(validation, answer)
        if validation.uncertainty_note:
            turn.trace.append(f"Validation: {validation.status.value}")
            turn.trace.append(f"  ↳ {validation.uncertainty_note}")
        else:
            turn.trace.append(f"Validation: {validation.status.value}")
        turn.answer = final_answer

        # ── 8. record the turn (with its language + intent, so a later
        #        language switch can still resolve context) ─────────────────
        self.memory.add_turn(
            user_message,
            turn.answer,
            image_id=image_id or plan.image_id,
            route=turn.route,
            validation_status=validation.status.value,
            metadata={"evidence": [e.citation() for e in turn.evidence]},
            language=language.code,
            response_language=intent.response_language,
            intent=intent.intent,
            subjects=intent.entities,
        )
        return turn

    def _answer_text(
        self,
        user_message: str,
        response_language: str,
        context: str,
        reference_context: str,
        turn: AssistantTurn,
    ) -> tuple[str, Optional[str]]:
        """
        Answer a text-only (or follow-up) question in the user's language.

        Uses the language-aware `MultilingualModel`, which pins the answer
        language and preserves the conversation's meaning across switches, then
        verifies the answer really came back in the requested language and
        re-expresses it once if it did not.
        """
        result = self.multilingual.answer(
            user_message,
            response_language,
            context=context,
            reference_context=reference_context,
            conversation_languages=self.memory.language_history(),
        )
        if result.corrected:
            turn.response_language_corrected = True
            turn.trace.append(
                f"Answer language corrected to "
                f"{result.requested_language} (model first replied in "
                f"{result.response_language})"
            )
        turn.trace.append(
            f"Text model answer ({result.model}, {result.latency_s}s, "
            f"language={result.response_language})"
        )
        return result.text, result.error

    # ── helpers ─────────────────────────────────────────────────────────────

    def _validate(
        self,
        answer: str,
        plan: RoutePlan,
        diagnostics: Optional[ImageDiagnostics],
        user_message: str,
        image_summary: str,
        ocr,
        evidence: list[Evidence],
        vlm_text: str = "",
        response_language: str = "en",
    ) -> ValidationReport:
        ocr_text = ""
        ocr_untrusted = False
        if ocr is not None:
            if getattr(ocr, "succeeded", False):
                ocr_text = getattr(ocr, "text", "")
                # Low OCR confidence is itself evidence of poor legibility.
                if getattr(ocr, "confidence", 0.0) and ocr.confidence < 0.5:
                    ocr_untrusted = True
        if not plan.trust_ocr:
            ocr_untrusted = True

        context = ValidationContext(
            user_message=user_message,
            plan=plan,
            diagnostics=diagnostics,
            image_summary=image_summary,
            ocr_text=ocr_text,
            vlm_text=vlm_text,
            evidence=evidence,
            conversation_context=self.memory.context_for_prompt(),
            ocr_was_untrusted=ocr_untrusted,
            response_language=response_language,
        )
        return self.validator.validate(answer, context)

    @staticmethod
    def _model_error_message(error: str, plan: RoutePlan) -> str:
        lowered = error.lower()

        # Known upstream breakage: recent Ollama llama.cpp builds refuse to load
        # the 'mllama' architecture, so llama3.2-vision dies before inference.
        # Diagnose it explicitly rather than dumping the raw error.
        if "mllama" in lowered or "unknown model architecture" in lowered:
            return (
                f"⚠️ The model `{config.VISION_MODEL}` failed to load — the Ollama "
                "build on this machine can't load its architecture "
                "(`mllama`).\n\n"
                "This is an upstream Ollama issue, not a problem with your "
                "image. Two options:\n\n"
                "**1. Use a model Ollama can load** (recommended):\n"
                "```\nollama pull qwen2.5vl:7b\n```\n"
                "then restart with `VISION_MODEL=qwen2.5vl:7b`.\n\n"
                "**2. Downgrade Ollama** to a release that still supports "
                "`mllama`."
            )

        # Match the *specific* signals Ollama emits for a missing model. Do NOT
        # match a bare "pull" — a perfectly good answer can mention
        # "ollama pull" and would then be replaced by this error notice.
        missing_model = (
            "model not found" in lowered
            or "no such model" in lowered
            or "model does not exist" in lowered
            or ("not found" in lowered and "model" in lowered)
            or ("pull" in lowered and "manifest" in lowered)
        )
        if missing_model:
            # Name the model that is actually missing. A text-only turn uses the
            # text model, so blaming the vision model would send the user off to
            # pull something irrelevant.
            wanted = config.VISION_MODEL if plan.use_vision else config.TEXT_MODEL
            role = "vision model" if plan.use_vision else "text model"
            return (
                f"⚠️ The {role} (`{wanted}`) isn't available locally, so I can't "
                "complete that request.\n\nRun this in a terminal and try "
                "again:\n"
                f"```\nollama pull {wanted}\n```\n\n"
                f"_Or point at a model you already have:_\n"
                f"`{'VISION_MODEL' if plan.use_vision else 'TEXT_MODEL'}="
                f"<model-name> streamlit run app.py`"
            )
        connection_problem = (
            "connection" in lowered
            or "refused" in lowered
            or "connect" in lowered
            or "not running" in lowered
        )
        if connection_problem:
            return (
                "⚠️ I couldn't reach the local model server (Ollama).\n\nStart it "
                "with:\n```\nollama serve\n```\n"
                "and make sure the model is pulled."
            )
        return (
            "⚠️ I wasn't able to complete that request because the model call "
            f"failed:\n\n`{error}`\n\nI'd rather report the failure than invent an "
            "answer."
        )

    # ── session helpers for the UI ──────────────────────────────────────────

    def reset(self) -> None:
        self.memory.reset()
