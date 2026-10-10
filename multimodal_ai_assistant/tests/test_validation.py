"""
Tests for response validation (V1–V6) and the end-to-end pipeline behaviour
that depends on it: grounding, numeric fidelity, fabrication, uncertainty, and
evidence retrieval.
"""
from __future__ import annotations

import pytest

from conversation_memory import ConversationMemory
from evidence_retriever import Evidence
from pipeline import AssistantPipeline
from reasoning_router import ReasoningRouter, Route, RoutePlan
from response_validator import (
    ResponseValidator,
    ValidationContext,
    ValidationStatus,
)


def _ctx(**kwargs) -> ValidationContext:
    kwargs.setdefault("user_message", "how many people are in this image?")
    kwargs.setdefault("plan", RoutePlan(route=Route.ANSWER_WITH_IMAGE, use_vision=True))
    return ValidationContext(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# V0 / V5 — format and safety
# ─────────────────────────────────────────────────────────────────────────────


class TestFormat:
    def test_empty_answer_fails(self, validator):
        report = validator.validate("", _ctx())
        assert report.status == ValidationStatus.FAILED
        assert report.confidence == 0.0

    def test_too_short_answer_flagged(self, validator):
        report = validator.validate("Yes.", _ctx())
        assert any(c.name == "V5-format" and not c.passed for c in report.checks)

    def test_normal_answer_passes_format(self, validator):
        report = validator.validate(
            "I can see four people standing near a doorway.", _ctx()
        )
        fmt = next(c for c in report.checks if c.name == "V5-format")
        assert fmt.passed

    def test_capability_refusal_flagged(self, validator):
        report = validator.validate(
            "As an AI language model, I don't have access to the images you "
            "upload, so I am not able to help with that.",
            _ctx(),
        )
        safety = next(c for c in report.checks if c.name == "V5-safety")
        assert not safety.passed
        assert any("refusal" in w.lower() for w in report.warnings)

    def test_scoped_hedge_is_not_a_refusal(self, validator, good_diagnostics):
        """'I can't confirm X, but I can see Y' is good hedging, not a refusal."""
        report = validator.validate(
            "I can't confirm the exact count from this angle, but I can see "
            "three people clearly and one partially obscured figure.",
            _ctx(
                user_message="how many people?",
                diagnostics=good_diagnostics,
                image_summary="four figures, one partly behind a dark box",
            ),
        )
        safety = next(c for c in report.checks if c.name == "V5-safety")
        assert safety.passed, safety.detail

    def test_answering_after_uncertainty_is_not_refusal(self, validator):
        """A model that hedges then answers must not be marked as refusing."""
        report = validator.validate(
            "I cannot read the smallest axis label, but the validation loss "
            "clearly rises after epoch 6.",
            _ctx(
                user_message="what happens after epoch 6?",
                image_summary="validation loss rises after epoch 6",
            ),
        )
        safety = next(c for c in report.checks if c.name == "V5-safety")
        assert safety.passed, safety.detail


# ─────────────────────────────────────────────────────────────────────────────
# V2 — grounding
# ─────────────────────────────────────────────────────────────────────────────


class TestGrounding:
    def test_grounded_answer_passes(self, validator, ok_ocr):
        report = validator.validate(
            "The training loss is 0.42 and the validation loss is 0.71.",
            _ctx(ocr_text=ok_ocr.text, image_summary=ok_ocr.text),
        )
        grounding = next(c for c in report.checks if c.name == "V2-grounding")
        assert grounding.passed

    def test_ungrounded_answer_is_flagged(self, validator):
        report = validator.validate(
            "The chart shows a sinusoidal oscillation with amplitude 9.7.",
            _ctx(ocr_text="Epoch 10 loss curve", image_summary="a simple line chart"),
        )
        grounding = next(c for c in report.checks if c.name == "V2-grounding")
        assert not grounding.passed

    def test_no_evidence_at_all_is_flagged(self, validator):
        report = validator.validate(
            "There are exactly seven objects in the picture.", _ctx()
        )
        grounding = next(c for c in report.checks if c.name == "V2-grounding")
        assert not grounding.passed
        assert any("without any supporting evidence" in w for w in report.warnings)

    def test_small_talk_needs_no_evidence(self, validator):
        report = validator.validate(
            "Hello! How can I help you today?",
            _ctx(plan=RoutePlan(route=Route.SMALL_TALK), user_message="hi"),
        )
        grounding = next(c for c in report.checks if c.name == "V2-grounding")
        assert grounding.passed

    def test_long_answer_to_short_question_is_relevant(self, validator):
        """A long structured answer must not fail relevance just for being long."""
        answer = (
            "Chart Type: Line graph. Title: Model Loss over Training. "
            "X-axis: Epoch. Y-axis: Loss. The training loss decreases steadily. "
            "The validation loss starts increasing after epoch 6 because the "
            "model begins to overfit the training noise."
        )
        report = validator.validate(
            answer,
            _ctx(user_message="when does validation loss start increasing and why?"),
        )
        relevance = next(c for c in report.checks if c.name == "V1-relevance")
        assert relevance.passed, relevance.detail

    def test_off_topic_answer_still_fails_relevance(self, validator):
        report = validator.validate(
            "Bananas are yellow and rich in potassium.",
            _ctx(user_message="when does validation loss start increasing and why?"),
        )
        relevance = next(c for c in report.checks if c.name == "V1-relevance")
        assert not relevance.passed

    def test_clarification_needs_no_grounding(self, validator):
        # A clarifying question asserts nothing, so it must not be flagged for
        # lacking evidence — that would be a false positive on every clarify turn.
        report = validator.validate(
            "Would you like me to describe the image or read its text?",
            _ctx(user_message="check this", plan=RoutePlan(route=Route.CLARIFY)),
        )
        grounding = next(c for c in report.checks if c.name == "V2-grounding")
        assert grounding.passed
        assert not any("without any supporting evidence" in w for w in report.warnings)

    def test_retrieved_evidence_counts_as_grounds(self, validator):
        evidence = [
            Evidence(
                text="Overfitting occurs when a model learns training noise, so "
                     "validation loss rises while training loss falls.",
                source="ml_basics.md",
                score=0.8,
            )
        ]
        report = validator.validate(
            "Overfitting is when the model fits training noise and validation "
            "loss starts rising even though training loss keeps falling.",
            _ctx(
                user_message="what is overfitting?",
                evidence=evidence,
                plan=RoutePlan(route=Route.GENERAL_QUESTION, use_retrieval=True),
            ),
        )
        grounding = next(c for c in report.checks if c.name == "V2-grounding")
        assert grounding.passed


# ─────────────────────────────────────────────────────────────────────────────
# V3 — numeric fidelity
# ─────────────────────────────────────────────────────────────────────────────


class TestNumericFidelity:
    def test_numbers_present_in_evidence_pass(self, validator):
        report = validator.validate(
            "Validation loss rises to 0.71 by epoch 10.",
            _ctx(
                user_message="what happens to validation loss?",
                image_summary="Epoch 10, validation loss 0.71",
            ),
        )
        nums = next(c for c in report.checks if c.name == "V3-numbers")
        assert nums.passed

    def test_fabricated_numbers_flagged(self, validator):
        report = validator.validate(
            "The accuracy reaches 0.99 at epoch 47.",
            _ctx(
                user_message="what accuracy does it reach?",
                image_summary="Epoch 10, validation loss 0.71",
            ),
        )
        nums = next(c for c in report.checks if c.name == "V3-numbers")
        assert not nums.passed
        assert any("0.99" in w or "47" in w for w in report.warnings)

    def test_no_numbers_is_fine(self, validator):
        report = validator.validate(
            "The curve clearly trends upward towards the end.", _ctx(image_summary="a curve")
        )
        nums = next(c for c in report.checks if c.name == "V3-numbers")
        assert nums.passed


# ─────────────────────────────────────────────────────────────────────────────
# V4 — uncertainty on unclear images
# ─────────────────────────────────────────────────────────────────────────────


class TestUncertainty:
    def test_confident_answer_on_blurry_image_is_flagged(self, validator, blurry_diagnostics):
        report = validator.validate(
            "There are exactly four people in the photo.",
            _ctx(
                user_message="how many people?",
                diagnostics=blurry_diagnostics,
            ),
        )
        unc = next(c for c in report.checks if c.name == "V4-uncertainty")
        assert not unc.passed
        assert any("uncertainty" in w.lower() for w in report.warnings)

    def test_hedged_answer_on_blurry_image_passes(self, validator, blurry_diagnostics):
        report = validator.validate(
            "I can clearly identify three people, but part of the image is "
            "obscured and blurred, so the total count is uncertain.",
            _ctx(user_message="how many people?", diagnostics=blurry_diagnostics),
        )
        unc = next(c for c in report.checks if c.name == "V4-uncertainty")
        assert unc.passed

    def test_morphological_hedges_are_detected(self, validator, blurry_diagnostics):
        """"blurriness"/"indistinct" must count as hedging, not just "blurry"."""
        report = validator.validate(
            "The figures are indistinct because of the blurriness, so I cannot "
            "definitively state the count.",
            _ctx(user_message="how many people?", diagnostics=blurry_diagnostics),
        )
        unc = next(c for c in report.checks if c.name == "V4-uncertainty")
        assert unc.passed, unc.detail

    def test_untrusted_ocr_triggers_uncertainty_check(self, validator, good_diagnostics):
        report = validator.validate(
            "The label reads 'Validation Loss'.",
            _ctx(
                user_message="what does the text say?",
                diagnostics=good_diagnostics,
                ocr_was_untrusted=True,
            ),
        )
        unc = next(c for c in report.checks if c.name == "V4-uncertainty")
        assert not unc.passed, "Low-confidence OCR must demand hedging"

    def test_clean_image_needs_no_hedging(self, validator, good_diagnostics):
        report = validator.validate(
            "The chart is a line plot of loss against epoch.",
            _ctx(
                user_message="what kind of chart is this?",
                diagnostics=good_diagnostics,
                image_summary="line plot of loss against epoch",
            ),
        )
        unc = next(c for c in report.checks if c.name == "V4-uncertainty")
        assert unc.passed


# ─────────────────────────────────────────────────────────────────────────────
# V6 — fabrication of entities
# ─────────────────────────────────────────────────────────────────────────────


class TestFabrication:
    def test_invented_entity_flagged(self, validator):
        report = validator.validate(
            "The chart is from the ResNet paper by Kaiming He.",
            _ctx(
                user_message="what is this chart?",
                image_summary="A line chart of loss over epochs.",
            ),
        )
        fab = next(c for c in report.checks if c.name == "V6-fabrication")
        assert not fab.passed

    def test_entity_present_in_evidence_passes(self, validator):
        report = validator.validate(
            "This shows the training loss curve.",
            _ctx(
                user_message="what is this?",
                image_summary="The training loss curve over epochs.",
            ),
        )
        fab = next(c for c in report.checks if c.name == "V6-fabrication")
        assert fab.passed


# ─────────────────────────────────────────────────────────────────────────────
# Status decisions & rewriting
# ─────────────────────────────────────────────────────────────────────────────


class TestStatusDecision:
    def test_clean_grounded_answer_passes(self, validator):
        report = validator.validate(
            "The image is a line chart showing loss against epoch.",
            _ctx(
                user_message="what does the chart show?",
                diagnostics=None,
                image_summary="line chart showing loss against epoch",
            ),
        )
        assert report.status in (ValidationStatus.PASS, ValidationStatus.QUALIFIED)

    def test_ungrounded_answer_is_qualified_not_passed(self, validator):
        report = validator.validate(
            "The graph shows an exponential decay with a half-life of 3.5.",
            _ctx(
                user_message="describe the graph",
                image_summary="a simple bar chart of counts",
            ),
        )
        assert report.status == ValidationStatus.QUALIFIED
        assert report.uncertainty_note

    def test_clarify_route_gets_needs_clarification_status(self, validator, good_diagnostics):
        report = validator.validate(
            "Would you like me to describe the image or extract its text?",
            _ctx(
                user_message="check this",
                plan=RoutePlan(route=Route.CLARIFY, image_id="i"),
                diagnostics=good_diagnostics,
            ),
        )
        assert report.status == ValidationStatus.NEEDS_CLARIFICATION

    def test_uncertainty_note_is_prepended(self, validator):
        report = validator.validate(
            "The graph shows an exponential decay with a half-life of 3.5.",
            _ctx(user_message="describe the graph", image_summary="a bar chart"),
        )
        rewritten = ResponseValidator.apply_uncertainty_note(
            report, report.final_answer
        )
        assert rewritten.startswith("_") or "Validation caveat" in rewritten

    def test_badge_reflects_status(self, validator):
        report = validator.validate("", _ctx())
        assert "failed" in report.as_badge()


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline (fake model, no Ollama)
# ─────────────────────────────────────────────────────────────────────────────


class _NoRetrieval:
    """Retriever stub so pipeline tests never touch the filesystem or downloads."""

    backend = "keyword"

    def build_index(self, force: bool = False):
        return "keyword"

    def retrieve(self, *a, **k):
        return []

    def format_for_prompt(self, evidence):
        return ""

    def stats(self):
        return {"backend": "keyword", "chunks": 0, "documents": []}


class TestPipelineEndToEnd:
    def _pipeline(self, fake_vision, tmp_path):
        return AssistantPipeline(
            vision_model=fake_vision,
            retriever=_NoRetrieval(),
            memory=ConversationMemory(),
            router=ReasoningRouter(classifier=None),
            validator=ResponseValidator(embedder=None, entailment_checker=None),
        )

    def test_vague_image_prompt_asks_for_clarification(
        self, fake_vision, sharp_text_image, tmp_path
    ):
        pipeline = self._pipeline(fake_vision, tmp_path)
        turn = pipeline.handle_turn("check this", sharp_text_image, "chart.png")
        assert turn.route == Route.CLARIFY.value
        assert "describe" in turn.answer.lower()
        assert "describe_image" not in fake_vision.calls, (
            "The model must not be called when clarifying"
        )

    def test_chart_question_uses_vision_and_validates(
        self, fake_vision, sharp_text_image, tmp_path
    ):
        pipeline = self._pipeline(fake_vision, tmp_path)
        turn = pipeline.handle_turn(
            "when does validation loss start increasing in this graph?",
            sharp_text_image,
            "chart.png",
        )
        assert turn.route in (Route.ANALYSE_CHART.value, Route.ANSWER_WITH_IMAGE.value)
        assert fake_vision.calls, "Vision model should have been invoked"
        assert turn.validation is not None
        assert turn.answer

    def test_follow_up_reuses_memory_without_reupload(
        self, fake_vision, sharp_text_image, tmp_path
    ):
        pipeline = self._pipeline(fake_vision, tmp_path)
        pipeline.handle_turn(
            "explain the validation loss trend in this graph",
            sharp_text_image,
            "chart.png",
        )
        turn2 = pipeline.handle_turn("why does it increase?", None, "")
        # Second turn must be a follow-up, not a fresh image request.
        assert turn2.route == Route.FOLLOW_UP.value
        assert "answer_without_image" in fake_vision.calls

    def test_broken_image_never_reaches_the_model(self, fake_vision, corrupt_bytes, tmp_path):
        pipeline = self._pipeline(fake_vision, tmp_path)
        turn = pipeline.handle_turn("what is in this image?", corrupt_bytes, "bad.png")
        assert turn.route == Route.CLARIFY.value
        assert not fake_vision.calls

    def test_trace_is_recorded(self, fake_vision, sharp_text_image, tmp_path):
        pipeline = self._pipeline(fake_vision, tmp_path)
        turn = pipeline.handle_turn(
            "read the text in this screenshot", sharp_text_image, "shot.png"
        )
        assert turn.trace
        assert any("Router chose" in step for step in turn.trace)

    def test_model_failure_reports_instead_of_fabricating(
        self, sharp_text_image, tmp_path
    ):
        """When the model errors, the answer must be an honest failure notice."""
        class BrokenVision:
            text_model = vision_model = "broken"

            def classify_request(self, *a, **k):
                return {"fallback": True}

            def analyse_chart(self, *a, **k):
                from vision_model import ModelResponse

                return ModelResponse(
                    model="broken",
                    success=False,
                    error="ResponseError: unknown model architecture: 'mllama'",
                )

            def answer_with_image(self, *a, **k):
                return self.analyse_chart()

            def answer_without_image(self, *a, **k):
                return self.analyse_chart()

            def describe_image(self, *a, **k):
                return self.analyse_chart()

            def extract_text_from_image(self, *a, **k):
                return self.analyse_chart()

            def check_entailment(self, *a, **k):
                return {"supported": None, "confidence": 0.0, "reason": ""}

        pipeline = AssistantPipeline(
            vision_model=BrokenVision(),
            retriever=_NoRetrieval(),
            memory=ConversationMemory(),
            router=ReasoningRouter(classifier=None),
            validator=ResponseValidator(embedder=None),
        )
        turn = pipeline.handle_turn(
            "describe this graph in detail", sharp_text_image, "chart.png"
        )
        # The failure must be surfaced, with actionable guidance — not an
        # invented description of an image the model never saw.
        assert turn.error
        assert "mllama" in turn.answer.lower() or "failed to load" in turn.answer.lower()
        assert "ollama pull" in turn.answer.lower()

    def test_error_message_does_not_hijack_answers_mentioning_pull(self):
        """Regression: an answer containing 'ollama pull' must not be replaced.

        The missing-model branch previously matched a bare "pull", so any error
        text quoting 'ollama pull ...' was misreported as a missing model —
        including errors from a completely unrelated, working call.
        """
        from reasoning_router import Route, RoutePlan

        benign = "ValueError: could not parse response; see `ollama pull x`"
        msg = AssistantPipeline._model_error_message(
            benign, RoutePlan(route=Route.GENERAL_QUESTION)
        )
        assert "isn't available locally" not in msg

        # The genuine signals must still be recognised.
        for real in (
            "model not found",
            "error: model does not exist",
            "pull model manifest: not found",
        ):
            out = AssistantPipeline._model_error_message(
                real, RoutePlan(route=Route.GENERAL_QUESTION)
            )
            assert "ollama pull" in out, real

    def test_missing_model_message_names_the_right_model(self):
        """A text-only failure must not tell the user to pull the vision model."""
        import config
        from reasoning_router import Route, RoutePlan

        text_msg = AssistantPipeline._model_error_message(
            "model not found",
            RoutePlan(route=Route.GENERAL_QUESTION, use_vision=False),
        )
        assert config.TEXT_MODEL in text_msg
        assert config.VISION_MODEL not in text_msg
        assert "text model" in text_msg.lower()

        vision_msg = AssistantPipeline._model_error_message(
            "model not found",
            RoutePlan(route=Route.ANSWER_WITH_IMAGE, use_vision=True),
        )
        assert config.VISION_MODEL in vision_msg
        assert "vision model" in vision_msg.lower()

    def test_text_model_auto_fallback(self):
        """If the configured text model is absent, any local text model beats failing."""
        from vision_model import VisionModel

        chosen = VisionModel._pick_text_fallback(
            ["qwen2.5vl:7b", "llama3:latest", "nomic-embed-text:latest"]
        )
        # Must not pick the vision model for text-only work when a text model exists.
        assert chosen == "llama3:latest"

        # With only a vision model present, use it rather than nothing.
        assert VisionModel._pick_text_fallback(["qwen2.5vl:7b"]) == "qwen2.5vl:7b"
        assert VisionModel._pick_text_fallback([]) is None

    def test_memory_records_turns(self, fake_vision, sharp_text_image, tmp_path):
        pipeline = self._pipeline(fake_vision, tmp_path)
        pipeline.handle_turn(
            "describe this image please", sharp_text_image, "chart.png"
        )
        assert len(pipeline.memory.turns) == 1
        assert pipeline.memory.turns[0].route


# ─────────────────────────────────────────────────────────────────────────────
# Evidence retriever (keyword backend, no downloads)
# ─────────────────────────────────────────────────────────────────────────────


class TestEvidenceRetriever:
    def test_retrieves_relevant_document(self, tmp_path):
        from evidence_retriever import EvidenceRetriever

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text(
            "Overfitting occurs when validation loss rises while training loss "
            "continues to fall. Early stopping is a remedy."
        )
        (docs / "b.md").write_text(
            "Bananas are a yellow fruit rich in potassium and fibre."
        )

        retriever = EvidenceRetriever(
            documents_dir=docs, persist_dir=tmp_path / "chroma",
            embedding_model="__nonexistent_model__",
        )
        backend = retriever.build_index()
        assert backend == "keyword"

        hits = retriever.retrieve("what is overfitting", min_score=0.01)
        assert hits
        assert hits[0].source == "a.md"
        assert "validation loss" in hits[0].text.lower()

    def test_irrelevant_query_returns_nothing(self, tmp_path):
        from evidence_retriever import EvidenceRetriever

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("Overfitting is a modelling problem.")
        retriever = EvidenceRetriever(
            documents_dir=docs, persist_dir=tmp_path / "chroma",
            embedding_model="__nonexistent_model__",
        )
        retriever.build_index()
        assert retriever.retrieve("zzzz qqqq xxxx", min_score=0.01) == []

    def test_format_for_prompt_cites_sources(self, tmp_path):
        from evidence_retriever import EvidenceRetriever

        retriever = EvidenceRetriever(documents_dir=tmp_path, persist_dir=tmp_path / "c")
        formatted = retriever.format_for_prompt(
            [Evidence(text="Some fact.", source="doc.md", score=0.9)]
        )
        assert "doc.md" in formatted
        assert "Some fact." in formatted
