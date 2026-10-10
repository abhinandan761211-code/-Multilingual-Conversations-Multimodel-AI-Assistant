"""
Tests for conversational context: storing turns, resolving follow-ups to the
right image, reusing image knowledge, and ambiguity/clarification routing.
"""
from __future__ import annotations

from conversation_memory import ConversationMemory
from reasoning_router import ReasoningRouter, Route


# ─────────────────────────────────────────────────────────────────────────────
# Basic storage
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryStorage:
    def test_adding_turns_preserves_order(self, memory):
        memory.add_turn("first", "reply 1")
        memory.add_turn("second", "reply 2")
        assert [t.user_message for t in memory.turns] == ["first", "second"]
        assert memory.last_turn().assistant_message == "reply 2"

    def test_image_registration_is_content_addressed(self, memory):
        a = memory.register_image(b"AAA", filename="a.png")
        b = memory.register_image(b"AAA", filename="a.png")
        assert a.image_id == b.image_id            # same bytes → same id
        assert len(memory.images) == 1

    def test_different_images_get_different_ids(self, memory):
        memory.register_image(b"AAA", filename="a.png")
        memory.register_image(b"BBB", filename="b.png")
        assert len(memory.images) == 2

    def test_registering_image_makes_it_active(self, memory):
        record = memory.register_image(b"AAA", filename="a.png")
        assert memory.active_image.image_id == record.image_id

    def test_pruning_keeps_last_n_turns_but_keeps_images(self, memory):
        memory.max_turns = 2
        memory.register_image(b"AAA", filename="a.png")
        for i in range(5):
            memory.add_turn(f"msg {i}", f"reply {i}")
        assert len(memory.turns) == 2
        assert memory.turns[-1].user_message == "msg 4"
        assert len(memory.images) == 1, "Images must survive turn pruning"

    def test_annotate_image_stores_summary_and_text(self, memory):
        record = memory.register_image(b"AAA", filename="chart.png")
        memory.annotate_image(
            record.image_id, summary="A line chart of loss", extracted_text="Epoch 10"
        )
        stored = memory.get_image(record.image_id)
        assert "line chart" in stored.summary
        assert stored.extracted_text == "Epoch 10"


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up detection & resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestFollowUpResolution:
    def test_pronoun_is_detected_as_follow_up(self, memory):
        memory.register_image(b"AAA", filename="chart.png")
        memory.add_turn("describe this image", "It is a chart.", image_id=memory.active_image.image_id)
        assert memory.is_follow_up("why does it increase?")

    def test_referential_cue_words_detected(self, memory):
        memory.register_image(b"AAA", filename="chart.png")
        for phrase in ["explain that part", "what about this", "the previous answer"]:
            assert memory.is_follow_up(phrase), phrase

    def test_part_reference_detected_when_image_exists(self, memory):
        memory.register_image(b"AAA", filename="nn.png")
        memory.add_turn("what is this", "a neural network diagram")
        assert memory.is_follow_up("what does the middle layer do?")

    def test_standalone_question_is_not_a_follow_up(self, memory):
        memory.add_turn("hello", "hi")
        assert not memory.is_follow_up("What is gradient descent?")

    def test_resolve_returns_active_image_for_follow_up(self, memory):
        record = memory.register_image(b"AAA", filename="chart.png")
        memory.add_turn("describe this image", "a chart", image_id=record.image_id)
        resolved = memory.resolve_image_for("why does it increase?")
        assert resolved is not None
        assert resolved.image_id == record.image_id

    def test_resolve_by_filename(self, memory):
        record = memory.register_image(b"AAA", filename="loss_curve.png")
        memory.register_image(b"BBB", filename="other.png")
        resolved = memory.resolve_image_for("compare with loss_curve.png")
        assert resolved.image_id == record.image_id

    def test_context_for_prompt_includes_turns_and_image_note(self, memory):
        record = memory.register_image(b"AAA", filename="chart.png")
        memory.annotate_image(record.image_id, summary="A loss curve")
        memory.add_turn("describe this", "It's a loss curve", image_id=record.image_id)
        context = memory.context_for_prompt()
        assert "describe this" in context
        assert "chart.png" in context
        assert "Assistant" in context

    def test_exclude_images_drops_image_turns_from_context(self, memory):
        """
        Regression: a self-contained question must not be answered with the
        earlier image description in context, or the model replies
        "the image does not provide information about X" instead of answering.
        """
        record = memory.register_image(b"AAA", filename="loss_curve.png")
        memory.annotate_image(record.image_id, summary="a loss curve chart")
        memory.add_turn("analyse this chart", "a loss curve", image_id=record.image_id)
        memory.add_turn("thanks", "you're welcome")

        full = memory.context_for_prompt()
        assert "loss_curve.png" in full
        assert "analyse this chart" in full

        filtered = memory.context_for_prompt(exclude_images=True)
        assert "loss_curve.png" not in filtered
        assert "loss curve" not in filtered.lower()
        # Non-image turns survive, so the conversation still has continuity.
        assert "thanks" in filtered

    def test_exclude_images_with_only_image_turns_returns_empty(self, memory):
        record = memory.register_image(b"AAA", filename="chart.png")
        memory.add_turn("describe this", "a chart", image_id=record.image_id)
        assert memory.context_for_prompt(exclude_images=True) == ""

    def test_image_context_reuses_previous_knowledge(self, memory):
        record = memory.register_image(b"AAA", filename="chart.png")
        memory.annotate_image(
            record.image_id,
            summary="Loss curve with training and validation series",
            extracted_text="Epoch 10",
        )
        context = memory.image_context(record)
        assert "Loss curve" in context
        assert "Epoch 10" in context


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation (Streamlit session_state round-trip)
# ─────────────────────────────────────────────────────────────────────────────


class TestSerialisation:
    def test_round_trip_preserves_state(self, memory):
        record = memory.register_image(b"AAA", filename="a.png", width=10, height=10)
        memory.annotate_image(record.image_id, summary="something")
        memory.add_turn("hello", "hi", image_id=record.image_id)

        state = memory.to_state()
        restored = ConversationMemory.from_state(state)

        assert len(restored.turns) == 1
        assert len(restored.images) == 1
        assert restored.active_image.image_id == record.image_id
        assert restored.active_image.summary == "something"

    def test_reset_clears_everything(self, memory):
        memory.register_image(b"AAA", filename="a.png")
        memory.add_turn("hello", "hi")
        memory.reset()
        assert not memory.turns
        assert not memory.images
        assert memory.active_image is None


# ─────────────────────────────────────────────────────────────────────────────
# Ambiguity handling in the router
# ─────────────────────────────────────────────────────────────────────────────


class TestAmbiguityRouting:
    def test_vague_prompt_with_image_asks_for_clarification(self, router, good_diagnostics):
        plan = router.plan(
            "check this image",
            diagnostics=good_diagnostics,
            new_image_id="img1",
        )
        assert plan.route == Route.CLARIFY
        assert plan.clarifying_question
        assert "describe" in plan.clarifying_question.lower()

    def test_single_word_prompt_is_vague(self, router, good_diagnostics):
        plan = router.plan("analyse", diagnostics=good_diagnostics, new_image_id="i")
        assert plan.route == Route.CLARIFY

    def test_specific_prompt_is_not_clarified(self, router, good_diagnostics):
        plan = router.plan(
            "how many people are in this image?",
            diagnostics=good_diagnostics,
            new_image_id="i",
        )
        assert plan.route != Route.CLARIFY

    def test_clarification_question_mentions_blur_when_blurry(self, router, blurry_diagnostics):
        plan = router.plan("check this", diagnostics=blurry_diagnostics, new_image_id="i")
        assert plan.route == Route.CLARIFY
        assert "blur" in plan.clarifying_question.lower()

    def test_greeting_without_image_is_small_talk(self, router):
        plan = router.plan("hello", diagnostics=None)
        assert plan.route == Route.SMALL_TALK

    def test_router_plan_records_rationale(self, router, good_diagnostics):
        plan = router.plan(
            "explain the trend in this graph",
            diagnostics=good_diagnostics,
            new_image_id="i",
        )
        assert plan.rationale, "Every route must explain itself"
        assert plan.confidence > 0

    def test_follow_up_route_when_referring_to_stored_image(self, router, memory):
        record = memory.register_image(b"AAA", filename="chart.png")
        memory.annotate_image(record.image_id, summary="a loss curve")
        memory.add_turn("describe this", "a loss curve", image_id=record.image_id)

        plan = router.plan(
            "why does it increase?",
            diagnostics=None,
            memory=memory,
        )
        assert plan.route == Route.FOLLOW_UP
        assert plan.image_id == record.image_id

    def test_text_only_question_enables_retrieval(self, router, memory):
        plan = router.plan("what is overfitting?", diagnostics=None, memory=memory)
        assert plan.route == Route.GENERAL_QUESTION
        assert plan.use_retrieval

    # ── Regression: an image in context must not capture every later question ──

    def test_self_contained_question_ignores_pinned_image(self, router, memory):
        """
        Regression: with an image pinned, every question used to be routed
        through the vision model — so "What is the capital of France?" was
        answered *about the chart*. A question that stands on its own must be
        answered as a text question.
        """
        record = memory.register_image(b"AAA", filename="loss_curve.png")
        memory.annotate_image(
            record.image_id, summary="a loss curve chart with training and validation lines"
        )
        memory.add_turn("analyse this", "a loss curve", image_id=record.image_id)

        for question in (
            "What is the capital of France?",
            "Write a Python function to reverse a string",
            "What is 17 times 23?",
            "Who wrote the novel Moby Dick?",
        ):
            plan = router.plan(question, diagnostics=None, memory=memory)
            assert plan.route == Route.GENERAL_QUESTION, question
            assert not plan.use_vision, f"should not use vision: {question}"

    def test_question_about_the_image_still_uses_vision(self, router, memory):
        """The relevance guard must not swing too far and lose real follow-ups."""
        record = memory.register_image(b"AAA", filename="loss_curve.png")
        memory.annotate_image(
            record.image_id, summary="a loss curve chart with training and validation lines"
        )
        memory.add_turn("analyse this", "a loss curve", image_id=record.image_id)

        for question in (
            "Why does the validation loss increase in this graph?",
            "What colour is the validation line?",
            "Explain the trend shown in the chart",
            "What does the middle layer do?",
            "in the graph?",
            "and why?",
        ):
            plan = router.plan(question, diagnostics=None, memory=memory)
            assert plan.use_vision, f"should use the image: {question}"
            assert plan.route == Route.FOLLOW_UP, question

    def test_follow_up_works_even_without_an_image_summary(self, router, memory):
        """An image with no stored summary still counts as visual context."""
        record = memory.register_image(b"AAA", filename="chart.png")
        memory.add_turn("what is this", "a chart", image_id=record.image_id)
        plan = router.plan("what about the blue line?", diagnostics=None, memory=memory)
        assert plan.route == Route.FOLLOW_UP

    def test_new_image_plus_question_always_uses_that_image(self, router, good_diagnostics):
        """A freshly attached image wins regardless of vocabulary."""
        plan = router.plan(
            "What is the capital of France?",
            diagnostics=good_diagnostics,
            new_image_id="fresh",
        )
        assert plan.use_vision
        assert plan.image_is_new

    def test_open_ended_explanations_enable_retrieval_on_follow_up(self, router, memory):
        # "what does X do" is an explanation request, so the follow-up should
        # pull in supporting documents rather than answering from memory alone.
        record = memory.register_image(b"AAA", filename="nn.png")
        memory.annotate_image(record.image_id, summary="a neural network diagram")
        memory.add_turn("what is this", "a diagram", image_id=record.image_id)

        for question in (
            "What does the middle layer do?",
            "How does the hidden layer work?",
            "Explain the output layer",
        ):
            plan = router.plan(question, diagnostics=None, memory=memory)
            assert plan.route == Route.FOLLOW_UP, question
            assert plan.use_retrieval, f"retrieval not enabled for: {question}"

    def test_short_literal_follow_up_skips_retrieval(self, router, memory):
        # A purely visual follow-up needs no external documents.
        record = memory.register_image(b"AAA", filename="chart.png")
        memory.annotate_image(record.image_id, summary="a loss curve")
        memory.add_turn("describe this", "a loss curve", image_id=record.image_id)
        plan = router.plan("what about the blue one?", diagnostics=None, memory=memory)
        assert plan.route == Route.FOLLOW_UP
