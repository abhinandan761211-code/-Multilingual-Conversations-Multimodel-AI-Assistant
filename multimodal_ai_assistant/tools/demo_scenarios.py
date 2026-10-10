"""
Run the six evaluation scenarios end-to-end against the *real* pipeline,
printing what the router decided and how the validator scored each answer.

This is the scripted version of the manual demo — useful for a quick sanity
check before presenting, and as evidence for the report.

    python tools/demo_scenarios.py                 # all scenarios
    python tools/demo_scenarios.py 1 4             # only scenarios 1 and 4

Requires Ollama running with the vision model pulled.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conversation_memory import ConversationMemory  # noqa: E402
from pipeline import AssistantPipeline  # noqa: E402
from reasoning_router import ReasoningRouter  # noqa: E402
from response_validator import ResponseValidator  # noqa: E402
from vision_model import VisionModel  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_images"

BAR = "─" * 78


def banner(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def show(turn, answer_chars: int = 1200) -> None:
    print(f"\nROUTE      : {turn.route}")
    print(f"PLAN       : {turn.plan.explain() if turn.plan else 'n/a'}")
    print(f"VALIDATION : {turn.validation.as_badge() if turn.validation else 'n/a'} "
          f"(confidence {turn.validation.confidence:.2f})"
          if turn.validation else "VALIDATION : n/a")
    if turn.validation and turn.validation.warnings:
        for w in turn.validation.warnings:
            print(f"  ! {w}")
    print(f"\nANSWER:\n{turn.answer[:answer_chars]}")
    print(f"\nTRACE:")
    for step in turn.trace:
        print(f"  {step}")


def load(name: str) -> bytes:
    path = DEMO_DIR / name
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Generate demo images first:\n"
            f"  python tools/make_demo_images.py"
        )
    return path.read_bytes()


def make_pipeline() -> AssistantPipeline:
    vision = VisionModel()
    return AssistantPipeline(
        vision_model=vision,
        memory=ConversationMemory(),
        router=ReasoningRouter(classifier=vision.classify_request),
        validator=ResponseValidator(entailment_checker=vision.check_entailment),
    )


# ── the six scenarios ───────────────────────────────────────────────────────


def scenario_1(pipeline: AssistantPipeline) -> None:
    """Image + clear question → grounded visual explanation."""
    banner("SCENARIO 1 — Image + clear question")
    print('Input: loss_curve.png + "When does the validation loss start '
          'increasing, and why?"')
    turn = pipeline.handle_turn(
        "When does the validation loss start increasing in this graph, and why?",
        load("loss_curve.png"),
        "loss_curve.png",
    )
    show(turn)


def scenario_2(pipeline: AssistantPipeline) -> None:
    """Image without a clear question → clarification, no model call."""
    banner("SCENARIO 2 — Image without a clear question (ambiguity)")
    print('Input: handwritten_math.png + "check this"')
    turn = pipeline.handle_turn("check this", load("handwritten_math.png"),
                                "handwritten_math.png")
    show(turn)
    print("\n>>> Note: no vision call was made — the system chose to ask first.")


def scenario_3(pipeline: AssistantPipeline) -> None:
    """Follow-up on the same image → resolved from memory."""
    banner("SCENARIO 3 — Follow-up on the same image (memory)")
    print('Turn 1: neural_net.png + "What kind of diagram is this?"')
    pipeline.handle_turn(
        "What kind of diagram is this?", load("neural_net.png"), "neural_net.png"
    )
    print('Turn 2 (no image re-uploaded): "What does the middle layer do?"')
    turn = pipeline.handle_turn("What does the middle layer do?", None, "")
    show(turn)


def scenario_4(pipeline: AssistantPipeline) -> None:
    """Blurry image → uncertainty reporting."""
    banner("SCENARIO 4 — Blurry / partly obscured image (uncertainty)")
    print('Input: blurred_scene.png + "How many people are in this image?"')
    turn = pipeline.handle_turn(
        "How many people are in this image?", load("blurred_scene.png"),
        "blurred_scene.png",
    )
    show(turn)
    if turn.diagnostics:
        print(f"\n>>> Image diagnostics: blur={turn.diagnostics.blur_score:.1f} "
              f"blurry={turn.diagnostics.is_blurry} "
              f"({turn.diagnostics.width}x{turn.diagnostics.height})")


def scenario_5(pipeline: AssistantPipeline) -> None:
    """Question needing external facts → retrieval."""
    banner("SCENARIO 5 — Question requiring external facts (retrieval)")
    print('Input (text only): "What is overfitting and how do you prevent it?"')
    turn = pipeline.handle_turn(
        "What is overfitting and how do you prevent it?", None, ""
    )
    show(turn)
    print(f"\n>>> Retrieved {len(turn.evidence)} evidence chunk(s):")
    for e in turn.evidence:
        print(f"    - {e.citation()} (score {e.score})")


def scenario_6(pipeline: AssistantPipeline) -> None:
    """Unsupported claim → validator qualifies the answer."""
    banner("SCENARIO 6 — Question the evidence cannot support (validation)")
    print('Input: neural_net.png + "What is the exact accuracy value shown?"')
    turn = pipeline.handle_turn(
        "What is the exact accuracy value shown in this diagram?",
        load("neural_net.png"),
        "neural_net.png",
    )
    show(turn)
    print("\n>>> The validator checks numbers against the visible evidence and "
          "downgrades the answer if a value cannot be traced.")


SCENARIOS = {
    1: scenario_1,
    2: scenario_2,
    3: scenario_3,
    4: scenario_4,
    5: scenario_5,
    6: scenario_6,
}


def main() -> None:
    vision = VisionModel()
    health = vision.health_check()
    print(f"Model health: {health.summary()}")

    wanted = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(SCENARIOS)

    if not health.vision_model_available:
        print(f"\n⚠️  '{vision.vision_model}' is not available. Image scenarios "
              f"will produce an instruction message rather than an analysis.")
        print(f"   Run: ollama pull {vision.vision_model}\n")

    for number in wanted:
        scenario = SCENARIOS.get(number)
        if scenario is None:
            print(f"Unknown scenario {number}")
            continue
        # Fresh pipeline per scenario so memory doesn't bleed across them.
        scenario(make_pipeline())

    banner("DONE")


if __name__ == "__main__":
    main()
