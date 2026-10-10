# Assistant Architecture Notes

## Processing Pipeline

Every user turn passes through an explicit pipeline rather than a single model
call:

1. **Image validation and measurement** — the uploaded image is decoded,
   checked for size, format and resolution, and measured for sharpness and
   brightness. Failures become first-class outcomes, not exceptions.
2. **Intent routing** — deterministic heuristics read the user's message for
   signals (describe, read text, analyse chart, count, explain, follow-up,
   vague). The router then selects one strategy and records its rationale.
3. **Tool selection** — OCR runs only when the image plausibly contains text;
   the vision model runs whenever pixels matter; retrieval runs when the
   question needs external facts.
4. **Generation** — the selected tool(s) produce a candidate answer.
5. **Validation** — the candidate is checked for relevance, grounding, numeric
   fidelity, uncertainty reporting and fabricated entities.
6. **Presentation** — the answer ships with its validation status and, where
   warranted, an explicit uncertainty note.

## Why Not Just Send the Image to a Model?

Sending an image to a vision model and printing the reply delegates every
decision to the model: whether text should be read, whether a count is safe,
whether the question was ambiguous at all. That produces confident wrong answers
on blurry images and unsupported answers on ambiguous prompts. The router plus
validator make those decisions explicit, inspectable and testable.

## Conversation Memory

Memory stores each turn together with the image it used. Follow-up questions are
resolved by looking for referring expressions ("it", "that graph", "the middle
layer") and mapping them to the most recently used image. The image's earlier
description and any text extracted from it are reused, so a follow-up does not
require re-uploading or re-analysing the pixels.

## Grounding and Fabrication

Grounding means every factual claim traces back to the image contents, the OCR
transcript, or a retrieved document. The fabrication check flags proper nouns
and numbers in the answer that appear nowhere in the evidence. When grounding is
weak, the answer is downgraded to "qualified" and carries a caveat.
