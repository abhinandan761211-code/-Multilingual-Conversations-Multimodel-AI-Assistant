# Architecture

## Why the design is shaped this way

The assignment brief says, in effect: *a single model call is not enough — the
system must decide how to analyse the image, whether the information is
reliable, and what the user's question actually means.*

That sentence drives the whole design. The model is treated as **one tool among
several**, invoked by a decision layer, and its output is treated as a
**candidate answer** that must pass validation before it is shown.

```
                        ┌──────────────────────────┐
   text ───────────────▶│                          │
                        │      Reasoning Router    │◀── conversation memory
   image ──────────────▶│   (heuristics + opt. LLM)│◀── image diagnostics
                        │                          │
                        └────────────┬─────────────┘
                                     │  RoutePlan {route, use_ocr, use_vision,
                                     │             use_retrieval, rationale}
             ┌───────────┬───────────┼───────────┬────────────┐
             ▼           ▼           ▼           ▼            ▼
        ┌────────┐  ┌────────┐  ┌─────────┐ ┌────────┐  ┌──────────┐
        │  OCR   │  │ Vision │  │Retrieval│ │ Memory │  │ Clarify  │
        │EasyOCR │  │Llama   │  │ChromaDB │ │ image  │  │ (no model│
        │/Tesser.│  │3.2-v   │  │+ MiniLM │ │ registry│  │  call)   │
        └───┬────┘  └───┬────┘  └────┬────┘ └───┬────┘  └────┬─────┘
            └───────────┴────────────┴──────────┘            │
                                     │                       │
                                     ▼                       │
                        ┌──────────────────────────┐         │
                        │    Response Validator    │         │
                        │  V1 relevance            │         │
                        │  V2 grounding            │         │
                        │  V2b entailment (LLM)    │         │
                        │  V3 numeric fidelity     │         │
                        │  V4 uncertainty          │         │
                        │  V5 format / safety      │         │
                        │  V6 fabrication          │         │
                        │  V7 response language    │         │
                        └────────────┬─────────────┘         │
                                     ▼                       ▼
                        Answer + status + uncertainty note
```

## The decision layer, in detail

`ReasoningRouter.plan()` walks a fixed priority order. Earlier rules win:

| # | Condition | Route | Model called? |
|---|---|---|---|
| 1 | Message is a greeting, no image | `small_talk` | no |
| 2 | Image present but **unreadable** | `clarify` | no |
| 3 | No image at all | `general_question` | text model |
| 4 | **New image + vague prompt** | `clarify` | no |
| 5 | Refers to earlier turn/image | `follow_up` | text model only |
| 6 | Concrete request + image | *strategy selection* | yes |

Strategy selection at step 6 then picks between `read_text`, `analyse_chart`,
`describe_image` and `answer_with_image` using the user's phrasing **and** the
measured pixel structure of the image.

### Rule 4 is the interesting one

"This is an image where the user has not said what they want" is precisely the
case where a naive system silently picks a task and presents its output as if it
were the answer. Here it produces a menu instead, and records why:

```
Router chose 'clarify' — route=clarify | confidence=0.90
  ↳ A new image was supplied but the request is ambiguous
  ↳ Detected vague phrasing (words=3)
```

If an LLM classifier is configured, it gets a *second opinion* at this point —
but only if it claims the task is actually concrete. The heuristics are the
default because they cannot hallucinate a route.

## Where uncertainty comes from

Uncertainty is not invented by the model at answer time. It is **measured
upstream** and carried through:

| Source | Signal | Consequence |
|---|---|---|
| `analyze_image` | Laplacian variance low | `is_blurry` → OCR untrusted, V4 armed |
| `analyze_image` | min dimension < 64 | `is_low_resolution` → V4 armed |
| `analyze_image` | mean brightness extremes | quality note added |
| `run_ocr` | mean confidence < 0.5 | `ocr_was_untrusted` → V4 armed |
| `RoutePlan` | blurry/low-res image | `trust_ocr = False` |
| `evidence_retriever` | top score < threshold | weaker grounding → V2 may fail |

V4 then checks the answer *for hedging* whenever any of those signals fired. A
confident answer on a blurry image fails V4 and its caveat is surfaced to the
user — the model does not get to decide whether it was sure.

## Memory and follow-up resolution

```
Turn 1: [image chart.png]  "describe this"     → "a loss curve…"
           │
           ├─ ImageRecord registered (content-hash id)
           └─ annotated: summary="a loss curve…", extracted_text="Epoch 10"

Turn 2:                    "why does it increase?"
           │
           ├─ is_follow_up?  → pronoun "it"    ✓
           ├─ resolve_image_for → active image  ✓
           └─ image_context() → reuses the stored summary
```

The second turn **does not re-encode or re-analyse the image**. This is both a
cost saving and a correctness property: the follow-up is answered from the same
description the user already saw, so the two turns cannot disagree.

## Multilingual layer (English · Hindi · Odia · Bengali)

The assistant is multilingual **end to end**, not by bolting a translator onto
the front. Language is detected first and then carried through routing,
generation, validation and memory, so meaning survives a language switch.

```
  user message (en / hi / or / bn / mixed)
        │
        ▼
  ┌──────────────────┐   LanguageReport{code, confidence, mixed, script}
  │ language_detector│──▶ Unicode script blocks + Roman-Hinglish lexicon
  └────────┬─────────┘   (+ optional FastText lid.176 / lingua)
           ▼
  ┌──────────────────┐   NormalizedIntent{intent, entities, response_language,
  │  intent_manager  │──▶ needs_clarification}          ← Pydantic-validated
  └────────┬─────────┘
           ▼
  ┌──────────────────┐   the router's language-agnostic overrides: greeting,
  │ reasoning_router │──▶ clarification, and a stamped language per plan
  └────────┬─────────┘
           ▼
  ┌──────────────────┐   language-pinned system prompt; technical terms kept
  │multilingual_model│──▶ verbatim; answer-language correction pass
  └────────┬─────────┘
           ▼
  ┌──────────────────┐   V7: was the answer in the language the user needs?
  │response_validator│──▶ (plus the existing V1–V6 checks)
  └────────┬─────────┘
           ▼
      answer in the user's language, same meaning as any other language
```

### Design rules

1. **Automatic detection, never a language selector.** Unicode script blocks
   identify Devanagari (Hindi), Odia and Bengali exactly and offline. Roman
   text is resolved to English vs Hinglish by a curated lexicon, with an
   optional statistical backend. A detection confidence is always reported.
2. **Intent is normalised, not translated.** Every supported language maps onto
   one canonical intent (`explain_concept`, `follow_up`, `compare`, …), so the
   router behaves identically regardless of the language the request arrived in.
3. **Technical terms are never translated.** `Python`, `API`, `overfitting` stay
   verbatim in every language; this is enforced in the prompt and visible in the
   preserved entity list.
4. **Language switches are first-class events.** `ConversationMemory` tags each
   turn with its language, so `previous_language()` can (a) inherit language for
   a short follow-up like "Why?" and (b) tell the model a switch happened, with
   instructions to keep the topic.
5. **Ambiguity asks, it does not guess.** "Isko improve kaise karun?" with two
   candidate subjects produces a clarifying question *in the user's language*,
   not a confident wrong answer.
6. **Response language is verified.** V7 detects the language the answer came
   back in; a mismatch is re-expressed once, then flagged if still wrong.

### The normalised representation

`IntentManager` emits exactly the structure the brief specifies, validated by a
Pydantic model so a malformed LLM response can never inject a bogus intent:

```json
{
  "detected_language": "hi",
  "mixed_languages": ["en"],
  "intent": "follow_up",
  "entities": ["training data", "machine learning"],
  "response_language": "hi",
  "needs_clarification": false,
  "confidence": 0.82,
  "source_of_resolution": "heuristic"
}
```

### A worked cross-lingual conversation

```
User (en): Explain machine learning.
  → lang=en, intent=explain_concept
  → answer in English

User (hi): Ab isko easy Hindi mein samjhao.
  → lang=hi (Devanagari), switch en→hi detected
  → intent=follow_up, entities=[machine learning]     ← "isko" resolves to ML
  → response_language=hi  → answer in Hindi, same meaning

User (mix): Isme training data ka kya role hai?
  → lang=en+hi mixed (Roman), intent=follow_up
  → context retained: still about ML, now about training data

User (or): ଏହା ଓଡ଼ିଆରେ ବୁଝାନ୍ତୁ।
  → lang=or (Odia script), switch hi→or detected
  → context retained across two earlier languages → answer in Odia
```

### Graceful degradation

Every optional dependency is optional for a reason — a demo machine is not a
production cluster:

| Missing | Behaviour |
|---|---|
| ChromaDB | in-process numpy cosine over the same embeddings |
| Sentence-Transformers | keyword/IDF scoring; validator falls back to lexical overlap |
| EasyOCR | Tesseract; if neither, OCR reports `no backend` and the VLM reads the text |
| Ollama server | clear "start it with `ollama serve`" message, never a stack trace |
| Vision model | route still chosen, answer explains the missing model and how to pull it |
| FastText / lingua | detection falls back to script blocks + Roman-Hinglish lexicon |
| Language correction fails | answer is still shown, V7 flags the language mismatch |

The system is never *silently* degraded: the sidebar reports the active retrieval
backend and whether each model is available.
