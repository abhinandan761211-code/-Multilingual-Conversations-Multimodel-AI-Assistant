# Multilingual Conversations

This document describes the multilingual layer that extends the existing
Multimodal AI Assistant. The core promise is simple to state and hard to build:

> **The language may change between turns. The meaning must not.**

Translating a reply is not enough. The assistant has to detect the language,
understand the *intent* behind a mixed-language message, keep the conversation's
context when the user switches language, resolve ambiguous references by asking
instead of guessing, and answer in the language the user is actually using.

---

## 1. Supported languages

| Code | Language | Script           | Notes                                              |
|------|----------|------------------|----------------------------------------------------|
| `en` | English  | Latin            | Default / existing language                        |
| `hi` | Hindi    | Devanagari + Latin (Hinglish) | Handles Roman-script code-mixing      |
| `or` | Odia     | Odia             | Regional language support                          |
| `bn` | Bengali  | Bengali          | Regional language support                          |

Any other language is *recognised* (reported as out-of-scope) and answered in a
fallback language (`UNSUPPORTED_LANGUAGE_FALLBACK`, default English) rather than
being silently mis-classified as English.

---

## 2. Architecture

The layer sits *around* the existing reasoning pipeline — the vision path is
untouched, so image understanding keeps working exactly as before.

```text
User message (en / hi / or / bn / mixed)
        │
        ▼
┌──────────────────────────┐
│ 1. language_detector.py  │  Dominant language + code-mixing + confidence
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 2. intent_manager.py     │  Normalised intent / entities / needs_clarification
└──────────────────────────┘  (validated with Pydantic)
        │
        ▼
┌──────────────────────────┐
│ 3. reasoning_router.py   │  Chooses the route *and* the response language
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 4. multilingual_model.py │  Language-steered generation (Qwen3 via Ollama)
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 5. response_validator.py │  V7: is the answer in the requested language?
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│ 6. conversation_memory   │  Per-turn language, intent, subjects → continuity
└──────────────────────────┘
        │
        ▼
   Final answer, in the user's language
```

Each stage is deterministic-first: heuristics always run and always decide, and
the model is consulted only as a *refinement*. A missing model, a missing
FastText file, or a malformed JSON reply degrades the quality of a decision but
never breaks a turn.

---

## 3. The normalised representation

Every message is reduced to a language-independent object before it influences
routing. This is the contract between the language layer and the rest of the
system (`intent_manager.NormalizedIntent`):

```json
{
  "detected_language": "hi",
  "mixed_languages": ["en"],
  "intent": "explain_concept",
  "entities": ["training data", "machine learning"],
  "response_language": "hi",
  "needs_clarification": false,
  "is_follow_up": true,
  "language_confidence": 0.83,
  "confidence": 0.7,
  "source_of_resolution": "heuristic"
}
```

`intent` is drawn from a fixed vocabulary, so "explain machine learning",
"machine learning samjhao" and "ମେସିନ ଲର୍ନିଂ ବୁଝାନ୍ତୁ" all become the *same*
value and therefore take the same path through the pipeline.

---

## 4. How each requirement is met

### A. Automatic language identification
`language_detector.py` combines two signals:

* **Unicode script blocks** — Devanagari, Odia and Bengali are distinct ranges,
  so native-script text is identified exactly, offline, with no dependency.
* **A Roman-Hinglish lexicon** — because "mera model train nahi ho raha" is all
  Latin. A curated Hindi marker list (`kya`, `nahi`, `samjhao`, `matlab`, …) and
  an English marker list are scored; the winner and the margin give a confidence.

The shared danda `।` (used by Hindi, Odia **and** Bengali) is deliberately
excluded, so an Odia sentence is never mistaken for Hindi.

Optional statistical backends (FastText `lid.176` or `lingua`) can be enabled
for extra accuracy on short Roman text; both are optional.

### B. Mixed-language understanding
Inputs like *"Mera Python code error de raha hai, please explain in English"* are
handled in three parts:

* **Intent** — recognised from multilingual keyword tables (`error`, `explain`).
* **Code-mixing** — reported as `mixed_languages`, not silently collapsed.
* **Desired output language** — the explicit request (*"in English"*) is parsed
  and overrides the detected input language.

Technical terms (API, Python, model, training data, overfitting, …) are extracted
as `entities` and explicitly protected from translation in every prompt.

### C. Context retention across a switch
`conversation_memory.py` tags every turn with its language, intent and subject
nouns, and:

* `previous_language()` carries the language across short, signal-less messages,
* `context_for_prompt()` prepends a `[Conversation languages: English, Hindi. …]`
  header so the model treats the switch as *continuity*, not a topic change,
* `recent_subjects()` keeps the candidate subjects a later pronoun may refer to.

The scenario from the brief works end to end: **EN question → HI instruction →
EN follow-up** stays on topic, and the memory records `["en", "hi", "en"]`.

### D. Ambiguity resolution
When a pronoun-led request (`"Isko improve kaise karun?"`) has more than one
plausible subject, `intent_manager` sets `needs_clarification` and the router
chooses the **clarify** route. The clarification is generated in the user's
language and names the candidate subjects, e.g.:

> Aap kis cheez ko improve karna chahte hain — accuracy, user interface?

Guessing is never the fallback.

### E. Consistent responses
Two mechanisms keep meaning stable across languages:

* The pipeline re-expresses an answer once if it came back in the wrong
  language (`multilingual_model.translate_answer`), preserving technical terms.
* The validator's **V7-language** check compares the detected language of the
  answer against the requested one and flags a mismatch as a warning.

---

## 5. Configuration

All knobs live in `config.py` and can be overridden by environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `TEXT_MODEL` | `qwen3:4b` | Multilingual reasoning model |
| `LANGUAGE_BACKEND` | `auto` | `auto` \| `heuristic` (offline, no deps) |
| `FASTTEXT_LID_PATH` | — | Path to a FastText `lid.176` model |
| `LANGUAGE_CONFIDENCE_THRESHOLD` | `0.55` | Below this, ask instead of assume |
| `DEFAULT_LANGUAGE` | `en` | Fallback for unclassifiable input |
| `UNSUPPORTED_LANGUAGE_FALLBACK` | `en` | Reply language for other languages |
| `ENFORCE_RESPONSE_LANGUAGE` | `1` | Enable the corrective re-expression pass |

The **vision model is unchanged**; multilingual support wraps the existing
pipeline rather than replacing any part of it.

### Setup

```bash
# 1. Pull a model that is strong in all four languages.
ollama pull qwen3:4b

# 2. (Optional) improve short Roman-script detection with FastText.
pip install fasttext-wheel
mkdir -p models && cd models
curl -LO https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
cd ..

# 3. Run.
streamlit run app.py
```

With no FastText model present, `LANGUAGE_BACKEND` falls back to the fully
offline heuristic path automatically.

---

## 6. Seeing it work

The UI shows a **language chip** under every answer (`🌐 हि · Hindi`), plus
`⇄ language switch` and `mixed` chips when those apply.

**Developer mode** adds a *Language* section to each turn showing:

* detected language, confidence, method (`script` / `fasttext` / `lingua`) and script,
* code-mixing, with the other languages listed,
* the language the answer was actually written in,
* whether a switch was handled and whether the answer was re-expressed.

---

## 7. Evaluation

`tests/` contains three multilingual suites (55 tests):

| File | What it proves |
|------|----------------|
| `test_languages.py` | All four languages identified; Hinglish, mixed input, explicit output-language requests |
| `test_language_switching.py` | Cross-lingual intent stability, switch detection, ambiguity → clarification, V7 language validation |
| `test_conversation_context.py` | Context retention across switches, subject tracking, native-script serialisation |

Run them with:

```bash
python -m pytest tests/ -q
```

The assignment's evaluation table maps directly onto these tests:

| Test | Covered by |
|------|-----------|
| Question in English → English response | `TestNativeScriptDetection`, `TestResponseLanguageValidation` |
| Hindi follow-up → previous context preserved | `TestContextAcrossLanguages`, `TestLanguageSwitchRouting` |
| Odia / Bengali question → appropriate-language response | `TestNativeScriptDetection` |
| Hinglish + English technical terms → correct intent | `TestRomanScriptDetection`, `TestCrossLingualIntent` |
| Short query "Why?" → resolves previous message | `TestContextAcrossLanguages` |
| Ambiguous query → clarification, not a guess | `TestAmbiguityResolution` |
| Knowledge question → consistent facts across languages | `TestCrossLingualConsistency` |
