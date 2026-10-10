# 🧠 Multimodal AI Assistant

An assistant that reasons over **text and images** — and shows its work.

This is *not* "upload an image, send it to a model, print the answer". Every turn
goes through an explicit pipeline that **decides** how to handle the input:

```
User Input (text / image / both)
        │
        ▼
┌───────────────────────┐
│ 1. Image validation   │  decode · size · format · sharpness · brightness
│    & measurement      │  → never raises; failures are first-class outcomes
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ 2. Reasoning Router   │  heuristics (+ optional LLM classifier) pick ONE path
│    (the decision hub) │  and record WHY
└───────────┬───────────┘
            ▼
   ┌────────┼─────────┬──────────────┐
   ▼        ▼         ▼              ▼
 OCR     Vision    Retrieval      Memory        ← tools chosen per request
 EasyOCR  Llama    ChromaDB +     conversation
 /Tesseract 3.2v   MiniLM         + image reg.
   └────────┴─────────┴──────────────┘
            ▼
┌───────────────────────┐
│ 3. Response Validator │  V1 relevance · V2 grounding · V3 numbers
│    (V1–V6 checks)     │  V4 uncertainty · V5 format/safety · V6 fabrication
└───────────┬───────────┘
            ▼
   Answer + validation status + uncertainty note
```

---

## 1. Quick start

```bash
# 1. Python dependencies
pip install -r requirements.txt

# 2. Start Ollama and pull the vision model (one-time, ~6 GB)
ollama serve &
ollama pull qwen2.5vl:7b

# 3. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### ⚠️ A note on `llama3.2-vision`

The brief suggests `llama3.2-vision`. Current Ollama builds bundle a llama.cpp
that **no longer loads its `mllama` architecture**, so it fails at load time
with `unknown model architecture: 'mllama'` before any inference happens — see
[ollama#16490](https://github.com/ollama/ollama/issues/16490),
[#16547](https://github.com/ollama/ollama/issues/16547),
[#18486](https://github.com/ollama/ollama/issues/18486). This is an upstream
bug, not something this project can fix.

The default is therefore **`qwen2.5vl:7b`** — a modern vision-language model
with the same image+text interface, and stronger at OCR and chart reading.

If you have an Ollama build that still supports `mllama`, the original model
works unchanged:

```bash
ollama pull llama3.2-vision:11b
VISION_MODEL=llama3.2-vision:11b streamlit run app.py
```

The assistant detects the `mllama` failure specifically and explains it in the
chat rather than showing a raw stack trace.

**Text model.** The routing classifier and follow-up answers use `TEXT_MODEL`
(default `llama3.2:3b`). If it isn't pulled, the assistant **automatically
switches to another local text model** at startup and says so in the sidebar,
rather than failing every text-only question. You can also set it explicitly:

```bash
ollama pull llama3.2:3b          # or:
TEXT_MODEL=llama3:latest streamlit run app.py
```

**Optional OCR backends** — the assistant degrades gracefully without them:

```bash
brew install tesseract        # then: pip install pytesseract
# or
pip install easyocr           # heavier; pulls torch
```

**Optional faster embeddings** — first run downloads `all-MiniLM-L6-v2` (~80 MB).
Without it, retrieval falls back to keyword scoring automatically.

### Offline / CI

```bash
pytest tests/ -q      # 86 tests, no Ollama, no network, no GPU required
```

### Run all six demo scenarios at once

```bash
python tools/demo_scenarios.py      # all
python tools/demo_scenarios.py 2 4  # just scenarios 2 and 4
```

This drives the real pipeline and prints the chosen route, the rationale, the
validation report and the final answer for each scenario.

---

## 2. How each assignment requirement is met

| Requirement | Where it lives | What actually happens |
|---|---|---|
| **A. Multimodal understanding** | `vision_model.py`, `pipeline.py` | Text-only, image-only and text+image all supported. The VLM is called only when pixels matter. |
| **B. Visual analysis & extraction** | `image_processor.py`, `vision_model.py` | OCR runs *only* when pixel heuristics say the image contains text. Charts get a dedicated structured-analysis prompt. Broken images never reach the model. |
| **C. Conversational context** | `conversation_memory.py` | Turns + images are registered. Referring expressions ("it", "that graph", "the middle layer") are resolved to the right image, whose earlier description is reused — no re-upload needed. |
| **D. Evidence-based responses** | `evidence_retriever.py`, `response_validator.py` | Document retrieval (ChromaDB + embeddings, keyword fallback) plus grounding checks. Claims only ship if traceable. |
| **E. Ambiguity handling** | `reasoning_router.py` | "check this image" → a clarifying question (describe / extract text / specific detail) instead of a silent guess. |
| **F. Response validation** | `response_validator.py` | Six named checks (V1–V6); weak answers are downgraded to `qualified` with an explicit caveat. |
| **G. Intelligent decision-making** | `reasoning_router.py` | Not one pipeline for all: nine distinct routes, each with a recorded rationale. |

### The routes the router can choose

| Route | Triggered when | Model calls |
|---|---|---|
| `clarify` | Vague prompt, or unreadable image | **none** — asks instead |
| `small_talk` | Greeting | none |
| `describe_image` | "describe this image" | vision |
| `read_text` | "read / extract the text" | OCR + vision |
| `analyse_chart` | graph/plot detected or asked about | OCR + vision |
| `answer_with_image` | specific question + image | vision (+ retrieval) |
| `follow_up` | refers to an earlier image | text model only (memory reuse) |
| `general_question` | no image | text model (+ retrieval) |
| `refuse_unsupported` | evidence insufficient | none |

---

## 3. The validation checks (V1–V6)

| Check | Question it answers | Method |
|---|---|---|
| **V1 relevance** | Does the reply address what was asked? | embedding similarity + token overlap (+ numeral bonus for count questions) |
| **V2 grounding** | Is every claim traceable to evidence? | similarity vs. the evidence blob (image summary + OCR + retrieved docs) |
| **V2b entailment** | Does the model agree a key claim is supported? | optional LLM spot-check on numeric claims |
| **V3 numbers** | Do stated values appear in the evidence? | deterministic set difference on numerals |
| **V4 uncertainty** | Is hedging present when the image is unclear? | hedge-word detection, triggered by blur/low-res/tiny/OCR-confidence |
| **V5 format/safety** | Sensible length? No capability refusals? | deterministic rules |
| **V6 fabrication** | Are there invented names/details? | proper-noun extraction vs. evidence |

Statuses: `✅ pass` · `⚠️ qualified` · `❓ needs_clarification` · `⛔ failed`.

---

## 4. Project structure

```
multimodal_ai_assistant/
├── app.py                    # Streamlit UI: chat, pinned image, history sidebar
├── ui_theme.py               # Styling + internal-state → user-language mapping
├── chat_store.py             # Persistent conversations (data/chats/*.json)
├── vision_model.py           # Ollama client: describe / read / chart / answer
├── pipeline.py               # Orchestrator: wires every stage together
├── config.py                 # All tunables, env-overridable
│
├── image_processor.py        # Validation, quality metrics, OCR, encoding
├── vision_model.py           # Ollama client: describe / read / chart / answer
├── reasoning_router.py       # ★ Intelligent decision-making
├── conversation_memory.py    # Turns, image registry, follow-up resolution
├── evidence_retriever.py     # ChromaDB + embeddings, keyword fallback
├── response_validator.py     # ★ V1–V6 checks and status decisions
│
├── data/
│   ├── documents/            # Trusted corpus (3 seeded .md files)
│   ├── chats/                # Saved conversations + index.json
│   ├── chroma_store/         # Persistent vector store
│   ├── demo_images/          # Generated fixtures for the evaluation
│   └── upload_cache/
│
├── tools/
│   ├── make_demo_images.py   # Regenerate the demo fixtures
│   └── demo_scenarios.py     # ★ Runs all six scenarios end-to-end (scripted demo)
│
├── docs/
│   ├── ARCHITECTURE.md       # Design rationale in depth
│   └── DEMO_OUTPUT.txt       # Real captured output of the six scenarios
│
└── tests/
    ├── conftest.py           # Synthetic images + FakeVisionModel
    ├── test_image_input.py   # Validation, quality, OCR gating, routing
    ├── test_memory.py        # Context, follow-up resolution, ambiguity
    ├── test_chat_store.py    # Persistence, history, one-image-per-chat
    └── test_validation.py    # V1–V6 + end-to-end pipeline + retriever
```

---

## 5. Verified results

All six scenarios were run against the real pipeline (`qwen2.5vl:7b` + `llama3`).
Full transcript: [`docs/DEMO_OUTPUT.txt`](docs/DEMO_OUTPUT.txt). Regenerate with:

```bash
python tools/make_demo_images.py      # once, to create the fixtures
python tools/demo_scenarios.py        # all six, or pass 1 3 4 for a subset
```

| # | Scenario | Route chosen | Validation | Key observed behaviour |
|---|---|---|---|---|
| 1 | Chart + specific question | `analyse_chart` | ✅ pass | Read the real axes/legend and identified **epoch 6** as the divergence point |
| 2 | Image + "check this" | `clarify` | ❓ needs_clarification | Asked for clarification; **made no model call at all** |
| 3 | Follow-up "the middle layer" | `follow_up` | ✅ pass | Resolved to the stored image (no re-upload) and pulled **4 evidence chunks** |
| 4 | Blurry image + count question | `answer_with_image` | ✅ pass | `sharpness=0.3` measured, OCR marked **untrusted**, answer hedged explicitly |
| 5 | "What is overfitting?" | `general_question` | ✅ pass | Retrieved 3 documents (top score **0.71**) and cited their remedies |
| 6 | "Exact accuracy value?" | `answer_with_image` | ✅ pass | Model **refused to fabricate** — correctly said no value exists in the diagram |

**Also verified through the running UI** (real browser, real file upload, not the
scripted pipeline): uploading `loss_curve.png` and asking *"When does the
validation loss start increasing in this graph, and why?"* returned:

> Chart Type: Line graph. Title: Model Loss over Training. X-axis: Epoch
> (0 to 20). Y-axis: Loss (0.0 to 1.0). Series/Legend: Training loss (blue),
> Validation loss (orange). The validation loss decreases initially but starts
> increasing after epoch 6. Validation loss minimum: approximately 0.3 at
> epoch 6 … likely due to overfitting.

It read the real title, axes and legend colours, and identified the correct
epoch — with 0 console errors and 0 failed network requests.

Additionally verified directly (OCR path, `data/demo_images/handwritten_math.png`):

```
looks_like_text=True  chart=False  blur=13
ocr attempted=True  succeeded=True  backend=tesseract  conf=0.959
text: "Question 3: A model trains for 20 epochs. Training loss falls from
       0.90 to 0.12. Validation loss falls to 0.31 at epoch 6 and then
       rises to 0.72 by epoch 20. Why does this happen?"
```

Tesseract transcribed the simulated handwriting at **95.9% confidence**, and
on `loss_curve.png` it picked up the axis labels (`0.0`–`1.2`, `Epoch`).

Two behaviours worth highlighting for the evaluation:

- **Scenario 4** — the uncertainty was *measured* (`blur_score=0.3` → `is_blurry`),
  propagated into the plan (`trust_ocr=False`), and then confirmed by V4. The
  model's own answer independently hedged, which is the behaviour we wanted to
  induce rather than merely check for.
- **Scenario 6** — asked for a value that does not exist, the system declined to
  invent one. This is the failure mode the whole validator exists to prevent.

---

## 6. Using the interface

The chat works the way you'd expect from a modern assistant:

**Upload once, ask many times.** Drop an image in and it is *attached to the
conversation*, not to a single message. A small strip above the composer shows
what's attached (`🖼 loss_curve.png · Active`), and every later question is
answered against it — no re-uploading:

```
You:      [loss_curve.png]  Describe this image and point out anything important.
Assistant: A line chart titled "Model Loss over Training" …
You:      Why does the validation loss increase after that point?   ← still the same image
Assistant: Because the model has started fitting training noise …
You:      What should be done to prevent this?                      ← still the same image
Assistant: Early stopping, regularisation, more data …
```

The image renders **once** in the transcript (where it was introduced), not on
every follow-up. Click **Detach image** to start fresh text-only questions.

### An attached image does not capture every later question

Having an image in context is not a reason to answer *everything* through it.
The router checks whether a new question actually relates to the pinned image,
and answers it directly when it doesn't:

| Question (with a chart attached) | Answered as |
|---|---|
| "Why does the validation loss increase in this graph?" | about the image |
| "What colour is the validation line?" | about the image |
| "What does the middle layer do?" | about the image (names a part) |
| "in the graph?" / "and why?" | about the image (continues the subject) |
| "What is the capital of France?" | **on its own** — not dragged to the chart |
| "Write a Python function to reverse a string" | **on its own** |
| "Who wrote Moby Dick?" | **on its own** |

The test is vocabulary overlap: a question naming a subject the image summary
does not mention is treated as a new topic. Erring toward "answer it directly"
is deliberate — a well-answered text question beats a confident answer about the
wrong thing.

**Chat history is saved.** Every conversation is written to `data/chats/` and
listed in the sidebar under **Recent**, newest first, with an auto-generated
title from your first message. Clicking a conversation reloads the full
transcript *and* re-attaches its image, so follow-ups still work after a
refresh, a Streamlit restart, or a reboot. Use **＋ New chat** to start over, or
**✕** to delete one.

Because the image is stored once per conversation (not per message), a long
Q&A over one chart stays small on disk.

**Two surfaces.** By default you see a clean product view: the answer, a trust
indicator (`✓ Verified` / `⚠️ Double-check this` / `❓ Question for you`) and
which reference documents informed it. Flip **Developer mode** in the sidebar to
additionally see the routing decision, rationale, image diagnostics, OCR output,
retrieval scores and every V1–V6 validation check.

---

## 7. Demo script (for the evaluation)

Run these in order. Each one exercises a different capability.

### 1️⃣ Image + clear question → visual explanation
Upload an ML training-curve screenshot, ask:
> *"When does the validation loss start increasing, and why?"*

Expect: route `analyse_chart` or `answer_with_image`, a grounded answer citing the
epoch and the divergence, status `✅ pass`. Open **Reasoning trace** to show the
router's rationale.

### 2️⃣ Image without a clear question → clarification
Upload anything, ask:
> *"check this image"*

Expect: route `clarify`, a numbered menu (describe / extract text / specific
detail), and **no model call at all** — visible in the trace. This proves the
system decides rather than blindly generating.

### 3️⃣ Follow-up on the same image → memory
After turn 1, ask (with no re-upload):
> *"why does it increase?"*

Expect: route `follow_up`, answered from the stored image context. The trace
shows *"Reused stored image context (no re-upload needed)"*.

### 4️⃣ Blurry image → uncertainty
Upload a deliberately blurred photo and ask:
> *"how many people are in this image?"*

Expect: an answer that gives what is clearly visible **and** flags the obscured
region (V4 uncertainty passes *because* it hedged). A confident unhedged count
would be flagged and downgraded.

### 5️⃣ External-fact question → retrieval
Ask with no image:
> *"what is overfitting and how do you fix it?"*

Expect: route `general_question` with retrieval on; the sidebar shows which
documents were indexed; retrieved chunks are listed under **Image diagnostics &
evidence**.

### 6️⃣ Unsupported claim → qualified
Ask something the evidence cannot support, e.g. about a chart with an invented
value:
> *"what is the exact accuracy at epoch 500?"*

Expect: validation status `⚠️ qualified` with a caveat naming the untraceable
number (V3), rather than a confident fabrication.

---

## 8. Configuration

All settings are environment-overridable (see `config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `VISION_MODEL` | `qwen2.5vl:7b` | Ollama vision model (see note above) |
| `TEXT_MODEL` | `llama3.2:3b` | Model for routing/follow-ups |
| `OCR_ENABLED` | `1` | Master OCR switch |
| `OCR_BACKEND` | `auto` | `auto`/`easyocr`/`tesseract`/`none` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Retrieval + validation embeddings |
| `RETRIEVAL_MIN_SCORE` | `0.25` | Evidence acceptance threshold |
| `GROUNDING_MIN_SCORE` | `0.30` | Minimum grounding score to pass V2 |
| `MAX_TURNS` | `12` | Turns kept in full in memory |

Example:
```bash
TEXT_MODEL=llama3:latest OCR_BACKEND=tesseract streamlit run app.py
```

---

## 9. Design decisions worth defending

- **The router never *depends* on the LLM.** Heuristics are deterministic and
  always available. The model classifier is opt-in and only consulted when the
  heuristics are genuinely ambiguous — an LLM cannot hallucinate a route.
- **Failure is a first-class outcome.** A corrupt upload produces
  `ImageDiagnostics(valid=False)`, which routes to `clarify` — it never raises
  and never reaches the model.
- **OCR is gated, not always-on.** Pixel heuristics decide whether OCR is worth
  running, and image quality decides whether its output can be trusted. Blurry
  OCR is a *signal for uncertainty*, not a fact.
- **Validation is mostly deterministic.** V3/V4/V6 need no model at all, so the
  safety net cannot itself be talked out of working by a bad generation.
- **Everything degrades gracefully.** No ChromaDB → keyword search. No embedding
  model → lexical checks. No OCR → VLM-only reading. No Ollama → a clear
  instruction, not a stack trace.
