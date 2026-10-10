# Multimodal AI Assistant

A local, image-aware assistant for asking questions about images and reference
documents. It uses a reasoning pipeline to choose between image analysis, OCR,
document retrieval, conversation memory, or a clarification question, then
checks the candidate answer before showing it.

The interface is built with Streamlit. Model inference runs locally through
[Ollama](https://ollama.com/).

## Features

- Ask questions about an image, extract visible text, inspect charts, or ask
  text-only questions.
- Attach an image once and ask follow-up questions about it in the same chat.
- Route ambiguous image requests to clarification instead of guessing silently.
- Measure image quality and surface uncertainty for blurry or low-resolution
  inputs.
- Use OCR when appropriate, with EasyOCR and Tesseract as optional backends.
- Retrieve supporting passages from the Markdown/text files in the local
  knowledge base.
- Validate relevance, grounding, numeric claims, uncertainty, format, and
  possible fabrication before presenting an answer.
- Save chat history locally and optionally enable a developer view with routing,
  OCR, diagnostics, retrieved evidence, and validation details.
- Fall back to keyword retrieval and lexical validation when optional embedding
  or vector-store components are unavailable.

## How it works

```text
Text and/or image
        |
        v
Image validation and quality checks
        |
        v
Reasoning router
  |        |         |          |
  OCR    Vision   Retrieval   Memory
  |        |         |          |
  +--------+---------+----------+
                   |
                   v
          Response validation
                   |
                   v
         Answer and trust signal
```

The router records why it selected a path. A greeting or unclear image request
may not need a model call; image questions use the vision model; text questions
can use the local language model and retrieve relevant passages. Follow-up
questions can reuse the current conversation's image context.

## Requirements

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running locally.
- The models used by the app pulled into Ollama (see below).
- Git and pip.

Image text extraction can use Tesseract or EasyOCR. Both are optional: without
an OCR backend the vision model can still read images. Semantic retrieval uses
Sentence-Transformers and ChromaDB when available, with a keyword fallback.

## Quick start

From the repository root:

```bash
git clone https://github.com/abhinandan761211-code/Multimodal-AI-Assistant.git
cd Multimodal-AI-Assistant/multimodal_ai_assistant

python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Install and start Ollama, then pull the default models:

```bash
ollama pull qwen2.5vl:7b
ollama pull llama3.2:3b
ollama serve
```

In a second terminal, from `multimodal_ai_assistant/`, start the app:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (usually
<http://localhost:8501>).

The default vision model is `qwen2.5vl:7b`; the default text model is
`llama3.2:3b`. If the configured text model is unavailable, the app attempts to
select an installed local text model. Model inference stays on the Ollama
instance configured for the app.

### Optional OCR setup

Tesseract requires both the system executable and the Python wrapper (the
wrapper is listed in `requirements.txt`):

```bash
brew install tesseract                         # macOS
sudo apt-get install tesseract-ocr              # Debian/Ubuntu
```

EasyOCR is another optional backend and can be installed with:

```bash
python -m pip install easyocr
```

## Using the app

1. Upload an image using **Add an image**, or enter a text-only question.
2. Ask a specific question, such as “When does the validation loss start
   increasing?” or “Extract the text from this image.”
3. Continue with follow-up questions without uploading the image again.
4. Use **Remove image** to detach the image from the current chat.
5. Enable **Developer mode** in the sidebar to inspect how an answer was
   produced.

Uploads accepted by the interface: PNG, JPEG, WEBP, BMP, and GIF. The configured
upload size limit is 20 MB.

## Configuration

Settings are read from environment variables in
[`config.py`](multimodal_ai_assistant/config.py). For example:

```bash
VISION_MODEL=qwen2.5vl:7b \
TEXT_MODEL=llama3.2:3b \
OCR_BACKEND=tesseract \
streamlit run app.py
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `VISION_MODEL` | `qwen2.5vl:7b` | Image-capable model |
| `TEXT_MODEL` | `llama3.2:3b` | Text and follow-up model |
| `OCR_ENABLED` | `1` | Enable/disable OCR |
| `OCR_BACKEND` | `auto` | `auto`, `easyocr`, `tesseract`, or `none` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Retrieval/validation embedding model |
| `RETRIEVAL_TOP_K` | `4` | Maximum retrieved passages |
| `RETRIEVAL_MIN_SCORE` | `0.25` | Minimum retrieval score |
| `GROUNDING_MIN_SCORE` | `0.30` | Minimum grounding score |
| `MAX_TURNS` | `12` | Conversation turns retained in memory |
| `PERSIST_CHATS` | `1` | Persist chat history locally |

Other image, timeout, temperature, and storage settings are also available in
`config.py`.

## Knowledge base and local data

Add `.md` or `.txt` reference material to
[`data/documents/`](multimodal_ai_assistant/data/documents/). The app indexes
those files to retrieve evidence for relevant questions.

Chat history is stored under `data/chats/`; the vector index and upload cache
are also stored locally. These directories can contain user-provided content.
Do not commit personal chats, private uploads, or other sensitive data.

The first use of semantic embeddings may download the configured model from
Hugging Face. Ollama model downloads also require network access. After models
and dependencies are installed, inference is performed locally.

## Development

Run the test suite from `multimodal_ai_assistant/`:

```bash
pytest tests/ -q
```

The unit tests use a fake vision model and synthetic images, so they do not
require a running Ollama service, a GPU, or model inference.

Run the end-to-end demo scenarios (requires Ollama and the vision model):

```bash
python tools/make_demo_images.py
python tools/demo_scenarios.py             # all scenarios
python tools/demo_scenarios.py 2 4         # selected scenarios
```

## Project structure

```text
multimodal_ai_assistant/
├── app.py                    # Streamlit chat interface
├── config.py                 # Defaults and environment-based configuration
├── pipeline.py               # Coordinates processing and answer validation
├── reasoning_router.py       # Chooses the processing strategy
├── vision_model.py           # Ollama model client
├── image_processor.py        # Image checks, OCR, and encoding
├── evidence_retriever.py     # Local document retrieval
├── response_validator.py    # Relevance, grounding, and safety checks
├── conversation_memory.py   # In-memory conversation and image context
├── chat_store.py             # Local conversation persistence
├── ui_theme.py               # Streamlit presentation and styling
├── data/
│   ├── documents/            # Local reference corpus
│   └── chats/                # Persisted conversations
├── docs/
│   ├── ARCHITECTURE.md       # Design and processing details
│   └── DEMO_OUTPUT.txt       # Captured demo output
├── tests/                    # Offline unit tests
└── tools/                    # Demo image and scenario utilities
```

For more implementation details, see
[`docs/ARCHITECTURE.md`](multimodal_ai_assistant/docs/ARCHITECTURE.md).
