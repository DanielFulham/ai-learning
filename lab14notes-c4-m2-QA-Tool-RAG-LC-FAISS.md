# Course 4 — Lab 14: AI-Powered YouTube Summariser and QA Tool

RAG pipeline over YouTube video transcripts. Summarises videos and answers questions about their content using FAISS for retrieval, MiniLM for embeddings, and Claude Haiku as the LLM.

Built as part of the IBM RAG and Agentic AI Professional Certificate — Course 4, Module 2.

---

## What It Does

- **Summarise** — fetches a YouTube transcript and generates a concise summary via Claude Haiku
- **Q&A** — chunks the transcript, builds a FAISS index, retrieves relevant chunks, and answers specific questions about the video

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Local, no API cost, already in Rovers stack |
| LLM | Claude Haiku (`claude-haiku-4-5`) | Own credentials, LCEL compatible |
| Vector store | FAISS via LangChain | `FAISS.from_texts()` — same interface as ChromaDB |
| UI | Gradio `gr.Blocks` | Sufficient for prototype validation |

---

## Architecture

Onion architecture — dependencies injected from the composition root (`ytbot.py`). No globals, no cross-layer imports. LLM and embedding model initialised once at startup in `config.py`.

```
course4-module2-lab-2/
├── ytbot.py                  — Composition root, Gradio UI, dependency wiring
├── config.py                 — Singletons: llm, embedding_model, credentials
├── application/
│   ├── summarise.py          — summarize_video()
│   └── qa.py                 — answer_question(), _generate_answer()
├── domain/
│   └── transcript.py         — get_transcript(), process(), chunk_transcript()
├── infra/
│   ├── embedding.py          — setup_embedding_model()
│   ├── llm.py                — initialize_llm()
│   └── vector_store.py       — create_faiss_index(), retrieve()
├── .env                      — ANTHROPIC_API_KEY (never committed)
└── requirements.txt
```

### Layer Rules

- `domain/` — pure Python, no framework imports, no AI libraries
- `infra/` — wraps external libraries (FAISS, HuggingFace, Anthropic)
- `application/` — orchestrates domain + infra, contains business logic. Private methods prefixed with `_`
- `config.py` — singletons initialised once at startup, imported by `ytbot.py`
- `ytbot.py` — wires everything together, Gradio only, no business logic

---

## Two Pipeline Paths

**Summarise** — no retrieval, full transcript sent directly to LLM:
```
get_transcript() → process() → LLM
```

**Q&A** — FAISS retrieval before LLM:
```
get_transcript() → process() → chunk() → FAISS index → retrieve() → LLM
```

The FAISS index is built once during summarisation and passed via `gr.State()` to all subsequent Q&A queries — no rebuild per question.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

Run:
```bash
python ytbot.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Key Decisions

**Singletons in `config.py`** — `llm` and `embedding_model` are initialised once when the module is first imported. Python module imports are cached — no re-initialisation on subsequent button clicks.

**`gr.State()` over globals** — Gradio runs each button click in a separate thread. Module-level globals are unreliable across threads. `gr.State()` passes state explicitly between components, eliminating the threading issue entirely.

**Index built once at summarise time** — embedding and indexing is an offline cost paid once when the transcript is fetched, not on every question. The index is stored in `gr.State()` and reused for all Q&A queries.

**Dependency injection via composition root** — `ytbot.py` imports all dependencies and passes them into application functions. Application and domain layers have no knowledge of where their dependencies come from. Same pattern as the books search Onion architecture from Lab 9.

**LCEL over `LLMChain`** — `LLMChain` is removed in current LangChain. `prompt | llm` with `.invoke()` is the current standard.

**MiniLM over IBM SLATE-30M** — same pipeline role, no IBM credentials required, already used in the Rovers chatbot stack.

---

## Dependency Notes

Significant version conflicts between Gradio, huggingface_hub, transformers, and sentence-transformers. Working pinned versions in `requirements.txt`. Key constraint: `huggingface_hub==1.5.0` to satisfy both Gradio 5.23.0 and transformers.