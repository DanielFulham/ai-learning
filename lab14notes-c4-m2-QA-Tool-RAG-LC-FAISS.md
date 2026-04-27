# Lab 14 Notes — AI-Powered YouTube Summariser and QA Tool
## Course 4, Module 2 Lab 2

---

## What This Lab Covers

End-to-end RAG pipeline over YouTube video transcripts. Two distinct paths:

- **Summarise** — full transcript sent directly to LLM, no retrieval
- **Q&A** — transcript chunked, embedded, indexed with FAISS, relevant chunks
  retrieved before LLM generates answer

Built with LangChain FAISS wrapper, MiniLM embeddings, Claude Haiku via LCEL,
and Gradio for the UI.

---

## Stack

| Component | Lab (IBM) | This implementation | Reason |
|---|---|---|---|
| Embedding model | IBM SLATE-30M | `all-MiniLM-L6-v2` | Local, no API cost, already in Rovers stack |
| LLM | IBM Granite | Claude Haiku (`claude-haiku-4-5`) | Own credentials, LCEL compatible |
| Vector store | FAISS | FAISS | Same — LangChain wrapper |
| UI | Gradio | Gradio | Same |

SLATE-30M = IBM's equivalent of MiniLM-L6-v2. Same role in the pipeline, different provider.

---

## Key Concepts

### Two Pipeline Paths — Summarise vs Q&A

The application has two fundamentally different paths:

**Summarise path:**
```
get_transcript() → process() → full transcript → LLM
```
No chunking, no embedding, no FAISS. The full processed transcript is sent
directly to Claude in one prompt. Summarisation needs the full picture.

**Q&A path:**
```
get_transcript() → process() → chunk() → embed() → FAISS index → retrieve() → LLM
```
Chunking and retrieval are required for Q&A — the full transcript would exceed
context limits and dilute relevance. FAISS retrieves only the chunks relevant
to the specific question.

### FAISS as a LangChain Vector Store

`FAISS.from_texts()` is the LangChain wrapper that handles embed → index in one call:

```python
from langchain_community.vectorstores import FAISS

faiss_index = FAISS.from_texts(chunks, embedding_model)
```

Returns a LangChain-compatible vector store with `.similarity_search()` built in.
Equivalent to ChromaDB's `Chroma.from_documents()` — same interface, different
underlying store.

The manual position mapping from Lab 13 (raw FAISS) is handled internally.
FAISS returns documents directly, not just indices.

### LCEL Chains Replacing LLMChain

`LLMChain` is removed in current LangChain. LCEL replacement:

```python
# OLD
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run({"transcript": text})

# NEW
chain = prompt | llm
result = chain.invoke({"transcript": text}).content
```

`ChatAnthropic` returns an `AIMessage` object — `.content` extracts the string.

### Prompt Template — Values Injected at Invoke Time

The prompt template is defined once as a reusable structure. Values are
injected only at `invoke()` time:

```python
# Define shape — no data
prompt = PromptTemplate(
    input_variables=["transcript"],
    template="Summarise: {transcript}"
)

# Wire to LLM — still no data
chain = prompt | llm

# Inject data at runtime
result = chain.invoke({"transcript": processed_transcript}).content
```

The key name in `invoke()` must exactly match `input_variables`. This is the
contract between the template and the call site.

Hardcoding the value at template definition time would make the chain reusable
for only one transcript. Keeping it as a placeholder makes the same chain work
for any video.

### Index Build Once, Query Many Times

The FAISS index should be built once when the transcript is fetched — not
rebuilt on every question. The current implementation builds it in
`summarize_video()` and reuses it in `answer_question()` via global state.

```python
# Built once in summarize_video()
faiss_index = create_faiss_index(chunks, embedding_model)

# Reused in answer_question() — no rebuild
answer = generate_answer(user_question, faiss_index, qa_chain)
```

Same principle as the Rovers chatbot — ChromaDB index built at startup,
not on every query.

---

## Production Notes

- **Global state in multi-threaded Gradio is unreliable** — Gradio runs each
  button click in a separate thread. Globals written in one thread may not be
  visible in another. Use `gr.State()` for explicit state passing in production.
- **`youtube-transcript-api` v1.2.1 breaking change** — snippets changed from
  plain dicts to `FetchedTranscriptSnippet` objects. Use `i.text` and `i.start`
  not `i["text"]` and `i["start"]`. Guard with `hasattr()` if supporting both.
- **SLATE-30M = MiniLM equivalent** — IBM's embedding model. Same pipeline
  role, different provider. Swap freely without changing FAISS or LLM code.
- **Summarise path skips FAISS entirely** — FAISS is only needed for retrieval.
  Direct LLM summarisation over the full transcript is valid for short videos.
  For long videos, summarise-via-retrieval or map-reduce chunking would be needed.
- **`chunk_size` should match source content** — transcript lines are short.
  Default `chunk_size=200` is oversized. Tuned to 100 for this lab.
- **IBM lab instructions used Streamlit in description, Gradio in code** —
  ignore all Streamlit references. The implementation is entirely Gradio.
- **`LLMChain` is removed in current LangChain** — use LCEL (`prompt | llm`)
  and `.invoke()`. `.run()` and `.predict()` are also removed.

---

## Dependency Notes

This lab has significant dependency conflicts between Gradio,
huggingface_hub, transformers, and sentence-transformers.
Working pinned versions:

```
gradio==5.23.0
huggingface_hub==1.5.0
sentence-transformers (latest)
langchain-huggingface (latest)
```

The `HfFolder` import error in older Gradio versions is caused by
`huggingface_hub>=0.30` removing `HfFolder`. Resolved by upgrading to
Gradio 5.23.0 which no longer references it.

---

## Files

- `ytbot.py` — full implementation
- `.env` — `ANTHROPIC_API_KEY` (never committed)
- `requirements.txt` — pinned dependencies