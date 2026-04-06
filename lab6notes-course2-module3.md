# Course 2 — Lab 6: Build an AI Icebreaker Bot with LlamaIndex & IBM Granite

## Lab Overview

**Course:** IBM RAG and Agentic AI Professional Certificate — Course 2  
**Lab:** Build an AI Icebreaker Bot with LlamaIndex & IBM Granite  
**Completed:** 6 April 2026  
**Time taken:** ~3 hours

Multi-file LlamaIndex RAG application using IBM Watsonx embeddings and Granite LLM. Covers the full LlamaIndex pipeline — document loading, chunking into nodes, vector indexing, and query engine — wrapped in both a CLI and Gradio web interface.

---

## What Was Built

A LinkedIn profile analyser that:
1. Fetches a LinkedIn profile as JSON (mock data from IBM S3)
2. Chunks the JSON into nodes using `SentenceSplitter`
3. Embeds nodes using IBM Watsonx `slate-125m-english-rtrvr-v2`
4. Indexes into `VectorStoreIndex`
5. Queries with `as_query_engine()` using custom `PromptTemplate`
6. Generates 3 facts about the person and answers follow-up questions

---

## Project Structure

```
icebreaker/
  config.py               — prompt templates, model IDs, chunk settings
  main.py                 — CLI entry point, orchestrates the pipeline
  app.py                  — Gradio web UI wrapping the same logic
  modules/
    data_extraction.py    — fetches mock LinkedIn JSON from IBM S3
    data_processing.py    — SentenceSplitter → VectorStoreIndex
    llm_interface.py      — WatsonxEmbeddings + WatsonxLLM
    query_engine.py       — as_query_engine() for facts + Q&A
```

---

## Key Concepts

### LlamaIndex RAG Pipeline

```
OFFLINE (index build)
─────────────────────────────────────────
JSON dict
  → json.dumps() → string
  → Document(text=json_string)
  → SentenceSplitter(chunk_size=400)     # chunk_size in TOKENS not characters
  → nodes (23 from mock LinkedIn profile)
  → VectorStoreIndex(nodes, embed_model) # embeds + stores in one call

ONLINE (per query)
─────────────────────────────────────────
User query
  → index.as_query_engine(llm=watsonx_llm, text_qa_template=prompt)
  → query_engine.query("question")       # embed + retrieve + augment + generate
  → response.response                    # string answer
```

### LlamaIndex vs LangChain Vocabulary

| LangChain | LlamaIndex |
|---|---|
| `Document` chunk | `Node` |
| `RecursiveCharacterTextSplitter` | `SentenceSplitter` |
| `VectorStore` + `.as_retriever()` | `VectorStoreIndex` (unified) |
| `RetrievalQA` chain | `QueryEngine` |
| `chunk_size` in characters | `chunk_size` in **tokens** |

### Three Levels of LlamaIndex Abstraction

```python
# Level 1 — manual retrieval
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("question")

# Level 2 — response synthesizer handles augmentation + LLM
response = response_synthesizer.synthesize(query="question", nodes=nodes)

# Level 3 — query engine, everything in one call (used in this lab)
query_engine = index.as_query_engine(llm=llm, text_qa_template=prompt)
response = query_engine.query("question")
```

### Custom Prompt Template in LlamaIndex

LlamaIndex uses `{context_str}` and `{query_str}` as reserved placeholders — equivalent to LangChain's `{context}` and `{question}`:

```python
from llama_index.core import PromptTemplate

template = """
Context: {context_str}
Question: {query_str}
Answer using only the context. Say "I don't know" if not available.
"""

prompt = PromptTemplate(template=template)
query_engine = index.as_query_engine(
    llm=watsonx_llm,
    text_qa_template=prompt,
    similarity_top_k=5
)
```

### IBM Watsonx in LlamaIndex

This lab uses LlamaIndex-native IBM integrations — different packages and import paths from the LangChain IBM integrations used in earlier labs:

```python
# LlamaIndex IBM integrations (this lab)
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM

# LangChain IBM integrations (previous labs)
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
```

Both use the same Watsonx credentials — just different wrappers.

```python
# Embedding model
watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    apikey=os.getenv("WATSONX_APIKEY"),
    truncate_input_tokens=3,
)

# LLM
watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-4-h-small",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    apikey=os.getenv("WATSONX_APIKEY"),
    temperature=0.0,
    max_new_tokens=500,
    additional_params={
        "decoding_method": "sample",
        "min_new_tokens": 1,
        "top_k": 50,
        "top_p": 1,
    }
)
```

### Mock Data Pattern

Proxycurl (the LinkedIn API used in the lab) was discontinued in February 2025. The lab falls back to mock data — a pre-built LinkedIn JSON file hosted on IBM S3:

```python
# data_extraction.py
if mock:
    response = requests.get(config.MOCK_DATA_URL, timeout=30)
    data = response.json()  # returns LinkedIn-shaped dict
```

The mock data is fetched over HTTP at runtime — not a local file. The full pipeline runs identically against it.

---

## Bugs Hit & Fixed

| Bug | Cause | Fix |
|---|---|---|
| `NameError: name 'config' is not defined` | `import config` missing from `main.py` | Add `import config` at top |
| `Did not find 'apikey' or 'token'` | `apikey` not passed to `WatsonxEmbeddings` / `WatsonxLLM` | Add `apikey=config.WATSONX_API_KEY` to both |
| `module 'config' has no attribute 'WATSONX_API_KEY'` | Attribute missing from `config.py` | Add `WATSONX_API_KEY = os.getenv("WATSONX_APIKEY")` |
| `Model 'ibm/granite-3-2-8b-instruct' is not supported` | Stale model ID in lab | Update to `ibm/granite-4-h-small` |
| `TypeError: unhashable type: 'dict'` in Gradio | Jinja2/Gradio 4.19.2 version incompatibility | `pip install --upgrade gradio` |
| `Data incompatible with messages format` | Old list-of-lists chat history `[[user, bot]]` incompatible with upgraded Gradio | Update `chat_with_profile` to use `[{"role": "user", "content": ...}]` format |


---

## Local vs Lab Environment

| Lab (Linux cloud IDE) | Local (Windows PowerShell) |
|---|---|
| `wget -qO- url \| tar -xf - -C dir` | `Invoke-WebRequest -Uri url -OutFile file.tar` then `tar -xf file.tar -C dir` |
| `python3.11 script.py` | `python script.py` |
| `source venv/bin/activate` | `venv\Scripts\Activate.ps1` |
| `project_id="skills-network"` (free lab auth) | Real credentials from `.env` via `python-dotenv` |
| Gradio 4.19.2 (pinned) | Upgrade to latest — pinned version incompatible with newer Jinja2 |

---

## Key Classes Reference

```python
# LlamaIndex core
from llama_index.core import Document, VectorStoreIndex, PromptTemplate, Settings
from llama_index.core.node_parser import SentenceSplitter

# IBM integrations
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM

# Pipeline
documents = SimpleDirectoryReader(input_files=["file.txt"]).load_data()
nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes, embed_model=embedding_model)
query_engine = index.as_query_engine(llm=llm, text_qa_template=prompt, similarity_top_k=5)
response = query_engine.query("question")
print(response.response)  # string answer
```

---

## Production Notes

**`chunk_size` in LlamaIndex is tokens, not characters.** 512 tokens ≈ 380 words. Most common source of confusion when switching from LangChain where `chunk_size` is in characters.

**`VectorStoreIndex` defaults to in-memory storage.** For production, persist to disk and load on Lambda cold start — same FAISS + S3 pattern from Module 1, just LlamaIndex syntax.

**`verify_embeddings()` uses private internals (`index._storage_context`).** Works for the lab but not production-safe — underscore prefix means it's not part of the public API. In production, validate retrieval quality by running a sample query instead.

**Gradio chat history format changed in newer versions.** Old format: `[[user, bot], ...]`. New format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`. Always check Gradio changelog when upgrading.

**LlamaIndex is the right tool for document Q&A prototypes.** Less boilerplate than LangChain for the standard RAG path. But for anything involving agents, multi-step chains, or tool use — use LangChain.