# ai-learning

IBM RAG and Agentic AI Professional Certificate — course exercises and labs.

## About

Working through the IBM RAG and Agentic AI Professional Certificate as part of my transition from Engineering Manager to AI Integration Engineer.

## Course Progress

- Course 1: Develop Generative AI Applications: Get Started ✅
- Course 2: Build RAG Applications: Get Started ✅
- Course 3: Vector Databases for RAG: An Introduction ✅
- Course 4: Advanced RAG with Vector Databases and Retrievers ✅
- Course 5: Build Multimodal Generative AI Applications 🔄 (Module 1 in progress)
- Course 6: Fundamentals of Building AI Agents
- Course 7: Agentic AI with LangChain and LangGraph
- Course 8: Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI
- Course 9: Build AI Agents using MCP

## Labs

| Course | Lab | Notes | Key Concepts |
|---|---|---|---|
| Course 1 - Module 1 | Prompt Engineering and LangChain PromptTemplates | [lab1-notes.md](lab1-notes.md) | LCEL chains, JsonOutputParser, prompting techniques |
| Course 1 - Module 2 | Build Smarter AI Apps: Empower LLMs with LangChain | [lab2-notes.md](lab2-notes.md) | Output parsers, RAG retrieval, memory, chains, ReAct agents |
| Course 1 - Module 3 | Build Your First GenAI Application The Right Way | [lab3notes.md](lab3notes.md) | Flask + LangChain, multi-model comparison, Pydantic structured outputs |
| Course 2 - Module 1 | Summarize Private Documents Using RAG, LangChain, and LLMs | [lab4notes.md](lab4notes.md) | RAG pipeline, document splitting, ChromaDB, RetrievalQA, prompt templates, conversational memory |
| Course 2 - Module 2 | Set Up a Simple Gradio Interface to Interact with Your Models | [lab5notes-course2-module2.md](lab5notes-course2-module2.md) | Gradio Interface, common components, wiring Gradio to a Watsonx LLM |
| Course 2 - Module 3 | Build an AI Icebreaker Bot with LlamaIndex & IBM Granite | [lab6notes-course2-module3.md](lab6notes-course2-module3.md) | LlamaIndex RAG pipeline, SentenceSplitter, VectorStoreIndex, QueryEngine, custom PromptTemplate, LlamaIndex IBM integrations |
| Course 3 - Module 1 | Similarity Search by Hand | [lab7notes-c3-m1-similarity-search.md](lab7notes-c3-m1-similarity-search.md) | L2 distance, dot product, cosine similarity, normalisation, similarity search pipeline |
| Course 3 - Module 1 | Similarity Search on Text Using Chroma DB and Python | [lab8notes-c3-m1-similarity-search-chroma.md](lab8notes-c3-m1-similarity-search-chroma.md) | Direct ChromaDB client, HNSW cosine config, multi-query batching, nested results iteration |
| Course 3 - Module 2 | Similarity Search on Employee Records using Python and Chroma DB | [lab9notes-c3-m2-employee-similarity-search.md](lab9notes-c3-m2-employee-similarity-search.md) | Structured data serialisation, dual-layer pattern, combined search, Onion architecture |
| Course 3 - Module 2 | Food Recommendation System Using Chroma DB | [lab10notes-c3-m2-recommend-system-chroma.md](lab10notes-c3-m2-recommend-system-chroma.md) | RAG chatbot, metadata filtering, similarity threshold, two-query pattern, conversation history |
| Course 4 - Module 1 | Build a Smarter Search with LangChain Context Retrieval | [lab11notes-c4-m1-LangChain-Context-Retrieval.md](lab11notes-c4-m1-LangChain-Context-Retrieval.md) | Vector store retriever, MMR, similarity threshold, multi-query retriever, self-querying retriever, parent document retriever |
| Course 4 - Module 1 | Explore Advanced Retrievers in LlamaIndex | [lab12notes-c4-m1-advanced-retrievers-llamaindex.md](lab12notes-c4-m1-advanced-retrievers-llamaindex.md) | VectorIndexRetriever, BM25Retriever, DocumentSummaryIndex (LLM + embedding), AutoMergingRetriever, RecursiveRetriever, QueryFusionRetriever (RRF/relative score/distribution-based), hybrid retrieval, production RAG pipeline with evaluation harness |
| Course 4 - Module 2 | Semantic Similarity with FAISS | [lab13notes-c4-m2-faiss-seman-similarity.md](lab13notes-c4-m2-faiss-seman-similarity.md) | FAISS IndexFlatL2, USE embeddings, manual position mapping, embed→store→search separation |
| Course 4 - Module 2 | AI-Powered YouTube Summariser and QA Tool | [lab14notes-c4-m2-QA-Tool-RAG-LC-FAISS.md](lab14notes-c4-m2-QA-Tool-RAG-LC-FAISS.md) | FAISS via LangChain wrapper, MiniLM embeddings, LCEL chains, gr.State(), Onion architecture, dependency injection, pytest with fixtures |
| Course 5 - Module 1 | Personal Storyteller with Mistral and gTTS | [lab15notes-c5-m1-personal-storyteller.md](lab15notes-c5-m1-personal-storyteller.md) | LLM→TTS pipeline, Ollama local swap, gTTS, notebook→script refactor, pytest mocking patterns |


## Production Notes

Things that also matter in production:

- LLMs are non-deterministic by default — same prompt, different output.
- JsonOutputParser requires your prompt to explicitly request JSON output — prompt and parser must agree on format. Use `get_format_instructions()` over custom instructions.
- Model versions deprecate. Always check supported models if you hit a `WMLClientError` on `model_id`. Same applies to Anthropic — `claude-haiku-20240307` does not exist, use `claude-haiku-4-5`.
- Output quality monitoring is essential — LLM failures don't throw exceptions, they silently return bad output.
- Temperature controls creativity, not verbosity. Low temperature models still hallucinate — they just do it consistently.
- Retrieval quality determines LLM output quality. The model has no way to know if the context it received is wrong.
- Match your wrapper type to your model type — passing chat messages to a completion model causes hallucinated conversation turns.
- Each LLM family requires different special token formatting — `ChatPromptTemplate` handles this automatically. Never manage special tokens manually in production.
- `{format_prompt}` must be injected into the system block of the template, not the user block — otherwise the model ignores JSON formatting instructions.
- `JsonOutputParser` returns a Python dict, not an `AIMessage` — access fields directly, no `.content`.
- Granite produces the cleanest structured output of the IBM-hosted models tested. Prefer it for structured output tasks over Llama or Mistral.
- Silent model selection bugs produce no errors — wrong model answers quietly. Name LLM variables clearly (`granite_llm`, `llama_llm`, `mistral_llm`) and verify model selection in tests.
- `chunk_overlap=0` is a lab shortcut. Use `RecursiveCharacterTextSplitter` with overlap (100-200 chars) in production to preserve context at chunk boundaries.
- Prompt templates are not optional in production — without one the model will hallucinate answers to questions not in the document.
- `return_source_documents=True` should be logged in production even if not shown to the user. It is the primary tool for diagnosing retrieval failures.
- Lambda is stateless — `while True` loops don't work. Session memory requires an external store (DynamoDB with TTL).
- The offline/online split is an infrastructure decision. Index building is a batch job. Query serving is a Lambda function. Never mix them.
- `gr.Interface` is single-turn. For a chatbot with memory use `gr.ChatInterface`. Gradio is for prototyping and internal tooling — not end-user production frontends.

**After LlamaIndex lab (Course 2 - Module 3):**
- LlamaIndex `chunk_size` is in tokens, not characters. 512 tokens ≈ 380 words. LangChain `chunk_size` is in characters. Same parameter name, different units — the most common cross-framework bug.
- LlamaIndex `VectorStoreIndex` defaults to in-memory storage. Persist with `index.storage_context.persist()` and load on cold start for Lambda deployments.
- Use LangChain for agents, multi-step pipelines, and tool use. Use LlamaIndex for pure document Q&A prototypes where you want less boilerplate. Never mix frameworks in the same pipeline.
- Gradio chat history format changed in newer versions — old `[[user, bot]]` list-of-lists format replaced by `[{"role": "user", "content": "..."}]` dict format. Always check Gradio changelog when upgrading.

**After Employee Similarity Search lab (Course 3 - Module 2):**
- Structured data must be serialised to natural language before embedding — embedding models are trained on text, not JSON or key-value pairs. Numeric attributes (ratings, counts, years) belong in metadata for filtering, not in the embedded document.
- `collection.get()` returns flat results, `collection.query()` returns nested results grouped by query. Index with `[i]` vs `[0][i]` respectively — the most common ChromaDB indexing bug.
- `where` on delete has no confirmation and no row count — resolve IDs first with `get()`, inspect, then delete by ID.
- `Protocol` is the Pythonic interface — structural typing, no explicit inheritance required. Prefer over `ABC` for clean layer boundaries in Onion architecture.
- Python has no DI container — wire interfaces to concretes in a composition root (`container.py`). Entry point imports the interface and the factory, never the concrete class directly.

**After Food Recommendation System lab (Course 3 - Module 2):**
- Vector search always returns a result — there is no concept of "no match." Apply a similarity threshold post-retrieval to prevent weak matches reaching the LLM. Threshold value should be calibrated from real query data, not guessed.
- Decouple result count from quality threshold — `n_results` controls how many ChromaDB returns, the threshold controls how many you show. Two independent knobs tuned separately.
- Metadata filters reduce the candidate pool before similarity search runs — overly restrictive filters return the best of a bad pool, not the best overall. Post-filter after a wider search when recall matters more than precision.
- Metadata filter values are exact string match, case sensitive — normalise at ingest time. Use uppercase controlled enum values for category-like fields (`"WOMENS_TICKETS"` not `"womens_tickets"`).
- Raw user input should never go directly into a metadata filter — extract intent first, map to a controlled value, then filter.
- Embedding model and generation model are completely independent — swap the generation model freely, never swap the embedding model without rebuilding the entire index.
- Pin embedding model to a specific named version — floating aliases can change silently and cause retrieval regression with no error thrown. Store the model name and version in collection metadata at index time.
- For Lambda: use an external embedding API (Bedrock Titan or Cloudflare Workers AI) — local SentenceTransformer models exceed the Lambda layer size limit.
- Conversation history tracked in a list is not the same as conversation memory — the list must be passed into the LLM prompt to have any effect. Three previous turns in-memory covers most conversational use cases without requiring DynamoDB.
- Two-query pattern for split UX: run filtered and unfiltered searches in parallel, deduplicate results by ID before display.
- Pydantic models for document schema validation — fail loudly at ingest time on malformed data rather than silently storing bad embeddings that return wrong answers later.

**After LangChain Context Retrieval lab (Course 4 - Module 1):**
- LangChain retriever interface vs direct ChromaDB — direct gives transparency and debuggability, LC interface gives composability and access to the advanced retriever ecosystem. Production almost always wants the interface. Build direct first to understand what's under the hood, then migrate to the interface.
- MMR `lambda_mult` is a tunable dial — 0.5 default, 0.7 for focused FAQ domains (lean relevance), lower for broad exploratory search (lean diversity). Calibrate from real query data not guesswork.
- `similarity_score_threshold` is the native LC implementation of the manual quality gate built in SRMC-995. Same concept, standardised interface.
- Multi-query retriever changes the text going into the query embedding, not the retrieval mechanism. Same cosine similarity, different starting vectors. The LLM generates reformulations — it does not see the index or documents.
- Multi-query is query expansion. EnsembleRetriever is method combination. Different tools solving different problems.
- Self-query retriever depends on the LLM generating valid, correctly typed filter expressions — not guaranteed. Requires retry logic and fallback to unfiltered search in production. Numeric type coercion is a known fragility with older LangChain versions and newer Claude models.
- `lark` is a required dependency for self-query retriever — not in the standard LangChain install. Add to requirements explicitly.
- Parent document retriever uses two stores — Chroma vector store for child chunk embeddings, InMemoryStore docstore for parent documents. `add_documents()` populates both simultaneously. Do not use `Chroma.from_documents()` for this pattern.
- LongContextReorder always comes after retrieval, never before. Reorder what you have, not what you're about to fetch.
- Load dotenv before llm() is called — the LLM instance captures the API key at construction time. Stale instances hold no key even after dotenv loads.
- `| tail -n 1` is Linux only. Remove from all pip install commands when running on Windows.
- `wget` is Linux only. Use `urllib.request.urlretrieve()` for file downloads on Windows.

**After LlamaIndex Advanced Retrievers lab (Course 4 - Module 1):**
- LlamaIndex makes the retriever/query engine separation explicit — `index.as_retriever()` returns nodes, `RetrieverQueryEngine(retriever=...)` adds synthesis. LangChain's `RetrievalQA` bundles both. Understanding the separation makes custom pipelines easier to reason about.
- BM25 is 30 years old (1994) — standard in Elasticsearch, Lucene, Solr. Deterministic, no embedding calls, no API cost. Use as a pre-filter before vector search on large corpora to reduce the candidate set before the embedding call runs.
- BM25 scores are unbounded positive floats, cosine similarity scores are 0-1. Cannot combine directly — normalise each list independently before fusion.
- When combining retrievers with weighted average fusion, match by text content not node ID — the same document has different node IDs in different retrievers.
- RRF (Reciprocal Rank Fusion) is the default fusion strategy — position-based, scale-invariant, no score calibration required. Relative Score and Distribution-Based require trust in score calibration. RRF is robust when you don't know how well-calibrated your embedding model is.
- `DocumentSummaryIndex` calls the LLM at index build time to generate summaries — budget this as an offline cost, not a per-query cost. Use the embedding-based retriever variant in production (deterministic, cheaper per query). Use the LLM-based variant as a development diagnostic tool only.
- `AutoMergingRetriever` needs two objects: `base_retriever` (does the vector search) and `storage_context` (navigates the parent/child hierarchy). Neither works without the other in this pattern.
- `RecursiveRetriever` requires a meaningful reference graph at index time. Synthetic positional references (doc_0 references doc_1) demonstrate the mechanism but not the value. Real value requires real citations, links, or intentional cross-references in metadata.
- `QueryFusionRetriever` combines multi-retriever fusion AND multi-query generation in one class. LangChain keeps these as separate classes (`EnsembleRetriever` and `MultiQueryRetriever`). LlamaIndex's fusion strategy options (RRF, relative score, distribution-based) are more sophisticated than LangChain's weighted average.
- The cost architecture hierarchy: BM25 (free) → embedding call (cheap) → small LLM for reformulation/filtering (cheap) → large LLM for synthesis (expensive). Never run an expensive operation on a candidate set a cheaper operation could have filtered first.


**After YouTube Summariser and QA Tool lab (Course 4 - Module 2):**
- FAISS via LangChain wrapper (`FAISS.from_texts()`) returns documents directly — no manual position mapping required. Raw FAISS returns indices; the LangChain wrapper abstracts the lookup.
- `LLMChain` is removed in current LangChain — use LCEL (`prompt | llm`) with `.invoke()`. `.run()` and `.predict()` are also removed. `.content` extracts the string from the returned `AIMessage`.
- Global state in multi-threaded Gradio is unreliable — use `gr.State()` for explicit state passing between button clicks. Each click runs in a separate thread; globals written in one thread may not be visible in another.
- Singletons initialised at module import time are cached by Python — `llm` and `embedding_model` constructed once in `config.py` are reused on every subsequent import. No re-initialisation per request.
- Index building is a one-time offline cost — build the FAISS index when the document is loaded, not on every query. Store in `gr.State()` and reuse.
- Summarise path and Q&A path are architecturally distinct — summarisation sends the full transcript directly to the LLM (no retrieval). Q&A requires chunking, embedding, and FAISS retrieval first. Never conflate the two.
- `sys.modules["config"] = Mock()` before importing the entry point prevents singleton initialisation during tests — essential when `config.py` makes API calls or loads models at import time.
- pytest fixtures replace repeated Mock setup — define shared dependencies once in `@pytest.fixture`, inject by parameter name. Same pattern as `[SetUp]` in NUnit.

**After Personal Storyteller lab (Course 5 - Module 1):**
- `ibm_watsonx_ai.ModelInference.generate_text()` → `langchain_ollama.OllamaLLM.invoke()` for local Ollama swap. No `prompt=` keyword — `invoke()` takes the string positionally.
- IBM's `DECODING_METHOD: "greedy"` ≡ `temperature=0.0` in Ollama. Same algorithm, different API surface. Use greedy when reproduciblity matters (labs, tests); use temperature > 0 for creative variation.
- `gTTS(text)` is inert — constructor stores config, no network call. Network hits on `tts.write_to_fp()` or `tts.save()`. Calling both on the same object makes two separate round-trips to Google — save to file once, reuse the file.
- gTTS is not production-ready — undocumented Google Translate endpoint, no SLA, no auth, IP-based rate limiting. For production TTS use ElevenLabs / Azure Speech / AWS Polly.
- `python -m pytest` over bare `pytest` — uses whichever Python is active on PATH, avoiding silent system-Python invocation when venv is not activated.
- Empty `conftest.py` is unreliable on modern pytest (7+) — always add the explicit `sys.path.insert(0, str(Path(__file__).parent))` for test discovery across subdirectories.
