# ai-langchain-learning

IBM RAG and Agentic AI Professional Certificate — course exercises and labs.

## About

Working through the IBM RAG and Agentic AI Professional Certificate as part of my transition from Engineering Manager to AI Integration Engineer.

## Course Progress

- Course 1: Develop Generative AI Applications: Get Started ✅
- Course 2: Build RAG Applications: Get Started ✅
- Course 3: Vector Databases for RAG: An Introduction (in progress)
- Course 4: Advanced RAG with Vector Databases and Retrievers
- Course 5: Build Multimodal Generative AI Applications
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


## Production Notes

Things that also matter in production:

- LLMs are non-deterministic by default — same prompt, different output.
- JsonOutputParser requires your prompt to explicitly request JSON output — prompt and parser must agree on format. Use `get_format_instructions()` over custom instructions.
- Model versions deprecate. Always check supported models if you hit a `WMLClientError` on `model_id`.
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