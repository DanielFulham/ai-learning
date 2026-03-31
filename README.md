# langchain-learning
IBM RAG and Agentic AI Professional Certificate — course exercises and labs.

## About
Working through the IBM RAG and Agentic AI Professional Certificate as part of
my transition from Engineering Manager to AI Integration Engineer.

## Course Progress
- [x] Course 1: Develop Generative AI Applications: Get Started (in progress)
- [ ] Course 2: Build RAG Applications: Get Started
- [ ] Course 3: Vector Databases for RAG: An Introduction
- [ ] Course 4: Advanced RAG with Vector Databases and Retrievers
- [ ] Course 5: Build Multimodal Generative AI Applications
- [ ] Course 6: Fundamentals of Building AI Agents
- [ ] Course 7: Agentic AI with LangChain and LangGraph
- [ ] Course 8: Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI
- [ ] Course 9: Build AI Agents using MCP

## Labs
| Course | Lab | Notes | Key Concepts |
|--------|-----|-------|--------------|
| Course 1 - Module 1 | Prompt Engineering and LangChain PromptTemplates | [lab1-notes.md](course-1-genai-fundamentals/lab1-notes.md) | LCEL chains, JsonOutputParser, prompting techniques |
| Course 1 - Module 2 | Build Smarter AI Apps: Empower LLMs with LangChain | [lab2-notes.md](course-1-genai-fundamentals/lab2-notes.md) | Output parsers, RAG retrieval, memory, chains, ReAct agents |


## Production Notes
Things that also matter in production:

- LLMs are non-deterministic by default — same prompt, different output.
- `JsonOutputParser` requires your prompt to explicitly request JSON output —
  prompt and parser must agree on format. Use `get_format_instructions()` over custom instructions.
- Model versions deprecate. Always check supported models if you hit a
  `WMLClientError` on model_id.
- Output quality monitoring is essential — LLM failures don't throw exceptions,
  they silently return bad output.
- Temperature controls creativity, not verbosity. Low temperature models still hallucinate — they just do it consistently.
- Retrieval quality determines LLM output quality. The model has no way to know if the context it received is wrong.
- Match your wrapper type to your model type — passing chat messages to a completion model causes hallucinated conversation turns.
