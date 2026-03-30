# langchain-learning
IBM RAG and Agentic AI Professional Certificate — course exercises and labs.

## About
Working through the IBM RAG and Agentic AI Professional Certificate as part of
my transition from Engineering Manager to AI Integration Engineer.

## Course Progress
- [ ] Develop Generative AI Applications: Get Started
- [ ] Build RAG Applications: Get Started
- [ ] Vector Databases for RAG: An Introduction
- [ ] Advanced RAG with Vector Databases and Retrievers
- [ ] Build Multimodal Generative AI Applications
- [ ] Fundamentals of Building AI Agents
- [ ] Agentic AI with LangChain and LangGraph
- [ ] Agentic AI with LangGraph, CrewAI, AutoGen and BeeAI
- [ ] Build AI Agents using MCP

## Labs
| Course | Lab | Key Concept |
|--------|-----|-------------|
| Course 1 - Module 1 | Prompt Engineering and LangChain PromptTemplates | LCEL chains, JsonOutputParser, prompting techniques |

## Production Notes
Things that also matter in production:

- LLMs are non-deterministic by default — same prompt, different output.
- JsonOutputParser requires your prompt to explicitly request JSON output —
  prompt and parser must agree on format.
- Model versions deprecate. Always check supported models if you hit a
  WMLClientError on model_id.
- Output quality monitoring is essential — LLM failures don't throw exceptions,
  they silently return bad output.
