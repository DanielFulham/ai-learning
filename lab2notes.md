# Course 1 — Lab 2: Build Smarter AI Apps: Empower LLMs with LangChain

## Lab Overview

**Course:** IBM RAG and Agentic AI Professional Certificate — Course 1  
**Lab:** Build Smarter AI Apps: Empower LLMs with LangChain  
**Completed:** 31 March 2026  

7 exercises covering the full LangChain component stack: models, prompts, output parsers, documents, retrieval, memory, chains, and agents.

---

## Key Concepts

### Models and Chat Models

`WatsonxLLM` wraps an IBM `ModelInference` object to make it compatible with LangChain. Two distinct types:

- **Completion model** (`WatsonxLLM`) — takes a plain string, completes it. Stateless.
- **Chat model** — takes a list of typed messages (`SystemMessage`, `HumanMessage`, `AIMessage`), understands roles natively.

**Critical lesson:** passing a list of messages to a completion model causes it to serialise them into a flat string and continue the pattern — it will hallucinate additional conversation turns until it hits `MAX_NEW_TOKENS`. Always match wrapper type to model type.

```python
# Completion model — takes a string
llama_llm.invoke("Who is man's best friend?")

# Chat model — takes a list of typed messages
llama_llm.invoke([
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capital of France?")
])
```

---

### ChatPromptTemplate and MessagesPlaceholder

`ChatPromptTemplate` structures input before it reaches the LLM. Keeps system instructions and user messages as distinct roles.

`MessagesPlaceholder` accepts a list of message objects — useful for injecting conversation history into a template.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])
input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}
prompt.invoke(input_)
# Returns ChatPromptValue with structured roles — not a flat string
```

**The flow:**  
Input dict → `ChatPromptTemplate` fills placeholders → `ChatPromptValue` (structured) → LLM → response

---

### Output Parsers

Output parsers control the shape of what comes back from the LLM.

| Parser | Output | Use case |
|---|---|---|
| `StrOutputParser` | Plain string | Most chains |
| `JsonOutputParser` | Python dict | Structured data |
| `CommaSeparatedListOutputParser` | Python list | Simple lists (fragile) |

**Key lesson — parsers are strict:**  
`JsonOutputParser` fails if the LLM adds any text before or after the JSON. One word of preamble like "Here is the JSON:" causes a `JSONDecodeError`. The fix is to strengthen the prompt instruction and use `json_parser.get_format_instructions()` rather than writing custom format instructions.

```python
# Correct pattern
json_parser = JsonOutputParser(pydantic_object=Movie)
format_instructions = json_parser.get_format_instructions()  # use built-in, not custom

prompt = PromptTemplate(
    template="Return ONLY a JSON object about {movie_name}. No other text.\n{format_instructions}\n",
    input_variables=["movie_name"],
    partial_variables={"format_instructions": format_instructions},
)
chain = prompt | llama_llm | json_parser
```

**`CommaSeparatedListOutputParser` is fragile** — chatty models add reasoning steps that get caught in the comma split. Don't rely on it in production. Use `JsonOutputParser` with a typed Pydantic schema instead.

---

### Documents, Loaders, and Text Splitters

LangChain `Document` objects have two fields: `page_content` (text) and `metadata` (source, page number, etc.).

**Loaders:**
- `PyPDFLoader` — loads PDF files, one `Document` per page
- `WebBaseLoader` — scrapes a URL into a `Document`

**Text Splitters:**

| Splitter | Separator strategy | Use case |
|---|---|---|
| `CharacterTextSplitter` | Single separator (e.g. `\n`) | Simple, predictable content |
| `RecursiveCharacterTextSplitter` | Tries `\n\n` → `\n` → ` ` → `""` in order | Production default |

`RecursiveCharacterTextSplitter` does NOT accept a `separator` argument — it manages its own list internally.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
# Metadata (source, page) is preserved on every chunk
```

**chunk_overlap** repeats the last N characters of chunk N at the start of chunk N+1 — preserves context at boundaries.

---

### Embeddings and Vector Stores

Text chunks are converted to vectors (lists of numbers) by an embedding model. Similar meaning = similar vectors = close in vector space.

```python
embedding_model = WatsonxEmbeddings(model_id="ibm/slate-125m-english-rtrvr-v2", ...)
# Chroma stores chunks + their vectors in one step
vector_store = Chroma.from_documents(chunks, embedding_model)
```

The raw vector values (`[-0.011, 0.017, ...]`) are meaningless to read. They only matter to the similarity search algorithm.

---

### Retrievers

A retriever takes a string query, embeds it, and returns the most semantically similar document chunks.

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("What is LangChain?")
```

**Retrieval quality lesson:** poor results don't throw errors — they silently return irrelevant chunks. The LLM then confidently answers using wrong context. Retrieval quality is the AI Integration Engineer's responsibility, not the model's.

**ParentDocumentRetriever** — stores small child chunks for precise embedding search, but returns larger parent chunks for full context on retrieval. Best for large document sets where both precision and context matter.

---

### Memory

LLMs are stateless — every API call is independent. Memory is the mechanism for faking statefulness by passing conversation history back on every call.

**ConversationBufferMemory** — stores full verbatim transcript. Accurate but grows indefinitely.

**ConversationSummaryMemory** — periodically summarises older turns into a compressed paragraph. Token-efficient for long conversations but introduces drift over time.

```python
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llama_llm, memory=memory, verbose=True)
conversation.invoke(input="My favourite colour is blue.")
# Memory buffer now contains this turn — passed back on every subsequent call
```

**Production constraint:** context window limits mean very long conversations silently drop the oldest messages. Strategies: buffer window (keep last N turns), summary memory, or database-backed memory.

---

### Chains — Legacy vs LCEL

**Legacy (SequentialChain):** configure chains with explicit `input_variables` and `output_variables`. Verbose, less readable.

```python
overall_chain = SequentialChain(
    chains=[sentiment_chain, summary_chain, response_chain],
    input_variables=["review"],
    output_variables=["sentiment", "summary", "response"],
    verbose=True
)
```

**LCEL (modern):** pipe operator `|` composes steps. `RunnablePassthrough.assign()` adds new keys to the dict without removing existing ones — this is how intermediate outputs flow through the chain.

```python
lcel_chain = (
    RunnablePassthrough.assign(sentiment=lambda x: sentiment_chain_lcel.invoke({"review": x["review"]}))
    | RunnablePassthrough.assign(summary=lambda x: summary_chain_lcel.invoke({"review": x["review"], "sentiment": x["sentiment"]}))
    | RunnablePassthrough.assign(response=lambda x: response_chain_lcel.invoke({"review": x["review"], "sentiment": x["sentiment"], "summary": x["summary"]}))
)
```

**Key lesson — garbage in, garbage out:**  
Each chain in a sequence receives the output of the previous one as input. If step 1 returns a rambling paragraph instead of a clean sentiment label, step 2 gets that noise as its `{sentiment}` variable. Output parsers between steps are the production fix.

**LCEL produced better results** than the legacy chain in testing — more explicit variable passing gives the model cleaner context at each step.

---

### Agents — ReAct Pattern

An agent uses an LLM to decide which tool to call, calls it, observes the result, and loops until it has a final answer. The LLM doesn't execute tools directly — it outputs structured text that the agent runtime parses and executes.

**ReAct pattern** (Reason + Act) — the prompt template enforces this reasoning loop:

```
Question: the user's question
Thought: think about what to do
Action: the tool to use
Action Input: the input to the tool
Observation: the result from the tool
Thought: I now know the final answer
Final Answer: the answer
```

```python
tools = [
    Tool(name="calculator", func=calculator, description="For maths calculations"),
    Tool(name="format_text", func=format_text, description="For formatting text")
]

agent = create_react_agent(llm=llama_llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

result = agent_executor.invoke({"input": "What is 25 + 63?"})
```

**Tool descriptions matter** — the LLM reads descriptions to decide which tool fits the query. Vague descriptions = wrong tool selection.

**`handle_parsing_errors=True`** — prevents the agent crashing when the LLM produces malformed output. Essential in production.

---

## Production Insights

**Prompt engineering is model-specific.** What works on GPT-4 won't always work on Llama. Custom format instructions that look clear to a human can cause a chatty model to reproduce them verbatim in its output.

**Temperature controls creativity, not verbosity.** A `temperature: 0.1` model will confidently hallucinate extra conversation turns — it just does it consistently. Low temperature ≠ well-behaved output.

**Retrieval quality determines LLM output quality.** The model has no way to know if the context it received is wrong. If retrieval is bad, answers are confidently wrong. Retrieval evaluation is as important as model selection.

**No guardrails in demo code.** Lab exercises show happy path only — no output validation between chain steps, no retry logic, no error handling beyond `handle_parsing_errors`. Production chains need all of these.

**Legacy vs LCEL:** `SequentialChain` still exists in production codebases. Recognise it, understand it, prefer LCEL for new code.

---

## Exercises Completed

| Exercise | Topic | Notes |
|---|---|---|
| 1 | Model comparison (temperature + model) | Llama 4 cleaner output than Granite for structured tasks |
| 2 | JSON Output Parser with Pydantic schema | Built `Movie` class — `get_format_instructions()` > custom instructions |
| 3 | Document Loaders + Text Splitters | PDF + Web, CharacterTextSplitter vs RecursiveCharacterTextSplitter |
| 4 | Simple Retrieval System (RAG) | ChromaDB + WatsonxEmbeddings + ParentDocumentRetriever |
| 5 | Chatbot with Memory | ConversationBufferMemory vs ConversationSummaryMemory |
| 6 | Multi-step chains (Legacy + LCEL) | Product review sentiment → summary → response |
| 7 | ReAct Agent with tools | Calculator + text formatter tools, full ReAct loop |

---

## Key Classes Reference

```python
# Models
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

# Messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Prompts
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

# Output Parsers
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser

# Documents
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Store + Embeddings
from langchain.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings

# Retrieval
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ChatMessageHistory

# Chains
from langchain.chains import LLMChain, SequentialChain, ConversationChain
from langchain_core.runnables import RunnablePassthrough

# Agents
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
```

---

## Football Club Connection (real use cases)

Concepts from this lab map directly to the Football club AI chatbot V1.0 plan:

- **ChatPromptTemplate + system prompt** → restricts chatbot to team-only topics
- **ConversationBufferMemory** → short session memory (10 turn limit) for match-day chatbot
- **Document Loaders + Text Splitters + ChromaDB** → V2.0 RAG over club FAQ documents
- **ReAct Agent** → future agentic version that queries ticketing api, faq etc.
- **Output parsers** → structured fan data extraction for CDP integration
