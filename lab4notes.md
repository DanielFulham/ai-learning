# Course 2 — Lab 4: Summarize Private Documents Using RAG, LangChain, and LLMs

## Lab Overview

**Course:** IBM RAG and Agentic AI Professional Certificate — Course 2  
**Lab:** Summarize Private Documents Using RAG, LangChain, and LLMs  
**Completed:** 4 April 2026  
**Time taken:** ~3 hours

Full RAG pipeline from document ingestion to conversational retrieval with memory. Covers the complete offline indexing phase (load → split → embed → store) and online retrieval phase (embed query → retrieve → augment → generate).

---

## Key Concepts

### The Offline / Online Split

The most important architectural insight from this lab. The RAG pipeline has two distinct phases:

**Offline (build time — happens once or on a schedule):**
1. Load source documents
2. Split into chunks
3. Embed chunks into vectors
4. Store in vector database (ChromaDB)

**Online (runtime — happens per user request):**
5. Embed the user query
6. Retrieve relevant chunks via similarity search
7. Augment prompt with retrieved chunks
8. Generate response via LLM

The vector index is pre-built. No model is called on page load — only when a user asks a question. This is the correct production architecture.

---

### Document Loading and Splitting

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))  # 16 chunks for the company policies doc
```

**`CharacterTextSplitter`** uses `\n\n` as default separator and splits at `chunk_size` characters. Simple but can split mid-sentence.

**Production note:** Use `RecursiveCharacterTextSplitter` with overlap (100-200 chars) to avoid losing context at chunk boundaries. `chunk_overlap=0` is intentionally simple for this lab.

```python
# Production version
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
```

---

### Embedding and Vector Storage

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings()  # defaults to sentence-transformers/all-mpnet-base-v2
docsearch = Chroma.from_documents(texts, embeddings)
print('document ingested')
```

**`HuggingFaceEmbeddings()` with no arguments** downloads `sentence-transformers/all-mpnet-base-v2` (~438MB) and runs it **locally**. First run downloads the model to `~/.cache/huggingface/`, subsequent runs load from cache.

**Critical:** The same embedding model must be used for both document indexing and query embedding. Mismatching models produces nonsensical similarity scores.

**Production note (Lambda):** A 438MB local embedding model is too heavy for Lambda (250MB layer limit). Use Amazon Bedrock embeddings API instead — Lambda calls a Bedrock endpoint, no local model download required.

---

### Model Construction

```python
# granite-3-3-8b-instruct requires deploy-on-demand (not available in free lab environment)
# granite-4-h-small is in the "provided with watsonx.ai" tier — works without API keys
model_id = 'ibm/granite-4-h-small'

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 130,
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5
}

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)

flan_ul2_llm = WatsonxLLM(model=model)
```

**Name LLM variables clearly.** When swapping models across cells, use consistent names (`granite_llm`, `llama_llm`, `mistral_llm`) to prevent silent bugs where the wrong model answers queries.

---

### RetrievalQA — Basic Retrieval Chain

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=flan_ul2_llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=False
)
query = "what is mobile policy?"
qa.invoke(query)
```

**`chain_type="stuff"`** — retrieved chunks are stuffed directly into the prompt as-is. Simplest strategy, correct for Q&A use cases.

**Other chain types:**
- `map_reduce` — processes each chunk separately then combines results. For summarisation over large documents. Hit token limit errors in this lab (`1998 > 1024`) even on a small document.
- `refine` — iteratively refines the answer across chunks. More token-efficient than map_reduce.

**`docsearch.as_retriever()`** defaults to top-K=4. Tune in production:
```python
docsearch.as_retriever(search_kwargs={"k": 3})
```

---

### Prompt Template — Controlling Hallucination

Without a prompt template the model fabricates answers to questions not in the document. With one it correctly declines.

```python
from langchain.prompts import PromptTemplate

prompt_template = """Use the information from the document to answer the question at the end. \
If you don't know the answer, just say that you don't know, definitely do not try to make up an answer.

{context}

Question: {question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=llama_3_llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
    return_source_documents=False
)

query = "Can I eat in company vehicles?"
qa.invoke(query)
# Result: "The document does not mention eating in company vehicles, it only mentions smoking.
#          Therefore, the answer is: I don't know."
```

**`{context}` and `{question}` are reserved keywords** in `RetrievalQA` — LangChain automatically injects retrieved chunks and the user query into these placeholders.

**Rovers chatbot prompt template:**
```python
prompt_template = """You are a helpful assistant for Shamrock Rovers FC.
Use only the information provided to answer questions about matchday experience
at Tallaght Stadium. If you don't have the answer, say:
"I don't have that information — please contact the club at info@shamrockrovers.ie"

{context}

Question: {question}
"""
```

---

### Conversational Memory

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)

qa = ConversationalRetrievalChain.from_llm(
    llm=llama_3_llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    memory=memory,
    get_chat_history=lambda h: h,
    return_source_documents=False
)

history = []

# Turn 1
query = "What is mobile policy?"
result = qa.invoke({"question": query}, {"chat_history": history})
print(result["answer"])
history.append((query, result["answer"]))

# Turn 2 — references previous turn implicitly via "it"
query = "List points in it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])
history.append((query, result["answer"]))

# Turn 3
query = "What is the aim of it?"
result = qa({"question": query}, {"chat_history": history})
print(result["answer"])
```

**Memory in production vs notebook:**

The `while True` loop in the lab is a notebook convenience only. Lambda is stateless — every invocation is independent, history stored as a Python list disappears when the function finishes.

**Production pattern — DynamoDB-backed session memory:**

```python
from langchain.memory import DynamoDBChatMessageHistory

def handler(event, context):
    session_id = event["session_id"]  # GUID generated by frontend on session start

    history = DynamoDBChatMessageHistory(
        table_name="rovers-chat-history",
        session_id=session_id
    )
    # history persists across Lambda invocations for the same session
```

**Session ID pattern:** Frontend generates a UUID on page load (`crypto.randomUUID()`), passes it with every request. DynamoDB stores history keyed by session ID with TTL (e.g. 2 hours) for automatic cleanup.

**DynamoDB cost at Rovers scale:** ~$0.25 per million reads/writes (on-demand). 500 conversations per matchday = negligible. The LLM API call is the real cost to monitor.

---

### Agent Loop

```python
def qa():
    memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llama_3_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
        return_source_documents=False
    )
    history = []
    while True:
        query = input("Question: ")
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break
        result = qa({"question": query}, {"chat_history": history})
        history.append((query, result["answer"]))
        print("Answer: ", result["answer"])

qa()
```

**Note:** Function name `qa()` shadows the chain variable `qa` — sloppy but functional in notebook context. Never do this in production code.

---

## Bugs Hit & Fixed

| Bug | Cause | Fix |
|---|---|---|
| `granite-3-3-8b-instruct` not available in lab | Deploy-on-demand model, requires API key | Swapped to `ibm/granite-4-h-small` (free tier) |
| `Token indices sequence length is longer than specified maximum (1998 > 1024)` | `map_reduce` sends all chunks through model, exceeded context limit | Reverted to `chain_type="stuff"` |
| `map_reduce` returned `"I don't know"` | Token overflow caused malformed combine step, model correctly applied "I don't know" guardrail | Expected behaviour given overflow — not a guardrail bug |
| Exercise 3: Mistral constructed but Llama answered | `mistral_llm` variable created but `llama_3_llm` passed to `RetrievalQA` — silent bug, no error thrown | Named new model `mistral_llm` and used it consistently |
| Exercise 1 lab hint skips re-indexing steps | Lab only shows `wget.download()` for new document | Manually added load → split → embed → rebuild index before querying |

---

## Exercises

### Exercise 1: Load and Index a New Document

**Key insight the lab skipped:** switching documents requires rebuilding the full index. Downloading the file alone is not enough.

```python
# 1. Download
filename = 'speech.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/XVnuuEg94sAE4S_xAsGxBA.txt'
wget.download(url, out=filename)
print('file downloaded')

# 2. Load and split — REQUIRED, lab hint omits this
loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(len(texts))

# 3. Re-embed and rebuild index — REQUIRED
embeddings = HuggingFaceEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
print('document ingested')
```

**Rovers relevance:** When the fan guide is updated or email Q&As are added, the index must be rebuilt. This is an offline batch process separate from the query API.

### Exercise 2: Return Source Documents

```python
qa = RetrievalQA.from_chain_type(
    llm=llama_3_llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True  # key change
)
query = "Can you summarize this speech in three sentences?"
results = qa.invoke(query)

print(results['result'])               # the answer
print(results['source_documents'][0])  # the chunk that was retrieved
```

**Production value of `return_source_documents=True`:**
- **Debuggability** — when the chatbot gives a wrong answer, inspect the retrieved chunk to diagnose whether retrieval or generation failed
- **Citation** — surface source to the user to build trust: "Based on our fan guide..."
- **Quality monitoring** — log retrieved chunks per invocation to tune retrieval quality over time

### Exercise 3: Swap the LLM Model

```python
model_id = 'mistralai/mistral-small-3-1-24b-instruct-2503'
model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
mistral_llm = WatsonxLLM(model=model)  # named clearly, not reusing existing variable

qa = RetrievalQA.from_chain_type(
    llm=mistral_llm,  # use the correct variable
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=False
)
query = "Can you summarize this speech in three sentences?"
results = qa.invoke(query)
print(results['result'])
```

---

## Key Classes Reference

```python
# Document loading
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Embeddings and vector store
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Retrieval chains
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

# Prompts
from langchain.prompts import PromptTemplate

# Memory
from langchain.memory import ConversationBufferMemory

# WatsonX
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
```

**Note:** These imports use `langchain==0.1.16` paths (lab pinned version). Current LangChain moves these to `langchain_community`:
```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
```

---

## Production Notes

**Offline vs online is an infrastructure decision.** Steps 1-3 (index building) run separately from steps 4-8 (query serving). Don't mix them. The index build is a batch job; the query handler is a Lambda function.

**`chunk_overlap=0` is a lab shortcut.** Production RAG needs overlap to preserve context at chunk boundaries. `RecursiveCharacterTextSplitter` with `chunk_overlap=150` is the production baseline.

**`chain_type="stuff"` is correct for FAQ-style Q&A.** `map_reduce` is for full-document summarisation but requires careful token budget management — hit token limit errors in this lab even with a small document and a model with only 1024 token context.

**Prompt templates are not optional in production.** Without one the model will hallucinate answers to questions not in the document. The "I don't know" instruction is a guardrail, not a nicety.

**`return_source_documents=True` should be logged in production** even if not shown to the user. It is the primary tool for diagnosing retrieval failures and tuning chunk quality over time.

**Lambda is stateless — `while True` loops don't work.** Session memory requires an external store. DynamoDB with session-keyed history and TTL is the correct pattern. Cost is negligible at small scale ($0.25/million requests).

**The embedding model and LLM are separate components with separate failure modes.** A great LLM with a poor embedding model still produces bad answers. Retrieval quality determines output quality.

**Silent model selection bugs are real.** Constructing a new model object but passing an old variable name to the chain produces no error — wrong model answers silently. Unit test model selection in production.

---

## Rovers Connection

This lab is the direct blueprint for the Rovers FAQ chatbot V1:

| Lab component | Rovers equivalent |
|---|---|
| `companyPolicies.txt` | Fan guide + cleaned email Q&As |
| `HuggingFaceEmbeddings()` | Amazon Bedrock embeddings (lightweight on Lambda) |
| `ChromaDB` | FAISS index stored in S3 (pre-built offline) |
| `RetrievalQA` | Lambda RAG chain |
| IBM WatsonX endpoint | Amazon Bedrock (Claude Haiku or Titan) |
| Jupyter agent loop | API Gateway → Lambda → JS widget on shamrockrovers.ie |
| In-memory history list | DynamoDB session history keyed by GUID (V2) |

**V1 is stateless** — no memory, no DynamoDB. Each supporter question is independent. Most matchday FAQ questions are standalone and don't need conversation history.

**V2** adds DynamoDB-backed session memory with GUID session IDs and 2-hour TTL.

**V3** adds scheduled index rebuild when fan guide or email Q&As are updated.
