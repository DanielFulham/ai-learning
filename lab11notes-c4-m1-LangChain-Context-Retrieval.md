# Lab 11 Notes — Build a Smarter Search with LangChain Context Retrieval
## Course 4, Module 1 Lab 1

---

## What This Lab Covers

Four types of LangChain retrievers applied to two datasets — a company policies
text file and a LangChain academic paper PDF. Covers the standard vector
store-backed retriever, multi-query retriever, self-querying retriever, and
parent document retriever. Each retriever is demonstrated with the same query
infrastructure to make the differences explicit.

---

## Local Environment Setup

This lab was run locally on Windows with VS Code + Jupyter notebook kernel,
not in the IBM Skills Network browser environment.

**Key Windows adaptations required:**

- `| tail -n 1` — Linux command, does not exist on Windows. Remove from all
  `!pip install` lines. Packages install cleanly without it, output is just
  more verbose.
- `wget` — Linux command for file download. Replace with Python urllib:
  ```python
  import urllib.request
  url = "https://..."
  urllib.request.urlretrieve(url, "companypolicies.txt")
  ```
- Pinned versions (`langchain==0.2.1` etc.) are built for Python 3.11 on
  Linux. They install on Windows but cause dependency conflicts when adding
  additional packages like `langchain-anthropic`.
- `sentence-transformers` not included in the lab's install list — required
  for `HuggingFaceEmbeddings`. Install separately:
  ```python
  !pip install sentence-transformers
  ```
- `ipykernel` required for VS Code notebook kernel registration:
  ```python
  pip install ipykernel
  ```

**IBM credentials exhausted — swapped to local/Anthropic stack:**

```python
# Original IBM embedding — replaced
# from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
# from langchain_ibm import WatsonxEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings

def watsonx_embedding():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
```

```python
# Original IBM LLM — replaced
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

def llm():
    return ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0.5,
        max_tokens=256
    )
```

**Critical lesson — load dotenv before llm() is called:**
The `llm()` function captures the API key at construction time. If
`load_dotenv()` hasn't run when the retriever is built, the retriever
holds a stale LLM instance with no API key. Re-run `load_dotenv()`,
re-run `llm()` definition, re-run retriever construction — in that order.

**Model ID lesson:**
`claude-haiku-20240307` does not exist. Use `claude-haiku-4-5` for the
cheapest current Claude model. Model IDs change — always verify against
current Anthropic model list when hitting NotFoundError.

---

## The Four Retrievers

### 1. Vector Store-Backed Retriever

The simplest retriever. Queries an existing vector store and returns the most
similar chunks by cosine similarity.

```python
vectordb = Chroma.from_documents(chunks_txt, watsonx_embedding())
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
docs = retriever.invoke(query)
```

**Three search modes available:**

```python
# Standard similarity search — returns top k regardless of quality
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# MMR — balances relevance with diversity
retriever = vectordb.as_retriever(search_type="mmr")

# Similarity score threshold — only returns results above score floor
retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.4}
)
```

**MMR observation:**
Query "email policy" with pure similarity returned three near-duplicate
Internet and Email Policy chunks. MMR returned one email policy chunk and
three chunks from different policy sections — same k=4, but diversity
penalised redundant selections. For broad queries MMR is better. For narrow
specific queries similarity search is better.

**Threshold observation:**
Higher threshold = stricter gate = fewer but higher quality results. Same
concept as the manually implemented `QUALITY_THRESHOLD` in SRMC-995's
`retriever.py`. LangChain's `similarity_score_threshold` is the native
implementation of that same pattern.

---

### 2. Multi-Query Retriever

Uses an LLM to generate multiple alternative phrasings of the original query,
runs each phrasing through the base retriever independently, then returns the
unique union of all results.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm()
)
docs = retriever.invoke(query)
```

**What it does under the hood:**
```
Original query: "What is LangChain?"
  → LLM generates 3-5 reformulations:
      "LangChain framework overview"
      "LangChain capabilities and features"
      "How does LangChain work"
      "LangChain use cases"
  → Each reformulation embedded independently → different vector
  → Each vector runs cosine similarity against same index
  → Results combined → duplicates removed → unique union returned
```

The retrieval mechanism never changes — always cosine similarity. Only the
text going into the embedding changes. Same torch, four angles.

**Result:** 15 documents returned vs 4 from standard retriever. Broader
recall, but 3 of 15 were noise (figure captions, reference list entries)
that should never have been indexed. Ingestion quality problem, not a
retrieval problem.

**Cost:** One additional LLM call per query for reformulation generation.
The LLM is doing a mechanical task — rephrase this question four ways. A
small cheap model handles this perfectly well. Production pattern: cheap
model for reformulation, expensive model for generation only.

**When to use:**
- Broad questions where the answer is distributed across a document
- Domains with vocabulary mismatch between user queries and indexed content
- Fan segmentation queries where "fans who drifted away" doesn't match
  profile language like "lapsed membership, reduced attendance"

**When not to use:**
- Narrow specific queries ("what time do gates open?") — single query finds
  the answer, multiple queries add cost for no gain
- Well-structured indexes with consistent vocabulary

---

### 3. Self-Querying Retriever

Decomposes a natural language query into two components: a semantic search
string and a structured metadata filter. The LLM generates both from a
description of your metadata schema.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(name="year", description="The year the movie was released", type="integer"),
    AttributeInfo(name="rating", description="IMDB rating of the movie", type="float"),
    AttributeInfo(name="director", description="Director of the movie", type="string"),
]

document_content_description = "Brief summary of a movie"

retriever = SelfQueryRetriever.from_llm(
    llm(),
    vectordb,
    document_content_description,
    metadata_field_info,
)

retriever.invoke("I want to watch a movie directed by Christopher Nolan")
```

**What happens under the hood:**
```
Query → LLM call with metadata schema description
     → LLM generates structured query:
         {"query": "movie to watch", "filter": "eq(\"director\", \"Christopher Nolan\")"}
     → lark parses filter expression
     → translated to ChromaDB native filter: where={"director": "Christopher Nolan"}
     → semantic search runs only against filtered subset
     → results returned
```

**lark dependency:**
`lark` is a parsing library required specifically by self-query retriever to
parse the filter expressions the LLM generates. Not in the standard LangChain
install. Add to requirements:
```python
!pip install lark
```

**Production fragility — numeric type coercion bug:**
With `langchain==0.2.1` and `claude-haiku-4-5`, numeric filter values are
returned as strings rather than native Python types. ChromaDB rejects string
operands for `$gt` and `$lt` operators:

```
ValueError: Expected operand value to be an int or a float for operator $gt, got 8.5
```

This is a known version compatibility issue between older LangChain pinned
versions and newer Claude models. String equality filters work correctly —
only numeric comparisons are affected.

**Production implications:**
- Self-query retriever depends on LLM output being valid and correctly typed —
  not guaranteed, especially across framework version boundaries
- Requires retry logic and fallback to unfiltered search in production
- The lab's own note: "re-run several times if you get blank content"
- Directly observed: same error on every run with numeric filters regardless
  of query wording

**When self-query is valuable:**
- Rich metadata schemas with multiple filterable dimensions
- Queries that naturally reference filterable attributes ("show me womens
  team content", "what changed last month")
- Replacing hand-rolled intent extraction like SRMC-995's
  `extract_category_intent()` — same pattern, automated via LLM

**The before/after insight:**
SRMC-995's `extract_category_intent()` is a manual self-query implementation:
keyword rules that map raw fan queries to category enum values before
semantic search. Self-query retriever does the same thing automatically with
an LLM instead of keyword rules. More flexible, less predictable, requires
more guardrails.

---

### 4. Parent Document Retriever

Solves the chunking tension: small chunks produce better embeddings (precise
semantic meaning), large chunks provide better context for the LLM. Indexes
small child chunks for retrieval precision but returns large parent documents
for generation richness.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import InMemoryStore

parent_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separator='\n')
child_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator='\n')

vectordb = Chroma(collection_name="split_parents", embedding_function=watsonx_embedding())
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectordb,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(txt_data)
```

**Two stores, one retriever:**
- `Chroma` vector store — holds child chunk embeddings for precision retrieval
- `InMemoryStore` docstore — holds full parent documents keyed by parent_id
- Child chunks carry a `parent_id` in metadata linking them to their parent
- `add_documents()` populates both stores simultaneously — do not use
  `Chroma.from_documents()` for this pattern

**The before/after verification:**
```python
# Shows small child chunk that matched
sub_docs = vectordb.similarity_search("smoking policy")
print(sub_docs[0].page_content)
# Output: "Designated Smoking Areas: Smoking is only permitted in..."

# Shows full parent document returned
retrieved_docs = retriever.invoke("smoking policy")
print(retrieved_docs[0].page_content)
# Output: full smoking policy section, ~1000 characters
```

The vector score found the right child. The retriever ignored the child
content and returned its parent. Score was navigation, parent was the
destination.

**Cost trade-off:**
More tokens sent to LLM per query — full parent documents instead of small
chunks. More expensive generation call but richer context. At Rovers scale
the cost difference is negligible. At Officeworks scale with high query
volume, evaluate whether the answer quality improvement justifies the token
cost increase.

**When to use:**
- Documents with naturally coherent parent sections (policy documents,
  FAQ pages, product descriptions)
- When answers to specific questions benefit from surrounding context
- Rovers matchday guide — fan asks "what time do gates open?" but the LLM
  benefits from knowing about parking, food, and family areas too

---

## Exercises

### Exercise 1 — Top 2 Results Using Vector Store-Backed Retriever

```python
vectordb = Chroma.from_documents(documents=chunks_txt, embedding=watsonx_embedding())
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
query = "smoking policy"
docs = retriever.invoke(query)
```

Result: 2 chunks — section heading "5. Smoking Policy" and the policy purpose
paragraph. The heading chunk is essentially noise — a 200-character chunk that
adds no value to LLM context. Production fix: minimum content length filter
at ingestion time, or use parent document retriever to return the full section.

### Exercise 2 — Self-Querying Retriever With Filter

Numeric filters failed with ValueError due to LangChain/Claude version
mismatch. String equality filter worked correctly:

```python
retriever.invoke("I want to watch a movie directed by Christopher Nolan")
# Returns: Inception (2010, rating 8.2, director Christopher Nolan)
```

Used simplified document_content_description to avoid confusing the LLM
with irrelevant metadata descriptions:

```python
document_content_description = "Brief summary of smoking policy"
retriever.invoke("I want to know the policy")
# Returns: 4 policy chunks — no filter applied (no filterable metadata intent)
```

---

## Production Notes

**Retriever selection decision tree for Rovers chatbot:**

```
Query type                          → Retriever
Narrow factual ("gate opening time") → Vector store, similarity search
Broad exploratory ("tell me about matchday") → Vector store, MMR
Category-specific ("womens tickets") → Self-query or manual intent + filter
Multi-dimensional document context  → Parent document retriever
Poor recall on specific domain terms → Multi-query retriever
```

**LC retriever interface vs direct ChromaDB (SRMC-995 vs SRMC-1005):**

SRMC-995 used direct ChromaDB client — full transparency, manual threshold
gate, manual intent extraction. Correct for a prototype where understanding
each step matters.

SRMC-1005 should use LC retriever interface — composability, LangSmith
instrumentation, advanced retriever access, standard patterns the team
understands. `ParentDocumentRetriever` and `MultiQueryRetriever` require
the LC interface — they wrap a base retriever, not a raw ChromaDB collection.

**Fan guide refresh is easier with LC interface:**
`add_documents()` on the retriever updates both stores consistently.
Direct ChromaDB requires manual CRUD across collection and docstore separately.

**Self-query retriever in production requires:**
- Retry logic (LLM output non-deterministic)
- Fallback to unfiltered search on parse failure
- Well-described metadata field descriptions — quality of filter generation
  depends entirely on how well you describe the schema to the LLM
- Version pinning — LangChain/LLM compatibility must be validated end-to-end

---

## Key Concepts Summary

| Retriever | What changes | Cost vs standard | Best for |
|---|---|---|---|
| Vector store (similarity) | Nothing — baseline | Baseline | Narrow specific queries |
| Vector store (MMR) | Result selection algorithm | Same | Broad queries, diversity |
| Vector store (threshold) | Filter by quality score | Same | Quality gate |
| Multi-query | Text going into query embedding | +1 LLM call | Recall improvement |
| Self-query | Adds metadata filter from NL query | +1 LLM call | Filtered retrieval |
| Parent document | Returns parent instead of child | +tokens to LLM | Context richness |

---

## Files

- `LangChain_retriever-v1.ipynb` — main lab notebook
- `companypolicies.txt` — company policies document (9 policies)
- `.env` — Anthropic API key (never commit)