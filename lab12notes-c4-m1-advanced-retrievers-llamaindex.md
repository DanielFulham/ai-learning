# Lab 12 Notes — Explore Advanced Retrievers in LlamaIndex
## Course 4, Module 1 Lab 2

---

## What This Lab Covers

Six LlamaIndex retriever types demonstrated against a 10-document AI/ML
corpus. Covers VectorIndexRetriever, BM25Retriever, DocumentSummaryIndex
retrievers (LLM and embedding variants), AutoMergingRetriever,
RecursiveRetriever, and QueryFusionRetriever with three fusion modes.
Two exercises: hybrid retrieval implementation and a production RAG pipeline
with evaluation harness.

---

## Local Environment Setup

Run locally on Windows with VS Code + Jupyter notebook kernel.

**Key Windows adaptations required:**

- `| tail -n 1` — Linux only. Remove from all `!pip install` lines.
- IBM Watsonx credentials not required — swapped to Anthropic stack throughout.
- `python-dotenv` must be installed in the same kernel environment as the
  notebook. Use `sys.executable` to ensure correct install target:
  ```python
  import sys
  !{sys.executable} -m pip install python-dotenv
  ```
- `resource` module not available on Windows — LlamaIndex checks for it at
  import time and prints a warning. Harmless, ignore it.

**IBM → Anthropic swap:**

```python
from llama_index.llms.anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

def create_llm():
    try:
        llm = Anthropic(
            model="claude-haiku-4-5",
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            temperature=0.9
        )
        print("✅ Anthropic Claude Haiku initialized")
        return llm
    except Exception as e:
        print(f"⚠️ Anthropic initialization error: {e}")
        from llama_index.core.llms.mock import MockLLM
        return MockLLM(max_tokens=512)
```

Install:
```python
!pip install llama-index \
    llama-index-embeddings-huggingface \
    llama-index-llms-anthropic \
    llama-index-retrievers-bm25 \
    sentence-transformers \
    rank-bm25 \
    PyStemmer \
    python-dotenv
```

**Critical — load dotenv before create_llm() is called.** The LLM instance
captures the API key at construction time. Re-run `load_dotenv()`, re-run
`create_llm()`, re-run `Settings` cell — in that order — if auth fails.

---

## Three Index Types

LlamaIndex has three core index types, each designed for a different retrieval
strategy:

| Index | How it retrieves | Best for |
|---|---|---|
| `VectorStoreIndex` | Cosine similarity on embeddings | Semantic / conversational queries |
| `DocumentSummaryIndex` | LLM or embedding match on summaries | Large diverse document sets |
| `KeywordTableIndex` | Exact keyword extraction and mapping | Exact term matching, hybrid search |

**`DocumentSummaryIndex` calls the LLM at index build time** — generates one
summary per document during the offline phase. Observable as `current doc id:`
log lines during construction. Budget for this cost at index build time, not
query time.

---

## The Six Retrievers

### 1. VectorIndexRetriever

Baseline semantic retriever. Two creation methods — explicit constructor and
shorthand:

```python
# Explicit — consistent pattern across all retriever types
retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)

# Shorthand — syntactic sugar, same result
retriever = index.as_retriever(similarity_top_k=3)
```

Results come back as `NodeWithScore` — node content + relevance score bundled
together. Score is cosine similarity, surfaced automatically. In SRMC-995 this
had to be extracted manually from ChromaDB distances.

**The explicit/query engine separation:**
```python
retriever = index.as_retriever(...)          # returns nodes
query_engine = RetrieverQueryEngine(retriever=retriever)  # retrieval + synthesis
```
LlamaIndex makes this distinction first-class. LangChain's `RetrievalQA`
bundles both — LlamaIndex unpacks them. Understanding the separation makes
it easier to compose custom pipelines.

---

### 2. BM25Retriever

Keyword-based retrieval. No embeddings, no LLM, no API calls. Pure in-memory
statistical scoring.

```python
import Stemmer

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=3,
    stemmer=Stemmer.Stemmer("english"),
    language="english"
)
```

**BM25 vs TF-IDF:**
- TF-IDF: linear term frequency scaling, no document length normalisation
- BM25: term frequency saturation (diminishing returns on repeated terms),
  document length normalisation via `b` parameter

**Key parameters:**
- `k1 ≈ 1.2` — term frequency saturation rate
- `b ≈ 0.75` — document length normalisation (0 = none, 1 = full)

**BM25 scores are not cosine similarity** — they're unbounded positive floats.
A score of 2.52 is not the same scale as a vector score of 0.87. Cannot be
combined directly without normalisation.

**BM25 is 30 years old** (1994). Standard in Elasticsearch, Lucene, Solr.
Well understood, deterministic, no calibration required.

**Production use case — BM25 as pre-filter before vector search:**
```
BM25 (in-memory, zero API cost) → reduce 1000 docs to 20 candidates
        ↓
Vector search (embedding API call) → rerank 20 to top 5
        ↓
LLM (expensive) → synthesise answer from 5
```
Each stage reduces the candidate set. Cheap operations eliminate before
expensive operations run. Same instinct as the similarity threshold gate
in SRMC-995.

*Rovers backlog note (SRMC-XXXX): Revisit BM25 pre-filter when corpus exceeds
~500 documents or embedding API costs become measurable at query volume.*

---

### 3. DocumentSummaryIndex Retrievers

Two variants operating over the same index:

**LLM-based:**
```python
retriever = DocumentSummaryIndexLLMRetriever(
    document_summary_index,
    choice_top_k=3
)
```
Sends query + all document summaries to LLM. LLM rates each summary's
relevance and selects top k. Returns original documents, not summaries.
Score is a rating (e.g. 9.0/10), not cosine similarity — non-deterministic.

**Embedding-based:**
```python
retriever = DocumentSummaryIndexEmbeddingRetriever(
    document_summary_index,
    similarity_top_k=3
)
```
Embeds query, runs cosine similarity against summary embeddings. Returns
original documents. Deterministic, no LLM call at query time.

**Key insight:** The summary is a cheap proxy used for filtering. The LLM
never sees the summary — it sees the original document. Same concept as a
library card catalogue — the catalogue entry decides which shelf to visit,
but you still read the actual book.

**Production default: embedding-based.** Deterministic, cheaper per query,
consistently better or equal to LLM-based on straightforward queries.

**LLM-based use case:** Diagnostic and development tool during index
evaluation. Run LLM retriever against query set to identify where summaries
are failing, then fix the index. Once embedding retriever quality matches
LLM retriever, ship embedding retriever to production.

**Cost architecture principle:** LLM retriever = expensive at query time.
Embedding retriever = pay once at index build (summary generation), cheap
at query time. For high query volume / stable documents → embedding retriever
is significantly cheaper.

---

### 4. AutoMergingRetriever

LlamaIndex's equivalent of LangChain's `ParentDocumentRetriever`. Indexes
documents at multiple chunk sizes (parent/child/grandchild hierarchy). Base
retriever finds the most relevant small chunks; AutoMergingRetriever checks
whether enough sibling chunks from the same parent were retrieved, and if so,
returns the parent instead.

```python
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 256, 128]  # parent, child, grandchild
)
hier_nodes = node_parser.get_nodes_from_documents(documents)

docstore = SimpleDocumentStore()
docstore.add_documents(hier_nodes)
storage_context = StorageContext.from_defaults(docstore=docstore)

base_index = VectorStoreIndex(hier_nodes, storage_context=storage_context)
base_retriever = base_index.as_retriever(similarity_top_k=6)

auto_merging_retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True  # shows merge decisions, useful during development
)
```

**Two objects, two jobs:**
- `base_retriever` — finds relevant small chunks via vector search
- `AutoMergingRetriever` — decides whether to merge siblings into parent

`AutoMergingRetriever` cannot work standalone — it needs `base_retriever`
to do the vector search first, and `storage_context` to look up parent nodes.

**`verbose=True` output:**
```
> Merging 1 nodes into parent node.
> Parent node id: xxx
> Parent node text: Deep learning uses neural networks...
```
Each line is one merge decision. "Merging 1 nodes" means only one child
matched — threshold for merging siblings was not met. On a real corpus with
naturally coherent sections you'd see "Merging 3 nodes into parent" where
multiple related child chunks trigger a full parent return.

**The lab corpus (single-sentence documents) is too small to demonstrate
dramatic merging.** The pattern works correctly, it just collapses 6 base
results to 2 parent nodes rather than 6 to 1 as it would on a real document.

**On real Rovers documents:**
Fan asks "where do I park on matchday?" → base retriever finds 3 parking
child chunks (locations, costs, times) → all from same parent → merge →
LLM receives full parking section, not 3 fragments.

---

### 5. RecursiveRetriever

Follows node relationship references across documents. Designed for corpora
with explicit cross-document links — academic paper citations, documentation
cross-references, linked FAQ entries.

```python
recursive_retriever = RecursiveRetriever(
    "vector",                    # root entry point
    retriever_dict=retriever_dict,  # maps node IDs to retrievers
    query_engine_dict={},
    verbose=True
)
```

**How it works:**
1. Queries root retriever (`"vector"`) — returns initial nodes
2. Reads `references` metadata from each retrieved node
3. Looks up each reference ID in `retriever_dict`
4. Retrieves from referenced retrievers
5. Repeats until no more references to follow

**Best mental model:** Wikipedia "See also" links. A query about deep
learning retrieves the deep learning article, which references neural
networks and backpropagation — RecursiveRetriever follows those links and
retrieves those articles too.

**The lab example is synthetic** — references are assigned positionally
(doc_0 references doc_1 and doc_2) with no semantic relationship. The
pattern is demonstrated mechanically but the value proposition requires a
real reference graph to be visible.

**When it's genuinely useful:**
- Academic paper corpora with real citations
- Documentation with genuine cross-references
- Knowledge bases with intentionally linked content

**Production setup requirement:** References must be meaningful at index
build time. Either extracted from source content (citations in PDFs) or
defined explicitly in metadata schema at ingestion.

---

### 6. QueryFusionRetriever

Combines multiple retrieval strategies and optionally generates multiple
query variants. Two ideas in one retriever:

**Idea 1 — Multi-retriever fusion:** Combine vector + BM25 results
(equivalent to LangChain's `EnsembleRetriever` but with more sophisticated
merging strategies).

**Idea 2 — Multi-query generation:** LLM generates query variants, each
runs through the retriever independently, results fused
(equivalent to LangChain's `MultiQueryRetriever`).

```python
fusion_retriever = QueryFusionRetriever(
    [base_retriever],
    similarity_top_k=3,
    num_queries=3,       # original + 2 LLM-generated variants
    mode="reciprocal_rerank",
    use_async=False,
    verbose=True
)
```

**Three fusion modes:**

| Mode | Mechanism | Use when |
|---|---|---|
| `reciprocal_rerank` | Position-based scoring: `1/(rank+60)` summed across queries | Default — robust, no score calibration required |
| `relative_score` | Normalises scores within each result set before merging | You trust your embedding model's score calibration |
| `dist_based_score` | Z-score / percentile normalisation across score distributions | Combining retrievers with very different score scales and distributions |

**RRF explained:**
Instead of comparing raw scores (BM25: 2.52 vs cosine: 0.87 — incomparable
scales), RRF uses rank position. A document ranked 1st in both lists scores
higher than a document ranked 1st in one and 5th in the other. Scale-invariant,
robust to outliers, no parameters to tune. Production default.

**Cost model:**
- 1x LLM call — generates query variants (cheap model appropriate)
- Nx embedding calls — one per query variant (N = num_queries)
- 0x LLM calls during retrieval itself
- Fusion step — pure maths, no API calls

LLM cost at query time, not index time. Opposite to DocumentSummaryIndex
which pays at index time. For high query volume, this adds up.

**When to use for Rovers:**
Broad exploratory queries ("tell me about matchday") benefit from multiple
query reformulations. Narrow factual queries ("what time do gates open?")
don't — single query finds the answer, multiple variants add cost for no
gain. Revisit when volunteer evaluation (SRMC-1011) shows recall failures
on broad queries.

---

## Exercises

### Exercise 1 — Hybrid Retriever (Manual Score Fusion)

Implemented a weighted average fusion of VectorIndexRetriever and
BM25Retriever scores. Key challenge: node IDs differ across retrievers for
the same content — match by text content, not ID.

```python
def hybrid_retrieve(query, top_k=5):
    vector_nodes = vector_retriever.retrieve(query)
    bm25_nodes = bm25_retriever.retrieve(query)

    # Normalise to 0-1 range (scores are on different scales)
    vector_max = max([n.score for n in vector_nodes], default=1)
    bm25_max = max([n.score for n in bm25_nodes], default=1)

    # Two-pass dict keyed by text content
    combined = {}
    for node in vector_nodes:
        combined[node.text] = {
            "text": node.text,
            "vector_score": (node.score or 0) / vector_max,
            "bm25_score": 0.0
        }
    for node in bm25_nodes:
        if node.text in combined:
            combined[node.text]["bm25_score"] = (node.score or 0) / bm25_max
        else:
            combined[node.text] = {
                "text": node.text,
                "vector_score": 0.0,
                "bm25_score": (node.score or 0) / bm25_max
            }

    # Weighted average fusion
    for item in combined.values():
        item["combined_score"] = (0.6 * item["vector_score"]) + (0.4 * item["bm25_score"])

    return sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)[:top_k]
```

**Why 0.6/0.4:** Starting point based on convention — semantic search
generally outperforms keyword search for conversational queries. Not a rule.
Tune against real query data. `EnsembleRetriever` in LangChain uses the same
weighted average pattern. RRF avoids the need to tune weights entirely.

**Observed results:** Documents appearing near the top of both retrievers
scored highest combined. Documents appearing in only one retriever were
penalised by the 0.0 default for the missing score. The combined score
correctly rewarded inter-retriever agreement.

### Exercise 2 — Production RAG Pipeline

```python
class ProductionRAGPipeline:
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.vector_retriever = VectorIndexRetriever(
            index=self.index, similarity_top_k=3
        )
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=lab.nodes, similarity_top_k=3,
            stemmer=Stemmer.Stemmer("english"), language="english"
        )

    def query(self, question, strategy="auto"):
        if strategy == "vector":
            nodes = self.vector_retriever.retrieve(question)
        elif strategy == "bm25":
            nodes = self.bm25_retriever.retrieve(question)
        elif strategy == "hybrid":
            nodes = hybrid_retrieve(question)
        elif strategy == "auto":
            # Routing heuristic: short queries → keyword likely, long → semantic
            if len(question.split()) <= 3:
                nodes = self.bm25_retriever.retrieve(question)
            else:
                nodes = self.vector_retriever.retrieve(question)
        else:
            nodes = self.vector_retriever.retrieve(question)
        return nodes

    def evaluate(self, test_queries, expected_answers):
        passed = 0
        total = len(test_queries)
        for query, expected in zip(test_queries, expected_answers):
            nodes = self.query(query)
            top_node = nodes[0] if nodes else None
            if top_node and expected.lower() in top_node.text.lower():
                passed += 1
        print(f"Overall: {passed}/{total} ({passed/total*100:.0f}%)")
        return passed / total
```

**Result:** 3/3 queries passed, recall score 1.0 on the test dataset.

**Production code quality notes (would not PR as-is):**
- `lab.nodes` hardcoded inside `__init__` — should derive from `self.index`
- No type hints
- `hybrid_retrieve` called as global function — should be a method or injected
- Routing heuristic (`len <= 3`) is untested, needs calibration
- `evaluate` mixes printing with metric calculation — separate concerns in production

The exercise demonstrates the concepts correctly. Production version would
address all of the above.

---

## LangChain vs LlamaIndex Retriever Mapping

| LangChain | LlamaIndex | Notes |
|---|---|---|
| `VectorStore.as_retriever()` | `VectorIndexRetriever` | Same concept, LlamaIndex more explicit |
| `EnsembleRetriever` | `QueryFusionRetriever` | LlamaIndex has more fusion strategy options |
| `MultiQueryRetriever` | `QueryFusionRetriever` (num_queries > 1) | LlamaIndex combines both in one class |
| `ParentDocumentRetriever` | `AutoMergingRetriever` | LlamaIndex uses node graph natively |
| No direct equivalent | `DocumentSummaryIndexRetriever` | Summarise-then-filter is LlamaIndex-specific |
| No direct equivalent | `RecursiveRetriever` | Reference graph traversal, LlamaIndex-specific |

---

## Production Notes

**Retriever selection for Rovers chatbot:**

```
Query type                              → Retriever
Narrow factual ("gate opening time")    → VectorIndex, similarity
Broad exploratory ("tell me about matchday") → QueryFusion or MMR
Category-specific ("womens tickets")    → Self-query or manual intent filter
Long document context needed            → AutoMerging
Exact proper noun (player names)        → BM25 or hybrid
Large diverse corpus                    → DocumentSummaryIndex pre-filter
```

**DocumentSummaryIndex in production:**
Use embedding-based retriever, not LLM-based. LLM-based is for development
evaluation only — use it to identify index quality gaps, then fix the corpus
and switch to embedding-based for production.

**Fusion strategy selection:**
- Default → RRF. No calibration required, scale-invariant.
- Trust your embedding scores → Relative Score
- Different score distributions → Distribution-Based
- For Rovers at current corpus size → RRF if using fusion at all

**BM25 + vector hybrid consideration:**
BM25 pre-filter before embedding call reduces API costs at scale. No
additional infrastructure — BM25 index built offline, loaded at Lambda cold
start. Revisit when corpus exceeds ~500 documents (SRMC-XXXX).

**AutoMerging threshold tuning:**
Too aggressive (merge on 1 child) → large parents, irrelevant content reaches
LLM, costs more. Too conservative → small fragments, context lost. Tune the
merge threshold against real queries on a real corpus, not toy data.

**The cost architecture hierarchy:**
```
BM25 (free) → pre-filter
Embedding call (cheap) → vector search
Small LLM (cheap) → query reformulation, summary filtering
Large LLM (expensive) → synthesis only
```
Each stage eliminates candidates before the next stage runs. Never call an
expensive operation on a candidate set that a cheaper operation could have
filtered.

---

## Key Concepts Summary

| Retriever | LLM at index time | LLM at query time | Cost vs baseline |
|---|---|---|---|
| VectorIndex | No | No | Baseline |
| BM25 | No | No | Cheaper (no embedding) |
| DocSummary (embed) | Yes (summaries) | No | Same per query |
| DocSummary (LLM) | Yes (summaries) | Yes (selection) | +1 LLM call |
| AutoMerging | No | No | +tokens to LLM |
| Recursive | No | No | Depends on depth |
| QueryFusion | No | Yes (variants) | +1 LLM call |

---

## Files

- `Advanced_retrievers_in_LlamaIndex-v1.ipynb` — main lab notebook
- `.env` — Anthropic API key (never commit)