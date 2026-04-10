# Course 3 — Lab 7: Similarity Search by Hand

## Lab Overview

**Course:** IBM RAG and Agentic AI Professional Certificate — Course 3
**Lab:** Similarity Search by Hand
**Completed:** 10 April 2026
**Time taken:** ~2 hours

Hands-on implementation of the three core distance and similarity metrics used in vector search — L2 distance, dot product, and cosine similarity — built manually from scratch before using library equivalents. Covers the full similarity search pipeline: embed documents, embed a query, normalise, compute similarity, return best match.

---

## What Was Built

A similarity search pipeline that:
1. Embeds 4 sentences using `sentence-transformers` (`all-MiniLM-L6-v2`)
2. Computes L2 distance, dot product similarity, and cosine similarity manually
3. Validates manual implementations against numpy, scipy, and PyTorch equivalents
4. Performs a query similarity search — embed a query, compare against documents, return highest scoring match

---

## Key Concepts

### The Three Metrics

| Metric | Formula | Sensitive to Magnitude | Best For |
|---|---|---|---|
| L2 Distance | `sqrt(sum((a - b)²))` | Yes | Spatial data, clustering |
| Dot Product | `sum(a * b)` | Yes | Recommender systems, neural networks |
| Cosine Similarity | `dot(a,b) / (‖a‖ × ‖b‖)` | No | Text, embeddings, NLP |

**One-liner summary:**
- L2 = how far apart
- Cosine = how aligned
- Dot product = how aligned AND how strong

### Why Cosine for RAG

Cosine similarity ignores magnitude — a short document and a long document about the same topic score as equally similar. For text retrieval, document length shouldn't affect relevance. Cosine handles this correctly; L2 does not.

### The Relationship Between the Three

They are mathematically connected:

```
dot product = ‖a‖ × ‖b‖ × cos(angle between a and b)
cosine similarity = dot product / (‖a‖ × ‖b‖)
```

Cosine similarity is dot product with magnitudes divided out. When vectors are normalised (magnitude = 1), dot product and cosine similarity are identical — which is why many embedding models normalise by default.

### Normalisation

Dividing a vector by its L2 norm scales it to magnitude 1. Preserves direction, removes magnitude. After normalisation, cosine similarity = dot product (cheaper to compute).

```python
# Manual
l2_norm = np.linalg.norm(vector)
normalised = vector / l2_norm

# PyTorch
normalised = torch.nn.functional.normalize(torch.from_numpy(vector)).numpy()
```

A normalised vector has the property: `sum(component²) = 1`

### Matrix Multiplication Shortcut

The full pairwise similarity matrix can be computed in one line on normalised embeddings:

```python
cosine_similarity = normalised_embeddings @ normalised_embeddings.T
```

Three equivalent ways to do matrix multiplication in Python:

```python
np.dot(A, B)       # older style
np.matmul(A, B)    # recommended for 2D arrays
A @ B              # Python operator, calls __matmul__ under the hood
```

`np.matmul` and `@` are preferred — `np.dot` behaves differently for 3D+ arrays which can cause subtle bugs.

### Similarity Search Pipeline (Full)

```
OFFLINE (build index)
documents
  → model.encode(documents)        # embed → shape (4, 384)
  → normalise each vector          # magnitude = 1

ONLINE (per query)
query string
  → model.encode(query)            # embed → shape (384,)
  → normalise query vector
  → normalised_docs @ query.T      # dot product = cosine similarity
  → np.argmax(scores)              # highest score = best match
  → documents[best_match_idx]      # return document
```

---

## Key Code Patterns

### L2 Distance

```python
# Manual
def euclidean_distance_fn(v1, v2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))

# Efficient manual (upper triangle only, mirror result)
for i in range(n):
    for j in range(n):
        if j > i:
            matrix[i,j] = np.linalg.norm(embeddings[i] - embeddings[j])
            matrix[j,i] = matrix[i,j]

# Library
scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
```

### Dot Product

```python
# Manual
def dot_product_fn(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

# Matrix (all pairs at once)
dot_product_matrix = embeddings @ embeddings.T
# or
dot_product_matrix = np.matmul(embeddings, embeddings.T)

# Dot product distance (flip sign)
dot_product_distance = -dot_product_matrix
```

### Cosine Similarity

```python
# Manual (normalise first, then dot product)
l2_norms = np.sqrt(np.sum(embeddings**2, axis=1))
l2_norms_reshaped = l2_norms.reshape(-1, 1)   # shape (4,1) for broadcasting
normalised = embeddings / l2_norms_reshaped

cosine_similarity = normalised @ normalised.T

# Cosine distance
cosine_distance = 1 - cosine_similarity
```

### Query Similarity Search

```python
# Embed and normalise query
query_embedding = model.encode(query)
query_norm = np.linalg.norm(query_embedding)
normalised_query = query_embedding / query_norm

# Score against all documents
scores = normalised_embeddings @ normalised_query

# Return best match
best_idx = np.argmax(scores)
print(documents[best_idx])
```

### Production Version (library)

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

scores = cosine_similarity(query_embedding, doc_embeddings)[0]
best_idx = scores.argmax()
print(documents[best_idx])
```

---

## Results

4 sentences about "bugs" — 2 software context, 2 insect context.

**Cosine similarity matrix:**
```
                    Sentence 0   Sentence 1   Sentence 2   Sentence 3
Sentence 0 (SW)     1.00         0.51         0.24         0.23
Sentence 1 (SW)     0.51         1.00         0.21         0.24
Sentence 2 (IN)     0.24         0.21         1.00         0.50
Sentence 3 (IN)     0.23         0.24         0.50         1.00
```

Same-topic pairs score ~0.50. Cross-topic pairs score ~0.23. The model correctly identified semantic meaning despite all four sentences containing the same keyword "bugs."

**Query result:**
Query: *"Who is responsible for a coding project and fixing others' mistakes?"*
Best match: *"Bugs introduced by the intern had to be squashed by the lead developer."* (score: 0.3856)

---

## Production Notes

**All of this is hidden inside ChromaDB / pgvector.** When you configure `hnsw:space: cosine` in ChromaDB or use the `<=>` operator in pgvector, every calculation in this lab runs automatically on every insert and query. The lab exists so you understand what you're choosing and why.

**pgvector operators:**
```sql
embedding <=>  -- cosine distance
embedding <->  -- L2 distance
embedding <#>  -- inner product (dot product)
```

**Threshold matters in production.** Vector search always returns a result even when nothing is relevant. Set a minimum similarity threshold to avoid passing poor context to the LLM:

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.70}
)
```

Threshold should be validated against real test queries — not picked arbitrarily. High-stakes domains (legal, medical) warrant higher thresholds.

**Different embedding models produce different absolute values.** The relationships hold across good models, but absolute scores are model-specific. Don't compare scores across models — only within the same model.

**Floating point rounding** means normalised vector magnitudes may show as `1.0000002` rather than exactly `1.0`. Use `np.allclose` not `==` when comparing float results.

**`axis=1` in numpy** sums across columns (dimensions) for each row (document), collapsing a `(4, 384)` matrix to `(4,)`. Without it you'd sum everything into one number — wrong.

**`reshape(-1, 1)`** converts a flat `(4,)` array to a `(4, 1)` column so numpy can broadcast division correctly across a `(4, 384)` matrix.

---

## Local Setup Notes

| Issue | Fix |
|---|---|
| `!pip install ... \| tail -n 1` fails on Windows | Comment out `\| tail -n 1` — Linux command not available on Windows |
| Notebook kernel not finding venv | Use **Enter interpreter path** in VS Code kernel picker and paste `venv\Scripts\python.exe` path directly |
| Install directly to venv | `C:\path\to\venv\Scripts\pip.exe install numpy scipy torch sentence-transformers==4.1.0` |
| `TqdmWarning: IProgress not found` | Cosmetic warning only — safe to ignore |

---

## Key Classes Reference

```python
import math
import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer

# Embed
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)   # shape (n, 384)

# L2 norm
np.linalg.norm(vector)
scipy.spatial.distance.cdist(A, B, 'euclidean')

# Normalise
torch.nn.functional.normalize(torch.from_numpy(embeddings)).numpy()

# Matrix multiply
A @ B
np.matmul(A, B)

# Verify equality (float safe)
np.allclose(A, B, atol=1e-05)
```

---

## Production Insights

These are real architectural considerations that emerge when you move similarity search from a lab into a production system. I've noted these below from some insights while doing the labs.

### 1. The Empty Result Problem and Hallucination Risk

Vector search always returns a result — even when nothing in the knowledge base is relevant to the query. There is no concept of "no match." The retriever will return the least-dissimilar chunks regardless of how poor the similarity score is. If those weak chunks get passed to the LLM, the model fills the gap with plausible-sounding fabrication — hallucination.

The failure chain:
1. User asks something with no good match in the knowledge base
2. Vector search returns the least-worst chunks anyway
3. LLM receives poor context but tries to answer
4. LLM hallucinates a confident but wrong answer

The fix is two layers of defence — a similarity threshold that prevents weak context reaching the LLM, and a system prompt guardrail that instructs the model to say "I don't know" rather than fabricate.

### 2. Similarity Threshold — Set it Deliberately

The threshold is not a library default — you choose it based on your use case and validate it against real queries. The right approach:

1. Run test queries you know should match — note the scores
2. Run queries you know shouldn't match — note the scores
3. Pick a threshold that cleanly separates the two groups

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.70}
)
```

**Threshold by domain risk:**
- General chatbot (e.g. fan FAQ) — 0.65–0.75 — wrong answers are inconvenient, not dangerous
- Legal or compliance information — 0.85+ — wrong answers carry real risk; better to say "I don't know"
- Medical or clinical systems — 0.85+ — patient safety requires high confidence before answering

A system that knows its own limitations is more trustworthy than one that always tries to answer.

### 3. Bounded Contexts — Separate Indexes Per Domain (sport system)

Trying to teach one embedding model to handle multiple divergent domains leads to retrieval bleeding — a query about pricing might pull back results from customer support, or vice versa. Contrastive fine-tuning can help but gets unwieldy fast as the number of domain pairs grows.

The cleaner architecture is separate indexes per bounded context for a sport domain:

```
Pricing index     → queries about market prices and odds
Resulting index   → queries about match/event outcomes
Support index     → queries about customer account issues
```

Each index only needs to understand its own domain. Updates to one don't affect others. Retrieval is scoped — a pricing query never pulls from support content.

This maps directly to Domain-Driven Design bounded contexts. In practice: separate ChromaDB collections, separate pgvector tables, or separate Pinecone namespaces. Same pattern, different infrastructure.

For a chatbot covering multiple document types (Shop and Ticketing), separate indexes with a routing layer is cleaner than one combined index — especially for questions that span both, where you query each index independently and let the LLM synthesise across the results.

### 4. Fine-tuning vs Metadata Filtering

When retrieval quality is poor — similar-sounding items being confused, domain-specific terminology not being understood — there are two ways to fix it:

**Fine-tuning (contrastive learning):** Give the model examples of what should be similar and what shouldn't, and retrain. The model learns domain-specific relationships. Powerful but expensive — requires labelled training data, compute, and ongoing maintenance as the domain evolves.

**Metadata filtering:** Tag your chunks with categories and filter at query time. Simpler, faster, and requires no training data.

```python
# Metadata filtering in ChromaDB — no fine-tuning needed
collection.query(
    query_embeddings=[query_vector],
    where={"ticket_type": "mens"},
    n_results=5
)
```

In most production cases, metadata filtering solves the problem without the overhead of fine-tuning. Fine-tuning becomes worth it when the base model is genuinely producing poor embeddings for your domain and filtering alone can't compensate — typically in highly specialised fields (clinical, legal, financial) with significant training data available.

### 5. The Chatbot Knowledge Problem — High Threshold + Tool Calls + Fallback

A RAG chatbot with a static knowledge base will always struggle with live data questions. The architecture that handles this cleanly is not "try to answer everything from documents" — it's tiered:

- **Evergreen content** (FAQs, policies, ground information) — high cosine threshold, answer from RAG index with confidence
- **Live data** (fixtures, ticket availability, results) — don't attempt from documents; call an external API via an agent tool
- **Out of scope** (anything else) — return a graceful fallback, direct the user elsewhere

```
Fan asks: "When's the next home game?"
→ Similarity score against FAQ docs: 0.31 (below threshold)
→ Route to fixture lookup tool → call Ticketing API
→ Return live answer

Fan asks: "How do I get to the Stadium?"
→ Similarity score against FAQ docs: 0.84 (above threshold)
→ Return RAG answer from fan guide

Fan asks: "What's the weather tomorrow?"
→ Similarity score: 0.18 (well below threshold)
→ No tool match
→ Return fallback: "I can only help with Shamrock Rovers questions."
```

This isn't a limitation of the chatbot — it's a well-designed system that knows what it knows, knows what it doesn't, and handles both correctly. The metric for value is not "does it answer everything" but "what percentage of real fan enquiries does it handle correctly." Even a 60% deflection rate on evergreen questions represents significant reduction in manual email handling — before the live data layer is even built.