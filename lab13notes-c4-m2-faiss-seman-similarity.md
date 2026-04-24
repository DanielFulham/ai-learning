# Course 4 — Lab 13: Semantic Similarity with FAISS

## Lab Overview

**Course:** IBM RAG and Agentic AI Professional Certificate — Course 4 Module 2
**Lab:** Semantic Similarity with FAISS
**Completed:** 24 April 2026
**Dataset:** 20 Newsgroups (18,846 posts across 20 categories)
**Embedding model:** Universal Sentence Encoder (USE) v4 — Google, via TensorFlow Hub
**Vector store:** FAISS IndexFlatL2

---

## Key Concepts

### The Embed → Store → Search Separation

USE and FAISS are completely independent components. The course video didn't make this explicit — the lab does.

- **USE** — embedding model. Text in, 512-dimensional vector out. One vector per document regardless of length.
- **FAISS** — vector index. Vectors in, positions and distances out. Indifferent to which embedding model produced the vectors.

FAISS would accept vectors from MiniLM, OpenAI, Bedrock, or any other embedding model — only the dimension number changes. This is the same separation as ChromaDB, just more explicit because the two steps are wired manually.

### IndexFlatL2

Brute force exact search. Compares query vector against every vector in the index on each search. No approximation — always returns the true nearest neighbours.

```python
dimension = X_use.shape[1]           # 512 for USE
index = faiss.IndexFlatL2(dimension)
index.add(X_use)                     # load all vectors — instant, no computation
```

Index build is near-instant — no clustering, no pre-computation. Cost is paid at query time.

### The Manual Position Mapping

FAISS returns positions and distances — not documents. The lookup is your responsibility.

```python
distances, indices = index.search(query_vector.astype('float32'), k)

for i, idx in enumerate(indices[0]):
    print(documents[idx])  # manual lookup — position in FAISS = position in documents list
```

ChromaDB stores document text alongside the vector and returns it directly. FAISS returns a position number. You maintain the mapping.

**Critical:** `documents`, `processed_documents`, and the FAISS index must all be the same length and in the same order. If preprocessing filters or skips documents, positions drift and wrong documents come back with no error thrown.

### float32 Requirement

FAISS requires float32 specifically. numpy defaults to float64. Always cast before searching:

```python
query_vector.astype('float32')
```

### Chunking vs Sentence Encoding

USE encodes whatever unit of text you give it — one vector per input regardless of length. Chunking is upstream preprocessing that creates meaningful units before encoding.

- Short text (newsgroup post, FAQ answer) → encode directly
- Long text (full document, article) → chunk first, then encode each chunk

The lab skips chunking because newsgroup posts are already short. In production RAG pipelines you always chunk first.

---

## Production Notes

- **Parallel array drift** — `documents`, `processed_documents`, and FAISS index positions must stay in sync. Any filtering, skipping, or reordering during preprocessing breaks the mapping silently. Safer pattern: store everything in a single list of dicts before indexing, maintain position integrity explicitly.
- **Deduplication before indexing** — the lab loads `subset='all'` (train + test combined) producing duplicate posts. Same content produces near-identical vectors and redundant results. Deduplicate on content hash before building the index.
- **IndexFlatL2 at scale** — fine for small datasets (<10k vectors). Switch to `IndexIVFFlat` for larger corpora — k-means clustering at index build time, `nprobe` controls speed vs recall trade-off at query time.
- **Preprocessing removes numbers** — the lab strips all non-alpha characters including numbers. Acceptable for newsgroup posts, wrong for domains where numbers carry meaning (Rovers FAQ: "gates open 90 minutes before kick-off", "tickets from €15").
- **Embedding is free at inference time** — USE loads locally from TF Hub. No API calls, no token costs after initial model download. Same as MiniLM in the Rovers stack.

---

## Index Type Reference

| Index | Mechanism | Build cost | Query cost | Use when |
|---|---|---|---|---|
| `IndexFlatL2` | Brute force, exact | Instant | O(n) per query | <10k vectors, learning |
| `IndexIVFFlat` | k-means clusters, search nearest clusters | k-means upfront | Sub-linear | 10k–1M vectors |
| `IndexIVFPQ` | Clusters + vector compression | Higher | Sub-linear, less memory | Memory constrained, large scale |

---

## Relevance to SRMC-1008

SRMC-1008 (Fan Segmentation local prototype) uses the same FAISS pattern:
- Synthetic supporter profiles replace newsgroup posts
- MiniLM replaces USE (384 dims vs 512, no TensorFlow dependency)
- `IndexFlatL2` is appropriate for 100 synthetic profiles
- Manual position mapping applies — store profiles and FAISS index in same order

The lab validated the pipeline. SRMC-1008 applies it to a real domain.