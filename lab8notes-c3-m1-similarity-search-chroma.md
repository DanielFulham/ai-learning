# Lab 8 Notes — Similarity Search on Text Using Chroma DB and Python
## Course 3, Module 1

---

## What This Lab Covers

First use of the ChromaDB Python client directly — no LangChain abstraction. 
Creates a collection, adds documents, and performs similarity search using 
the HNSW index with cosine distance.

---

## Key Concepts

**Direct ChromaDB client vs LangChain abstraction:**
- Previous labs used `langchain.vectorstores.Chroma` — LangChain managed 
  collection creation, embedding, and querying behind the scenes
- This lab uses `chromadb.Client()` directly — explicit control over every step
- Same underlying ChromaDB, different interface

**Collection creation with explicit HNSW configuration:**
```python
collection = client.create_collection(
    name="my_grocery_collection",
    metadata={"description": "A collection for storing grocery data"},
    configuration={
        "hnsw": {"space": "cosine"},  # Always set explicitly — default is L2
        "embedding_function": ef
    }
)
```

**Adding documents — ChromaDB handles embedding automatically:**
```python
collection.add(
    documents=texts,
    metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts],
    ids=ids
)
```
No manual embedding step — ChromaDB calls the configured embedding function 
internally. In production, prefer handling embeddings yourself for explicit 
model control.

**Querying with multiple query terms in one call:**
```python
results = collection.query(
    query_texts=query_term,  # List of terms — batched in one call
    n_results=3
)
```
`query_texts` accepts a list — ChromaDB batches the queries internally and 
returns a result set per term. `results['ids']` is a list of lists — one inner 
list per query term.

**Defensive type check pattern:**
```python
if isinstance(query_term, str):
    query_term = [query_term]
```
Makes the function accept either a single string or a list. Common Python 
pattern for flexible function inputs.

**Iterating over multiple query results:**
```python
for q in range(len(query_term)):
    for i in range(min(3, len(results['ids'][q]))):
        doc_id = results['ids'][q][i]
        score = results['distances'][q][i]
        text = results['documents'][q][i]
```
Outer loop over query terms, inner loop over results per term. Index with `[q]` 
not `[0]` when handling multiple queries.

---

## Observations

**Semantic search working as intended:**
Query "apple" returned "golden apple", "fresh red apples", "red fruit" — 
the third result has no lexical match to "apple" but the embedding model 
knows apples are red fruit. A SQL LIKE '%apple%' would miss it entirely.

**Distance scores are cosine distance (lower = more similar):**
- `golden apple` — 0.3825 — strongest match
- `fresh red apples` — 0.4809
- `red fruit` — 0.5965 — no word match, purely semantic

**Query batching returns nested results:**
Two query terms → `results['ids']` has two inner lists. Easy to get wrong 
if you index with `[0]` assuming a single query.

---

## Production Notes

- Always set `"space": "cosine"` explicitly — default L2 is wrong for text/RAG
- `chromadb.Client()` is in-memory — data is lost when script exits. Use 
  `PersistentClient(path="./chroma_db")` for anything that needs to survive restarts
- Letting ChromaDB handle embeddings is convenient but hides the model choice. 
  In production, embed explicitly and pass vectors — keeps embedding model 
  auditable and consistent
- `results['ids']` is always a list of lists — `[0]` for single query, 
  `[q]` when iterating over multiple queries. Easy indexing bug if you forget this
- Telemetry errors (`capture() takes 1 positional argument`) are a known bug 
  in this ChromaDB version — ignorable, doesn't affect functionality

---

## Files

- `similarity_search.py` — main lab implementation
- `requirements.txt` — pinned dependencies