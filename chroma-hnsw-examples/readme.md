# ChromaDB HNSW Examples

Demonstrates HNSW index configuration and similarity search using the ChromaDB 
Python client directly.

## Setup

```bash
python3.11 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Downloads `all-MiniLM-L6-v2` on first run (~90MB, cached locally after that).

## Run

```bash
python hnsw_examples.py
```

## HNSW Parameters

| Parameter | Default | Tuning guidance |
|---|---|---|
| `space` | `l2` | Always set `cosine` for text/RAG. Cannot be changed after collection creation. |
| `ef_search` | `100` | Online cost — paid per query. Tune for latency vs recall tradeoff. |
| `ef_construction` | `100` | Offline cost — paid once at index build. Set higher (200+) in production. |
| `max_neighbors` | `16` | Higher = denser graph, better search, more memory. Cannot be changed after creation. |

## Examples

| # | Query | Filter | Demonstrates |
|---|---|---|---|
| 1 | `cats` | None | Semantic similarity — animal docs rank above library docs |
| 2 | `polar bear` | None | Failure mode — "polars" library matches over bear species |
| 3 | `polar bear` | Metadata `topic = animals` | Metadata filter fixes the failure |
| 4 | `polar bear` | Document `$not_contains library` | Document filter as alternative fix |
| 5 | `polar bear` | Both combined | Belt and braces — metadata + document filter together |

## Notes

- Default distance metric is L2 — always set `cosine` explicitly for text use cases
- `space` and `max_neighbors` cannot be changed after collection creation — get them right upfront
- `ef_construction` is an offline cost — be generous, default of 100 is conservative
- `ef_search` is an online cost — default of 100 is reasonable, tune based on observed latency
- Based on IBM RAG and Agentic AI Certificate — Course 3, Module 1