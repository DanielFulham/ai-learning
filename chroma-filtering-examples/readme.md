# ChromaDB Filtering Examples

Demonstrates metadata and document content filtering using the ChromaDB Python client directly (no LangChain abstraction).

## Setup

```bash
python3.11 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
python filtering_examples.py
```

Downloads `all-MiniLM-L6-v2` on first run (~90MB, cached locally after that).

## Examples

| # | Type | Filter Used |
|---|---|---|
| 1 | Metadata | `$eq` — exact source match |
| 2 | Metadata | `$and` + `$eq` + `$lt` |
| 3 | Metadata | `$and` + `$in` + `$lt` |
| 4 | Document content | `$contains` |
| 5 | Combined | Metadata `$gt` + document `$or` + `$contains` |

## Notes

- Filtering is case-sensitive
- `where` filters on metadata, `where_document` filters on document text
- Default client is in-memory — collection resets on every run
- Based on IBM RAG and Agentic AI Certificate — Course 3, Module 1