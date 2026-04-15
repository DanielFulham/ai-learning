# Lab 9 Notes — Similarity Search on Employee Records using Python and Chroma DB
## Course 3, Module 2

---

## What This Lab Covers

Similarity search on structured employee data using ChromaDB. 
Extends the Module 1 direct client work to handle key-value pair data 
(not just free text), with both semantic similarity search and metadata filtering.

Also used as the basis for the free-hand books search project — extended into 
full Onion architecture as a production design exercise.

---

## Key Concepts

**Serialising structured data for embedding:**

The critical decision in this lab — structured employee dicts cannot be embedded 
directly. They must be converted to natural language sentences first:

```python
document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
document += f"Employment type: {employee['employment_type']}."
```

The embedding model is trained on natural language — a sentence produces a 
semantically meaningful vector. A JSON string or comma-separated list does not.

**The dual-layer pattern:**

Two representations of the same record serving two different purposes:
- `documents` — natural language sentence, gets embedded, drives semantic search
- `metadatas` — raw structured fields, drives hard filters

Attributes you want to filter on exactly (department, location, experience range) 
go in metadata. Attributes that describe meaning (role, skills) go in the document.

```python
collection.add(
    ids=[employee["id"] for employee in employees],
    documents=employee_documents,
    metadatas=[{
        "name": employee["name"],
        "department": employee["department"],
        "experience": employee["experience"],
        "location": employee["location"],
    } for employee in employees]
)
```

**`get` vs `query` result structure:**

A consistent source of bugs when switching between the two:

```python
# collection.get() — flat result
results['ids']          # flat list
results['metadatas'][i] # direct index

# collection.query() — nested result (one inner list per query)
results['ids'][0]          # [0] for single query
results['metadatas'][0][i] # nested index
```

**Combined search — semantic + metadata filter:**

```python
results = collection.query(
    query_texts=["senior Python developer full-stack"],
    n_results=5,
    where={
        "$and": [
            {"experience": {"$gte": 8}},
            {"location": {"$in": ["San Francisco", "New York", "Seattle"]}}
        ]
    }
)
```

Semantic retrieval narrows to the most relevant candidates. 
Metadata filter enforces hard constraints. Neither alone is as powerful.

---

## Observations

**Semantic search limitations on structured data:**

Query "Python developer with web development experience" returned Matthew Garcia 
(Junior Software Engineer, no Python) above Alex Rodriguez (Lead, Python + React). 
The web development signal in the query text pulled in HTML/CSS skills 
over actual Python experience.

Lesson: attributes you need to match precisely (skill presence, seniority) 
should be metadata fields, not buried in the embedded document text.

**Leadership title vs leadership role:**

Query "team leader manager with experience" returned Jane Smith (Marketing Manager) 
above David Lee (Engineering Manager, 15 years, mentoring skills). The model 
picked up on "Manager" in job titles rather than the leadership/mentoring skills 
the query implied.

Lesson: semantic search finds conceptual proximity, not intent. 
Ambiguous queries produce ambiguous results.

**delete by `where` vs delete by ID:**

Prefer resolve-then-delete over direct `where` delete in production:

```python
# Fragile — silent if metadata was set incorrectly
collection.delete(where={"type": "match_preview"})

# Safer — audit step before deletion
items = collection.get(where={"type": "match_preview"})
# inspect items['ids'] before proceeding
collection.delete(ids=items['ids'])
```

---

## Production Notes

- Numeric attributes (experience, rating, page count) don't embed meaningfully — 
  put them in metadata for range filters, not in document text
- `where` on delete has no confirmation — resolve IDs first, delete by ID
- `update` re-embeds automatically on change — in batch pipelines, avoid 
  loop-and-update. Batch writes instead
- Embedding model cannot be changed on an existing collection — plan the 
  embedding model choice upfront. Changing it = full collection rebuild
- `upsert` is the production write pattern — idempotent, handles create and 
  update in one call. `add` throws on duplicate IDs

---

## Architecture Extension — Books Free-Hand Project

The employee lab pattern was extended into a full Onion architecture for the 
books search free-hand exercise. Key additions:

**Onion layer structure:**
```
books_search/
├── application/
│   ├── interfaces/         # Protocols for app layer consumers
│   ├── book_loader.py      # load_books() — converts raw dicts to Book domain objects
│   ├── book_document_builder.py  # build_book_documents(), build_book_metadatas()
│   ├── book_search_service.py    # All search use cases
│   ├── container.py        # Composition root — wires interfaces to concretes
│   └── result_printer.py   # Presentation helpers
├── data/
│   └── books.py            # Raw dicts — never modified
├── domain/
│   └── book.py             # Book dataclass with to_document() and to_metadata()
├── infra/
│   ├── book_repository.py  # create_collection(), add_books()
│   ├── client.py           # ChromaDB client
│   └── embedding.py        # SentenceTransformer embedding function
├── interfaces/             # Protocols that infra implements
│   ├── book_repository_interface.py
│   ├── client_interface.py
│   └── embedding_interface.py
└── books_advanced_search.py  # main() only — entry point
```

**Key design decisions:**

Domain logic (serialisation) lives on the `Book` dataclass itself:
```python
@dataclass
class Book:
    ...
    def to_document(self) -> str:
        ...
    def to_metadata(self) -> dict:
        ...
```

Interfaces use `Protocol` (structural typing) not `ABC` (nominal typing):
```python
from typing import Protocol

class BookSearchServiceInterface(Protocol):
    def search_by_similarity(self, query_text: str, n_results: int) -> None: ...
```

`Protocol` is closer to a true interface — no explicit inheritance required. 
A class satisfies the protocol if it has the right methods, regardless of 
its inheritance chain.

Composition root wires concretes without the entry point knowing about them:
```python
# container.py
def build_search_service(collection):
    return BookSearchService(collection)

# books_advanced_search.py — main() only references the interface
service: BookSearchServiceInterface = build_search_service(collection)
```

**Why `Protocol` over `ABC`:**

| Capability | ABC | Protocol |
|---|---|---|
| Runtime checking | ✓ | ✓ |
| Static checking | ✓ | ✓ |
| Implicit interface (no inheritance) | 0.5 | ✓ |
| Callback interface | ✗ | ✓ |

Protocol wins for clean Onion architecture — infra doesn't need to 
explicitly inherit from the interface it implements.

---

## Files

**Lab (IBM course):**
- `similarity_employeedata.py` — employee search implementation
- `employees.py` — extracted employee data module

**Free-hand extension:**
- `books_search/` — full Onion architecture books search project
- `books_search/books_advanced_search.py` — entry point

---

## Test Suite

31 tests across all layers, all passing. Run with:

```powershell
pytest tests/ -v
```

**Coverage by layer:**

| Layer | File | Tests | What's covered |
|---|---|---|---|
| Domain | `test_book.py` | 3 | Book creation, `to_document()`, `to_metadata()` |
| Application | `test_book_document_builder.py` | 5 | Document and metadata builder functions, ID exclusion |
| Application | `test_book_loader.py` | 2 | List type, length, first book field values |
| Application | `test_book_search_service.py` | 12 | All search methods — happy paths and empty result unhappy paths |
| Infra | `test_book_repository.py` | 4 | Collection creation config, add books, duplicate name, empty list guard |
| Integration | `test_books_advanced_search.py` | 5 | Full pipeline — document count, similarity search, genre filter, rating filter, combined search |

**Key patterns used:**

- `@pytest.fixture` for shared test data — equivalent of `[SetUp]` in NUnit
- `Mock()` from `unittest.mock` for isolating infra dependencies
- `yield` fixture for integration test teardown — creates and deletes ChromaDB collection per test
- `side_effect` for testing exception paths
- `assert_called_once_with` for verifying exact call signatures
- `assert_not_called` for verifying guards (empty list check in repository)

**Note:** Integration tests require `chromadb` and `sentence-transformers` installed 
in the venv. Not suitable for CI without those dependencies available.