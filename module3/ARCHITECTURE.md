# Architecture Vision

This documents the target layered architecture for AI applications in this repo,
based on patterns from production TypeScript Lambda work.

The lab exercises start simple (single file, flat structure) and will be refactored
toward this pattern as complexity grows — particularly from Course 2 (RAG) onwards.

## Target Structure
```
app/
  app.py                  # Application layer — Flask/Lambda entry point (thin, just routing)
  domain/
    __init__.py
    chat.py               # Domain logic, business rules
    retrieval.py          # Retrieval domain logic
  infrastructure/
    __init__.py
    vector_store.py       # Vector store (ChromaDB)
    llm_client.py         # LangChain model wrapper
  middleware/
    __init__.py
    logging.py            # Request/response logging
  models/
    __init__.py
    request.py            # Input shapes
    response.py           # Output shapes
```

## Layer Responsibilities

- **Application layer** (`app.py`) — entry point only, wires middleware and routes
- **Domain layer** — business logic, no infrastructure imports
- **Infrastructure layer** — LLM clients, vector stores, external APIs
- **Middleware** — logging, error handling, registered in app.py

## C# / TypeScript Parallel

| Pattern | TypeScript Lambda | Python Flask/Lambda |
|---|---|---|
| Entry point | `app.ts` | `app.py` |
| Domain logic | `domain/` | `domain/` |
| Data access | `repository/` | `infrastructure/` |
| Cross-cutting | `middleware/` | `middleware/` |
| Input/output shapes | interfaces | Pydantic models |

## When to Apply This Structure

- Module 3 lab: flat structure acceptable (single concern, learning exercise)
- Course 2 RAG work: introduce `infrastructure/` for vector store
- Rovers project: full layered structure from day one