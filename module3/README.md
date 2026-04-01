# Module 3: Build a Generative AI Application with LangChain

## Local Setup

- Python 3.11 (via `py -3.11`)
- Virtual environment: `venv/` (scoped to this module)
- Flask for local API serving

## Project Structure
```
module3/
  app.py                  # Application layer — Flask entry point (thin, just routing)
  domain/
    __init__.py
    chat.py               # Domain logic
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
  .env                    # API keys — never committed
  requirements.txt        # Pinned dependencies
```

## Architecture Notes

Follows the same layered pattern as TypeScript Lambda work:
- `app.py` is the entry point — thin, just wires middleware and routes
- Domain logic lives in `domain/` — no Flask or infrastructure imports here
- Infrastructure concerns (LLM client, vector store) live in `infrastructure/`
- Middleware registered in `app.py` via `register_middleware(app)`

## Running Locally
```bash
# Activate venv
venv\Scripts\activate

# Run Flask dev server
python app.py
```