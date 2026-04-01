# Module 3: Build a Generative AI Application with LangChain

## Local Setup

- Python 3.11 (via `py -3.11`)
- Virtual environment: `venv/` (scoped to this module)
- Flask for local API serving

## What This Module Builds

A Flask API with multiple LLM backends (Llama, Granite, Mistral) via IBM Watsonx,
using LangChain for prompt templating and structured JSON output via Pydantic models.

## Project Structure
```
module3/
  app.py                  # Flask entry point — routes /generate POST endpoint
  model.py                # LangChain chains — model init, prompt templates, response functions
  config.py               # Model IDs, Watsonx credentials, generation parameters
  llm_test.py             # Sanity check — calls all three models directly
  templates/
    index.html            # Chat UI
  static/
    script.js             # Frontend — calls /generate endpoint
    styles.css            # Styling
  .env                    # API keys — never committed
  requirements.txt        # Pinned dependencies
```

## Architecture Notes

- `app.py` is thin — just routing and error handling, imports from `model.py`
- `model.py` owns all AI logic — model initialisation, prompt templates, chains
- `config.py` centralises model IDs and parameters — swap models by changing one value
- Each model has its own prompt template to handle model-specific special tokens
- `JsonOutputParser` + Pydantic enforces structured output from all three models

## Key Learnings

- Each LLM family (Llama, Granite, Mistral) requires different special token formatting
- `{format_prompt}` must be injected into the system block of each template
- Parser returns a Python dict — not an AIMessage object — access fields directly
- Granite produced the cleanest structured output of the three models tested
- Model IDs in course materials go stale — always verify against supported models list

## Running Locally
```bash
# Activate venv
venv\Scripts\activate

# Run Flask dev server
python app.py

# Test models directly without Flask
python llm_test.py
```

## Planned Extension

Port to OpenAI locally (Watsonx credentials only available in IBM cloud environment).
Refactor toward layered architecture for Course 2 RAG work.