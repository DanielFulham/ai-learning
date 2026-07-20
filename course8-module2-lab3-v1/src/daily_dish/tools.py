"""Tool factories for the Daily Dish lab.

Two tools shared across the three approach scripts:

- `build_serper_tool()`  — web search via Serper.dev (SerperDevTool)
- `build_pdf_search_tool()` — RAG over the Daily Dish FAQ PDF (PDFSearchTool)

Factory functions rather than module-scope instances so each script owns
its own tool instance (avoids any hidden state carry-over between runs
of different approaches in the same interpreter, and matches the
build_llm() factory shape from config.py).
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from crewai.rag.embeddings.providers.sentence_transformer.types import (
    SentenceTransformerProviderConfig,
    SentenceTransformerProviderSpec,
)
from crewai_tools import PDFSearchTool, SerperDevTool
from crewai_tools.tools.rag.types import RagToolConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_PDF_LOCAL_PATH = _DATA_DIR / "daily_dish_faq.pdf"
_PDF_SOURCE_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "7vgNfis17dQfjHAiIKkBOg/The-Daily-Dish-FAQ.pdf"
)


def build_serper_tool() -> SerperDevTool:
    """Construct a SerperDevTool for web search.

    Reads SERPER_API_KEY from environment (populated by config.py's
    dotenv load). Checked here rather than in config.py because
    task_centric.py never calls this function and shouldn't be forced
    to have a Serper key just to run. No arguments to SerperDevTool
    itself — it picks up the key from env automatically once present.
    """
    if not os.environ.get("SERPER_API_KEY"):
        raise RuntimeError(
            "SERPER_API_KEY missing from environment. Copy .env.example to .env and populate it."
        )
    return SerperDevTool()


def _ensure_faq_pdf() -> Path:
    """Download the Daily Dish FAQ PDF once, cache locally. Idempotent."""
    if _PDF_LOCAL_PATH.exists():
        return _PDF_LOCAL_PATH
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_PDF_SOURCE_URL, _PDF_LOCAL_PATH)
    return _PDF_LOCAL_PATH


def build_pdf_search_tool() -> PDFSearchTool:
    """Construct a PDFSearchTool over the Daily Dish FAQ PDF.

    Uses the `sentence-transformer` provider — local embeddings via the
    sentence_transformers package (pinned in requirements.txt),
    all-MiniLM-L6-v2 model. No API key required, no cross-provider mix
    with the Anthropic LLM.

    F-M2 finding: the current provider name for local sentence-transformers
    embeddings is `sentence-transformer` (hyphen, no 's'). CrewAI's
    `huggingface` provider now means the HuggingFace Inference API
    (remote, requires api_key). Type inspection of RagToolConfig on
    crewai-tools==1.15.2 confirmed both shape and semantics.

    First-run cost: ~90MB MiniLM model download to HuggingFace cache,
    plus the FAQ PDF download to data/. Cached thereafter.
    """
    pdf_path = _ensure_faq_pdf()
    embedding_config: SentenceTransformerProviderConfig = {
        "model_name": "all-MiniLM-L6-v2",
    }
    embedding_spec: SentenceTransformerProviderSpec = {
        "provider": "sentence-transformer",
        "config": embedding_config,
    }
    rag_config: RagToolConfig = {
        "embedding_model": embedding_spec,
    }
    return PDFSearchTool(
        pdf=str(pdf_path),
        config=rag_config,
    )
