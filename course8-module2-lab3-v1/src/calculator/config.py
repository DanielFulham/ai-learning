"""LLM and environment configuration for the calculator lab.

Mirrors daily_dish/config.py in shape but scoped to what the calculator
actually needs — no SERPER_API_KEY (no web search), no sandbox flag
(no file I/O). CREWAI_TRACING_ENABLED still suppressed because the
interactive tracing prompt fires after every crew.kickoff() regardless
of what the crew does.
"""

from __future__ import annotations

import os
from pathlib import Path

from crewai import LLM
from dotenv import load_dotenv

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_ENV_PATH)

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise RuntimeError(
        "ANTHROPIC_API_KEY missing from environment. "
        f"Expected in {_ENV_PATH} — copy .env.example and populate."
    )


def build_llm() -> LLM:
    """Construct the LLM instance for the calculator agent."""
    return LLM(
        model="anthropic/claude-haiku-4-5",
        temperature=0.0,
    )
