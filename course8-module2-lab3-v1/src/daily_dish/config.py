"""Environment configuration for the Daily Dish lab.

Handles .env loading with fail-loud checks on ANTHROPIC_API_KEY and
SERPER_API_KEY. Sets CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true to bypass
crewai-tools' path-validation sandbox (see F-M4 — data lives outside
process cwd under V1's run shape; correct-by-construction fix deferred
as L36-2). Suppresses CREWAI_TRACING_ENABLED to skip the 20-second
interactive tracing prompt after every kickoff. Exports build_llm() as
the LLM factory for all three approach scripts.
"""

from __future__ import annotations

import os
from pathlib import Path

from crewai import LLM
from dotenv import load_dotenv

os.environ.setdefault("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY missing from environment. "
        f"Expected in {_ENV_PATH} — copy .env.example and populate."
    )
if not SERPER_API_KEY:
    raise RuntimeError(
        "SERPER_API_KEY missing from environment. "
        f"Expected in {_ENV_PATH} — copy .env.example and populate."
    )


def build_llm() -> LLM:
    """Construct the shared LLM instance for all agents in this lab."""
    return LLM(
        model="anthropic/claude-haiku-4-5",
        temperature=0.0,
    )
