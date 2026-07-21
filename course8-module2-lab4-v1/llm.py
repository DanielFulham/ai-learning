"""Anthropic model plumbing shared across the NourishBot lab.

Two factories against the same model: `get_client()` returns a direct SDK
client for the vision-in-tool multimodal work in tools.py, and
`build_crewai_llm()` returns a CrewAI LLM wrapping the same model via its
native AnthropicCompletion provider. `MODEL` is the single source of
truth so the version can't drift between the two paths.

Auth: ANTHROPIC_API_KEY env var, loaded from .env at the entry point.
"""

from __future__ import annotations

from typing import Final

from anthropic import Anthropic
from crewai import LLM

MODEL: Final[str] = "claude-haiku-4-5"

# Private: only build_crewai_llm() needs the provider-prefixed form;
# direct-SDK callers use MODEL.
_CREWAI_MODEL: Final[str] = f"anthropic/{MODEL}"

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def build_crewai_llm() -> LLM:
    """The 'anthropic/' prefix triggers native AnthropicCompletion routing
    when the anthropic SDK is installed — L35 correction: crewai does NOT
    install anthropic transitively; requirements.txt pins it explicitly.

    Cheap to construct; not cached, each crew builder calls fresh.
    """
    return LLM(model=_CREWAI_MODEL)
