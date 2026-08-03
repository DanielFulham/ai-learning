"""Model configuration factory.

Concentrates the provider seam in one place. The notebook redefines an
``llm_config`` dict in six separate cells; this replaces all of them.

AG2 v1.0 uses typed provider configs (``AnthropicConfig``, ``OpenAIConfig``,
...) implementing a common ``ModelConfig`` protocol, rather than Classic's
untyped ``{"config_list": [{"model": ...}]}`` dict.
"""

from __future__ import annotations

import os

from ag2.config.anthropic import AnthropicConfig
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5"
"""Standardised across the cert (L34-L38) for cross-lab cost comparability."""


def build_config(
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    prompt_caching: bool = True,
) -> AnthropicConfig:
    """Build the Anthropic model config used by every script in this lab.

    ``prompt_caching`` defaults to True in ``AnthropicConfig`` itself; it is
    surfaced here explicitly so the caching probe (F-L39-4) can flip it
    without editing call sites.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and populate it."
        )

    return AnthropicConfig(
        model=MODEL,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt_caching=prompt_caching,
    )
