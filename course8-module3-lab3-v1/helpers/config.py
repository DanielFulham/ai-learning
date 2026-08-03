"""Provider seam. Carried forward from L39's `course8-module3-lab2-v1/helpers/config.py`.

Reconstructed here from the installed package rather than copied — replace
this file with L39's if it has diverged.

`prompt_caching` defaults True on `AnthropicConfig` and
`anthropic_client._inject_cache_control` writes ephemeral markers
(F-L39-4). Left at the default so the probe's cache columns carry
whatever the provider actually reports.
"""

import os

from ag2.config import AnthropicConfig
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1024


def anthropic_config() -> AnthropicConfig:
    """Build the shared model config. Fails loud on a missing key."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and populate it, "
            "or run with --offline to exercise the wiring without a provider."
        )
    return AnthropicConfig(model=MODEL, api_key=api_key, max_tokens=MAX_TOKENS)
