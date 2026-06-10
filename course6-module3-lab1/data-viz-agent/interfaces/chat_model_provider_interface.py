"""Protocol for LLM provider construction.

Infra implementations build the concrete chat model. Application code types
against this interface so the agent never knows which provider it's using.
The seam exists primarily for testability — tests inject a mock provider
and the OpenAI client is never instantiated, so no API key is needed.
"""

from typing import Protocol

from langchain_core.language_models import BaseChatModel


class ChatModelProviderInterface(Protocol):
    """A provider that constructs a configured BaseChatModel."""

    def create(self) -> BaseChatModel:
        """Construct and return a chat model ready for tool binding and invocation."""
        ...