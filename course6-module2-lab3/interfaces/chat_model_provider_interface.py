from typing import Protocol

from langchain_core.language_models import BaseChatModel


class ChatModelProviderInterface(Protocol):
    """Provides a BaseChatModel instance. Allows the LLM provider to be swapped at the seam."""

    def create(self) -> BaseChatModel:
        ...