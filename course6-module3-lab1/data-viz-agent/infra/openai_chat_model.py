"""OpenAI implementation of ChatModelProviderInterface.

Constructs a configured chat model via LangChain's init_chat_model dispatcher.
This is the only file in the application that names a concrete LLM provider.

Configuration (model name, temperature, etc.) lives here, not in application
code. Swapping providers means writing a sibling file and changing one line
in container.py.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


class OpenAIChatModelProvider:
    """Constructs a configured OpenAI chat model."""

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature

    def create(self) -> BaseChatModel:
        """Build and return the configured BaseChatModel."""
        return init_chat_model(
            self._model_name,
            model_provider="openai",
            temperature=self._temperature,
        )