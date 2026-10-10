from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel


class AnthropicChatModelProvider:
    """Anthropic-backed chat model provider.

    Requires `ANTHROPIC_API_KEY` in the environment. The key is checked when
    `create()` is called, not at import time, so tests that inject a mock
    provider never need one.
    """

    def __init__(self, model_name: str = "claude-haiku-4-5") -> None:
        self._model_name = model_name

    def create(self) -> BaseChatModel:
        return ChatAnthropic(model_name=self._model_name, timeout=None, stop=None)
