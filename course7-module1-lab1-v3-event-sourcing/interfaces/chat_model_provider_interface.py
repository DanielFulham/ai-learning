from typing import Protocol

from langchain_core.language_models import BaseChatModel


class ChatModelProviderInterface(Protocol):
    """Factory for a configured `BaseChatModel`.

    Two concretes in V3a:
    - `OllamaChatModelProvider` — local model via Ollama, no API key
    - `OpenAIChatModelProvider` — OpenAI cloud, requires OPENAI_API_KEY

    The provider's job is construction. The returned `BaseChatModel` is
    used by application-layer nodes that call it directly. Tool-binding
    extensions (`bind_tools(...)` on the adapter) land in Lab 28-30 work.
    """

    def create(self) -> BaseChatModel: ...