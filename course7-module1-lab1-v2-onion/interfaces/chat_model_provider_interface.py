from typing import Protocol

from langchain_core.language_models import BaseChatModel


class ChatModelProviderInterface(Protocol):
    """Provider seam for chat models.

    Concretes wrap framework constructors (Ollama, OpenAI, future Anthropic
    or others) and hand back a configured `BaseChatModel`. Application code
    calls `.create()` once at composition time and works with the returned
    model directly.

    Narrowing `BaseMessage.content` from `str | list` to `str` is *not* the
    provider's job — it lives in `application/llm_text.py` so the seam
    stays focused on construction. Same shape as Lab 26's SQL Agent.
    """

    def create(self) -> BaseChatModel: ...