from typing import Protocol

from langchain_core.language_models import BaseChatModel


class ChatModelProviderInterface(Protocol):
    """Provider seam for chat models.

    Concretes wrap framework constructors and hand back a configured
    `BaseChatModel`. Application code calls `.create()` once at composition
    time and works with the returned model directly.
    """

    def create(self) -> BaseChatModel: ...
