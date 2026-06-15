from typing import Protocol

from langchain_core.language_models import BaseChatModel


class ChatModelProviderInterface(Protocol):
    def create(self) -> BaseChatModel: ...