from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from interfaces.chat_model_provider_interface import ChatModelProviderInterface


class OpenAIChatModelProvider(ChatModelProviderInterface):
    """Concrete provider for OpenAI chat models via langchain.chat_models.init_chat_model."""

    def __init__(self, model_name: str = "gpt-4.1-nano") -> None:
        self._model_name = model_name

    def create(self) -> BaseChatModel:
        return init_chat_model(self._model_name, model_provider="openai")