"""OpenAI implementation of ChatModelProviderInterface."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


class OpenAIChatModelProvider:
    """Builds a LangChain chat model backed by OpenAI."""

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        self._model = model

    def create(self) -> BaseChatModel:
        return init_chat_model(self._model, model_provider="openai")