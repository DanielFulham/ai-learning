from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


class OpenAIChatModelProvider:
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature

    def create(self) -> BaseChatModel:
        return init_chat_model(
            self._model_name,
            model_provider="openai",
            temperature=self._temperature,
        )