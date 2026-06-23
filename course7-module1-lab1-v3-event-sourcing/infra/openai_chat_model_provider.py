from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


class OpenAIChatModelProvider:
    """OpenAI provider — cloud model. Requires OPENAI_API_KEY in env.

    Defaults to `gpt-4o-mini` at `temperature=0.0`. Construction lazily
    validates the API key when `create()` is called and the underlying
    client is initialised; the provider itself does not eager-check.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature

    def create(self) -> BaseChatModel:
        return ChatOpenAI(
            model=self._model_name,
            temperature=self._temperature,
        )