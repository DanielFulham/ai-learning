from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama


class OllamaChatModelProvider:
    """Default provider — local model via Ollama, no API key required.

    Defaults to `llama3.2:latest` at `temperature=0.0` for deterministic
    output. Requires `ollama serve` running on the host; the provider
    does not health-check at construction.
    """

    def __init__(
        self,
        model_name: str = "llama3.2:latest",
        temperature: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._temperature = temperature

    def create(self) -> BaseChatModel:
        return ChatOllama(
            model=self._model_name,
            temperature=self._temperature,
        )