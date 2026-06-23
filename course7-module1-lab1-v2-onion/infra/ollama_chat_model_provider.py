from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama


class OllamaChatModelProvider:
    """Local Ollama-backed chat model provider.

    Default concrete for V2. No API key needed — runs against a local
    Ollama server (`ollama serve`) with the model already pulled. Matches
    V1's `temperature=0` determinism finding so QA traces stay reproducible.
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