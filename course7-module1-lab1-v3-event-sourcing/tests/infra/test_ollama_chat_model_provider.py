from langchain_core.language_models import BaseChatModel

from infra.ollama_chat_model_provider import OllamaChatModelProvider
from interfaces.chat_model_provider_interface import ChatModelProviderInterface


def _accepts_provider(provider: ChatModelProviderInterface) -> None:
    """Type-guard helper."""


class TestOllamaChatModelProvider:

    def test_satisfies_chat_model_provider_interface(self) -> None:
        _accepts_provider(OllamaChatModelProvider())

    def test_create_returns_base_chat_model(self) -> None:
        """Pinned: create() returns a BaseChatModel instance. Construction
        is lazy enough that no Ollama server is required to call this —
        the ChatOllama instance is built but doesn't connect until invoked.
        """
        provider = OllamaChatModelProvider()
        model = provider.create()
        assert isinstance(model, BaseChatModel)

    def test_custom_model_name_and_temperature_preserved(self) -> None:
        provider = OllamaChatModelProvider(
            model_name="custom-model", temperature=0.5
        )
        assert provider._model_name == "custom-model"
        assert provider._temperature == 0.5