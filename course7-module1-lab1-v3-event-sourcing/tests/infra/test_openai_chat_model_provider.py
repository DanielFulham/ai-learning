import os
from unittest.mock import patch

from infra.openai_chat_model_provider import OpenAIChatModelProvider
from interfaces.chat_model_provider_interface import ChatModelProviderInterface


def _accepts_provider(provider: ChatModelProviderInterface) -> None:
    """Type-guard helper."""


class TestOpenAIChatModelProvider:

    def test_satisfies_chat_model_provider_interface(self) -> None:
        _accepts_provider(OpenAIChatModelProvider())

    def test_custom_model_name_and_temperature_preserved(self) -> None:
        provider = OpenAIChatModelProvider(
            model_name="gpt-4o", temperature=0.7
        )
        assert provider._model_name == "gpt-4o"
        assert provider._temperature == 0.7

    def test_create_with_dummy_api_key(self) -> None:
        """Pinned: create() works without a real API key for instantiation
        purposes — the OpenAI client validates lazily at request time.
        Pyright satisfaction and structural correctness are what this test
        proves; integration with the real API is out of scope for unit
        tests."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key-for-test"}):
            provider = OpenAIChatModelProvider()
            model = provider.create()
            assert model is not None