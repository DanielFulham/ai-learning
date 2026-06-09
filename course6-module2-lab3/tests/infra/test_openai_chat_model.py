from unittest.mock import MagicMock, patch

from infra.openai_chat_model import OpenAIChatModelProvider


def test_create_calls_init_chat_model_with_default_model() -> None:
    with patch("infra.openai_chat_model.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        provider = OpenAIChatModelProvider()

        provider.create()

        mock_init.assert_called_once_with("gpt-4.1-nano", model_provider="openai")


def test_create_passes_custom_model_name() -> None:
    with patch("infra.openai_chat_model.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        provider = OpenAIChatModelProvider(model_name="custom-model")

        provider.create()

        mock_init.assert_called_once_with("custom-model", model_provider="openai")


def test_create_returns_what_init_chat_model_returns() -> None:
    sentinel = MagicMock()
    with patch("infra.openai_chat_model.init_chat_model", return_value=sentinel):
        provider = OpenAIChatModelProvider()
        assert provider.create() is sentinel