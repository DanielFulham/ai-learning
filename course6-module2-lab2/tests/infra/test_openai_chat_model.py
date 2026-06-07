from unittest.mock import patch, MagicMock

from langchain_core.language_models import BaseChatModel

from infra.openai_chat_model import OpenAIChatModelProvider


def test_provider_stores_default_model():
    provider = OpenAIChatModelProvider()
    assert provider._model == "gpt-4.1-nano"


def test_provider_stores_custom_model():
    provider = OpenAIChatModelProvider(model="gpt-4o-mini")
    assert provider._model == "gpt-4o-mini"


@patch("infra.openai_chat_model.init_chat_model")
def test_create_calls_init_chat_model_with_openai_provider(mock_init):
    mock_init.return_value = MagicMock(spec=BaseChatModel)
    provider = OpenAIChatModelProvider(model="gpt-4.1-nano")

    provider.create()

    mock_init.assert_called_once_with("gpt-4.1-nano", model_provider="openai")


@patch("infra.openai_chat_model.init_chat_model")
def test_create_returns_a_chat_model(mock_init):
    fake_llm = MagicMock(spec=BaseChatModel)
    mock_init.return_value = fake_llm
    provider = OpenAIChatModelProvider()

    result = provider.create()

    assert result is fake_llm