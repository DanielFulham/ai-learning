from unittest.mock import patch, MagicMock

from langchain_core.language_models import BaseChatModel

from infra.openai_chat_model import OpenAIChatModelProvider


@patch("infra.openai_chat_model.init_chat_model")
def test_create_calls_init_chat_model_with_default_arguments(mock_init):
    mock_init.return_value = MagicMock(spec=BaseChatModel)
    provider = OpenAIChatModelProvider()

    provider.create()

    mock_init.assert_called_once_with(
        "gpt-4.1-mini",
        model_provider="openai",
        temperature=0.0,
    )


@patch("infra.openai_chat_model.init_chat_model")
def test_create_passes_custom_model_name(mock_init):
    mock_init.return_value = MagicMock(spec=BaseChatModel)
    provider = OpenAIChatModelProvider(model_name="gpt-4o-mini")

    provider.create()

    mock_init.assert_called_once_with(
        "gpt-4o-mini",
        model_provider="openai",
        temperature=0.0,
    )


@patch("infra.openai_chat_model.init_chat_model")
def test_create_passes_custom_temperature(mock_init):
    mock_init.return_value = MagicMock(spec=BaseChatModel)
    provider = OpenAIChatModelProvider(temperature=0.7)

    provider.create()

    mock_init.assert_called_once_with(
        "gpt-4.1-mini",
        model_provider="openai",
        temperature=0.7,
    )


@patch("infra.openai_chat_model.init_chat_model")
def test_create_returns_the_init_chat_model_result(mock_init):
    fake_model = MagicMock(spec=BaseChatModel)
    mock_init.return_value = fake_model
    provider = OpenAIChatModelProvider()

    result = provider.create()

    assert result is fake_model


@patch("infra.openai_chat_model.init_chat_model")
def test_create_can_be_called_multiple_times(mock_init):
    mock_init.return_value = MagicMock(spec=BaseChatModel)
    provider = OpenAIChatModelProvider()

    provider.create()
    provider.create()

    assert mock_init.call_count == 2