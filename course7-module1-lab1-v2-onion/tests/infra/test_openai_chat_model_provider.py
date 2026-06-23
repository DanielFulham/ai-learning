from unittest.mock import MagicMock, patch

from infra.openai_chat_model_provider import OpenAIChatModelProvider


def test_create_invokes_init_chat_model_with_defaults() -> None:
    with patch("infra.openai_chat_model_provider.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        provider = OpenAIChatModelProvider()
        provider.create()
        mock_init.assert_called_once_with(
            "gpt-4.1-mini",
            model_provider="openai",
            temperature=0.0,
        )


def test_create_invokes_init_chat_model_with_custom_args() -> None:
    with patch("infra.openai_chat_model_provider.init_chat_model") as mock_init:
        mock_init.return_value = MagicMock()
        provider = OpenAIChatModelProvider(model_name="gpt-4.1", temperature=0.5)
        provider.create()
        mock_init.assert_called_once_with(
            "gpt-4.1",
            model_provider="openai",
            temperature=0.5,
        )


def test_constructor_does_not_call_init_chat_model() -> None:
    """Construction is lazy — key validation happens on `.create()`, not __init__.

    This is what lets the test suite run without OPENAI_API_KEY set.
    """
    with patch("infra.openai_chat_model_provider.init_chat_model") as mock_init:
        OpenAIChatModelProvider()
        mock_init.assert_not_called()