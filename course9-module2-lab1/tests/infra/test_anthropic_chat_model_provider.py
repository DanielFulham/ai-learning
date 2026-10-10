from unittest.mock import patch

from infra.anthropic_chat_model_provider import AnthropicChatModelProvider


def test_create_builds_the_default_haiku_model() -> None:
    with patch("infra.anthropic_chat_model_provider.ChatAnthropic") as model_class:
        model = AnthropicChatModelProvider().create()
        model_class.assert_called_once_with(model_name="claude-haiku-4-5", timeout=None, stop=None)
        assert model is model_class.return_value


def test_create_uses_the_model_name_it_was_given() -> None:
    with patch("infra.anthropic_chat_model_provider.ChatAnthropic") as model_class:
        AnthropicChatModelProvider(model_name="TEST_MODEL").create()
        assert model_class.call_args.kwargs["model_name"] == "TEST_MODEL"


def test_constructing_the_provider_does_not_build_a_model() -> None:
    """Construction is lazy, so tests and imports never need an API key."""
    with patch("infra.anthropic_chat_model_provider.ChatAnthropic") as model_class:
        AnthropicChatModelProvider()
        model_class.assert_not_called()
