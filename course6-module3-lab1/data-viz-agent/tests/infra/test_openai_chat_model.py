"""Tests for infra.openai_chat_model.

OpenAIChatModelProvider wraps langchain's init_chat_model dispatcher. Tests
verify it stores constructor arguments correctly and passes them through
on .create() — without ever instantiating a real OpenAI client.

The init_chat_model symbol is patched at the point of use (the infra module's
namespace), so no API key is needed and no network call is made.
"""

from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel

from infra.openai_chat_model import OpenAIChatModelProvider


def test_create_calls_init_chat_model_with_default_args() -> None:
    fake_model = MagicMock(spec=BaseChatModel)

    with patch("infra.openai_chat_model.init_chat_model", return_value=fake_model) as init:
        provider = OpenAIChatModelProvider()
        result = provider.create()

    init.assert_called_once_with(
        "gpt-4.1-mini",
        model_provider="openai",
        temperature=0.0,
    )
    assert result is fake_model


def test_create_passes_custom_model_name() -> None:
    fake_model = MagicMock(spec=BaseChatModel)

    with patch("infra.openai_chat_model.init_chat_model", return_value=fake_model) as init:
        provider = OpenAIChatModelProvider(model_name="gpt-5-mini")
        provider.create()

    init.assert_called_once_with(
        "gpt-5-mini",
        model_provider="openai",
        temperature=0.0,
    )


def test_create_passes_custom_temperature() -> None:
    fake_model = MagicMock(spec=BaseChatModel)

    with patch("infra.openai_chat_model.init_chat_model", return_value=fake_model) as init:
        provider = OpenAIChatModelProvider(temperature=0.7)
        provider.create()

    init.assert_called_once_with(
        "gpt-4.1-mini",
        model_provider="openai",
        temperature=0.7,
    )


def test_create_can_be_called_multiple_times() -> None:
    """Each call to create() builds a fresh chat model. The provider itself
    is stateless beyond its construction arguments."""
    fake_model = MagicMock(spec=BaseChatModel)

    with patch("infra.openai_chat_model.init_chat_model", return_value=fake_model) as init:
        provider = OpenAIChatModelProvider()
        provider.create()
        provider.create()
        provider.create()

    assert init.call_count == 3


def test_provider_satisfies_interface_protocol() -> None:
    """Structural check: an OpenAIChatModelProvider is a valid
    ChatModelProviderInterface (has a .create() method returning a BaseChatModel).
    This is enforced by Protocol structural typing; the test makes the
    relationship explicit and runnable."""
    from interfaces.chat_model_provider_interface import ChatModelProviderInterface

    provider: ChatModelProviderInterface = OpenAIChatModelProvider()
    assert hasattr(provider, "create")
    assert callable(provider.create)