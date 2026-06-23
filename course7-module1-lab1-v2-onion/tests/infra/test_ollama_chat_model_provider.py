from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from infra.ollama_chat_model_provider import OllamaChatModelProvider


def test_create_returns_chat_ollama_instance() -> None:
    provider = OllamaChatModelProvider()
    model = provider.create()
    assert isinstance(model, ChatOllama)


def test_create_returns_base_chat_model() -> None:
    provider = OllamaChatModelProvider()
    model = provider.create()
    assert isinstance(model, BaseChatModel)


def test_defaults_to_llama32_at_temperature_zero() -> None:
    with patch("infra.ollama_chat_model_provider.ChatOllama") as mock_chat_ollama:
        provider = OllamaChatModelProvider()
        provider.create()
        mock_chat_ollama.assert_called_once_with(
            model="llama3.2:latest",
            temperature=0.0,
        )


def test_passes_custom_model_and_temperature_through() -> None:
    with patch("infra.ollama_chat_model_provider.ChatOllama") as mock_chat_ollama:
        provider = OllamaChatModelProvider(model_name="qwen2.5:7b", temperature=0.7)
        provider.create()
        mock_chat_ollama.assert_called_once_with(
            model="qwen2.5:7b",
            temperature=0.7,
        )