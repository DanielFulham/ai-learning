from unittest.mock import patch

from infra.console_input_provider import ConsoleInputProvider


def test_prompt_delegates_to_builtin_input() -> None:
    with patch("builtins.input", return_value="user_typed_this") as mock_input:
        provider = ConsoleInputProvider()
        result = provider.prompt("Enter your password: ")
        assert result == "user_typed_this"
        mock_input.assert_called_once_with("Enter your password: ")


def test_prompt_returns_exact_input_no_processing() -> None:
    """No strip, no transformation — the provider is a thin pass-through."""
    with patch("builtins.input", return_value="  spaces preserved  "):
        provider = ConsoleInputProvider()
        result = provider.prompt("> ")
        assert result == "  spaces preserved  "