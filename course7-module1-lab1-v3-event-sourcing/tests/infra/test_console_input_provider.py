from unittest.mock import patch

from infra.console_input_provider import ConsoleInputProvider
from interfaces.input_provider_interface import InputProviderInterface


def _accepts_input_provider(provider: InputProviderInterface) -> None:
    """Type-guard helper. Pyright fails if ConsoleInputProvider stops
    satisfying InputProviderInterface."""


class TestConsoleInputProviderInterfaceSatisfaction:

    def test_satisfies_input_provider_interface(self) -> None:
        """Pinned: ConsoleInputProvider is the production concrete for
        InputProviderInterface. Structural typing check via the guard."""
        provider = ConsoleInputProvider()
        _accepts_input_provider(provider)


class TestConsoleInputProviderPrompt:

    def test_prompt_calls_builtin_input_with_message(self) -> None:
        """Pinned: prompt() delegates to builtins.input with the message
        as the prompt string and returns whatever input() returns."""
        provider = ConsoleInputProvider()
        with patch("builtins.input", return_value="alice") as mock_input:
            result = provider.prompt("What is your username? ")

        mock_input.assert_called_once_with("What is your username? ")
        assert result == "alice"

    def test_prompt_returns_user_input(self) -> None:
        provider = ConsoleInputProvider()
        with patch("builtins.input", return_value="secure_password"):
            result = provider.prompt("Enter your password: ")

        assert result == "secure_password"
