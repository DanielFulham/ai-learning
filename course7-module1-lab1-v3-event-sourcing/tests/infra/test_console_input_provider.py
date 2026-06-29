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


class TestConsoleInputProviderPromptSecret:

    def test_prompt_secret_calls_getpass_with_message(self) -> None:
        """Pinned: prompt_secret() delegates to getpass.getpass with the
        message as the prompt string and returns its result. Patched at the
        import site (infra.console_input_provider.getpass), not the source."""
        provider = ConsoleInputProvider()
        with patch(
            "infra.console_input_provider.getpass.getpass",
            return_value="secure_password",
        ) as mock_getpass:
            result = provider.prompt_secret("Enter your password: ")

        mock_getpass.assert_called_once_with("Enter your password: ")
        assert result == "secure_password"

    def test_prompt_secret_returns_empty_string_without_validation(self) -> None:
        """Pinned: getpass.getpass returns '' when the user hits enter
        without typing. The concrete does not validate — it returns the
        empty string verbatim. Validation is the caller's concern."""
        provider = ConsoleInputProvider()
        with patch(
            "infra.console_input_provider.getpass.getpass", return_value=""
        ):
            result = provider.prompt_secret("Enter your password: ")

        assert result == ""
