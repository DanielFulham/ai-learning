import asyncio
from unittest.mock import MagicMock

import pytest

from application.container import initialise
from config import DEFAULT_SERVER_SCRIPT, SERVER_COMMAND
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface


def _patch_transports(monkeypatch: pytest.MonkeyPatch) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Replace the transports and the session class with mocks; return all three."""
    http, stdio, session_class = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr("application.container.HttpTransport", http)
    monkeypatch.setattr("application.container.StdioTransport", stdio)
    monkeypatch.setattr("application.container.OfficialMcpSession", session_class)
    return http, stdio, session_class


def test_injected_parts_are_wired_together() -> None:
    session = MagicMock(spec=McpSessionInterface)
    user_input = MagicMock(spec=UserInputInterface)
    user_input.ask.return_value = "quit"

    service = initialise(session=session, user_input=user_input)
    asyncio.run(service.run())

    session.connect.assert_awaited_once()
    session.close.assert_awaited_once()


@pytest.mark.parametrize("url", ["http://TEST_HOST:1/mcp", "https://TEST_HOST/mcp"])
def test_an_http_url_uses_the_http_transport(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    http, stdio, session_class = _patch_transports(monkeypatch)
    initialise(url)
    http.assert_called_once_with(url)
    stdio.assert_not_called()
    session_class.assert_called_once_with(http.return_value)


def test_a_script_path_uses_the_stdio_transport_with_the_configured_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http, stdio, session_class = _patch_transports(monkeypatch)
    initialise("TEST_SERVER_SCRIPT.py")
    stdio.assert_called_once_with(SERVER_COMMAND, ["TEST_SERVER_SCRIPT.py"])
    http.assert_not_called()
    session_class.assert_called_once_with(stdio.return_value)


def test_with_no_server_given_the_default_script_is_launched_over_stdio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http, stdio, _ = _patch_transports(monkeypatch)
    initialise()
    stdio.assert_called_once_with(SERVER_COMMAND, [DEFAULT_SERVER_SCRIPT])
    http.assert_not_called()


def test_an_injected_session_wins_over_the_server_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http, stdio, session_class = _patch_transports(monkeypatch)
    initialise("http://TEST_HOST/mcp", session=MagicMock(spec=McpSessionInterface))
    http.assert_not_called()
    stdio.assert_not_called()
    session_class.assert_not_called()


def test_each_call_builds_a_new_service() -> None:
    assert initialise("TEST_SERVER_SCRIPT.py") is not initialise("TEST_SERVER_SCRIPT.py")


def test_defaults_build_without_a_server_running() -> None:
    service = initialise("TEST_SERVER_SCRIPT.py")
    assert hasattr(service, "run")
