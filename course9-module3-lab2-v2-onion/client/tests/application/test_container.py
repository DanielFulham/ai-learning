import asyncio
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from application.container import initialise_gui, initialise_host
from config import DEFAULT_SERVER, MODEL_ID, SERVER_COMMAND, STDIO_SERVER_ARGS
from interfaces.llm_chat_model_interface import LlmChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface


def _patch_infra(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Replace the transports, the session class and the model class with mocks."""
    mocks = {
        "http": MagicMock(),
        "stdio": MagicMock(),
        "session": MagicMock(),
        "model": MagicMock(),
    }
    monkeypatch.setattr("application.container.HttpTransport", mocks["http"])
    monkeypatch.setattr("application.container.StdioTransport", mocks["stdio"])
    monkeypatch.setattr("application.container.McpSdkSession", mocks["session"])
    monkeypatch.setattr("application.container.OpenAiChatModel", mocks["model"])
    return mocks


@pytest.mark.parametrize("build", [initialise_gui, initialise_host])
@pytest.mark.parametrize(
    ("server", "url"),
    [
        ("http://TEST_HOST:1", "http://TEST_HOST:1/mcp"),
        ("http://TEST_HOST:1/", "http://TEST_HOST:1/mcp"),
        ("http://TEST_HOST:1/mcp", "http://TEST_HOST:1/mcp"),
        ("https://TEST_HOST", "https://TEST_HOST/mcp"),
    ],
)
def test_an_http_address_uses_the_http_transport_with_the_mcp_endpoint(
    monkeypatch: pytest.MonkeyPatch, build: Callable[[str], object], server: str, url: str
) -> None:
    mocks = _patch_infra(monkeypatch)
    build(server)
    mocks["http"].assert_called_once_with(url)
    mocks["stdio"].assert_not_called()
    mocks["session"].assert_called_once_with(mocks["http"].return_value)


def test_a_script_path_uses_the_stdio_transport_with_the_configured_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _patch_infra(monkeypatch)
    initialise_gui("TEST_SERVER_SCRIPT.py")
    mocks["stdio"].assert_called_once_with(
        SERVER_COMMAND, ["TEST_SERVER_SCRIPT.py", *STDIO_SERVER_ARGS]
    )
    mocks["http"].assert_not_called()
    mocks["session"].assert_called_once_with(mocks["stdio"].return_value)


def test_with_no_server_given_the_lab_servers_address_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _patch_infra(monkeypatch)
    initialise_host()
    mocks["http"].assert_called_once_with(DEFAULT_SERVER + "/mcp")


def test_an_injected_session_wins_over_the_server_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _patch_infra(monkeypatch)
    initialise_gui("http://TEST_HOST", session=MagicMock(spec=McpSessionInterface))
    initialise_host(
        "http://TEST_HOST",
        session=MagicMock(spec=McpSessionInterface),
        chat_model=MagicMock(spec=LlmChatModelInterface),
    )
    mocks["http"].assert_not_called()
    mocks["session"].assert_not_called()


def test_the_gui_never_builds_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    mocks = _patch_infra(monkeypatch)
    initialise_gui("http://TEST_HOST")
    mocks["model"].assert_not_called()


def test_the_host_builds_the_configured_model_unless_one_is_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mocks = _patch_infra(monkeypatch)
    initialise_host("http://TEST_HOST")
    mocks["model"].assert_called_once_with(MODEL_ID)
    mocks["model"].reset_mock()
    initialise_host("http://TEST_HOST", chat_model=MagicMock(spec=LlmChatModelInterface))
    mocks["model"].assert_not_called()


def test_each_call_builds_a_new_service() -> None:
    assert initialise_gui("http://TEST_HOST") is not initialise_gui("http://TEST_HOST")
    assert initialise_host("http://TEST_HOST") is not initialise_host("http://TEST_HOST")


def test_building_either_service_needs_no_server_and_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert hasattr(initialise_gui("http://TEST_HOST:1"), "list_tools")
    assert hasattr(initialise_host("http://TEST_HOST:1"), "chat")


def test_the_gui_service_is_wired_to_the_injected_session() -> None:
    session = MagicMock(spec=McpSessionInterface)
    session.list_tools.return_value = [
        {"name": "TEST_TOOL", "description": "D", "input_schema": {}}
    ]
    service = initialise_gui(session=session)
    assert asyncio.run(service.list_tools()) == ("• TEST_TOOL: D", ["TEST_TOOL"])
    session.connect.assert_awaited_once()
