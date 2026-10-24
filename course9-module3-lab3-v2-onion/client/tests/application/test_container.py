import asyncio
from unittest.mock import MagicMock

import pytest

from application.container import initialise_gui, initialise_host
from application.gui_service import GuiService
from application.host_service import HostService
from domain.conversation import ModelReply
from interfaces.approval_source_interface import ApprovalSourceInterface
from interfaces.llm_chat_model_interface import LlmChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface, RawTool


def _make_session() -> MagicMock:
    session = MagicMock(spec=McpSessionInterface)
    session.list_tools.return_value = [
        RawTool(name="TEST_TOOL", description="TEST_DESC", input_schema={})
    ]
    return session


def test_given_the_gui_container_when_it_builds_then_the_service_is_the_gui_service() -> None:
    assert isinstance(initialise_gui(session=_make_session()).service, GuiService)


def test_given_the_host_container_when_it_builds_then_the_service_is_the_host_service() -> None:
    app = initialise_host(session=_make_session(), chat_model=MagicMock(spec=LlmChatModelInterface))
    assert isinstance(app.service, HostService)


def test_given_an_injected_session_when_the_gui_service_lists_tools_then_it_uses_that_session() -> (
    None
):
    session = _make_session()
    text, names = asyncio.run(initialise_gui(session=session).service.list_tools())
    assert names == ["TEST_TOOL"]
    session.connect.assert_awaited_once()


def test_given_an_injected_session_when_the_host_runs_then_it_uses_that_session_and_model() -> None:
    session = _make_session()
    model = MagicMock(spec=LlmChatModelInterface)
    model.complete.return_value = ModelReply(text="TEST_ANSWER")
    app = initialise_host(session=session, chat_model=model)
    desk = app.desks.desk_for("TEST_SESSION")
    reply = asyncio.run(app.service.chat("TEST", (), desk))
    assert reply.text == "TEST_ANSWER"
    session.list_tools.assert_awaited()


def test_given_no_api_key_when_the_host_is_built_then_building_it_does_not_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI client raises without a key, so the model makes it on the first request."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    initialise_host(session=_make_session())


def test_given_two_calls_when_the_container_builds_then_each_is_a_new_instance() -> None:
    """Pins statelessness: the entry point decides what to keep, as the onion doc says."""
    session = _make_session()
    first, second = initialise_gui(session=session), initialise_gui(session=session)
    assert first is not second
    assert first.service is not second.service
    assert first.desks is not second.desks


def test_given_the_gui_app_when_two_sessions_ask_for_desks_then_they_get_different_ones() -> None:
    """Pins F-L47-18 at the wiring level: the desks come from a per-session registry."""
    app = initialise_gui(session=_make_session())
    first: ApprovalSourceInterface = app.desks.desk_for("TEST_SESSION_A")
    second: ApprovalSourceInterface = app.desks.desk_for("TEST_SESSION_B")
    assert first is not second
