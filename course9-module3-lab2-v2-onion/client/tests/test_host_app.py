import asyncio
from unittest.mock import MagicMock

import gradio as gr

from application.interfaces.host_service_interface import HostServiceInterface
from domain.conversation import ChatReply, Message, ToolActivity
from host_app import build_interface, chat_turn

_EARLIER = (Message(role="user", text="TEST_EARLIER"),)


def _make_service(answer: ChatReply | None = None) -> MagicMock:
    service = MagicMock(spec=HostServiceInterface)
    service.chat.return_value = answer or ChatReply("TEST_ANSWER", history=_EARLIER)
    return service


def test_a_message_adds_the_question_and_the_answer_to_the_conversation() -> None:
    service = _make_service()
    shown: list[dict[str, str | None]] = [{"role": "user", "content": "TEST_EARLIER"}]
    new_shown, _ = asyncio.run(chat_turn(service, "TEST_QUERY", shown, _EARLIER))
    service.chat.assert_awaited_once_with("TEST_QUERY", _EARLIER)
    assert new_shown == [
        {"role": "user", "content": "TEST_EARLIER"},
        {"role": "user", "content": "TEST_QUERY"},
        {"role": "assistant", "content": "TEST_ANSWER"},
    ]


def test_the_page_keeps_the_history_the_service_returns() -> None:
    kept = (*_EARLIER, Message(role="assistant", text="TEST_ANSWER"))
    service = _make_service(ChatReply("TEST_ANSWER", history=kept))
    _, conversation = asyncio.run(chat_turn(service, "TEST_QUERY", [], _EARLIER))
    assert conversation == kept


def test_a_blank_message_changes_nothing_and_asks_nothing() -> None:
    service = _make_service()
    shown: list[dict[str, str | None]] = [{"role": "user", "content": "TEST_EARLIER"}]
    assert asyncio.run(chat_turn(service, "   ", shown, _EARLIER)) == (shown, _EARLIER)
    service.chat.assert_not_called()


def test_the_tools_the_model_used_are_shown_with_its_answer() -> None:
    answer = ChatReply("TEST_ANSWER", (ToolActivity("TEST_TOOL", "{}", "TEST_RESULT"),))
    shown, _ = asyncio.run(chat_turn(_make_service(answer), "TEST_QUERY", [], ()))
    content = shown[-1]["content"] or ""
    assert "TEST_TOOL" in content and "TEST_RESULT" in content
    assert content.endswith("TEST_ANSWER")


def test_the_page_builds_without_launching() -> None:
    page = build_interface(MagicMock(spec=HostServiceInterface), "http://TEST_HOST", "TEST_ROOTS")
    assert isinstance(page, gr.Blocks)
    assert page.title == "MCP HTTP AI Host"
