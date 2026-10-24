import asyncio
from unittest.mock import MagicMock

import gradio as gr

from application.approval_desks import ApprovalDesks
from application.interfaces.approval_desk_interface import ApprovalDeskInterface
from application.interfaces.host_service_interface import HostServiceInterface
from domain.conversation import ChatReply, Message, ToolActivity
from host_app import build_interface, chat_turn

_EARLIER = (Message(role="user", text="TEST_EARLIER"),)
Shown = list[dict[str, str | None]]
Step = tuple[Shown, tuple[Message, ...]]


def _make_desk(pending: str | None = None) -> MagicMock:
    desk = MagicMock(spec=ApprovalDeskInterface)
    desk.pending_message.return_value = pending
    return desk


def _make_service(answer: ChatReply | None = None) -> MagicMock:
    service = MagicMock(spec=HostServiceInterface)
    service.chat.return_value = answer or ChatReply("TEST_ANSWER", history=_EARLIER)
    return service


def _run_turn(
    service: MagicMock,
    desk: MagicMock,
    message: str,
    shown: Shown | None = None,
    conversation: tuple[Message, ...] = _EARLIER,
) -> list[Step]:
    async def collect() -> list[Step]:
        return [
            step
            async for step in chat_turn(
                service, desk, message, shown or [], conversation, poll_seconds=0.01
            )
        ]

    return asyncio.run(collect())


def test_given_a_message_when_a_turn_runs_then_the_question_appears_first_then_the_answer() -> None:
    steps = _run_turn(_make_service(), _make_desk(), "TEST_QUERY")
    assert [len(shown) for shown, _ in steps] == [1, 2]
    assert steps[0][0] == [{"role": "user", "content": "TEST_QUERY"}]
    assert steps[-1][0][-1] == {"role": "assistant", "content": "TEST_ANSWER"}


def test_given_a_message_when_a_turn_runs_then_the_service_gets_the_message_history_and_desk() -> (
    None
):
    service, desk = _make_service(), _make_desk()
    _run_turn(service, desk, "TEST_QUERY")
    service.chat.assert_awaited_once_with("TEST_QUERY", _EARLIER, desk)


def test_given_a_service_that_returns_history_when_the_turn_ends_then_the_page_keeps_it() -> None:
    kept = (*_EARLIER, Message(role="assistant", text="TEST_ANSWER"))
    service = _make_service(ChatReply("TEST_ANSWER", history=kept))
    assert _run_turn(service, _make_desk(), "TEST_QUERY")[-1][1] == kept


def test_given_a_blank_message_when_a_turn_runs_then_nothing_changes_and_nothing_is_asked() -> None:
    service = _make_service()
    shown: Shown = [{"role": "user", "content": "TEST_EARLIER"}]
    assert _run_turn(service, _make_desk(), "   ", shown) == [(shown, _EARLIER)]
    service.chat.assert_not_called()


def test_given_tool_activity_when_the_turn_ends_then_the_tools_are_shown_with_the_answer() -> None:
    answer = ChatReply("TEST_ANSWER", (ToolActivity("TEST_TOOL", "{}", "TEST_RESULT"),))
    content = _run_turn(_make_service(answer), _make_desk(), "TEST")[-1][0][-1]["content"] or ""
    assert "TEST_TOOL" in content and "TEST_RESULT" in content
    assert content.endswith("TEST_ANSWER")


def test_given_the_server_asks_during_a_turn_then_the_exact_question_is_shown_to_the_user() -> None:
    """Pins F-L47-3 on the host page: the person sees the exact question before any "yes"."""
    desk = _make_desk()

    async def slow_chat(message: str, history: tuple[Message, ...], d: object) -> ChatReply:
        desk.pending_message.return_value = "TEST_QUESTION"
        await asyncio.sleep(0.05)
        desk.pending_message.return_value = None
        return ChatReply("TEST_ANSWER", history=_EARLIER)

    service = _make_service()
    service.chat.side_effect = slow_chat
    steps = _run_turn(service, desk, "TEST_QUERY")
    shown_texts = [item["content"] or "" for item in steps[-1][0]]
    assert any("TEST_QUESTION" in text and "yes or no" in text for text in shown_texts)
    assert shown_texts[-1] == "TEST_ANSWER"


def test_given_the_same_question_still_waiting_when_polled_again_then_it_is_shown_only_once() -> (
    None
):
    desk = _make_desk()

    async def slow_chat(message: str, history: tuple[Message, ...], d: object) -> ChatReply:
        desk.pending_message.return_value = "TEST_QUESTION"
        await asyncio.sleep(0.08)
        return ChatReply("TEST_ANSWER", history=_EARLIER)

    service = _make_service()
    service.chat.side_effect = slow_chat
    shown = _run_turn(service, desk, "TEST_QUERY")[-1][0]
    assert sum("TEST_QUESTION" in (item["content"] or "") for item in shown) == 1


def test_given_a_question_already_waiting_when_the_user_replies_then_it_is_answered_directly() -> (
    None
):
    """The reply to a waiting question goes straight to the service, with no second prompt shown."""
    service = _make_service(ChatReply("Sent your answer (accept).", history=_EARLIER))
    steps = _run_turn(service, _make_desk("TEST_QUESTION"), "yes")
    assert len(steps) == 1
    contents = [item["content"] for item in steps[0][0]]
    assert contents == ["yes", "Sent your answer (accept)."]


def test_given_a_host_service_when_the_page_is_built_then_it_builds_without_launching() -> None:
    page = build_interface(MagicMock(spec=HostServiceInterface), ApprovalDesks())
    assert isinstance(page, gr.Blocks)
    assert page.title == "MCP Permission AI Host"


def test_given_the_host_page_when_it_is_built_then_closing_the_page_is_handled() -> None:
    """Pins F-L47-27: the page registers an unload handler so a closed tab frees its desk."""
    page = build_interface(MagicMock(spec=HostServiceInterface), ApprovalDesks())
    assert any(("unload" in [t[1] for t in fn.targets]) for fn in page.fns.values())
