import asyncio
from collections.abc import Sequence
from unittest.mock import MagicMock

from application.chat_service import ChatService
from domain.conversation import Message, ModelReply, ToolCall, ToolResult, ToolSpec
from interfaces.chat_model_interface import ChatModelInterface
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface


def _make_call(call_id: str = "TEST_ID") -> ToolCall:
    return ToolCall(call_id, "TEST_TOOL", {"x": 1})


def _make_session() -> MagicMock:
    session = MagicMock(spec=McpSessionInterface)
    session.list_tools.return_value = [
        {"name": "TEST_TOOL", "description": "TEST_DESC", "input_schema": {}}
    ]
    session.call_tool.return_value = {"text": "TEST_OUTPUT", "is_error": False}
    return session


def _make_model(*replies: ModelReply) -> tuple[MagicMock, list[list[Message]]]:
    """A chat model that returns the replies in order and records what it was sent."""
    seen: list[list[Message]] = []
    queue = list(replies)

    async def complete(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ModelReply:
        seen.append(list(messages))
        return queue.pop(0)

    model = MagicMock(spec=ChatModelInterface)
    model.complete.side_effect = complete
    return model, seen


def _make_service(model: MagicMock, session: MagicMock) -> tuple[ChatService, MagicMock]:
    user_input = MagicMock(spec=UserInputInterface)
    return ChatService(model, session, user_input), user_input


def test_plain_answer_makes_one_model_call() -> None:
    model, seen = _make_model(ModelReply("TEST_ANSWER"))
    service, _ = _make_service(model, _make_session())
    assert asyncio.run(service.process_query("TEST_QUERY")) == "TEST_ANSWER"
    assert seen == [[Message(role="user", text="TEST_QUERY")]]


def test_tools_are_offered_to_the_model() -> None:
    model, _ = _make_model(ModelReply("TEST_ANSWER"))
    service, _ = _make_service(model, _make_session())
    asyncio.run(service.process_query("TEST_QUERY"))
    offered = model.complete.call_args.args[1]
    assert [t.name for t in offered] == ["TEST_TOOL"]


def test_tool_call_is_run_and_result_sent_back() -> None:
    call = _make_call()
    model, seen = _make_model(ModelReply("TEST_THINKING", (call,)), ModelReply("TEST_FINAL"))
    session = _make_session()
    service, _ = _make_service(model, session)

    assert asyncio.run(service.process_query("TEST_QUERY")) == "TEST_FINAL"

    session.call_tool.assert_called_once_with("TEST_TOOL", {"x": 1})
    second_call = seen[1]
    assert second_call[1] == Message(role="assistant", text="TEST_THINKING", tool_calls=(call,))
    assert second_call[2] == Message(
        role="user", tool_results=(ToolResult("TEST_ID", "TEST_OUTPUT", False),)
    )


def test_several_tool_calls_in_one_reply_are_all_run() -> None:
    calls = (_make_call("TEST_1"), _make_call("TEST_2"))
    model, seen = _make_model(ModelReply("", calls), ModelReply("TEST_FINAL"))
    session = _make_session()
    service, _ = _make_service(model, session)
    asyncio.run(service.process_query("TEST_QUERY"))
    assert session.call_tool.call_count == 2
    assert [r.tool_call_id for r in seen[1][2].tool_results] == ["TEST_1", "TEST_2"]


def test_tool_error_result_is_passed_on_as_an_error() -> None:
    model, seen = _make_model(ModelReply("", (_make_call(),)), ModelReply("TEST_FINAL"))
    session = _make_session()
    session.call_tool.return_value = {"text": "TEST_FAIL", "is_error": True}
    service, _ = _make_service(model, session)
    asyncio.run(service.process_query("TEST_QUERY"))
    assert seen[1][2].tool_results == (ToolResult("TEST_ID", "TEST_FAIL", True),)


def test_raised_tool_exception_becomes_an_error_result_and_is_shown() -> None:
    model, seen = _make_model(ModelReply("", (_make_call(),)), ModelReply("TEST_FINAL"))
    session = _make_session()
    session.call_tool.side_effect = RuntimeError("TEST_BOOM")
    service, user_input = _make_service(model, session)
    asyncio.run(service.process_query("TEST_QUERY"))
    assert seen[1][2].tool_results == (ToolResult("TEST_ID", "Error: TEST_BOOM", True),)
    user_input.show.assert_called_once_with("Error calling tool TEST_TOOL: TEST_BOOM")


def test_truncated_reply_shows_a_warning() -> None:
    model, _ = _make_model(ModelReply("TEST_PARTIAL", truncated=True))
    service, user_input = _make_service(model, _make_session())
    assert asyncio.run(service.process_query("TEST_QUERY")) == "TEST_PARTIAL"
    assert "max_tokens" in user_input.show.call_args.args[0]
