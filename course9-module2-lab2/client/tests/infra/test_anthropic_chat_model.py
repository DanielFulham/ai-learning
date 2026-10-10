import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from anthropic import AsyncAnthropic

from domain.conversation import Message, ModelReply, ToolCall, ToolResult, ToolSpec
from infra.anthropic_chat_model import AnthropicChatModel

TEST_TOOLS = [ToolSpec("TEST_TOOL", "TEST_DESC", {"type": "object"})]


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(call_id: str, name: str, tool_input: object) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input)


def _make_model(stop_reason: str, *blocks: SimpleNamespace) -> tuple[AnthropicChatModel, AsyncMock]:
    create = AsyncMock(return_value=SimpleNamespace(content=list(blocks), stop_reason=stop_reason))
    client = MagicMock(spec=AsyncAnthropic)
    client.messages = SimpleNamespace(create=create)
    return AnthropicChatModel("TEST_MODEL", 123, client), create


def _complete(model: AnthropicChatModel, messages: list[Message]) -> ModelReply:
    return asyncio.run(model.complete(messages, TEST_TOOLS))


def test_sends_model_limits_tools_and_messages() -> None:
    model, create = _make_model("end_turn", _text("hi"))
    _complete(model, [Message(role="user", text="TEST_QUERY")])
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "TEST_MODEL"
    assert kwargs["max_tokens"] == 123
    assert kwargs["messages"] == [{"role": "user", "content": "TEST_QUERY"}]
    assert kwargs["tools"] == [
        {"name": "TEST_TOOL", "description": "TEST_DESC", "input_schema": {"type": "object"}}
    ]


def test_text_blocks_are_joined() -> None:
    model, _ = _make_model("end_turn", _text("A"), _text("B"))
    reply = _complete(model, [Message(role="user", text="q")])
    assert (reply.text, reply.tool_calls, reply.truncated) == ("A\nB", (), False)


def test_tool_use_becomes_tool_calls() -> None:
    model, _ = _make_model("tool_use", _text("T"), _tool_use("id1", "TEST_TOOL", {"x": 1}))
    reply = _complete(model, [Message(role="user", text="q")])
    assert reply.text == "T"
    assert reply.tool_calls == (ToolCall("id1", "TEST_TOOL", {"x": 1}),)


def test_tool_input_that_is_not_an_object_becomes_empty_arguments() -> None:
    model, _ = _make_model("tool_use", _tool_use("id1", "TEST_TOOL", "oops"))
    reply = _complete(model, [Message(role="user", text="q")])
    assert reply.tool_calls == (ToolCall("id1", "TEST_TOOL", {}),)


def test_tool_calls_in_a_truncated_reply_are_dropped() -> None:
    model, _ = _make_model("max_tokens", _text("T"), _tool_use("id1", "TEST_TOOL", {"x": 1}))
    reply = _complete(model, [Message(role="user", text="q")])
    assert reply.tool_calls == ()
    assert reply.truncated is True


def test_assistant_and_tool_result_messages_are_converted() -> None:
    model, create = _make_model("end_turn", _text("ok"))
    call = ToolCall("id1", "TEST_TOOL", {"x": 1})
    messages = [
        Message(role="user", text="q"),
        Message(role="assistant", text="T", tool_calls=(call,)),
        Message(role="assistant", tool_calls=(call,)),
        Message(
            role="user",
            tool_results=(ToolResult("id1", "OUT", False), ToolResult("id2", "BAD", True)),
        ),
    ]
    _complete(model, messages)
    sent = create.call_args.kwargs["messages"]
    tool_use = {"type": "tool_use", "id": "id1", "name": "TEST_TOOL", "input": {"x": 1}}
    assert sent[1] == {"role": "assistant", "content": [{"type": "text", "text": "T"}, tool_use]}
    assert sent[2] == {"role": "assistant", "content": [tool_use]}
    assert sent[3] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "id1", "content": "OUT"},
            {"type": "tool_result", "tool_use_id": "id2", "content": "BAD", "is_error": True},
        ],
    }
