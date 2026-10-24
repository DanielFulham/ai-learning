import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI, Omit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
    Custom,
)
from openai.types.chat.chat_completion_message_function_tool_call import Function

from domain.conversation import Message, ModelReply, ToolCall
from domain.tool_info import ToolInfo
from infra.openai_chat_model import OpenAiChatModel
from interfaces.llm_chat_model_interface import LlmChatModelInterface

TEST_MODEL = "TEST_MODEL"


def _accepts_chat_model(model: LlmChatModelInterface) -> None:
    """Type guard: only compiles if the object satisfies the interface."""


def _completion(
    content: str | None = "TEST_TEXT",
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None = None,
    empty: bool = False,
) -> ChatCompletion:
    choices = []
    if not empty:
        message = ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)
        choices = [Choice(finish_reason="stop", index=0, message=message)]
    return ChatCompletion(
        id="TEST_ID", created=0, model=TEST_MODEL, object="chat.completion", choices=choices
    )


def _make_client(completion: ChatCompletion) -> MagicMock:
    client = MagicMock(spec=AsyncOpenAI)
    client.chat = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


def _complete(
    model: OpenAiChatModel, messages: list[Message], tools: list[ToolInfo]
) -> ModelReply | None:
    return asyncio.run(model.complete(messages, tools))


def _reply(model: OpenAiChatModel) -> ModelReply:
    reply = _complete(model, [Message(role="user", text="TEST_QUERY")], [])
    assert reply is not None
    return reply


def test_given_the_chat_model_when_typed_as_the_interface_then_it_satisfies_it() -> None:
    _accepts_chat_model(OpenAiChatModel(TEST_MODEL))


def test_given_no_api_key_when_the_model_is_built_then_building_it_does_not_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins F-L47-11's shape: the OpenAI client raises without a key, so it is made on first use."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    OpenAiChatModel(TEST_MODEL)


def test_given_two_requests_when_made_then_the_client_is_created_once_on_the_first() -> None:
    created: list[str] = []
    fake_client = _make_client(_completion())

    def fake_factory() -> MagicMock:
        created.append("client")
        return fake_client

    model = OpenAiChatModel(TEST_MODEL, client_factory=fake_factory)
    assert created == []
    _complete(model, [Message(role="user", text="TEST_QUERY")], [])
    _complete(model, [Message(role="user", text="TEST_QUERY")], [])
    assert created == ["client"]


def test_given_a_text_reply_when_read_then_it_has_the_text_and_no_tool_calls() -> None:
    reply = _reply(OpenAiChatModel(TEST_MODEL, client=_make_client(_completion("TEST_ANSWER"))))
    assert (reply.text, reply.tool_calls) == ("TEST_ANSWER", ())


def test_given_a_tool_call_reply_when_read_then_it_has_the_id_name_and_argument_text() -> None:
    call = ChatCompletionMessageFunctionToolCall(
        id="TEST_CALL", type="function", function=Function(name="TEST_TOOL", arguments='{"a": 1}')
    )
    reply = _reply(OpenAiChatModel(TEST_MODEL, client=_make_client(_completion(None, [call]))))
    assert reply.text is None
    assert reply.tool_calls == (ToolCall("TEST_CALL", "TEST_TOOL", '{"a": 1}'),)


def test_given_a_custom_tool_call_when_read_then_it_is_skipped() -> None:
    """Pins that only function tools are offered, so a custom tool call is never run."""
    call = ChatCompletionMessageCustomToolCall(
        id="TEST_CALL", type="custom", custom=Custom(input="TEST_INPUT", name="TEST_TOOL")
    )
    reply = _reply(OpenAiChatModel(TEST_MODEL, client=_make_client(_completion(None, [call]))))
    assert reply.tool_calls == ()


def test_given_a_reply_with_no_choices_when_read_then_the_result_is_none() -> None:
    model = OpenAiChatModel(TEST_MODEL, client=_make_client(_completion(empty=True)))
    assert _complete(model, [Message(role="user", text="TEST_QUERY")], []) is None


def test_given_tools_when_a_request_is_made_then_they_are_offered_with_auto_choice() -> None:
    client = _make_client(_completion())
    model = OpenAiChatModel(TEST_MODEL, client=client)
    tool = ToolInfo("TEST_TOOL", "TEST_DESC", {"type": "object", "properties": {}})
    _complete(model, [Message(role="user", text="TEST_QUERY")], [tool])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == TEST_MODEL
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "TEST_TOOL",
                "description": "TEST_DESC",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_given_no_tools_when_a_request_is_made_then_neither_tools_nor_choice_is_sent() -> None:
    client = _make_client(_completion())
    model = OpenAiChatModel(TEST_MODEL, client=client)
    _complete(model, [Message(role="user", text="TEST_QUERY")], [])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert isinstance(kwargs["tools"], Omit)
    assert isinstance(kwargs["tool_choice"], Omit)


def test_given_every_kind_of_message_when_a_request_is_made_then_each_has_the_wire_shape() -> None:
    client = _make_client(_completion())
    model = OpenAiChatModel(TEST_MODEL, client=client)
    call = ToolCall("TEST_CALL", "TEST_TOOL", "{}")
    messages = [
        Message(role="user", text="TEST_QUERY"),
        Message(role="assistant", text=None, tool_calls=(call,)),
        Message(role="tool", text="TEST_RESULT", tool_call_id="TEST_CALL"),
        Message(role="assistant", text="TEST_ANSWER"),
    ]
    _complete(model, messages, [])
    assert client.chat.completions.create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "TEST_QUERY"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "TEST_CALL",
                    "type": "function",
                    "function": {"name": "TEST_TOOL", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "TEST_CALL", "content": "TEST_RESULT"},
        {"role": "assistant", "content": "TEST_ANSWER"},
    ]
