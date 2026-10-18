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
from openai.types.chat.chat_completion_message_function_tool_call import Function

from domain.conversation import Message, ModelReply, ToolCall
from domain.tool_info import ToolInfo
from infra.openai_chat_model import OpenAiChatModel
from interfaces.llm_chat_model_interface import LlmChatModelInterface

TEST_MODEL = "TEST_MODEL"


def _accepts_chat_model(model: LlmChatModelInterface) -> None:
    """Type-guard helper."""


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


def test_satisfies_the_chat_model_interface() -> None:
    _accepts_chat_model(OpenAiChatModel(TEST_MODEL))


def test_building_it_does_not_need_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The OpenAI client raises without a key, so it is created on first use, not here.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    OpenAiChatModel(TEST_MODEL)


def test_the_client_is_created_on_the_first_request_not_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[str] = []
    fake_client = _make_client(_completion())

    def fake_factory() -> MagicMock:
        created.append("client")
        return fake_client

    monkeypatch.setattr("infra.openai_chat_model.AsyncOpenAI", fake_factory)
    model = OpenAiChatModel(TEST_MODEL)
    assert created == []
    _complete(model, [Message(role="user", text="TEST_QUERY")], [])
    _complete(model, [Message(role="user", text="TEST_QUERY")], [])
    assert created == ["client"]


def test_a_text_reply_has_no_tool_calls() -> None:
    reply = _reply(OpenAiChatModel(TEST_MODEL, client=_make_client(_completion("TEST_ANSWER"))))
    assert (reply.text, reply.tool_calls) == ("TEST_ANSWER", ())


def test_a_tool_call_reply_gives_the_id_name_and_argument_text() -> None:
    call = ChatCompletionMessageFunctionToolCall(
        id="TEST_CALL", type="function", function=Function(name="TEST_TOOL", arguments='{"a": 1}')
    )
    reply = _reply(OpenAiChatModel(TEST_MODEL, client=_make_client(_completion(None, [call]))))
    assert reply.text is None
    assert reply.tool_calls == (ToolCall("TEST_CALL", "TEST_TOOL", '{"a": 1}'),)


def test_a_reply_with_no_choices_is_none() -> None:
    model = OpenAiChatModel(TEST_MODEL, client=_make_client(_completion(empty=True)))
    assert _complete(model, [Message(role="user", text="TEST_QUERY")], []) is None


def test_the_request_uses_the_model_id_and_offers_the_tools_with_auto_choice() -> None:
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


def test_with_no_tools_neither_tools_nor_tool_choice_is_sent() -> None:
    client = _make_client(_completion())
    model = OpenAiChatModel(TEST_MODEL, client=client)
    _complete(model, [Message(role="user", text="TEST_QUERY")], [])
    kwargs = client.chat.completions.create.call_args.kwargs
    assert isinstance(kwargs["tools"], Omit)
    assert isinstance(kwargs["tool_choice"], Omit)


def test_each_kind_of_message_is_sent_in_the_chat_completions_shape() -> None:
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
