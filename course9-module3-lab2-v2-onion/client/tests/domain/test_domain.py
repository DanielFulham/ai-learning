from dataclasses import FrozenInstanceError

import pytest

from domain.conversation import ChatReply, Message, ModelReply, ToolActivity, ToolCall
from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_template_info import ResourceTemplateInfo
from domain.tool_info import ToolCallResult, ToolInfo


def test_tool_info_defaults_to_an_empty_schema_and_is_frozen() -> None:
    tool = ToolInfo(name="TEST_TOOL", description="TEST_DESC")
    assert tool.input_schema == {}
    with pytest.raises(FrozenInstanceError):
        setattr(tool, "name", "TEST_OTHER")


def test_tool_call_result_is_frozen() -> None:
    result = ToolCallResult(texts=("TEST_TEXT",), is_error=False)
    with pytest.raises(FrozenInstanceError):
        setattr(result, "is_error", True)


def test_resource_template_info_is_frozen() -> None:
    template = ResourceTemplateInfo(name="TEST_NAME", uri_template="TEST://{x}", description="D")
    with pytest.raises(FrozenInstanceError):
        setattr(template, "name", "TEST_OTHER")


def test_prompt_info_defaults_to_no_arguments_and_is_frozen() -> None:
    prompt = PromptInfo(name="TEST_PROMPT", description="TEST_DESC")
    assert prompt.argument_names == ()
    with pytest.raises(FrozenInstanceError):
        setattr(prompt, "name", "TEST_OTHER")


def test_rendered_prompt_defaults_to_no_messages_and_is_frozen() -> None:
    prompt = RenderedPrompt(description="TEST_DESC")
    assert prompt.messages == ()
    with pytest.raises(FrozenInstanceError):
        setattr(prompt, "description", "TEST_OTHER")


def test_prompt_message_is_frozen() -> None:
    message = PromptMessage(role="user", text="TEST_TEXT")
    with pytest.raises(FrozenInstanceError):
        setattr(message, "text", "TEST_OTHER")


def test_tool_call_keeps_the_arguments_as_the_text_the_model_wrote() -> None:
    call = ToolCall(id="TEST_ID", name="TEST_TOOL", arguments_json='{"a": 1}')
    assert call.arguments_json == '{"a": 1}'
    with pytest.raises(FrozenInstanceError):
        setattr(call, "name", "TEST_OTHER")


def test_message_defaults_to_no_text_no_calls_and_no_call_id() -> None:
    message = Message(role="user")
    assert (message.text, message.tool_calls, message.tool_call_id) == (None, (), None)
    with pytest.raises(FrozenInstanceError):
        setattr(message, "text", "TEST_OTHER")


def test_model_reply_defaults_to_no_tool_calls() -> None:
    reply = ModelReply(text="TEST_TEXT")
    assert reply.tool_calls == ()
    with pytest.raises(FrozenInstanceError):
        setattr(reply, "text", "TEST_OTHER")


def test_chat_reply_defaults_to_no_tool_activity() -> None:
    reply = ChatReply(text="TEST_TEXT")
    assert reply.tool_activity == ()
    assert reply.history == ()
    with pytest.raises(FrozenInstanceError):
        setattr(reply, "text", "TEST_OTHER")


def test_tool_activity_is_frozen() -> None:
    activity = ToolActivity(name="TEST_TOOL", arguments_json="{}", result="TEST_RESULT")
    with pytest.raises(FrozenInstanceError):
        setattr(activity, "result", "TEST_OTHER")
