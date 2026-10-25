from dataclasses import FrozenInstanceError, fields

import pytest

from domain.approval import ApprovalChoice
from domain.conversation import ChatReply, Message, ModelReply, ToolActivity, ToolCall
from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_info import ResourceInfo
from domain.tool_info import ToolCallResult, ToolInfo

# --- approval choices ----------------------------------------------------------------------


def test_given_the_approval_choices_when_listed_then_they_are_accept_decline_and_cancel() -> None:
    """Pins the three-way answer the server's question can get; a fourth has to break this."""
    assert {choice.value for choice in ApprovalChoice} == {"accept", "decline", "cancel"}


def test_given_the_text_accept_when_parsed_as_a_choice_then_it_is_the_accept_choice() -> None:
    assert ApprovalChoice("accept") is ApprovalChoice.ACCEPT


def test_given_text_that_is_not_a_choice_when_parsed_then_a_value_error_is_raised() -> None:
    with pytest.raises(ValueError):
        ApprovalChoice("yes")


# --- tools, resources, prompts -------------------------------------------------------------


def test_given_a_tool_with_no_schema_when_built_then_the_schema_defaults_to_empty() -> None:
    assert ToolInfo("TEST_TOOL", "TEST_DESCRIPTION").input_schema == {}


def test_given_two_tools_when_built_without_a_schema_then_they_do_not_share_one_dict() -> None:
    """Pins that the default schema is a fresh dict per tool, not one shared mutable object."""
    first, second = ToolInfo("A", "TEST"), ToolInfo("B", "TEST")
    assert first.input_schema is not second.input_schema


def test_given_a_tool_call_result_when_built_then_it_holds_the_texts_and_the_error_flag() -> None:
    result = ToolCallResult(texts=("TEST_ONE", "TEST_TWO"), is_error=True)
    assert result.texts == ("TEST_ONE", "TEST_TWO")
    assert result.is_error is True


def test_given_a_resource_when_built_then_it_holds_the_uri_name_and_description() -> None:
    resource = ResourceInfo("file://TEST/one", "TEST_NAME", "TEST_DESCRIPTION")
    assert (resource.uri, resource.name, resource.description) == (
        "file://TEST/one",
        "TEST_NAME",
        "TEST_DESCRIPTION",
    )


def test_given_a_prompt_with_no_arguments_when_built_then_the_names_default_to_empty() -> None:
    assert PromptInfo("TEST_PROMPT", "TEST_DESCRIPTION").argument_names == ()


def test_given_a_rendered_prompt_with_no_messages_when_built_then_the_messages_are_empty() -> None:
    assert RenderedPrompt("TEST_DESCRIPTION").messages == ()
    assert PromptMessage("user", "TEST_TEXT").role == "user"


# --- conversation --------------------------------------------------------------------------


def test_given_a_user_message_when_built_then_it_has_no_tool_calls_and_no_call_id() -> None:
    message = Message(role="user", text="TEST_TEXT")
    assert message.tool_calls == ()
    assert message.tool_call_id is None


def test_given_an_assistant_message_with_calls_when_built_then_it_keeps_the_calls_in_order() -> (
    None
):
    calls = (ToolCall("1", "TEST_A", "{}"), ToolCall("2", "TEST_B", "{}"))
    assert Message(role="assistant", tool_calls=calls).tool_calls == calls


def test_given_a_chat_reply_with_only_text_when_built_then_activity_and_history_are_empty() -> None:
    reply = ChatReply(text="TEST_TEXT")
    assert reply.tool_activity == ()
    assert reply.history == ()


def test_given_a_model_reply_with_no_text_when_built_then_the_text_is_none() -> None:
    """Pins that "the model only asked for tools" is None, not an empty string."""
    assert ModelReply(text=None, tool_calls=(ToolCall("1", "TEST", "{}"),)).text is None


@pytest.mark.parametrize(
    "value",
    [
        ToolInfo("TEST", "TEST"),
        ToolCallResult(("TEST",), False),
        ResourceInfo("file://TEST", "TEST", "TEST"),
        PromptInfo("TEST", "TEST"),
        Message(role="user", text="TEST"),
        ToolActivity("TEST", "{}", "TEST"),
        ChatReply(text="TEST"),
    ],
)
def test_given_any_domain_value_when_a_field_is_assigned_then_it_is_frozen(
    value: ToolInfo
    | ToolCallResult
    | ResourceInfo
    | PromptInfo
    | Message
    | ToolActivity
    | ChatReply,
) -> None:
    """Pins immutability for every domain type that crosses a layer boundary."""
    first_field = fields(value)[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(value, first_field, "TEST_CHANGED")
