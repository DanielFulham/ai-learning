import pytest

from application.mappers import (
    answer_from_choice,
    choice_from_reply,
    parse_json_object,
    parse_prompt_arguments,
    to_prompt_info,
    to_rendered_prompt,
    to_resource_info,
    to_tool_call_result,
    to_tool_info,
)
from domain.approval import ApprovalChoice
from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_info import ResourceInfo
from domain.tool_info import ToolCallResult, ToolInfo
from interfaces.mcp_session_interface import (
    RawPrompt,
    RawPromptMessage,
    RawPromptResult,
    RawResource,
    RawTool,
    RawToolResult,
)


def test_given_a_raw_tool_when_mapped_then_the_domain_tool_has_the_same_fields() -> None:
    raw = RawTool(name="TEST_TOOL", description="TEST_DESC", input_schema={"type": "object"})
    assert to_tool_info(raw) == ToolInfo("TEST_TOOL", "TEST_DESC", {"type": "object"})


def test_given_a_raw_tool_result_when_mapped_then_the_texts_become_a_tuple() -> None:
    """Pins the immutability contract: a list from infra becomes a tuple in the domain."""
    raw = RawToolResult(texts=["TEST_ONE", "TEST_TWO"], is_error=True)
    assert to_tool_call_result(raw) == ToolCallResult(("TEST_ONE", "TEST_TWO"), True)


def test_given_a_raw_resource_when_mapped_then_the_domain_resource_has_the_same_fields() -> None:
    raw = RawResource(uri="file://TEST/one", name="TEST_NAME", description="TEST_DESC")
    assert to_resource_info(raw) == ResourceInfo("file://TEST/one", "TEST_NAME", "TEST_DESC")


def test_given_a_raw_prompt_when_mapped_then_the_argument_names_become_a_tuple() -> None:
    raw = RawPrompt(name="TEST_PROMPT", description="TEST_DESC", argument_names=["a", "b"])
    assert to_prompt_info(raw) == PromptInfo("TEST_PROMPT", "TEST_DESC", ("a", "b"))


def test_given_a_raw_rendered_prompt_when_mapped_then_the_messages_become_domain_messages() -> None:
    raw = RawPromptResult(
        description="TEST_DESC",
        messages=[RawPromptMessage(role="user", text="TEST_TEXT")],
    )
    assert to_rendered_prompt(raw) == RenderedPrompt(
        "TEST_DESC", (PromptMessage("user", "TEST_TEXT"),)
    )


@pytest.mark.parametrize(
    ("choice", "answer"),
    [
        (ApprovalChoice.ACCEPT, "accept"),
        (ApprovalChoice.DECLINE, "decline"),
        (ApprovalChoice.CANCEL, "cancel"),
    ],
)
def test_given_a_choice_when_mapped_to_an_answer_then_it_is_the_matching_word(
    choice: ApprovalChoice, answer: str
) -> None:
    assert answer_from_choice(choice) == answer


@pytest.mark.parametrize("text", ["yes", "y", "approve", "ok", "confirm", "YES", "  Yes  "])
def test_given_a_yes_word_when_read_as_a_reply_then_the_choice_is_accept(text: str) -> None:
    assert choice_from_reply(text) is ApprovalChoice.ACCEPT


@pytest.mark.parametrize("text", ["no", "n", "deny", "decline", "NO", "  No  "])
def test_given_a_no_word_when_read_as_a_reply_then_the_choice_is_decline(text: str) -> None:
    assert choice_from_reply(text) is ApprovalChoice.DECLINE


def test_given_the_word_cancel_when_read_as_a_reply_then_the_choice_is_cancel() -> None:
    assert choice_from_reply("cancel") is ApprovalChoice.CANCEL


@pytest.mark.parametrize(
    "text", ["", "maybe", "yes please", "yess", "ok then", "write a file", "yes, run it"]
)
def test_given_anything_else_when_read_as_a_reply_then_there_is_no_choice(text: str) -> None:
    """Pins that only a bare yes or no word answers a question, never a sentence that has one."""
    assert choice_from_reply(text) is None


def test_given_empty_text_when_parsed_as_arguments_then_the_result_is_an_empty_dict() -> None:
    assert parse_json_object("") == {}
    assert parse_json_object("   ") == {}


def test_given_a_json_object_when_parsed_then_the_dict_comes_back() -> None:
    assert parse_json_object('{"filepath": "TEST.txt", "n": 1}') == {"filepath": "TEST.txt", "n": 1}


@pytest.mark.parametrize("text", ["not json", "{", '["a"]', '"text"', "42", "null", "true"])
def test_given_text_that_is_not_a_json_object_when_parsed_then_a_value_error_is_raised(
    text: str,
) -> None:
    """Pins F-L47-6's cousin: Lab 46 let a JSON list or number through as tool arguments."""
    with pytest.raises(ValueError):
        parse_json_object(text)


def test_given_string_values_when_parsed_as_prompt_arguments_then_the_dict_comes_back() -> None:
    assert parse_prompt_arguments('{"topic": "TEST_TOPIC"}') == {"topic": "TEST_TOPIC"}


def test_given_empty_text_when_parsed_as_prompt_arguments_then_the_result_is_an_empty_dict() -> (
    None
):
    assert parse_prompt_arguments("") == {}


@pytest.mark.parametrize("text", ['{"n": 1}', '{"flag": true}', '{"x": null}', '{"x": ["a"]}'])
def test_given_a_non_string_value_when_parsed_as_prompt_arguments_then_a_value_error_is_raised(
    text: str,
) -> None:
    """Prompt arguments are text by protocol: a number is refused, not silently converted."""
    with pytest.raises(ValueError):
        parse_prompt_arguments(text)
