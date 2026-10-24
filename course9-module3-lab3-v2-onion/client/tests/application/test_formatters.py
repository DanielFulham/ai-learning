from application.formatters import (
    format_chat_reply,
    format_prompt,
    format_rendered_prompt,
    format_resource,
    format_text_parts,
    format_tool,
)
from domain.conversation import ChatReply, ToolActivity
from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_info import ResourceInfo
from domain.tool_info import ToolInfo


def test_given_a_tool_when_formatted_then_it_is_a_bullet_with_name_and_description() -> None:
    assert format_tool(ToolInfo("TEST_TOOL", "TEST_DESC")) == "• TEST_TOOL: TEST_DESC"


def test_given_a_resource_when_formatted_then_it_shows_the_uri_name_and_description() -> None:
    text = format_resource(ResourceInfo("file://TEST/one", "TEST_NAME", "TEST_DESC"))
    assert text == "• file://TEST/one\n  TEST_NAME: TEST_DESC"


def test_given_a_prompt_with_arguments_when_formatted_then_the_arguments_are_listed() -> None:
    assert format_prompt(PromptInfo("TEST_P", "TEST_DESC", ("a", "b"))) == (
        "• TEST_P: TEST_DESC (args: a, b)"
    )


def test_given_a_prompt_with_no_arguments_when_formatted_then_there_is_no_args_note() -> None:
    assert format_prompt(PromptInfo("TEST_P", "TEST_DESC")) == "• TEST_P: TEST_DESC"


def test_given_text_parts_when_formatted_then_each_ends_with_a_newline() -> None:
    assert format_text_parts(("TEST_ONE", "TEST_TWO")) == "TEST_ONE\nTEST_TWO\n"


def test_given_no_text_parts_when_formatted_then_the_result_is_empty() -> None:
    assert format_text_parts(()) == ""


def test_given_a_rendered_prompt_when_formatted_then_it_has_a_header_and_role_lines() -> None:
    prompt = RenderedPrompt("TEST_DESC", (PromptMessage("user", "TEST_TEXT"),))
    assert format_rendered_prompt(prompt) == "--- Prompt: TEST_DESC ---\n\nuser: TEST_TEXT\n"


def test_given_a_reply_with_no_tool_activity_when_formatted_then_it_is_just_the_text() -> None:
    assert format_chat_reply(ChatReply(text="TEST_ANSWER")) == "TEST_ANSWER"


def test_given_a_reply_with_tool_activity_when_formatted_then_the_calls_lead_the_answer() -> None:
    reply = ChatReply(
        text="TEST_ANSWER",
        tool_activity=(ToolActivity("TEST_TOOL", '{"a": 1}', "TEST_RESULT"),),
    )
    assert format_chat_reply(reply) == (
        '**Tool calls**\n- `TEST_TOOL({"a": 1})` → TEST_RESULT\n\nTEST_ANSWER'
    )


def test_given_a_long_tool_result_when_formatted_then_it_is_cut_at_200_characters() -> None:
    reply = ChatReply(text="TEST", tool_activity=(ToolActivity("TEST_TOOL", "{}", "x" * 500),))
    line = format_chat_reply(reply).splitlines()[1]
    assert line.endswith("x" * 200 + "…")


def test_given_a_multiline_tool_result_when_formatted_then_it_is_one_line() -> None:
    reply = ChatReply(
        text="TEST", tool_activity=(ToolActivity("TEST_TOOL", "{}", "TEST_A\n\nTEST_B"),)
    )
    assert "TEST_A TEST_B" in format_chat_reply(reply)
