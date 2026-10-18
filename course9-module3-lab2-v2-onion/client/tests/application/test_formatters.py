from application.formatters import (
    format_chat_reply,
    format_prompt,
    format_rendered_prompt,
    format_resource_template,
    format_text_parts,
    format_tool,
)
from domain.conversation import ChatReply, ToolActivity
from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_template_info import ResourceTemplateInfo
from domain.tool_info import ToolInfo


def test_format_tool_shows_name_and_description() -> None:
    assert format_tool(ToolInfo("TEST_TOOL", "TEST_DESC")) == "• TEST_TOOL: TEST_DESC"


def test_format_resource_template_shows_name_and_uri_template_on_two_lines() -> None:
    template = ResourceTemplateInfo("TEST_NAME", "TEST://{x}", "TEST_DESC")
    assert format_resource_template(template) == "• TEST_NAME\n  URI template: TEST://{x}"


def test_format_prompt_lists_argument_names() -> None:
    prompt = PromptInfo("TEST_PROMPT", "TEST_DESC", ("TEST_A", "TEST_B"))
    assert format_prompt(prompt) == "• TEST_PROMPT: TEST_DESC (args: TEST_A, TEST_B)"


def test_format_prompt_without_arguments_has_no_args_part() -> None:
    assert format_prompt(PromptInfo("TEST_PROMPT", "TEST_DESC")) == "• TEST_PROMPT: TEST_DESC"


def test_format_text_parts_ends_each_part_with_a_newline() -> None:
    assert format_text_parts(("TEST_ONE", "TEST_TWO")) == "TEST_ONE\nTEST_TWO\n"


def test_format_text_parts_of_nothing_is_empty() -> None:
    assert format_text_parts(()) == ""


def test_format_rendered_prompt_has_a_header_and_one_line_per_message() -> None:
    prompt = RenderedPrompt("TEST_DESC", (PromptMessage("user", "TEST_ONE"),))
    assert format_rendered_prompt(prompt) == "--- Prompt: TEST_DESC ---\n\nuser: TEST_ONE\n"


def test_format_chat_reply_without_tools_is_just_the_text() -> None:
    assert format_chat_reply(ChatReply("TEST_ANSWER")) == "TEST_ANSWER"


def test_format_chat_reply_lists_each_tool_call_and_result_before_the_answer() -> None:
    reply = ChatReply(
        "TEST_ANSWER",
        (
            ToolActivity("TEST_TOOL", '{"a": "V"}', "TEST_RESULT"),
            ToolActivity("TEST_OTHER", "{}", "TEST_OTHER_RESULT"),
        ),
    )
    assert format_chat_reply(reply) == (
        "**Tool calls**\n"
        '- `TEST_TOOL({"a": "V"})` → TEST_RESULT\n'
        "- `TEST_OTHER({})` → TEST_OTHER_RESULT\n"
        "\n"
        "TEST_ANSWER"
    )


def test_format_chat_reply_truncates_a_long_result() -> None:
    reply = ChatReply("TEST_ANSWER", (ToolActivity("TEST_TOOL", "{}", "x" * 500),))
    shown = format_chat_reply(reply)
    assert "x" * 200 + "…" in shown
    assert "x" * 201 not in shown


def test_format_chat_reply_keeps_a_result_on_one_line() -> None:
    reply = ChatReply("TEST_ANSWER", (ToolActivity("TEST_TOOL", "{}", "line1\nline2"),))
    assert "→ line1 line2" in format_chat_reply(reply)
