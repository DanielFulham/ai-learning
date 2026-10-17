from application.formatters import (
    HELP_TEXT,
    format_history,
    format_prompt,
    format_rendered_prompt,
    format_resource_template,
    format_tool,
)
from domain.history_entry import HistoryEntry
from domain.prompt_info import PromptInfo, PromptMessage, RenderedPrompt
from domain.resource_template_info import ResourceTemplateInfo
from domain.tool_info import ToolInfo


def test_format_tool_shows_name_and_description() -> None:
    assert format_tool(ToolInfo("TEST_TOOL", "TEST_DESC")) == "  • TEST_TOOL: TEST_DESC"


def test_format_resource_template_shows_name_and_uri_template_on_two_lines() -> None:
    template = ResourceTemplateInfo("TEST_NAME", "TEST://{x}")
    assert format_resource_template(template) == "  • TEST_NAME\n    URI template: TEST://{x}"


def test_format_prompt_lists_argument_names() -> None:
    prompt = PromptInfo("TEST_PROMPT", "TEST_DESC", ("TEST_A", "TEST_B"))
    assert format_prompt(prompt) == "  • TEST_PROMPT: TEST_DESC (args: TEST_A, TEST_B)"


def test_format_prompt_without_arguments_has_no_args_part() -> None:
    assert format_prompt(PromptInfo("TEST_PROMPT", "TEST_DESC")) == "  • TEST_PROMPT: TEST_DESC"


def test_format_rendered_prompt_has_a_header_and_one_line_per_message() -> None:
    prompt = RenderedPrompt("TEST_DESC", (PromptMessage("user", "TEST_ONE"),))
    assert format_rendered_prompt(prompt) == "\n--- Prompt: TEST_DESC ---\nuser: TEST_ONE"


def test_format_history_numbers_entries_from_one() -> None:
    entries = [HistoryEntry("tool", "TEST_A"), HistoryEntry("tool", "TEST_B")]
    assert format_history(entries) == (
        "  Operation History:\n    1. tool: TEST_A\n    2. tool: TEST_B"
    )


def test_format_history_when_empty_says_so() -> None:
    assert format_history([]) == "  No operations yet"


def test_help_text_names_every_command() -> None:
    for command in ("tools", "call", "resources", "read", "prompts", "prompt", "history", "help"):
        assert f"  {command} " in HELP_TEXT
    assert "quit" in HELP_TEXT
