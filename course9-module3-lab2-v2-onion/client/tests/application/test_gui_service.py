import asyncio
from unittest.mock import MagicMock

import pytest

from application.gui_service import GuiService
from interfaces.mcp_session_interface import McpSessionInterface


def _make_session() -> MagicMock:
    session = MagicMock(spec=McpSessionInterface)
    session.list_tools.return_value = [
        {"name": "TEST_TOOL", "description": "TEST_DESC", "input_schema": {}}
    ]
    session.call_tool.return_value = {"texts": ["TEST_RESULT"], "is_error": False}
    session.list_resource_templates.return_value = [
        {"name": "TEST_TEMPLATE", "uri_template": "TEST://{x}", "description": "TEST_DOC"}
    ]
    session.read_resource.return_value = ["TEST_TEXT"]
    session.list_prompts.return_value = [
        {"name": "TEST_PROMPT", "description": "TEST_DESC", "argument_names": ["TEST_ARG"]}
    ]
    session.get_prompt.return_value = {
        "description": "TEST_RENDERED",
        "messages": [{"role": "user", "text": "TEST_MESSAGE"}],
    }
    return session


def test_list_tools_gives_bullet_lines_and_the_names_for_the_dropdown() -> None:
    session = _make_session()
    output, choices = asyncio.run(GuiService(session).list_tools())
    session.connect.assert_awaited_once()
    assert output == "• TEST_TOOL: TEST_DESC"
    assert choices == ["TEST_TOOL"]


def test_list_tools_when_there_are_none_gives_nothing() -> None:
    session = _make_session()
    session.list_tools.return_value = []
    assert asyncio.run(GuiService(session).list_tools()) == ("", [])


def test_call_tool_parses_the_json_and_ends_each_text_with_a_newline() -> None:
    session = _make_session()
    session.call_tool.return_value = {"texts": ["TEST_ONE", "TEST_TWO"], "is_error": False}
    output = asyncio.run(GuiService(session).call_tool("TEST_TOOL", '{"a": "TEST_VALUE"}'))
    session.call_tool.assert_awaited_once_with("TEST_TOOL", {"a": "TEST_VALUE"})
    assert output == "TEST_ONE\nTEST_TWO\n"


def test_call_tool_with_empty_arguments_sends_an_empty_dict() -> None:
    session = _make_session()
    asyncio.run(GuiService(session).call_tool("TEST_TOOL", ""))
    session.call_tool.assert_awaited_once_with("TEST_TOOL", {})


def test_call_tool_with_no_selection_asks_for_one_but_still_connects() -> None:
    # The lab connects before it checks the selection.
    session = _make_session()
    output = asyncio.run(GuiService(session).call_tool(None, "{}"))
    assert output == "Error: Please select a tool from the dropdown first"
    session.connect.assert_awaited_once()
    session.call_tool.assert_not_called()


def test_call_tool_with_invalid_json_does_not_call_the_tool() -> None:
    session = _make_session()
    output = asyncio.run(GuiService(session).call_tool("TEST_TOOL", "{'a': 1}"))
    assert output == "Error: Invalid JSON format"
    session.call_tool.assert_not_called()


def test_call_tool_with_no_text_says_no_response() -> None:
    session = _make_session()
    session.call_tool.return_value = {"texts": [], "is_error": False}
    assert asyncio.run(GuiService(session).call_tool("TEST_TOOL", "{}")) == "No response"


def test_call_tool_shows_a_result_the_server_flagged_as_an_error_as_an_error() -> None:
    # The lab ignores the error flag, so a failed call looked like a successful one.
    session = _make_session()
    session.call_tool.return_value = {"texts": ["TEST_BOOM"], "is_error": True}
    assert asyncio.run(GuiService(session).call_tool("TEST_TOOL", "{}")) == "Error: TEST_BOOM"


def test_call_tool_with_an_error_and_no_text_still_says_so() -> None:
    session = _make_session()
    session.call_tool.return_value = {"texts": [], "is_error": True}
    output = asyncio.run(GuiService(session).call_tool("TEST_TOOL", "{}"))
    assert output == "Error: the tool reported an error"


def test_call_tool_failure_is_shown_as_error_text() -> None:
    session = _make_session()
    session.call_tool.side_effect = RuntimeError("TEST_BOOM")
    output = asyncio.run(GuiService(session).call_tool("TEST_TOOL", "{}"))
    assert output == "Error: TEST_BOOM"


def test_a_connect_failure_is_not_caught_as_in_the_lab() -> None:
    session = _make_session()
    session.connect.side_effect = RuntimeError("TEST_NO_SERVER")
    with pytest.raises(RuntimeError, match="TEST_NO_SERVER"):
        asyncio.run(GuiService(session).call_tool("TEST_TOOL", "{}"))


def test_list_resources_shows_each_template() -> None:
    output = asyncio.run(GuiService(_make_session()).list_resources())
    assert output == "• TEST_TEMPLATE\n  URI template: TEST://{x}"


def test_list_resources_separates_templates_with_a_blank_line() -> None:
    session = _make_session()
    session.list_resource_templates.return_value = [
        {"name": "A", "uri_template": "A://{x}", "description": ""},
        {"name": "B", "uri_template": "B://{x}", "description": ""},
    ]
    output = asyncio.run(GuiService(session).list_resources())
    assert output == "• A\n  URI template: A://{x}\n\n• B\n  URI template: B://{x}"


def test_list_resources_when_there_are_none_says_so() -> None:
    session = _make_session()
    session.list_resource_templates.return_value = []
    assert asyncio.run(GuiService(session).list_resources()) == "No resources available"


def test_read_resource_sends_the_uri_as_typed_and_ends_each_text_with_a_newline() -> None:
    session = _make_session()
    output = asyncio.run(GuiService(session).read_resource("file://workspace/a.txt"))
    session.read_resource.assert_awaited_once_with("file://workspace/a.txt")
    assert output == "TEST_TEXT\n"


def test_read_resource_with_no_uri_asks_for_one() -> None:
    session = _make_session()
    assert (
        asyncio.run(GuiService(session).read_resource("")) == "Error: Please enter a resource URI"
    )
    session.read_resource.assert_not_called()


def test_read_resource_with_no_text_says_no_content() -> None:
    session = _make_session()
    session.read_resource.return_value = []
    assert asyncio.run(GuiService(session).read_resource("TEST://x")) == "No content"


def test_read_resource_failure_is_shown_as_error_text() -> None:
    session = _make_session()
    session.read_resource.side_effect = RuntimeError("TEST_BOOM")
    assert asyncio.run(GuiService(session).read_resource("TEST://x")) == "Error: TEST_BOOM"


def test_list_prompts_gives_lines_with_arguments_and_the_names_for_the_dropdown() -> None:
    output, choices = asyncio.run(GuiService(_make_session()).list_prompts())
    assert output == "• TEST_PROMPT: TEST_DESC (args: TEST_ARG)"
    assert choices == ["TEST_PROMPT"]


def test_get_prompt_parses_the_json_and_shows_the_header_and_messages() -> None:
    session = _make_session()
    output = asyncio.run(GuiService(session).get_prompt("TEST_PROMPT", '{"TEST_ARG": "V"}'))
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"TEST_ARG": "V"})
    assert output == "--- Prompt: TEST_RENDERED ---\n\nuser: TEST_MESSAGE\n"


def test_get_prompt_with_no_selection_asks_for_one() -> None:
    session = _make_session()
    output = asyncio.run(GuiService(session).get_prompt(None, "{}"))
    assert output == "Error: Please select a prompt from the dropdown first"
    session.get_prompt.assert_not_called()


def test_get_prompt_with_invalid_json_does_not_call_the_server() -> None:
    session = _make_session()
    assert asyncio.run(GuiService(session).get_prompt("P", "not json")) == (
        "Error: Invalid JSON format"
    )
    session.get_prompt.assert_not_called()


def test_get_prompt_failure_is_shown_as_error_text() -> None:
    session = _make_session()
    session.get_prompt.side_effect = RuntimeError("TEST_BOOM")
    assert asyncio.run(GuiService(session).get_prompt("P", "{}")) == "Error: TEST_BOOM"
