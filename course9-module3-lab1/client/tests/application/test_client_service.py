import asyncio
from unittest.mock import MagicMock

import pytest

from application.client_service import ClientService
from application.formatters import COMMANDS_LINE, CONNECTED, HELP_TEXT, WELCOME
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface


def _make_session() -> MagicMock:
    session = MagicMock(spec=McpSessionInterface)
    session.list_tools.return_value = [{"name": "TEST_TOOL", "description": "TEST_DESC"}]
    session.call_tool.return_value = {"texts": ["TEST_RESULT"], "is_error": False}
    session.list_resource_templates.return_value = [
        {"name": "TEST_TEMPLATE", "uri_template": "TEST://{x}"}
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


def _run(session: MagicMock, *answers: str) -> list[str]:
    """Run the client with scripted answers; return everything it showed."""
    user_input = MagicMock(spec=UserInputInterface)
    user_input.ask.side_effect = list(answers)
    asyncio.run(ClientService(session, user_input).run())
    return [c.args[0] for c in user_input.show.call_args_list]


def test_quit_connects_greets_then_closes() -> None:
    session = _make_session()
    shown = _run(session, "quit")
    session.connect.assert_awaited_once()
    session.close.assert_awaited_once()
    assert shown == [CONNECTED, WELCOME, COMMANDS_LINE]


def test_commands_are_case_insensitive() -> None:
    shown = _run(_make_session(), "TOOLS", "quit")
    assert "  • TEST_TOOL: TEST_DESC" in shown


def test_unknown_and_empty_commands_are_reported_and_the_loop_continues() -> None:
    shown = _run(_make_session(), "nope", "", "tools", "quit")
    assert shown.count("  Unknown command") == 2
    assert "  • TEST_TOOL: TEST_DESC" in shown


def test_connect_failure_propagates_and_the_session_is_still_closed() -> None:
    session = _make_session()
    session.connect.side_effect = RuntimeError("TEST_BOOM")
    with pytest.raises(RuntimeError, match="TEST_BOOM"):
        _run(session)
    session.close.assert_awaited_once()


def test_ctrl_c_shows_exiting_and_closes_the_session() -> None:
    session = MagicMock(spec=McpSessionInterface)
    user_input = MagicMock(spec=UserInputInterface)
    user_input.ask.side_effect = KeyboardInterrupt()
    asyncio.run(ClientService(session, user_input).run())
    assert "\nExiting..." in [c.args[0] for c in user_input.show.call_args_list]
    session.close.assert_awaited_once()


def test_tools_lists_each_tool() -> None:
    session = _make_session()
    shown = _run(session, "tools", "quit")
    session.list_tools.assert_awaited_once()
    assert "  • TEST_TOOL: TEST_DESC" in shown


def test_call_parses_the_json_and_shows_each_result_text() -> None:
    session = _make_session()
    session.call_tool.return_value = {"texts": ["TEST_ONE", "TEST_TWO"], "is_error": False}
    shown = _run(session, "call", "TEST_TOOL", '{"text": "TEST_VALUE"}', "quit")
    session.call_tool.assert_awaited_once_with("TEST_TOOL", {"text": "TEST_VALUE"})
    assert "  Result: TEST_ONE" in shown
    assert "  Result: TEST_TWO" in shown


def test_call_shows_the_lab_prompt_for_arguments() -> None:
    shown = _run(_make_session(), "call", "TEST_TOOL", "{}", "quit")
    assert '  Arguments (as JSON, for example, {"text": "hello"}): ' in shown


def test_call_shows_a_tool_error_result_as_an_ordinary_result() -> None:
    # The lab ignores the error flag, so a failed call prints like a successful one.
    # Pinned on purpose: a later step changes this.
    session = _make_session()
    session.call_tool.return_value = {"texts": ["TEST_BOOM"], "is_error": True}
    shown = _run(session, "call", "TEST_TOOL", "{}", "quit")
    assert "  Result: TEST_BOOM" in shown


def test_call_with_invalid_json_shows_a_hint_and_does_not_call_the_tool() -> None:
    session = _make_session()
    shown = _run(session, "call", "TEST_TOOL", "{'text': 'hi'}", "tools", "quit")
    session.call_tool.assert_not_called()
    assert "  Error: Invalid JSON format" in shown
    assert '  Hint: Use double quotes, for example, {"text": "hello"}' in shown
    assert "  • TEST_TOOL: TEST_DESC" in shown


def test_a_failing_call_shows_the_error_and_the_loop_continues() -> None:
    session = _make_session()
    session.call_tool.side_effect = RuntimeError("TEST_BOOM")
    shown = _run(session, "call", "TEST_TOOL", "{}", "tools", "quit")
    assert "  Error: TEST_BOOM" in shown
    assert "  Hint: Check the resource URI or filename" not in shown
    assert "  • TEST_TOOL: TEST_DESC" in shown


def test_an_error_that_says_not_found_adds_the_hint() -> None:
    session = _make_session()
    session.read_resource.side_effect = RuntimeError("TEST resource Not Found")
    shown = _run(session, "read", "TEST://x", "quit")
    assert "  Error: TEST resource Not Found" in shown
    assert "  Hint: Check the resource URI or filename" in shown


def test_history_is_empty_at_first() -> None:
    shown = _run(_make_session(), "history", "quit")
    assert "  No operations yet" in shown


def test_history_lists_successful_tool_calls_in_order() -> None:
    shown = _run(_make_session(), "call", "TEST_A", "{}", "call", "TEST_B", "{}", "history", "quit")
    assert "  Operation History:\n    1. tool: TEST_A\n    2. tool: TEST_B" in shown


def test_history_does_not_record_a_call_that_raised() -> None:
    session = _make_session()
    session.call_tool.side_effect = RuntimeError("TEST_BOOM")
    shown = _run(session, "call", "TEST_A", "{}", "history", "quit")
    assert "  No operations yet" in shown


def test_resources_lists_each_template() -> None:
    shown = _run(_make_session(), "resources", "quit")
    assert "  • TEST_TEMPLATE\n    URI template: TEST://{x}" in shown


def test_resources_when_there_are_none_says_so() -> None:
    session = _make_session()
    session.list_resource_templates.return_value = []
    shown = _run(session, "resources", "quit")
    assert "  No resources available" in shown


def test_read_sends_the_uri_as_typed_and_shows_each_text_part() -> None:
    session = _make_session()
    session.read_resource.return_value = ["TEST_ONE", "TEST_TWO"]
    shown = _run(session, "read", "file://resources/a.txt", "quit")
    session.read_resource.assert_awaited_once_with("file://resources/a.txt")
    assert "\nTEST_ONE" in shown
    assert "\nTEST_TWO" in shown


def test_prompts_lists_each_prompt_with_its_arguments() -> None:
    shown = _run(_make_session(), "prompts", "quit")
    assert "  • TEST_PROMPT: TEST_DESC (args: TEST_ARG)" in shown


def test_prompt_parses_the_json_and_shows_the_rendered_messages() -> None:
    session = _make_session()
    shown = _run(session, "prompt", "TEST_PROMPT", '{"TEST_ARG": "TEST_VALUE"}', "quit")
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"TEST_ARG": "TEST_VALUE"})
    assert "  Arguments (as JSON): " in shown
    assert "\n--- Prompt: TEST_RENDERED ---\nuser: TEST_MESSAGE" in shown


def test_prompt_with_invalid_json_does_not_call_the_server() -> None:
    session = _make_session()
    shown = _run(session, "prompt", "TEST_PROMPT", "not json", "quit")
    session.get_prompt.assert_not_called()
    assert "  Error: Invalid JSON format" in shown


def test_help_shows_the_help_text() -> None:
    shown = _run(_make_session(), "help", "quit")
    assert HELP_TEXT in shown
