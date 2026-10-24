import asyncio
from unittest.mock import MagicMock

from application.gui_service import GuiService
from interfaces.approval_source_interface import ApprovalSourceInterface
from interfaces.mcp_session_interface import (
    McpSessionInterface,
    RawPrompt,
    RawPromptMessage,
    RawPromptResult,
    RawResource,
    RawTool,
    RawToolResult,
)

POLICY_URI = "file://TEST/policy"
AUDIT_URI = "file://TEST/audit"


def _make_session() -> MagicMock:
    session = MagicMock(spec=McpSessionInterface)
    session.list_tools.return_value = [
        RawTool(name="TEST_TOOL", description="TEST_DESC", input_schema={})
    ]
    session.call_tool.return_value = RawToolResult(texts=["TEST_OUT"], is_error=False)
    session.list_resources.return_value = [
        RawResource(uri="file://TEST/one", name="TEST_NAME", description="TEST_DESC")
    ]
    session.read_resource.return_value = ["TEST_TEXT"]
    session.list_prompts.return_value = [
        RawPrompt(name="TEST_PROMPT", description="TEST_DESC", argument_names=["topic"])
    ]
    session.get_prompt.return_value = RawPromptResult(
        description="TEST_DESC", messages=[RawPromptMessage(role="user", text="TEST_TEXT")]
    )
    return session


def _make_approvals() -> MagicMock:
    return MagicMock(spec=ApprovalSourceInterface)


def _make_service(session: MagicMock | None = None) -> GuiService:
    return GuiService(session or _make_session(), POLICY_URI, AUDIT_URI)


def _call(
    service: GuiService,
    tool: str | None,
    arguments: str,
    approvals: MagicMock | None = None,
) -> str:
    return asyncio.run(service.call_tool(tool, arguments, approvals or _make_approvals()))


def _read(service: GuiService, uri: str, approvals: MagicMock | None = None) -> str:
    return asyncio.run(service.read_resource(uri, approvals or _make_approvals()))


# --- tools ---------------------------------------------------------------------------------


def test_given_a_server_when_tools_are_listed_then_the_text_and_the_names_come_back() -> None:
    text, names = asyncio.run(_make_service().list_tools())
    assert text == "• TEST_TOOL: TEST_DESC"
    assert names == ["TEST_TOOL"]


def test_given_a_service_when_tools_are_listed_then_it_connects_first() -> None:
    session = _make_session()
    asyncio.run(_make_service(session).list_tools())
    session.connect.assert_awaited_once()


def test_given_no_tool_selected_when_called_then_it_asks_for_one_and_calls_nothing() -> None:
    session = _make_session()
    text = _call(_make_service(session), None, "{}")
    assert text == "Error: Please select a tool from the dropdown first"
    session.call_tool.assert_not_called()


def test_given_arguments_and_an_approval_source_when_a_tool_is_called_then_both_are_passed() -> (
    None
):
    """Pins F-L47-18: the call carries the calling page session's own approval source."""
    session, approvals = _make_session(), _make_approvals()
    _call(_make_service(session), "TEST_TOOL", '{"a": 1}', approvals)
    session.call_tool.assert_awaited_once_with("TEST_TOOL", {"a": 1}, approvals)


def test_given_two_page_sessions_when_each_calls_a_tool_then_each_call_gets_its_own_source() -> (
    None
):
    """Pins that the service is stateless: it never keeps one session's approval source."""
    session = _make_session()
    service = _make_service(session)
    first, second = _make_approvals(), _make_approvals()
    _call(service, "TEST_TOOL", "{}", first)
    _call(service, "TEST_TOOL", "{}", second)
    sources = [recorded.args[2] for recorded in session.call_tool.await_args_list]
    assert sources == [first, second]


def test_given_empty_arguments_when_a_tool_is_called_then_an_empty_object_is_sent() -> None:
    session = _make_session()
    _call(_make_service(session), "TEST_TOOL", "")
    assert session.call_tool.await_args is not None
    assert session.call_tool.await_args.args[1] == {}


def test_given_a_tool_result_when_called_then_the_text_parts_are_shown() -> None:
    assert _call(_make_service(), "TEST_TOOL", "{}") == "TEST_OUT\n"


def test_given_a_tool_with_no_text_when_called_then_no_response_is_shown() -> None:
    session = _make_session()
    session.call_tool.return_value = RawToolResult(texts=[], is_error=False)
    assert _call(_make_service(session), "TEST_TOOL", "{}") == "No response"


def test_given_a_flagged_error_when_a_tool_is_called_then_it_is_shown_as_an_error() -> None:
    session = _make_session()
    session.call_tool.return_value = RawToolResult(texts=["TEST_BAD"], is_error=True)
    assert _call(_make_service(session), "TEST_TOOL", "{}") == "Error: TEST_BAD"


def test_given_a_flagged_error_with_no_text_when_called_then_a_plain_error_is_shown() -> None:
    session = _make_session()
    session.call_tool.return_value = RawToolResult(texts=[], is_error=True)
    assert _call(_make_service(session), "TEST_TOOL", "{}") == "Error: the tool reported an error"


def test_given_invalid_json_when_a_tool_is_called_then_nothing_is_sent_to_the_server() -> None:
    session = _make_session()
    text = _call(_make_service(session), "TEST_TOOL", "{not json")
    assert text == "Error: Arguments must be a JSON object"
    session.call_tool.assert_not_called()


def test_given_json_that_is_not_an_object_when_a_tool_is_called_then_it_is_refused() -> None:
    """Pins that a list or a number never reaches the server as tool arguments."""
    session = _make_session()
    text = _call(_make_service(session), "TEST_TOOL", "[1, 2]")
    assert text == "Error: Arguments must be a JSON object"
    session.call_tool.assert_not_called()


def test_given_a_session_that_raises_when_a_tool_is_called_then_the_message_is_shown() -> None:
    session = _make_session()
    session.call_tool.side_effect = RuntimeError("TEST_BOOM")
    assert _call(_make_service(session), "TEST_TOOL", "{}") == "Error: TEST_BOOM"


# --- resources -----------------------------------------------------------------------------


def test_given_a_server_when_resources_are_listed_then_each_is_shown() -> None:
    text = asyncio.run(_make_service().list_resources())
    assert text == "• file://TEST/one\n  TEST_NAME: TEST_DESC"


def test_given_a_server_with_no_resources_when_listed_then_it_says_so() -> None:
    session = _make_session()
    session.list_resources.return_value = []
    assert asyncio.run(_make_service(session).list_resources()) == "No resources available"


def test_given_no_uri_when_a_resource_is_read_then_it_asks_for_one_and_reads_nothing() -> None:
    session = _make_session()
    text = _read(_make_service(session), "")
    assert text == "Error: Please enter a resource URI"
    session.read_resource.assert_not_called()


def test_given_a_uri_and_an_approval_source_when_read_then_both_are_passed() -> None:
    session, approvals = _make_session(), _make_approvals()
    text = _read(_make_service(session), "file://TEST/one", approvals)
    assert text == "TEST_TEXT\n"
    session.read_resource.assert_awaited_once_with("file://TEST/one", approvals)


def test_given_a_resource_with_no_text_when_read_then_no_content_is_shown() -> None:
    session = _make_session()
    session.read_resource.return_value = []
    assert _read(_make_service(session), "file://TEST/one") == "No content"


def test_given_a_session_that_raises_when_a_resource_is_read_then_the_message_is_shown() -> None:
    session = _make_session()
    session.read_resource.side_effect = RuntimeError("TEST_BOOM")
    assert _read(_make_service(session), "file://TEST/one") == "Error: TEST_BOOM"


def test_given_the_policy_view_when_opened_then_the_policy_resource_is_read() -> None:
    """The Permissions tab is read-only: the server owns the policy, so the client only shows it."""
    session, approvals = _make_session(), _make_approvals()
    asyncio.run(_make_service(session).view_policy(approvals))
    session.read_resource.assert_awaited_once_with(POLICY_URI, approvals)


def test_given_the_audit_view_when_opened_then_the_audit_resource_is_read() -> None:
    session, approvals = _make_session(), _make_approvals()
    asyncio.run(_make_service(session).view_audit_log(approvals))
    session.read_resource.assert_awaited_once_with(AUDIT_URI, approvals)


def test_given_a_refused_audit_read_when_viewed_then_the_error_is_shown() -> None:
    session = _make_session()
    session.read_resource.side_effect = RuntimeError("Permission denied for tool: read_audit_log")
    text = asyncio.run(_make_service(session).view_audit_log(_make_approvals()))
    assert text == "Error: Permission denied for tool: read_audit_log"


# --- prompts -------------------------------------------------------------------------------


def test_given_a_server_when_prompts_are_listed_then_the_text_and_the_names_come_back() -> None:
    text, names = asyncio.run(_make_service().list_prompts())
    assert text == "• TEST_PROMPT: TEST_DESC (args: topic)"
    assert names == ["TEST_PROMPT"]


def test_given_no_prompt_selected_when_fetched_then_it_asks_for_one() -> None:
    session = _make_session()
    text = asyncio.run(_make_service(session).get_prompt(None, "{}"))
    assert text == "Error: Please select a prompt from the dropdown first"
    session.get_prompt.assert_not_called()


def test_given_string_arguments_when_a_prompt_is_fetched_then_the_rendered_text_is_shown() -> None:
    session = _make_session()
    text = asyncio.run(_make_service(session).get_prompt("TEST_PROMPT", '{"topic": "T"}'))
    assert text == "--- Prompt: TEST_DESC ---\n\nuser: TEST_TEXT\n"
    session.get_prompt.assert_awaited_once_with("TEST_PROMPT", {"topic": "T"})


def test_given_a_non_string_argument_when_a_prompt_is_fetched_then_it_is_refused() -> None:
    session = _make_session()
    text = asyncio.run(_make_service(session).get_prompt("TEST_PROMPT", '{"topic": 1}'))
    assert text == "Error: Prompt arguments must be a JSON object of text values"
    session.get_prompt.assert_not_called()


def test_given_invalid_json_when_a_prompt_is_fetched_then_it_is_refused() -> None:
    session = _make_session()
    text = asyncio.run(_make_service(session).get_prompt("TEST_PROMPT", "{oops"))
    assert text == "Error: Prompt arguments must be a JSON object of text values"
    session.get_prompt.assert_not_called()
