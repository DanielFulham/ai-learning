import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from application.chat_service import ChatService
from application.client_service import ClientService
from interfaces.mcp_session_interface import McpSessionInterface
from interfaces.user_input_interface import UserInputInterface


def _make_session() -> MagicMock:
    session = MagicMock(spec=McpSessionInterface)
    session.list_prompts.return_value = [
        {
            "name": "code_review",
            "arguments": [{"name": "file_path", "required": True}],
        },
        {"name": "documentation_generator", "arguments": []},
    ]
    return session


def _make_chat() -> MagicMock:
    chat = MagicMock(spec=ChatService)
    chat.process_query = AsyncMock(return_value="TEST_ANSWER")
    return chat


def _run(session: MagicMock, chat: MagicMock, *answers: str) -> list[str]:
    """Run the client with scripted answers; return everything it showed."""
    user_input = MagicMock(spec=UserInputInterface)
    user_input.ask.side_effect = list(answers)
    asyncio.run(ClientService(chat, session, user_input).run())
    return [c.args[0] for c in user_input.show.call_args_list]


def test_quit_connects_then_closes() -> None:
    session = _make_session()
    shown = _run(session, _make_chat(), "q")
    session.connect.assert_awaited_once()
    session.close.assert_awaited_once()
    assert "Exiting client..." in shown


def test_invalid_choice_is_reported_and_menu_continues() -> None:
    shown = _run(_make_session(), _make_chat(), "9", "quit")
    assert "Invalid choice. Please try again." in shown
    assert "Exiting client..." in shown


def test_connect_failure_is_shown_and_session_still_closed() -> None:
    session = _make_session()
    session.connect.side_effect = RuntimeError("TEST_BOOM")
    shown = _run(session, _make_chat())
    assert "Error: RuntimeError: TEST_BOOM" in shown
    session.close.assert_awaited_once()


def test_read_file_encodes_the_name_and_shows_content() -> None:
    session = _make_session()
    session.read_resource_text.return_value = json.dumps({"file_content": "TEST_TEXT"})
    shown = _run(session, _make_chat(), "3", "sub/a b.txt", "q")
    session.read_resource_text.assert_awaited_once_with("file:///sub%2Fa%20b.txt")
    assert "File Content:\n TEST_TEXT" in shown


def test_read_file_shows_the_servers_error() -> None:
    session = _make_session()
    session.read_resource_text.return_value = json.dumps({"error": "TEST_ERROR"})
    shown = _run(session, _make_chat(), "3", "a.txt", "q")
    assert "Error reading file: TEST_ERROR" in shown


def test_read_dir_shows_a_listing() -> None:
    session = _make_session()
    items = [{"name": "a.txt", "type": "file", "size": 3, "modified": "TEST_TIME"}]
    session.read_resource_text.return_value = json.dumps({"items": items})
    shown = _run(session, _make_chat(), "4", "q")
    session.read_resource_text.assert_awaited_once_with("dir://.")
    assert any("a.txt" in line for line in shown)


def test_review_prompt_asks_for_arguments_and_gives_the_model_no_tools() -> None:
    session = _make_session()
    session.get_prompt_text.return_value = "TEST_PROMPT_TEXT"
    chat = _make_chat()
    shown = _run(session, chat, "2", "m.py", "q")
    session.get_prompt_text.assert_awaited_once_with("code_review", {"file_path": "m.py"})
    chat.process_query.assert_awaited_once_with("TEST_PROMPT_TEXT", use_tools=False)
    assert "TEST_ANSWER" in shown


def test_prompt_without_a_required_argument_is_not_run() -> None:
    session = _make_session()
    shown = _run(session, _make_chat(), "2", "", "q")
    assert "Error: file_path is required" in shown
    session.get_prompt_text.assert_not_called()


def test_documentation_prompt_takes_no_arguments_here() -> None:
    session = _make_session()
    session.get_prompt_text.return_value = "TEST_PROMPT_TEXT"
    chat = _make_chat()
    _run(session, chat, "1", "q")
    session.get_prompt_text.assert_awaited_once_with("documentation_generator", {})
    chat.process_query.assert_awaited_once_with("TEST_PROMPT_TEXT", use_tools=True)


def test_missing_prompt_is_reported() -> None:
    session = _make_session()
    session.list_prompts.return_value = []
    shown = _run(session, _make_chat(), "2", "q")
    assert "Prompt 'code_review' not found" in shown


def test_prompt_failure_is_shown_and_menu_continues() -> None:
    session = _make_session()
    session.get_prompt_text.side_effect = ValueError("TEST_BOOM")
    shown = _run(session, _make_chat(), "1", "q")
    assert "Error: ValueError: TEST_BOOM\n" in shown
    assert "Exiting client..." in shown


def test_converse_answers_until_quit() -> None:
    chat = _make_chat()
    shown = _run(_make_session(), chat, "5", "TEST_QUERY", "", "q", "q")
    chat.process_query.assert_awaited_once_with("TEST_QUERY")
    assert "\nTEST_ANSWER" in shown
    assert "Please enter a query" in shown


def test_converse_shows_a_failed_query_and_carries_on() -> None:
    chat = _make_chat()
    chat.process_query.side_effect = RuntimeError("TEST_BOOM")
    shown = _run(_make_session(), chat, "5", "TEST_QUERY", "q", "q")
    assert "Error processing query: TEST_BOOM" in shown
