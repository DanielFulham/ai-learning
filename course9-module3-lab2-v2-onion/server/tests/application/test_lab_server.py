import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Client
from mcp.types import CallToolResult, TextContent, TextResourceContents

from application.lab_server import build_server
from interfaces.file_store_interface import FileStoreInterface, PathOutsideRootError

TEST_NAME = "TEST_SERVER"
DENIED = "Error: Access denied - path outside workspace roots"


def _make_store() -> MagicMock:
    store = MagicMock(spec=FileStoreInterface)
    store.kind_of.return_value = "file"
    store.read_text.return_value = "TEST_CONTENT"
    store.list_entries.return_value = []
    return store


def _make_client(store: MagicMock) -> Client[Any]:
    return Client(build_server(store, TEST_NAME))


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _call_tool(store: MagicMock, tool: str, args: dict[str, Any]) -> CallToolResult:
    async def run() -> CallToolResult:
        async with _make_client(store) as client:
            return await client.call_tool_mcp(tool, args)

    return asyncio.run(run())


def _read_resource(store: MagicMock, uri: str) -> str:
    async def run() -> str:
        async with _make_client(store) as client:
            result = await client.read_resource_mcp(uri)
            block = result.contents[0]
            assert isinstance(block, TextResourceContents)
            return block.text

    return asyncio.run(run())


def _get_prompt(name: str, args: dict[str, Any]) -> str:
    async def run() -> str:
        async with _make_client(_make_store()) as client:
            result = await client.get_prompt_mcp(name, args)
            block = result.messages[0].content
            assert isinstance(block, TextContent)
            return block.text

    return asyncio.run(run())


def test_server_has_the_given_name() -> None:
    assert build_server(_make_store(), TEST_NAME).name == TEST_NAME


def test_lists_the_four_tools_with_the_lab_descriptions() -> None:
    async def run() -> dict[str, str | None]:
        async with _make_client(_make_store()) as client:
            return {t.name: t.description for t in await client.list_tools()}

    descriptions = asyncio.run(run())
    assert set(descriptions) == {"read_file", "write_file", "list_files", "analyze_code"}
    assert descriptions["read_file"] == "Read a file from the workspace directory."
    assert descriptions["write_file"] == "Write content to a file in the workspace directory."
    assert descriptions["list_files"] == "List files in a directory within the workspace."


def test_lists_the_resource_template_as_the_lab_does() -> None:
    async def run() -> list[tuple[str, str]]:
        async with _make_client(_make_store()) as client:
            return [(t.name, t.uriTemplate) for t in await client.list_resource_templates()]

    assert asyncio.run(run()) == [("get_workspace_file", "file://workspace/{filename}")]


def test_lists_the_two_prompts_with_their_argument() -> None:
    async def run() -> list[tuple[str, list[str]]]:
        async with _make_client(_make_store()) as client:
            prompts = await client.list_prompts()
            return [(p.name, [a.name for a in p.arguments or []]) for p in prompts]

    assert sorted(asyncio.run(run())) == [
        ("analyze_security", ["filename"]),
        ("review_code", ["filename"]),
    ]


def test_read_file_returns_the_content() -> None:
    store = _make_store()
    result = _call_tool(store, "read_file", {"filepath": "a.txt"})
    store.read_text.assert_called_once_with("a.txt")
    assert _text(result) == "TEST_CONTENT"


def test_read_file_of_a_missing_file_is_error_text_not_an_error_result() -> None:
    # The lab returns every failure from the tools as ordinary text. Pinned on purpose.
    store = _make_store()
    store.kind_of.return_value = "missing"
    result = _call_tool(store, "read_file", {"filepath": "gone.txt"})
    assert result.isError is False
    assert _text(result) == "Error: File not found: gone.txt"
    store.read_text.assert_not_called()


def test_read_file_outside_the_root_is_denied_as_text() -> None:
    store = _make_store()
    store.kind_of.side_effect = PathOutsideRootError("outside")
    result = _call_tool(store, "read_file", {"filepath": "../x"})
    assert result.isError is False
    assert _text(result) == DENIED
    store.read_text.assert_not_called()


def test_read_file_failure_is_reported_as_text() -> None:
    store = _make_store()
    store.read_text.side_effect = OSError("TEST_BOOM")
    assert _text(_call_tool(store, "read_file", {"filepath": "a.txt"})) == (
        "Error reading file: TEST_BOOM"
    )


def test_write_file_writes_through_the_store_and_reports_the_length() -> None:
    store = _make_store()
    result = _call_tool(store, "write_file", {"filepath": "a.txt", "content": "TEST_CONTENT"})
    store.write_text.assert_called_once_with("a.txt", "TEST_CONTENT")
    assert _text(result) == "Successfully wrote 12 characters to a.txt"


def test_write_file_outside_the_root_is_denied_as_text() -> None:
    store = _make_store()
    store.write_text.side_effect = PathOutsideRootError("outside")
    result = _call_tool(store, "write_file", {"filepath": "../x", "content": "TEST_CONTENT"})
    assert result.isError is False
    assert _text(result) == DENIED


def test_write_file_failure_is_reported_as_text() -> None:
    store = _make_store()
    store.write_text.side_effect = OSError("TEST_BOOM")
    result = _call_tool(store, "write_file", {"filepath": "a.txt", "content": "x"})
    assert _text(result) == "Error writing file: TEST_BOOM"


def test_list_files_formats_each_entry_and_defaults_to_the_root() -> None:
    store = _make_store()
    store.kind_of.return_value = "directory"
    store.list_entries.return_value = [
        {"name": "a.txt", "path": "a.txt", "is_dir": False, "size": 50},
        {"name": "sub", "path": "sub", "is_dir": True, "size": 0},
    ]
    result = _call_tool(store, "list_files", {})
    store.kind_of.assert_called_once_with(".")
    assert _text(result) == "FILE: a.txt (50 bytes)\nDIR: sub (0 bytes)"


def test_list_files_of_an_empty_folder_says_so() -> None:
    store = _make_store()
    store.kind_of.return_value = "directory"
    assert _text(_call_tool(store, "list_files", {})) == "Directory is empty"


def test_list_files_of_a_missing_folder_and_of_a_file() -> None:
    store = _make_store()
    store.kind_of.return_value = "missing"
    assert _text(_call_tool(store, "list_files", {"directory": "nope"})) == (
        "Error: Directory not found: nope"
    )
    store.kind_of.return_value = "file"
    assert _text(_call_tool(store, "list_files", {"directory": "a.txt"})) == (
        "Error: Not a directory: a.txt"
    )


def test_list_files_outside_the_root_is_denied_as_text() -> None:
    store = _make_store()
    store.kind_of.side_effect = PathOutsideRootError("outside")
    assert _text(_call_tool(store, "list_files", {"directory": ".."})) == DENIED


def test_list_files_failure_is_reported_as_text() -> None:
    store = _make_store()
    store.kind_of.return_value = "directory"
    store.list_entries.side_effect = OSError("TEST_BOOM")
    assert _text(_call_tool(store, "list_files", {})) == "Error listing directory: TEST_BOOM"


def test_analyze_code_is_the_lab_stub_and_sends_no_request() -> None:
    store = _make_store()
    text = _text(_call_tool(store, "analyze_code", {"code": "TEST_CODE", "focus": "TEST_FOCUS"}))
    assert text.startswith("[SAMPLING TRIGGER]")
    assert "Analyze this code for TEST_FOCUS:" in text
    assert "TEST_CODE..." in text
    assert text.endswith("This simplified version demonstrates the concept.")
    assert store.mock_calls == []


def test_analyze_code_shows_only_the_first_fifty_characters_and_defaults_the_focus() -> None:
    text = _text(_call_tool(_make_store(), "analyze_code", {"code": "x" * 60}))
    assert "Analyze this code for quality:" in text
    assert ("x" * 50 + "...") in text
    assert ("x" * 51) not in text


def test_resource_reads_the_file() -> None:
    store = _make_store()
    assert _read_resource(store, "file://workspace/a.txt") == "TEST_CONTENT"
    store.read_text.assert_called_once_with("a.txt")


def test_resource_for_a_missing_file_raises_unlike_the_tools() -> None:
    store = _make_store()
    store.kind_of.return_value = "missing"
    with pytest.raises(Exception, match="File not found: gone.txt"):
        _read_resource(store, "file://workspace/gone.txt")


def test_resource_outside_the_root_raises() -> None:
    store = _make_store()
    store.kind_of.side_effect = PathOutsideRootError("outside")
    with pytest.raises(Exception, match="Access denied - path outside workspace roots"):
        _read_resource(store, "file://workspace/x")
    store.read_text.assert_not_called()


def test_review_code_prompt_names_the_file_and_lists_the_five_points() -> None:
    text = _get_prompt("review_code", {"filename": "TEST_FILE.py"})
    assert text.startswith("Please review the code in file 'TEST_FILE.py' and provide:")
    for point in ("1. A summary of what the code does", "5. Code quality assessment"):
        assert point in text
    assert text.endswith("Focus on readability, maintainability, and best practices.")


def test_analyze_security_prompt_names_the_file_and_lists_the_five_points() -> None:
    text = _get_prompt("analyze_security", {"filename": "TEST_FILE.py"})
    assert text.startswith("Perform a security analysis of 'TEST_FILE.py' focusing on:")
    for point in ("1. Input validation and sanitization", "5. Error handling security"):
        assert point in text
    assert text.endswith("Provide specific line numbers and remediation suggestions.")
