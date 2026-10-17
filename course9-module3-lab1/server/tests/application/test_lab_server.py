import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Client
from mcp.types import CallToolResult, TextContent, TextResourceContents

from application.lab_server import build_server
from interfaces.file_store_interface import FileStoreInterface

TEST_NAME = "TEST_SERVER"


def _make_store(read_text: str = "TEST_CONTENT", exists: bool = True) -> MagicMock:
    store = MagicMock(spec=FileStoreInterface)
    store.read_text.return_value = read_text
    store.exists.return_value = exists
    return store


def _make_client(
    file_store: MagicMock | None = None, resource_store: MagicMock | None = None
) -> Client[Any]:
    server = build_server(file_store or _make_store(), resource_store or _make_store(), TEST_NAME)
    return Client(server)


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _call_tool(store: MagicMock, tool: str, args: dict[str, Any]) -> CallToolResult:
    async def run() -> CallToolResult:
        async with _make_client(file_store=store) as client:
            return await client.call_tool_mcp(tool, args)

    return asyncio.run(run())


def _read_resource(store: MagicMock, uri: str) -> str:
    async def run() -> str:
        async with _make_client(resource_store=store) as client:
            result = await client.read_resource_mcp(uri)
            block = result.contents[0]
            assert isinstance(block, TextResourceContents)
            return block.text

    return asyncio.run(run())


def _get_prompt(name: str, args: dict[str, Any]) -> str:
    async def run() -> str:
        async with _make_client() as client:
            result = await client.get_prompt_mcp(name, args)
            block = result.messages[0].content
            assert isinstance(block, TextContent)
            return block.text

    return asyncio.run(run())


def test_server_has_the_given_name() -> None:
    server = build_server(_make_store(), _make_store(), TEST_NAME)
    assert server.name == TEST_NAME


def test_lists_the_two_tools_with_the_lab_descriptions() -> None:
    async def run() -> dict[str, str | None]:
        async with _make_client() as client:
            return {t.name: t.description for t in await client.list_tools()}

    assert asyncio.run(run()) == {
        "echo": "Echo back the input text.",
        "write_file": "Write content to a file.",
    }


def test_lists_the_resource_template_as_the_lab_does() -> None:
    async def run() -> list[tuple[str, str]]:
        async with _make_client() as client:
            templates = await client.list_resource_templates()
            return [(t.name, t.uriTemplate) for t in templates]

    assert asyncio.run(run()) == [("read_resource_file", "file://resources/{filename}")]


def test_lists_the_prompt_with_its_argument() -> None:
    async def run() -> list[tuple[str, str | None, list[str]]]:
        async with _make_client() as client:
            prompts = await client.list_prompts()
            return [(p.name, p.description, [a.name for a in p.arguments or []]) for p in prompts]

    assert asyncio.run(run()) == [
        ("review_file", "Generate a prompt to review a file's contents.", ["filename"])
    ]


def test_echo_returns_the_text_with_a_prefix() -> None:
    result = _call_tool(_make_store(), "echo", {"text": "TEST_TEXT"})
    assert _text(result) == "Echo: TEST_TEXT"


def test_write_file_writes_through_the_store_and_reports_the_path() -> None:
    store = _make_store()
    result = _call_tool(store, "write_file", {"path": "a.txt", "content": "TEST_CONTENT"})
    store.write_text.assert_called_once_with("a.txt", "TEST_CONTENT")
    assert result.isError is False
    assert _text(result) == "Successfully wrote to a.txt"


def test_write_file_does_not_touch_the_resource_store() -> None:
    file_store = _make_store()
    resource_store = _make_store()

    async def run() -> None:
        async with _make_client(file_store, resource_store) as client:
            await client.call_tool_mcp("write_file", {"path": "a.txt", "content": "TEST_CONTENT"})

    asyncio.run(run())
    assert resource_store.mock_calls == []


def test_resource_reads_through_the_resource_store() -> None:
    store = _make_store(read_text="TEST_FILE_TEXT")
    assert _read_resource(store, "file://resources/a.txt") == "TEST_FILE_TEXT"
    store.exists.assert_called_once_with("a.txt")
    store.read_text.assert_called_once_with("a.txt")


def test_resource_for_a_missing_file_is_ordinary_text_not_an_error() -> None:
    # The lab returns the message as the resource content (compare Lab 43 F-L43-5).
    store = _make_store(exists=False)
    assert _read_resource(store, "file://resources/gone.txt") == "File not found: gone.txt"
    store.read_text.assert_not_called()


def test_resource_does_not_use_the_file_store() -> None:
    file_store = _make_store()

    async def run() -> None:
        async with _make_client(file_store=file_store) as client:
            await client.read_resource_mcp("file://resources/a.txt")

    asyncio.run(run())
    assert file_store.mock_calls == []


def test_prompt_names_the_file_and_lists_the_four_review_points() -> None:
    text = _get_prompt("review_file", {"filename": "TEST_FILE.txt"})
    assert text.startswith("Please review the file 'TEST_FILE.txt' and provide:")
    for point in (
        "1. A summary of its contents",
        "2. Key points or sections",
        "3. Any suggestions for improvement",
        "4. Overall quality assessment",
    ):
        assert point in text
    assert text.endswith("Use the appropriate tools to read the file if needed.")


def test_failing_write_is_an_error_result() -> None:
    # The lab's write_file has no handler, so a failing write reaches the client as an error.
    store = _make_store()
    store.write_text.side_effect = OSError("TEST_BOOM")
    result = _call_tool(store, "write_file", {"path": "a.txt", "content": "TEST_CONTENT"})
    assert result.isError is True
    assert "TEST_BOOM" in _text(result)


def test_failing_resource_read_is_an_error_not_text() -> None:
    # Only a missing file is turned into text; any other read failure is not caught.
    store = _make_store()
    store.read_text.side_effect = OSError("TEST_BOOM")
    with pytest.raises(Exception, match="TEST_BOOM"):
        _read_resource(store, "file://resources/a.txt")
