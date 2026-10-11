import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from fastmcp import Client
from fastmcp.client.elicitation import ElicitResult
from fastmcp.client.progress import ProgressHandler
from mcp.shared.context import RequestContext
from mcp.types import (
    CallToolResult,
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    TextContent,
    TextResourceContents,
)

from application.file_server import DocumentGeneratorSchema, build_server
from interfaces.file_store_interface import FileStoreInterface, PathOutsideRootError

TEST_NAME = "Test Server"

Action = Literal["accept", "decline", "cancel"]

ElicitationHandler = Callable[
    [
        str,
        type[DocumentGeneratorSchema] | None,
        ElicitRequestFormParams | ElicitRequestURLParams,
        RequestContext[Any, Any, Any],
    ],
    Awaitable[DocumentGeneratorSchema | ElicitResult[DocumentGeneratorSchema]],
]


def _make_store() -> MagicMock:
    store = MagicMock(spec=FileStoreInterface)
    store.kind_of.return_value = "file"
    store.read_text.return_value = "TEST_CODE"
    store.list_entries.return_value = []
    return store


def _make_client(store: MagicMock, handler: ElicitationHandler | None = None) -> Client[Any]:
    server = build_server(store, TEST_NAME, chunk_delay=0)
    return Client(server, elicitation_handler=handler)


def _text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def _call_tool(
    store: MagicMock, tool: str, args: dict[str, Any], progress: ProgressHandler | None = None
) -> CallToolResult:
    async def run() -> CallToolResult:
        async with _make_client(store) as client:
            return await client.call_tool_mcp(tool, args, progress_handler=progress)

    return asyncio.run(run())


def _read_resource(store: MagicMock, uri: str) -> str:
    async def run() -> str:
        async with _make_client(store) as client:
            result = await client.read_resource_mcp(uri)
            block = result.contents[0]
            assert isinstance(block, TextResourceContents)
            return block.text

    return asyncio.run(run())


def _get_prompt(
    store: MagicMock,
    name: str,
    args: dict[str, Any] | None = None,
    handler: ElicitationHandler | None = None,
) -> str:
    async def run() -> str:
        async with _make_client(store, handler) as client:
            result = await client.get_prompt_mcp(name, args)
            block = result.messages[0].content
            assert isinstance(block, TextContent)
            return block.text

    return asyncio.run(run())


def _make_elicitation_handler(action: Action) -> ElicitationHandler:
    async def handler(
        message: str,
        response_type: type[DocumentGeneratorSchema] | None,
        params: ElicitRequestFormParams | ElicitRequestURLParams,
        context: RequestContext[Any, Any, Any],
    ) -> DocumentGeneratorSchema | ElicitResult[DocumentGeneratorSchema]:
        if action == "accept":
            return DocumentGeneratorSchema(file_path="m.py", name="TEST_DOC")
        return ElicitResult(action=action)

    return handler


def test_write_file_creates_then_appends_chunks() -> None:
    store = _make_store()
    result = _call_tool(store, "write_file", {"file_path": "a.txt", "content": "x" * 20})
    assert result.isError is False
    calls = store.write_text.call_args_list
    assert calls[0].args == ("a.txt", "")
    assert "".join(c.args[1] for c in calls[1:]) == "x" * 20
    assert all(c.kwargs == {"append": True} for c in calls[1:])


def test_write_file_empty_content_still_creates_file() -> None:
    store = _make_store()
    _call_tool(store, "write_file", {"file_path": "a.txt", "content": ""})
    store.write_text.assert_called_once_with("a.txt", "")


def test_write_file_reports_progress() -> None:
    store = _make_store()
    seen: list[tuple[float, float | None, str | None]] = []

    async def handler(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))

    args = {"file_path": "a.txt", "content": "x" * 20}
    _call_tool(store, "write_file", args, progress=handler)
    assert seen[-1] == (20, 20, "Write complete")
    assert len(seen) == 11


def test_concurrent_writes_do_not_interleave() -> None:
    store = _make_store()

    async def run() -> None:
        async with _make_client(store) as client:
            await asyncio.gather(
                client.call_tool_mcp("write_file", {"file_path": "a.txt", "content": "a" * 20}),
                client.call_tool_mcp("write_file", {"file_path": "a.txt", "content": "b" * 20}),
            )

    asyncio.run(run())
    contents = [c.args[1] for c in store.write_text.call_args_list]
    # Each write is one empty create plus 10 chunks, so with the lock the second
    # create comes right after the first write's 10 chunks.
    assert [i for i, text in enumerate(contents) if text == ""] == [0, 11]
    assert len(set(contents[1:11])) == 1
    assert len(set(contents[12:])) == 1
    assert contents[1] != contents[12]


def test_write_file_outside_root_is_an_error() -> None:
    store = _make_store()
    store.write_text.side_effect = PathOutsideRootError("outside")
    result = _call_tool(store, "write_file", {"file_path": "../x", "content": "hi"})
    assert result.isError is True


def test_delete_file_deletes_a_file() -> None:
    store = _make_store()
    result = _call_tool(store, "delete_file", {"file_path": "a.txt"})
    store.delete.assert_called_once_with("a.txt")
    assert "Successfully deleted" in _text(result)


@pytest.mark.parametrize(
    ("kind", "text"), [("directory", "is a directory"), ("missing", "not found")]
)
def test_delete_file_refuses_directory_and_missing(kind: str, text: str) -> None:
    store = _make_store()
    store.kind_of.return_value = kind
    result = _call_tool(store, "delete_file", {"file_path": "p"})
    store.delete.assert_not_called()
    assert text in _text(result)


def test_delete_file_outside_root_returns_text_not_error() -> None:
    store = _make_store()
    store.kind_of.side_effect = PathOutsideRootError("outside")
    result = _call_tool(store, "delete_file", {"file_path": "../x"})
    assert result.isError is False
    assert "Error deleting file" in _text(result)


def test_read_file_resource() -> None:
    assert "TEST_CODE" in _read_resource(_make_store(), "file:///a.txt")


def test_read_file_resource_not_a_file() -> None:
    store = _make_store()
    store.kind_of.return_value = "missing"
    assert "not a valid file" in _read_resource(store, "file:///a.txt")


def test_list_files_resource_maps_entries() -> None:
    store = _make_store()
    store.kind_of.return_value = "directory"
    store.list_entries.return_value = [
        {
            "name": "a.txt",
            "path": "a.txt",
            "is_dir": False,
            "size": 3,
            "modified": 0.0,
            "created": 0.0,
        }
    ]
    text = _read_resource(store, "dir://.")
    assert '"name": "a.txt"' in text
    assert '"type": "file"' in text


def test_list_files_resource_root_not_a_directory() -> None:
    store = _make_store()
    store.kind_of.return_value = "missing"
    assert "not a valid directory" in _read_resource(store, "dir://.")


def test_code_review_prompt() -> None:
    text = _get_prompt(_make_store(), "code_review", {"file_path": "m.py"})
    assert "TEST_CODE" in text
    assert ".py" in text


def test_code_review_missing_file_is_an_error() -> None:
    store = _make_store()
    store.kind_of.return_value = "missing"
    with pytest.raises(Exception, match="not a valid file"):
        _get_prompt(store, "code_review", {"file_path": "m.py"})


def test_documentation_generator_accepted() -> None:
    handler = _make_elicitation_handler("accept")
    text = _get_prompt(_make_store(), "documentation_generator", handler=handler)
    assert "TEST_DOC" in text
    assert "TEST_CODE" in text


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_documentation_generator_not_accepted(action: Literal["decline", "cancel"]) -> None:
    handler = _make_elicitation_handler(action)
    with pytest.raises(Exception, match="not accepted"):
        _get_prompt(_make_store(), "documentation_generator", handler=handler)
