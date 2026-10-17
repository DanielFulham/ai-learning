import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Client
from mcp.types import CallToolResult, ReadResourceResult, TextContent, TextResourceContents

from application.container import initialise_server
from config import SERVER_NAME
from interfaces.file_store_interface import FileStoreInterface


def _make_store() -> MagicMock:
    store = MagicMock(spec=FileStoreInterface)
    store.exists.return_value = True
    store.read_text.return_value = "TEST_CONTENT"
    return store


def test_server_uses_the_configured_name() -> None:
    server = initialise_server(_make_store(), _make_store())
    assert server.name == SERVER_NAME


def test_each_call_builds_a_new_server() -> None:
    assert initialise_server(_make_store(), _make_store()) is not initialise_server(
        _make_store(), _make_store()
    )


def test_injected_stores_are_used_for_their_own_job() -> None:
    file_store = _make_store()
    resource_store = _make_store()
    server = initialise_server(file_store, resource_store)

    async def run() -> None:
        async with Client(server) as client:
            await client.call_tool_mcp("write_file", {"path": "a.txt", "content": "TEST_CONTENT"})
            await client.read_resource_mcp("file://resources/b.txt")

    asyncio.run(run())
    file_store.write_text.assert_called_once_with("a.txt", "TEST_CONTENT")
    resource_store.read_text.assert_called_once_with("b.txt")
    file_store.read_text.assert_not_called()
    resource_store.write_text.assert_not_called()


def test_server_exposes_tools_resources_and_prompts() -> None:
    server = initialise_server(_make_store(), _make_store())

    async def run() -> tuple[set[str], set[str], set[str]]:
        async with Client(server) as client:
            tools = {t.name for t in await client.list_tools()}
            templates = {t.uriTemplate for t in await client.list_resource_templates()}
            prompts = {p.name for p in await client.list_prompts()}
            return tools, templates, prompts

    tools, templates, prompts = asyncio.run(run())
    assert tools == {"echo", "write_file"}
    assert templates == {"file://resources/{filename}"}
    assert prompts == {"review_file"}


def test_default_stores_write_to_the_root_and_read_from_its_resources_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("application.container.ROOT_DIR", tmp_path)
    monkeypatch.setattr("application.container.RESOURCES_DIR", tmp_path / "resources")
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "r.txt").write_text("TEST_RESOURCE", encoding="utf-8")
    server = initialise_server()

    async def run() -> tuple[CallToolResult, ReadResourceResult]:
        async with Client(server) as client:
            written = await client.call_tool_mcp(
                "write_file", {"path": "sub/w.txt", "content": "TEST_WRITTEN"}
            )
            read = await client.read_resource_mcp("file://resources/r.txt")
            return written, read

    written, read = asyncio.run(run())
    assert isinstance(written.content[0], TextContent)
    assert (tmp_path / "sub" / "w.txt").read_text(encoding="utf-8") == "TEST_WRITTEN"
    block = read.contents[0]
    assert isinstance(block, TextResourceContents)
    assert block.text == "TEST_RESOURCE"


@pytest.mark.parametrize(
    "uri",
    [
        "file://resources/..%2Fsecret.txt",
        "file://resources/%2e%2e%2fsecret.txt",
        "file://resources/..%5Csecret.txt",
    ],
)
def test_encoded_traversal_in_a_resource_uri_does_not_reach_a_file_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uri: str
) -> None:
    # Before containment, the server decoded these into `../secret.txt` and returned the file.
    monkeypatch.setattr("application.container.ROOT_DIR", tmp_path)
    monkeypatch.setattr("application.container.RESOURCES_DIR", tmp_path / "resources")
    (tmp_path / "resources").mkdir()
    (tmp_path / "secret.txt").write_text("TEST_OUTSIDE_SECRET", encoding="utf-8")
    server = initialise_server()

    async def run() -> str:
        async with Client(server) as client:
            try:
                result = await client.read_resource_mcp(uri)
            except Exception as e:
                return f"ERROR: {e}"
            block = result.contents[0]
            assert isinstance(block, TextResourceContents)
            return block.text

    assert "TEST_OUTSIDE_SECRET" not in asyncio.run(run())


def test_a_write_outside_the_root_through_the_tool_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    monkeypatch.setattr("application.container.ROOT_DIR", root)
    monkeypatch.setattr("application.container.RESOURCES_DIR", root / "resources")
    server = initialise_server()

    async def run() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool_mcp(
                "write_file", {"path": "../escaped.txt", "content": "TEST_CONTENT"}
            )

    result = asyncio.run(run())
    assert result.isError is True
    assert not (tmp_path / "escaped.txt").exists()
