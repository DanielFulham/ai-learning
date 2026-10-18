import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Client, FastMCP
from mcp.types import CallToolResult, TextContent, TextResourceContents

from application.container import initialise_server
from config import SERVER_NAME
from interfaces.file_store_interface import FileStoreInterface

DENIED = "Error: Access denied - path outside workspace roots"


def _make_store() -> MagicMock:
    store = MagicMock(spec=FileStoreInterface)
    store.kind_of.return_value = "file"
    store.read_text.return_value = "TEST_CONTENT"
    return store


def _use_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr("application.container.ROOT_DIR", root)


def _call_tool(server: FastMCP, tool: str, args: dict[str, Any]) -> str:
    async def run() -> str:
        async with Client(server) as client:
            result: CallToolResult = await client.call_tool_mcp(tool, args)
            block = result.content[0]
            assert isinstance(block, TextContent)
            return block.text

    return asyncio.run(run())


def _read_resource_outcome(server: FastMCP, uri: str) -> str:
    """The resource's text, or "ERROR: ..." if reading it raised."""

    async def run() -> str:
        async with Client(server) as client:
            try:
                result = await client.read_resource_mcp(uri)
            except Exception as e:
                return f"ERROR: {e}"
            block = result.contents[0]
            assert isinstance(block, TextResourceContents)
            return block.text

    return asyncio.run(run())


def test_server_uses_the_configured_name() -> None:
    assert initialise_server(_make_store()).name == SERVER_NAME


def test_each_call_builds_a_new_server() -> None:
    assert initialise_server(_make_store()) is not initialise_server(_make_store())


def test_injected_store_is_used() -> None:
    store = _make_store()
    server = initialise_server(store)
    assert _call_tool(server, "read_file", {"filepath": "a.txt"}) == "TEST_CONTENT"
    store.read_text.assert_called_once_with("a.txt")


def test_server_exposes_tools_resources_and_prompts() -> None:
    server = initialise_server(_make_store())

    async def run() -> tuple[set[str], set[str], set[str]]:
        async with Client(server) as client:
            tools = {t.name for t in await client.list_tools()}
            templates = {t.uriTemplate for t in await client.list_resource_templates()}
            prompts = {p.name for p in await client.list_prompts()}
            return tools, templates, prompts

    tools, templates, prompts = asyncio.run(run())
    assert tools == {"read_file", "write_file", "list_files", "analyze_code"}
    assert templates == {"file://workspace/{filename}"}
    assert prompts == {"review_code", "analyze_security"}


def test_default_store_is_rooted_at_the_configured_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    _use_root(monkeypatch, root)
    server = initialise_server()
    assert root.is_dir()
    assert _call_tool(server, "list_files", {}) == "Directory is empty"
    _call_tool(server, "write_file", {"filepath": "sub/a.txt", "content": "TEST_CONTENT"})
    assert (root / "sub" / "a.txt").read_text(encoding="utf-8") == "TEST_CONTENT"
    assert _call_tool(server, "read_file", {"filepath": "sub/a.txt"}) == "TEST_CONTENT"


def test_a_path_that_goes_up_and_back_inside_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    _use_root(monkeypatch, root)
    server = initialise_server()
    _call_tool(server, "write_file", {"filepath": "a.txt", "content": "TEST_CONTENT"})
    assert _call_tool(server, "read_file", {"filepath": "sub/../a.txt"}) == "TEST_CONTENT"


def test_tools_refuse_paths_outside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    _use_root(monkeypatch, root)
    (tmp_path / "secret.txt").write_text("TEST_OUTSIDE_SECRET", encoding="utf-8")
    server = initialise_server()
    assert _call_tool(server, "read_file", {"filepath": "../secret.txt"}) == DENIED
    assert _call_tool(server, "read_file", {"filepath": str(tmp_path / "secret.txt")}) == DENIED
    assert _call_tool(server, "list_files", {"directory": ".."}) == DENIED
    outside = {"filepath": "../escaped.txt", "content": "TEST_CONTENT"}
    assert _call_tool(server, "write_file", outside) == DENIED
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize(
    "uri",
    [
        "file://workspace/..%2Fsecret.txt",
        "file://workspace/%2e%2e%2fsecret.txt",
        "file://workspace/..%5Csecret.txt",
    ],
)
def test_encoded_traversal_in_a_resource_uri_does_not_reach_a_file_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uri: str
) -> None:
    # In Lab 45 the server decoded these into `../secret.txt` before the check.
    root = tmp_path / "root"
    _use_root(monkeypatch, root)
    (tmp_path / "secret.txt").write_text("TEST_OUTSIDE_SECRET", encoding="utf-8")
    server = initialise_server()
    assert "TEST_OUTSIDE_SECRET" not in _read_resource_outcome(server, uri)


def test_resource_reads_a_file_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    _use_root(monkeypatch, root)
    server = initialise_server()
    _call_tool(server, "write_file", {"filepath": "ok.txt", "content": "TEST_CONTENT"})
    assert _read_resource_outcome(server, "file://workspace/ok.txt") == "TEST_CONTENT"
