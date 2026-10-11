import asyncio
from unittest.mock import MagicMock

from fastmcp import Client

from application.container import initialise_server
from config import SERVER_NAME
from interfaces.file_store_interface import FileStoreInterface


def test_server_uses_the_configured_name() -> None:
    server = initialise_server(MagicMock(spec=FileStoreInterface))
    assert server.name == SERVER_NAME


def test_injected_file_store_is_used() -> None:
    store = MagicMock(spec=FileStoreInterface)
    store.kind_of.return_value = "missing"
    server = initialise_server(store)

    async def run() -> None:
        async with Client(server) as client:
            await client.call_tool_mcp("delete_file", {"file_path": "a.txt"})

    asyncio.run(run())
    store.kind_of.assert_called_once_with("a.txt")


def test_server_exposes_tools_resources_and_prompts() -> None:
    server = initialise_server(MagicMock(spec=FileStoreInterface))

    async def run() -> tuple[set[str], set[str], set[str]]:
        async with Client(server) as client:
            tools = {t.name for t in await client.list_tools()}
            templates = {t.uriTemplate for t in await client.list_resource_templates()}
            prompts = {p.name for p in await client.list_prompts()}
            return tools, templates, prompts

    tools, templates, prompts = asyncio.run(run())
    assert tools == {"write_file", "delete_file"}
    assert templates == {"file:///{file_name}"}
    assert prompts == {"code_review", "documentation_generator"}
