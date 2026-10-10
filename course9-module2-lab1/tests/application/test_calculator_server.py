import asyncio
from unittest.mock import MagicMock

from fastmcp import Client
from mcp.types import TextContent, TextResourceContents

from application.calculator_server import build_server
from interfaces.document_store_interface import DocumentStoreInterface


def make_store(text: str | None) -> MagicMock:
    store = MagicMock(spec=DocumentStoreInterface)
    store.read.return_value = text
    return store


def test_server_offers_add_and_subtract() -> None:
    async def run() -> set[str]:
        async with Client(build_server(make_store(None))) as client:
            return {tool.name for tool in await client.list_tools()}

    assert asyncio.run(run()) == {"add", "subtract"}


def test_add_and_subtract_compute() -> None:
    async def run() -> tuple[object, object]:
        async with Client(build_server(make_store(None))) as client:
            added = await client.call_tool("add", {"a": 4, "b": 5})
            subtracted = await client.call_tool("subtract", {"a": 4, "b": 5})
            return added.data, subtracted.data

    assert asyncio.run(run()) == (9, -1)


def test_template_resource_does_not_touch_the_store() -> None:
    store = make_store("TEST_CONTENT")

    async def run() -> str:
        async with Client(build_server(store)) as client:
            contents = await client.read_resource("file:///endpoint/TEST_DOC.txt")
            assert isinstance(contents[0], TextResourceContents)
            return contents[0].text

    assert asyncio.run(run()) == "Document contents of TEST_DOC.txt"
    store.read.assert_not_called()


def test_document_resource_returns_the_stores_text() -> None:
    store = make_store("TEST_CONTENT")

    async def run() -> str:
        async with Client(build_server(store)) as client:
            contents = await client.read_resource("file://endpoint2/TEST_DOC.txt")
            assert isinstance(contents[0], TextResourceContents)
            return contents[0].text

    assert asyncio.run(run()) == "TEST_CONTENT"
    store.read.assert_called_once_with("TEST_DOC.txt")


def test_document_resource_reports_a_missing_document_as_text() -> None:
    async def run() -> str:
        async with Client(build_server(make_store(None))) as client:
            contents = await client.read_resource("file://endpoint2/TEST_MISSING.txt")
            assert isinstance(contents[0], TextResourceContents)
            return contents[0].text

    assert asyncio.run(run()) == "Document 'TEST_MISSING.txt' not found in path directory"


def test_review_code_prompt_wraps_the_code() -> None:
    async def run() -> str:
        async with Client(build_server(make_store(None))) as client:
            result = await client.get_prompt("review_code", {"code": "TEST_CODE"})
            content = result.messages[0].content
            assert isinstance(content, TextContent)
            return content.text

    assert asyncio.run(run()) == "Please review this code:\n\nTEST_CODE"
