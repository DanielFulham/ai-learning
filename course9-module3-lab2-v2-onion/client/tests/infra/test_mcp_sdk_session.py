import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

import pytest
from mcp import types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.memory import MessageStream, create_client_server_memory_streams
from pydantic import AnyUrl

from infra.mcp_sdk_session import McpSdkSession

_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


def _make_test_server() -> Server[Any]:
    server: Server[Any] = Server("TEST_SERVER")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(name="echo", description="TEST_ECHO_DOC", inputSchema=_SCHEMA),
            types.Tool(name="fail", description=None, inputSchema={"type": "object"}),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name == "fail":
            raise ValueError("TEST_FAIL")
        return [
            types.TextContent(type="text", text=f"echo:{arguments['text']}"),
            types.TextContent(type="text", text="TEST_SECOND"),
        ]

    @server.list_resource_templates()
    async def list_resource_templates() -> list[types.ResourceTemplate]:
        return [
            types.ResourceTemplate(
                name="TEST_FILE", uriTemplate="file://workspace/{name}", description="TEST_DOC"
            ),
            types.ResourceTemplate(name="TEST_BARE", uriTemplate="file://bare/{name}"),
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        return [ReadResourceContents(content=f"TEST_FILE:{uri}", mime_type="text/plain")]

    @server.list_prompts()
    async def list_prompts() -> list[types.Prompt]:
        arguments = [types.PromptArgument(name="who", required=True)]
        return [types.Prompt(name="greet", description="TEST_PROMPT_DOC", arguments=arguments)]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        text = types.TextContent(type="text", text=f"Hello {(arguments or {})['who']}")
        return types.GetPromptResult(
            description="TEST_RENDERED",
            messages=[types.PromptMessage(role="user", content=text)],
        )

    return server


class _InMemoryTransport:
    """A test double transport: a real SDK server on the other end of in-memory streams."""

    def __init__(self, server: Server[Any]) -> None:
        self._server = server
        self.opened = 0

    @asynccontextmanager
    async def open(self) -> AsyncIterator[MessageStream]:
        self.opened += 1
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            server_read, server_write = server_streams
            options = self._server.create_initialization_options()
            task = asyncio.create_task(self._server.run(server_read, server_write, options))
            try:
                yield client_streams
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


def _make_session() -> McpSdkSession:
    return McpSdkSession(_InMemoryTransport(_make_test_server()))


def test_not_connected_raises() -> None:
    with pytest.raises(RuntimeError, match="connect"):
        asyncio.run(_make_session().list_tools())


def test_connecting_twice_opens_the_transport_once() -> None:
    # The lab's connect() is "safe to call multiple times".
    transport = _InMemoryTransport(_make_test_server())

    async def run() -> None:
        session = McpSdkSession(transport)
        await session.connect()
        try:
            await session.connect()
            await session.list_tools()
        finally:
            await session.close()

    asyncio.run(run())
    assert transport.opened == 1


class _SlowFakeClientSession:
    """Stands in for the SDK's session: a handshake that takes a moment, as a real one does."""

    def __init__(self, read: object, write: object) -> None:
        pass

    async def __aenter__(self) -> _SlowFakeClientSession:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def initialize(self) -> None:
        await asyncio.sleep(0)


class _SlowCountingTransport:
    def __init__(self) -> None:
        self.opened = 0

    @asynccontextmanager
    async def open(self) -> AsyncIterator[MessageStream]:
        self.opened += 1
        await asyncio.sleep(0)
        async with create_client_server_memory_streams() as (client_streams, _):
            yield client_streams


def test_connecting_at_the_same_time_opens_the_transport_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two Gradio events can start together. The lab's connect() checked a flag that was only set
    # after the handshake, so overlapping calls would each connect.
    monkeypatch.setattr("infra.mcp_sdk_session.ClientSession", _SlowFakeClientSession)
    transport = _SlowCountingTransport()

    async def run() -> None:
        session = McpSdkSession(transport)
        await asyncio.gather(session.connect(), session.connect(), session.connect())
        await session.close()

    asyncio.run(run())
    assert transport.opened == 1


def test_lists_tools_with_their_schema_and_an_empty_string_for_a_missing_description() -> None:
    async def run() -> list[tuple[str, str, dict[str, object]]]:
        session = _make_session()
        await session.connect()
        try:
            tools = await session.list_tools()
            return [(t["name"], t["description"], t["input_schema"]) for t in tools]
        finally:
            await session.close()

    assert asyncio.run(run()) == [
        ("echo", "TEST_ECHO_DOC", _SCHEMA),
        ("fail", "", {"type": "object"}),
    ]


def test_call_tool_returns_every_text_part_and_no_error() -> None:
    async def run() -> dict[str, Any]:
        session = _make_session()
        await session.connect()
        try:
            return dict(await session.call_tool("echo", {"text": "TEST_TEXT"}))
        finally:
            await session.close()

    assert asyncio.run(run()) == {"texts": ["echo:TEST_TEXT", "TEST_SECOND"], "is_error": False}


def test_call_tool_failure_is_returned_with_the_error_flag_not_raised() -> None:
    async def run() -> tuple[list[str], bool]:
        session = _make_session()
        await session.connect()
        try:
            result = await session.call_tool("fail", {})
            return result["texts"], result["is_error"]
        finally:
            await session.close()

    texts, is_error = asyncio.run(run())
    assert is_error is True
    assert "TEST_FAIL" in texts[0]


def test_lists_resource_templates_with_an_empty_string_for_a_missing_description() -> None:
    async def run() -> list[tuple[str, str, str]]:
        session = _make_session()
        await session.connect()
        try:
            found = await session.list_resource_templates()
            return [(t["name"], t["uri_template"], t["description"]) for t in found]
        finally:
            await session.close()

    assert asyncio.run(run()) == [
        ("TEST_FILE", "file://workspace/{name}", "TEST_DOC"),
        ("TEST_BARE", "file://bare/{name}", ""),
    ]


def test_read_resource_returns_the_text_and_keeps_the_uri() -> None:
    async def run() -> list[str]:
        session = _make_session()
        await session.connect()
        try:
            return await session.read_resource("file://workspace/a.txt")
        finally:
            await session.close()

    assert asyncio.run(run()) == ["TEST_FILE:file://workspace/a.txt"]


def test_lists_prompts_with_argument_names() -> None:
    async def run() -> list[Any]:
        session = _make_session()
        await session.connect()
        try:
            return list(await session.list_prompts())
        finally:
            await session.close()

    assert asyncio.run(run()) == [
        {"name": "greet", "description": "TEST_PROMPT_DOC", "argument_names": ["who"]}
    ]


def test_get_prompt_returns_the_description_and_message_text() -> None:
    async def run() -> dict[str, Any]:
        session = _make_session()
        await session.connect()
        try:
            return dict(await session.get_prompt("greet", {"who": "TEST"}))
        finally:
            await session.close()

    assert asyncio.run(run()) == {
        "description": "TEST_RENDERED",
        "messages": [{"role": "user", "text": "Hello TEST"}],
    }


def test_close_then_use_raises_not_connected() -> None:
    async def run() -> None:
        session = _make_session()
        await session.connect()
        await session.close()
        await session.list_tools()

    with pytest.raises(RuntimeError, match="connect"):
        asyncio.run(run())
