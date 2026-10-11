import asyncio

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

from infra.fastmcp_session import FastmcpSession


def _make_test_server() -> FastMCP:
    server = FastMCP("TEST_SERVER")

    @server.tool()
    async def echo(text: str, ctx: Context) -> str:
        """TEST_ECHO_DOC"""
        await ctx.report_progress(progress=1, total=2, message="TEST_PROGRESS")
        return f"echo:{text}"

    @server.tool()
    async def fail() -> str:
        """Always fails."""
        raise ValueError("TEST_FAIL")

    @server.tool()
    async def ask_name(ctx: Context) -> str:
        """Asks the client for a name."""
        from fastmcp.server.elicitation import AcceptedElicitation
        from pydantic import BaseModel

        class Name(BaseModel):
            first: str
            last: str

        result = await ctx.elicit(message="TEST_WHO", response_type=Name)
        if isinstance(result, AcceptedElicitation):
            return f"{result.data.first} {result.data.last}"
        return f"not accepted: {result.action}"

    @server.resource("file:///{name}")
    async def a_file(name: str) -> str:
        return f"TEST_FILE:{name}"

    @server.prompt()
    async def greet(who: str) -> str:
        """TEST_PROMPT_DOC"""
        return f"Hello {who}"

    return server


class _Recorder:
    def __init__(self, elicitation_answer: dict[str, str] | None = None) -> None:
        self.elicitations: list[tuple[str, list[str]]] = []
        self.progress: list[tuple[float, float | None, str | None]] = []
        self.notifications: list[str] = []
        self._answer = elicitation_answer

    async def on_elicitation(self, message: str, fields: list[str]) -> dict[str, str] | None:
        self.elicitations.append((message, fields))
        return self._answer

    async def on_progress(self, progress: float, total: float | None, message: str | None) -> None:
        self.progress.append((progress, total, message))

    async def on_notification(self, method: str) -> None:
        self.notifications.append(method)


def _make_session(recorder: _Recorder) -> FastmcpSession:
    return FastmcpSession(
        _make_test_server(),
        recorder.on_elicitation,
        recorder.on_progress,
        recorder.on_notification,
    )


def test_not_connected_raises() -> None:
    session = _make_session(_Recorder())
    with pytest.raises(RuntimeError, match="connect"):
        asyncio.run(session.list_tools())


def test_lists_tools_and_prompts_as_primitives() -> None:
    async def run() -> None:
        session = _make_session(_Recorder())
        await session.connect()
        try:
            tools = {t["name"]: t for t in await session.list_tools()}
            assert tools["echo"]["description"] == "TEST_ECHO_DOC"
            assert "text" in tools["echo"]["input_schema"]["properties"]
            prompts = await session.list_prompts()
            assert prompts == [{"name": "greet", "arguments": [{"name": "who", "required": True}]}]
        finally:
            await session.close()

    asyncio.run(run())


def test_call_tool_returns_text_and_reports_progress() -> None:
    recorder = _Recorder()

    async def run() -> None:
        session = _make_session(recorder)
        await session.connect()
        try:
            result = await session.call_tool("echo", {"text": "hi"})
            assert result == {"text": "echo:hi", "is_error": False}
        finally:
            await session.close()

    asyncio.run(run())
    assert recorder.progress == [(1, 2, "TEST_PROGRESS")]


def test_call_tool_failure_is_returned_not_raised() -> None:
    async def run() -> None:
        session = _make_session(_Recorder())
        await session.connect()
        try:
            result = await session.call_tool("fail", {})
            assert result["is_error"] is True
            assert "TEST_FAIL" in result["text"]
        finally:
            await session.close()

    asyncio.run(run())


def test_get_prompt_text_and_read_resource_text() -> None:
    async def run() -> None:
        session = _make_session(_Recorder())
        await session.connect()
        try:
            assert await session.get_prompt_text("greet", {"who": "TEST"}) == "Hello TEST"
            assert await session.read_resource_text("file:///x.txt") == "TEST_FILE:x.txt"
        finally:
            await session.close()

    asyncio.run(run())


def test_elicitation_accepted_returns_the_values() -> None:
    recorder = _Recorder({"first": "TEST_A", "last": "TEST_B"})

    async def run() -> str:
        session = _make_session(recorder)
        await session.connect()
        try:
            return (await session.call_tool("ask_name", {}))["text"]
        finally:
            await session.close()

    assert asyncio.run(run()) == "TEST_A TEST_B"
    assert recorder.elicitations == [("TEST_WHO", ["first", "last"])]


def test_elicitation_declined() -> None:
    async def run() -> str:
        session = _make_session(_Recorder(None))
        await session.connect()
        try:
            return (await session.call_tool("ask_name", {}))["text"]
        finally:
            await session.close()

    assert asyncio.run(run()) == "not accepted: decline"


def test_for_http_targets_the_url() -> None:
    recorder = _Recorder()
    session = FastmcpSession.for_http(
        "http://127.0.0.1:1/mcp",
        recorder.on_elicitation,
        recorder.on_progress,
        recorder.on_notification,
    )
    assert isinstance(session._target, StreamableHttpTransport)
    assert session._target.url == "http://127.0.0.1:1/mcp"


def test_for_stdio_launches_the_command() -> None:
    recorder = _Recorder()
    session = FastmcpSession.for_stdio(
        "TEST_COMMAND",
        ["TEST_SCRIPT.py"],
        recorder.on_elicitation,
        recorder.on_progress,
        recorder.on_notification,
    )
    assert isinstance(session._target, StdioTransport)
    assert session._target.command == "TEST_COMMAND"
    assert session._target.args == ["TEST_SCRIPT.py"]
