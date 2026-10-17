import asyncio
import sys
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from mcp import ClientSession, StdioServerParameters

from infra.stdio_transport import StdioTransport

_TEST_SERVER = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("TEST_SERVER")


    @server.tool()
    def echo(text: str) -> str:
        return f"echo:{text}"


    if __name__ == "__main__":
        server.run()
    """
)


def _make_fake_stdio_client() -> MagicMock:
    @asynccontextmanager
    async def fake(params: StdioServerParameters) -> AsyncIterator[tuple[str, str]]:
        yield "TEST_READ", "TEST_WRITE"

    return MagicMock(side_effect=fake)


def test_open_launches_the_given_command_and_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _make_fake_stdio_client()
    monkeypatch.setattr("infra.stdio_transport.stdio_client", fake)

    async def run() -> None:
        async with StdioTransport("TEST_COMMAND", ["TEST_SCRIPT.py"]).open():
            pass

    asyncio.run(run())
    params = fake.call_args.args[0]
    assert isinstance(params, StdioServerParameters)
    assert (params.command, params.args) == ("TEST_COMMAND", ["TEST_SCRIPT.py"])


def test_open_inherits_the_environment_rather_than_setting_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The lab passes env=None. Pinned so a change is deliberate.
    fake = _make_fake_stdio_client()
    monkeypatch.setattr("infra.stdio_transport.stdio_client", fake)

    async def run() -> None:
        async with StdioTransport("TEST_COMMAND", []).open():
            pass

    asyncio.run(run())
    assert fake.call_args.args[0].env is None


def test_open_yields_the_streams_the_sdk_gives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("infra.stdio_transport.stdio_client", _make_fake_stdio_client())
    seen: list[object] = []

    async def run() -> None:
        async with StdioTransport("TEST_COMMAND", []).open() as streams:
            seen.extend(streams)

    asyncio.run(run())
    assert seen == ["TEST_READ", "TEST_WRITE"]


def test_open_connects_a_real_session_to_a_stdio_server(tmp_path: Path) -> None:
    script = tmp_path / "server_script.py"
    script.write_text(_TEST_SERVER, encoding="utf-8")

    async def run() -> list[str]:
        async with StdioTransport(sys.executable, [str(script)]).open() as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return [t.name for t in (await session.list_tools()).tools]

    assert asyncio.run(run()) == ["echo"]
