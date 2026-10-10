import asyncio
import socket

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport

from infra.in_process_http_server_host import InProcessHttpServerHost


def free_port() -> int:
    """A port that was free a moment ago."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


def make_server() -> FastMCP:
    server = FastMCP(name="TestServer")

    @server.tool
    def ping() -> str:
        """Reply with a fixed word."""
        return "pong"

    return server


def test_port_is_not_in_use_when_nothing_is_listening() -> None:
    assert InProcessHttpServerHost(free_port()).is_port_in_use() is False


def test_port_is_in_use_when_a_socket_holds_it() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        assert InProcessHttpServerHost(port).is_port_in_use() is True


def test_serve_refuses_a_port_that_is_already_in_use() -> None:
    async def run() -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(("127.0.0.1", 0))
            holder.listen()
            host = InProcessHttpServerHost(holder.getsockname()[1])
            async with host.serve(make_server()):
                pass

    with pytest.raises(RuntimeError, match="already in use"):
        asyncio.run(run())


def test_serve_makes_the_server_reachable_over_http() -> None:
    port = free_port()
    host = InProcessHttpServerHost(port)

    async def run() -> tuple[bool, str]:
        async with host.serve(make_server()):
            in_use = host.is_port_in_use()
            client = Client(StreamableHttpTransport(url=f"http://127.0.0.1:{port}/mcp"))
            async with client:
                result = await client.call_tool("ping", {})
            return in_use, str(result.data)

    assert asyncio.run(run()) == (True, "pong")
