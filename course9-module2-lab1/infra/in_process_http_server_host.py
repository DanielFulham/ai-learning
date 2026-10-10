import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp import FastMCP


class InProcessHttpServerHost:
    """Serves an MCP server over HTTP from inside the current process."""

    def __init__(self, port: int) -> None:
        self._port = port

    def is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", self._port))
                return False
            except OSError:
                return True

    @asynccontextmanager
    async def serve(self, server: FastMCP) -> AsyncIterator[None]:
        if self.is_port_in_use():
            raise RuntimeError(f"Port {self._port} is already in use; change PORT and rerun")
        server_task = asyncio.create_task(server.run_http_async(port=self._port, show_banner=False))
        try:
            while not self.is_port_in_use():  # True once the server has bound the port
                await asyncio.sleep(0.1)
            yield
        finally:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
