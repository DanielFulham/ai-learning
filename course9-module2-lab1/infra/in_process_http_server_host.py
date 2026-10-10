import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
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
        config = uvicorn.Config(server.http_app(), host="127.0.0.1", port=self._port)
        uvicorn_server = uvicorn.Server(config)
        server_task = asyncio.create_task(uvicorn_server.serve())
        try:
            while not uvicorn_server.started:
                if server_task.done():
                    raise RuntimeError(f"HTTP server on port {self._port} failed to start")
                await asyncio.sleep(0.05)
            yield
        finally:
            uvicorn_server.should_exit = True  # graceful stop, so lifespan shutdown completes
            await asyncio.gather(server_task, return_exceptions=True)
