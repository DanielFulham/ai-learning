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

    async def _run(self, uvicorn_server: uvicorn.Server) -> None:
        try:
            await uvicorn_server.serve()
        except SystemExit as exc:  # uvicorn exits instead of raising when it cannot bind
            raise RuntimeError(f"HTTP server on port {self._port} failed to start") from exc

    @asynccontextmanager
    async def serve(self, server: FastMCP) -> AsyncIterator[None]:
        if self.is_port_in_use():
            raise RuntimeError(f"Port {self._port} is already in use; change PORT and rerun")
        config = uvicorn.Config(server.http_app(), host="127.0.0.1", port=self._port)
        uvicorn_server = uvicorn.Server(config)
        server_task = asyncio.create_task(self._run(uvicorn_server))
        try:
            while not uvicorn_server.started:
                if server_task.done():
                    server_task.result()  # re-raises the startup failure
                    raise RuntimeError(f"HTTP server on port {self._port} stopped before starting")
                await asyncio.sleep(0.05)
            yield
        finally:
            uvicorn_server.should_exit = True  # graceful stop, so lifespan shutdown completes
            await asyncio.gather(server_task, return_exceptions=True)
