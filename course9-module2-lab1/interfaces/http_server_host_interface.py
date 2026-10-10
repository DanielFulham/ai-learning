from contextlib import AbstractAsyncContextManager
from typing import Protocol

from fastmcp import FastMCP


class HttpServerHostInterface(Protocol):
    """Runs an MCP server over HTTP for the length of a block, then stops it."""

    def is_port_in_use(self) -> bool: ...

    def serve(self, server: FastMCP) -> AbstractAsyncContextManager[None]: ...
