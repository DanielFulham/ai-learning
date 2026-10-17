from contextlib import AbstractAsyncContextManager
from typing import Protocol

from mcp.shared.memory import MessageStream


class McpTransportInterface(Protocol):
    """How the client reaches one MCP server: stdio, HTTP, or anything that gives the
    session a read stream and a write stream."""

    def open(self) -> AbstractAsyncContextManager[MessageStream]:
        """Connect, yield the (read, write) streams, and disconnect on exit."""
        ...
