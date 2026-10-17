from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import MessageStream

from interfaces.mcp_transport_interface import McpTransportInterface


class HttpTransport(McpTransportInterface):
    """Connects to a server that is already running at a Streamable HTTP address."""

    def __init__(self, url: str) -> None:
        self._url = url

    @asynccontextmanager
    async def open(self) -> AsyncIterator[MessageStream]:
        # The SDK also yields a callable that returns the session id; the session does not use it.
        async with streamable_http_client(self._url) as (read, write, _):
            yield read, write
