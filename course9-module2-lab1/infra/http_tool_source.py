from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class HttpToolSource:
    """Tools from an MCP server reached over Streamable HTTP."""

    def __init__(self, url: str) -> None:
        self._url = url

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Sequence[BaseTool]]:
        async with streamable_http_client(self._url) as (read, write, _sid):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield await load_mcp_tools(session)
