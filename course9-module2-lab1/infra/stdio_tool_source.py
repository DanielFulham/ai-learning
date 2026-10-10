from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class StdioToolSource:
    """Tools from an MCP server launched as a subprocess over stdio."""

    def __init__(self, command: str, args: list[str]) -> None:
        self._params = StdioServerParameters(command=command, args=args)

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Sequence[BaseTool]]:
        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield await load_mcp_tools(session)
