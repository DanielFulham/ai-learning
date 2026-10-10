from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MultiServerToolSource:
    """Tools from one stdio server and one HTTP server, in a single client.

    Tool names are prefixed with the server name (`stdio-client_add`,
    `http-client_add`) so two servers offering the same tool don't collide.
    """

    def __init__(self, stdio_command: str, stdio_args: list[str], http_url: str) -> None:
        self._client = MultiServerMCPClient(
            {
                "stdio-client": {
                    "command": stdio_command,
                    "args": stdio_args,
                    "transport": "stdio",
                },
                "http-client": {
                    "url": http_url,
                    "transport": "streamable_http",
                },
            },
            tool_name_prefix=True,
        )

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Sequence[BaseTool]]:
        # Each tool call opens its own session, so there is no connection to hold open.
        yield await self._client.get_tools()
