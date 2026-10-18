import logging
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    Prompt,
    ReadResourceResult,
    ResourceTemplate,
    Tool,
)
from pydantic import AnyUrl

logging.getLogger("mcp").setLevel(logging.WARNING)


class MCPHTTPClient:
    """Base MCP HTTP client with pure protocol logic - no GUI dependencies."""

    def __init__(self, server_url: str, roots_dir: str) -> None:
        self.server_url = server_url
        self.roots_dir = roots_dir
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self._connected = False

    def _require_session(self) -> ClientSession:
        if self.session is None:
            raise RuntimeError("Not connected to a server; call connect() first")
        return self.session

    async def connect(self) -> None:
        """Connect to HTTP MCP server via Streamable HTTP. Safe to call multiple times."""
        if self._connected:
            return

        # FastMCP uses /mcp endpoint for streamable HTTP
        mcp_url = f"{self.server_url}/mcp"
        read, write, _ = await self.exit_stack.enter_async_context(
            streamablehttp_client(mcp_url)
        )

        session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await session.initialize()
        self.session = session
        self._connected = True

    async def list_tools(self) -> list[Tool]:
        """List all available tools from the HTTP server."""
        result = await self._require_session().list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> CallToolResult:
        """Execute a tool on the HTTP server."""
        result = await self._require_session().call_tool(tool_name, arguments)
        return result

    async def list_resources(self) -> list[ResourceTemplate]:
        """List all available resource templates from the HTTP server."""
        result = await self._require_session().list_resource_templates()
        return result.resourceTemplates

    async def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a resource by URI from the HTTP server."""
        result = await self._require_session().read_resource(AnyUrl(uri))
        return result

    async def list_prompts(self) -> list[Prompt]:
        """List all available prompts from the HTTP server."""
        result = await self._require_session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name: str, arguments: dict[str, str]) -> GetPromptResult:
        """Get a rendered prompt template from the HTTP server."""
        result = await self._require_session().get_prompt(prompt_name, arguments)
        return result

    async def cleanup(self) -> None:
        """Clean up resources and close HTTP connection."""
        await self.exit_stack.aclose()
