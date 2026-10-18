import asyncio
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.types import TextContent, TextResourceContents
from pydantic import AnyUrl

from interfaces.mcp_session_interface import (
    McpSessionInterface,
    RawPrompt,
    RawPromptResult,
    RawResourceTemplate,
    RawTool,
    RawToolResult,
)
from interfaces.mcp_transport_interface import McpTransportInterface


class McpSdkSession(McpSessionInterface):
    """A connection to one MCP server through the official SDK's `ClientSession`.

    How the server is reached (stdio, HTTP) is the transport's job; this class only speaks
    the protocol and returns primitives.
    """

    def __init__(self, transport: McpTransportInterface) -> None:
        self._transport = transport
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._connect_lock = asyncio.Lock()

    async def connect(self) -> None:
        # The lock makes calls that overlap wait for the first one to finish the handshake.
        async with self._connect_lock:
            if self._session is not None:
                return

            read, write = await self._stack.enter_async_context(self._transport.open())
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session

    async def close(self) -> None:
        await self._stack.aclose()
        self._session = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Not connected to a server; call connect() first")
        return self._session

    async def list_tools(self) -> list[RawTool]:
        result = await self._require_session().list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": dict(t.inputSchema),
            }
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, object]) -> RawToolResult:
        result = await self._require_session().call_tool(name, arguments)
        texts = [c.text for c in result.content if isinstance(c, TextContent)]
        return {"texts": texts, "is_error": result.isError}

    async def list_resource_templates(self) -> list[RawResourceTemplate]:
        result = await self._require_session().list_resource_templates()
        return [
            {
                "name": t.name,
                "uri_template": t.uriTemplate,
                "description": t.description or "",
            }
            for t in result.resourceTemplates
        ]

    async def read_resource(self, uri: str) -> list[str]:
        result = await self._require_session().read_resource(AnyUrl(uri))
        return [c.text for c in result.contents if isinstance(c, TextResourceContents)]

    async def list_prompts(self) -> list[RawPrompt]:
        result = await self._require_session().list_prompts()
        return [
            {
                "name": p.name,
                "description": p.description or "",
                "argument_names": [a.name for a in p.arguments or []],
            }
            for p in result.prompts
        ]

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> RawPromptResult:
        result = await self._require_session().get_prompt(name, arguments)
        return {
            "description": result.description or "",
            "messages": [
                {
                    "role": m.role,
                    "text": m.content.text if isinstance(m.content, TextContent) else "",
                }
                for m in result.messages
            ],
        }
