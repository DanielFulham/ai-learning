from contextlib import AsyncExitStack
from typing import Any, Self

from fastmcp import Client, FastMCP
from fastmcp.client.elicitation import ElicitResult
from fastmcp.client.transports import ClientTransport, StdioTransport, StreamableHttpTransport
from mcp.shared.context import RequestContext
from mcp.shared.session import RequestResponder
from mcp.types import (
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    ServerNotification,
    TextContent,
    TextResourceContents,
)

from interfaces.mcp_session_interface import (
    ElicitationCallback,
    McpSessionInterface,
    NotificationCallback,
    ProgressCallback,
    RawPrompt,
    RawTool,
    RawToolResult,
)


class FastmcpSession(McpSessionInterface):
    """A FastMCP client connection to one server.

    `target` is a transport (for a real server) or a FastMCP instance (in-memory,
    for tests). The callbacks are plain callables so this class knows nothing about
    how the application answers the server.
    """

    def __init__(
        self,
        target: ClientTransport | FastMCP[Any],
        on_elicitation: ElicitationCallback,
        on_progress: ProgressCallback,
        on_notification: NotificationCallback,
    ) -> None:
        self._target = target
        self._on_elicitation = on_elicitation
        self._on_progress = on_progress
        self._on_notification = on_notification
        self._stack = AsyncExitStack()
        self._client: Client[Any] | None = None

    @classmethod
    def for_stdio(
        cls,
        command: str,
        args: list[str],
        on_elicitation: ElicitationCallback,
        on_progress: ProgressCallback,
        on_notification: NotificationCallback,
    ) -> Self:
        """A session that launches the server as a subprocess and talks to it over stdio."""
        transport = StdioTransport(command=command, args=args)
        return cls(transport, on_elicitation, on_progress, on_notification)

    @classmethod
    def for_http(
        cls,
        url: str,
        on_elicitation: ElicitationCallback,
        on_progress: ProgressCallback,
        on_notification: NotificationCallback,
    ) -> Self:
        """A session that connects to a server already running at an HTTP address."""
        transport = StreamableHttpTransport(url=url)
        return cls(transport, on_elicitation, on_progress, on_notification)

    async def connect(self) -> None:
        self._client = Client(
            self._target,
            elicitation_handler=self._handle_elicitation,
            progress_handler=self._on_progress,
            message_handler=self._handle_message,
        )
        await self._stack.enter_async_context(self._client)

    async def close(self) -> None:
        await self._stack.aclose()
        self._client = None

    def _require_client(self) -> Client[Any]:
        if self._client is None:
            raise RuntimeError("Not connected to a server; call connect() first")
        return self._client

    async def list_tools(self) -> list[RawTool]:
        tools = await self._require_client().list_tools()
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
            for t in tools
        ]

    async def list_prompts(self) -> list[RawPrompt]:
        prompts = await self._require_client().list_prompts()
        return [
            {
                "name": p.name,
                "arguments": [
                    {"name": a.name, "required": bool(a.required)} for a in (p.arguments or [])
                ],
            }
            for p in prompts
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> RawToolResult:
        result = await self._require_client().call_tool(name, arguments, raise_on_error=False)
        text = "\n".join(c.text if isinstance(c, TextContent) else str(c) for c in result.content)
        return {"text": text, "is_error": result.is_error}

    async def get_prompt_text(self, name: str, arguments: dict[str, str]) -> str:
        result = await self._require_client().get_prompt_mcp(name, arguments)
        content = result.messages[0].content
        if not isinstance(content, TextContent):
            raise ValueError("Prompt did not return text content")
        return content.text

    async def read_resource_text(self, uri: str) -> str:
        result = await self._require_client().read_resource_mcp(uri)
        first = result.contents[0]
        if not isinstance(first, TextResourceContents):
            raise ValueError("Resource did not return text content")
        return first.text

    async def _handle_elicitation(
        self,
        message: str,
        response_type: type[Any] | None,
        params: ElicitRequestFormParams | ElicitRequestURLParams,
        context: RequestContext[Any, Any, Any],
    ) -> dict[str, str] | ElicitResult[dict[str, str]]:
        fields = (
            list(params.requestedSchema.get("properties", {}))
            if isinstance(params, ElicitRequestFormParams)
            else []
        )
        values = await self._on_elicitation(message, fields)
        if values is None:
            return ElicitResult(action="decline")
        return values

    async def _handle_message(
        self,
        message: RequestResponder[Any, Any] | ServerNotification | Exception,
    ) -> None:
        if isinstance(message, ServerNotification):
            await self._on_notification(message.root.method)
