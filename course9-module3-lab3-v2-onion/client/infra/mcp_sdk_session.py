import asyncio
from collections.abc import Sequence
from contextlib import suppress
from contextvars import ContextVar

from mcp import StdioServerParameters
from mcp.client.client import Client
from mcp.client.context import ClientRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.types import (
    ElicitRequestFormParams,
    ElicitRequestParams,
    ElicitResult,
    ErrorData,
    TextContent,
    TextResourceContents,
)

from interfaces.approval_source_interface import ApprovalSourceInterface
from interfaces.mcp_session_interface import (
    McpSessionInterface,
    RawPrompt,
    RawPromptMessage,
    RawPromptResult,
    RawResource,
    RawTool,
    RawToolResult,
)

ClientTarget = StdioServerParameters | MCPServer

# The source that answers the server's questions for the call now in progress. The SDK runs its
# callback in a child task of the call that triggered it, and a child task inherits this value,
# so one shared connection can serve several users who are each asked only about their own calls.
_approvals: ContextVar[ApprovalSourceInterface | None] = ContextVar("approvals", default=None)


def _boolean_field(schema: dict[str, object]) -> str | None:
    """The name of the question's yes-or-no field, read from the server's own schema."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    for name, spec in properties.items():
        if isinstance(spec, dict) and spec.get("type") == "boolean":
            return str(name)
    return None


async def _answer_elicitation(
    context: ClientRequestContext, params: ElicitRequestParams
) -> ElicitResult | ErrorData:
    """Pass the server's question to the current call's approval source.

    Anything unexpected is a cancel, never an accept: no source for this call, a question this
    client cannot render, a form with no yes-or-no field, or a source that raises.
    """
    source = _approvals.get()
    if source is None or not isinstance(params, ElicitRequestFormParams):
        return ElicitResult(action="cancel")
    flag = _boolean_field(params.requested_schema)
    if flag is None:
        return ElicitResult(action="cancel")
    try:
        answer = await source.decide(params.message)
    except Exception:  # a broken source must never turn into an accept
        return ElicitResult(action="cancel")
    if answer == "accept":
        return ElicitResult(action="accept", content={flag: True})
    return ElicitResult(action=answer)


def _texts(blocks: list[object]) -> list[str]:
    return [block.text for block in blocks if isinstance(block, TextContent)]


class McpSdkSession(McpSessionInterface):
    """One connection to one MCP server, on the official `mcp` client (protocol auto-negotiated).

    The SDK requires the task that opens a connection to be the one that closes it. Here the
    client lives inside one dedicated background task, so `connect()` and `close()` can be
    called from any task, for example from different Gradio events. Overlapping `connect()`
    calls open one connection, and a failed connect leaves the session closed and retryable.
    """

    def __init__(self, target: ClientTarget) -> None:
        self._target = target
        self._client: Client | None = None
        self._runner: asyncio.Task[None] | None = None
        self._stop: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    @classmethod
    def for_stdio(cls, command: str, args: Sequence[str], cwd: str | None = None) -> McpSdkSession:
        """A session that launches `command args` as a subprocess and talks to it over stdio."""
        return cls(StdioServerParameters(command=command, args=list(args), cwd=cwd))

    async def connect(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            ready: asyncio.Future[Client] = asyncio.get_running_loop().create_future()
            stop = asyncio.Event()
            runner = asyncio.create_task(self._run(ready, stop))
            try:
                client = await ready
            except BaseException:
                runner.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await runner
                raise
            self._client, self._runner, self._stop = client, runner, stop

    async def _run(self, ready: asyncio.Future[Client], stop: asyncio.Event) -> None:
        """Hold the client open from inside one task until told to stop."""
        try:
            async with Client(self._target, elicitation_callback=_answer_elicitation) as client:
                ready.set_result(client)
                await stop.wait()
        except BaseException as error:
            if ready.done():
                raise
            ready.set_exception(error)

    async def close(self) -> None:
        async with self._lock:
            runner, stop = self._runner, self._stop
            self._client = self._runner = self._stop = None
        if runner is not None and stop is not None:
            stop.set()
            await runner

    def _require_client(self) -> Client:
        if self._client is None:
            raise RuntimeError("Not connected to a server; call connect() first")
        return self._client

    async def list_tools(self) -> list[RawTool]:
        result = await self._require_client().list_tools()
        return [
            RawTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema),
            )
            for tool in result.tools
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, object], approvals: ApprovalSourceInterface
    ) -> RawToolResult:
        client = self._require_client()
        token = _approvals.set(approvals)
        try:
            result = await client.call_tool(name, arguments)
        finally:
            _approvals.reset(token)
        return RawToolResult(texts=_texts(list(result.content)), is_error=bool(result.is_error))

    async def list_resources(self) -> list[RawResource]:
        result = await self._require_client().list_resources()
        return [
            RawResource(
                uri=str(resource.uri), name=resource.name, description=resource.description or ""
            )
            for resource in result.resources
        ]

    async def read_resource(self, uri: str, approvals: ApprovalSourceInterface) -> list[str]:
        client = self._require_client()
        token = _approvals.set(approvals)
        try:
            result = await client.read_resource(uri)
        finally:
            _approvals.reset(token)
        return [part.text for part in result.contents if isinstance(part, TextResourceContents)]

    async def list_prompts(self) -> list[RawPrompt]:
        result = await self._require_client().list_prompts()
        return [
            RawPrompt(
                name=prompt.name,
                description=prompt.description or "",
                argument_names=[argument.name for argument in prompt.arguments or []],
            )
            for prompt in result.prompts
        ]

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> RawPromptResult:
        result = await self._require_client().get_prompt(name, arguments)
        return RawPromptResult(
            description=result.description or "",
            messages=[
                RawPromptMessage(
                    role=message.role,
                    text=message.content.text if isinstance(message.content, TextContent) else "",
                )
                for message in result.messages
            ],
        )
