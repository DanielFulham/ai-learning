from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypedDict


class RawTool(TypedDict):
    name: str
    description: str
    input_schema: dict[str, Any]


class RawPromptArgument(TypedDict):
    name: str
    required: bool


class RawPrompt(TypedDict):
    name: str
    arguments: list[RawPromptArgument]


class RawToolResult(TypedDict):
    text: str
    is_error: bool


# Asked with the server's message and the names of the fields it wants filled in.
# Returns the values by field name, or None when the person declines.
ElicitationCallback = Callable[[str, list[str]], Awaitable[dict[str, str] | None]]

# Called with progress, total (None if unknown) and an optional message.
ProgressCallback = Callable[[float, float | None, str | None], Awaitable[None]]

# Called with the method name of each notification the server sends.
NotificationCallback = Callable[[str], Awaitable[None]]


class McpSessionInterface(Protocol):
    """A live connection to one MCP server."""

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def list_tools(self) -> list[RawTool]: ...

    async def list_prompts(self) -> list[RawPrompt]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> RawToolResult: ...

    async def get_prompt_text(self, name: str, arguments: dict[str, str]) -> str: ...

    async def read_resource_text(self, uri: str) -> str: ...
