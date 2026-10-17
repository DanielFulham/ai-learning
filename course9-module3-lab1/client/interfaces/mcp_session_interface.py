from typing import Protocol, TypedDict


class RawTool(TypedDict):
    name: str
    description: str


class RawToolResult(TypedDict):
    texts: list[str]
    is_error: bool


class RawResourceTemplate(TypedDict):
    name: str
    uri_template: str


class RawPrompt(TypedDict):
    name: str
    description: str
    argument_names: list[str]


class RawPromptMessage(TypedDict):
    role: str
    text: str


class RawPromptResult(TypedDict):
    description: str
    messages: list[RawPromptMessage]


class McpSessionInterface(Protocol):
    """A live connection to one MCP server."""

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def list_tools(self) -> list[RawTool]: ...

    async def call_tool(self, name: str, arguments: dict[str, object]) -> RawToolResult: ...

    async def list_resource_templates(self) -> list[RawResourceTemplate]: ...

    async def read_resource(self, uri: str) -> list[str]:
        """The text of each text part of the resource."""
        ...

    async def list_prompts(self) -> list[RawPrompt]: ...

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> RawPromptResult: ...
