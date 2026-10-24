from typing import Protocol, TypedDict

from interfaces.approval_source_interface import ApprovalSourceInterface


class RawTool(TypedDict):
    name: str
    description: str
    input_schema: dict[str, object]


class RawToolResult(TypedDict):
    texts: list[str]
    is_error: bool


class RawResource(TypedDict):
    uri: str
    name: str
    description: str


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
    """A live connection to one MCP server.

    The server may ask a question before it runs a tool or serves a resource. Each call that can
    trigger one takes the `ApprovalSourceInterface` that answers it, so several users can share
    one connection and each is only ever asked about their own calls.
    """

    async def connect(self) -> None:
        """Connect and initialise. Safe to call again: a connected session does nothing."""
        ...

    async def close(self) -> None: ...

    def shutdown(self) -> None:
        """Close the connection from outside the session's own event loop, for example from the
        main thread when the app exits. Does nothing if it was never connected."""
        ...

    async def list_tools(self) -> list[RawTool]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, object], approvals: ApprovalSourceInterface
    ) -> RawToolResult: ...

    async def list_resources(self) -> list[RawResource]: ...

    async def read_resource(self, uri: str, approvals: ApprovalSourceInterface) -> list[str]:
        """The text of each text part of the resource."""
        ...

    async def list_prompts(self) -> list[RawPrompt]: ...

    async def get_prompt(self, name: str, arguments: dict[str, str]) -> RawPromptResult: ...
