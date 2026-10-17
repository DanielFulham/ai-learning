from contextlib import AbstractAsyncContextManager

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import MessageStream

from interfaces.mcp_transport_interface import McpTransportInterface


class StdioTransport(McpTransportInterface):
    """Launches the server as a subprocess and talks to it over its stdin and stdout."""

    def __init__(self, command: str, args: list[str]) -> None:
        self._command = command
        self._args = args

    def open(self) -> AbstractAsyncContextManager[MessageStream]:
        params = StdioServerParameters(command=self._command, args=self._args, env=None)
        return stdio_client(params)
