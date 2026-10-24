import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

from infra.mcp_sdk_session import McpSdkSession
from interfaces.approval_source_interface import ApprovalSourceInterface

_SERVER_SCRIPT = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("TEST_STDIO_SERVER")


@server.tool()
def echo(text: str) -> str:
    return text


if __name__ == "__main__":
    server.run("stdio")
"""


def test_given_a_script_when_a_stdio_session_is_made_then_it_launches_and_calls_the_server(
    tmp_path: Path,
) -> None:
    """A real subprocess over stdio: pins that the factory builds a working launch."""
    script = tmp_path / "TEST_SERVER.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")

    async def run() -> tuple[list[str], list[str]]:
        session = McpSdkSession.for_stdio(sys.executable, [str(script)], cwd=str(tmp_path))
        await session.connect()
        try:
            tools = [tool["name"] for tool in await session.list_tools()]
            approvals = MagicMock(spec=ApprovalSourceInterface)
            result = await session.call_tool("echo", {"text": "TEST_TEXT"}, approvals)
            return tools, result["texts"]
        finally:
            await session.close()

    assert asyncio.run(run()) == (["echo"], ["TEST_TEXT"])
