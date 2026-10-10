"""Call the calculator MCP server directly with the raw client, one transport at a time.

No model is involved. The in-memory demo covers tools, both resources and the prompt; the
http and stdio demos repeat the tool call over a real transport; local-tool calls a plain
LangChain tool for contrast, with no MCP server. See agent_demo.py for the same server used
by an agent.

    python raw_client_demo.py in-memory|http|stdio|local-tool
"""

import argparse
import asyncio
import sys
from collections.abc import Callable

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport
from langchain.tools import tool
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    BlobResourceContents,
    GetPromptResult,
    TextContent,
    TextResourceContents,
    Tool,
)

from application.container import initialise_http_host, initialise_server
from config import HTTP_URL, PORT, SERVER_ARGS, SERVER_COMMAND
from scripts.download_samples import prepare_sample_files

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def call_add(client: Client, a: int, b: int) -> CallToolResult:
    async with client:
        return await client.call_tool("add", {"a": a, "b": b})


async def call_subtract(client: Client, a: int, b: int) -> CallToolResult:
    async with client:
        return await client.call_tool("subtract", {"a": a, "b": b})


async def read_resource(
    client: Client, uri: str
) -> list[TextResourceContents | BlobResourceContents]:
    async with client:
        return await client.read_resource(uri)


async def get_prompt(client: Client, code: str) -> GetPromptResult:
    async with client:
        return await client.get_prompt("review_code", {"code": code})


async def list_tools(client: Client) -> list[Tool]:
    async with client:
        return await client.list_tools()


def result_text(response: CallToolResult) -> str | None:
    """The text of a tool result's first content block, or None if it is not text."""
    first = response.content[0]
    return first.text if isinstance(first, TextContent) else None


def resource_text(contents: list[TextResourceContents | BlobResourceContents]) -> str | None:
    """The text of a resource's first contents, or None if it is binary."""
    first = contents[0]
    return first.text if isinstance(first, TextResourceContents) else None


def print_stream_info(read: object, write: object, session_id: Callable[[], str | None]) -> None:
    """Print the read/write streams and session ID behind a connection."""
    print("READ (receives FROM server):")
    print(read)
    print()

    print("WRITE (sends TO server):")
    print(write)
    print()

    print("SESSION ID:")
    print(session_id())


def print_add_result(response: CallToolResult) -> None:
    """Print the three ways of reading a tool result."""
    print(response)

    # The actual answer/data
    print("\nResult Data .data :")
    print(response.data)  # 9

    # Content (text format)
    print("\nContent (as text):")
    print(result_text(response))  # "9"

    # Structured content (as dictionary)
    print("\nStructured Content:")
    print(response.structured_content)


async def demo_local_tool() -> None:
    """Call a LangChain tool defined in this file; no MCP server is involved."""
    print(multiply.name)
    print(multiply.description)
    print(multiply.args)

    print("What is 2 x 3?")
    print("Answer: " + str(multiply.invoke({"a": 2, "b": 3})))


async def show_tool_calls(client: Client) -> None:
    print_add_result(await call_add(client, 4, 5))
    print(result_text(await call_subtract(client, 4, 5)))


async def show_tool_listing(client: Client) -> None:
    tools = await list_tools(client)
    print("Available tools:")
    for listed_tool in tools:
        print(f"- {listed_tool.name}: {listed_tool.description}")

    first_tool = tools[0]
    print(first_tool)
    print(first_tool.inputSchema)
    print(first_tool.outputSchema)


async def show_resources(client: Client) -> None:
    """Two resources: the first returns a template string, the second reads a real file."""
    print(resource_text(await read_resource(client, "file:///endpoint/README.txt")))

    text = resource_text(await read_resource(client, "file://endpoint2/README.txt"))
    print(text[:300] if text is not None else None)

    # A file that doesn't exist: the store returns None, so the resource returns a
    # normal text result, not an error.
    missing = (await read_resource(client, "file://endpoint2/random.txt"))[0]
    print(f"uri:      {missing.uri}")
    print(f"mimeType: {missing.mimeType}")
    print(f"meta:     {missing.meta}")
    if isinstance(missing, TextResourceContents):
        print(f"text:     {missing.text}")


async def show_prompt(client: Client) -> None:
    prompt = await get_prompt(client, "CODE TO BE REVIEWED")
    message = prompt.messages[0]
    print(f"Prompt Role: {message.role}")
    if isinstance(message.content, TextContent):
        print(f"Prompt Content: {message.content.text}")


async def demo_in_memory() -> None:
    """Call the tools, read both resources and fetch the prompt, with no transport."""
    mcp = initialise_server()
    print("mcp object", mcp)

    prepare_sample_files()

    client = Client(mcp)
    print(f"client: {client}")

    await show_tool_calls(client)
    await show_tool_listing(client)
    await show_resources(client)
    await show_prompt(client)


async def demo_http() -> None:
    """Serve the same server over Streamable HTTP from this process and call it."""
    host = initialise_http_host()
    print(f"Port {PORT} is available: {not host.is_port_in_use()}")
    http_client = Client(StreamableHttpTransport(url=HTTP_URL))
    print(f"http_client: {http_client}")

    async with host.serve(initialise_server()):
        print(f"HTTP MCP Server started in background on port {PORT}")
        print(result_text(await call_add(http_client, 4, 5)))

        print(await list_tools(http_client))

        # The raw read/write streams behind an HTTP connection.
        async with streamable_http_client(HTTP_URL) as (read, write, session_id):
            print_stream_info(read, write, session_id)


async def demo_stdio() -> None:
    """Launch the same server as a subprocess and call it over stdio."""
    stdio_client = Client(StdioTransport(command=SERVER_COMMAND, args=SERVER_ARGS))
    print(f"stdio_transport_client: {stdio_client}")

    print(result_text(await call_add(stdio_client, 4, 5)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raw_client_demo",
        description="Call the calculator MCP server with the raw client, one transport at a time",
    )
    parser.add_argument(
        "transport",
        choices=["in-memory", "http", "stdio", "local-tool"],
        help="Which demo to run (local-tool uses no MCP server)",
    )
    args = parser.parse_args(argv)
    demos = {
        "in-memory": demo_in_memory,
        "http": demo_http,
        "stdio": demo_stdio,
        "local-tool": demo_local_tool,
    }
    asyncio.run(demos[args.transport]())
    return 0


if __name__ == "__main__":
    sys.exit(main())
