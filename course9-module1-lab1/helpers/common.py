"""Code shared by stdio_transport.py and http_transport.py.

The two scripts differ only in the transport that builds their client. Everything
that happens once a client exists lives here.
"""

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent, Tool

LIBRARY_NAME = "fastmcp"
RESOLVE_QUERY = "I want to create a new MCP server using the fastmcp Python framework"
# Fixed by the notebook; it is not parsed from the resolve-library-id output.
LIBRARY_ID = "/llmstxt/gofastmcp_llms-full_txt"
DOCS_QUERY = "I want to fetch the code snippets and the documentation"


def first_text(result: CallToolResult) -> str | None:
    # The notebook reads content[0].text unnarrowed; only TextContent has .text.
    # call_tool() raises ToolError on is_error results by default, so is_error is never
    # True here. A backend failure returned as plain text (F-L41-7) arrives as content.
    block = result.content[0]
    return block.text if isinstance(block, TextContent) else None


def show_tool(tool: Tool) -> None:
    print(tool.name)
    print(tool.description)
    print(tool.input_schema)
    print()


async def show_tools(mcp_client: Client) -> None:
    async with mcp_client as client:
        # List of tools the server provides
        tools = await client.list_tools()
        print(f"mode={client.mode} protocol_version={client.protocol_version}")
        print(f"server_info={client.server_info}")

    print("Done")
    print(len(tools))
    for tool in tools:
        show_tool(tool)


async def show_library_id(
    mcp_client: Client,
    library_name: str = LIBRARY_NAME,
    query: str = RESOLVE_QUERY,
    max_chars: int | None = None,
) -> None:
    async with mcp_client as client:
        # Find a library ID via a search query
        response = await client.call_tool(
            "resolve-library-id",
            {"libraryName": library_name, "query": query},
        )

    print("Response block:")
    text = first_text(response)
    if text is not None:
        print(text[:max_chars])


async def show_docs(
    mcp_client: Client,
    library_id: str = LIBRARY_ID,
    query: str = DOCS_QUERY,
    max_chars: int = 1000,
) -> None:
    async with mcp_client as client:
        # Fetch documentation for the library ID
        docs = await client.call_tool(
            "query-docs",
            {"libraryId": library_id, "query": query},
        )

    text = first_text(docs)
    if text is not None:
        print(text[:max_chars])


async def run_lab_steps(mcp_client: Client) -> None:
    print("\n--- Tools ---")
    await show_tools(mcp_client)

    print("\n--- resolve-library-id ---")
    await show_library_id(mcp_client)

    print("\n--- query-docs ---")
    await show_docs(mcp_client)