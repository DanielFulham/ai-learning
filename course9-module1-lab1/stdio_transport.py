import asyncio

from fastmcp import Client

# StdioTransport is one of the transport options for the client.
from fastmcp.client.transports import StdioTransport

from helpers.common import run_lab_steps, show_docs, show_library_id

CONTEXT7_VERSION = "4.1.1"

STDIO_COMMAND = "npx"
STDIO_ARGS = ["-y", f"@upstash/context7-mcp@{CONTEXT7_VERSION}"]


def build_stdio_client() -> Client:
    stdio_transport = StdioTransport(command=STDIO_COMMAND, args=STDIO_ARGS)
    print(stdio_transport)
    return Client(stdio_transport)


async def main() -> None:
    stdio_client = build_stdio_client()
    await run_lab_steps(stdio_client)

    print("\n--- Question 1: resolve-library-id for scikit-learn ---")
    await show_library_id(
        stdio_client,
        library_name="scikit-learn",
        query="I want to use scikit-learn package",
        max_chars=1500,
    )

    print("\n--- Question 2: query-docs for scikit-learn ---")
    await show_docs(
        stdio_client,
        library_id="/scikit-learn/scikit-learn",
        query="I want to fetch the documentation of the package.",
    )


if __name__ == "__main__":
    asyncio.run(main())