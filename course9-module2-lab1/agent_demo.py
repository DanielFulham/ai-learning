"""Ask the agent a question that needs the calculator server's tools.

Unlike raw_client_demo.py, which calls the MCP server directly with no model, this goes
through the agent service: a model decides when to call the tools, and the tools reach it
over the chosen transport.

    python agent_demo.py http|stdio|multi    # needs ANTHROPIC_API_KEY in .env
"""

import argparse
import asyncio
import sys

from dotenv import load_dotenv

from application.container import (
    Transport,
    initialise,
    initialise_http_host,
    initialise_server,
)
from application.interfaces.agent_service_interface import AgentServiceInterface
from config import PORT
from domain.conversation import AgentReply

load_dotenv()

PROMPTS: dict[Transport, str] = {
    "http": "Use the add tool to add 2 and 1 and let me know if you used a tool.",
    "stdio": "Use the add tool to add 2 and 1.",
    "multi": "whats 8 + 7? use tools",
}


def print_reply(reply: AgentReply) -> None:
    """Print each turn of the conversation, naming any tools the AI called."""
    for turn in reply.turns:
        label = turn.role.upper()
        if turn.text or not turn.tool_names:
            print(f"[{label}] {turn.text}")
        if turn.tool_names:
            print(f"[{label}] tool call: {', '.join(turn.tool_names)}")


async def run(transport: Transport) -> None:
    """Ask the agent one question over the chosen transport."""
    service: AgentServiceInterface = initialise(transport=transport)
    if transport == "stdio":
        reply = await service.ask(PROMPTS[transport])
    else:
        # The http and multi tool sources need the HTTP server running.
        host = initialise_http_host()
        async with host.serve(initialise_server()):
            print(f"HTTP MCP Server started in background on port {PORT}")
            reply = await service.ask(PROMPTS[transport])
    print_reply(reply)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_demo",
        description="Ask an agent a question that needs the calculator server's tools",
    )
    parser.add_argument(
        "transport",
        choices=["http", "stdio", "multi"],
        help="Which transport the agent's tools come over",
    )
    args = parser.parse_args(argv)
    asyncio.run(run(args.transport))
    return 0


if __name__ == "__main__":
    sys.exit(main())
