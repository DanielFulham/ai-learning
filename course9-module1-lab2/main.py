import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

TRACE_RESULT_CHARS = 200


def trace_turn(messages: list[BaseMessage]) -> None:
    """Print the tools called and tokens used since the last user message."""
    last_user = max(i for i, message in enumerate(messages) if message.type == "human")
    input_tokens = 0
    output_tokens = 0
    for message in messages[last_user + 1 :]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"  tool call:   {call['name']} {str(call['args'])[:150]}")
            if message.usage_metadata:
                input_tokens += message.usage_metadata["input_tokens"]
                output_tokens += message.usage_metadata["output_tokens"]
        elif isinstance(message, ToolMessage):
            text = " ".join(message.text.split())
            shown = text[:TRACE_RESULT_CHARS] + ("..." if len(text) > TRACE_RESULT_CHARS else "")
            print(f"  tool result: {message.name} status={message.status} {shown}")
    print(f"  tokens this turn: {input_tokens} in, {output_tokens} out")


async def main() -> None:
    """Set up an agent with access to the Context7 and Met Museum MCP servers."""

    # Initialize the multi-server MCP client with the Context7 and Met Museum servers
    client = MultiServerMCPClient(
        {
            "context7": {
                "url": "https://mcp.context7.com/mcp",
                "transport": "streamable_http",
            },
            "met-museum": {
                "command": "npx",
                # The lab's metmuseum-mcp uses the retired /v1/search; this server uses /v1.1
                "args": ["-y", "@cyanheads/met-museum-mcp-server@0.7.0"],
                "transport": "stdio",
            },
        }
    )

    # Language model that powers the agent's reasoning and responses
    model = ChatAnthropic(
        model_name="claude-haiku-4-5",
        timeout=None,
        stop=None,
    )

    print("Retrieving tools from the MCP client...")
    tools = await client.get_tools()

    # Conversation memory: lets the agent remember earlier messages in the session
    checkpointer = InMemorySaver()

    # The thread_id groups every message in this session together
    config: RunnableConfig = {"configurable": {"thread_id": "conversation_id"}}

    # The agent reasons, calls MCP tools, observes the results, and repeats until it can answer
    agent = create_agent(model=model, tools=tools, checkpointer=checkpointer)

    # Introduce the agent and its capabilities; this also seeds the conversation thread
    print("Introducing the agent and its capabilities...")
    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a smart, useful agent with tools to access "
                        "code library documentation and the Met Museum collection."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Give a brief introduction of what you do and the tools you can access."
                    ),
                },
            ]
        },
        config=config,
    )

    # Print the agent's reply (the last message in the conversation)
    print("Agent's introduction response:")
    print(response["messages"][-1].text)
    trace_turn(response["messages"])

    # Main interaction loop: keep talking to the agent until you quit
    while True:
        choice = input(
            """
Menu:
1. Ask the agent a question
2. Quit
Enter your choice (1 or 2): """
        )

        if choice == "1":
            print("Your question")
            query = input("> ")

            # The agent sees the full conversation history through the checkpointer
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
            )
            print(response["messages"][-1].text)
            trace_turn(response["messages"])
        else:
            # Any choice other than "1" exits, including typos
            print("Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
