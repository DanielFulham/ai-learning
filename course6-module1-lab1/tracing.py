"""Shared message-list inspection helper.

`create_agent` returns a state dict containing a `messages` list of
HumanMessage / AIMessage / ToolMessage objects. `print_trace` walks the list
and surfaces the fields most useful for understanding what the agent did:
content, tool name (for ToolMessage), and tool_calls (for AIMessage).
"""


def print_trace(response: dict) -> None:
    """Walk the message list and print what the agent did."""
    for i, msg in enumerate(response["messages"]):
        msg_type = type(msg).__name__
        print(f"\n--- Message {i+1} [{msg_type}] ---")
        if hasattr(msg, "content") and msg.content:
            print(f"Content: {msg.content}")
        if hasattr(msg, "name") and msg.name:
            print(f"Tool name: {msg.name}")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"Tool calls: {msg.tool_calls}")