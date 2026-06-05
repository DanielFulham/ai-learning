from typing import Protocol


class ToolCallingAgentInterface(Protocol):
    """An agent that takes a natural-language query and answers it using tools."""

    def ask(self, query: str) -> str:
        """Run the agent loop until a final answer is produced. Return the answer."""
        ...