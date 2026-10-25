from typing import Literal, Protocol

ApprovalAnswer = Literal["accept", "decline", "cancel"]


class ApprovalSourceInterface(Protocol):
    """Whatever answers the server's question before it runs a tool: a person at a page."""

    async def decide(self, message: str) -> ApprovalAnswer:
        """Show the question, wait for the answer, and return it."""
        ...
