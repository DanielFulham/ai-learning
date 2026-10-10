from typing import Protocol

from domain.conversation import AgentReply


class AgentServiceInterface(Protocol):
    """Service contract for running an agent over MCP tools."""

    async def ask(self, prompt: str) -> AgentReply: ...
