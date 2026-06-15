from typing import Protocol

from domain.models import AgentTrace


class SqlAgentInterface(Protocol):
    def ask(self, question: str) -> str: ...

    @property
    def trace(self) -> AgentTrace: ...