from typing import Protocol

from domain.qa_exchange import QAExchange


class QAAgentServiceInterface(Protocol):
    """Service contract for the QA workflow."""

    def run(self, question: str) -> QAExchange: ...