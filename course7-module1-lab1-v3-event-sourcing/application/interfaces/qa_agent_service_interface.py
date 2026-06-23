from typing import Protocol

from domain.qa_exchange import QAExchange


class QAAgentServiceInterface(Protocol):
    """Contract for the QA workflow's application service.

    The entry point depends on this Protocol, not on the concrete
    `QAAgentService`. Lets the entry point be unaware of the V3a
    constructor surface (event store, inner consumer, clock).
    """

    def run(self, question: str) -> QAExchange: ...