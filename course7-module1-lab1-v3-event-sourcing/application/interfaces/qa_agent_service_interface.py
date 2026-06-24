from typing import Protocol

from domain.run_result import RunResult


class QAAgentServiceInterface(Protocol):
    """Contract for the QA workflow's application service.

    The entry point depends on this Protocol, not on the concrete
    `QAAgentService`. Lets the entry point be unaware of the V3a
    constructor surface (event store, inner consumer, clock).

    `run()` returns a `RunResult` carrying both the workflow's
    QAExchange and the run_id the service generated. Callers that
    need the run_id (the demo, projection consumers) access
    `result.run_id`; callers that only care about the answer access
    `result.exchange.answer`.
    """

    def run(self, question: str) -> RunResult: ...