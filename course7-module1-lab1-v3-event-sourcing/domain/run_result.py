from dataclasses import dataclass
from uuid import UUID

from domain.qa_exchange import QAExchange


@dataclass(frozen=True)
class RunResult:
    """One QA run's outcome plus its run_id.

    Returned by `QAAgentService.run()`. The exchange carries the
    workflow's answer (or the user-safe failure message); the run_id
    correlates this run with its events in the event store, so callers
    can replay-by-run for projections or composition assertions.

    Without the run_id, the demo and any projection caller can't ask
    "what happened in this run?" against the event log — the service
    generates the UUID internally and would otherwise discard it.
    """

    exchange: QAExchange
    run_id: UUID