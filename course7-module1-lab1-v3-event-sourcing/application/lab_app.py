from dataclasses import dataclass

from application.interfaces.auth_agent_service_interface import (
    AuthAgentServiceInterface,
)
from application.interfaces.qa_agent_service_interface import (
    QAAgentServiceInterface,
)
from interfaces.agent_checkpointer_interface import AgentCheckpointerInterface
from interfaces.agent_event_store_interface import AgentEventStoreInterface


@dataclass(frozen=True)
class LabApp:
    """V3a-shrunk lab application bundle.

    Carries the QA service, the shared event store, and the checkpointer.
    Exposing the event store on the bundle makes the singleton contract
    visible — when V3b and V3c add Auth and Counter services, all three
    services will hold this same reference, and the demo's read-side
    (RunSummaryProjection, ThreadHistoryProjection in V3c) reads from
    the same store the services wrote to. The checkpointer is exposed on
    the bundle so the resume path is reachable from `initialise()`'s
    output — it shares the SQLite database file with the event store when
    persistence is enabled.

    Frozen — the bundle is immutable after construction. To rewire,
    call `initialise()` again.
    """

    qa: QAAgentServiceInterface
    event_store: AgentEventStoreInterface
    checkpointer: AgentCheckpointerInterface
    auth: AuthAgentServiceInterface