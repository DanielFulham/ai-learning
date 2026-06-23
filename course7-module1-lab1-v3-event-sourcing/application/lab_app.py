from dataclasses import dataclass

from application.interfaces.qa_agent_service_interface import (
    QAAgentServiceInterface,
)


@dataclass(frozen=True)
class LabApp:
    """V3a-shrunk lab application bundle.

    V2 carried three service fields (auth, qa, counter). V3a ships only
    the QA workflow integrated with the event-sourced substrate; Auth
    and Counter are added back as V3b and V3c land each workflow's
    translator, sensitive-field policy, and projection joins.

    Frozen — the bundle is immutable after construction. To rewire,
    call `initialise()` again.
    """

    qa: QAAgentServiceInterface