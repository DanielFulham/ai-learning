from dataclasses import dataclass

from application.interfaces.auth_agent_service_interface import AuthAgentServiceInterface
from application.interfaces.counter_agent_service_interface import (
    CounterAgentServiceInterface,
)
from application.interfaces.qa_agent_service_interface import QAAgentServiceInterface


@dataclass(frozen=True)
class LabApp:
    """The three agent services bundled for the entry point.

    Returned by `application.container.initialise()`. `demo.py` picks
    which service to run based on the CLI subcommand. The dataclass is
    frozen so an entry point can't mutate the wired-up app structure
    after composition.
    """

    auth: AuthAgentServiceInterface
    qa: QAAgentServiceInterface
    counter: CounterAgentServiceInterface