from typing import Protocol

from domain.auth_result import AuthResult


class AuthAgentServiceInterface(Protocol):
    """Contract for the Auth workflow's application service.

    The entry point depends on this Protocol, not on the concrete
    `AuthAgentService`. Lets the entry point be unaware of the V3b
    constructor surface (graph, event store, inner consumer, clock).

    `run()` is parameterless — the auth flow takes its input interactively
    through the graph's input provider (InputNode prompts for the
    username/password), so there is no argument to pass. It returns an
    `AuthResult` carrying the final credentials (verdict + message) and
    the run_id the service generated. Callers that need the run_id (the
    demo, projection consumers) access `result.run_id`; callers that care
    about the verdict access `result.credentials.is_authenticated`.
    """

    def run(self) -> AuthResult: ...
