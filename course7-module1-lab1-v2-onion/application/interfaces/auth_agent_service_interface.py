from typing import Protocol

from domain.auth_credentials import AuthCredentials
from domain.state_schemas import AuthState


class AuthAgentServiceInterface(Protocol):
    """Service contract the entry point consumes for the Auth workflow.

    `application/interfaces/` rather than root `interfaces/` because the
    application service is the only implementer and the entry point is
    the only consumer — no competing concretes.
    """

    def run(self, initial_state: AuthState | None = None) -> AuthCredentials: ...