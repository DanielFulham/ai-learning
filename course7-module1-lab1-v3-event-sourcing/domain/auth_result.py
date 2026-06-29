from dataclasses import dataclass
from uuid import UUID

from domain.auth_credentials import AuthCredentials


@dataclass(frozen=True)
class AuthResult:
    """One auth run's outcome plus its run_id.

    Returned by `AuthAgentService.run()`. The credentials carry the final
    verdict (`is_authenticated`) and the UI message; the run_id correlates
    this run with its events in the event store, so callers can
    replay-by-run for projections or composition assertions.

    Mirrors `RunResult`'s shape (one domain object + run_id) for the auth
    workflow. The two are deliberately separate types, not a shared
    generic — duplication is acceptable at two workflows, the same
    reasoning that keeps the two services and two translators separate.
    """

    credentials: AuthCredentials
    run_id: UUID
