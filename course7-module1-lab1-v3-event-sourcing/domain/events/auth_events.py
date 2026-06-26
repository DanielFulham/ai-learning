from dataclasses import dataclass

from domain.events.base import BaseAgentEvent


@dataclass(frozen=True, kw_only=True)
class LoginAttempted(BaseAgentEvent):
    """Credentials submitted; validation has not yet run.

    `username` is non-nullable (str, not str | None). V2's
    AuthCredentials.username is nullable because the carrier starts empty
    and fills in. By the time LoginAttempted fires, username is a real
    string — events pin the strongest contract.

    `password` is absent by design. F03's sensitive-field policy is
    enforced structurally at the dataclass shape in C1, and procedurally
    by the translator's codec helper in C3 (belt and braces). The
    persisted payload never carries a password field.
    """

    username: str


@dataclass(frozen=True, kw_only=True)
class LoginSucceeded(BaseAgentEvent):
    """Credentials validated successfully.

    `username` is non-nullable (str, not str | None). V2's
    AuthCredentials.username is nullable because the carrier starts empty
    and fills in. By the time LoginSucceeded fires, username is a real
    string — events pin the strongest contract.

    `message` is non-nullable because the success node stamps it.
    message is absent from LoginAttempted because V2 stamps no message at
    attempt time.
    """

    username: str
    message: str


@dataclass(frozen=True, kw_only=True)
class LoginFailed(BaseAgentEvent):
    """Credentials rejected.

    `username` is non-nullable (str, not str | None). V2's
    AuthCredentials.username is nullable because the carrier starts empty
    and fills in. By the time LoginFailed fires, username is a real
    string — events pin the strongest contract.

    `message` is non-nullable because the failure node stamps it.
    message is absent from LoginAttempted because V2 stamps no message at
    attempt time.
    """

    username: str
    message: str


AuthEvent = LoginAttempted | LoginSucceeded | LoginFailed
"""Closed union over the Auth service's event types.

Open across services (QAEvent and CounterEvent are separate unions),
closed within (the Auth translator returns `list[AuthEvent]` and pyright
enforces exhaustiveness at the translator boundary via assert_never).

Adding a new Auth event type is a one-line union extension and a new
dataclass under this module. Removing one is a breaking schema change
that should be a deliberate version bump on the affected event,
not a silent union shrink.
"""
