from dataclasses import dataclass


@dataclass(frozen=True)
class RunNow:
    """The policy allows the call; the tool may run."""


@dataclass(frozen=True)
class Refuse:
    """The call must not run, because of the policy or the user's answer."""

    reason: str


@dataclass(frozen=True)
class AskUser:
    """The policy says ask. `token` binds the question to these exact arguments."""

    message: str
    token: str


GateDecision = RunNow | Refuse | AskUser
