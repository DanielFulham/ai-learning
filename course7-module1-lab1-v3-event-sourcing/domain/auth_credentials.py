from dataclasses import dataclass


@dataclass(frozen=True)
class AuthCredentials:
    """V2's auth state carrier, lifted into V3b.

    Carries username, password, verdict, and message through the entire
    Auth flow. All fields default to None — the carrier starts empty
    at input_node and fills in across the graph traversal.

    V3b note: this is the state-carrier vocabulary. The event vocabulary
    (LoginAttempted, LoginSucceeded, LoginFailed) is decoupled from this
    via the auth translator's dispatch — events pin the strongest contract
    (str, not str | None), the carrier admits the in-progress states.
    """

    username: str | None = None
    password: str | None = None
    is_authenticated: bool | None = None
    message: str | None = None
