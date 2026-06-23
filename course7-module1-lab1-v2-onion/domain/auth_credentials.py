from dataclasses import dataclass


@dataclass(frozen=True)
class AuthCredentials:
    """Immutable credentials carrying the auth verdict.

    Invariant: an authentication verdict of True can only exist alongside
    a non-empty username. A True verdict without a username is structurally
    impossible — the verdict is *about* the credentials it carries.
    """

    username: str | None = None
    password: str | None = None
    is_authenticated: bool | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.is_authenticated is True:
            if self.username is None or self.username == "":
                raise ValueError(
                    "AuthCredentials with is_authenticated=True requires a non-empty username"
                )