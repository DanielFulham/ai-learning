from typing import Protocol


class InvalidPolicyError(Exception):
    """Raised when the stored policy cannot be trusted: unreadable, malformed or a bad value."""


class PolicyStoreInterface(Protocol):
    def load_rules(self) -> dict[str, str]:
        """Return the stored rules as tool name -> policy text, as written in the file."""
        ...
