from typing import Protocol

from domain.approval import ApprovalChoice
from interfaces.approval_source_interface import ApprovalSourceInterface


class ApprovalDeskInterface(ApprovalSourceInterface, Protocol):
    """One page session's approval desk, as its entry point and host service use it.

    The session awaits `decide` when the server asks. The page shows `pending_message` and
    calls `respond` from a button or a typed reply.
    """

    def pending_message(self) -> str | None:
        """The oldest question still waiting for an answer, or None."""
        ...

    def begin_turn(self) -> bool:
        """Claim the session's one ordinary turn. False means a turn is already running."""
        ...

    def end_turn(self) -> None:
        """Give the turn back, whether it finished or failed."""
        ...

    def respond(self, choice: ApprovalChoice) -> bool:
        """Answer the oldest waiting question. False means nothing was waiting."""
        ...
