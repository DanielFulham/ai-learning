import asyncio
from collections import deque

from application.mappers import answer_from_choice
from domain.approval import ApprovalChoice
from interfaces.approval_source_interface import ApprovalAnswer, ApprovalSourceInterface


class ApprovalBroker(ApprovalSourceInterface):
    """Holds the server's questions until a person answers them. No UI, just asyncio.

    `decide` is what the session awaits when the server asks. The page shows `pending_message`
    and calls `respond` from a button or a typed "yes". Questions are answered oldest first.
    A question nobody answers within `timeout_seconds` counts as a cancel, never an accept, and
    a question whose call was abandoned stops being pending.
    """

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._waiting: deque[tuple[str, asyncio.Future[ApprovalAnswer]]] = deque()

    async def decide(self, message: str) -> ApprovalAnswer:
        future: asyncio.Future[ApprovalAnswer] = asyncio.get_running_loop().create_future()
        entry = (message, future)
        self._waiting.append(entry)
        try:
            return await asyncio.wait_for(future, self._timeout_seconds)
        except TimeoutError:
            return "cancel"
        finally:
            if entry in self._waiting:
                self._waiting.remove(entry)

    def waiting_count(self) -> int:
        """How many questions the broker is still holding: answered, timed-out and abandoned
        ones are removed, so this is also how a leak would show."""
        return len(self._waiting)

    def cancel_all(self) -> None:
        """Answer every waiting question with cancel, for a page that has closed."""
        for _, future in self._waiting:
            if not future.done():
                future.set_result("cancel")

    def pending_message(self) -> str | None:
        """The oldest question still waiting for an answer, or None."""
        for message, future in self._waiting:
            if not future.done():
                return message
        return None

    def respond(self, choice: ApprovalChoice) -> bool:
        """Answer the oldest waiting question. False means nothing was waiting."""
        for _, future in self._waiting:
            if not future.done():
                future.set_result(answer_from_choice(choice))
                return True
        return False
