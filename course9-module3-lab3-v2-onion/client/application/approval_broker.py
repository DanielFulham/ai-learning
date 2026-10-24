import asyncio
import threading
from collections import deque
from dataclasses import dataclass

from application.mappers import answer_from_choice
from domain.approval import ApprovalChoice
from interfaces.approval_source_interface import ApprovalAnswer, ApprovalSourceInterface


@dataclass
class _Waiting:
    message: str
    future: asyncio.Future[ApprovalAnswer]
    loop: asyncio.AbstractEventLoop
    claimed: bool = False


def _set_if_open(future: asyncio.Future[ApprovalAnswer], answer: ApprovalAnswer) -> None:
    if not future.done():
        future.set_result(answer)


class ApprovalBroker(ApprovalSourceInterface):
    """Holds the server's questions until a person answers them. No UI, just asyncio.

    `decide` is what the session awaits when the server asks. The page shows `pending_message`
    and calls `respond` from a button or a typed "yes". Questions are answered oldest first.
    A question nobody answers within `timeout_seconds` counts as a cancel, never an accept, and
    a question whose call was abandoned stops being pending.

    Gradio runs button callbacks on worker threads, while `decide` waits on the event loop's
    thread. So an answer is claimed under a lock and handed to the waiting loop with
    `call_soon_threadsafe`; a question can be claimed once, by whichever thread gets there first.
    """

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._waiting: deque[_Waiting] = deque()
        self._turn_running = False

    async def decide(self, message: str) -> ApprovalAnswer:
        loop = asyncio.get_running_loop()
        entry = _Waiting(message, loop.create_future(), loop)
        with self._lock:
            self._waiting.append(entry)
        try:
            return await asyncio.wait_for(entry.future, self._timeout_seconds)
        except TimeoutError:
            return "cancel"
        finally:
            with self._lock:
                if entry in self._waiting:
                    self._waiting.remove(entry)

    def begin_turn(self) -> bool:
        """Claim the session's one ordinary turn. False means a turn is already running."""
        with self._lock:
            if self._turn_running:
                return False
            self._turn_running = True
            return True

    def end_turn(self) -> None:
        with self._lock:
            self._turn_running = False

    def waiting_count(self) -> int:
        """How many questions the broker is still holding: answered, timed-out and abandoned
        ones are removed, so this is also how a leak would show."""
        with self._lock:
            return len(self._waiting)

    def pending_message(self) -> str | None:
        """The oldest question still waiting for an answer, or None."""
        with self._lock:
            for entry in self._waiting:
                if not entry.claimed:
                    return entry.message
        return None

    def respond(self, choice: ApprovalChoice) -> bool:
        """Answer the oldest waiting question. False means nothing was waiting."""
        with self._lock:
            entry = next((e for e in self._waiting if not e.claimed), None)
            if entry is None:
                return False
            entry.claimed = True
        self._deliver(entry, answer_from_choice(choice))
        return True

    def cancel_all(self) -> None:
        """Answer every waiting question with cancel, for a page that has closed."""
        with self._lock:
            entries = [e for e in self._waiting if not e.claimed]
            for entry in entries:
                entry.claimed = True
        for entry in entries:
            self._deliver(entry, "cancel")

    @staticmethod
    def _deliver(entry: _Waiting, answer: ApprovalAnswer) -> None:
        try:
            entry.loop.call_soon_threadsafe(_set_if_open, entry.future, answer)
        except RuntimeError:
            # The loop has closed, so the call that asked is gone and nobody is waiting.
            pass
