from application.approval_broker import ApprovalBroker
from application.interfaces.approval_desk_interface import ApprovalDeskInterface


class ApprovalDesks:
    """One approval desk per page session, made on first use.

    The services are shared by every browser session, so the question a tool call raises has to
    go to the desk of the session that made the call, and a typed "yes" has to land on that same
    desk. A session id (Gradio's session hash) picks the desk.
    """

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._desks: dict[str, ApprovalBroker] = {}

    def desk_for(self, session_id: str) -> ApprovalDeskInterface:
        if session_id not in self._desks:
            self._desks[session_id] = ApprovalBroker(self._timeout_seconds)
        return self._desks[session_id]

    def forget(self, session_id: str) -> None:
        """Drop a session's desk. A question still waiting on it is cancelled, not timed out."""
        desk = self._desks.pop(session_id, None)
        if desk is not None:
            desk.cancel_all()

    def count(self) -> int:
        return len(self._desks)
