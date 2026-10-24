import json
import logging
import secrets
import time
from collections.abc import Callable, Mapping

from application.policy_book import PolicyBook
from domain.approval_outcome import ApprovalOutcome
from domain.audit_event import AuditEvent
from domain.gate_decision import AskUser, GateDecision, Refuse, RunNow
from domain.gate_result import Completed, Failed
from domain.policy import Policy
from interfaces.audit_log_interface import AuditLogInterface
from interfaces.policy_store_interface import InvalidPolicyError

_logger = logging.getLogger(__name__)


class PermissionGate:
    """Decides whether a tool call may run, and audits each step. Never waits for a person.

    When the policy says ask, `decide` returns an `AskUser` carrying a random token that the gate
    remembers, with the tool and arguments it was issued for. The client's answer comes back on a
    retry with that token. A token works once, for those exact arguments, within `ttl_seconds`;
    any use spends it, and an unknown, spent, expired or mismatched token is refused. At most
    `max_pending` questions are held, so unanswered ones cannot grow memory without limit. There
    is no "approved" flag a caller can pass.

    The audit record of an intent (allowed, asked, approved, refused) is written before the tool
    runs, and a failure there stops the call. The record of how the tool ended is written after;
    if that one fails the result is still returned and the failure is logged.

    The token shows the server asked this question. It does not show a person answered it: a
    client that holds the token can send "accept" itself, so the user has to trust the client.
    """

    def __init__(
        self,
        policy_book: PolicyBook,
        audit_log: AuditLogInterface,
        ttl_seconds: float = 300.0,
        max_pending: int = 100,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy_book = policy_book
        self._audit_log = audit_log
        self._ttl_seconds = ttl_seconds
        self._max_pending = max_pending
        self._clock = clock
        self._pending: dict[str, tuple[str, str, float]] = {}

    def pending_count(self) -> int:
        """How many questions are waiting for an answer, not counting expired ones."""
        self._drop_expired()
        return len(self._pending)

    def decide(
        self,
        tool: str,
        arguments: Mapping[str, object],
        outcome: ApprovalOutcome | None,
        token: str | None,
    ) -> GateDecision:
        arguments_json = json.dumps(arguments, sort_keys=True)
        try:
            policy = self._policy_book.policy_for(tool)
        except InvalidPolicyError as error:
            self._record(tool, arguments_json, AuditEvent.DENIED, f"policy unreadable: {error}")
            return Refuse(f"The policy could not be read, so {tool} was not run: {error}")

        if policy is Policy.DENY:
            self._record(tool, arguments_json, AuditEvent.DENIED, "policy: deny")
            return Refuse(f"Permission denied for tool: {tool}")
        if policy is Policy.ALLOW:
            self._record(tool, arguments_json, AuditEvent.ALLOWED, "policy: allow")
            return RunNow()

        if outcome is None:
            self._drop_expired()
            if len(self._pending) >= self._max_pending:
                self._record(tool, arguments_json, AuditEvent.DENIED, "too many open questions")
                return Refuse(f"Too many approvals are waiting, so {tool} was not run.")
            issued = secrets.token_urlsafe(32)
            self._pending[issued] = (tool, arguments_json, self._clock() + self._ttl_seconds)
            self._record(tool, arguments_json, AuditEvent.APPROVAL_REQUESTED, "policy: ask")
            return AskUser(f"Run {tool} with arguments {arguments_json}?", issued)
        if not self._spend(token, tool, arguments_json):
            detail = "the approval is missing, used, expired or for different arguments"
            self._record(tool, arguments_json, AuditEvent.DENIED, detail)
            return Refuse(
                f"The approval was missing, used, expired or for different arguments, "
                f"so {tool} was not run."
            )
        if outcome is ApprovalOutcome.DECLINED:
            self._record(tool, arguments_json, AuditEvent.DECLINED, "declined by the user")
            return Refuse(f"The user declined to run {tool}.")
        if outcome is ApprovalOutcome.CANCELLED:
            self._record(tool, arguments_json, AuditEvent.CANCELLED, "cancelled by the user")
            return Refuse(f"The request to run {tool} was cancelled.")
        self._record(tool, arguments_json, AuditEvent.APPROVED, "accepted by the user")
        return RunNow()

    def execute(
        self, tool: str, arguments: Mapping[str, object], action: Callable[[], str]
    ) -> Completed | Failed:
        arguments_json = json.dumps(arguments, sort_keys=True)
        try:
            text = action()
        except Exception as error:  # a tool failure is reported, not raised
            self._record_completion(tool, arguments_json, AuditEvent.FAILED, str(error))
            return Failed(f"{tool} failed: {error}")
        self._record_completion(tool, arguments_json, AuditEvent.SUCCEEDED, "")
        return Completed(text)

    def _record(self, tool: str, arguments_json: str, event: AuditEvent, detail: str) -> None:
        self._audit_log.append(tool, arguments_json, event.value, detail)

    def _record_completion(
        self, tool: str, arguments_json: str, event: AuditEvent, detail: str
    ) -> None:
        """Record how a tool ended. The tool has already run, so a log failure here cannot undo
        it: the result is still returned and the gap goes to the server's own log. Records made
        before the tool runs are not treated this way; if they fail, the tool does not run."""
        try:
            self._record(tool, arguments_json, event, detail)
        except OSError:
            _logger.exception("Could not audit %s for %s after it ran", event.value, tool)

    def _spend(self, token: str | None, tool: str, arguments_json: str) -> bool:
        """Use up a token. True only if it was issued for this call and has not expired."""
        if token is None:
            return False
        issued = self._pending.pop(token, None)
        if issued is None:
            return False
        issued_tool, issued_arguments, expires_at = issued
        return (
            issued_tool == tool
            and issued_arguments == arguments_json
            and self._clock() < expires_at
        )

    def _drop_expired(self) -> None:
        now = self._clock()
        for token in [t for t, (_, _, expires_at) in self._pending.items() if now >= expires_at]:
            del self._pending[token]
