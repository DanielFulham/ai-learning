import hashlib
import json
from collections.abc import Callable, Mapping

from application.policy_book import PolicyBook
from domain.approval_outcome import ApprovalOutcome
from domain.audit_event import AuditEvent
from domain.gate_decision import AskUser, GateDecision, Refuse, RunNow
from domain.gate_result import Completed, Failed
from domain.policy import Policy
from interfaces.audit_log_interface import AuditLogInterface
from interfaces.policy_store_interface import InvalidPolicyError


class PermissionGate:
    """Decides whether a tool call may run, and audits each step. Stateless and synchronous.

    The gate never waits for a person. When the policy says ask, `decide` returns an `AskUser`
    carrying a token bound to the exact tool and arguments. The client's answer comes back on a
    retry together with that token, and only a matching token lets an accept through. There is
    no "approved" flag a caller can pass. Every path that is not an explicit allow, or an
    accept with the matching token, leaves the tool unrun.
    """

    def __init__(self, policy_book: PolicyBook, audit_log: AuditLogInterface) -> None:
        self._policy_book = policy_book
        self._audit_log = audit_log

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

        expected_token = _token_for(tool, arguments_json)
        if outcome is None:
            self._record(tool, arguments_json, AuditEvent.APPROVAL_REQUESTED, "policy: ask")
            return AskUser(f"Run {tool} with arguments {arguments_json}?", expected_token)
        if token != expected_token:
            detail = "the approval does not match these arguments"
            self._record(tool, arguments_json, AuditEvent.DENIED, detail)
            return Refuse(f"The approval was for different arguments, so {tool} was not run.")
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
            self._record(tool, arguments_json, AuditEvent.FAILED, str(error))
            return Failed(f"{tool} failed: {error}")
        self._record(tool, arguments_json, AuditEvent.SUCCEEDED, "")
        return Completed(text)

    def _record(self, tool: str, arguments_json: str, event: AuditEvent, detail: str) -> None:
        self._audit_log.append(tool, arguments_json, event.value, detail)


def _token_for(tool: str, arguments_json: str) -> str:
    return hashlib.sha256(f"{tool}\n{arguments_json}".encode()).hexdigest()
