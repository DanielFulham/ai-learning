import hashlib
import inspect
from unittest.mock import MagicMock, call, create_autospec

import pytest

from application.permission_gate import PermissionGate
from application.policy_book import PolicyBook
from domain.approval_outcome import ApprovalOutcome
from domain.gate_decision import AskUser, GateDecision, Refuse, RunNow
from domain.gate_result import Completed, Failed
from interfaces.audit_log_interface import AuditLogInterface
from interfaces.policy_store_interface import InvalidPolicyError, PolicyStoreInterface


def _make_policy_book(rules: dict[str, str] | None = None) -> PolicyBook:
    store = MagicMock(spec=PolicyStoreInterface)
    store.load_rules.return_value = rules or {}
    return PolicyBook(store)


def _make_audit() -> MagicMock:
    return MagicMock(spec=AuditLogInterface)


def _make_gate(policy: str | None = None, audit: MagicMock | None = None) -> PermissionGate:
    rules = {"TEST_TOOL": policy} if policy else {}
    return PermissionGate(_make_policy_book(rules), audit or _make_audit())


def _tool_action() -> str:
    return ""


def _make_action(text: str = "TEST_RESULT") -> MagicMock:
    action = create_autospec(_tool_action)
    action.return_value = text
    return action


def _decide(
    gate: PermissionGate,
    outcome: ApprovalOutcome | None = None,
    token: str | None = None,
    tool: str = "TEST_TOOL",
    arguments: dict[str, object] | None = None,
) -> GateDecision:
    return gate.decide(tool, arguments or {}, outcome, token)


def _token_for(
    gate: PermissionGate, tool: str = "TEST_TOOL", arguments: dict[str, object] | None = None
) -> str:
    """The token the gate hands out for a call, read from its first-round answer."""
    decision = gate.decide(tool, arguments or {}, None, None)
    assert isinstance(decision, AskUser)
    return decision.token


def _events(audit: MagicMock) -> list[str]:
    return [recorded.args[2] for recorded in audit.append.call_args_list]


# --- allow ---------------------------------------------------------------------------------


def test_given_an_allow_policy_when_decided_then_the_tool_may_run_now() -> None:
    assert _decide(_make_gate("allow")) == RunNow()


def test_given_an_allow_policy_when_decided_then_allowed_is_audited_with_the_arguments() -> None:
    audit = _make_audit()
    _decide(_make_gate("allow", audit), arguments={"b": 1, "a": 2})
    assert audit.append.call_args_list == [
        call("TEST_TOOL", '{"a": 2, "b": 1}', "allowed", "policy: allow")
    ]


# --- deny ----------------------------------------------------------------------------------


def test_given_a_deny_policy_when_decided_then_it_is_refused_with_the_lab_text() -> None:
    assert _decide(_make_gate("deny")) == Refuse("Permission denied for tool: TEST_TOOL")


def test_given_a_deny_policy_when_decided_then_only_denied_is_audited() -> None:
    audit = _make_audit()
    _decide(_make_gate("deny", audit))
    assert audit.append.call_args_list == [call("TEST_TOOL", "{}", "denied", "policy: deny")]


@pytest.mark.parametrize("outcome", list(ApprovalOutcome))
def test_given_a_deny_policy_when_any_outcome_is_supplied_then_it_is_still_refused(
    outcome: ApprovalOutcome,
) -> None:
    """Pins F-L47-4: a deny is final, so no answer from the client can unlock it."""
    decision = _decide(_make_gate("deny"), outcome, token="TEST_ANY_TOKEN")
    assert isinstance(decision, Refuse)


# --- ask -----------------------------------------------------------------------------------


def test_given_an_ask_policy_and_no_answer_when_decided_then_the_user_is_asked() -> None:
    assert isinstance(_decide(_make_gate("ask")), AskUser)


def test_given_an_ask_policy_when_decided_then_the_question_names_the_tool_and_arguments() -> None:
    """Pins F-L47-3: the lab asked about arguments the user never saw."""
    decision = _decide(_make_gate("ask"), arguments={"filepath": "TEST.txt"})
    assert isinstance(decision, AskUser)
    assert "TEST_TOOL" in decision.message
    assert '"filepath": "TEST.txt"' in decision.message


def test_given_an_ask_policy_when_decided_then_the_request_is_audited() -> None:
    audit = _make_audit()
    _decide(_make_gate("ask", audit))
    assert _events(audit) == ["approval_requested"]


def test_given_the_same_call_when_asked_twice_then_each_question_gets_its_own_token() -> None:
    """Pins F-L47-28: tokens are random and server-issued, so none can be worked out in advance."""
    gate = _make_gate("ask")
    assert _token_for(gate, arguments={"a": 1}) != _token_for(gate, arguments={"a": 1})


def test_given_a_token_when_it_is_read_then_it_is_not_derived_from_the_call() -> None:
    """The old token was a hash of tool and arguments, which any client could compute itself."""
    gate = _make_gate("ask")
    token = _token_for(gate, arguments={"a": 1})
    assert token != hashlib.sha256(b'TEST_TOOL\n{"a": 1}').hexdigest()


def test_given_the_same_arguments_in_another_order_when_accepted_then_the_tool_may_run() -> None:
    gate = _make_gate("ask")
    token = _token_for(gate, arguments={"a": 1, "b": 2})
    assert _decide(gate, ApprovalOutcome.ACCEPTED, token, arguments={"b": 2, "a": 1}) == RunNow()


def test_given_an_accept_for_another_tool_when_decided_then_it_is_refused() -> None:
    gate = PermissionGate(_make_policy_book({"TEST_A": "ask", "TEST_B": "ask"}), _make_audit())
    token = _token_for(gate, "TEST_A")
    assert isinstance(_decide(gate, ApprovalOutcome.ACCEPTED, token, tool="TEST_B"), Refuse)


def test_given_a_token_that_was_used_when_it_is_used_again_then_the_second_use_is_refused() -> None:
    """Pins F-L47-28: one question, one answer. A replayed accept must not run the tool again."""
    gate = _make_gate("ask")
    token = _token_for(gate)
    assert _decide(gate, ApprovalOutcome.ACCEPTED, token) == RunNow()
    assert isinstance(_decide(gate, ApprovalOutcome.ACCEPTED, token), Refuse)


def test_given_a_declined_token_when_it_is_then_accepted_then_the_accept_is_refused() -> None:
    gate = _make_gate("ask")
    token = _token_for(gate)
    _decide(gate, ApprovalOutcome.DECLINED, token)
    assert isinstance(_decide(gate, ApprovalOutcome.ACCEPTED, token), Refuse)


def test_given_a_token_used_with_other_arguments_when_used_again_correctly_then_it_is_gone() -> (
    None
):
    """A wrong guess burns the token: the client has to be asked again."""
    gate = _make_gate("ask")
    token = _token_for(gate, arguments={"a": 1})
    _decide(gate, ApprovalOutcome.ACCEPTED, token, arguments={"a": 2})
    assert isinstance(_decide(gate, ApprovalOutcome.ACCEPTED, token, arguments={"a": 1}), Refuse)


def test_given_a_token_past_its_lifetime_when_accepted_then_it_is_refused() -> None:
    """Pins F-L47-28: an old question cannot be answered long after the user saw it."""
    clock = [0.0]
    gate = PermissionGate(
        _make_policy_book({"TEST_TOOL": "ask"}),
        _make_audit(),
        ttl_seconds=60,
        clock=lambda: clock[0],
    )
    token = _token_for(gate)
    clock[0] = 61.0
    assert isinstance(_decide(gate, ApprovalOutcome.ACCEPTED, token), Refuse)


def test_given_a_token_inside_its_lifetime_when_accepted_then_the_tool_may_run() -> None:
    clock = [0.0]
    gate = PermissionGate(
        _make_policy_book({"TEST_TOOL": "ask"}),
        _make_audit(),
        ttl_seconds=60,
        clock=lambda: clock[0],
    )
    token = _token_for(gate)
    clock[0] = 59.0
    assert _decide(gate, ApprovalOutcome.ACCEPTED, token) == RunNow()


def test_given_open_questions_when_counted_then_used_ones_are_not_held() -> None:
    gate = _make_gate("ask")
    first, _second = _token_for(gate), _token_for(gate)
    assert gate.pending_count() == 2
    _decide(gate, ApprovalOutcome.ACCEPTED, first)
    assert gate.pending_count() == 1


def test_given_too_many_open_questions_when_another_is_asked_then_it_is_refused() -> None:
    """Pins F-L47-28: unanswered questions cannot grow the gate's memory without limit."""
    gate = PermissionGate(_make_policy_book({"TEST_TOOL": "ask"}), _make_audit(), max_pending=2)
    _token_for(gate)
    _token_for(gate)
    decision = _decide(gate)
    assert isinstance(decision, Refuse)
    assert gate.pending_count() == 2


def test_given_a_full_gate_when_old_questions_expire_then_a_new_one_is_accepted() -> None:
    clock = [0.0]
    gate = PermissionGate(
        _make_policy_book({"TEST_TOOL": "ask"}),
        _make_audit(),
        ttl_seconds=60,
        max_pending=1,
        clock=lambda: clock[0],
    )
    _token_for(gate)
    clock[0] = 61.0
    assert isinstance(_decide(gate), AskUser)
    assert gate.pending_count() == 1


def test_given_an_accept_with_the_matching_token_when_decided_then_the_tool_may_run() -> None:
    gate = _make_gate("ask")
    token = _token_for(gate, arguments={"a": 1})
    decision = _decide(gate, ApprovalOutcome.ACCEPTED, token, arguments={"a": 1})
    assert decision == RunNow()


def test_given_an_accept_with_the_matching_token_when_decided_then_approved_is_audited() -> None:
    """Pins F-L47-7: the lab logged an approved run exactly like an automatic allow."""
    audit = _make_audit()
    gate = _make_gate("ask", audit)
    token = _token_for(gate)
    audit.reset_mock()
    _decide(gate, ApprovalOutcome.ACCEPTED, token)
    assert _events(audit) == ["approved"]


def test_given_an_accept_for_other_arguments_when_decided_then_it_is_refused() -> None:
    """Pins F-L47-3: an approval cannot be replayed against arguments the user never saw."""
    gate = _make_gate("ask")
    token = _token_for(gate, arguments={"filepath": "SAFE.txt"})
    decision = _decide(gate, ApprovalOutcome.ACCEPTED, token, arguments={"filepath": "OTHER.txt"})
    assert isinstance(decision, Refuse)
    assert "different arguments" in decision.reason


def test_given_an_accept_with_no_token_when_decided_then_it_is_refused() -> None:
    """Pins that "accepted" alone is not enough: the caller must present the bound token."""
    assert isinstance(_decide(_make_gate("ask"), ApprovalOutcome.ACCEPTED, None), Refuse)


def test_given_an_accept_with_a_made_up_token_when_decided_then_it_is_refused_and_audited() -> None:
    audit = _make_audit()
    decision = _decide(_make_gate("ask", audit), ApprovalOutcome.ACCEPTED, "TEST_MADE_UP")
    assert isinstance(decision, Refuse)
    assert _events(audit) == ["denied"]


@pytest.mark.parametrize(
    ("outcome", "event", "word"),
    [
        (ApprovalOutcome.DECLINED, "declined", "declined"),
        (ApprovalOutcome.CANCELLED, "cancelled", "cancelled"),
    ],
)
def test_given_no_approval_with_the_matching_token_when_decided_then_the_tool_is_refused(
    outcome: ApprovalOutcome, event: str, word: str
) -> None:
    audit = _make_audit()
    gate = _make_gate("ask", audit)
    token = _token_for(gate)
    audit.reset_mock()
    decision = _decide(gate, outcome, token)
    assert isinstance(decision, Refuse)
    assert word in decision.reason
    assert _events(audit) == [event]


def test_given_a_tool_with_no_rule_when_decided_then_the_user_is_asked() -> None:
    """Pins fail-safe: an unknown tool is never allowed to run unprompted."""
    gate = _make_gate()
    assert isinstance(_decide(gate, tool="TEST_UNKNOWN_TOOL"), AskUser)


def test_given_the_gate_when_its_decide_method_is_inspected_then_it_takes_no_approved_flag() -> (
    None
):
    """Pins F-L47-3: the lab's approved=True was a boolean any caller could pass."""
    assert "approved" not in inspect.signature(PermissionGate.decide).parameters


# --- unreadable policy ---------------------------------------------------------------------


def test_given_an_unreadable_policy_file_when_decided_then_it_is_refused_and_audited() -> None:
    """Pins F-L47-5: a bad policy file refuses every call instead of running the tool."""
    store = MagicMock(spec=PolicyStoreInterface)
    store.load_rules.side_effect = InvalidPolicyError("TEST_BAD_FILE")
    audit = _make_audit()
    decision = PermissionGate(PolicyBook(store), audit).decide("TEST_TOOL", {}, None, None)
    assert isinstance(decision, Refuse)
    assert "TEST_BAD_FILE" in decision.reason
    assert _events(audit) == ["denied"]


# --- execute -------------------------------------------------------------------------------


def test_given_an_action_when_executed_then_its_text_comes_back_and_succeeded_is_audited() -> None:
    audit = _make_audit()
    action = _make_action("TEST_RESULT")
    result = _make_gate("allow", audit).execute("TEST_TOOL", {"a": 1}, action)
    assert result == Completed("TEST_RESULT")
    action.assert_called_once_with()
    assert audit.append.call_args_list == [call("TEST_TOOL", '{"a": 1}', "succeeded", "")]


def test_given_an_action_that_raises_when_executed_then_failed_is_returned_and_audited() -> None:
    audit = _make_audit()
    action = _make_action()
    action.side_effect = FileNotFoundError("TEST_MISSING")
    result = _make_gate("allow", audit).execute("TEST_TOOL", {}, action)
    assert isinstance(result, Failed)
    assert "TEST_MISSING" in result.reason
    assert audit.append.call_args_list == [call("TEST_TOOL", "{}", "failed", "TEST_MISSING")]


def test_given_the_gate_when_decide_and_execute_are_inspected_then_neither_waits_for_the_user() -> (
    None
):
    """Pins the stateless design: no coroutine here ever blocks on a person."""
    assert not inspect.iscoroutinefunction(PermissionGate.decide)
    assert not inspect.iscoroutinefunction(PermissionGate.execute)


# --- audit failures after the tool has run ---------------------------------------------------


def _audit_failing_on(event: str) -> MagicMock:
    audit = _make_audit()

    def append(tool: str, arguments_json: str, recorded: str, detail: str) -> None:
        if recorded == event:
            raise OSError("TEST_DISK_FULL")

    audit.append.side_effect = append
    return audit


def test_given_the_log_fails_after_a_successful_tool_when_executed_then_the_result_still_comes_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pins F-L47-30: the change happened, so the client is told it did, and the gap is logged."""
    gate = _make_gate("allow", _audit_failing_on("succeeded"))
    result = gate.execute("TEST_TOOL", {}, _make_action("TEST_RESULT"))
    assert result == Completed("TEST_RESULT")
    assert "TEST_TOOL" in caplog.text
    assert "succeeded" in caplog.text


def test_given_the_log_fails_after_a_failed_tool_when_executed_then_the_failure_comes_back() -> (
    None
):
    action = _make_action()
    action.side_effect = FileNotFoundError("TEST_MISSING")
    gate = _make_gate("allow", _audit_failing_on("failed"))
    result = gate.execute("TEST_TOOL", {}, action)
    assert isinstance(result, Failed)
    assert "TEST_MISSING" in result.reason


def test_given_the_log_fails_before_the_tool_when_decided_then_the_call_is_not_allowed() -> None:
    """Pins fail-closed: with no record of the intent, the tool must not run."""
    gate = _make_gate("allow", _audit_failing_on("allowed"))
    with pytest.raises(OSError, match="TEST_DISK_FULL"):
        _decide(gate)
