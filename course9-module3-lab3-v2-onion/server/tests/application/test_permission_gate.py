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


def test_given_the_same_call_when_asked_twice_then_the_token_is_the_same() -> None:
    gate = _make_gate("ask")
    assert _token_for(gate, arguments={"a": 1}) == _token_for(gate, arguments={"a": 1})


def test_given_different_arguments_when_asked_then_the_tokens_differ() -> None:
    """Pins that the token is bound to the exact arguments the user is shown."""
    gate = _make_gate("ask")
    assert _token_for(gate, arguments={"a": 1}) != _token_for(gate, arguments={"a": 2})


def test_given_a_different_tool_when_asked_then_the_token_differs() -> None:
    gate = PermissionGate(_make_policy_book({"TEST_A": "ask", "TEST_B": "ask"}), _make_audit())
    assert _token_for(gate, "TEST_A") != _token_for(gate, "TEST_B")


def test_given_the_same_arguments_in_another_order_when_asked_then_the_token_is_the_same() -> None:
    gate = _make_gate("ask")
    first = _token_for(gate, arguments={"a": 1, "b": 2})
    assert first == _token_for(gate, arguments={"b": 2, "a": 1})


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
