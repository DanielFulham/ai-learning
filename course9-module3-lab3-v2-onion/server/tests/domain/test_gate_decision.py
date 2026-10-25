from dataclasses import FrozenInstanceError

import pytest

from domain.gate_decision import AskUser, GateDecision, Refuse, RunNow


def test_given_an_ask_user_decision_when_built_then_it_holds_the_message_and_the_token() -> None:
    decision = AskUser(message="TEST_MESSAGE", token="TEST_TOKEN")
    assert decision.message == "TEST_MESSAGE"
    assert decision.token == "TEST_TOKEN"


def test_given_a_refuse_decision_when_built_then_it_holds_the_reason() -> None:
    assert Refuse("TEST_REASON").reason == "TEST_REASON"


def test_given_two_run_now_decisions_when_compared_then_they_are_equal() -> None:
    assert RunNow() == RunNow()


def test_given_the_three_decisions_when_compared_then_none_equals_another() -> None:
    """Pins that the caller can tell run, refuse and ask apart by type."""
    decisions: list[GateDecision] = [RunNow(), Refuse("TEST"), AskUser("TEST", "TEST")]
    assert len({type(decision) for decision in decisions}) == 3


@pytest.mark.parametrize(
    ("decision", "field"),
    [(Refuse("TEST"), "reason"), (AskUser("TEST", "TEST"), "message")],
)
def test_given_a_decision_when_a_field_is_assigned_then_it_is_frozen(
    decision: Refuse | AskUser, field: str
) -> None:
    """Pins immutability: a decision cannot be rewritten between the gate and the tool."""
    with pytest.raises(FrozenInstanceError):
        setattr(decision, field, "TEST_CHANGED")
