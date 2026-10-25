from dataclasses import FrozenInstanceError

import pytest

from domain.gate_result import Completed, Failed


def test_given_a_completed_result_when_built_then_it_holds_the_text() -> None:
    assert Completed("TEST_TEXT").text == "TEST_TEXT"


def test_given_a_failed_result_when_built_then_it_holds_the_reason() -> None:
    assert Failed("TEST_REASON").reason == "TEST_REASON"


def test_given_completed_and_failed_with_the_same_text_when_compared_then_they_differ() -> None:
    """Pins that the caller can tell a tool that ran from a tool that broke."""
    assert Completed("TEST") != Failed("TEST")


@pytest.mark.parametrize("result", [Completed("TEST"), Failed("TEST")])
def test_given_any_result_when_a_field_is_assigned_then_it_is_frozen(
    result: Completed | Failed,
) -> None:
    """Pins immutability: a result cannot be rewritten after the gate returns it."""
    field = "text" if isinstance(result, Completed) else "reason"
    with pytest.raises(FrozenInstanceError):
        setattr(result, field, "TEST_CHANGED")
