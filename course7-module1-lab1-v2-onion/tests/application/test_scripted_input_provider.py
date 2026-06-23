import pytest

from application.scripted_input_provider import ScriptedInputProvider


def test_returns_responses_in_order() -> None:
    provider = ScriptedInputProvider(["first", "second", "third"])
    assert provider.prompt("ignored") == "first"
    assert provider.prompt("ignored") == "second"
    assert provider.prompt("ignored") == "third"


def test_message_argument_is_ignored() -> None:
    """The scripted provider returns canned responses regardless of message."""
    provider = ScriptedInputProvider(["canned"])
    assert provider.prompt("any message") == "canned"


def test_empty_responses_list_raises_on_first_call() -> None:
    provider = ScriptedInputProvider([])
    with pytest.raises(IndexError, match="exhausted after 0 responses"):
        provider.prompt("anything")


def test_exhaustion_raises_with_helpful_message() -> None:
    provider = ScriptedInputProvider(["only_one"])
    provider.prompt("first call")
    with pytest.raises(IndexError, match="exhausted after 1 responses"):
        provider.prompt("second call")