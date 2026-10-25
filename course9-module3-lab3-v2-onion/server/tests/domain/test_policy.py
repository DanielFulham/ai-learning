import pytest

from domain.policy import Policy


def test_given_the_text_allow_when_parsed_as_a_policy_then_it_is_the_allow_policy() -> None:
    assert Policy("allow") is Policy.ALLOW


def test_given_the_text_deny_when_parsed_as_a_policy_then_it_is_the_deny_policy() -> None:
    assert Policy("deny") is Policy.DENY


def test_given_the_text_ask_when_parsed_as_a_policy_then_it_is_the_ask_policy() -> None:
    assert Policy("ask") is Policy.ASK


@pytest.mark.parametrize("text", ["alow", "ALLOW", "Allow", "", " allow", "TEST_UNKNOWN"])
def test_given_text_that_is_not_exactly_a_policy_when_parsed_then_a_value_error_is_raised(
    text: str,
) -> None:
    """Pins F-L47-5: the lab let any string other than deny or ask run the tool."""
    with pytest.raises(ValueError):
        Policy(text)


def test_given_the_policy_type_when_listed_then_only_allow_deny_and_ask_exist() -> None:
    """Pins the closed set: adding a fourth policy has to break this test."""
    assert {policy.value for policy in Policy} == {"allow", "deny", "ask"}


def test_given_a_policy_when_compared_with_its_text_then_they_are_equal() -> None:
    """Policy is a string enum, so it round-trips through the JSON policy file unchanged."""
    assert Policy.ASK == "ask"
