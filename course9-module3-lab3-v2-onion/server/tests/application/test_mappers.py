import pytest

from application.mappers import parse_policy_rules
from domain.policy import Policy
from interfaces.policy_store_interface import InvalidPolicyError


def test_given_valid_rule_text_when_parsed_then_each_tool_gets_its_policy() -> None:
    raw = {"TEST_TOOL_A": "allow", "TEST_TOOL_B": "deny", "TEST_TOOL_C": "ask"}
    rules = parse_policy_rules(raw)
    assert rules == {
        "TEST_TOOL_A": Policy.ALLOW,
        "TEST_TOOL_B": Policy.DENY,
        "TEST_TOOL_C": Policy.ASK,
    }


def test_given_no_rules_when_parsed_then_the_result_is_an_empty_dict() -> None:
    assert parse_policy_rules({}) == {}


@pytest.mark.parametrize("value", ["alow", "ALLOW", "", " allow", "TEST_UNKNOWN"])
def test_given_a_value_that_is_not_a_policy_when_parsed_then_invalid_policy_is_raised(
    value: str,
) -> None:
    """Pins F-L47-5: the lab ran the tool for any value other than deny or ask."""
    with pytest.raises(InvalidPolicyError):
        parse_policy_rules({"TEST_TOOL": value})


def test_given_a_bad_value_when_parsed_then_the_error_names_the_tool_and_the_value() -> None:
    """The message is read by the person editing the file, so it says what to change."""
    with pytest.raises(InvalidPolicyError) as error:
        parse_policy_rules({"TEST_TOOL": "alow"})
    assert "TEST_TOOL" in str(error.value)
    assert "alow" in str(error.value)


def test_given_an_argument_level_key_when_parsed_then_invalid_policy_is_raised() -> None:
    """Pins F-L47-5: an argument-level rule beat a tool-level deny, so they are not supported."""
    with pytest.raises(InvalidPolicyError):
        parse_policy_rules({'TEST_TOOL:{"filepath": "a.txt"}': "allow"})
