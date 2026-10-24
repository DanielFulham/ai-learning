from unittest.mock import MagicMock

import pytest

from application.policy_book import PolicyBook
from domain.policy import Policy
from interfaces.policy_store_interface import InvalidPolicyError, PolicyStoreInterface


def _make_store(rules: dict[str, str] | None = None) -> MagicMock:
    store = MagicMock(spec=PolicyStoreInterface)
    store.load_rules.return_value = rules or {}
    return store


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("read_file", Policy.ALLOW),
        ("write_file", Policy.ASK),
        ("delete_file", Policy.DENY),
        ("execute_command", Policy.DENY),
    ],
)
def test_given_an_empty_policy_file_when_asked_about_a_lab_tool_then_the_default_applies(
    tool: str, expected: Policy
) -> None:
    assert PolicyBook(_make_store()).policy_for(tool) is expected


def test_given_a_file_rule_when_asked_about_that_tool_then_it_overrides_the_default() -> None:
    book = PolicyBook(_make_store({"write_file": "allow"}))
    assert book.policy_for("write_file") is Policy.ALLOW


def test_given_a_file_rule_for_one_tool_when_asked_about_another_then_its_default_stays() -> None:
    book = PolicyBook(_make_store({"write_file": "allow"}))
    assert book.policy_for("delete_file") is Policy.DENY


def test_given_a_tool_with_no_rule_and_no_default_when_asked_then_the_answer_is_ask() -> None:
    """Pins fail-safe: an unknown tool is never allowed to run unprompted."""
    assert PolicyBook(_make_store()).policy_for("TEST_UNKNOWN_TOOL") is Policy.ASK


def test_given_a_file_rule_for_a_new_tool_when_asked_then_that_rule_applies() -> None:
    book = PolicyBook(_make_store({"TEST_NEW_TOOL": "deny"}))
    assert book.policy_for("TEST_NEW_TOOL") is Policy.DENY


def test_given_a_misspelt_policy_when_asked_then_invalid_policy_is_raised() -> None:
    """Pins F-L47-5: the answer is an error the gate can refuse on, never a tool that runs."""
    book = PolicyBook(_make_store({"read_file": "alow"}))
    with pytest.raises(InvalidPolicyError):
        book.policy_for("read_file")


def test_given_a_misspelt_policy_for_one_tool_when_asked_about_another_then_it_still_fails() -> (
    None
):
    """Pins that a bad file is never partly trusted: one bad value invalidates the whole file."""
    book = PolicyBook(_make_store({"read_file": "alow"}))
    with pytest.raises(InvalidPolicyError):
        book.policy_for("delete_file")


def test_given_the_store_raises_invalid_policy_when_asked_then_the_error_propagates() -> None:
    store = _make_store()
    store.load_rules.side_effect = InvalidPolicyError("TEST_NOT_JSON")
    with pytest.raises(InvalidPolicyError):
        PolicyBook(store).policy_for("read_file")


def test_given_a_file_rule_when_all_policies_are_listed_then_defaults_and_rules_are_merged() -> (
    None
):
    book = PolicyBook(_make_store({"write_file": "allow", "TEST_NEW_TOOL": "deny"}))
    assert book.all_policies() == {
        "read_file": Policy.ALLOW,
        "write_file": Policy.ALLOW,
        "delete_file": Policy.DENY,
        "execute_command": Policy.DENY,
        "read_audit_log": Policy.ASK,
        "read_permissions": Policy.ALLOW,
        "TEST_NEW_TOOL": Policy.DENY,
    }


def test_given_a_misspelt_policy_when_all_policies_are_listed_then_invalid_policy_is_raised() -> (
    None
):
    book = PolicyBook(_make_store({"read_file": "alow"}))
    with pytest.raises(InvalidPolicyError):
        book.all_policies()


def test_given_the_file_changes_between_questions_when_asked_twice_then_the_new_rule_applies() -> (
    None
):
    """Pins no caching: an edit to the policy file takes effect on the next call, no restart."""
    store = _make_store({"write_file": "ask"})
    book = PolicyBook(store)
    assert book.policy_for("write_file") is Policy.ASK
    store.load_rules.return_value = {"write_file": "deny"}
    assert book.policy_for("write_file") is Policy.DENY
