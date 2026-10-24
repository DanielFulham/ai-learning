import pytest

from domain.default_policies import DEFAULT_POLICIES
from domain.policy import Policy


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("read_file", Policy.ALLOW),
        ("write_file", Policy.ASK),
        ("delete_file", Policy.DENY),
        ("execute_command", Policy.DENY),
        ("read_audit_log", Policy.ALLOW),
        ("read_permissions", Policy.ALLOW),
    ],
)
def test_given_the_built_in_defaults_when_a_name_is_looked_up_then_it_has_the_expected_policy(
    name: str, expected: Policy
) -> None:
    assert DEFAULT_POLICIES[name] is expected


def test_given_the_built_in_defaults_when_listed_then_they_cover_the_tools_and_the_resources() -> (
    None
):
    """Pins that a new tool or resource gets no silent default: it falls back to ask."""
    assert set(DEFAULT_POLICIES) == {
        "read_file",
        "write_file",
        "delete_file",
        "execute_command",
        "read_audit_log",
        "read_permissions",
    }
