import pytest

from domain.risk_level import RiskLevel
from domain.tool_risks import TOOL_RISKS


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("read_file", RiskLevel.LOW),
        ("write_file", RiskLevel.MEDIUM),
        ("delete_file", RiskLevel.HIGH),
        ("execute_command", RiskLevel.CRITICAL),
    ],
)
def test_given_the_lab_tools_when_their_risk_is_looked_up_then_it_matches_the_lab(
    tool: str, expected: RiskLevel
) -> None:
    assert TOOL_RISKS[tool] is expected


def test_given_the_tool_risks_when_listed_then_they_cover_exactly_the_four_tools() -> None:
    assert set(TOOL_RISKS) == {"read_file", "write_file", "delete_file", "execute_command"}
