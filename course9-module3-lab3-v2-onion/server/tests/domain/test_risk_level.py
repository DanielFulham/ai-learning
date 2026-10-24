import pytest

from domain.risk_level import RiskLevel


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("low", RiskLevel.LOW),
        ("medium", RiskLevel.MEDIUM),
        ("high", RiskLevel.HIGH),
        ("critical", RiskLevel.CRITICAL),
    ],
)
def test_given_a_risk_level_name_when_parsed_then_it_is_the_matching_level(
    text: str, expected: RiskLevel
) -> None:
    assert RiskLevel(text) is expected


def test_given_text_that_is_not_a_risk_level_when_parsed_then_a_value_error_is_raised() -> None:
    with pytest.raises(ValueError):
        RiskLevel("TEST_UNKNOWN")


def test_given_the_levels_when_ranked_then_low_medium_high_critical_is_the_order() -> None:
    """Pins the order the host and the audit log can rely on for sorting and thresholds."""
    ranked = sorted(RiskLevel, key=lambda level: level.rank)
    assert ranked == [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
