import pytest

from application.tools.tip import calculate_tip


def test_calculate_tip_round_percent():
    assert calculate_tip.invoke({"total_bill": 100, "tip_percent": 20}) == 20.0


def test_calculate_tip_realistic_bill():
    result = calculate_tip.invoke({"total_bill": 87.43, "tip_percent": 15})
    assert result == pytest.approx(13.1145)


def test_calculate_tip_fractional_percent():
    assert calculate_tip.invoke({"total_bill": 50, "tip_percent": 12.5}) == 6.25


def test_calculate_tip_zero_bill():
    assert calculate_tip.invoke({"total_bill": 0, "tip_percent": 20}) == 0.0


def test_calculate_tip_zero_percent():
    assert calculate_tip.invoke({"total_bill": 100, "tip_percent": 0}) == 0.0