"""Tip calculation tool for the tool-calling agent."""

from langchain.tools import tool


@tool
def calculate_tip(total_bill: float, tip_percent: float) -> float:
    """
    Calculate the tip amount for a bill.

    Args:
        total_bill: the total bill amount in currency units
        tip_percent: the tip percentage as a number (e.g. 15 for 15%, not 0.15)

    Returns:
        The tip amount in the same currency units as total_bill.
    """
    return total_bill * tip_percent / 100