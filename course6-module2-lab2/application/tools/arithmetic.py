"""Arithmetic tools for the tool-calling agent."""

from langchain.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add a and b and return the sum."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a and return the difference."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b and return the product."""
    return a * b