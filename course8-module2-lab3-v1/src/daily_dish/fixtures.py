"""Shared fixtures for the Daily Dish lab.

Three FAQ-answerable questions plus one deliberately-not-in-FAQ question.
Kept in a shared module so each approach script depends on the fixture
list, not on a sibling approach's implementation.
"""

from __future__ import annotations

FIXTURE_QUERIES: list[str] = [
    "What are the timings?",
    "What is the phone number?",
    "What is the location?",
    "What are some nearby parking options?",
]
