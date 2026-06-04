"""Section 1 — Direct tool tests (no LLM involved).

Demonstrates that @tool-decorated functions can be invoked directly via .invoke()
with a dict matching the schema. Useful as a sanity check before wiring them
into an agent: if the direct call is wrong, the LLM-mediated call will be wrong
in subtler ways.

Specifically tests `add_numbers_with_options` with the `absolute` flag toggled,
confirming the boolean default and the typed-list input behave as expected.
"""
from tools import add_numbers_with_options


def run() -> None:
    print("=== Section 1: Direct tool tests (no LLM) ===")
    print(add_numbers_with_options.invoke({"numbers": [-1.1, -2.1, -3.0], "absolute": False}))
    print(add_numbers_with_options.invoke({"numbers": [-1.1, -2.1, -3.0], "absolute": True}))