"""Section 3 — Tool introspection and direct invocation.

Two things this section demonstrates that the LLM-mediated sections hide:

1. The @tool decorator exposes name / description / args schema as inspectable
   attributes. These are exactly what the LLM receives — printing them shows
   the runtime contract the agent will see.

2. Direct .invoke() with messy strings exposes the tool's parsing brittleness
   without the LLM layer compensating. Inputs containing word-numbers ("four")
   or non-numeric noise ("a b") get silently dropped — the LLM-mediated call
   would have normalised "four" → 4 before reaching the tool.

The gap between direct-test output and agent-test output is itself a lesson:
LLM-mediated tests hide tool brittleness that direct tests expose.
"""
from tools import subtract_numbers, multiply_numbers, divide_numbers


def run() -> None:
    print("=== Section 3a: Tool introspection (subtract_numbers) ===")
    print(f"Name:        {subtract_numbers.name}")
    print(f"Description: {subtract_numbers.description}")
    print(f"Args schema: {subtract_numbers.args}")

    print("\n=== Section 3b: Direct invoke with non-numeric noise ===")
    test_input = "10 20 30 and four a b"
    print(f"subtract_numbers.invoke({test_input!r}) = {subtract_numbers.invoke(test_input)}")

    print("\n=== Section 3c: Direct invoke — multiply with 'four' as word ===")
    multiply_test = "2, 3, and four "
    print(f"multiply_numbers.invoke({multiply_test!r}) = {multiply_numbers.invoke(multiply_test)}")

    print("\n=== Section 3d: Direct invoke — divide with 'two' as word ===")
    divide_test = "100, 5, two"
    print(f"divide_numbers.invoke({divide_test!r}) = {divide_numbers.invoke(divide_test)}")