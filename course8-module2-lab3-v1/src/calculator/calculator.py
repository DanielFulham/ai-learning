"""Custom Functions appendix — @tool decorator demo.

Demonstrates CrewAI's `@tool` decorator for defining custom Python
functions as agent-callable tools. Distinct from the SerperDevTool /
PDFSearchTool factory pattern used in the daily_dish scripts:

- Factory pattern: instantiate a pre-built tool class from crewai_tools
- @tool decorator: wrap an arbitrary Python function with a docstring;
  CrewAI reads the docstring to generate the tool description that gets
  injected into the LLM prompt

The docstring is load-bearing — it's what the LLM sees when deciding
whether to invoke the tool. Not a comment, not documentation-only.

Kept as a smoke test that the @tool decorator works on our modernised
stack, not as a source of new empirical findings.

Self-contained subpackage: does NOT depend on daily_dish/. Own config.py
duplicates build_llm + env loading rather than sharing across peers.
"""

from __future__ import annotations

import re

from config import build_llm
from crewai import Agent, Crew, Process, Task
from crewai.crews.crew_output import CrewOutput
from crewai.tools import tool


@tool("Add Numbers Tool")
def add_numbers(input_string: str) -> str:
    """Extracts all integers from the input string and returns their sum.
    Use this tool when the query asks for addition, total, or sum.
    """
    numbers = [int(n) for n in re.findall(r"-?\d+", input_string)]
    if not numbers:
        return "No numbers found in the input."
    return f"The sum of {numbers} is {sum(numbers)}."


@tool("Multiply Numbers Tool")
def multiply_numbers(input_string: str) -> str:
    """Extracts all integers from the input string and returns their product.
    Use this tool when the query asks for multiplication or product.
    """
    numbers = [int(n) for n in re.findall(r"-?\d+", input_string)]
    if not numbers:
        return "No numbers found in the input."
    product = 1
    for n in numbers:
        product *= n
    return f"The product of {numbers} is {product}."


CALCULATOR_QUERIES: list[str] = [
    "Add 7 and 8, also 9, don't forget 10.",
    "Multiply 3 and 4, then also 5.",
]


def build_crew() -> Crew:
    """Construct the Calculator crew.

    Single agent with two custom @tool-decorated Python functions.
    """
    llm = build_llm()

    calculator = Agent(
        role="Calculator",
        goal=(
            "Extract numbers from the user's request and perform the "
            "requested arithmetic operation using the available tools."
        ),
        backstory=(
            "You are a specialist calculator agent. You interpret natural "
            "language requests containing numbers and arithmetic instructions, "
            "then invoke the appropriate tool — addition or multiplication — "
            "to produce the correct result."
        ),
        tools=[add_numbers, multiply_numbers],
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    calculation_task = Task(
        description=(
            "Answer the following user request: '{user_query}'. "
            "Extract the numbers, decide whether to add or multiply based "
            "on the request, and use the appropriate tool. Return the result."
        ),
        expected_output="A clear response containing the arithmetic result.",
        agent=calculator,
    )

    return Crew(
        agents=[calculator],
        tasks=[calculation_task],
        process=Process.sequential,
        verbose=False,
    )


def run_calculator_queries() -> None:
    """Run the calculator crew against the fixture queries."""
    crew = build_crew()
    prev_prompt = 0
    prev_completion = 0
    for query in CALCULATOR_QUERIES:
        print(f"\n{'=' * 72}")
        print(f"QUERY: {query}")
        print(f"{'=' * 72}")
        result = crew.kickoff(inputs={"user_query": query})
        if not isinstance(result, CrewOutput):
            raise TypeError(f"Expected CrewOutput from kickoff, got {type(result).__name__}")
        print(f"\n--- ANSWER ---\n{result.raw}\n")
        if result.token_usage is not None:
            delta_prompt = result.token_usage.prompt_tokens - prev_prompt
            delta_completion = result.token_usage.completion_tokens - prev_completion
            print(
                f"--- USAGE (this query) ---\n"
                f"prompt_tokens={delta_prompt} "
                f"completion_tokens={delta_completion} "
                f"cached_prompt_tokens={result.token_usage.cached_prompt_tokens}\n"
            )
            prev_prompt = result.token_usage.prompt_tokens
            prev_completion = result.token_usage.completion_tokens


if __name__ == "__main__":
    run_calculator_queries()
