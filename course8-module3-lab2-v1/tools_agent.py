"""Notebook cell 22: tool registration.

Classic (what the notebook ships):
    def is_prime(n: Annotated[int, "Positive integer"]) -> str: ...
    register_function(is_prime, caller=math_asker, executor=math_checker,
                      description="Check if a number is prime.")

Two agents, two roles: one *decides* to call the tool, the other *runs* it.

v1.0 (here):
    @tool
    def is_prime(n: int) -> str: ...
    Agent("math_agent", ..., tools=[is_prime])

The caller/executor split is GONE. There is no ``register_function`` and no
executor agent - the agent that decides also executes, and the JSON schema
is generated from type hints rather than ``Annotated`` metadata.

That removal is a finding, not an inconvenience. The split was AG2's most
distinctive decomposition against the field; v1.0 abandons it and
converges on the shape everyone else already had.

The docstring is the tool description: ``tool()`` resolves
``description or f.__doc__ or ""``. It does NOT fail on a missing
docstring - it silently ships an empty description, which degrades tool
selection with no error. Docstrings on @tool functions stay exempt from
the cut-restatement rule.
"""

from __future__ import annotations

import asyncio

from ag2 import Agent, tool

from helpers.config import build_config
from helpers.metrics import print_usage

QUESTIONS = (
    "Is 72 a prime number?",
    "Is 97 a prime number?",
)


@tool
def is_prime(n: int) -> str:
    """Check whether a positive integer is prime. Returns "Yes" or "No"."""
    if n < 2:
        return "No"
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return "No"
    return "Yes"


async def main() -> None:
    agent = Agent(
        "math_agent",
        "You answer questions about whether numbers are prime. Always use the "
        "is_prime tool; never compute mentally.",
        config=build_config(),
        tools=[is_prime],
    )

    for question in QUESTIONS:
        reply = await agent.ask(question)
        body = reply.body
        assert body is not None, f"no text body for question: {question}"
        print(f"\n[{question}]\n{body}")
        print_usage(await reply.usage(), label=question)


if __name__ == "__main__":
    asyncio.run(main())
